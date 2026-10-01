package httpx

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"github.com/heurry/cloudnative-infra-platform/server/internal/store"
	"github.com/heurry/cloudnative-infra-platform/server/internal/training"
)

// Phase F / F2：训练任务 runner（镜像 deploy_runner 的形态）。submitTrainingJob 后台启它：
// 提交 PyTorchJob → 轮询状态 → 相位/副本进度写 training_jobs.metadata → 成败回写 status + 审计。
// 安全：调用前已过 trainingWriteGuard（ALLOW_TRAINING + 命名空间守卫，serving 硬禁）。
// 注：F2 不做「成功后自动注册到 C1」——那是 F3（finishTraining 留了钩子注释）。

const (
	trainingPollInterval = 5 * time.Second
	trainingTimeout      = 6 * time.Hour // 训练可能很长；超时即判失败并停止轮询
)

// trainingRegistration 是训练成功后注册进 C1（模型注册中心）的目标（F3：训练→注册闭环）。
type trainingRegistration struct {
	ModelID       string
	Version       string
	ParentVersion string // 血缘父版本（base 的 version；空则无血缘）
	BaseModel     string
	LoraAdapter   string
	ArtifactKey   string // 训练容器写 adapter 的 MinIO key（经 env OUTPUT_URI 注入）
}

// trainingTarget 是一次训练任务的目标 spec、发起人与注册目标。
type trainingTarget struct {
	JobID       string
	Spec        training.JobSpec
	Operator    string
	Reg         trainingRegistration
	InitialMeta map[string]any
}

// runTrainingJob 是后台训练主流程（meta 由本 goroutine 独占持有并增量持久化）。
func (a *API) runTrainingJob(t trainingTarget) {
	meta := t.InitialMeta
	if meta == nil {
		meta = map[string]any{}
	}
	defer func() {
		if rec := recover(); rec != nil {
			slog.Error("training job panic", "id", t.JobID, "err", rec)
			a.trainingEvent(context.Background(), t.JobID, meta, "failed", fmt.Sprintf("panic: %v", rec))
			_ = a.Store.UpdateTrainingJobStatus(context.Background(), t.JobID, "failed")
		}
	}()
	ctx := context.Background()
	ref := t.Spec.Namespace + "/" + t.Spec.Name

	a.trainingEvent(ctx, t.JobID, meta, "submitting", fmt.Sprintf("submit PyTorchJob %s", ref))
	if _, err := a.Training.SubmitJob(ctx, t.Spec); err != nil {
		meta["error"] = err.Error()
		a.trainingEvent(ctx, t.JobID, meta, "failed", "submit failed: "+err.Error())
		_ = a.Store.UpdateTrainingJobStatus(ctx, t.JobID, "failed")
		a.Store.Audit(ctx, t.Operator, "operator", "training.failed", "training_job", ref,
			map[string]any{"error": err.Error()})
		return
	}
	_ = a.Store.UpdateTrainingJobStatus(ctx, t.JobID, "running")
	a.trainingEvent(ctx, t.JobID, meta, "running", "PyTorchJob submitted; polling status")

	deadline := time.Now().Add(trainingTimeout)
	ticker := time.NewTicker(trainingPollInterval)
	defer ticker.Stop()
	for range ticker.C {
		job, jobErr := a.Store.GetTrainingJob(ctx, t.JobID)
		if jobErr == nil && job.Status == "cancelled" {
			a.trainingEvent(ctx, t.JobID, meta, "cancelled", "training cancelled by operator")
			return
		}
		st, err := a.Training.GetJob(ctx, t.Spec.Namespace, t.Spec.Name)
		if err != nil {
			// 瞬时读失败不致命，记一笔继续轮询直到超时。
			meta["message"] = "status read error: " + err.Error()
			a.persistTrainingMeta(ctx, t.JobID, meta)
			if time.Now().After(deadline) {
				a.finishTraining(ctx, t, meta, "failed", "training timed out")
				return
			}
			continue
		}
		a.trainingProgress(ctx, t.JobID, meta, st)
		switch st.Phase {
		case "Succeeded":
			a.finishTraining(ctx, t, meta, "succeeded", "training complete")
			return
		case "Failed":
			a.finishTraining(ctx, t, meta, "failed", "PyTorchJob failed: "+st.Message)
			return
		}
		if time.Now().After(deadline) {
			a.finishTraining(ctx, t, meta, "failed", "training timed out")
			return
		}
	}
}

// finishTraining 落最终相位 + 回写 status + 审计。
func (a *API) finishTraining(ctx context.Context, t trainingTarget, meta map[string]any, status, message string) {
	ref := t.Spec.Namespace + "/" + t.Spec.Name
	a.trainingEvent(ctx, t.JobID, meta, status, message)
	_ = a.Store.UpdateTrainingJobStatus(ctx, t.JobID, status)
	action := "training.failed"
	if status == "succeeded" {
		action = "training.succeeded"
	}
	a.Store.Audit(ctx, t.Operator, "operator", action, "training_job", ref, map[string]any{"message": message})
	if status == "succeeded" {
		a.registerTrainedModel(ctx, t, meta)
	}
}

// registerTrainedModel（F3）：训练成功后把产出注册进 C1，血缘指回 base，回填 job 的 output/version。
// adapter 实体由训练容器写到 ArtifactKey（MinIO）；此处只登记注册中心条目（不做上传）。
// 重复注册 (model_id, version) 唯一冲突时记一笔跳过，不让已成功的训练任务变失败。
func (a *API) registerTrainedModel(ctx context.Context, t trainingTarget, meta map[string]any) {
	reg := t.Reg
	if reg.ModelID == "" || reg.Version == "" {
		return
	}
	baseModel, lora, artifact := reg.BaseModel, reg.LoraAdapter, reg.ArtifactKey
	mvID, err := a.Store.RegisterModelVersion(ctx, store.RegisterModelParams{
		ModelID:       reg.ModelID,
		Version:       reg.Version,
		BaseModel:     &baseModel,
		LoraAdapter:   &lora,
		ParentVersion: nilIfEmpty(reg.ParentVersion),
		ArtifactURI:   &artifact,
		Tags:          []string{"trained", "lora"},
		Status:        "registered",
		CreatedBy:     t.Operator,
	})
	if err != nil {
		reason := "register skipped: " + err.Error()
		if isUniqueViolation(err) {
			reason = fmt.Sprintf("register skipped: %s:%s already exists", reg.ModelID, reg.Version)
		}
		a.trainingEvent(ctx, t.JobID, meta, "registered_skipped", reason)
		return
	}
	_ = a.Store.SetTrainingJobOutput(ctx, t.JobID, artifact, mvID)
	a.trainingEvent(ctx, t.JobID, meta, "registered", fmt.Sprintf("registered %s:%s (version id=%s)", reg.ModelID, reg.Version, mvID))
	a.Store.Audit(ctx, t.Operator, "operator", "training.registered", "model", reg.ModelID+":"+reg.Version,
		map[string]any{"version_id": mvID, "parent_version": reg.ParentVersion, "artifact_uri": artifact})
}

// trainingProgress 把相位 + 副本计数写入 metadata（不追加事件），持久化。
func (a *API) trainingProgress(ctx context.Context, id string, meta map[string]any, st training.JobStatus) {
	meta["phase"] = st.Phase
	if st.Reason != "" {
		meta["reason"] = st.Reason
	}
	repl := map[string]any{}
	for role, rs := range st.ReplicaStatuses {
		repl[role] = map[string]any{"active": rs.Active, "succeeded": rs.Succeeded, "failed": rs.Failed}
	}
	meta["replica_statuses"] = repl
	if st.StartTime != "" {
		meta["start_time"] = st.StartTime
	}
	if st.CompletionTime != "" {
		meta["completion_time"] = st.CompletionTime
	}
	a.persistTrainingMeta(ctx, id, meta)
}

// trainingEvent 设置当前相位 + 追加事件日志 + 持久化。
func (a *API) trainingEvent(ctx context.Context, id string, meta map[string]any, phase, message string) {
	meta["phase"] = phase
	meta["message"] = message
	ev, _ := meta["events"].([]any)
	meta["events"] = append(ev, map[string]any{
		"at": time.Now().UTC().Format(time.RFC3339), "phase": phase, "message": message,
	})
	a.persistTrainingMeta(ctx, id, meta)
	level := "info"
	if phase == "failed" {
		level = "error"
	}
	_ = a.recordPlatformLog(ctx, platformLogInput{
		Level: level, Source: "training-runner", ResourceType: "training_job", ResourceID: id,
		Message: message, Attributes: map[string]any{"phase": phase},
	})
}

// persistTrainingMeta best-effort 把整份 metadata 写回 training_jobs 行（goroutine 独占，无并发写）。
func (a *API) persistTrainingMeta(ctx context.Context, id string, meta map[string]any) {
	b, err := json.Marshal(meta)
	if err != nil {
		return
	}
	_, _ = a.Pool.Exec(ctx, `UPDATE training_jobs SET metadata = $2, updated_at = now() WHERE id = $1::uuid`, id, b)
}
