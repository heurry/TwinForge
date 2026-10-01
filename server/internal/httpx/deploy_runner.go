package httpx

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"github.com/heurry/cloudnative-infra-platform/server/internal/k8s"
)

// A1：真实 rollout 执行器。triggerDeployment 给出 image 时走这里——脱离请求 context 后台跑：
// patch 镜像 → 轮询 Deployment 滚动发布状态 → 成功回写 success；失败/超时自动回滚到上一个镜像 + failed。
// 实时进度写进 deployments.metadata（phase/progress/ready/desired + 小事件日志），前端轮询列表即可见。
// 安全：调用前已过 A2 的 k8sWriteGuard（feature flag + 命名空间守卫），serving 命名空间不会到这里。

const (
	rolloutPollInterval = 2 * time.Second
	rolloutTimeout      = 180 * time.Second
)

// rolloutTarget 是一次真实 rollout 的目标与发起人。
type rolloutTarget struct {
	Namespace string
	Name      string
	Image     string
	Operator  string
}

// runDeploymentRollout 是后台 rollout 主流程（meta 由本 goroutine 独占持有并增量持久化）。
func (a *API) runDeploymentRollout(id string, t rolloutTarget, meta map[string]any) {
	defer func() {
		if rec := recover(); rec != nil {
			slog.Error("deployment rollout panic", "id", id, "err", rec)
			a.deployEvent(context.Background(), id, meta, "failed", fmt.Sprintf("panic: %v", rec))
			_, _ = a.Store.FinishDeployment(context.Background(), id, "failed")
		}
	}()
	ctx := context.Background()

	a.deployEvent(ctx, id, meta, "patching", fmt.Sprintf("patch image → %s", t.Image))
	prev, err := a.K8s.PatchDeploymentImage(ctx, t.Namespace, t.Name, t.Image)
	if err != nil {
		meta["error"] = err.Error()
		a.deployEvent(ctx, id, meta, "failed", "patch failed: "+err.Error())
		_, _ = a.Store.FinishDeployment(ctx, id, "failed")
		a.Store.Audit(ctx, t.Operator, "operator", "deployment.rollout.failed", "deployment", t.Namespace+"/"+t.Name,
			map[string]any{"image": t.Image, "error": err.Error()})
		return
	}
	meta["previous_image"] = prev
	a.deployEvent(ctx, id, meta, "progressing", fmt.Sprintf("rolling out %s (was %s)", t.Image, prev))

	deadline := time.Now().Add(rolloutTimeout)
	ticker := time.NewTicker(rolloutPollInterval)
	defer ticker.Stop()
	for range ticker.C {
		st, err := a.K8s.RolloutStatus(ctx, t.Namespace, t.Name)
		if err != nil {
			// 瞬时读失败不致命，记一笔继续轮询直到超时。
			meta["message"] = "status read error: " + err.Error()
			a.persistDeployMeta(ctx, id, meta)
			if time.Now().After(deadline) {
				a.rollbackRollout(ctx, id, t, prev, meta, "rollout timed out")
				return
			}
			continue
		}
		a.deployProgress(ctx, id, meta, st)
		switch {
		case st.Complete:
			a.deployEvent(ctx, id, meta, "succeeded", fmt.Sprintf("rollout complete: %d/%d ready", st.Ready, st.Desired))
			_, _ = a.Store.FinishDeployment(ctx, id, "success")
			a.Store.Audit(ctx, t.Operator, "operator", "deployment.rollout.succeeded", "deployment", t.Namespace+"/"+t.Name,
				map[string]any{"image": t.Image, "ready": st.Ready, "desired": st.Desired})
			return
		case st.Failed:
			a.rollbackRollout(ctx, id, t, prev, meta, "progress deadline exceeded: "+st.Message)
			return
		case time.Now().After(deadline):
			a.rollbackRollout(ctx, id, t, prev, meta, "rollout timed out")
			return
		}
	}
}

// rollbackRollout 把镜像 patch 回 prev（自动回滚），落 failed + 两条审计。
func (a *API) rollbackRollout(ctx context.Context, id string, t rolloutTarget, prev string, meta map[string]any, reason string) {
	meta["error"] = reason
	a.deployEvent(ctx, id, meta, "rolling_back", "auto-rollback: "+reason)
	if prev != "" {
		if _, err := a.K8s.PatchDeploymentImage(ctx, t.Namespace, t.Name, prev); err != nil {
			a.deployEvent(ctx, id, meta, "rolling_back", "rollback patch failed: "+err.Error())
		}
	}
	meta["phase"] = "rolled_back"
	meta["message"] = "auto-rolled back to " + prev
	a.appendDeployEvent(meta, "rolled_back", "auto-rolled back to "+prev)
	a.persistDeployMeta(ctx, id, meta)
	_, _ = a.Store.FinishDeployment(ctx, id, "failed")
	a.Store.Audit(ctx, t.Operator, "operator", "deployment.rollout.failed", "deployment", t.Namespace+"/"+t.Name,
		map[string]any{"image": t.Image, "reason": reason})
	a.Store.Audit(ctx, t.Operator, "operator", "deployment.rollout.rolledback", "deployment", t.Namespace+"/"+t.Name,
		map[string]any{"to": prev})
}

// deployProgress 更新进度字段（ready/desired/updated/progress），不追加事件，持久化。
func (a *API) deployProgress(ctx context.Context, id string, meta map[string]any, st k8s.RolloutSnapshot) {
	progress := 0
	if st.Desired > 0 {
		progress = int(st.Ready) * 100 / int(st.Desired)
	}
	meta["ready"] = st.Ready
	meta["desired"] = st.Desired
	meta["updated"] = st.Updated
	meta["progress"] = progress
	a.persistDeployMeta(ctx, id, meta)
}

// persistDeployMeta best-effort 把整份 metadata 写回 deployments 行（goroutine 独占，无并发写）。
func (a *API) persistDeployMeta(ctx context.Context, id string, meta map[string]any) {
	b, err := json.Marshal(meta)
	if err != nil {
		return
	}
	_, _ = a.Pool.Exec(ctx, `UPDATE deployments SET metadata = $2 WHERE id = $1::uuid`, id, b)
}

// appendDeployEvent 往 metadata.events 追加一条小日志（有界：仅相位转换时调用）。
func (a *API) appendDeployEvent(meta map[string]any, phase, message string) {
	ev, _ := meta["events"].([]any)
	meta["events"] = append(ev, map[string]any{
		"at": time.Now().UTC().Format(time.RFC3339), "phase": phase, "message": message,
	})
}

// deployEvent 设置当前相位 + 追加事件日志 + 持久化。
func (a *API) deployEvent(ctx context.Context, id string, meta map[string]any, phase, message string) {
	meta["phase"] = phase
	meta["message"] = message
	a.appendDeployEvent(meta, phase, message)
	a.persistDeployMeta(ctx, id, meta)
	level := "info"
	if phase == "failed" || phase == "rolling_back" {
		level = "error"
	}
	_ = a.recordPlatformLog(ctx, platformLogInput{
		Level: level, Source: "deployment-runner", ResourceType: "deployment", ResourceID: id,
		Message: message, Attributes: map[string]any{"phase": phase, "mode": meta["mode"]},
	})
}
