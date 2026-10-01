package httpx

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/heurry/cloudnative-infra-platform/server/internal/k8s"
	"github.com/heurry/cloudnative-infra-platform/server/internal/store"
	"github.com/heurry/cloudnative-infra-platform/server/internal/training"
)

// Phase F / F2：分布式训练任务（Kubeflow PyTorchJob）生命周期。
// 提交即建台账（pending）+ 启 runner（training_runner.go）轮询状态回写；
// 写操作受 ALLOW_TRAINING + TRAINING_NAMESPACES + guardNamespace（serving 硬禁）三重约束。

const (
	maxTrainingWorkers = 8   // 单任务 Worker 上限（防误配打爆集群）
	maxGPUsPerReplica  = 8   // 每副本 GPU 上限
	trainingLogTail    = 200 // 日志查看默认尾行数
)

// trainingWriteGuard：client 就绪 + ALLOW_TRAINING + 命名空间放行（复用 k8s 写路径的 guardNamespace，serving 硬禁）。
func (a *API) trainingWriteGuard(w http.ResponseWriter, r *http.Request, namespace string) bool {
	if a.Training == nil {
		WriteError(w, r, http.StatusServiceUnavailable, "training_unavailable", "训练编排未配置（Kubeflow 客户端不可用）")
		return false
	}
	if !a.AllowTraining {
		WriteError(w, r, http.StatusForbidden, "training_disabled", "训练写操作未启用（设置 ALLOW_TRAINING=true）")
		return false
	}
	// 训练任务名通常包含 qwen/model 名称；serving 隔离应按 namespace 判断，
	// 否则合法的 qwen35-lora PyTorchJob 会被 serving 名称保护规则误杀。
	if err := guardNamespace(namespace, "", a.TrainingNamespaces); err != nil {
		WriteError(w, r, http.StatusForbidden, "training_namespace_protected", err.Error())
		return false
	}
	return true
}

type submitTrainingReq struct {
	Name          string            `json:"name"`
	Namespace     string            `json:"namespace"`
	BaseModel     string            `json:"base_model"`
	DatasetURI    string            `json:"dataset_uri"`
	Image         string            `json:"image"`
	Workers       int32             `json:"workers"`
	GPUsPerWorker int32             `json:"gpus_per_worker"`
	Hyperparams   map[string]any    `json:"hyperparams"`
	Env           map[string]string `json:"env"`
	// F3：训练成功后注册进 C1 的目标（可选）。model_id 缺省取 name；version 缺省自动生成；base_version=血缘父版本。
	ModelID       string `json:"model_id"`
	Version       string `json:"version"`
	BaseVersion   string `json:"base_version"`
	ConfigItemID  string `json:"config_item_id"`
	ConfigVersion int    `json:"config_version"`
	ConfigKey     string `json:"config_key"`
	ConfigEnv     string `json:"config_env"`
	Operator      string `json:"operator"`
}

// POST /api/training/jobs —— 提交训练任务（建台账 + 启 runner 异步提交 PyTorchJob）。
func (a *API) submitTrainingJob(w http.ResponseWriter, r *http.Request) {
	var req submitTrainingReq
	if err := decodeBody(r, &req); err != nil {
		a.badRequest(w, r, "invalid body")
		return
	}
	req.Name = strings.TrimSpace(req.Name)
	req.BaseModel = strings.TrimSpace(req.BaseModel)
	req.Image = strings.TrimSpace(req.Image)
	if req.Name == "" || req.BaseModel == "" || req.Image == "" {
		a.badRequest(w, r, "name, base_model, image are required")
		return
	}
	if req.Workers < 0 || req.Workers > maxTrainingWorkers {
		a.badRequest(w, r, fmt.Sprintf("workers must be between 0 and %d", maxTrainingWorkers))
		return
	}
	if req.GPUsPerWorker < 0 || req.GPUsPerWorker > maxGPUsPerReplica {
		a.badRequest(w, r, fmt.Sprintf("gpus_per_worker must be between 0 and %d", maxGPUsPerReplica))
		return
	}

	namespace := strings.TrimSpace(req.Namespace)
	if namespace == "" {
		namespace = "training"
		if len(a.TrainingNamespaces) > 0 {
			namespace = a.TrainingNamespaces[0]
		}
	}
	if !a.trainingWriteGuard(w, r, namespace) {
		return
	}
	if benchmarkID, active, err := a.activeBenchmarkRun(r.Context()); err != nil {
		a.fail(w, r, err)
		return
	} else if active {
		WriteError(w, r, http.StatusConflict, "gpu_lane_busy", "推理压测 "+benchmarkID+" 正在占用单机 GPU 实验通道，请先停止压测")
		return
	}
	if gpu := a.benchmarkGPUSnapshot(r.Context()); gpu["available"] == true {
		if used, ok := asFloat(gpu["max_memory_utilization_percent"]); ok && used >= 20 {
			WriteError(w, r, http.StatusConflict, "gpu_lane_busy", fmt.Sprintf("GPU 显存占用 %.1f%%，请先停止 vLLM 推理服务并释放显存", used))
			return
		}
	}

	operator := a.actor(r, req.Operator)
	datasetURI := nilIfEmpty(req.DatasetURI)
	ref := namespace + "/" + req.Name
	id, err := a.Store.CreateTrainingJob(r.Context(), store.CreateTrainingJobParams{
		Name:          req.Name,
		Framework:     "pytorch",
		Namespace:     namespace,
		BaseModel:     req.BaseModel,
		DatasetURI:    datasetURI,
		Image:         req.Image,
		Workers:       req.Workers,
		GPUsPerWorker: req.GPUsPerWorker,
		Hyperparams:   req.Hyperparams,
		K8sJobRef:     &ref,
		CreatedBy:     operator,
	})
	if err != nil {
		a.fail(w, r, err)
		return
	}
	a.Store.Audit(r.Context(), operator, "operator", "training.submit", "training_job", ref,
		map[string]any{"base_model": req.BaseModel, "workers": req.Workers, "gpus_per_worker": req.GPUsPerWorker,
			"config_item_id": req.ConfigItemID, "config_version": req.ConfigVersion, "config_key": req.ConfigKey})

	// 训练容器经 env 拿数据集/基座；runner 脱离请求 context 后台提交 + 轮询。
	env := map[string]string{}
	for k, v := range req.Env {
		env[k] = v
	}
	if datasetURI != nil {
		env["DATASET_URI"] = *datasetURI
	}
	env["BASE_MODEL"] = req.BaseModel
	env["NNODES"] = fmt.Sprintf("%d", req.Workers+1)
	env["GPUS_PER_NODE"] = fmt.Sprintf("%d", req.GPUsPerWorker)
	copyTrainingHyperparams(env, req.Hyperparams)

	// F3：训练成功后把产出注册进 C1。model_id 缺省取 name；version 缺省 lora-<jobid 前 8 位>。
	// adapter 由训练容器写到 OUTPUT_URI（MinIO key）；成功后 runner 据此登记注册中心条目。
	modelID := orDefault(strings.TrimSpace(req.ModelID), req.Name)
	version := strings.TrimSpace(req.Version)
	if version == "" {
		version = "lora-" + id[:8]
	}
	artifactKey := fmt.Sprintf("models/%s/%s/adapter", modelID, version)
	env["OUTPUT_URI"] = artifactKey
	env["OUTPUT_DIR"] = fmt.Sprintf("%s/%s", strings.TrimRight(a.TrainingOutputRoot, "/"), id)

	hostPathMounts := []training.HostPathMount{}
	if a.TrainingHostPath != "" && a.TrainingMountPath != "" {
		hostPathMounts = append(hostPathMounts, training.HostPathMount{
			Name: "training-data", HostPath: a.TrainingHostPath, MountPath: a.TrainingMountPath,
		})
	}
	envFromSecrets := []string{}
	if a.TrainingArtifactSecret != "" {
		envFromSecrets = append(envFromSecrets, a.TrainingArtifactSecret)
	}

	go a.runTrainingJob(trainingTarget{
		JobID:    id,
		Operator: operator,
		InitialMeta: map[string]any{"config_ref": map[string]any{
			"item_id": req.ConfigItemID, "version": req.ConfigVersion, "key": req.ConfigKey, "env": req.ConfigEnv,
		}},
		Spec: training.JobSpec{
			Namespace:      namespace,
			Name:           req.Name,
			Image:          req.Image,
			Env:            env,
			Workers:        req.Workers,
			GPUsPerWorker:  req.GPUsPerWorker,
			HostPathMounts: hostPathMounts,
			EnvFromSecrets: envFromSecrets,
		},
		Reg: trainingRegistration{
			ModelID:       modelID,
			Version:       version,
			ParentVersion: strings.TrimSpace(req.BaseVersion),
			BaseModel:     req.BaseModel,
			LoraAdapter:   req.Name,
			ArtifactKey:   artifactKey,
		},
	})

	WriteJSON(w, http.StatusAccepted, map[string]any{"id": id, "name": req.Name, "namespace": namespace, "status": "pending", "registers_as": modelID + ":" + version,
		"config_ref": map[string]any{"item_id": req.ConfigItemID, "version": req.ConfigVersion, "key": req.ConfigKey}})
}

func copyTrainingHyperparams(env map[string]string, values map[string]any) {
	keys := map[string]string{
		"learning_rate":               "LEARNING_RATE",
		"epochs":                      "EPOCHS",
		"lora_rank":                   "LORA_RANK",
		"lora_alpha":                  "LORA_ALPHA",
		"lora_dropout":                "LORA_DROPOUT",
		"per_device_train_batch_size": "PER_DEVICE_TRAIN_BATCH_SIZE",
		"gradient_accumulation_steps": "GRADIENT_ACCUMULATION_STEPS",
		"precision":                   "PRECISION",
		"deepspeed":                   "DEEPSPEED",
		"gradient_checkpointing":      "GRADIENT_CHECKPOINTING",
		"max_seq_length":              "MAX_SEQ_LENGTH",
		"max_samples":                 "MAX_SAMPLES",
		"seed":                        "SEED",
	}
	for key, envKey := range keys {
		if value, ok := values[key]; ok {
			env[envKey] = fmt.Sprint(value)
		}
	}
}

func (a *API) activeBenchmarkRun(ctx context.Context) (string, bool, error) {
	var id string
	err := a.Pool.QueryRow(ctx, `
		SELECT run_id FROM benchmark_runs
		 WHERE status IN ('queued','running')
		 ORDER BY created_at DESC LIMIT 1`).Scan(&id)
	if isNoRows(err) {
		return "", false, nil
	}
	return id, err == nil, err
}

// GET /api/training/jobs
func (a *API) listTrainingJobs(w http.ResponseWriter, r *http.Request) {
	jobs, err := a.Store.ListTrainingJobs(r.Context())
	if err != nil {
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{"jobs": mapSlice(jobs, toTrainingJobDTO)})
}

// GET /api/training/jobs/{id}
func (a *API) getTrainingJob(w http.ResponseWriter, r *http.Request) {
	job, err := a.Store.GetTrainingJob(r.Context(), chi.URLParam(r, "id"))
	if err != nil {
		if isNoRows(err) {
			WriteError(w, r, http.StatusNotFound, "not_found", "training job not found")
			return
		}
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, toTrainingJobDTO(job))
}

// DELETE /api/training/jobs/{id} —— 取消：删 PyTorchJob（幂等）+ 置 cancelled。
func (a *API) cancelTrainingJob(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	job, err := a.Store.GetTrainingJob(r.Context(), id)
	if err != nil {
		if isNoRows(err) {
			WriteError(w, r, http.StatusNotFound, "not_found", "training job not found")
			return
		}
		a.fail(w, r, err)
		return
	}
	if !a.trainingWriteGuard(w, r, job.Namespace) {
		return
	}
	if job.K8sJobRef != nil {
		if err := a.Training.DeleteJob(r.Context(), job.Namespace, job.Name); err != nil {
			a.fail(w, r, err)
			return
		}
	}
	if err := a.Store.UpdateTrainingJobStatus(r.Context(), id, "cancelled"); err != nil {
		a.fail(w, r, err)
		return
	}
	a.Store.Audit(r.Context(), a.actor(r, ""), "operator", "training.cancel", "training_job", job.Namespace+"/"+job.Name, map[string]any{})
	WriteJSON(w, http.StatusOK, map[string]any{"id": id, "status": "cancelled"})
}

// GET /api/training/jobs/{id}/kubernetes —— 当前任务对应的 PyTorchJob、Pods 与 K8s Events。
// 集群不可达时仍返回任务期望资源和明确错误，详情页无需把“台账存在”误判成“CRD 已创建”。
func (a *API) trainingJobKubernetes(w http.ResponseWriter, r *http.Request) {
	job, err := a.Store.GetTrainingJob(r.Context(), chi.URLParam(r, "id"))
	if err != nil {
		if isNoRows(err) {
			WriteError(w, r, http.StatusNotFound, "not_found", "training job not found")
			return
		}
		a.fail(w, r, err)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 8*time.Second)
	defer cancel()
	type liveResult struct {
		status training.JobStatus
		err    error
	}
	type snapshotResult struct {
		snapshot k8s.Snapshot
	}
	liveCh := make(chan liveResult, 1)
	snapshotCh := make(chan snapshotResult, 1)
	if a.Training != nil {
		go func() {
			status, liveErr := a.Training.GetJob(ctx, job.Namespace, job.Name)
			liveCh <- liveResult{status: status, err: liveErr}
		}()
	} else {
		liveCh <- liveResult{err: fmt.Errorf("%s", orDefault(a.TrainingErr, "training client unavailable"))}
	}
	if a.K8s != nil {
		go func() {
			snapshotCh <- snapshotResult{snapshot: a.K8s.CollectSnapshot(ctx, true, false, true, false, false)}
		}()
	} else {
		snapshotCh <- snapshotResult{snapshot: k8s.Snapshot{Available: false, Error: orDefault(a.K8sErr, "kubernetes integration not configured"), Pods: []k8s.PodSnapshot{}, Events: []k8s.EventSnapshot{}}}
	}

	live := <-liveCh
	snapshot := (<-snapshotCh).snapshot
	resource := map[string]any{
		"api_version": "kubeflow.org/v1",
		"kind":        "PyTorchJob",
		"namespace":   job.Namespace,
		"name":        job.Name,
		"ref":         job.Namespace + "/" + job.Name,
		"available":   live.err == nil,
	}
	if live.err != nil {
		resource["error"] = live.err.Error()
	} else {
		resource["phase"] = live.status.Phase
		resource["reason"] = live.status.Reason
		resource["message"] = live.status.Message
		resource["replica_statuses"] = live.status.ReplicaStatuses
		resource["start_time"] = live.status.StartTime
		resource["completion_time"] = live.status.CompletionTime
	}

	WriteJSON(w, http.StatusOK, map[string]any{
		"resource": resource,
		"cluster":  map[string]any{"available": snapshot.Available, "error": snapshot.Error},
		"pods":     filterTrainingPods(snapshot.Pods, job.Namespace, job.Name),
		"events":   filterTrainingEvents(snapshot.Events, job.Namespace, job.Name),
	})
}

func filterTrainingPods(pods []k8s.PodSnapshot, namespace, jobName string) []k8s.PodSnapshot {
	result := make([]k8s.PodSnapshot, 0)
	for _, pod := range pods {
		if pod.Namespace == namespace && strings.HasPrefix(pod.Name, jobName+"-") {
			result = append(result, pod)
		}
	}
	return result
}

func filterTrainingEvents(events []k8s.EventSnapshot, namespace, jobName string) []k8s.EventSnapshot {
	result := make([]k8s.EventSnapshot, 0)
	for _, event := range events {
		if event.Namespace != namespace {
			continue
		}
		if event.ResourceName == jobName || strings.HasPrefix(event.ResourceName, jobName+"-") {
			result = append(result, event)
		}
	}
	return result
}

// GET /api/training/jobs/{id}/logs —— 读训练 Pod 日志（master 优先；k8s/Pod 不可用时优雅降级）。
func (a *API) trainingJobLogs(w http.ResponseWriter, r *http.Request) {
	job, err := a.Store.GetTrainingJob(r.Context(), chi.URLParam(r, "id"))
	if err != nil {
		if isNoRows(err) {
			WriteError(w, r, http.StatusNotFound, "not_found", "training job not found")
			return
		}
		a.fail(w, r, err)
		return
	}
	if a.K8s == nil {
		WriteJSON(w, http.StatusOK, map[string]any{"pod": nil, "logs": "", "note": "kubernetes integration not configured"})
		return
	}
	pods, err := a.K8s.Pods(r.Context())
	if err != nil {
		WriteJSON(w, http.StatusOK, map[string]any{"pod": nil, "logs": "", "note": "cluster unavailable: " + err.Error()})
		return
	}
	pod := pickTrainingPod(pods, job.Namespace, job.Name)
	if pod == "" {
		WriteJSON(w, http.StatusOK, map[string]any{"pod": nil, "logs": "", "note": "no pods for this job yet"})
		return
	}
	logs, err := a.K8s.PodLogs(r.Context(), job.Namespace, pod, trainingLogTail)
	if err != nil {
		WriteJSON(w, http.StatusOK, map[string]any{"pod": pod, "logs": "", "note": "logs unavailable: " + err.Error()})
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{"pod": pod, "logs": logs})
}

// pickTrainingPod 在命名空间内按名字前缀匹配训练任务的 Pod，master 优先，否则名字最小的。
func pickTrainingPod(pods []k8s.PodSnapshot, namespace, jobName string) string {
	candidates := []string{}
	for _, p := range pods {
		if p.Namespace != namespace || !strings.HasPrefix(p.Name, jobName+"-") {
			continue
		}
		if strings.Contains(p.Name, "master") {
			return p.Name
		}
		candidates = append(candidates, p.Name)
	}
	if len(candidates) == 0 {
		return ""
	}
	sort.Strings(candidates)
	return candidates[0]
}

// ---- DTO ----

type trainingJobDTO struct {
	ID                string         `json:"id"`
	Name              string         `json:"name"`
	Framework         string         `json:"framework"`
	Namespace         string         `json:"namespace"`
	BaseModel         string         `json:"base_model"`
	DatasetURI        *string        `json:"dataset_uri"`
	Image             string         `json:"image"`
	Workers           int32          `json:"workers"`
	GPUsPerWorker     int32          `json:"gpus_per_worker"`
	Hyperparams       map[string]any `json:"hyperparams"`
	Status            string         `json:"status"`
	K8sJobRef         *string        `json:"k8s_job_ref"`
	OutputArtifactURI *string        `json:"output_artifact_uri"`
	ModelVersionID    *string        `json:"model_version_id"`
	Metadata          map[string]any `json:"metadata"`
	CreatedBy         *string        `json:"created_by"`
	CreatedAt         string         `json:"created_at"`
	UpdatedAt         string         `json:"updated_at"`
}

func toTrainingJobDTO(j store.TrainingJob) trainingJobDTO {
	return trainingJobDTO{
		ID: j.ID, Name: j.Name, Framework: j.Framework, Namespace: j.Namespace, BaseModel: j.BaseModel,
		DatasetURI: j.DatasetURI, Image: j.Image, Workers: j.Workers, GPUsPerWorker: j.GPUsPerWorker,
		Hyperparams: j.Hyperparams, Status: j.Status, K8sJobRef: j.K8sJobRef, OutputArtifactURI: j.OutputArtifactURI,
		ModelVersionID: j.ModelVersionID, Metadata: j.Metadata, CreatedBy: j.CreatedBy,
		CreatedAt: j.CreatedAt.UTC().Format(time.RFC3339), UpdatedAt: j.UpdatedAt.UTC().Format(time.RFC3339),
	}
}
