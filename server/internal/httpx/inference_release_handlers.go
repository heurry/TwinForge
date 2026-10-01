package httpx

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/heurry/cloudnative-infra-platform/server/internal/k8s"
	"github.com/jackc/pgx/v5"
)

const (
	inferenceReleaseModelID  = "qwen36-27b-fp8"
	inferenceReleaseEndpoint = "qwen36-27b-fp8-vllm"
	inferenceReleaseName     = "qwen36-27b-fp8-production"
	inferenceReleaseTimeout  = 10 * time.Minute
	// 本地双 RTX 3090 的首 token 延迟会受权重/编译和长上下文影响，
	// 5s/150ms 对当前演示环境过于严格；保留质量门槛，仅放宽性能 SLO。
	inferenceReleaseMaxP95TTFTMs = 10000.0
	inferenceReleaseMaxP95TPOTMs = 250.0
)

// inferenceModelSpec 是发布中心与运行时之间的唯一模型能力契约。
// 以前发布中心把 27B 双卡参数写死，切换到 4B 时仍会带上 TP=2、FP8 和 Prefix Cache，
// 最终导致模型加载失败或压测证据无法匹配。模型注册表负责选择版本，这里负责给出该模型
// 在当前单机实验环境可用的 endpoint、并行方式和默认调度参数。
type inferenceModelSpec struct {
	ModelID          string
	Endpoint         string
	DeploymentName   string
	TensorParallel   int
	PipelineParallel int
	GPUCount         int
	PrefixCaching    bool
	MaxModelLen      int
	MaxNumSeqs       int
	MaxBatchedTokens int
	GPUMemory        float64
	KVCacheDType     string
}

func inferenceModelSpecFor(modelID string) (inferenceModelSpec, bool) {
	modelID = strings.TrimSpace(modelID)
	switch modelID {
	case "qwen38-27b-fp8":
		return inferenceModelSpec{ModelID: modelID, Endpoint: modelID + "-vllm", DeploymentName: modelID + "-production", TensorParallel: 2, PipelineParallel: 1, GPUCount: 2, PrefixCaching: true, MaxModelLen: 24576, MaxNumSeqs: 4, MaxBatchedTokens: 8192, GPUMemory: 0.88, KVCacheDType: "auto"}, true
	case "qwen36-27b-fp8":
		return inferenceModelSpec{ModelID: modelID, Endpoint: modelID + "-vllm", DeploymentName: modelID + "-production", TensorParallel: 2, PipelineParallel: 1, GPUCount: 2, PrefixCaching: true, MaxModelLen: 4096, MaxNumSeqs: 8, MaxBatchedTokens: 4096, GPUMemory: 0.9, KVCacheDType: "auto"}, true
	case "qwen36-27b-awq":
		return inferenceModelSpec{ModelID: modelID, Endpoint: modelID + "-vllm", DeploymentName: modelID + "-production", TensorParallel: 2, PipelineParallel: 1, GPUCount: 2, PrefixCaching: true, MaxModelLen: 4096, MaxNumSeqs: 8, MaxBatchedTokens: 4096, GPUMemory: 0.9, KVCacheDType: "auto"}, true
	case "qwen35-4b-customer", "qwen3-4b-customer":
		return inferenceModelSpec{ModelID: modelID, Endpoint: modelID + "-vllm", DeploymentName: modelID + "-production", TensorParallel: 1, PipelineParallel: 1, GPUCount: 1, PrefixCaching: false, MaxModelLen: 4096, MaxNumSeqs: 8, MaxBatchedTokens: 4096, GPUMemory: 0.9, KVCacheDType: "auto"}, true
	default:
		return inferenceModelSpec{}, false
	}
}

func inferenceModelIDFromEndpoint(endpoint string) string {
	return strings.TrimSuffix(strings.TrimSpace(endpoint), "-vllm")
}

func supportedInferenceModelSpecs() []inferenceModelSpec {
	ids := []string{"qwen38-27b-fp8", "qwen36-27b-fp8", "qwen36-27b-awq", "qwen35-4b-customer", "qwen3-4b-customer"}
	out := make([]inferenceModelSpec, 0, len(ids))
	for _, id := range ids {
		if spec, ok := inferenceModelSpecFor(id); ok {
			out = append(out, spec)
		}
	}
	return out
}

type inferenceReleaseProfile struct {
	Key              string
	Label            string
	Description      string
	MaxNumSeqs       int
	MaxBatchedTokens int
	PrefixCaching    bool
	RuntimeRequest   map[string]any
}

func inferenceReleaseProfilesForModel(modelID string) []inferenceReleaseProfile {
	spec, ok := inferenceModelSpecFor(modelID)
	if !ok {
		spec, _ = inferenceModelSpecFor(inferenceReleaseModelID)
	}
	base := map[string]any{
		"profile": "scheduler", "model_id": spec.ModelID, "tensor_parallel_size": spec.TensorParallel,
		"pipeline_parallel_size": spec.PipelineParallel, "max_num_seqs": spec.MaxNumSeqs,
		"max_num_batched_tokens": spec.MaxBatchedTokens, "prefix_caching": spec.PrefixCaching,
		"scheduling_policy": "fcfs", "max_num_partial_prefills": 1,
		"max_long_partial_prefills": 1, "long_prefill_token_threshold": 0,
		"scheduler_reserve_full_isl": true, "disable_custom_all_reduce": true,
		"stream_interval": 1, "async_scheduling": true, "gpu_memory_utilization": spec.GPUMemory,
		"max_model_len": spec.MaxModelLen, "kv_cache_dtype": spec.KVCacheDType, "speculative_decoding": "none",
	}
	schedulerBase := make(map[string]any, len(base))
	for k, v := range base {
		schedulerBase[k] = v
	}
	if modelID == inferenceReleaseModelID {
		// 27B 的稳定模板使用专门的 prefix_cache profile，让 agent 保留旧版兼容参数。
		base = map[string]any{"profile": "prefix_cache", "model_id": spec.ModelID}
	}
	high := make(map[string]any, len(schedulerBase))
	for k, v := range schedulerBase {
		high[k] = v
	}
	high["profile"] = "scheduler"
	high["max_num_seqs"] = 16
	high["max_num_batched_tokens"] = 8192
	return []inferenceReleaseProfile{
		{Key: "balanced", Label: "均衡档", Description: "优先控制 TPOT，适合默认业务流量。", MaxNumSeqs: spec.MaxNumSeqs, MaxBatchedTokens: spec.MaxBatchedTokens, PrefixCaching: spec.PrefixCaching, RuntimeRequest: base},
		{Key: "high_throughput", Label: "高并发档", Description: "提高并发和吞吐，TPOT 可能增加。", MaxNumSeqs: 16, MaxBatchedTokens: 8192, PrefixCaching: spec.PrefixCaching, RuntimeRequest: high},
	}
}

func inferenceReleaseProfiles() []inferenceReleaseProfile {
	return inferenceReleaseProfilesForModel(inferenceReleaseModelID)
}

func inferenceReleaseProfileByKey(key string) (inferenceReleaseProfile, bool) {
	return inferenceReleaseProfileByKeyForModel(key, inferenceReleaseModelID)
}

func inferenceReleaseProfileByKeyForModel(key, modelID string) (inferenceReleaseProfile, bool) {
	for _, profile := range inferenceReleaseProfilesForModel(modelID) {
		if profile.Key == key {
			return profile, true
		}
	}
	return inferenceReleaseProfile{}, false
}

func customInferenceReleaseProfileForModel(request map[string]any, modelID string) (inferenceReleaseProfile, error) {
	spec, ok := inferenceModelSpecFor(modelID)
	if !ok {
		return inferenceReleaseProfile{}, fmt.Errorf("unsupported inference model %q", modelID)
	}
	if len(request) == 0 {
		return inferenceReleaseProfile{}, errors.New("runtime_request required")
	}
	runtimeRequest := make(map[string]any, len(request)+1)
	for key, value := range request {
		runtimeRequest[key] = value
	}
	runtimeRequest["model_id"] = spec.ModelID
	profile := strings.TrimSpace(stringValue(runtimeRequest["profile"]))
	if profile == "" {
		profile = "scheduler"
		runtimeRequest["profile"] = profile
	}
	if profile != "scheduler" {
		return inferenceReleaseProfile{}, errors.New("custom release requires profile=scheduler")
	}
	maxNumSeqs := intFromAny(runtimeRequest["max_num_seqs"])
	maxBatchedTokens := intFromAny(runtimeRequest["max_num_batched_tokens"])
	if !allowedReleaseInt(maxNumSeqs, 4, 8, 12, 16, 24, 32) {
		return inferenceReleaseProfile{}, errors.New("max_num_seqs must be one of 4, 8, 12, 16, 24 or 32")
	}
	if !allowedReleaseInt(maxBatchedTokens, 2048, 4096, 8192) {
		return inferenceReleaseProfile{}, errors.New("max_num_batched_tokens must be one of 2048, 4096 or 8192")
	}
	tp := intFromAny(runtimeRequest["tensor_parallel_size"])
	pp := intFromAny(runtimeRequest["pipeline_parallel_size"])
	if tp == 0 {
		tp = spec.TensorParallel
		runtimeRequest["tensor_parallel_size"] = tp
	}
	if pp == 0 {
		pp = spec.PipelineParallel
		runtimeRequest["pipeline_parallel_size"] = pp
	}
	if tp*pp != spec.GPUCount {
		return inferenceReleaseProfile{}, fmt.Errorf("%s requires tensor_parallel_size * pipeline_parallel_size = %d", modelID, spec.GPUCount)
	}
	prefixCaching := spec.PrefixCaching
	if value, ok := runtimeRequest["prefix_caching"].(bool); ok {
		prefixCaching = value
	} else {
		runtimeRequest["prefix_caching"] = spec.PrefixCaching
	}
	return inferenceReleaseProfile{
		Key:   fmt.Sprintf("custom-%d-%d", maxNumSeqs, maxBatchedTokens),
		Label: "自定义配置", Description: "由发布参数或 YAML 生成的受控运行时配置。",
		MaxNumSeqs: maxNumSeqs, MaxBatchedTokens: maxBatchedTokens, PrefixCaching: prefixCaching,
		RuntimeRequest: runtimeRequest,
	}, nil
}

func customInferenceReleaseProfile(request map[string]any) (inferenceReleaseProfile, error) {
	return customInferenceReleaseProfileForModel(request, inferenceReleaseModelID)
}

func allowedReleaseInt(value int, allowed ...int) bool {
	for _, candidate := range allowed {
		if value == candidate {
			return true
		}
	}
	return false
}

type inferenceReleaseCandidate struct {
	Profile             string         `json:"profile"`
	Label               string         `json:"label"`
	Description         string         `json:"description"`
	Available           bool           `json:"available"`
	GatePassed          bool           `json:"gate_passed"`
	RunID               string         `json:"run_id,omitempty"`
	ReportPath          string         `json:"report_path,omitempty"`
	Scenarios           int            `json:"scenarios"`
	MinSuccessRate      float64        `json:"min_success_rate"`
	MinQualityRate      float64        `json:"min_quality_rate"`
	AverageP95TTFTMs    float64        `json:"average_p95_ttft_ms"`
	AverageP95TPOTMs    float64        `json:"average_p95_tpot_ms"`
	AverageThroughput   float64        `json:"average_output_tokens_per_second"`
	MaxP95TTFTMs        float64        `json:"max_p95_ttft_ms"`
	MaxP95TPOTMs        float64        `json:"max_p95_tpot_ms"`
	SLOTTFTLimitMs      float64        `json:"slo_ttft_limit_ms"`
	SLOTPOTLimitMs      float64        `json:"slo_tpot_limit_ms"`
	MaxNumSeqs          int            `json:"max_num_seqs"`
	MaxNumBatchedTokens int            `json:"max_num_batched_tokens"`
	RuntimeRequest      map[string]any `json:"runtime_request"`
	Error               string         `json:"error,omitempty"`
}

type inferenceReleaseProgressStage struct {
	Key    string `json:"key"`
	Label  string `json:"label"`
	State  string `json:"state"`
	Detail string `json:"detail,omitempty"`
}

type inferenceReleaseProgress struct {
	ActiveStage   string                          `json:"active_stage"`
	WeightPercent int                             `json:"weight_percent,omitempty"`
	Stages        []inferenceReleaseProgressStage `json:"stages"`
}

var inferenceWeightProgressPattern = regexp.MustCompile(`Loading safetensors checkpoint shards:\s+(\d+)% Completed`)

func deriveInferenceReleaseProgress(runtime, logs map[string]any, releaseActive bool) inferenceReleaseProgress {
	stages := []inferenceReleaseProgressStage{
		{Key: "gate", Label: "发布门禁", State: "pending"},
		{Key: "container", Label: "创建容器", State: "pending"},
		{Key: "weights", Label: "加载权重", State: "pending"},
		{Key: "compile", Label: "编译与缓存", State: "pending"},
		{Key: "health", Label: "健康检查", State: "pending"},
	}
	status, _ := runtime["status"].(string)
	if !releaseActive && status != "starting" {
		if status == "ready" {
			stages[1].State = "complete"
			stages[1].Detail = "检测到运行中的 vLLM workload"
			return inferenceReleaseProgress{ActiveStage: "manual_runtime", WeightPercent: 100, Stages: stages}
		}
		return inferenceReleaseProgress{ActiveStage: "idle", Stages: stages}
	}
	stages[0].State = "complete"
	stages[0].Detail = "模型、参数与压测证据匹配"
	if status == "ready" {
		for index := 1; index < len(stages); index++ {
			stages[index].State = "complete"
		}
		stages[1].Detail = "vLLM workload 已运行"
		modelLabel := stringValue(runtime["model"])
		if modelLabel == "" {
			modelLabel = inferenceReleaseModelID
		}
		stages[2].Detail = modelLabel + " 已加载"
		stages[3].Detail = "执行图与 KV Cache 已初始化"
		stages[4].Detail = "OpenAI-Compatible API 已就绪"
		return inferenceReleaseProgress{ActiveStage: "ready", WeightPercent: 100, Stages: stages}
	}
	stages[1].State = "complete"
	stages[1].Detail = "vLLM workload 已创建"
	stages[2].State = "active"
	stages[2].Detail = "等待模型加载日志"
	progress := inferenceReleaseProgress{ActiveStage: "weights", Stages: stages}

	lines, _ := logs["lines"].([]any)
	var text strings.Builder
	for _, raw := range lines {
		line, _ := raw.(string)
		text.WriteString(line)
		text.WriteByte('\n')
		for _, match := range inferenceWeightProgressPattern.FindAllStringSubmatch(line, -1) {
			value, _ := strconv.Atoi(match[1])
			if value > progress.WeightPercent {
				progress.WeightPercent = value
			}
		}
	}
	logText := text.String()
	if progress.WeightPercent > 0 {
		stages[2].Detail = fmt.Sprintf("checkpoint shards %d%%", progress.WeightPercent)
	}
	if strings.Contains(logText, "Model loading took") || progress.WeightPercent >= 100 {
		stages[2].State = "complete"
		stages[2].Detail = "checkpoint shards 100%"
		stages[3].State = "active"
		stages[3].Detail = "初始化执行图与 KV Cache"
		progress.ActiveStage = "compile"
		progress.WeightPercent = 100
	}
	if strings.Contains(logText, "torch.compile took") {
		stages[3].State = "complete"
		stages[3].Detail = "torch.compile 已完成"
		stages[4].State = "active"
		stages[4].Detail = "轮询 /health 与 /v1/models"
		progress.ActiveStage = "health"
	}
	progress.Stages = stages
	return progress
}

func summarizeReleaseCandidate(profile inferenceReleaseProfile, evidence inferenceBenchmarkEvidence) inferenceReleaseCandidate {
	result := inferenceReleaseCandidate{
		Profile: profile.Key, Label: profile.Label, Description: profile.Description,
		Available: true, RunID: evidence.RunID, ReportPath: evidence.ReportPath,
		MaxNumSeqs: profile.MaxNumSeqs, MaxNumBatchedTokens: profile.MaxBatchedTokens,
		RuntimeRequest: profile.RuntimeRequest, MinSuccessRate: 1, MinQualityRate: 1,
		SLOTTFTLimitMs: inferenceReleaseMaxP95TTFTMs, SLOTPOTLimitMs: inferenceReleaseMaxP95TPOTMs,
	}
	scenarios, _ := evidence.Summary["scenarios"].([]any)
	for _, raw := range scenarios {
		scenario, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		result.Scenarios++
		success, _ := asFloat(scenario["success_rate"])
		quality, _ := asFloat(scenario["quality_gate_pass_rate"])
		ttft, _ := asFloat(scenario["p95_ttft_ms"])
		tpot, _ := asFloat(scenario["p95_tpot_ms"])
		throughput, _ := asFloat(scenario["output_tokens_per_second"])
		if success < result.MinSuccessRate {
			result.MinSuccessRate = success
		}
		if quality < result.MinQualityRate {
			result.MinQualityRate = quality
		}
		if ttft > result.MaxP95TTFTMs {
			result.MaxP95TTFTMs = ttft
		}
		if tpot > result.MaxP95TPOTMs {
			result.MaxP95TPOTMs = tpot
		}
		result.AverageP95TTFTMs += ttft
		result.AverageP95TPOTMs += tpot
		result.AverageThroughput += throughput
	}
	if result.Scenarios > 0 {
		count := float64(result.Scenarios)
		result.AverageP95TTFTMs /= count
		result.AverageP95TPOTMs /= count
		result.AverageThroughput /= count
	}
	result.GatePassed = result.Scenarios > 0 && result.MinSuccessRate >= 0.99 && result.MinQualityRate >= 0.99 &&
		result.MaxP95TTFTMs <= inferenceReleaseMaxP95TTFTMs && result.MaxP95TPOTMs <= inferenceReleaseMaxP95TPOTMs
	return result
}

func (a *API) loadInferenceReleaseCandidate(ctx context.Context, profile inferenceReleaseProfile, runID, modelID string) (inferenceReleaseCandidate, error) {
	spec, ok := inferenceModelSpecFor(modelID)
	if !ok {
		return inferenceReleaseCandidate{}, fmt.Errorf("unsupported inference model %q", modelID)
	}
	row := a.Pool.QueryRow(ctx, `
		SELECT run_id, COALESCE(endpoint_id,''), COALESCE(workload,''), config, summary,
		       COALESCE(report_path,''), created_at, updated_at
		FROM benchmark_runs
		WHERE status='completed' AND endpoint_id=$1
		  AND COALESCE((config->'vllm'->>'prefix_caching')::boolean, false)=$2
		  AND COALESCE((config->'vllm'->>'max_num_seqs')::int, 0)=$3
		  AND COALESCE((config->'vllm'->>'max_num_batched_tokens')::int, 0)=$4
		  AND ($5='' OR run_id::text=$5)
		ORDER BY jsonb_array_length(COALESCE(summary->'scenarios','[]'::jsonb)) DESC, updated_at DESC
	LIMIT 1`, spec.Endpoint, profile.PrefixCaching, profile.MaxNumSeqs, profile.MaxBatchedTokens, runID)
	evidence, err := scanBenchmarkEvidence(row)
	if err != nil {
		return inferenceReleaseCandidate{}, err
	}
	return summarizeReleaseCandidate(profile, evidence), nil
}

// GET /api/inference/releases returns server-selected, parameter-matched release evidence.
func (a *API) inferenceReleaseCandidates(w http.ResponseWriter, r *http.Request) {
	requestedModelID := strings.TrimSpace(r.URL.Query().Get("model_id"))
	runtime, _, runtimeErr := a.Agent.RequestObject(r.Context(), http.MethodGet, "/api/inference/runtime", nil)
	if runtimeErr != nil {
		runtime = map[string]any{"available": false, "status": "unavailable", "error": runtimeErr.Error()}
	}
	modelID := requestedModelID
	if modelID == "" {
		if runtimeModel, ok := runtime["model"].(string); ok {
			modelID = inferenceModelIDFromEndpoint(runtimeModel)
		}
	}
	if _, ok := inferenceModelSpecFor(modelID); !ok {
		modelID = inferenceReleaseModelID
	}
	servingStatus := a.inspectAIBrixServingStatus(r.Context(), modelID)
	// A formal AIBrix release is a Kubernetes workload, not the host :8020
	// container. Use its persisted configuration, but derive the current status
	// exclusively from live workload/gateway/model-route observations.
	var activeReleaseMeta []byte
	if err := a.Pool.QueryRow(r.Context(), `SELECT metadata FROM deployments
		WHERE metadata->>'mode'='inference_runtime' AND metadata->>'release_target'='aibrix'
		  AND metadata->>'model_id'=$1 AND status IN ('running','success')
		ORDER BY started_at DESC LIMIT 1`, modelID).Scan(&activeReleaseMeta); err == nil {
		meta := jsonbObject(activeReleaseMeta)
		if releaseRuntime, ok := meta["runtime"].(map[string]any); ok {
			runtime = releaseRuntime
			runtime["status"] = servingStatus.Overall
			runtime["endpoint"] = servingStatus.GatewayEndpoint
		}
	}
	spec, _ := inferenceModelSpecFor(modelID)
	profiles := inferenceReleaseProfilesForModel(modelID)
	candidates := make([]inferenceReleaseCandidate, 0, len(profiles))
	for _, profile := range profiles {
		candidate, err := a.loadInferenceReleaseCandidate(r.Context(), profile, "", modelID)
		if err != nil {
			candidate = inferenceReleaseCandidate{
				Profile: profile.Key, Label: profile.Label, Description: profile.Description,
				MaxNumSeqs: profile.MaxNumSeqs, MaxNumBatchedTokens: profile.MaxBatchedTokens,
				RuntimeRequest: profile.RuntimeRequest, Error: err.Error(),
				SLOTTFTLimitMs: inferenceReleaseMaxP95TTFTMs, SLOTPOTLimitMs: inferenceReleaseMaxP95TPOTMs,
			}
		}
		candidates = append(candidates, candidate)
	}
	var releaseActive bool
	_ = a.Pool.QueryRow(r.Context(), `SELECT EXISTS(SELECT 1 FROM deployments
		WHERE metadata->>'mode'='inference_runtime' AND metadata->>'model_id'=$1
		  AND status IN ('running','success'))`, modelID).Scan(&releaseActive)
	logs := map[string]any{}
	if runtime["status"] == "starting" {
		logs = a.Agent.FetchObject(r.Context(), "/api/inference/runtime/logs")
	}
	payload := map[string]any{
		"model_id": modelID, "endpoint_id": spec.Endpoint,
		"release_endpoint_id": modelID + "-aibrix",
		"candidates":          candidates, "runtime": runtime,
		"serving_status": servingStatus,
		"serving_models": a.listInferenceServingModels(r.Context(), runtime),
		"progress":       deriveInferenceReleaseProgress(runtime, logs, releaseActive),
	}
	modelOptions := make([]map[string]any, 0, len(supportedInferenceModelSpecs()))
	for _, supported := range supportedInferenceModelSpecs() {
		modelOptions = append(modelOptions, map[string]any{"model_id": supported.ModelID, "endpoint_id": supported.Endpoint, "gpu_count": supported.GPUCount, "tensor_parallel_size": supported.TensorParallel, "pipeline_parallel_size": supported.PipelineParallel, "prefix_caching": supported.PrefixCaching, "aibrix_supported": supported.GPUCount == 1 && strings.Contains(supported.ModelID, "4b")})
	}
	payload["supported_models"] = modelOptions
	if maxNumSeqs, seqErr := strconv.Atoi(r.URL.Query().Get("max_num_seqs")); seqErr == nil {
		if maxBatchedTokens, tokenErr := strconv.Atoi(r.URL.Query().Get("max_num_batched_tokens")); tokenErr == nil {
			prefixCaching := r.URL.Query().Get("prefix_caching") != "false"
			requested, profileErr := customInferenceReleaseProfileForModel(map[string]any{
				"profile": "scheduler", "max_num_seqs": maxNumSeqs,
				"max_num_batched_tokens": maxBatchedTokens, "prefix_caching": prefixCaching,
			}, modelID)
			if profileErr == nil {
				candidate, candidateErr := a.loadInferenceReleaseCandidate(r.Context(), requested, r.URL.Query().Get("benchmark_run_id"), modelID)
				if candidateErr != nil {
					candidate = inferenceReleaseCandidate{Profile: requested.Key, Label: requested.Label, Description: requested.Description,
						MaxNumSeqs: requested.MaxNumSeqs, MaxNumBatchedTokens: requested.MaxBatchedTokens,
						RuntimeRequest: requested.RuntimeRequest, Error: candidateErr.Error(),
						SLOTTFTLimitMs: inferenceReleaseMaxP95TTFTMs, SLOTPOTLimitMs: inferenceReleaseMaxP95TPOTMs}
				}
				payload["requested_candidate"] = candidate
			}
		}
	}
	WriteJSON(w, http.StatusOK, payload)
}

type submitInferenceReleaseRequest struct {
	ModelVersionID  string         `json:"model_version_id"`
	Profile         string         `json:"profile"`
	RuntimeRequest  map[string]any `json:"runtime_request"`
	ReleaseSpec     string         `json:"release_spec"`
	BenchmarkRunID  string         `json:"benchmark_run_id"`
	Env             string         `json:"env"`
	Operator        string         `json:"operator"`
	RolloutStrategy string         `json:"rollout_strategy"`
	StableEndpoint  string         `json:"stable_endpoint"`
	StableModel     string         `json:"stable_model"`
	CanaryWeight    int            `json:"canary_weight"`
	ReleaseTarget   string         `json:"release_target"`
}

func (a *API) submitInferenceRelease(w http.ResponseWriter, r *http.Request) {
	var req submitInferenceReleaseRequest
	if err := decodeBody(r, &req); err != nil {
		a.badRequest(w, r, "invalid body")
		return
	}
	if req.ModelVersionID == "" {
		a.badRequest(w, r, "model_version_id required")
		return
	}
	model, err := a.Store.GetModelVersion(r.Context(), req.ModelVersionID)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			WriteError(w, r, http.StatusNotFound, "model_version_not_found", "model version not found")
			return
		}
		a.fail(w, r, err)
		return
	}
	modelSpec, supported := inferenceModelSpecFor(model.ModelID)
	if !supported {
		WriteError(w, r, http.StatusConflict, "unsupported_inference_model", fmt.Sprintf("模型 %s 尚未配置本机推理运行时，请先登记支持的 vLLM 模型", model.ModelID))
		return
	}
	releaseTarget := strings.ToLower(strings.TrimSpace(req.ReleaseTarget))
	if releaseTarget == "" {
		releaseTarget = "aibrix"
	}
	if releaseTarget != "aibrix" && releaseTarget != "direct" {
		a.badRequest(w, r, "release_target must be aibrix or direct")
		return
	}
	if releaseTarget == "aibrix" {
		if a.K8s == nil {
			WriteError(w, r, http.StatusServiceUnavailable, "k8s_unavailable", "AIBrix 发布需要可用的 Kubernetes 控制面")
			return
		}
		if !a.AllowK8sWrites {
			WriteError(w, r, http.StatusForbidden, "k8s_writes_disabled", "AIBrix 发布未启用，请设置 ALLOW_K8S_WRITES=true")
			return
		}
		if modelSpec.GPUCount != 1 || !strings.Contains(model.ModelID, "4b") {
			WriteError(w, r, http.StatusConflict, "aibrix_model_unsupported", "当前本地 AIBrix PVC 只支持单卡 4B 模型；27B 请使用本机调试运行时或准备独立模型卷")
			return
		}
	}
	var profile inferenceReleaseProfile
	var ok bool
	var profileErr error
	if len(req.RuntimeRequest) > 0 {
		profile, profileErr = customInferenceReleaseProfileForModel(req.RuntimeRequest, model.ModelID)
	} else {
		profile, ok = inferenceReleaseProfileByKeyForModel(req.Profile, model.ModelID)
		if !ok {
			profileErr = errors.New("profile must be a known template or runtime_request must be provided")
		}
	}
	if profileErr != nil {
		a.badRequest(w, r, profileErr.Error())
		return
	}
	if model.Status == "deprecated" {
		WriteError(w, r, http.StatusConflict, "release_model_deprecated", "deprecated model version cannot be released")
		return
	}
	rolloutStrategy := strings.ToLower(strings.TrimSpace(req.RolloutStrategy))
	if rolloutStrategy == "" {
		rolloutStrategy = "full"
	}
	if rolloutStrategy != "full" && rolloutStrategy != "canary" {
		a.badRequest(w, r, "rollout_strategy must be full or canary")
		return
	}
	stableModel := strings.TrimSpace(req.StableModel)
	releaseEndpointID := modelSpec.Endpoint
	if releaseTarget == "aibrix" {
		releaseEndpointID = model.ModelID + "-aibrix"
	}
	if rolloutStrategy == "canary" {
		req.StableEndpoint = strings.TrimSpace(req.StableEndpoint)
		if req.StableEndpoint == "" || req.StableEndpoint == releaseEndpointID {
			a.badRequest(w, r, "canary release requires a distinct stable_endpoint")
			return
		}
		if req.CanaryWeight == 0 {
			req.CanaryWeight = 10
		}
		if req.CanaryWeight < 1 || req.CanaryWeight > 50 {
			a.badRequest(w, r, "canary_weight must be between 1 and 50")
			return
		}
		var stableStatus, registeredModel string
		err := a.Pool.QueryRow(r.Context(), `SELECT status, COALESCE(model_id,'') FROM service_instances WHERE name=$1`, req.StableEndpoint).Scan(&stableStatus, &registeredModel)
		if err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				WriteError(w, r, http.StatusConflict, "stable_endpoint_missing", "灰度基线服务不存在："+req.StableEndpoint)
				return
			}
			a.fail(w, r, err)
			return
		}
		if stableStatus != "healthy" {
			WriteError(w, r, http.StatusConflict, "stable_endpoint_unhealthy", "灰度基线服务当前不可用："+req.StableEndpoint)
			return
		}
		if stableModel == "" {
			stableModel = registeredModel
		}
	}
	candidate, err := a.loadInferenceReleaseCandidate(r.Context(), profile, req.BenchmarkRunID, model.ModelID)
	if err != nil {
		WriteError(w, r, http.StatusConflict, "release_evidence_missing", "没有与当前核心运行参数完全匹配的已完成压测")
		return
	}
	if !candidate.GatePassed {
		WriteError(w, r, http.StatusConflict, "release_gate_failed", fmt.Sprintf(
			"发布门禁未通过：成功率/质量需 >=99%%，所有场景 P95 TTFT <= %.0fms、P95 TPOT <= %.0fms",
			inferenceReleaseMaxP95TTFTMs, inferenceReleaseMaxP95TPOTMs))
		return
	}
	if trainingID, active, err := a.activeTrainingJob(r.Context()); err != nil {
		a.fail(w, r, err)
		return
	} else if active {
		WriteError(w, r, http.StatusConflict, "gpu_lane_busy", "训练任务 "+trainingID+" 正在占用 GPU 实验通道")
		return
	}
	if benchmarkID, active, err := a.activeBenchmarkRun(r.Context()); err != nil {
		a.fail(w, r, err)
		return
	} else if active {
		WriteError(w, r, http.StatusConflict, "benchmark_active", "推理压测 "+benchmarkID+" 仍在运行")
		return
	}
	var activeRelease bool
	if err := a.Pool.QueryRow(r.Context(), `SELECT EXISTS(
		SELECT 1 FROM deployments WHERE status='running' AND metadata->>'mode'='inference_runtime'
	)`).Scan(&activeRelease); err != nil {
		a.fail(w, r, err)
		return
	}
	if activeRelease {
		WriteError(w, r, http.StatusConflict, "release_in_progress", "已有推理发布正在执行")
		return
	}

	operator := a.actor(r, req.Operator)
	env := orDefault(req.Env, "prod")
	previous, _, previousErr := a.Agent.RequestObject(r.Context(), http.MethodGet, "/api/inference/runtime", nil)
	if previousErr != nil {
		previous = map[string]any{"status": "unavailable", "error": previousErr.Error()}
	}
	var previousReleaseID string
	_ = a.Pool.QueryRow(r.Context(), `SELECT id FROM deployments
		WHERE status='success' AND metadata->>'mode'='inference_runtime'
		ORDER BY started_at DESC LIMIT 1`).Scan(&previousReleaseID)
	meta := map[string]any{
		"owner": operator, "mode": "inference_runtime", "phase": "queued",
		"model_id": model.ModelID, "model_version_id": model.ID, "endpoint_id": releaseEndpointID,
		"release_target":  releaseTarget,
		"release_profile": profile.Key, "runtime_request": profile.RuntimeRequest,
		"release_spec":     req.ReleaseSpec,
		"benchmark_run_id": candidate.RunID, "benchmark_report": candidate.ReportPath,
		"gate": map[string]any{"passed": true, "scenarios": candidate.Scenarios, "min_success_rate": candidate.MinSuccessRate, "min_quality_rate": candidate.MinQualityRate,
			"max_p95_ttft_ms": candidate.MaxP95TTFTMs, "max_p95_tpot_ms": candidate.MaxP95TPOTMs,
			"slo_ttft_limit_ms": candidate.SLOTTFTLimitMs, "slo_tpot_limit_ms": candidate.SLOTPOTLimitMs},
		"previous_runtime": previous,
		"rollout": map[string]any{"strategy": rolloutStrategy, "stable_endpoint": req.StableEndpoint,
			"stable_model": stableModel, "canary_weight": req.CanaryWeight},
	}
	if releaseTarget == "aibrix" {
		meta["k8s_namespace"] = "default"
		meta["k8s_deployment"] = model.ModelID + "-candidate"
		meta["aibrix_gateway_endpoint"] = "aibrix-gateway"
	}
	if previousReleaseID != "" {
		meta["previous_release_id"] = previousReleaseID
	}
	id, err := a.Store.CreateDeploymentMeta(r.Context(), modelSpec.DeploymentName, model.Version, env, meta)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	a.Store.Audit(r.Context(), operator, "operator", "inference.release.trigger", "model", model.ModelID+":"+model.Version,
		map[string]any{"deployment_id": id, "profile": profile.Key, "benchmark_run_id": candidate.RunID, "runtime_request": profile.RuntimeRequest})
	go a.runInferenceRelease(id, model.ID, model.ModelID, model.Version, operator, profile, meta)
	WriteJSON(w, http.StatusAccepted, map[string]any{"id": id, "status": "running", "profile": profile.Key, "benchmark_run_id": candidate.RunID})
}

func (a *API) runInferenceRelease(id, modelVersionID, modelID, version, operator string, profile inferenceReleaseProfile, meta map[string]any) {
	defer func() {
		if rec := recover(); rec != nil {
			slog.Error("inference release panic", "id", id, "err", rec)
			if stringValue(meta["release_target"]) == "aibrix" {
				a.failAIBrixInferenceRelease(context.Background(), id, modelVersionID, modelID, version, operator, meta, fmt.Sprintf("panic: %v", rec))
			} else {
				a.failInferenceRelease(context.Background(), id, modelVersionID, modelID, version, operator, meta, fmt.Sprintf("panic: %v", rec))
			}
		}
	}()
	if stringValue(meta["release_target"]) == "aibrix" {
		a.runAIBrixInferenceRelease(id, modelVersionID, modelID, version, operator, profile, meta)
		return
	}
	ctx := context.Background()
	a.inferenceReleaseEvent(ctx, id, meta, "starting", "starting validated vLLM production workload")
	previous, _, _ := a.Agent.RequestObject(ctx, http.MethodGet, "/api/inference/runtime", nil)
	if status, _ := previous["status"].(string); status == "ready" || status == "starting" {
		a.inferenceReleaseEvent(ctx, id, meta, "replacing", "stopping previous inference workload")
		if _, _, err := a.Agent.RequestObject(ctx, http.MethodDelete, "/api/inference/runtime", nil); err != nil {
			a.failInferenceRelease(ctx, id, modelVersionID, modelID, version, operator, meta, "failed to stop previous runtime: "+err.Error())
			return
		}
	}
	result, _, err := a.Agent.RequestObject(ctx, http.MethodPost, "/api/inference/runtime", profile.RuntimeRequest)
	if err != nil {
		a.failInferenceRelease(ctx, id, modelVersionID, modelID, version, operator, meta, "runtime start failed: "+err.Error())
		return
	}
	meta["runtime"] = result
	a.inferenceReleaseEvent(ctx, id, meta, "warming", "container created; waiting for OpenAI-compatible health check")

	deadline := time.Now().Add(inferenceReleaseTimeout)
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	for range ticker.C {
		status, _, statusErr := a.Agent.RequestObject(ctx, http.MethodGet, "/api/inference/runtime", nil)
		if statusErr != nil {
			meta["message"] = "runtime status unavailable: " + statusErr.Error()
			a.persistDeployMeta(ctx, id, meta)
		} else {
			meta["runtime"] = status
			a.persistDeployMeta(ctx, id, meta)
			switch runtimeStatus, _ := status["status"].(string); runtimeStatus {
			case "ready":
				a.finishInferenceRelease(ctx, id, modelVersionID, modelID, version, operator, meta, "succeeded", "vLLM production endpoint is ready")
				return
			case "error", "stopped":
				a.failInferenceRelease(ctx, id, modelVersionID, modelID, version, operator, meta, "runtime entered "+runtimeStatus)
				return
			}
		}
		if time.Now().After(deadline) {
			a.failInferenceRelease(ctx, id, modelVersionID, modelID, version, operator, meta, "runtime readiness timed out")
			return
		}
	}
}

func (a *API) runAIBrixInferenceRelease(id, modelVersionID, modelID, version, operator string, profile inferenceReleaseProfile, meta map[string]any) {
	ctx := context.Background()
	namespace := orDefault(stringValue(meta["k8s_namespace"]), "default")
	deploymentName := orDefault(stringValue(meta["k8s_deployment"]), modelID+"-candidate")
	a.inferenceReleaseEvent(ctx, id, meta, "starting", "creating AIBrix candidate workload")

	// The host debug runtime and the Kubernetes candidate share the second GPU
	// on the local two-card node. Formal publication owns that lane, so stop the
	// debug runtime before asking the scheduler to place the candidate Pod.
	previous, _, _ := a.Agent.RequestObject(ctx, http.MethodGet, "/api/inference/runtime", nil)
	if status := stringValue(previous["status"]); status == "ready" || status == "starting" {
		meta["previous_runtime"] = previous
		a.inferenceReleaseEvent(ctx, id, meta, "replacing", "stopping host debug runtime to release the candidate GPU")
		if _, _, err := a.Agent.RequestObject(ctx, http.MethodDelete, "/api/inference/runtime", nil); err != nil {
			a.failAIBrixInferenceRelease(ctx, id, modelVersionID, modelID, version, operator, meta, "failed to stop host debug runtime: "+err.Error())
			return
		}
	}

	request := profile.RuntimeRequest
	rollout, _ := meta["rollout"].(map[string]any)
	track := "candidate"
	if strings.EqualFold(stringValue(rollout["strategy"]), "full") {
		track = "stable"
	}
	spec := k8s.AIBrixModelSpec{
		Namespace: namespace, DeploymentName: deploymentName, ServiceName: modelID, ModelID: modelID, Version: version, Track: track,
		Image: "local/vllm-openai:qwen35-v0.19.1-deepepfix", PVCName: "qwen35-4b-model-store", ModelPath: "/models/Qwen3.5-4B",
		TensorParallel: intFromAny(request["tensor_parallel_size"]), PipelineParallel: intFromAny(request["pipeline_parallel_size"]),
		MaxModelLen: intFromAny(request["max_model_len"]), MaxNumSeqs: intFromAny(request["max_num_seqs"]),
		MaxNumBatchedTokens:  intFromAny(request["max_num_batched_tokens"]),
		GPUMemoryUtilization: releaseFloat(request["gpu_memory_utilization"], 0.9),
		PrefixCaching:        releaseBool(request["prefix_caching"], false), AsyncScheduling: releaseBool(request["async_scheduling"], true),
		KVCacheDType: orDefault(stringValue(request["kv_cache_dtype"]), "auto"),
	}
	if spec.TensorParallel == 0 {
		spec.TensorParallel = 1
	}
	if spec.PipelineParallel == 0 {
		spec.PipelineParallel = 1
	}
	if spec.MaxModelLen == 0 {
		spec.MaxModelLen = 4096
	}
	if spec.MaxNumSeqs == 0 {
		spec.MaxNumSeqs = 8
	}
	if spec.MaxNumBatchedTokens == 0 {
		spec.MaxNumBatchedTokens = 4096
	}
	if err := a.K8s.UpsertAIBrixModelDeployment(ctx, spec); err != nil {
		a.failAIBrixInferenceRelease(ctx, id, modelVersionID, modelID, version, operator, meta, "failed to apply AIBrix workload: "+err.Error())
		return
	}
	meta["runtime"] = map[string]any{
		"status": "starting", "model": modelID, "endpoint": a.aibrixGatewayEndpoint(), "release_target": "aibrix",
		"k8s_namespace": namespace, "k8s_deployment": deploymentName, "config": request,
	}
	a.inferenceReleaseEvent(ctx, id, meta, "warming", "AIBrix candidate Pod created; waiting for Kubernetes readiness")

	deadline := time.Now().Add(inferenceReleaseTimeout)
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	gatewayBaseURL := a.aibrixGatewayEndpoint()
	for range ticker.C {
		status, err := a.K8s.RolloutStatus(ctx, namespace, deploymentName)
		if err != nil {
			meta["message"] = "AIBrix rollout status unavailable: " + err.Error()
			a.persistDeployMeta(ctx, id, meta)
		} else {
			meta["k8s_rollout"] = map[string]any{"desired": status.Desired, "updated": status.Updated, "ready": status.Ready, "available": status.Available, "reason": status.Reason, "message": status.Message}
			if status.Failed {
				a.failAIBrixInferenceRelease(ctx, id, modelVersionID, modelID, version, operator, meta, "AIBrix rollout failed: "+status.Message)
				return
			}
			if status.Complete {
				probeURL, probeStatus, probeDetail, probeErr := probeAIBrix(ctx, gatewayBaseURL, modelID)
				meta["aibrix_probe"] = map[string]any{
					"url": probeURL, "http_status": probeStatus, "detail": probeDetail,
					"checked_at": time.Now().UTC().Format(time.RFC3339Nano),
				}
				if probeErr != nil {
					meta["message"] = "Kubernetes workload is ready; waiting for AIBrix route: " + probeErr.Error()
					a.persistDeployMeta(ctx, id, meta)
					if time.Now().After(deadline) {
						a.failAIBrixInferenceRelease(ctx, id, modelVersionID, modelID, version, operator, meta, "AIBrix gateway readiness timed out: "+probeErr.Error()+": "+probeDetail)
						return
					}
					continue
				}
				meta["runtime"] = map[string]any{
					"status": "ready", "model": modelID, "endpoint": a.aibrixGatewayEndpoint(), "release_target": "aibrix",
					"k8s_namespace": namespace, "k8s_deployment": deploymentName, "config": request,
				}
				a.finishInferenceRelease(ctx, id, modelVersionID, modelID, version, operator, meta, "succeeded", "AIBrix workload and gateway route are ready")
				return
			}
			a.persistDeployMeta(ctx, id, meta)
		}
		if time.Now().After(deadline) {
			a.failAIBrixInferenceRelease(ctx, id, modelVersionID, modelID, version, operator, meta, "AIBrix rollout readiness timed out")
			return
		}
	}
}

func (a *API) failAIBrixInferenceRelease(ctx context.Context, id, modelVersionID, modelID, version, operator string, meta map[string]any, message string) {
	namespace := orDefault(stringValue(meta["k8s_namespace"]), "default")
	deploymentName := stringValue(meta["k8s_deployment"])
	if deploymentName != "" && a.K8s != nil {
		if err := a.K8s.ScaleAIBrixModelDeployment(ctx, namespace, deploymentName, 0); err != nil {
			meta["rollback_error"] = err.Error()
		}
	}
	previous, _ := meta["previous_runtime"].(map[string]any)
	if request, ok := runtimeRequestForRestore(previous); ok {
		a.inferenceReleaseEvent(ctx, id, meta, "rolling_back", "restoring previous host debug runtime")
		if restored, _, err := a.Agent.RequestObject(ctx, http.MethodPost, "/api/inference/runtime", request); err != nil {
			meta["runtime_restore_error"] = err.Error()
		} else {
			meta["rollback_runtime"] = restored
		}
	}
	a.finishInferenceRelease(ctx, id, modelVersionID, modelID, version, operator, meta, "failed", message)
}

func releaseFloat(value any, fallback float64) float64 {
	switch typed := value.(type) {
	case float64:
		return typed
	case float32:
		return float64(typed)
	case int:
		return float64(typed)
	default:
		return fallback
	}
}

func releaseBool(value any, fallback bool) bool {
	typed, ok := value.(bool)
	if !ok {
		return fallback
	}
	return typed
}

func (a *API) failInferenceRelease(ctx context.Context, id, modelVersionID, modelID, version, operator string, meta map[string]any, message string) {
	// 在删除失败容器前抓取 vLLM stderr 摘要，避免最终只剩下笼统的 stopped。
	if logs, _, err := a.Agent.RequestObject(ctx, http.MethodGet, "/api/inference/runtime/logs", nil); err == nil {
		if excerpt := runtimeFailureExcerpt(logs); excerpt != "" {
			message += ": " + excerpt
			meta["runtime_error_excerpt"] = excerpt
		}
		if signatures, ok := logs["signatures"]; ok {
			meta["runtime_log_signatures"] = signatures
		}
	}
	_, _, _ = a.Agent.RequestObject(ctx, http.MethodDelete, "/api/inference/runtime", nil)
	previous, _ := meta["previous_runtime"].(map[string]any)
	if request, ok := runtimeRequestForRestore(previous); ok {
		a.inferenceReleaseEvent(ctx, id, meta, "rolling_back", "restoring previous inference runtime")
		if restored, _, err := a.Agent.RequestObject(ctx, http.MethodPost, "/api/inference/runtime", request); err != nil {
			meta["rollback_error"] = err.Error()
		} else {
			meta["rollback_runtime"] = restored
			meta["rollback_status"] = "starting"
		}
	}
	a.finishInferenceRelease(ctx, id, modelVersionID, modelID, version, operator, meta, "failed", message)
}

// runtimeFailureExcerpt 从 Agent 返回的容器日志中提取少量错误行，写入部署元数据。
// 完整日志仍可在容器存在时通过 /api/inference/runtime/logs 查看，数据库只保留摘要。
func runtimeFailureExcerpt(logs map[string]any) string {
	raw, ok := logs["lines"].([]any)
	if !ok {
		return ""
	}
	selected := make([]string, 0, 3)
	seen := map[string]bool{}
	// Root-cause exceptions are normally near the end of a Python traceback.
	// Prefer those over the first generic "EngineCore failed" lines.
	priorities := [][]string{
		{"free memory on device", "cuda out of memory", "insufficient gpu memory"},
		{"valueerror:", "runtimeerror:"},
		{"error", "traceback"},
	}
	for _, patterns := range priorities {
		for index := len(raw) - 1; index >= 0 && len(selected) < 3; index-- {
			line, ok := raw[index].(string)
			if !ok {
				continue
			}
			lower := strings.ToLower(line)
			matched := false
			for _, pattern := range patterns {
				if strings.Contains(lower, pattern) {
					matched = true
					break
				}
			}
			line = strings.TrimSpace(line)
			if !matched || line == "" || seen[line] {
				continue
			}
			seen[line] = true
			if len([]rune(line)) > 360 {
				line = string([]rune(line)[:360]) + "…"
			}
			selected = append(selected, line)
		}
	}
	return strings.Join(selected, " | ")
}

func runtimeRequestForRestore(status map[string]any) (map[string]any, bool) {
	if status == nil || status["status"] != "ready" {
		return nil, false
	}
	profile, _ := status["profile"].(string)
	modelID := stringValue(status["model"])
	if profile == "baseline" || profile == "prefix_cache" {
		request := map[string]any{"profile": profile}
		if modelID != "" {
			request["model_id"] = modelID
		}
		return request, true
	}
	if profile != "scheduler" {
		return nil, false
	}
	config, _ := status["config"].(map[string]any)
	request := map[string]any{"profile": "scheduler"}
	if modelID != "" {
		request["model_id"] = modelID
	}
	for _, key := range []string{
		"tensor_parallel_size", "pipeline_parallel_size", "pipeline_layer_partition",
		"max_num_seqs", "max_num_batched_tokens", "scheduling_policy", "max_num_partial_prefills",
		"max_long_partial_prefills", "long_prefill_token_threshold", "stream_interval", "prefix_caching",
		"async_scheduling", "scheduler_reserve_full_isl", "disable_custom_all_reduce", "profiling",
		"gpu_memory_utilization", "max_model_len", "kv_cache_dtype", "speculative_decoding",
	} {
		if value, exists := config[key]; exists {
			request[key] = value
		}
	}
	return request, true
}

func (a *API) finishInferenceRelease(ctx context.Context, id, modelVersionID, modelID, version, operator string, meta map[string]any, phase, message string) {
	a.inferenceReleaseEvent(ctx, id, meta, phase, message)
	status := "failed"
	if phase == "succeeded" {
		status = "success"
		_, _, _ = a.Store.UpdateModelStatus(ctx, modelVersionID, "serving")
		if previousID, _ := meta["previous_release_id"].(string); previousID != "" && previousID != id {
			_, _ = a.Pool.Exec(ctx, `UPDATE deployments SET status='rolled_back', finished_at=now(),
				metadata=jsonb_set(metadata,'{phase}','"superseded"'::jsonb) WHERE id=$1::uuid`, previousID)
		}
		endpointID := stringValue(meta["endpoint_id"])
		if endpointID == "" {
			if spec, ok := inferenceModelSpecFor(modelID); ok {
				endpointID = spec.Endpoint
			}
		}
		if endpointID != "" {
			runtime, _ := meta["runtime"].(map[string]any)
			releaseTarget := orDefault(stringValue(meta["release_target"]), "direct")
			gpuID := gpuDeviceIDFromRuntime(runtime)
			baseURL := "http://127.0.0.1:8020/v1"
			kind := "vllm"
			if releaseTarget == "aibrix" {
				baseURL = a.aibrixGatewayEndpoint()
				kind = "aibrix"
				gpuID = "k8s"
			}
			serviceMeta, _ := json.Marshal(map[string]any{
				"managed_by": "model-release", "scope": "production", "release_target": releaseTarget,
				"release_id": id, "k8s_namespace": meta["k8s_namespace"], "k8s_deployment": meta["k8s_deployment"],
			})
			_, _ = a.Pool.Exec(ctx, `INSERT INTO service_instances
				(name, base_url, model_id, kind, gpu_id, routing_role, status, metadata, last_checked_at)
				VALUES ($1,$2,$3,$4,$5,'model','healthy',$6::jsonb,now())
				ON CONFLICT (name) DO UPDATE SET base_url=EXCLUDED.base_url, model_id=EXCLUDED.model_id,
					kind=EXCLUDED.kind, gpu_id=EXCLUDED.gpu_id, status='healthy', last_checked_at=now(),
					metadata=service_instances.metadata || EXCLUDED.metadata`, endpointID, baseURL, modelID, kind, gpuID, serviceMeta)
			// 每个模型拥有独立的生产路由策略。灰度发布保留一个已健康的稳定入口，
			// 把少量流量导向新版本；直接全量则只保留当前已验收 endpoint。
			policyName := inferenceProductionPolicyName(modelID, releaseTarget)
			rollout, _ := meta["rollout"].(map[string]any)
			rolloutStrategy := orDefault(stringValue(rollout["strategy"]), "full")
			rolloutPhase := "full"
			variants := []routingVariant{{Label: modelID + "-" + version + "-stable", Endpoint: endpointID, Model: modelID, Weight: 100}}
			previousVariants := []routingVariant{}
			if rolloutStrategy == "canary" {
				stableEndpoint := stringValue(rollout["stable_endpoint"])
				stableModel := stringValue(rollout["stable_model"])
				canaryWeight := intValue(rollout["canary_weight"])
				if canaryWeight < 1 || canaryWeight > 50 {
					canaryWeight = 10
				}
				stableLabel := orDefault(stableModel, stableEndpoint) + "-stable"
				previousVariants = []routingVariant{
					{Label: stableLabel, Endpoint: stableEndpoint, Model: stableModel, Weight: 100},
					{Label: modelID + "-" + version + "-canary", Endpoint: endpointID, Model: modelID, Weight: 0},
				}
				variants = []routingVariant{
					{Label: stableLabel, Endpoint: stableEndpoint, Model: stableModel, Weight: 100 - canaryWeight},
					{Label: modelID + "-" + version + "-canary", Endpoint: endpointID, Model: modelID, Weight: canaryWeight},
				}
				rolloutPhase = "canary"
			} else {
				var currentJSON []byte
				if err := a.Pool.QueryRow(ctx, `SELECT variants FROM routing_policies WHERE name=$1`, policyName).Scan(&currentJSON); err == nil {
					_ = json.Unmarshal(currentJSON, &previousVariants)
				}
			}
			variantsJSON, _ := json.Marshal(variants)
			routingMeta := map[string]any{"source": "model_release", "rollout_strategy": rolloutStrategy,
				"rollout_phase": rolloutPhase, "deployment_id": id, "model_version_id": modelVersionID,
				"canary_weight": intValue(rollout["canary_weight"]), "release_target": releaseTarget,
				"k8s_namespace": meta["k8s_namespace"], "k8s_deployment": meta["k8s_deployment"]}
			if len(previousVariants) > 0 {
				routingMeta["prev_variants"] = previousVariants
			}
			routingMetaJSON, _ := json.Marshal(routingMeta)
			description := "模型 " + modelID + " 的正式发布入口（发布中心同步 · " + map[string]string{"canary": "灰度", "full": "全量"}[rolloutPhase] + "）"
			_, _ = a.Pool.Exec(ctx, `INSERT INTO routing_policies (name, description, enabled, variants, metadata, created_by)
				VALUES ($1, $2, true, $3::jsonb, $4::jsonb, $5)
				ON CONFLICT (name) DO UPDATE SET enabled=true, variants=EXCLUDED.variants,
					metadata=COALESCE(routing_policies.metadata,'{}'::jsonb) || EXCLUDED.metadata,
					description=EXCLUDED.description, updated_at=now()`, policyName, description, variantsJSON, routingMetaJSON, operator)
			a.Store.Audit(ctx, operator, "operator", "inference.release.routing_synced", "routing_policy", policyName,
				map[string]any{"deployment_id": id, "rollout_phase": rolloutPhase, "variants": variants})
		}
	}
	_, _ = a.Store.FinishDeployment(ctx, id, status)
	a.Store.Audit(ctx, operator, "operator", "inference.release."+phase, "model", modelID+":"+version,
		map[string]any{"deployment_id": id, "message": message})
}

func inferenceProductionPolicyName(modelID, releaseTarget string) string {
	if releaseTarget == "aibrix" {
		return modelID + "-production"
	}
	return modelID + "-direct-production"
}

func (a *API) inferenceReleaseEvent(ctx context.Context, id string, meta map[string]any, phase, message string) {
	a.deployEvent(ctx, id, meta, phase, message)
}

// DELETE /api/inference/releases/{id} takes that exact managed release offline.
// Multiple model workloads may coexist, so "latest globally" is not a valid
// ownership or liveness check.
func (a *API) stopInferenceRelease(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if benchmarkID, active, err := a.activeBenchmarkRun(r.Context()); err != nil {
		a.fail(w, r, err)
		return
	} else if active {
		WriteError(w, r, http.StatusConflict, "benchmark_active", "推理压测 "+benchmarkID+" 仍在运行")
		return
	}
	var latestStatus string
	err := a.Pool.QueryRow(r.Context(), `SELECT status FROM deployments
		WHERE id=$1::uuid AND metadata->>'mode'='inference_runtime'`, id).Scan(&latestStatus)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			WriteError(w, r, http.StatusConflict, "release_not_active", "没有运行中的推理发布")
			return
		}
		a.fail(w, r, err)
		return
	}
	if latestStatus != "running" && latestStatus != "success" {
		WriteError(w, r, http.StatusConflict, "release_not_active", "当前推理发布已经下线")
		return
	}
	var modelVersionID, releaseTarget, k8sNamespace, k8sDeployment string
	_ = a.Pool.QueryRow(r.Context(), `SELECT COALESCE(metadata->>'model_version_id',''),
		COALESCE(metadata->>'release_target','direct'), COALESCE(metadata->>'k8s_namespace','default'),
		COALESCE(metadata->>'k8s_deployment','') FROM deployments WHERE id=$1::uuid`, id).
		Scan(&modelVersionID, &releaseTarget, &k8sNamespace, &k8sDeployment)
	if releaseTarget == "aibrix" {
		if a.K8s == nil || k8sDeployment == "" {
			WriteError(w, r, http.StatusServiceUnavailable, "k8s_unavailable", "AIBrix workload information is unavailable")
			return
		}
		if err := a.K8s.ScaleAIBrixModelDeployment(r.Context(), k8sNamespace, k8sDeployment, 0); err != nil {
			WriteError(w, r, http.StatusBadGateway, "inference_stop_failed", err.Error())
			return
		}
	} else if _, _, err := a.Agent.RequestObject(r.Context(), http.MethodDelete, "/api/inference/runtime", nil); err != nil {
		WriteError(w, r, http.StatusBadGateway, "inference_stop_failed", err.Error())
		return
	}
	_, _ = a.Pool.Exec(r.Context(), `UPDATE deployments SET status='rolled_back', finished_at=now(),
		metadata=jsonb_set(jsonb_set(metadata,'{phase}','"stopped"'::jsonb),'{message}','"production workload stopped"'::jsonb)
		WHERE id=$1::uuid`, id)
	if modelVersionID != "" {
		_, _, _ = a.Store.UpdateModelStatus(r.Context(), modelVersionID, "registered")
	}
	var endpointID string
	_ = a.Pool.QueryRow(r.Context(), `SELECT COALESCE(metadata->>'endpoint_id','') FROM deployments WHERE id=$1::uuid`, id).Scan(&endpointID)
	if endpointID == "" {
		endpointID = inferenceReleaseEndpoint
	}
	_, _ = a.Pool.Exec(r.Context(), `UPDATE service_instances SET status='unreachable', last_checked_at=now() WHERE name=$1`, endpointID)
	modelID := strings.TrimSuffix(strings.TrimSuffix(endpointID, "-aibrix"), "-vllm")
	if modelID != "" {
		_, _ = a.Pool.Exec(r.Context(), `UPDATE routing_policies SET enabled=false, updated_at=now() WHERE name=$1`, inferenceProductionPolicyName(modelID, releaseTarget))
	}
	operator := a.actor(r, "")
	a.Store.Audit(r.Context(), operator, "operator", "inference.release.stopped", "deployment", id, nil)
	WriteJSON(w, http.StatusOK, map[string]any{"id": id, "status": "rolled_back", "runtime_status": "stopped"})
}

// DELETE /api/inference/serving-models/{model_id}?target=aibrix|direct
// reconciles the selected live workload to zero even when it predates the
// release controller. This is intentionally separate from deleting release
// history: immutable deployment/audit records remain available after stop.
func (a *API) stopInferenceServingModel(w http.ResponseWriter, r *http.Request) {
	modelID := strings.TrimSpace(chi.URLParam(r, "model_id"))
	if _, supported := inferenceModelSpecFor(modelID); !supported {
		WriteError(w, r, http.StatusNotFound, "inference_model_not_found", "不支持的推理模型")
		return
	}
	if benchmarkID, active, err := a.activeBenchmarkRun(r.Context()); err != nil {
		a.fail(w, r, err)
		return
	} else if active {
		WriteError(w, r, http.StatusConflict, "benchmark_active", "推理压测 "+benchmarkID+" 仍在运行")
		return
	}
	target := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("target")))
	if target == "" {
		target = "aibrix"
	}
	if target != "aibrix" && target != "direct" {
		WriteError(w, r, http.StatusBadRequest, "invalid_release_target", "target 必须是 aibrix 或 direct")
		return
	}

	operator := a.actor(r, "")
	if target == "aibrix" {
		truth := a.inspectAIBrixServingStatus(r.Context(), modelID)
		if !truth.CanStop {
			WriteError(w, r, http.StatusConflict, "serving_model_not_active", "该模型的 AIBrix workload 当前未运行")
			return
		}
		if a.K8s == nil {
			WriteError(w, r, http.StatusServiceUnavailable, "k8s_unavailable", "Kubernetes collector unavailable")
			return
		}
		if err := a.K8s.ScaleAIBrixModelDeployment(r.Context(), truth.Namespace, truth.Deployment, 0); err != nil {
			WriteError(w, r, http.StatusBadGateway, "inference_stop_failed", err.Error())
			return
		}
		_, _ = a.Pool.Exec(r.Context(), `UPDATE deployments SET status='rolled_back', finished_at=now(),
			metadata=jsonb_set(jsonb_set(metadata,'{phase}','"stopped"'::jsonb),'{message}','"production workload stopped"'::jsonb)
			WHERE metadata->>'mode'='inference_runtime' AND metadata->>'model_id'=$1
			  AND COALESCE(metadata->>'release_target','direct')='aibrix' AND status IN ('running','success')`, modelID)
		_, _ = a.Pool.Exec(r.Context(), `UPDATE service_instances SET status='unreachable', last_checked_at=now()
			WHERE model_id=$1 AND kind='aibrix' AND routing_role='model'`, modelID)
		_, _ = a.Pool.Exec(r.Context(), `UPDATE routing_policies SET enabled=false, updated_at=now() WHERE name=$1`, inferenceProductionPolicyName(modelID, "aibrix"))
	} else {
		runtime, _, err := a.Agent.RequestObject(r.Context(), http.MethodGet, "/api/inference/runtime", nil)
		if err != nil {
			WriteError(w, r, http.StatusBadGateway, "runtime_unavailable", err.Error())
			return
		}
		actualModel := inferenceModelIDFromEndpoint(stringValue(runtime["model"]))
		status := strings.ToLower(strings.TrimSpace(stringValue(runtime["status"])))
		if actualModel != modelID || (status != "ready" && status != "starting") {
			WriteError(w, r, http.StatusConflict, "serving_model_not_active", "该模型的直连 vLLM runtime 当前未运行")
			return
		}
		if _, _, err := a.Agent.RequestObject(r.Context(), http.MethodDelete, "/api/inference/runtime", nil); err != nil {
			WriteError(w, r, http.StatusBadGateway, "inference_stop_failed", err.Error())
			return
		}
		_, _ = a.Pool.Exec(r.Context(), `UPDATE deployments SET status='rolled_back', finished_at=now(),
			metadata=jsonb_set(jsonb_set(metadata,'{phase}','"stopped"'::jsonb),'{message}','"direct runtime stopped"'::jsonb)
			WHERE metadata->>'mode'='inference_runtime_manual' AND metadata->'runtime'->>'model'=$1
			  AND status IN ('running','success')`, modelID)
		_, _ = a.Pool.Exec(r.Context(), `UPDATE service_instances SET status='unreachable', last_checked_at=now()
			WHERE model_id=$1 AND kind='vllm'`, modelID)
		_, _ = a.Pool.Exec(r.Context(), `UPDATE routing_policies SET enabled=false, updated_at=now() WHERE name=$1`, inferenceProductionPolicyName(modelID, "direct"))
	}
	a.Store.Audit(r.Context(), operator, "operator", "inference.serving_model.stopped", "model", modelID,
		map[string]any{"target": target, "history_preserved": true})
	WriteJSON(w, http.StatusOK, map[string]any{"model_id": modelID, "target": target, "status": "stopped"})
}
