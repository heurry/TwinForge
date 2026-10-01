package httpx

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
)

// isNoRows 判定 pgx QueryRow 无行。
func isNoRows(err error) bool { return errors.Is(err, pgx.ErrNoRows) }

// 6A：benchmarks 组绞杀——Go 原生 serving 压测（复刻 legacy src/api + src/jobs/benchmark.py 契约）。
// 复用既有 PG benchmark_runs / benchmark_samples（000001，是 SQLite 的超集）；
// 压测执行从「Go 起 Python 子进程跑 11_benchmark_serving.py」改为「Go 进程内并发负载」
// （见 benchmark_runner.go），复用 6A proxy 的端点解析与 5B.4b 的 MinIO 报告。
// 历史 25 行 SQLite 旧 run 不迁（前端只轮询新建 run_id；批量回填留待 6B）。

// benchmarkRunRequest 对齐 customer_support.schemas.BenchmarkRunRequest。
type benchmarkRunRequest struct {
	EndpointID        string                 `json:"endpoint_id"`
	Dataset           string                 `json:"dataset"`
	Workload          string                 `json:"workload"`
	RoutingStrategy   string                 `json:"routing_strategy"`
	ConcurrencyLevels []int                  `json:"concurrency_levels"`
	ContextLengths    []int                  `json:"context_lengths"`
	ContextMix        []int                  `json:"context_mix,omitempty"`
	PriorityByContext bool                   `json:"priority_by_context,omitempty"`
	RequestsPerLevel  int                    `json:"requests_per_level"`
	MaxTokens         int                    `json:"max_tokens"`
	VLLM              vllmOptimizationParams `json:"vllm"`
}

type vllmOptimizationParams struct {
	TensorParallelSize       int      `json:"tensor_parallel_size,omitempty"`
	PipelineParallelSize     int      `json:"pipeline_parallel_size,omitempty"`
	PipelineLayerPartition   string   `json:"pipeline_layer_partition,omitempty"`
	EnableDBO                *bool    `json:"enable_dbo,omitempty"`
	DBODecodeTokenThreshold  int      `json:"dbo_decode_token_threshold,omitempty"`
	DBOPrefillTokenThreshold int      `json:"dbo_prefill_token_threshold,omitempty"`
	MaxNumSeqs               int      `json:"max_num_seqs,omitempty"`
	MaxNumBatchedTokens      int      `json:"max_num_batched_tokens,omitempty"`
	GPUMemoryUtilization     float64  `json:"gpu_memory_utilization,omitempty"`
	MaxModelLen              int      `json:"max_model_len,omitempty"`
	PrefixCaching            *bool    `json:"prefix_caching,omitempty"`
	ChunkedPrefill           *bool    `json:"chunked_prefill,omitempty"`
	EnableThinking           *bool    `json:"enable_thinking,omitempty"`
	Quantization             string   `json:"quantization,omitempty"`
	Scheduler                string   `json:"scheduler,omitempty"`
	AsyncScheduling          *bool    `json:"async_scheduling,omitempty"`
	MaxNumPartialPrefills    int      `json:"max_num_partial_prefills,omitempty"`
	MaxLongPartialPrefills   int      `json:"max_long_partial_prefills,omitempty"`
	LongPrefillThreshold     int      `json:"long_prefill_token_threshold,omitempty"`
	SchedulerReserveFullISL  *bool    `json:"scheduler_reserve_full_isl,omitempty"`
	StreamInterval           int      `json:"stream_interval,omitempty"`
	DisableCustomAllReduce   *bool    `json:"disable_custom_all_reduce,omitempty"`
	KVCacheDType             string   `json:"kv_cache_dtype,omitempty"`
	FrequencyPenalty         float64  `json:"frequency_penalty,omitempty"`
	Stop                     []string `json:"stop,omitempty"`
	Profiling                bool     `json:"profiling,omitempty"`
	SpeculativeDecoding      string   `json:"speculative_decoding,omitempty"`
}

func boolValue(value *bool) bool {
	return value != nil && *value
}

func (r *benchmarkRunRequest) applyDefaults() {
	if r.EndpointID == "" {
		r.EndpointID = "aibrix-gateway"
	}
	if r.Dataset == "" {
		r.Dataset = "DianJin/DianJin-CSC-Data"
	}
	if r.Workload == "" {
		r.Workload = "customer_support_shared_prefix"
	}
	if r.RoutingStrategy == "" {
		r.RoutingStrategy = "least-request"
	}
	if len(r.ConcurrencyLevels) == 0 {
		r.ConcurrencyLevels = []int{1, 2, 4, 8, 16}
	}
	if len(r.ContextLengths) == 0 {
		r.ContextLengths = []int{1024, 2048}
	}
	if r.RequestsPerLevel <= 0 {
		r.RequestsPerLevel = 16
	}
	if r.MaxTokens <= 0 {
		r.MaxTokens = 256
	}
	if r.VLLM.EnableThinking == nil {
		disabled := false
		r.VLLM.EnableThinking = &disabled
	}
}

// POST /api/benchmarks/serving：建 run（queued）→ 解析端点 → 后台跑负载 → 返回 {run_id,status}。
func (a *API) createServingBenchmark(w http.ResponseWriter, r *http.Request) {
	var req benchmarkRunRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		a.badRequest(w, r, "invalid JSON body")
		return
	}
	req.applyDefaults()
	if err := req.validate(); err != nil {
		a.badRequest(w, r, err.Error())
		return
	}
	if trainingID, active, err := a.activeTrainingJob(r.Context()); err != nil {
		a.fail(w, r, err)
		return
	} else if active {
		WriteError(w, r, http.StatusConflict, "gpu_lane_busy", "训练任务 "+trainingID+" 正在占用单机 GPU 实验通道，请先停止训练")
		return
	}

	ep, status, err := a.resolveEndpoint(r.Context(), req.EndpointID)
	if err != nil {
		WriteError(w, r, status, errCodeForStatus(status), err.Error())
		return
	}
	if err := a.validateLocalInferenceConfig(r.Context(), req); err != nil {
		WriteError(w, r, http.StatusConflict, "inference_config_mismatch", err.Error())
		return
	}

	runID := uuid.NewString()
	cfg, _ := json.Marshal(req)
	if _, err := a.Pool.Exec(r.Context(), `
		INSERT INTO benchmark_runs (run_id, status, endpoint_id, workload, routing_strategy, config)
		VALUES ($1, 'queued', $2, $3, $4, $5)`,
		runID, req.EndpointID, req.Workload, req.RoutingStrategy, cfg); err != nil {
		a.fail(w, r, err)
		return
	}
	a.appendBenchmarkEvent(r.Context(), runID, "queued",
		map[string]any{"endpoint_id": req.EndpointID, "workload": req.Workload})

	// 后台跑（脱离请求 context，随进程存活）。
	ctx, cancel := context.WithCancel(context.Background())
	benchmarkCancels.Store(runID, cancel)
	go a.runServingBenchmark(ctx, runID, ep, req)

	WriteJSON(w, http.StatusOK, map[string]any{"run_id": runID, "status": "queued"})
}

func (a *API) validateLocalInferenceConfig(ctx context.Context, req benchmarkRunRequest) error {
	if req.EndpointID != "qwen36-27b-fp8-vllm" {
		return nil
	}
	runtime, _, err := a.Agent.RequestObject(ctx, http.MethodGet, "/api/inference/runtime", nil)
	if err != nil {
		return fmt.Errorf("cannot verify local inference runtime: %w", err)
	}
	if runtime["status"] != "ready" {
		return fmt.Errorf("local inference runtime is not ready (status=%v)", runtime["status"])
	}
	actual, ok := runtime["config"].(map[string]any)
	if !ok {
		return errors.New("local inference runtime did not return its launch config")
	}
	mismatches := []string{}
	checkNumber := func(key string, expected float64, enabled bool) {
		if !enabled {
			return
		}
		value, found := asFloat(actual[key])
		if !found || value != expected {
			mismatches = append(mismatches, fmt.Sprintf("%s report=%v runtime=%v", key, expected, actual[key]))
		}
	}
	checkBool := func(key string, expected *bool) {
		if expected == nil {
			return
		}
		value, found := actual[key].(bool)
		if !found || value != *expected {
			mismatches = append(mismatches, fmt.Sprintf("%s report=%v runtime=%v", key, *expected, actual[key]))
		}
	}
	checkString := func(key, expected string) {
		if expected != "" && actual[key] != expected {
			mismatches = append(mismatches, fmt.Sprintf("%s report=%s runtime=%v", key, expected, actual[key]))
		}
	}

	checkNumber("max_num_seqs", float64(req.VLLM.MaxNumSeqs), req.VLLM.MaxNumSeqs > 0)
	checkNumber("tensor_parallel_size", float64(req.VLLM.TensorParallelSize), req.VLLM.TensorParallelSize > 0)
	checkNumber("pipeline_parallel_size", float64(req.VLLM.PipelineParallelSize), req.VLLM.PipelineParallelSize > 0)
	checkNumber("dbo_decode_token_threshold", float64(req.VLLM.DBODecodeTokenThreshold), req.VLLM.DBODecodeTokenThreshold > 0)
	checkNumber("dbo_prefill_token_threshold", float64(req.VLLM.DBOPrefillTokenThreshold), req.VLLM.DBOPrefillTokenThreshold > 0)
	checkNumber("max_num_batched_tokens", float64(req.VLLM.MaxNumBatchedTokens), req.VLLM.MaxNumBatchedTokens > 0)
	checkNumber("gpu_memory_utilization", req.VLLM.GPUMemoryUtilization, req.VLLM.GPUMemoryUtilization > 0)
	checkNumber("max_model_len", float64(req.VLLM.MaxModelLen), req.VLLM.MaxModelLen > 0)
	checkNumber("max_num_partial_prefills", float64(req.VLLM.MaxNumPartialPrefills), req.VLLM.MaxNumPartialPrefills > 0)
	checkNumber("max_long_partial_prefills", float64(req.VLLM.MaxLongPartialPrefills), req.VLLM.MaxLongPartialPrefills > 0)
	checkNumber("long_prefill_token_threshold", float64(req.VLLM.LongPrefillThreshold), req.VLLM.LongPrefillThreshold > 0)
	checkNumber("stream_interval", float64(req.VLLM.StreamInterval), req.VLLM.StreamInterval > 0)
	checkBool("prefix_caching", req.VLLM.PrefixCaching)
	checkBool("chunked_prefill", req.VLLM.ChunkedPrefill)
	checkBool("async_scheduling", req.VLLM.AsyncScheduling)
	checkBool("scheduler_reserve_full_isl", req.VLLM.SchedulerReserveFullISL)
	checkBool("disable_custom_all_reduce", req.VLLM.DisableCustomAllReduce)
	checkBool("enable_dbo", req.VLLM.EnableDBO)
	checkString("scheduling_policy", req.VLLM.Scheduler)
	checkString("kv_cache_dtype", req.VLLM.KVCacheDType)
	checkString("speculative_decoding", req.VLLM.SpeculativeDecoding)
	checkString("pipeline_layer_partition", req.VLLM.PipelineLayerPartition)
	if req.VLLM.Quantization != "" && req.VLLM.Quantization != "fp8" {
		mismatches = append(mismatches, "quantization report="+req.VLLM.Quantization+" runtime=fp8")
	}
	if req.VLLM.Profiling {
		enabled := true
		checkBool("profiling", &enabled)
	}
	if len(mismatches) > 0 {
		return errors.New("benchmark parameters do not match the running service: " + strings.Join(mismatches, "; "))
	}
	return nil
}

// DELETE /api/benchmarks/{run_id}：取消在途请求并把 run 置为 cancelled。
func (a *API) cancelServingBenchmark(w http.ResponseWriter, r *http.Request) {
	runID := chi.URLParam(r, "run_id")
	var status string
	if err := a.Pool.QueryRow(r.Context(), `SELECT status FROM benchmark_runs WHERE run_id=$1`, runID).Scan(&status); err != nil {
		if isNoRows(err) {
			WriteError(w, r, http.StatusNotFound, "not_found", "benchmark run not found")
			return
		}
		a.fail(w, r, err)
		return
	}
	if status == "completed" || status == "failed" || status == "cancelled" {
		WriteJSON(w, http.StatusOK, map[string]any{"run_id": runID, "status": status})
		return
	}
	if value, ok := benchmarkCancels.Load(runID); ok {
		value.(context.CancelFunc)()
	}
	_, err := a.Pool.Exec(r.Context(), `UPDATE benchmark_runs SET status='cancelled', updated_at=now() WHERE run_id=$1`, runID)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	a.appendBenchmarkEvent(r.Context(), runID, "process_exit", map[string]any{"status": "cancelled"})
	WriteJSON(w, http.StatusOK, map[string]any{"run_id": runID, "status": "cancelled"})
}

func (a *API) activeTrainingJob(ctx context.Context) (string, bool, error) {
	var id string
	err := a.Pool.QueryRow(ctx, `
		SELECT id FROM training_jobs
		 WHERE status IN ('pending','running')
		 ORDER BY created_at DESC LIMIT 1`).Scan(&id)
	if isNoRows(err) {
		return "", false, nil
	}
	return id, err == nil, err
}

func (r benchmarkRunRequest) validate() error {
	if len(r.ContextLengths) > 8 || len(r.ConcurrencyLevels) > 16 {
		return errors.New("benchmark matrix is too large")
	}
	if r.RequestsPerLevel > 1000 || r.MaxTokens > 4096 {
		return errors.New("requests_per_level or max_tokens exceeds the safety limit")
	}
	for _, contextLength := range r.ContextLengths {
		if contextLength <= 0 || contextLength > 262144 {
			return errors.New("context_lengths must be between 1 and 262144")
		}
		if r.VLLM.MaxModelLen > 0 && contextLength+r.MaxTokens > r.VLLM.MaxModelLen {
			return errors.New("context_length plus max_tokens exceeds vllm.max_model_len")
		}
	}
	for _, contextLength := range r.ContextMix {
		if contextLength <= 0 || contextLength > 262144 {
			return errors.New("context_mix values must be between 1 and 262144")
		}
		if r.VLLM.MaxModelLen > 0 && contextLength+r.MaxTokens > r.VLLM.MaxModelLen {
			return errors.New("context_mix value plus max_tokens exceeds vllm.max_model_len")
		}
	}
	if len(r.ContextMix) > 16 {
		return errors.New("context_mix supports at most 16 weighted entries")
	}
	for _, concurrency := range r.ConcurrencyLevels {
		if concurrency <= 0 || concurrency > 256 {
			return errors.New("concurrency_levels must be between 1 and 256")
		}
		if r.RequestsPerLevel < concurrency {
			return errors.New("requests_per_level must be greater than or equal to every concurrency level")
		}
	}
	if r.VLLM.GPUMemoryUtilization < 0 || r.VLLM.GPUMemoryUtilization > 1 {
		return errors.New("vllm.gpu_memory_utilization must be between 0 and 1")
	}
	if r.VLLM.FrequencyPenalty < -2 || r.VLLM.FrequencyPenalty > 2 {
		return errors.New("vllm.frequency_penalty must be between -2 and 2")
	}
	return nil
}

// GET /api/benchmarks/{run_id}（复刻：返回行 + config/summary 解析为对象）。
func (a *API) benchmarkRun(w http.ResponseWriter, r *http.Request) {
	runID := chi.URLParam(r, "run_id")
	var status string
	var endpointID, workload, routingStrategy, reportPath, errText *string
	var config, summary []byte
	var createdAt, updatedAt time.Time
	err := a.Pool.QueryRow(r.Context(), `
		SELECT status, endpoint_id, workload, routing_strategy, config, summary, report_path, error, created_at, updated_at
		  FROM benchmark_runs WHERE run_id = $1`, runID).
		Scan(&status, &endpointID, &workload, &routingStrategy, &config, &summary, &reportPath, &errText, &createdAt, &updatedAt)
	if err != nil {
		if isNoRows(err) {
			WriteError(w, r, http.StatusNotFound, "not_found", "benchmark run not found")
			return
		}
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{
		"run_id": runID, "status": status,
		"endpoint_id": endpointID, "workload": workload, "routing_strategy": routingStrategy,
		"config": jsonbObject(config), "summary": jsonbObject(summary),
		"report_path": reportPath, "error": errText,
		"created_at": createdAt.UTC().Format(time.RFC3339Nano),
		"updated_at": updatedAt.UTC().Format(time.RFC3339Nano),
	})
}

// GET /api/benchmarks/{run_id}/events（复刻：benchmark_samples 按序，payload 解析为对象）。
func (a *API) benchmarkEvents(w http.ResponseWriter, r *http.Request) {
	runID := chi.URLParam(r, "run_id")
	rows, err := a.Pool.Query(r.Context(), `
		SELECT event_type, payload, created_at FROM benchmark_samples
		 WHERE run_id = $1 ORDER BY id ASC`, runID)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	defer rows.Close()
	events := []map[string]any{}
	for rows.Next() {
		var eventType *string
		var payload []byte
		var createdAt time.Time
		if err := rows.Scan(&eventType, &payload, &createdAt); err != nil {
			a.fail(w, r, err)
			return
		}
		events = append(events, map[string]any{
			"event_type": eventType, "payload": jsonbObject(payload),
			"created_at": createdAt.UTC().Format(time.RFC3339Nano),
		})
	}
	if err := rows.Err(); err != nil {
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{"run_id": runID, "events": events})
}

// appendBenchmarkEvent best-effort 落一条 benchmark_samples 事件。
func (a *API) appendBenchmarkEvent(ctx context.Context, runID, eventType string, payload map[string]any) {
	b, err := json.Marshal(payload)
	if err != nil {
		b = []byte("{}")
	}
	_, _ = a.Pool.Exec(ctx,
		`INSERT INTO benchmark_samples (run_id, event_type, payload) VALUES ($1, $2, $3)`,
		runID, eventType, b)
}
