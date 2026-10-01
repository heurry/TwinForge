package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	inferenceModelName = "qwen36-27b-fp8"
	inferenceEndpoint  = "http://127.0.0.1:8020/v1"
)

type inferenceRuntime struct {
	docker           *http.Client
	image            string
	defaultModelPath string
	modelPath        string
	cachePath        string
	containerName    string
	modelID          string
	healthClient     *http.Client
}

type inferenceStatus struct {
	Available       bool                  `json:"available"`
	Status          string                `json:"status"`
	ContainerStatus string                `json:"container_status,omitempty"`
	ContainerName   string                `json:"container_name"`
	Model           string                `json:"model"`
	Endpoint        string                `json:"endpoint"`
	Profile         string                `json:"profile"`
	PrefixCaching   bool                  `json:"prefix_caching"`
	Config          inferenceLaunchConfig `json:"config"`
	Message         string                `json:"message,omitempty"`
}

type inferenceStartRequest struct {
	ModelID                  string  `json:"model_id"`
	Profile                  string  `json:"profile"`
	TensorParallelSize       int     `json:"tensor_parallel_size"`
	PipelineParallelSize     int     `json:"pipeline_parallel_size"`
	PipelineLayerPartition   string  `json:"pipeline_layer_partition"`
	EnableDBO                *bool   `json:"enable_dbo"`
	DBODecodeTokenThreshold  int     `json:"dbo_decode_token_threshold"`
	DBOPrefillTokenThreshold int     `json:"dbo_prefill_token_threshold"`
	MaxNumSeqs               int     `json:"max_num_seqs"`
	MaxNumBatchedTokens      int     `json:"max_num_batched_tokens"`
	SchedulingPolicy         string  `json:"scheduling_policy"`
	MaxNumPartialPrefills    int     `json:"max_num_partial_prefills"`
	MaxLongPartialPrefills   int     `json:"max_long_partial_prefills"`
	LongPrefillThreshold     int     `json:"long_prefill_token_threshold"`
	StreamInterval           int     `json:"stream_interval"`
	PrefixCaching            *bool   `json:"prefix_caching"`
	AsyncScheduling          *bool   `json:"async_scheduling"`
	SchedulerReserveFullISL  *bool   `json:"scheduler_reserve_full_isl"`
	DisableCustomAllReduce   *bool   `json:"disable_custom_all_reduce"`
	Profiling                bool    `json:"profiling"`
	GPUMemoryUtilization     float64 `json:"gpu_memory_utilization"`
	MaxModelLen              int     `json:"max_model_len"`
	KVCacheDType             string  `json:"kv_cache_dtype"`
	SpeculativeDecoding      string  `json:"speculative_decoding"`
}

type inferenceLaunchConfig struct {
	TensorParallelSize       int      `json:"tensor_parallel_size"`
	PipelineParallelSize     int      `json:"pipeline_parallel_size"`
	PipelineLayerPartition   string   `json:"pipeline_layer_partition"`
	EnableDBO                bool     `json:"enable_dbo"`
	DBODecodeTokenThreshold  int      `json:"dbo_decode_token_threshold"`
	DBOPrefillTokenThreshold int      `json:"dbo_prefill_token_threshold"`
	MaxNumSeqs               int      `json:"max_num_seqs"`
	MaxNumBatchedTokens      int      `json:"max_num_batched_tokens"`
	SchedulingPolicy         string   `json:"scheduling_policy"`
	MaxNumPartialPrefills    int      `json:"max_num_partial_prefills"`
	MaxLongPartialPrefills   int      `json:"max_long_partial_prefills"`
	LongPrefillThreshold     int      `json:"long_prefill_token_threshold"`
	StreamInterval           int      `json:"stream_interval"`
	PrefixCaching            bool     `json:"prefix_caching"`
	ChunkedPrefill           bool     `json:"chunked_prefill"`
	AsyncScheduling          *bool    `json:"async_scheduling"`
	SchedulerReserveFullISL  bool     `json:"scheduler_reserve_full_isl"`
	DisableCustomAllReduce   bool     `json:"disable_custom_all_reduce"`
	GPUMemoryUtilization     float64  `json:"gpu_memory_utilization"`
	MaxModelLen              int      `json:"max_model_len"`
	KVCacheDType             string   `json:"kv_cache_dtype"`
	Profiling                bool     `json:"profiling"`
	SpeculativeDecoding      string   `json:"speculative_decoding"`
	GPUDeviceIDs             []string `json:"gpu_device_ids,omitempty"`
}

func normalizeInferenceStart(req inferenceStartRequest) (inferenceLaunchConfig, error) {
	profile := strings.TrimSpace(req.Profile)
	if profile == "" {
		profile = "baseline"
	}
	if profile != "baseline" && profile != "prefix_cache" && profile != "scheduler" {
		return inferenceLaunchConfig{}, errors.New("profile must be baseline, prefix_cache or scheduler")
	}
	asyncScheduling := true
	config := inferenceLaunchConfig{
		TensorParallelSize:       2,
		PipelineParallelSize:     1,
		DBODecodeTokenThreshold:  32,
		DBOPrefillTokenThreshold: 512,
		MaxNumSeqs:               8,
		MaxNumBatchedTokens:      4096,
		SchedulingPolicy:         "fcfs",
		MaxNumPartialPrefills:    1,
		MaxLongPartialPrefills:   1,
		LongPrefillThreshold:     0,
		StreamInterval:           1,
		PrefixCaching:            profile != "baseline",
		ChunkedPrefill:           true,
		AsyncScheduling:          &asyncScheduling,
		SchedulerReserveFullISL:  true,
		DisableCustomAllReduce:   true,
		GPUMemoryUtilization:     0.9,
		MaxModelLen:              4096,
		KVCacheDType:             "auto",
		SpeculativeDecoding:      "none",
	}
	if profile != "scheduler" {
		if req.TensorParallelSize != 0 || req.PipelineParallelSize != 0 || req.PipelineLayerPartition != "" || req.EnableDBO != nil ||
			req.DBODecodeTokenThreshold != 0 || req.DBOPrefillTokenThreshold != 0 ||
			req.MaxNumSeqs != 0 || req.MaxNumBatchedTokens != 0 || req.SchedulingPolicy != "" ||
			req.MaxNumPartialPrefills != 0 || req.MaxLongPartialPrefills != 0 || req.LongPrefillThreshold != 0 ||
			req.StreamInterval != 0 || req.PrefixCaching != nil || req.AsyncScheduling != nil ||
			req.SchedulerReserveFullISL != nil || req.DisableCustomAllReduce != nil || req.Profiling ||
			req.GPUMemoryUtilization != 0 || req.MaxModelLen != 0 || req.KVCacheDType != "" || req.SpeculativeDecoding != "" {
			return inferenceLaunchConfig{}, errors.New("scheduler overrides require profile=scheduler")
		}
		return config, nil
	}
	if req.TensorParallelSize != 0 {
		config.TensorParallelSize = req.TensorParallelSize
	}
	if req.PipelineParallelSize != 0 {
		config.PipelineParallelSize = req.PipelineParallelSize
	}
	if req.PipelineLayerPartition != "" {
		config.PipelineLayerPartition = strings.TrimSpace(req.PipelineLayerPartition)
	}
	if req.EnableDBO != nil {
		config.EnableDBO = *req.EnableDBO
	}
	if req.DBODecodeTokenThreshold != 0 {
		config.DBODecodeTokenThreshold = req.DBODecodeTokenThreshold
	}
	if req.DBOPrefillTokenThreshold != 0 {
		config.DBOPrefillTokenThreshold = req.DBOPrefillTokenThreshold
	}
	if req.MaxNumSeqs != 0 {
		config.MaxNumSeqs = req.MaxNumSeqs
	}
	if req.MaxNumBatchedTokens != 0 {
		config.MaxNumBatchedTokens = req.MaxNumBatchedTokens
	}
	if req.SchedulingPolicy != "" {
		config.SchedulingPolicy = req.SchedulingPolicy
	}
	if req.MaxNumPartialPrefills != 0 {
		config.MaxNumPartialPrefills = req.MaxNumPartialPrefills
	}
	if req.MaxLongPartialPrefills != 0 {
		config.MaxLongPartialPrefills = req.MaxLongPartialPrefills
	}
	if req.LongPrefillThreshold != 0 {
		config.LongPrefillThreshold = req.LongPrefillThreshold
	}
	if req.StreamInterval != 0 {
		config.StreamInterval = req.StreamInterval
	}
	if req.PrefixCaching != nil {
		config.PrefixCaching = *req.PrefixCaching
	}
	if req.AsyncScheduling != nil {
		config.AsyncScheduling = req.AsyncScheduling
	}
	config.Profiling = req.Profiling
	if req.GPUMemoryUtilization != 0 {
		config.GPUMemoryUtilization = req.GPUMemoryUtilization
	}
	if req.MaxModelLen != 0 {
		config.MaxModelLen = req.MaxModelLen
	}
	if req.KVCacheDType != "" {
		config.KVCacheDType = req.KVCacheDType
	}
	if req.SpeculativeDecoding != "" {
		config.SpeculativeDecoding = req.SpeculativeDecoding
	}
	if req.SchedulerReserveFullISL != nil {
		config.SchedulerReserveFullISL = *req.SchedulerReserveFullISL
	}
	if req.DisableCustomAllReduce != nil {
		config.DisableCustomAllReduce = *req.DisableCustomAllReduce
	}

	if !slices.Contains([]int{4, 8, 12, 16, 24, 32}, config.MaxNumSeqs) {
		return inferenceLaunchConfig{}, errors.New("max_num_seqs must be one of 4, 8, 12, 16, 24 or 32")
	}
	if !slices.Contains([]int{2048, 4096, 8192}, config.MaxNumBatchedTokens) {
		return inferenceLaunchConfig{}, errors.New("max_num_batched_tokens must be one of 2048, 4096 or 8192")
	}
	if config.SchedulingPolicy != "fcfs" && config.SchedulingPolicy != "priority" {
		return inferenceLaunchConfig{}, errors.New("scheduling_policy must be fcfs or priority")
	}
	if config.MaxNumPartialPrefills != 1 {
		return inferenceLaunchConfig{}, errors.New("concurrent partial prefill is not supported by Qwen3.6 on vLLM 0.19.1; max_num_partial_prefills must be 1")
	}
	if config.MaxLongPartialPrefills < 1 || config.MaxLongPartialPrefills > config.MaxNumPartialPrefills {
		return inferenceLaunchConfig{}, errors.New("max_long_partial_prefills must be between 1 and max_num_partial_prefills")
	}
	if !slices.Contains([]int{0, 1024, 1536, 2048}, config.LongPrefillThreshold) {
		return inferenceLaunchConfig{}, errors.New("long_prefill_token_threshold must be one of 0, 1024, 1536 or 2048")
	}
	if !slices.Contains([]int{1, 4, 8}, config.StreamInterval) {
		return inferenceLaunchConfig{}, errors.New("stream_interval must be one of 1, 4 or 8")
	}
	if config.GPUMemoryUtilization < 0.35 || config.GPUMemoryUtilization > 0.95 {
		return inferenceLaunchConfig{}, errors.New("gpu_memory_utilization must be between 0.35 and 0.95")
	}
	if !slices.Contains([]int{3072, 4096, 8192, 16384, 24576}, config.MaxModelLen) {
		return inferenceLaunchConfig{}, errors.New("max_model_len must be 3072, 4096, 8192, 16384 or 24576")
	}
	if config.KVCacheDType != "auto" && config.KVCacheDType != "fp8" {
		return inferenceLaunchConfig{}, errors.New("kv_cache_dtype must be auto or fp8")
	}
	if config.SpeculativeDecoding != "none" && config.SpeculativeDecoding != "ngram" {
		return inferenceLaunchConfig{}, errors.New("speculative_decoding must be none or ngram")
	}
	if config.SpeculativeDecoding != "none" && config.AsyncScheduling != nil && *config.AsyncScheduling {
		return inferenceLaunchConfig{}, errors.New("speculative decoding requires async_scheduling=false")
	}
	if config.TensorParallelSize < 1 || config.PipelineParallelSize < 1 ||
		(config.TensorParallelSize*config.PipelineParallelSize != 1 && config.TensorParallelSize*config.PipelineParallelSize != 2) {
		return inferenceLaunchConfig{}, errors.New("runtime supports tensor_parallel_size * pipeline_parallel_size = 1 (4B) or 2 (27B)")
	}
	if config.PipelineLayerPartition != "" {
		if config.PipelineParallelSize != 2 {
			return inferenceLaunchConfig{}, errors.New("pipeline_layer_partition requires pipeline_parallel_size=2")
		}
		if !slices.Contains([]string{"32,32", "34,30", "35,29", "36,28"}, config.PipelineLayerPartition) {
			return inferenceLaunchConfig{}, errors.New("pipeline_layer_partition must be one of 32,32, 34,30, 35,29 or 36,28")
		}
	}
	if config.EnableDBO {
		return inferenceLaunchConfig{}, errors.New("DBO requires a DeepEP all-to-all backend and kernels, which are unavailable in the dual RTX 3090 runtime")
	}
	if !slices.Contains([]int{1, 4, 8, 16, 32}, config.DBODecodeTokenThreshold) {
		return inferenceLaunchConfig{}, errors.New("dbo_decode_token_threshold must be one of 1, 4, 8, 16 or 32")
	}
	if !slices.Contains([]int{256, 512, 1024}, config.DBOPrefillTokenThreshold) {
		return inferenceLaunchConfig{}, errors.New("dbo_prefill_token_threshold must be one of 256, 512 or 1024")
	}
	return config, nil
}

func sameInferenceConfig(left, right inferenceLaunchConfig) bool {
	// Device selection is derived from live capacity and must not make an
	// otherwise identical start request look like a profile change.
	left.GPUDeviceIDs = nil
	right.GPUDeviceIDs = nil
	leftJSON, _ := json.Marshal(left)
	rightJSON, _ := json.Marshal(right)
	return bytes.Equal(leftJSON, rightJSON)
}

func newInferenceRuntime(cfg agentConfig) (*inferenceRuntime, error) {
	dockerClient := &http.Client{Transport: &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", "/var/run/docker.sock")
		},
	}}
	return &inferenceRuntime{
		docker:           dockerClient,
		image:            cfg.inferenceImage,
		defaultModelPath: cfg.inferenceModelPath,
		modelPath:        cfg.inferenceModelPath,
		cachePath:        cfg.inferenceCachePath,
		containerName:    cfg.inferenceContainerName,
		modelID:          inferenceModelName,
		healthClient:     &http.Client{Timeout: 800 * time.Millisecond},
	}, nil
}

func (a *agent) registerInferenceHandlers(mux *http.ServeMux) {
	mux.HandleFunc("/api/inference/runtime", func(w http.ResponseWriter, r *http.Request) {
		if a.inference == nil {
			writeJSON(w, http.StatusServiceUnavailable, inferenceStatus{Available: false, Status: "disabled", Message: "inference runtime control is disabled"})
			return
		}
		switch r.Method {
		case http.MethodGet:
			writeJSON(w, http.StatusOK, a.inference.status(r.Context()))
		case http.MethodPost:
			var req inferenceStartRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				writeJSON(w, http.StatusBadRequest, map[string]any{"error": "invalid JSON body"})
				return
			}
			status, err := a.inference.start(r.Context(), req)
			if err != nil {
				writeJSON(w, http.StatusConflict, map[string]any{"error": err.Error(), "runtime": status})
				return
			}
			writeJSON(w, http.StatusAccepted, status)
		case http.MethodDelete:
			status, err := a.inference.stop(r.Context())
			if err != nil {
				writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error(), "runtime": status})
				return
			}
			writeJSON(w, http.StatusOK, status)
		default:
			w.Header().Set("Allow", "GET, POST, DELETE")
			writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"error": "method not allowed"})
		}
	})
	mux.HandleFunc("/api/inference/runtime/logs", func(w http.ResponseWriter, r *http.Request) {
		if a.inference == nil {
			writeJSON(w, http.StatusServiceUnavailable, map[string]any{"available": false, "error": "inference runtime control is disabled"})
			return
		}
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", "GET")
			writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"error": "method not allowed"})
			return
		}
		result, err := a.inference.logs(r.Context())
		if err != nil {
			writeJSON(w, http.StatusServiceUnavailable, map[string]any{"available": false, "error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, result)
	})
}

func (r *inferenceRuntime) logs(ctx context.Context) (map[string]any, error) {
	path := "/containers/" + url.PathEscape(r.containerName) + "/logs?stdout=true&stderr=true&timestamps=true&tail=2000"
	response, err := r.rawEngineRequest(ctx, http.MethodGet, path, nil)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode == http.StatusNotFound {
		return map[string]any{"available": false, "lines": []string{}, "signatures": map[string]int{}}, nil
	}
	if response.StatusCode != http.StatusOK {
		return nil, dockerResponseError(response)
	}
	raw, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	text := decodeDockerLogStream(raw)
	lines := strings.Split(strings.TrimSpace(text), "\n")
	if len(lines) == 1 && lines[0] == "" {
		lines = []string{}
	}
	signatures := map[string]int{}
	patterns := map[string]string{
		"marlin_fp8_fallback":         "does not have native support for FP8 computation",
		"gpu_p2p_unavailable":         "lacks GPU P2P capability",
		"fp8_kv_uncalibrated":         "Using uncalibrated q_scale",
		"partial_prefill_unsupported": "Concurrent Partial Prefill is not supported",
		"prefix_cache_experimental":   "support for Mamba layers is experimental",
		"possible_tensor_mismatch":    "potential format mismatch",
		"cuda_out_of_memory":          "CUDA out of memory",
		"insufficient_gpu_memory":     "Free memory on device",
	}
	for name, pattern := range patterns {
		for _, line := range lines {
			if strings.Contains(line, pattern) {
				signatures[name]++
			}
		}
	}
	return map[string]any{"available": true, "lines": lines, "signatures": signatures}, nil
}

func decodeDockerLogStream(raw []byte) string {
	var decoded strings.Builder
	for len(raw) >= 8 && (raw[0] == 1 || raw[0] == 2) {
		size := int(binary.BigEndian.Uint32(raw[4:8]))
		if size < 0 || size > len(raw)-8 {
			break
		}
		decoded.Write(raw[8 : 8+size])
		raw = raw[8+size:]
	}
	if decoded.Len() == 0 {
		return string(raw)
	}
	decoded.Write(raw)
	return decoded.String()
}

func (r *inferenceRuntime) status(ctx context.Context) inferenceStatus {
	config, _ := normalizeInferenceStart(inferenceStartRequest{Profile: "baseline"})
	result := inferenceStatus{
		Available:     true,
		Status:        "stopped",
		ContainerName: r.containerName,
		Model:         r.modelID,
		Endpoint:      inferenceEndpoint,
		Profile:       "baseline",
		Config:        config,
	}
	containerJSON, found, err := r.inspect(ctx)
	if err != nil {
		result.Status = "error"
		result.Message = err.Error()
		return result
	}
	if !found {
		return result
	}
	result.ContainerStatus = containerJSON.State.Status
	for i, arg := range containerJSON.Config.Cmd {
		if arg == "--served-model-name" && i+1 < len(containerJSON.Config.Cmd) {
			result.Model = containerJSON.Config.Cmd[i+1]
			r.modelID = result.Model
		}
	}
	if stored := containerJSON.Config.Labels["twinforge.inference.config"]; stored != "" {
		_ = json.Unmarshal([]byte(stored), &result.Config)
	} else {
		result.Config.PrefixCaching = slices.Contains(containerJSON.Config.Cmd, "--enable-prefix-caching")
	}
	result.PrefixCaching = result.Config.PrefixCaching
	if profile := containerJSON.Config.Labels["twinforge.profile"]; profile != "" {
		result.Profile = profile
	} else if result.PrefixCaching {
		result.Profile = "prefix_cache"
	}
	if !containerJSON.State.Running {
		return result
	}
	result.Status = "starting"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://127.0.0.1:8020/health", nil)
	if err == nil {
		if response, healthErr := r.healthClient.Do(req); healthErr == nil {
			response.Body.Close()
			if response.StatusCode == http.StatusOK {
				result.Status = "ready"
			}
		}
	}
	return result
}

func (r *inferenceRuntime) start(ctx context.Context, req inferenceStartRequest) (inferenceStatus, error) {
	modelID := strings.TrimSpace(req.ModelID)
	if modelID == "" {
		modelID = inferenceModelName
	}
	// 每次启动都根据 model_id 重新选择路径，不能复用上一次模型的可变路径。
	// 否则先启动 4B、再发布 27B 时会以 FP8 参数加载 Qwen3.5-4B，导致
	// Marlin 重排失败并表现为 runtime entered stopped。
	modelPath := r.defaultModelPath
	switch modelID {
	case inferenceModelName:
		// defaultModelPath 指向配置的 Qwen3.6-27B-FP8 权重。
	case "qwen38-27b-fp8":
		modelPath = "/mnt/nvme-data/models/LLM_model/Qwen3.8-27B-FP8"
	case "qwen36-27b-awq":
		modelPath = "/mnt/nvme-data/models/LLM_model/Qwen3.6-27B-AWQ-INT4"
	case "qwen35-4b-customer", "qwen3-4b-customer":
		modelPath = "/mnt/nvme-data/LLM/llm_train_platform_miniv2/model/Qwen3.5-4B"
	default:
		return r.status(ctx), errors.New("unsupported model_id; supported models: qwen38-27b-fp8, qwen36-27b-fp8, qwen36-27b-awq, qwen35-4b-customer, qwen3-4b-customer")
	}
	config, err := normalizeInferenceStart(req)
	if err != nil {
		return r.status(ctx), err
	}
	profile := strings.TrimSpace(req.Profile)
	if profile == "" {
		profile = "baseline"
	}
	if profile != "baseline" && profile != "prefix_cache" {
		partition := strings.ReplaceAll(config.PipelineLayerPartition, ",", "-")
		if partition == "" {
			partition = "auto"
		}
		profile = fmt.Sprintf("scheduler-tp%d-pp%d-l%s-dbo%t-s%d-t%d", config.TensorParallelSize, config.PipelineParallelSize, partition, config.EnableDBO, config.MaxNumSeqs, config.MaxNumBatchedTokens)
	}
	current := r.status(ctx)
	if current.Status == "ready" || current.Status == "starting" {
		if sameInferenceConfig(current.Config, config) && current.Model == modelID {
			return current, nil
		}
		return current, errors.New("inference service is already running with another profile; stop it before changing profile")
	}

	if err := r.remove(ctx); err != nil {
		return r.status(ctx), err
	}
	deviceIDs, err := chooseInferenceGPUDevices(config.TensorParallelSize*config.PipelineParallelSize, config.GPUMemoryUtilization)
	if err != nil {
		return r.status(ctx), err
	}
	config.GPUDeviceIDs = deviceIDs
	r.modelID = modelID
	r.modelPath = modelPath
	cmd := []string{
		"serve", r.modelPath,
		"--served-model-name", modelID,
		"--tensor-parallel-size", fmt.Sprint(config.TensorParallelSize),
		"--pipeline-parallel-size", fmt.Sprint(config.PipelineParallelSize),
		"--dtype", "auto",
		"--max-model-len", fmt.Sprint(config.MaxModelLen),
		"--gpu-memory-utilization", fmt.Sprintf("%.2f", config.GPUMemoryUtilization),
		"--max-num-seqs", fmt.Sprint(config.MaxNumSeqs),
		"--max-num-batched-tokens", fmt.Sprint(config.MaxNumBatchedTokens),
		"--scheduling-policy", config.SchedulingPolicy,
		"--max-num-partial-prefills", fmt.Sprint(config.MaxNumPartialPrefills),
		"--max-long-partial-prefills", fmt.Sprint(config.MaxLongPartialPrefills),
		"--long-prefill-token-threshold", fmt.Sprint(config.LongPrefillThreshold),
		"--stream-interval", fmt.Sprint(config.StreamInterval),
		"--kv-cache-dtype", config.KVCacheDType,
		"--host", "0.0.0.0",
		"--port", "8020",
		"--enable-chunked-prefill",
		"--language-model-only",
		"--reasoning-parser", "qwen3",
		"--enable-auto-tool-choice",
		"--tool-call-parser", "qwen3_coder",
		"--generation-config", "vllm",
		"--trust-remote-code",
	}
	if modelID == inferenceModelName || modelID == "qwen38-27b-fp8" {
		cmd = append(cmd[:len(cmd)-1], "--quantization", "fp8", "--trust-remote-code")
	} else if modelID == "qwen36-27b-awq" {
		cmd = append(cmd[:len(cmd)-1], "--quantization", "awq", "--trust-remote-code")
	}
	if config.PrefixCaching {
		cmd = append(cmd, "--enable-prefix-caching")
	} else {
		cmd = append(cmd, "--no-enable-prefix-caching")
	}
	if config.AsyncScheduling != nil {
		if *config.AsyncScheduling {
			cmd = append(cmd, "--async-scheduling")
		} else {
			cmd = append(cmd, "--no-async-scheduling")
		}
	}
	if config.SchedulerReserveFullISL {
		cmd = append(cmd, "--scheduler-reserve-full-isl")
	} else {
		cmd = append(cmd, "--no-scheduler-reserve-full-isl")
	}
	if config.DisableCustomAllReduce {
		cmd = append(cmd, "--disable-custom-all-reduce")
	}
	if config.EnableDBO {
		cmd = append(cmd,
			"--enable-dbo",
			"--dbo-decode-token-threshold", fmt.Sprint(config.DBODecodeTokenThreshold),
			"--dbo-prefill-token-threshold", fmt.Sprint(config.DBOPrefillTokenThreshold),
		)
	}
	if config.Profiling {
		cmd = append(cmd,
			"--enable-logging-iteration-details",
			"--enable-mfu-metrics",
			"--cudagraph-metrics",
			"--kv-cache-metrics",
			"--profiler-config",
			`{"profiler":"torch","torch_profiler_dir":"/root/.cache/vllm/profiler","torch_profiler_with_stack":false,"torch_profiler_record_shapes":true,"torch_profiler_with_memory":true}`,
		)
	}
	if config.SpeculativeDecoding == "ngram" {
		cmd = append(cmd, "--speculative-config", `{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_min":2,"prompt_lookup_max":5}`)
	}
	created, err := r.create(ctx, cmd, profile, config, deviceIDs)
	if err != nil {
		return r.status(ctx), err
	}
	if err := r.engineRequest(ctx, http.MethodPost, "/containers/"+created.ID+"/start", nil, http.StatusNoContent); err != nil {
		return r.status(ctx), err
	}
	return r.status(ctx), nil
}

func (r *inferenceRuntime) stop(ctx context.Context) (inferenceStatus, error) {
	err := r.engineRequest(ctx, http.MethodPost, "/containers/"+url.PathEscape(r.containerName)+"/stop?t=20", nil, http.StatusNoContent, http.StatusNotModified, http.StatusNotFound)
	if err != nil {
		return r.status(ctx), err
	}
	if err := r.remove(ctx); err != nil {
		return r.status(ctx), err
	}
	return r.status(ctx), nil
}

type dockerInspect struct {
	State struct {
		Status  string `json:"Status"`
		Running bool   `json:"Running"`
	} `json:"State"`
	Config struct {
		Cmd    []string          `json:"Cmd"`
		Labels map[string]string `json:"Labels"`
	} `json:"Config"`
}

type dockerCreateResponse struct {
	ID string `json:"Id"`
}

func (r *inferenceRuntime) inspect(ctx context.Context) (dockerInspect, bool, error) {
	var result dockerInspect
	response, err := r.rawEngineRequest(ctx, http.MethodGet, "/containers/"+url.PathEscape(r.containerName)+"/json", nil)
	if err != nil {
		return result, false, err
	}
	defer response.Body.Close()
	if response.StatusCode == http.StatusNotFound {
		return result, false, nil
	}
	if response.StatusCode != http.StatusOK {
		return result, false, dockerResponseError(response)
	}
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		return result, false, err
	}
	return result, true, nil
}

func (r *inferenceRuntime) create(ctx context.Context, cmd []string, profile string, config inferenceLaunchConfig, deviceIDs []string) (dockerCreateResponse, error) {
	configJSON, _ := json.Marshal(config)
	env := []string{"NVIDIA_VISIBLE_DEVICES=" + strings.Join(deviceIDs, ","), "NVIDIA_DRIVER_CAPABILITIES=compute,utility"}
	if config.PipelineLayerPartition != "" {
		env = append(env, "VLLM_PP_LAYER_PARTITION="+config.PipelineLayerPartition)
	}
	payload := map[string]any{
		"Image":      r.image,
		"Entrypoint": []string{"vllm"},
		"Cmd":        cmd,
		"Env":        env,
		"Labels": map[string]string{
			"twinforge.workload":         "inference",
			"twinforge.profile":          profile,
			"twinforge.inference.config": string(configJSON),
		},
		"HostConfig": map[string]any{
			"Binds":       []string{r.modelPath + ":" + r.modelPath + ":ro", r.cachePath + ":/root/.cache/vllm"},
			"NetworkMode": "host",
			"IpcMode":     "host",
			"DeviceRequests": []map[string]any{{
				"DeviceIDs": deviceIDs, "Capabilities": [][]string{{"gpu"}},
			}},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return dockerCreateResponse{}, err
	}
	response, err := r.rawEngineRequest(ctx, http.MethodPost, "/containers/create?name="+url.QueryEscape(r.containerName), body)
	if err != nil {
		return dockerCreateResponse{}, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusCreated {
		return dockerCreateResponse{}, dockerResponseError(response)
	}
	var result dockerCreateResponse
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		return result, err
	}
	return result, nil
}

// chooseInferenceGPUDevices chooses the least occupied physical GPUs and refuses
// a launch before Docker creation when vLLM's requested memory fraction cannot
// fit. This matters on the two-GPU demo node where an AIBrix stable replica and
// a direct canary runtime deliberately coexist.
func chooseInferenceGPUDevices(rankCount int, utilization float64) ([]string, error) {
	gpus, errText := collectGPU(context.Background())
	if len(gpus) == 0 {
		if errText == "" {
			errText = "no GPU metrics returned"
		}
		return nil, fmt.Errorf("GPU capacity preflight unavailable: %s", errText)
	}
	if rankCount < 1 || rankCount > len(gpus) {
		return nil, fmt.Errorf("runtime requires %d GPU(s), but the node exposes %d", rankCount, len(gpus))
	}
	type candidate struct {
		index int
		free  float64
		need  float64
	}
	candidates := make([]candidate, 0, len(gpus))
	for _, gpu := range gpus {
		candidates = append(candidates, candidate{
			index: gpu.Index,
			free:  gpu.MemoryTotalMb - gpu.MemoryUsedMb,
			need:  gpu.MemoryTotalMb * utilization,
		})
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].free > candidates[j].free })
	selected := make([]string, 0, rankCount)
	for _, gpu := range candidates {
		if gpu.free+1 < gpu.need {
			continue
		}
		selected = append(selected, strconv.Itoa(gpu.index))
		if len(selected) == rankCount {
			return selected, nil
		}
	}
	best := candidates[0]
	return nil, fmt.Errorf(
		"insufficient GPU memory: runtime needs %d GPU(s) with %.0f MiB each at gpu_memory_utilization=%.2f; best device has %.0f MiB free (AIBrix or another process is using the GPUs)",
		rankCount, best.need, utilization, best.free,
	)
}

func (r *inferenceRuntime) remove(ctx context.Context) error {
	path := "/containers/" + url.PathEscape(r.containerName) + "?force=true"
	for attempt := 0; attempt < 20; attempt++ {
		response, err := r.rawEngineRequest(ctx, http.MethodDelete, path, nil)
		if err != nil {
			return err
		}
		if response.StatusCode == http.StatusNoContent || response.StatusCode == http.StatusNotFound {
			response.Body.Close()
			return nil
		}
		if response.StatusCode == http.StatusConflict {
			body, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
			response.Body.Close()
			if strings.Contains(strings.ToLower(string(body)), "removal") {
				select {
				case <-ctx.Done():
					return ctx.Err()
				case <-time.After(250 * time.Millisecond):
					continue
				}
			}
			return fmt.Errorf("docker engine returned %s: %s", response.Status, strings.TrimSpace(string(body)))
		}
		err = dockerResponseError(response)
		response.Body.Close()
		return err
	}
	return errors.New("timed out waiting for inference container removal")
}

func (r *inferenceRuntime) engineRequest(ctx context.Context, method, path string, body []byte, allowed ...int) error {
	response, err := r.rawEngineRequest(ctx, method, path, body)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if slices.Contains(allowed, response.StatusCode) {
		return nil
	}
	return dockerResponseError(response)
}

func (r *inferenceRuntime) rawEngineRequest(ctx context.Context, method, path string, body []byte) (*http.Response, error) {
	request, err := http.NewRequestWithContext(ctx, method, "http://docker/v1.44"+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	return r.docker.Do(request)
}

func dockerResponseError(response *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
	return fmt.Errorf("docker engine returned %s: %s", response.Status, strings.TrimSpace(string(body)))
}
