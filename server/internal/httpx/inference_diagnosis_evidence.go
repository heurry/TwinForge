package httpx

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"time"

	"github.com/jackc/pgx/v5"
)

type inferenceBenchmarkEvidence struct {
	RunID           string         `json:"run_id"`
	EndpointID      string         `json:"endpoint_id"`
	Workload        string         `json:"workload"`
	Config          map[string]any `json:"config"`
	Summary         map[string]any `json:"summary"`
	ReportPath      string         `json:"report_path"`
	CreatedAt       string         `json:"created_at"`
	UpdatedAt       string         `json:"updated_at"`
	PrefixCaching   bool           `json:"prefix_caching"`
	ChunkedPrefill  bool           `json:"chunked_prefill"`
	MaxNumSeqs      int            `json:"max_num_seqs"`
	MaxBatchedToken int            `json:"max_num_batched_tokens"`
}

// gatherInferenceEvidence 把最新或指定的正式压测、可比 baseline、运行时和 GPU 快照放进同一证据包。
func (a *API) gatherInferenceEvidence(ctx context.Context, runID string) (map[string]any, error) {
	current, err := a.loadBenchmarkEvidence(ctx, runID)
	if err != nil {
		return nil, err
	}

	evidence := map[string]any{"benchmark": current}
	if current.PrefixCaching {
		if baseline, baselineErr := a.loadComparableBaseline(ctx, current); baselineErr == nil {
			evidence["baseline"] = baseline
		} else if !errors.Is(baselineErr, pgx.ErrNoRows) {
			evidence["baseline_error"] = baselineErr.Error()
		}
	}

	runtime, _, runtimeErr := a.Agent.RequestObject(ctx, http.MethodGet, "/api/inference/runtime", nil)
	if runtimeErr != nil {
		runtime = map[string]any{"available": false, "status": "unavailable", "error": runtimeErr.Error()}
	}
	evidence["runtime"] = runtime
	evidence["runtime_logs"] = a.Agent.FetchObject(ctx, "/api/inference/runtime/logs")
	evidence["gpu"] = a.Agent.FetchObject(ctx, "/api/gpu")
	return evidence, nil
}

func (a *API) loadBenchmarkEvidence(ctx context.Context, runID string) (inferenceBenchmarkEvidence, error) {
	row := a.Pool.QueryRow(ctx, `
		SELECT run_id, COALESCE(endpoint_id,''), COALESCE(workload,''), config, summary,
		       COALESCE(report_path,''), created_at, updated_at
		FROM benchmark_runs
		WHERE status='completed' AND ($1='' OR run_id=$1)
		ORDER BY updated_at DESC
		LIMIT 1`, runID)
	return scanBenchmarkEvidence(row)
}

func (a *API) loadComparableBaseline(ctx context.Context, current inferenceBenchmarkEvidence) (inferenceBenchmarkEvidence, error) {
	row := a.Pool.QueryRow(ctx, `
		SELECT run_id, COALESCE(endpoint_id,''), COALESCE(workload,''), config, summary,
		       COALESCE(report_path,''), created_at, updated_at
		FROM benchmark_runs
		WHERE status='completed' AND run_id<>$1
		  AND COALESCE(endpoint_id,'')=$2 AND COALESCE(workload,'')=$3
		  AND COALESCE((config->'vllm'->>'prefix_caching')::boolean, false)=false
		ORDER BY updated_at DESC
		LIMIT 1`, current.RunID, current.EndpointID, current.Workload)
	return scanBenchmarkEvidence(row)
}

type benchmarkEvidenceRow interface {
	Scan(dest ...any) error
}

func scanBenchmarkEvidence(row benchmarkEvidenceRow) (inferenceBenchmarkEvidence, error) {
	var result inferenceBenchmarkEvidence
	var configJSON, summaryJSON []byte
	var createdAt, updatedAt time.Time
	if err := row.Scan(&result.RunID, &result.EndpointID, &result.Workload, &configJSON, &summaryJSON,
		&result.ReportPath, &createdAt, &updatedAt); err != nil {
		return result, err
	}
	result.Config = decodeJSONObject(configJSON)
	result.Summary = decodeJSONObject(summaryJSON)
	result.CreatedAt = createdAt.UTC().Format(time.RFC3339)
	result.UpdatedAt = updatedAt.UTC().Format(time.RFC3339)
	if vllm, ok := result.Config["vllm"].(map[string]any); ok {
		result.PrefixCaching, _ = vllm["prefix_caching"].(bool)
		result.ChunkedPrefill, _ = vllm["chunked_prefill"].(bool)
		result.MaxNumSeqs = intFromAny(vllm["max_num_seqs"])
		result.MaxBatchedToken = intFromAny(vllm["max_num_batched_tokens"])
	}
	return result, nil
}

func decodeJSONObject(raw []byte) map[string]any {
	result := map[string]any{}
	_ = json.Unmarshal(raw, &result)
	return result
}

func intFromAny(value any) int {
	switch v := value.(type) {
	case float64:
		return int(v)
	case int:
		return v
	default:
		return 0
	}
}

// GET /api/ai/inference/evidence：供 AIOps 页面预览当前会参与诊断的真实推理证据。
func (a *API) inferenceDiagnosisEvidence(w http.ResponseWriter, r *http.Request) {
	evidence, err := a.gatherInferenceEvidence(r.Context(), r.URL.Query().Get("run_id"))
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			WriteError(w, r, http.StatusNotFound, "benchmark_not_found", "没有可用于诊断的已完成推理压测")
			return
		}
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{"inference": evidence})
}
