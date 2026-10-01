package httpx

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"github.com/heurry/cloudnative-infra-platform/server/internal/aiclient"
	"github.com/heurry/cloudnative-infra-platform/server/internal/store"
	"github.com/jackc/pgx/v5"
)

// ===== AI 诊断 /api/ai =====
// 边界（Phase 3 D4）：Go 取证 → Python 推理 → Go 落库+审计+历史。
// 无 GPU 时 Python 走 stub 模式，本链路仍可端到端跑通（result.Mode=stub）。

type diagnoseReq struct {
	Question       string  `json:"question"`
	Scope          string  `json:"scope"`
	BenchmarkRunID string  `json:"benchmark_run_id"`
	TrainingJobID  string  `json:"training_job_id"`
	IncidentID     string  `json:"incident_id"`
	CreateIncident bool    `json:"create_incident"`
	MaxTokens      int     `json:"max_tokens"`
	Temperature    float64 `json:"temperature"`
	Operator       string  `json:"operator"`
}

func (a *API) aiStatus(w http.ResponseWriter, r *http.Request) {
	status, err := a.AI.Health(r.Context())
	if err != nil {
		WriteJSON(w, http.StatusOK, map[string]any{"status": "unavailable", "llm_connected": false, "effective_mode": "unavailable", "error": err.Error()})
		return
	}
	WriteJSON(w, http.StatusOK, status)
}

func (a *API) diagnose(w http.ResponseWriter, r *http.Request) {
	var req diagnoseReq
	if err := decodeBody(r, &req); err != nil {
		a.badRequest(w, r, "invalid body")
		return
	}
	question := strings.TrimSpace(req.Question)
	if question == "" {
		a.badRequest(w, r, "question required")
		return
	}
	operator := a.actor(r, req.Operator)
	ctx := r.Context()
	req.Scope = strings.TrimSpace(req.Scope)
	if req.Scope == "" {
		req.Scope = "general"
	}
	if req.Scope != "general" && req.Scope != "inference" && req.Scope != "training" {
		a.badRequest(w, r, "scope must be general, inference or training")
		return
	}
	if req.IncidentID != "" {
		if _, err := uuid.Parse(req.IncidentID); err != nil {
			a.badRequest(w, r, "invalid incident_id")
			return
		}
	}

	// 1) Go 取证：聚合 metrics / 事件 / 部署 / 配置 / k8s 成证据 bundle（best-effort）。
	evidence, evidenceErr := a.gatherEvidence(ctx, question, req.Scope, strings.TrimSpace(req.BenchmarkRunID), strings.TrimSpace(req.TrainingJobID))
	if evidenceErr != nil {
		if errors.Is(evidenceErr, pgx.ErrNoRows) {
			WriteError(w, r, http.StatusNotFound, "diagnosis_evidence_not_found", "没有找到该作用域可用于诊断的运行记录")
			return
		}
		a.fail(w, r, evidenceErr)
		return
	}

	// 2) 调 Python 推理。
	var opts *aiclient.DiagnoseOptions
	if req.MaxTokens > 0 || req.Temperature > 0 {
		opts = &aiclient.DiagnoseOptions{MaxTokens: req.MaxTokens, Temperature: req.Temperature}
	}
	result, aiErr := a.AI.Diagnose(ctx, aiclient.DiagnoseRequest{
		Question: question, Evidence: evidence, Options: opts,
	})

	// 3a) 上游失败：仍落一条 failed 诊断 + 审计（可观测），返回 502。
	if aiErr != nil {
		id, insErr := a.Store.InsertDiagnosis(ctx, store.DiagnosisInput{
			Question: question, Status: "failed", Error: aiErr.Error(),
		})
		if insErr != nil {
			slog.Error("persist failed-diagnosis errored", "err", insErr, "request_id", RequestIDFrom(ctx))
		}
		a.Store.Audit(ctx, operator, "operator", "ai.diagnose", "diagnosis", id,
			map[string]any{"status": "failed", "error": aiErr.Error()})
		slog.Error("diagnose upstream failed", "err", aiErr, "request_id", RequestIDFrom(ctx))
		WriteError(w, r, http.StatusBadGateway, "ai_unavailable", "AI service failed: "+classifyAIErr(aiErr))
		return
	}

	// 3b) 成功：落库 → 回读 → DTO（round-trip 保证「返回==落库」）。
	result.Evidence = diagnosisEvidenceWithClassification(result.Evidence, req.Scope, result.Category, result.Severity)
	id, err := a.Store.InsertDiagnosis(ctx, store.DiagnosisInput{
		Question:           question,
		Status:             result.Status,
		RootCause:          result.RootCause,
		Confidence:         result.Confidence,
		Impact:             result.Impact,
		Evidence:           []byte(result.Evidence),
		RecommendedActions: []byte(result.RecommendedActions),
		RelatedResources:   []byte(result.RelatedResources),
		ModelID:            result.ModelID,
		EndpointID:         result.EndpointID,
		LatencyMs:          result.LatencyMs,
		Error:              result.Error,
	})
	if err != nil {
		a.fail(w, r, err)
		return
	}
	incidentID := strings.TrimSpace(req.IncidentID)
	incidentCreated := false
	resourceKey, resourceID := diagnosisResource(evidence, req.Scope)
	if req.CreateIncident && incidentID == "" && result.Severity != "" && result.Severity != "info" && resourceID != "" {
		incidentID, incidentCreated, err = a.Store.EnsureWorkloadIncident(ctx, resourceKey, resourceID,
			diagnosisIncidentTitle(req.Scope, result.Category, result.ModelID), result.Severity, result.RootCause, id)
		if err != nil {
			a.fail(w, r, err)
			return
		}
	} else if incidentID != "" {
		if err := a.Store.LinkIncidentDiagnosis(ctx, incidentID, id, map[string]any{
			"scope": req.Scope, resourceKey: resourceID, "category": result.Category,
		}); err != nil {
			a.fail(w, r, err)
			return
		}
	}
	a.Store.Audit(ctx, operator, "operator", "ai.diagnose", "diagnosis", id,
		map[string]any{"status": result.Status, "mode": result.Mode, "model_id": result.ModelID,
			"scope": req.Scope, resourceKey: resourceID, "incident_id": incidentID})

	row, err := a.Store.GetDiagnosis(ctx, id)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{
		"diagnosis": rowDiagnosisWithClassification(toDiagnosisDTO(row), result.Category, result.Severity),
		"mode":      result.Mode, "incident_id": nilIfEmpty(incidentID), "incident_created": incidentCreated,
	})
}

// gatherEvidence 聚合证据 bundle；每块独立 best-effort，单块失败不拖垮整次诊断。
func (a *API) gatherEvidence(ctx context.Context, question, scope, benchmarkRunID, trainingJobID string) (map[string]any, error) {
	ev := map[string]any{}
	ev["scope"] = scope

	if m, err := a.Metrics.Current(ctx); err == nil {
		ev["metrics"] = m
	} else {
		ev["metrics"] = map[string]any{"error": err.Error()}
		slog.Warn("evidence metrics failed", "err", err)
	}

	if rows, err := a.Store.Incidents(ctx, ""); err == nil {
		ev["incidents"] = mapSlice(rows, toIncidentDTO)
	} else {
		ev["incidents"] = []any{}
		slog.Warn("evidence incidents failed", "err", err)
	}

	if rows, err := a.Store.Deployments(ctx); err == nil {
		ev["deployments"] = mapSlice(rows, toDeploymentDTO)
	} else {
		ev["deployments"] = []any{}
		slog.Warn("evidence deployments failed", "err", err)
	}

	if rows, err := a.Store.ConfigItems(ctx); err == nil {
		ev["config_items"] = mapSlice(rows, toConfigItemDTO)
	} else {
		ev["config_items"] = []any{}
		slog.Warn("evidence config_items failed", "err", err)
	}
	ev["config_changes"] = a.recentConfigChanges(ctx, 20)
	ev["request_traces"] = a.recentDiagnosisTraces(ctx, 50)
	ev["logs"] = a.recentDiagnosisLogs(ctx, 50)
	ev["rag_knowledge"] = a.diagnosisKnowledge(ctx, question, 5)

	// Kubernetes 读取已上提到 Go 控制面；诊断必须复用同一套 client-go
	// collector，不能再走已关闭 K8s 能力的 Node Agent，否则真实集群会被
	// 错误标成 unavailable。Snapshot 自带 disabled/error 降级语义。
	ev["kubernetes"] = a.k8sSnapshot(ctx, true, true, true, true, true)
	if scope == "inference" {
		inference, err := a.gatherInferenceEvidence(ctx, benchmarkRunID)
		if err != nil {
			return nil, err
		}
		ev["inference"] = inference
	} else if scope == "training" {
		training, err := a.gatherTrainingEvidence(ctx, trainingJobID)
		if err != nil {
			return nil, err
		}
		ev["training"] = training
	}

	return ev, nil
}

func inferenceRunID(evidence map[string]any) string {
	inference, _ := evidence["inference"].(map[string]any)
	benchmark, _ := inference["benchmark"].(inferenceBenchmarkEvidence)
	return benchmark.RunID
}

func diagnosisResource(evidence map[string]any, scope string) (string, string) {
	if scope == "training" {
		training, _ := evidence["training"].(map[string]any)
		job, _ := training["job"].(store.TrainingJob)
		return "training_job_id", job.ID
	}
	return "benchmark_run_id", inferenceRunID(evidence)
}

func diagnosisIncidentTitle(scope, category, modelID string) string {
	model := strings.TrimSpace(modelID)
	if slash := strings.LastIndex(model, "/"); slash >= 0 {
		model = model[slash+1:]
	}
	if model == "" || model == "stub" {
		if scope == "training" {
			model = "训练模型"
		} else {
			model = "推理模型"
		}
	}
	if scope == "training" {
		switch category {
		case "training_oom":
			return model + " LoRA 训练显存溢出"
		case "distributed_failure":
			return model + " 分布式训练通信异常"
		case "data_failure":
			return model + " 训练数据异常"
		case "artifact_failure":
			return model + " 训练产物归档异常"
		default:
			return model + " LoRA 训练任务异常"
		}
	}
	switch category {
	case "request_failure":
		return model + " 推理请求失败率异常"
	case "quality_regression":
		return model + " 推理输出质量异常"
	case "scheduler_saturation":
		return model + " 高并发调度饱和"
	case "memory_pressure":
		return model + " GPU 显存压力"
	case "decode_bottleneck":
		return model + " Decode 性能异常"
	default:
		return model + " 推理性能异常"
	}
}

func rowDiagnosisWithClassification(dto DiagnosisDTO, category, severity string) map[string]any {
	raw, _ := json.Marshal(dto)
	result := map[string]any{}
	_ = json.Unmarshal(raw, &result)
	result["category"] = category
	result["severity"] = severity
	return result
}

func diagnosisEvidenceWithClassification(raw json.RawMessage, scope, category, severity string) json.RawMessage {
	items := []map[string]any{}
	_ = json.Unmarshal(raw, &items)
	classification := map[string]any{
		"label":  "诊断分类",
		"detail": "scope=" + scope + ", category=" + category + ", severity=" + severity,
		"source": "classifier",
	}
	encoded, err := json.Marshal(append([]map[string]any{classification}, items...))
	if err != nil {
		return raw
	}
	return encoded
}

func (a *API) getDiagnosis(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if _, err := uuid.Parse(id); err != nil {
		a.badRequest(w, r, "invalid diagnosis id")
		return
	}
	row, err := a.Store.GetDiagnosis(r.Context(), id)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			WriteError(w, r, http.StatusNotFound, "not_found", "diagnosis not found")
			return
		}
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{"diagnosis": toDiagnosisDTO(row)})
}

func (a *API) listDiagnoses(w http.ResponseWriter, r *http.Request) {
	limit := clamp(queryInt(r, "limit", 20), 1, 200)
	rows, err := a.Store.ListDiagnoses(r.Context(), int32(limit))
	if err != nil {
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{"diagnoses": mapSlice(rows, toDiagnosisListItemDTO)})
}

// classifyAIErr 把 aiclient 的错误分类转成对外简短原因（不泄露内部细节）。
func classifyAIErr(err error) string {
	switch {
	case errors.Is(err, aiclient.ErrUnreachable):
		return "unreachable"
	case errors.Is(err, aiclient.ErrBadStatus):
		return "upstream error"
	case errors.Is(err, aiclient.ErrBadResponse):
		return "malformed response"
	default:
		return "unknown"
	}
}
