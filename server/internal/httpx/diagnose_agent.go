package httpx

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/heurry/cloudnative-infra-platform/server/internal/aiclient"
	"github.com/heurry/cloudnative-infra-platform/server/internal/store"
)

// E1：agentic 诊断。由 Go 编排多轮循环，ai-service（模型）当 reasoner：
// 模型按需调用「只读取证工具」（指标/部署/故障/k8s），多轮收集证据后给出结构化结论。
// Go 本地执行工具（复用既有只读数据访问，无 Python→Go 回调），并把「查了哪些证据」记成推理轨迹落库。

const maxAgentSteps = 5

const agentSystemPrompt = `你是云原生 AI 推理基础设施的运维诊断 agent。你可以多轮调用下列只读工具收集证据：
- recent_metrics：当前 serving 指标（qps / error_rate / p95 / p99 等）
- recent_deployments：最近的部署记录与状态（含训练/推理配置引用）
- recent_config_changes：当前与上一配置版本的内容和变更原因
- recent_logs：训练、推理、发布和健康检查日志
- knowledge_search：从历史故障、配置变更和基准知识库检索相似证据
- open_incidents：未解决的故障事件
- kubernetes_pods：集群 Pod 概况（是否有非 Running）

策略：先按需调用工具取证（每次可调一个），证据足够后停止调用工具，直接输出"最终诊断"为一个 JSON 对象：
{"root_cause":"一句话根因","confidence":0~1,"impact":"业务影响",
 "recommended_actions":[{"action":"","risk":"low|medium|high","impact":""}],
 "related_resources":[{"type":"deployment|incident|config|service","id":"","name":""}]}
结论须引用证据中的具体数值，用简体中文。`

// agentTool 是一个只读取证工具：schema 给模型，exec 在 Go 本地执行，summarize 出轨迹摘要。
type agentTool struct {
	schema    aiclient.AgentToolSchema
	exec      func(ctx context.Context) any
	summarize func(v any) string
}

// agentTraceStep 是一步取证（落库进 evidence，并在响应里返回，构成推理轨迹）。
type agentTraceStep struct {
	Step    int    `json:"step"`
	Tool    string `json:"tool"`
	Summary string `json:"summary"`
}

func noArgsSchema() map[string]any {
	return map[string]any{"type": "object", "properties": map[string]any{}}
}

func (a *API) agentTools(question string) []agentTool {
	tools := []agentTool{
		{
			schema: aiclient.AgentToolSchema{Name: "recent_metrics", Description: "当前 serving 指标快照（qps/error_rate/p95/p99/request_count）", Parameters: noArgsSchema()},
			exec: func(ctx context.Context) any {
				m, err := a.Metrics.Current(ctx)
				if err != nil {
					return map[string]any{"error": err.Error()}
				}
				keys := []string{"request_count", "qps", "error_rate", "p95_latency_ms", "p99_latency_ms", "mean_latency_ms"}
				out := map[string]any{}
				for _, k := range keys {
					if v, ok := m[k]; ok {
						out[k] = v
					}
				}
				return out
			},
			summarize: func(v any) string {
				m, _ := v.(map[string]any)
				errorRate, _ := asFloat(m["error_rate"])
				p95, _ := asFloat(m["p95_latency_ms"])
				qps, _ := asFloat(m["qps"])
				return fmt.Sprintf("error_rate=%.4f, p95=%.1fms, qps=%.4f", errorRate, p95, qps)
			},
		},
		{
			schema: aiclient.AgentToolSchema{Name: "recent_deployments", Description: "最近部署记录（名称/状态/版本）", Parameters: noArgsSchema()},
			exec: func(ctx context.Context) any {
				rows, err := a.Store.Deployments(ctx)
				if err != nil {
					return map[string]any{"error": err.Error()}
				}
				dtos := mapSlice(rows, toDeploymentDTO)
				if len(dtos) > 8 {
					dtos = dtos[:8]
				}
				return dtos
			},
			summarize: func(v any) string {
				if s, ok := v.([]DeploymentDTO); ok {
					return fmt.Sprintf("%d 条部署记录", len(s))
				}
				return "部署记录"
			},
		},
		{
			schema: aiclient.AgentToolSchema{Name: "open_incidents", Description: "未解决的故障事件（标题/级别/状态）", Parameters: noArgsSchema()},
			exec: func(ctx context.Context) any {
				rows, err := a.Store.Incidents(ctx, "")
				if err != nil {
					return map[string]any{"error": err.Error()}
				}
				all := mapSlice(rows, toIncidentDTO)
				active := make([]IncidentDTO, 0, len(all))
				for _, i := range all {
					if i.Status != "resolved" {
						active = append(active, i)
					}
				}
				return active
			},
			summarize: func(v any) string {
				if s, ok := v.([]IncidentDTO); ok {
					if len(s) == 0 {
						return "无未解决故障"
					}
					return fmt.Sprintf("%d 起未解决故障，最高 %s", len(s), s[0].Severity)
				}
				return "故障事件"
			},
		},
		{
			schema: aiclient.AgentToolSchema{Name: "kubernetes_pods", Description: "集群 Pod 概况（总数 + 非 Running 列表）", Parameters: noArgsSchema()},
			exec: func(ctx context.Context) any {
				snap := a.k8sSnapshot(ctx, true, false, false, false, false)
				notRunning := []string{}
				for _, p := range snap.Pods {
					// Succeeded 是 Job/训练 Pod 的健康终态，不能当成运行异常。
					if p.Phase != "Running" && p.Phase != "Succeeded" {
						notRunning = append(notRunning, p.Namespace+"/"+p.Name+"("+p.Phase+")")
					}
				}
				return map[string]any{"available": snap.Available, "pod_count": len(snap.Pods), "not_running": notRunning}
			},
			summarize: func(v any) string {
				m, _ := v.(map[string]any)
				nr, _ := m["not_running"].([]string)
				return fmt.Sprintf("Pod %v 个，非 Running %d 个", m["pod_count"], len(nr))
			},
		},
	}
	configChanges := agentTool{
		schema:    aiclient.AgentToolSchema{Name: "recent_config_changes", Description: "最近配置版本 Diff、原因和操作人", Parameters: noArgsSchema()},
		exec:      func(ctx context.Context) any { return a.recentConfigChanges(ctx, 12) },
		summarize: func(v any) string { return summarizeCollection("配置变更", v) },
	}
	recentLogs := agentTool{
		schema:    aiclient.AgentToolSchema{Name: "recent_logs", Description: "最近训练、推理、发布、健康检查平台日志", Parameters: noArgsSchema()},
		exec:      func(ctx context.Context) any { return a.recentDiagnosisLogs(ctx, 30) },
		summarize: func(v any) string { return summarizeCollection("日志", v) },
	}
	knowledgeSearch := agentTool{
		schema:    aiclient.AgentToolSchema{Name: "knowledge_search", Description: "检索与当前问题相似的历史故障、配置变更和性能证据", Parameters: noArgsSchema()},
		exec:      func(ctx context.Context) any { return a.diagnosisKnowledge(ctx, question, 5) },
		summarize: func(v any) string { return "已检索历史诊断知识" },
	}
	// Stub 模式按顺序取前五个工具，因此优先覆盖新增的配置、日志、RAG
	// 与真实 Kubernetes 证据；在线模型仍能看到并自由选择全部工具。
	return []agentTool{tools[0], configChanges, recentLogs, knowledgeSearch, tools[3], tools[1], tools[2]}
}

func summarizeCollection(label string, value any) string {
	if rows, ok := value.([]map[string]any); ok {
		return fmt.Sprintf("%s %d 条", label, len(rows))
	}
	return label
}

// runDiagnoseAgent 跑 agent 循环，返回最终结论 + 推理轨迹 + 模式（stub/live）。
func (a *API) runDiagnoseAgent(ctx context.Context, question string, opts *aiclient.DiagnoseOptions) (*aiclient.AgentFinal, []agentTraceStep, string, error) {
	tools := a.agentTools(question)
	byName := make(map[string]agentTool, len(tools))
	schemas := make([]aiclient.AgentToolSchema, 0, len(tools))
	for _, t := range tools {
		byName[t.schema.Name] = t
		schemas = append(schemas, t.schema)
	}
	maxTokens, temperature := 1024, 0.2
	if opts != nil {
		if opts.MaxTokens > 0 {
			maxTokens = opts.MaxTokens
		}
		if opts.Temperature > 0 {
			temperature = opts.Temperature
		}
	}
	messages := []map[string]any{
		{"role": "system", "content": agentSystemPrompt},
		{"role": "user", "content": question},
	}
	trace := []agentTraceStep{}
	lastMode := ""

	for step := 0; step < maxAgentSteps; step++ {
		res, err := a.AI.AgentStep(ctx, aiclient.AgentStepRequest{
			Messages: messages, Tools: schemas, MaxTokens: maxTokens, Temperature: temperature,
		})
		if err != nil {
			return nil, trace, lastMode, err
		}
		lastMode = res.Mode
		if res.Final != nil && strings.TrimSpace(res.Final.RootCause) != "" {
			return res.Final, trace, res.Mode, nil
		}
		if len(res.ToolCalls) == 0 {
			// 模型既无工具调用也无结构化结论 → 把原文当结论（降级）。
			return &aiclient.AgentFinal{RootCause: orDefault(strings.TrimSpace(res.Content), "未得出明确结论")}, trace, res.Mode, nil
		}
		for _, tc := range res.ToolCalls {
			tool, ok := byName[tc.Name]
			var result any
			var summary string
			if !ok {
				result = map[string]any{"error": "unknown tool"}
				summary = "未知工具 " + tc.Name
			} else {
				result = tool.exec(ctx)
				summary = tool.summarize(result)
			}
			resultJSON, _ := json.Marshal(result)
			trace = append(trace, agentTraceStep{Step: step, Tool: tc.Name, Summary: summary})
			messages = append(messages, map[string]any{"role": "tool", "name": tc.Name, "content": string(resultJSON)})
		}
	}

	// 步数耗尽仍未结论 → 强制要一次最终结论（不再给工具）。
	res, err := a.AI.AgentStep(ctx, aiclient.AgentStepRequest{
		Messages:    append(messages, map[string]any{"role": "user", "content": "已达取证步数上限，请基于已收集证据直接给出最终诊断 JSON。"}),
		MaxTokens:   maxTokens,
		Temperature: temperature,
	})
	if err != nil {
		return nil, trace, lastMode, err
	}
	if res.Final != nil && strings.TrimSpace(res.Final.RootCause) != "" {
		return res.Final, trace, res.Mode, nil
	}
	return &aiclient.AgentFinal{RootCause: orDefault(strings.TrimSpace(res.Content), "未能在限定步数内得出结论")}, trace, res.Mode, nil
}

// POST /api/ai/diagnose:agent —— agentic 诊断（多轮取证 + 推理轨迹）。
func (a *API) diagnoseAgent(w http.ResponseWriter, r *http.Request) {
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
	if a.AI == nil {
		WriteError(w, r, http.StatusBadGateway, "ai_unavailable", "AI service not configured")
		return
	}
	operator := a.actor(r, req.Operator)
	ctx := r.Context()

	var opts *aiclient.DiagnoseOptions
	if req.MaxTokens > 0 || req.Temperature > 0 {
		opts = &aiclient.DiagnoseOptions{MaxTokens: req.MaxTokens, Temperature: req.Temperature}
	}

	final, trace, mode, err := a.runDiagnoseAgent(ctx, question, opts)
	if err != nil {
		id, _ := a.Store.InsertDiagnosis(ctx, store.DiagnosisInput{Question: question, Status: "failed", Error: err.Error()})
		a.Store.Audit(ctx, operator, "operator", "ai.diagnose.agent", "diagnosis", id,
			map[string]any{"status": "failed", "error": err.Error(), "steps": len(trace)})
		WriteError(w, r, http.StatusBadGateway, "ai_unavailable", "agent diagnose failed: "+classifyAIErr(err))
		return
	}

	// 推理轨迹落进 evidence（每步取证 = 一条 EvidenceItem，source=agent）。
	evidenceItems := make([]map[string]any, 0, len(trace))
	for _, s := range trace {
		evidenceItems = append(evidenceItems, map[string]any{
			"label":  s.Tool,
			"detail": s.Summary,
			"source": "agent",
		})
	}
	evidenceJSON, _ := json.Marshal(evidenceItems)

	id, err := a.Store.InsertDiagnosis(ctx, store.DiagnosisInput{
		Question:           question,
		Status:             "completed",
		RootCause:          final.RootCause,
		Confidence:         final.Confidence,
		Impact:             final.Impact,
		Evidence:           evidenceJSON,
		RecommendedActions: []byte(final.RecommendedActions),
		RelatedResources:   []byte(final.RelatedResources),
		ModelID:            "agent-" + mode,
		EndpointID:         "agent",
	})
	if err != nil {
		a.fail(w, r, err)
		return
	}
	a.Store.Audit(ctx, operator, "operator", "ai.diagnose.agent", "diagnosis", id,
		map[string]any{"status": "completed", "mode": mode, "steps": len(trace)})

	row, err := a.Store.GetDiagnosis(ctx, id)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{
		"diagnosis": toDiagnosisDTO(row),
		"mode":      mode,
		"steps":     len(trace),
		"trace":     trace,
	})
}
