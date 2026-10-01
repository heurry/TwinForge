// Package aiclient 是 Go 控制面调用 Python AI 服务（FastAPI）的客户端。
// 边界（Phase 3 D4）：Go 取证打包 evidence → 本客户端 POST /internal/diagnose →
// Python 做 RAG+LLM 推理出结构化 JSON → Go 落库+审计。错误分三类，便于 handler 分流状态码。
package aiclient

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

// 错误分类：handler 据此区分「上游不可达/超时」(502) 与「上游返回非 2xx 或坏响应」。
var (
	ErrUnreachable  = errors.New("ai service unreachable")
	ErrBadStatus    = errors.New("ai service returned error status")
	ErrBadResponse  = errors.New("ai service returned malformed response")
	maxResponseSize = int64(1 << 20) // 1 MiB，防止异常大响应撑爆内存
)

type Client struct {
	base string
	http *http.Client
}

// New 构建客户端；timeout<=0 时取 60s（诊断含 LLM 推理，比普通调用宽）。
func New(base string, timeout time.Duration) *Client {
	if timeout <= 0 {
		timeout = 60 * time.Second
	}
	return &Client{
		base: strings.TrimRight(base, "/"),
		// D1：otelhttp 传输注入 W3C traceparent，串起 Go→Python 链路（未配 OTLP 时为 no-op）。
		http: &http.Client{Timeout: timeout, Transport: otelhttp.NewTransport(http.DefaultTransport)},
	}
}

// DiagnoseOptions 透传给 Python 的推理参数（可空，Python 侧有默认值）。
type DiagnoseOptions struct {
	MaxTokens   int     `json:"max_tokens,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
}

// DiagnoseRequest 是发往 /internal/diagnose 的契约：问题 + Go 聚合的证据 bundle。
// Evidence 用 any（handler 传 map），整体一次性 marshal。
type DiagnoseRequest struct {
	Question string           `json:"question"`
	Evidence any              `json:"evidence"`
	Options  *DiagnoseOptions `json:"options,omitempty"`
}

// DiagnoseResult 是 Python 返回的结构化诊断。
// 三个 JSON 数组（Evidence/RecommendedActions/RelatedResources）保留为 RawMessage：
// Go 不重建嵌套结构，直接作为 JSONB 落库，由 Python 拥有「结构化外形」，边界更稳。
type DiagnoseResult struct {
	Status             string          `json:"status"` // completed / failed
	RootCause          string          `json:"root_cause"`
	Confidence         *float64        `json:"confidence"`
	Impact             string          `json:"impact"`
	Evidence           json.RawMessage `json:"evidence"`
	RecommendedActions json.RawMessage `json:"recommended_actions"`
	RelatedResources   json.RawMessage `json:"related_resources"`
	ModelID            string          `json:"model_id"`
	EndpointID         string          `json:"endpoint_id"`
	LatencyMs          *float64        `json:"latency_ms"`
	Mode               string          `json:"mode"` // stub / live —— 便于排查无 GPU 时为何是假响应
	Error              string          `json:"error"`
	Category           string          `json:"category"`
	Severity           string          `json:"severity"`
}

// Diagnose 调 Python 推理。任何网络/超时归 ErrUnreachable；非 2xx 归 ErrBadStatus；
// 解析失败归 ErrBadResponse。ctx 取消会传导到底层请求。
func (c *Client) Diagnose(ctx context.Context, req DiagnoseRequest) (*DiagnoseResult, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal diagnose request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+"/internal/diagnose", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnreachable, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnreachable, err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseSize))
	if err != nil {
		return nil, fmt.Errorf("%w: read body: %v", ErrBadResponse, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("%w: status %d: %s", ErrBadStatus, resp.StatusCode, truncate(data, 300))
	}

	var out DiagnoseResult
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrBadResponse, err)
	}
	return &out, nil
}

// Health 返回 AI 服务及其真实 LLM 上游状态，供控制台明确区分 live 与规则降级。
func (c *Client) Health(ctx context.Context) (map[string]any, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+"/internal/health", nil)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnreachable, err)
	}
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnreachable, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseSize))
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrBadResponse, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("%w: status %d", ErrBadStatus, resp.StatusCode)
	}
	out := map[string]any{}
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrBadResponse, err)
	}
	return out, nil
}

// EmbedRequest / EmbedResult 对齐 ai-service /internal/embed（5B.4c）。
type embedRequest struct {
	Texts   []string `json:"texts"`
	IsQuery bool     `json:"is_query"`
}

type embedResult struct {
	Embeddings [][]float32 `json:"embeddings"`
	Model      string      `json:"model"`
	Dim        int         `json:"dim"`
	Mode       string      `json:"mode"`
}

// Embed 调 Python /internal/embed 取文本嵌入（is_query=true 时查询侧加 Qwen3 指令前缀）。
// 错误分类同 Diagnose。向量响应可较大，单独放宽读取上限。
func (c *Client) Embed(ctx context.Context, texts []string, isQuery bool) ([][]float32, error) {
	body, err := json.Marshal(embedRequest{Texts: texts, IsQuery: isQuery})
	if err != nil {
		return nil, fmt.Errorf("marshal embed request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+"/internal/embed", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnreachable, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnreachable, err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(io.LimitReader(resp.Body, 32<<20)) // 32 MiB：批量向量可观
	if err != nil {
		return nil, fmt.Errorf("%w: read body: %v", ErrBadResponse, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("%w: status %d: %s", ErrBadStatus, resp.StatusCode, truncate(data, 300))
	}
	var out embedResult
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrBadResponse, err)
	}
	return out.Embeddings, nil
}

// ===== E1：agentic 诊断的单步推理（Go 编排循环，ai-service 当 reasoner）=====

// AgentToolSchema 是暴露给模型的工具描述（read-only 取证端点）。
type AgentToolSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// AgentStepRequest：当前对话 + 可用工具，问模型「下一步调哪个工具」或「给最终结论」。
type AgentStepRequest struct {
	Messages []map[string]any `json:"messages"`
	// omitempty：最终结论那一步不带工具（nil 切片）；否则序列化成 "tools":null，
	// 被 ai-service 的 Pydantic（List 非 Optional）判 422。省略后服务端用默认空列表。
	Tools       []AgentToolSchema `json:"tools,omitempty"`
	MaxTokens   int               `json:"max_tokens,omitempty"`
	Temperature float64           `json:"temperature,omitempty"`
}

// AgentToolCall：模型要求调用的工具（arguments 可空）。
type AgentToolCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// AgentFinal：模型给出的结构化最终诊断（与 DiagnoseResult 文本字段对齐）。
type AgentFinal struct {
	RootCause          string          `json:"root_cause"`
	Confidence         *float64        `json:"confidence"`
	Impact             string          `json:"impact"`
	RecommendedActions json.RawMessage `json:"recommended_actions"`
	RelatedResources   json.RawMessage `json:"related_resources"`
}

// AgentStepResult：tool_calls（继续取证）或 final（下结论），二选一；content 保留原文供 trace。
type AgentStepResult struct {
	Mode      string          `json:"mode"` // stub | live
	ToolCalls []AgentToolCall `json:"tool_calls"`
	Final     *AgentFinal     `json:"final"`
	Content   string          `json:"content"`
}

// AgentStep 调 ai-service /internal/agent-step。错误分类同 Diagnose。
func (c *Client) AgentStep(ctx context.Context, req AgentStepRequest) (*AgentStepResult, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal agent step: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+"/internal/agent-step", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnreachable, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnreachable, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseSize))
	if err != nil {
		return nil, fmt.Errorf("%w: read body: %v", ErrBadResponse, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("%w: status %d: %s", ErrBadStatus, resp.StatusCode, truncate(data, 300))
	}
	var out AgentStepResult
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrBadResponse, err)
	}
	return &out, nil
}

// BaseURL 暴露给反向代理（httpx/proxy.go）拼上游地址。
func (c *Client) BaseURL() string { return c.base }

func truncate(b []byte, n int) string {
	s := strings.TrimSpace(string(b))
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}
