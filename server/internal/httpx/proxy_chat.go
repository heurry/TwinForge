package httpx

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/heurry/cloudnative-infra-platform/server/internal/obs"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

// 6A：proxy 组绞杀——Go 原生 POST /api/proxy/{endpoint_id}/v1/chat/completions。
// 复刻 legacy src/api proxy_chat_completions：按 endpoint 解析（auto_router / client_round_robin /
// 直连），再反向代理到上游 vLLM/AIBrix 的 /chat/completions（流式与非流式统一边读边刷）。
//
// 对齐与偏差：
//   - 端点解析（轮询轮转、auto 候选优选 aibrix-gateway→round-robin→首个可用、状态过滤、headers、
//     x-target-pod / x-routing-endpoint）逐项复刻。
//   - **唯一偏差**：legacy auto_router 的 _choose_aibrix_strategy 依据「近期延迟/错误率」自适应选
//     routing-strategy，依赖 legacy SQLite 指标（Go 不持有）；此处改用配置的首个策略（缺省
//     least-request）。结构选路一致，仅放弃该自适应微调（可在后续用 Go serving 指标重做）。

var (
	rrMu        sync.Mutex
	rrPositions = map[string]int{} // 轮询位置（进程内，镜像 legacy _rr_positions）
)

// instanceRow 是 service_instances 的一行（解析用）。
type instanceRow struct {
	Name        string
	BaseURL     string
	ModelID     string
	Kind        string
	RoutingRole string
	Status      string
	Metadata    []byte
}

// resolvedEndpoint 是选路后用于反代的目标。
type resolvedEndpoint struct {
	EndpointID      string
	BaseURL         string
	ModelID         string
	TargetPod       string // x-target-pod
	RoutingStrategy string // 仅 aibrix 网关时设置 → routing-strategy 头
}

func (a *API) proxyChatCompletions(w http.ResponseWriter, r *http.Request) {
	started := time.Now()
	endpointID := chi.URLParam(r, "endpoint_id")
	body, err := io.ReadAll(io.LimitReader(r.Body, 16<<20))
	if err != nil {
		a.badRequest(w, r, "failed to read request body")
		return
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		a.badRequest(w, r, "invalid JSON body")
		return
	}

	ep, status, err := a.resolveEndpoint(r.Context(), endpointID)
	if err != nil {
		go a.recordRequestTrace(requestTraceInput{
			RequestID: requestIDFor(r), EndpointID: endpointID, ModelID: payloadString(payload, "model"),
			TotalMs: msSince(started), Status: "error", Error: err.Error(), Metadata: map[string]any{"plane": "proxy"},
		})
		WriteError(w, r, status, errCodeForStatus(status), err.Error())
		return
	}

	payloadModel, _ := payload["model"].(string)
	// 流式与非流式统一处理：无整体超时（SSE 长连），靠请求 context 取消。
	// The endpoint binding is authoritative for the served model. Do not pass
	// the caller's stale alias as an override; routing variants explicitly use
	// dispatchUpstream's override parameter when they need one.
	resp, err := dispatchUpstream(r.Context(), ep, body, requestIDFor(r), r.Header.Get("Authorization"), "")
	if err != nil {
		obs.RecordServiceEdge(endpointID, "endpoint", ep.TargetPod, "serving", true)
		go a.recordRequestTrace(requestTraceInput{
			RequestID: requestIDFor(r), EndpointID: endpointID, TargetPod: ep.TargetPod, ModelID: modelFor("", payloadModel),
			TotalMs: msSince(started), Status: "error", Error: err.Error(), Metadata: map[string]any{"plane": "proxy"},
		})
		WriteError(w, r, http.StatusBadGateway, "upstream_unreachable", "upstream endpoint unreachable: "+err.Error())
		return
	}
	defer resp.Body.Close()

	ct := resp.Header.Get("Content-Type")
	if ct == "" {
		ct = "application/json"
	}
	w.Header().Set("Content-Type", ct)
	w.Header().Set("x-target-pod", ep.TargetPod)
	w.Header().Set("x-routing-endpoint", endpointID)
	w.WriteHeader(resp.StatusCode)
	written := streamCopy(w, resp)
	obs.RecordServiceEdge(endpointID, "endpoint", ep.TargetPod, "serving", resp.StatusCode < 200 || resp.StatusCode >= 400)
	traceStatus := "ok"
	traceError := ""
	if resp.StatusCode < 200 || resp.StatusCode >= 400 {
		traceStatus = "error"
		traceError = http.StatusText(resp.StatusCode)
	}
	go a.recordRequestTrace(requestTraceInput{
		RequestID: requestIDFor(r), EndpointID: endpointID, TargetPod: ep.TargetPod,
		ModelID: modelFor(ep.ModelID, payloadModel), TotalMs: msSince(started), Status: traceStatus, Error: traceError,
		Metadata: map[string]any{"plane": "proxy", "http_status": resp.StatusCode, "response_bytes": written},
	})
}

func payloadString(payload map[string]any, key string) string {
	value, _ := payload[key].(string)
	return value
}

// requestIDFor 取入站 x-request-id，缺失时回退到 RequestID 中间件生成的值。
func requestIDFor(r *http.Request) string {
	if id := r.Header.Get(headerRequestID); id != "" {
		return id
	}
	return RequestIDFrom(r.Context())
}

// dispatchUpstream 构造并发起到 ep 的 /chat/completions 反代请求（流式/非流式统一）。
// modelOverride 非空时覆盖上游 model 头（候选/影子的版本灰度）；为空则用 ep.ModelID。
// 调用方负责关闭返回响应的 Body。
func dispatchUpstream(ctx context.Context, ep *resolvedEndpoint, body []byte, reqID, authz, modelOverride string) (*http.Response, error) {
	upstream := upstreamBaseURL(ep.BaseURL) + "/chat/completions"
	model := ep.ModelID
	if modelOverride != "" {
		model = modelOverride
	}
	// The endpoint is the source of truth for the served model. Older seeded
	// aliases (for example qwen3-4b-platform) can otherwise be forwarded to a
	// gateway currently serving qwen3-4b-customer and fail with model_not_found.
	forwardBody := body
	if model != "" {
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err == nil {
			payload["model"] = model
			if encoded, err := json.Marshal(payload); err == nil {
				forwardBody = encoded
			}
		}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, upstream, bytes.NewReader(forwardBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-request-id", reqID)
	if authz != "" {
		req.Header.Set("Authorization", authz)
	} else {
		req.Header.Set("Authorization", "Bearer EMPTY")
	}
	req.Header.Set("model", model)
	if ep.RoutingStrategy != "" {
		req.Header.Set("routing-strategy", ep.RoutingStrategy)
	}
	return proxyHTTPClient.Do(req)
}

// streamCopy 把上游响应边读边刷到 w（流式 token 即时下发），返回写出的字节数。
func streamCopy(w http.ResponseWriter, resp *http.Response) int {
	flusher, _ := w.(http.Flusher)
	buf := make([]byte, 8192)
	total := 0
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return total
			}
			total += n
			if flusher != nil {
				flusher.Flush() // 流式 token 即时下发
			}
		}
		if rerr != nil {
			return total
		}
	}
}

// proxyHTTPClient 无整体 Timeout（SSE 流式需要）；连接级超时交给 Transport 默认值与请求 context。
// D1：otelhttp 包裹传输，把 traceparent 注入到 Go→vLLM/AIBrix 的上游调用（未配 OTLP 时 no-op）。
var proxyHTTPClient = &http.Client{
	Transport: otelhttp.NewTransport(&http.Transport{ResponseHeaderTimeout: 60 * time.Second}),
}

// resolveEndpoint 解析 endpoint_id → 反代目标（含 auto_router / round_robin 选路）。
// 返回 (ep, httpStatus, err)；err!=nil 时 status 为应回写的 HTTP 码。
func (a *API) resolveEndpoint(ctx context.Context, name string) (*resolvedEndpoint, int, error) {
	row, err := a.fetchInstance(ctx, name)
	if err != nil {
		return nil, http.StatusInternalServerError, err
	}
	if row == nil {
		return nil, http.StatusNotFound, fmt.Errorf("no service instance configured for %q", name)
	}
	switch {
	case row.Kind == "auto_router" || row.RoutingRole == "auto_router":
		return a.selectAuto(ctx, row)
	case row.Kind == "client_round_robin" || row.RoutingRole == "client_round_robin":
		return a.selectRoundRobin(ctx, row)
	default:
		return endpointFromRow(row), http.StatusOK, nil
	}
}

// selectRoundRobin 在 metadata.target_instances 间轮转（进程内计数器）。
func (a *API) selectRoundRobin(ctx context.Context, router *instanceRow) (*resolvedEndpoint, int, error) {
	targets := metaStringSlice(router.Metadata, "target_instances")
	if len(targets) == 0 {
		return nil, http.StatusBadRequest, fmt.Errorf("round-robin endpoint %q has no target_instances", router.Name)
	}
	ordered := make([]*instanceRow, 0, len(targets))
	for _, name := range targets {
		if row, err := a.fetchInstance(ctx, name); err == nil && row != nil {
			ordered = append(ordered, row)
		}
	}
	if len(ordered) == 0 {
		return nil, http.StatusNotFound, fmt.Errorf("round-robin endpoint %q has no configured targets", router.Name)
	}
	rrMu.Lock()
	pos := rrPositions[router.Name]
	rrPositions[router.Name] = pos + 1
	rrMu.Unlock()
	sel := ordered[pos%len(ordered)]
	ep := endpointFromRow(sel)
	ep.RoutingStrategy = "" // 轮转目标是 vllm 副本，非 aibrix 网关
	return ep, http.StatusOK, nil
}

// selectAuto 复刻 _select_auto_endpoint 的结构选路：优选 aibrix-gateway → direct-round-robin →
// 首个可用候选 → 503（无 legacy 的 fallback_endpoint 配置）。
func (a *API) selectAuto(ctx context.Context, router *instanceRow) (*resolvedEndpoint, int, error) {
	candidates := metaStringSlice(router.Metadata, "candidate_endpoints")
	if len(candidates) == 0 {
		candidates = []string{"aibrix-gateway", "direct-round-robin"}
	}
	avail := map[string]*instanceRow{}
	for _, name := range candidates {
		if row, err := a.fetchInstance(ctx, name); err == nil && row != nil && endpointAvailable(row.Status) {
			avail[name] = row
		}
	}
	if aibrix := avail["aibrix-gateway"]; aibrix != nil {
		return endpointFromRow(aibrix), http.StatusOK, nil
	}
	if rr := avail["direct-round-robin"]; rr != nil {
		return a.selectRoundRobin(ctx, rr)
	}
	for _, name := range candidates {
		if row := avail[name]; row != nil {
			return endpointFromRow(row), http.StatusOK, nil
		}
	}
	return nil, http.StatusServiceUnavailable, fmt.Errorf("auto router %q has no available endpoints", router.Name)
}

// endpointFromRow 把一行转成反代目标；aibrix 网关附带 routing-strategy。
func endpointFromRow(row *instanceRow) *resolvedEndpoint {
	ep := &resolvedEndpoint{EndpointID: row.Name, BaseURL: row.BaseURL, ModelID: row.ModelID, TargetPod: row.Name}
	if row.Kind == "aibrix" || row.RoutingRole == "gateway" {
		ep.RoutingStrategy = routingStrategyFor(row.Metadata)
	}
	return ep
}

// routingStrategyFor 取 metadata.routing_strategies 首个，缺省 least-request（见文件头偏差说明）。
func routingStrategyFor(metadata []byte) string {
	if ss := metaStringSlice(metadata, "routing_strategies"); len(ss) > 0 {
		return ss[0]
	}
	return "least-request"
}

func (a *API) fetchInstance(ctx context.Context, name string) (*instanceRow, error) {
	var row instanceRow
	var modelID, routingRole *string
	err := a.Pool.QueryRow(ctx,
		`SELECT name, base_url, model_id, kind, routing_role, status, metadata
		   FROM service_instances WHERE name = $1`, name).
		Scan(&row.Name, &row.BaseURL, &modelID, &row.Kind, &routingRole, &row.Status, &row.Metadata)
	if err != nil {
		if strings.Contains(err.Error(), "no rows") {
			return nil, nil
		}
		return nil, err
	}
	row.ModelID = derefOr(modelID, "")
	row.RoutingRole = derefOr(routingRole, "")
	return &row, nil
}

// normalizeBaseURL 保证以 /v1 结尾（复刻 legacy normalize_base_url）。
func normalizeBaseURL(b string) string {
	t := strings.TrimRight(strings.TrimSpace(b), "/")
	if strings.HasSuffix(t, "/v1") {
		return t
	}
	return t + "/v1"
}

// upstreamBaseURL 将容器内无法访问的 loopback 地址映射到宿主机网关。
// service_instances 保留 127.0.0.1 便于浏览器/宿主机直连；go-server 在
// Docker 网络中发起请求时，通过 UPSTREAM_LOOPBACK_HOST=host.docker.internal
// 访问同一个宿主机 vLLM。未设置该变量时保持原行为，兼容本地直接运行二进制。
func upstreamBaseURL(b string) string {
	normalized := normalizeBaseURL(b)
	loopbackHost := strings.TrimSpace(os.Getenv("UPSTREAM_LOOPBACK_HOST"))
	if loopbackHost == "" {
		return normalized
	}
	u, err := url.Parse(normalized)
	if err != nil {
		return normalized
	}
	host := strings.ToLower(u.Hostname())
	if host != "127.0.0.1" && host != "localhost" && host != "::1" {
		return normalized
	}
	if port := u.Port(); port != "" {
		u.Host = net.JoinHostPort(loopbackHost, port)
	} else {
		u.Host = loopbackHost
	}
	return u.String()
}

func endpointAvailable(status string) bool {
	s := strings.ToLower(strings.TrimSpace(status))
	return s != "missing" && s != "unreachable"
}

func metaStringSlice(metadata []byte, key string) []string {
	if len(metadata) == 0 {
		return nil
	}
	var m map[string]any
	if err := json.Unmarshal(metadata, &m); err != nil {
		return nil
	}
	arr, ok := m[key].([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(arr))
	for _, v := range arr {
		if s, ok := v.(string); ok && s != "" {
			out = append(out, s)
		}
	}
	return out
}

func derefOr(p *string, def string) string {
	if p != nil {
		return *p
	}
	return def
}

func errCodeForStatus(status int) string {
	switch status {
	case http.StatusNotFound:
		return "not_found"
	case http.StatusBadRequest:
		return "bad_request"
	case http.StatusServiceUnavailable:
		return "service_unavailable"
	default:
		return "internal_error"
	}
}
