package httpx

import (
	"context"
	"encoding/json"
	"io"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/heurry/cloudnative-infra-platform/server/internal/obs"
)

// E3 数据面：POST /api/routing/{policy}/v1/chat/completions。
// 按策略加权选主路候选 → 反代到上游（流式回客户端），可选把同一请求镜像到影子目标
// （丢弃响应、只采指标）。每次调用落 routing_samples 供 A/B 对比与全量/回滚决策。
// 与 proxyChatCompletions 共享 resolveEndpoint / dispatchUpstream / streamCopy。

// pickVariant 按权重做加权随机选择。返回命中候选与是否成功（总权重<=0 时失败）。
func pickVariant(variants []routingVariant) (routingVariant, bool) {
	total := 0
	for _, v := range variants {
		if v.Weight > 0 {
			total += v.Weight
		}
	}
	if total <= 0 {
		return routingVariant{}, false
	}
	n := rand.Intn(total)
	for _, v := range variants {
		if v.Weight <= 0 {
			continue
		}
		if n < v.Weight {
			return v, true
		}
		n -= v.Weight
	}
	return routingVariant{}, false // 不可达（权重和已 > 0）
}

// modelFor 候选/影子的 model 覆盖优先，否则沿用请求体内的 model。
func modelFor(override, payloadModel string) string {
	if override != "" {
		return override
	}
	return payloadModel
}

// rewriteModel 把请求体内的 "model" 字段改写为 model（候选版本覆盖时，body 与 header 一致）。
// model 为空或改写失败时原样返回。
func rewriteModel(body []byte, model string) []byte {
	if model == "" {
		return body
	}
	var payload map[string]any
	if json.Unmarshal(body, &payload) != nil {
		return body
	}
	payload["model"] = model
	out, err := json.Marshal(payload)
	if err != nil {
		return body
	}
	return out
}

// upstreamResult 是一次上游调用的指标（主路/影子共用）。
type upstreamResult struct {
	status  int // 0 = 连接失败
	latency time.Duration
	bytes   int
}

func (a *API) routedChatCompletions(w http.ResponseWriter, r *http.Request) {
	totalStart := time.Now()
	policyName := chi.URLParam(r, "policy")
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
	payloadModel, _ := payload["model"].(string)

	pol, err := a.loadRoutingPolicy(r.Context(), policyName)
	if err != nil {
		if isNoRows(err) {
			WriteError(w, r, http.StatusNotFound, "not_found", "routing policy not found: "+policyName)
			return
		}
		a.fail(w, r, err)
		return
	}
	if !pol.Enabled {
		WriteError(w, r, http.StatusServiceUnavailable, "policy_disabled", "routing policy disabled: "+policyName)
		return
	}
	variant, ok := pickVariant(pol.Variants)
	if !ok {
		WriteError(w, r, http.StatusServiceUnavailable, "no_active_variant", "routing policy has no active variant")
		return
	}
	// 模型生产入口必须绑定通过门禁的正式发布；手工启动的 vLLM 只允许内部调试，不能被路由代理暴露。
	if releaseModel, protected := inferenceReleaseVariant(variant); protected {
		var validated bool
		if err := a.Pool.QueryRow(r.Context(), `SELECT EXISTS(
			SELECT 1 FROM deployments WHERE metadata->>'mode'='inference_runtime'
			AND metadata->>'model_id'=$1 AND metadata->>'endpoint_id'=$2
			AND status IN ('running','success') AND COALESCE((metadata->'gate'->>'passed')::boolean, false)=true
		)`, releaseModel, variant.Endpoint).Scan(&validated); err != nil || !validated {
			WriteError(w, r, http.StatusServiceUnavailable, "inference_not_released", "模型运行时尚未通过发布门禁，当前只能在推理服务控制面内部调试")
			return
		}
	}
	ep, status, err := a.resolveEndpoint(r.Context(), variant.Endpoint)
	if err != nil {
		obs.RecordServiceEdge(policyName, "routing-policy", variant.Endpoint, "serving", true)
		// 落一条失败样本（status 0）后回写错误，便于灰度健康度观测。
		a.recordRoutingSample(policyName, requestIDFor(r), variant, variant.Endpoint, upstreamResult{status: 0}, nil, routingTarget{})
		go a.recordRequestTrace(requestTraceInput{
			RequestID: requestIDFor(r), EndpointID: policyName, TargetPod: variant.Endpoint,
			ModelID: modelFor(variant.Model, payloadModel), TotalMs: msSince(totalStart), Status: "error", Error: err.Error(),
			Metadata: map[string]any{"plane": "routing", "variant": variant.Label},
		})
		WriteError(w, r, status, errCodeForStatus(status), err.Error())
		return
	}

	reqID := requestIDFor(r)
	authz := r.Header.Get("Authorization")
	primaryModel := modelFor(variant.Model, payloadModel)

	// 影子镜像：与主路并发发起（同一请求体，独立 background context），结果经 channel 回收。
	var shadowCh chan upstreamResult
	var shadowTarget routingTarget
	if a.RoutingShadowEnabled && pol.Shadow != nil {
		shadowTarget = *pol.Shadow
		shadowCh = make(chan upstreamResult, 1)
		go func(t routingTarget) {
			shadowCh <- a.shadowCall(t, body, reqID, authz, payloadModel)
		}(shadowTarget)
	}

	start := time.Now()
	resp, err := dispatchUpstream(r.Context(), ep, rewriteModel(body, variant.Model), reqID, authz, primaryModel)
	if err != nil {
		obs.RecordServiceEdge(policyName, "routing-policy", ep.TargetPod, "serving", true)
		a.collectAndRecord(policyName, reqID, variant, ep.TargetPod, upstreamResult{status: 0, latency: time.Since(start)}, shadowCh, shadowTarget)
		go a.recordRequestTrace(requestTraceInput{
			RequestID: reqID, EndpointID: policyName, TargetPod: ep.TargetPod, ModelID: primaryModel,
			TotalMs: msSince(totalStart), QueueGatewayMs: float64(time.Since(start).Microseconds()) / 1000,
			Status: "error", Error: err.Error(), Metadata: map[string]any{"plane": "routing", "variant": variant.Label},
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
	w.Header().Set("x-routing-policy", policyName)
	w.Header().Set("x-routing-variant", variant.Label)
	w.Header().Set("x-target-pod", ep.TargetPod)
	w.WriteHeader(resp.StatusCode)
	written := streamCopy(w, resp)
	obs.RecordServiceEdge(policyName, "routing-policy", ep.TargetPod, "serving", resp.StatusCode < 200 || resp.StatusCode >= 400)

	a.collectAndRecord(policyName, reqID, variant, ep.TargetPod,
		upstreamResult{status: resp.StatusCode, latency: time.Since(start), bytes: written}, shadowCh, shadowTarget)
	traceStatus := "ok"
	traceError := ""
	if resp.StatusCode < 200 || resp.StatusCode >= 400 {
		traceStatus = "error"
		traceError = http.StatusText(resp.StatusCode)
	}
	go a.recordRequestTrace(requestTraceInput{
		RequestID: reqID, EndpointID: policyName, TargetPod: ep.TargetPod, ModelID: primaryModel,
		QueueGatewayMs: float64(time.Since(start).Microseconds()) / 1000, TotalMs: msSince(totalStart),
		Status: traceStatus, Error: traceError,
		Metadata: map[string]any{"plane": "routing", "variant": variant.Label, "http_status": resp.StatusCode, "response_bytes": written},
	})
}

// inferenceReleaseVariant 将已配置的模型生产 endpoint 标记为受发布门禁保护，
// 避免 qwen35 等新模型因为不在旧的 qwen36 常量判断中而绕过正式发布校验。
func inferenceReleaseVariant(variant routingVariant) (string, bool) {
	modelID := strings.TrimSpace(variant.Model)
	if modelID == "" {
		modelID = inferenceModelIDFromEndpoint(variant.Endpoint)
	}
	if _, ok := inferenceModelSpecFor(modelID); !ok {
		return "", false
	}
	spec, _ := inferenceModelSpecFor(modelID)
	// AIBrix 稳定基线（aibrix-gateway）可作为灰度主版本，无需绑定本次
	// 发布；发布中心创建的 direct 或 *-aibrix 候选 endpoint 必须有门禁证据。
	return modelID, variant.Endpoint == spec.Endpoint || strings.HasSuffix(variant.Endpoint, "-aibrix")
}

// shadowCall 把请求镜像到影子目标：独立 background context + 超时，读尽并丢弃响应、只采指标。
func (a *API) shadowCall(t routingTarget, body []byte, reqID, authz, payloadModel string) upstreamResult {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	ep, _, err := a.resolveEndpoint(ctx, t.Endpoint)
	if err != nil {
		return upstreamResult{status: 0}
	}
	start := time.Now()
	resp, err := dispatchUpstream(ctx, ep, rewriteModel(body, t.Model), reqID, authz, modelFor(t.Model, payloadModel))
	if err != nil {
		return upstreamResult{status: 0, latency: time.Since(start)}
	}
	defer resp.Body.Close()
	n, _ := io.Copy(io.Discard, resp.Body)
	return upstreamResult{status: resp.StatusCode, latency: time.Since(start), bytes: int(n)}
}

// collectAndRecord 在后台收集影子结果（限时等待）后落一条合并样本，不阻塞请求返回。
func (a *API) collectAndRecord(policy, reqID string, variant routingVariant, targetPod string, primary upstreamResult, shadowCh chan upstreamResult, shadowTarget routingTarget) {
	go func() {
		var shadow *upstreamResult
		if shadowCh != nil {
			select {
			case res := <-shadowCh:
				shadow = &res
			case <-time.After(65 * time.Second):
			}
		}
		a.recordRoutingSample(policy, reqID, variant, targetPod, primary, shadow, shadowTarget)
	}()
}

// recordRoutingSample 落一条 routing_samples（主路 + 可选影子指标）。
func (a *API) recordRoutingSample(policy, reqID string, variant routingVariant, targetPod string, primary upstreamResult, shadow *upstreamResult, shadowTarget routingTarget) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var shadowLabel, shadowEndpoint string
	var shadowStatus, shadowLatency, shadowBytes *int
	if shadow != nil {
		shadowLabel = shadowTarget.Label
		shadowEndpoint = shadowTarget.Endpoint
		st := shadow.status
		ms := int(shadow.latency.Milliseconds())
		by := shadow.bytes
		shadowStatus, shadowLatency, shadowBytes = &st, &ms, &by
	}
	_, _ = a.Pool.Exec(ctx,
		`INSERT INTO routing_samples
		   (policy_name, request_id, variant_label, variant_endpoint, primary_status, primary_latency_ms, primary_bytes,
		    shadow_label, shadow_endpoint, shadow_status, shadow_latency_ms, shadow_bytes)
		 VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)`,
		policy, reqID, variant.Label, targetPod, primary.status, int(primary.latency.Milliseconds()), primary.bytes,
		shadowLabel, shadowEndpoint, shadowStatus, shadowLatency, shadowBytes)
}
