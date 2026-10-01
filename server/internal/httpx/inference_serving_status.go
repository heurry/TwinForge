package httpx

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
)

type inferenceServingLayer struct {
	Status     string `json:"status"`
	Detail     string `json:"detail,omitempty"`
	Desired    int32  `json:"desired,omitempty"`
	Ready      int32  `json:"ready,omitempty"`
	Available  int32  `json:"available,omitempty"`
	HTTPStatus int    `json:"http_status,omitempty"`
}

type inferenceServingRelease struct {
	ID     string `json:"id,omitempty"`
	Status string `json:"status"`
	Phase  string `json:"phase,omitempty"`
}

type inferenceServingStatus struct {
	Overall         string                  `json:"overall"`
	ModelID         string                  `json:"model_id"`
	Target          string                  `json:"target"`
	Managed         bool                    `json:"managed"`
	CanStop         bool                    `json:"can_stop"`
	GatewayEndpoint string                  `json:"gateway_endpoint"`
	Namespace       string                  `json:"namespace"`
	Deployment      string                  `json:"deployment"`
	CheckedAt       string                  `json:"checked_at"`
	Release         inferenceServingRelease `json:"release"`
	Workload        inferenceServingLayer   `json:"workload"`
	Gateway         inferenceServingLayer   `json:"gateway"`
	ModelRoute      inferenceServingLayer   `json:"model_route"`
	ToolCalling     inferenceServingLayer   `json:"tool_calling"`
}

// RunInferenceServingReconciler continuously projects current serving truth back
// into service_instances for both AIBrix bindings and direct vLLM bindings.
// Routing and the service-management UI therefore cannot keep using an
// endpoint only because a historical health check once marked it healthy.
func (a *API) RunInferenceServingReconciler(ctx context.Context, interval time.Duration) {
	if interval <= 0 {
		interval = 30 * time.Second
	}
	reconcile := func() {
		rows, err := a.Pool.Query(ctx, `SELECT name, COALESCE(model_id,''), kind, base_url, routing_role, metadata
			FROM service_instances WHERE kind IN ('aibrix','vllm') ORDER BY name`)
		if err != nil {
			return
		}
		type binding struct {
			name, model, kind, baseURL, routingRole string
			metadata                                []byte
		}
		var bindings []binding
		for rows.Next() {
			var current binding
			if rows.Scan(&current.name, &current.model, &current.kind, &current.baseURL, &current.routingRole, &current.metadata) == nil && current.model != "" {
				bindings = append(bindings, current)
			}
		}
		rows.Close()
		observations := make(map[string]inferenceServingStatus)
		for _, current := range bindings {
			if current.kind == "vllm" {
				probeCtx, cancel := context.WithTimeout(ctx, healthProbeTimeout)
				started := time.Now()
				target := &instanceRow{
					Name: current.name, ModelID: current.model, Kind: current.kind,
					BaseURL: current.baseURL, RoutingRole: current.routingRole, Metadata: current.metadata,
				}
				probeURL, _, detail, probeErr := probeService(probeCtx, target.BaseURL, healthPaths(target))
				cancel()
				registryStatus := "healthy"
				if probeErr != nil {
					registryStatus = "unreachable"
					detail = probeErr.Error()
				}
				latencyMs := float64(time.Since(started).Microseconds()) / 1000
				_ = a.persistHealthResult(ctx, current.name, registryStatus, latencyMs, probeURL, detail)
				continue
			}
			observation, exists := observations[current.model]
			if !exists {
				observation = a.inspectAIBrixServingStatus(ctx, current.model)
				observations[current.model] = observation
			}
			registryStatus := "unreachable"
			if observation.Overall == "ready" {
				registryStatus = "healthy"
			}
			detail := fmt.Sprintf("overall=%s workload=%s gateway=%s model_route=%s",
				observation.Overall, observation.Workload.Status, observation.Gateway.Status, observation.ModelRoute.Status)
			_ = a.persistHealthResult(ctx, current.name, registryStatus, 0, observation.GatewayEndpoint, detail)
		}
	}
	reconcile()
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			reconcile()
		}
	}
}

// inspectAIBrixServingStatus keeps desired release state separate from current
// data-plane truth. A historical successful Deployment is never treated as a
// live endpoint without a current Kubernetes observation and model route probe.
func (a *API) inspectAIBrixServingStatus(ctx context.Context, modelID string) inferenceServingStatus {
	endpoint := a.aibrixGatewayEndpoint()
	status := inferenceServingStatus{
		Overall: "unavailable", ModelID: modelID, Target: "aibrix", GatewayEndpoint: endpoint,
		Namespace: "default", Deployment: modelID,
		CheckedAt:   time.Now().UTC().Format(time.RFC3339Nano),
		Release:     inferenceServingRelease{Status: "not_released"},
		Workload:    inferenceServingLayer{Status: "unknown"},
		Gateway:     inferenceServingLayer{Status: "unknown"},
		ModelRoute:  inferenceServingLayer{Status: "not_checked", Detail: "等待 workload 与 gateway 就绪"},
		ToolCalling: inferenceServingLayer{Status: "not_checked", Detail: "等待模型路由就绪"},
	}

	var releaseID, releaseStatus string
	var metadata []byte
	err := a.Pool.QueryRow(ctx, `SELECT id::text, status, metadata FROM deployments
		WHERE metadata->>'mode'='inference_runtime' AND metadata->>'release_target'='aibrix'
		  AND metadata->>'model_id'=$1
		ORDER BY started_at DESC LIMIT 1`, modelID).Scan(&releaseID, &releaseStatus, &metadata)
	if err == nil {
		meta := jsonbObject(metadata)
		status.Release = inferenceServingRelease{ID: releaseID, Status: releaseStatus, Phase: stringValue(meta["phase"])}
		status.Managed = true
		status.Namespace = orDefault(stringValue(meta["k8s_namespace"]), "default")
		status.Deployment = orDefault(stringValue(meta["k8s_deployment"]), modelID)
	} else if !errors.Is(err, pgx.ErrNoRows) {
		status.Release = inferenceServingRelease{Status: "unknown", Phase: err.Error()}
	}

	if a.K8s == nil {
		status.Workload = inferenceServingLayer{Status: "unavailable", Detail: orDefault(a.K8sErr, "Kubernetes collector unavailable")}
	} else {
		rollout, rolloutErr := a.K8s.RolloutStatus(ctx, status.Namespace, status.Deployment)
		switch {
		case rolloutErr != nil:
			status.Workload = inferenceServingLayer{Status: "missing", Detail: rolloutErr.Error()}
		case rollout.Failed:
			status.Workload = inferenceServingLayer{Status: "failed", Detail: rollout.Message, Desired: rollout.Desired, Ready: rollout.Ready, Available: rollout.Available}
		case rollout.Desired == 0:
			status.Workload = inferenceServingLayer{Status: "stopped", Detail: "Deployment 已缩容到 0", Desired: 0}
		case rollout.Complete:
			status.Workload = inferenceServingLayer{Status: "ready", Detail: rollout.Message, Desired: rollout.Desired, Ready: rollout.Ready, Available: rollout.Available}
		default:
			status.Workload = inferenceServingLayer{Status: "starting", Detail: rollout.Message, Desired: rollout.Desired, Ready: rollout.Ready, Available: rollout.Available}
		}
	}

	gatewayCode, gatewayDetail, gatewayErr := probeAIBrixGateway(ctx, endpoint)
	if gatewayErr != nil {
		status.Gateway = inferenceServingLayer{Status: "unreachable", Detail: gatewayErr.Error(), HTTPStatus: gatewayCode}
	} else {
		status.Gateway = inferenceServingLayer{Status: "ready", Detail: gatewayDetail, HTTPStatus: gatewayCode}
	}

	if status.Workload.Status == "ready" && status.Gateway.Status == "ready" {
		_, code, detail, routeErr := probeAIBrix(ctx, endpoint, modelID)
		if routeErr != nil {
			status.ModelRoute = inferenceServingLayer{Status: "unreachable", Detail: routeErr.Error() + detail, HTTPStatus: code}
		} else {
			status.ModelRoute = inferenceServingLayer{Status: "ready", Detail: "模型已通过 AIBrix 路由完成 1-token 探针", HTTPStatus: code}
			toolCode, toolDetail, toolErr := probeToolCalling(ctx, endpoint, modelID)
			if toolErr != nil {
				status.ToolCalling = inferenceServingLayer{Status: "unreachable", Detail: toolErr.Error() + toolDetail, HTTPStatus: toolCode}
			} else {
				status.ToolCalling = inferenceServingLayer{Status: "ready", Detail: "Tool Calling 请求已被模型端接受", HTTPStatus: toolCode}
			}
		}
	}
	status.Overall = deriveServingOverall(status.Workload.Status, status.Gateway.Status, status.ModelRoute.Status)
	if status.Overall == "ready" && status.ToolCalling.Status != "ready" {
		status.Overall = "degraded"
	}
	status.CanStop = status.Workload.Status == "ready" || status.Workload.Status == "starting"
	return status
}

// listInferenceServingModels returns data-plane truth for every supported
// serving target. It deliberately includes workloads that were started outside
// the release controller so operators can see (and explicitly take down) drift.
func (a *API) listInferenceServingModels(ctx context.Context, runtime map[string]any) []inferenceServingStatus {
	items := make([]inferenceServingStatus, 0, len(supportedInferenceModelSpecs())+1)
	for _, spec := range supportedInferenceModelSpecs() {
		if spec.GPUCount == 1 && strings.Contains(spec.ModelID, "4b") {
			items = append(items, a.inspectAIBrixServingStatus(ctx, spec.ModelID))
		}
	}
	runtimeStatus := strings.ToLower(strings.TrimSpace(stringValue(runtime["status"])))
	runtimeModel := inferenceModelIDFromEndpoint(stringValue(runtime["model"]))
	if runtimeModel != "" && (runtimeStatus == "ready" || runtimeStatus == "starting") {
		layerStatus := runtimeStatus
		if layerStatus == "ready" {
			layerStatus = "ready"
		}
		toolLayer := inferenceServingLayer{Status: "not_checked", Detail: "等待运行时就绪"}
		if runtimeStatus == "ready" {
			code, detail, toolErr := probeToolCalling(ctx, stringValue(runtime["endpoint"]), runtimeModel)
			if toolErr != nil {
				toolLayer = inferenceServingLayer{Status: "unreachable", Detail: toolErr.Error() + detail, HTTPStatus: code}
			} else {
				toolLayer = inferenceServingLayer{Status: "ready", Detail: "Tool Calling 请求已被模型端接受", HTTPStatus: code}
			}
		}
		overall := runtimeStatus
		if runtimeStatus == "ready" && toolLayer.Status != "ready" {
			overall = "degraded"
		}
		items = append(items, inferenceServingStatus{
			Overall: overall, ModelID: runtimeModel, Target: "direct", Managed: true, CanStop: true,
			GatewayEndpoint: stringValue(runtime["endpoint"]), CheckedAt: time.Now().UTC().Format(time.RFC3339Nano),
			Release:     inferenceServingRelease{Status: "running", Phase: runtimeStatus},
			Workload:    inferenceServingLayer{Status: layerStatus, Detail: "宿主 vLLM 调试运行时"},
			Gateway:     inferenceServingLayer{Status: "not_applicable", Detail: "直连模式不经过 AIBrix"},
			ModelRoute:  inferenceServingLayer{Status: layerStatus, Detail: "OpenAI-Compatible 直连 endpoint"},
			ToolCalling: toolLayer,
		})
	}
	return items
}

func (a *API) aibrixGatewayEndpoint() string {
	if endpoint := strings.TrimSpace(a.AIBrixGatewayBaseURL); endpoint != "" {
		return endpoint
	}
	return "http://minikube:30080/v1"
}

func probeAIBrixGateway(ctx context.Context, baseURL string) (int, string, error) {
	probeCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	url := strings.TrimRight(upstreamBaseURL(baseURL), "/") + "/models"
	req, err := http.NewRequestWithContext(probeCtx, http.MethodGet, url, nil)
	if err != nil {
		return 0, "", err
	}
	resp, err := (&http.Client{Timeout: 3 * time.Second}).Do(req)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4<<10))
	// AIBrix commonly returns 404/405 for /v1/models. Any HTTP response proves
	// that the stable NodePort and Envoy listener are reachable; model routing is
	// verified separately with an actual one-token completion.
	return resp.StatusCode, fmt.Sprintf("AIBrix Gateway 可达（HTTP %d；模型路由单独验证）", resp.StatusCode), nil
}

func deriveServingOverall(workload, gateway, route string) string {
	if workload == "ready" && gateway == "ready" && route == "ready" {
		return "ready"
	}
	if workload == "stopped" || workload == "missing" {
		return "stopped"
	}
	if workload == "failed" || gateway == "unreachable" || route == "unreachable" {
		return "degraded"
	}
	if workload == "starting" {
		return "starting"
	}
	return "unavailable"
}
