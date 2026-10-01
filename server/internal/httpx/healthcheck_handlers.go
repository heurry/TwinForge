package httpx

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
)

const healthProbeTimeout = 5 * time.Second

// serviceInstanceHealthcheck performs a real active probe. Model-serving
// endpoints use the OpenAI-compatible models endpoint; ordinary services use
// healthz/health. A metadata.health_path value overrides these defaults.
func (a *API) serviceInstanceHealthcheck(w http.ResponseWriter, r *http.Request) {
	name := chi.URLParam(r, "name")
	row, err := a.fetchInstance(r.Context(), name)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	if row == nil {
		WriteError(w, r, http.StatusNotFound, "not_found", "service instance not found")
		return
	}

	target := row
	if row.Kind == "auto_router" || row.Kind == "client_round_robin" || row.RoutingRole == "auto_router" || row.RoutingRole == "client_round_robin" {
		resolved, _, resolveErr := a.resolveEndpoint(r.Context(), row.Name)
		if resolveErr != nil {
			a.persistHealthResult(r.Context(), row.Name, "unreachable", 0, "", resolveErr.Error())
			WriteJSON(w, http.StatusOK, map[string]any{"name": name, "status": "unreachable", "detail": resolveErr.Error()})
			return
		}
		kind, role := "vllm", ""
		if resolved.RoutingStrategy != "" {
			kind, role = "aibrix", "gateway"
		}
		target = &instanceRow{Name: resolved.TargetPod, BaseURL: resolved.BaseURL, ModelID: resolved.ModelID, Kind: kind, RoutingRole: role}
	}

	paths := healthPaths(target)
	started := time.Now()
	var probeURL string
	var code int
	var detail string
	var probeErr error
	if strings.Contains(strings.ToLower(target.Kind+" "+target.RoutingRole), "aibrix") {
		probeURL, code, detail, probeErr = probeAIBrix(r.Context(), target.BaseURL, target.ModelID)
	} else {
		probeURL, code, detail, probeErr = probeService(r.Context(), target.BaseURL, paths)
	}
	latencyMs := float64(time.Since(started).Microseconds()) / 1000
	status := "healthy"
	if probeErr != nil {
		status = "unreachable"
		detail = probeErr.Error()
	}
	if err := a.persistHealthResult(r.Context(), row.Name, status, latencyMs, probeURL, detail); err != nil {
		a.fail(w, r, err)
		return
	}
	level := "info"
	if status != "healthy" {
		level = "error"
	}
	_ = a.recordPlatformLog(r.Context(), platformLogInput{
		Level: level, Source: "healthcheck", ResourceType: "service_instance", ResourceID: name,
		Message:    status + ": " + orDefault(detail, probeURL),
		Attributes: map[string]any{"target": target.Name, "url": probeURL, "http_status": code, "latency_ms": latencyMs},
	})
	operator := a.actor(r, "")
	a.Store.Audit(r.Context(), operator, "operator", "service.healthcheck", "service_instance", name, map[string]any{
		"status": status, "target": target.Name, "url": probeURL, "http_status": code, "latency_ms": latencyMs, "detail": detail,
	})
	WriteJSON(w, http.StatusOK, map[string]any{
		"name": name, "target": target.Name, "status": status, "url": probeURL,
		"http_status": code, "latency_ms": latencyMs, "detail": detail,
	})
}

// probeAIBrix uses a one-token Chat Completions request because the AIBrix
// data plane usually does not expose GET /v1/models.
func probeAIBrix(ctx context.Context, baseURL, model string) (string, int, string, error) {
	root := strings.TrimRight(strings.TrimSpace(upstreamBaseURL(baseURL)), "/")
	url := root + "/chat/completions"
	payload := map[string]any{
		"model":      model,
		"messages":   []map[string]string{{"role": "user", "content": "ping"}},
		"max_tokens": 1, "temperature": 0,
		"chat_template_kwargs": map[string]bool{"enable_thinking": false},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return url, 0, "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return url, 0, "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: healthProbeTimeout}).Do(req)
	if err != nil {
		return url, 0, "", err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
	detail := strings.TrimSpace(string(raw))
	if resp.StatusCode < 200 || resp.StatusCode >= 400 {
		return url, resp.StatusCode, detail, fmt.Errorf("%s returned HTTP %d", url, resp.StatusCode)
	}
	return url, resp.StatusCode, detail, nil
}

// probeToolCalling verifies the serving process accepts OpenAI-compatible tool
// schemas. A plain chat probe is insufficient for Agent workloads because vLLM
// requires explicit auto-tool-choice and parser startup flags.
func probeToolCalling(ctx context.Context, baseURL, model string) (int, string, error) {
	root := strings.TrimRight(strings.TrimSpace(upstreamBaseURL(baseURL)), "/")
	url := root + "/chat/completions"
	payload := map[string]any{
		"model":    model,
		"messages": []map[string]string{{"role": "user", "content": "Call the health_probe tool with an empty object."}},
		"tools": []map[string]any{{"type": "function", "function": map[string]any{
			"name": "health_probe", "description": "Tool-calling readiness probe",
			"parameters": map[string]any{"type": "object", "additionalProperties": false},
		}}},
		"tool_choice": "auto", "max_tokens": 16, "temperature": 0,
		"chat_template_kwargs": map[string]bool{"enable_thinking": false},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return 0, "", err
	}
	probeCtx, cancel := context.WithTimeout(ctx, healthProbeTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(probeCtx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return 0, "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: healthProbeTimeout}).Do(req)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
	detail := strings.TrimSpace(string(raw))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return resp.StatusCode, detail, fmt.Errorf("tool-calling probe returned HTTP %d: ", resp.StatusCode)
	}
	return resp.StatusCode, detail, nil
}

func healthPaths(row *instanceRow) []string {
	var metadata map[string]any
	_ = json.Unmarshal(row.Metadata, &metadata)
	if path, _ := metadata["health_path"].(string); strings.TrimSpace(path) != "" {
		return []string{path}
	}
	kind := strings.ToLower(row.Kind + " " + row.RoutingRole)
	if strings.Contains(kind, "vllm") || strings.Contains(kind, "model") || strings.Contains(kind, "aibrix") || row.ModelID != "" {
		return []string{"/v1/models", "/health", "/healthz"}
	}
	return []string{"/healthz", "/health", "/readyz"}
}

func probeService(ctx context.Context, baseURL string, paths []string) (string, int, string, error) {
	client := &http.Client{Timeout: healthProbeTimeout}
	root := strings.TrimRight(strings.TrimSpace(upstreamBaseURL(baseURL)), "/")
	if strings.HasSuffix(root, "/v1") {
		root = strings.TrimSuffix(root, "/v1")
	}
	var lastErr error
	for _, path := range paths {
		if !strings.HasPrefix(path, "/") {
			path = "/" + path
		}
		url := root + path
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			lastErr = err
			continue
		}
		resp, err := client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		resp.Body.Close()
		detail := strings.TrimSpace(string(raw))
		if resp.StatusCode >= 200 && resp.StatusCode < 400 {
			return url, resp.StatusCode, detail, nil
		}
		lastErr = fmt.Errorf("%s returned HTTP %d", url, resp.StatusCode)
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("no health probe path configured")
	}
	return "", 0, "", lastErr
}

func (a *API) persistHealthResult(ctx context.Context, name, status string, latencyMs float64, url, detail string) error {
	meta, _ := json.Marshal(map[string]any{
		"healthcheck": map[string]any{
			"status": status, "latency_ms": latencyMs, "url": url, "detail": detail,
			"checked_at": time.Now().UTC().Format(time.RFC3339Nano),
		},
	})
	_, err := a.Pool.Exec(ctx, `UPDATE service_instances
		SET status=$2, last_checked_at=now(), updated_at=now(), metadata=metadata || $3::jsonb
		WHERE name=$1`, name, status, meta)
	return err
}
