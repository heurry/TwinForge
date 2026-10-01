// Package httptool adapts versioned HTTP endpoints to tool.Handler.
package httptool

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/resource"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

const (
	defaultMaxResponse  = int64(1 << 20)
	absoluteMaxResponse = int64(16 << 20)
)

// Config supplies deployment policy and secret resolution.
type Config struct {
	AllowedHosts []string
	HTTPClient   *http.Client
	LookupEnv    func(string) (string, bool)
}

// NewHandler validates a versioned ToolSpec and returns its HTTP execution body.
func NewHandler(spec resource.ToolSpec, config Config) (tool.Handler, error) {
	if err := spec.Validate(); err != nil {
		return nil, err
	}
	endpoint, err := url.Parse(spec.HTTP.Endpoint)
	if err != nil {
		return nil, fmt.Errorf("parse tool endpoint: %w", err)
	}
	allowed := normalizeHosts(config.AllowedHosts)
	if !hostAllowed(endpoint, allowed) {
		return nil, fmt.Errorf("tool endpoint host %q is not allowed", endpoint.Host)
	}
	lookup := config.LookupEnv
	if lookup == nil {
		lookup = os.LookupEnv
	}
	client := cloneClient(config.HTTPClient)
	priorRedirect := client.CheckRedirect
	client.CheckRedirect = func(request *http.Request, via []*http.Request) error {
		if !hostAllowed(request.URL, allowed) {
			return fmt.Errorf("redirect host %q is not allowed", request.URL.Host)
		}
		if priorRedirect != nil {
			return priorRedirect(request, via)
		}
		if len(via) >= 5 {
			return errors.New("too many tool endpoint redirects")
		}
		return nil
	}
	maxResponse := spec.HTTP.MaxResponseBytes
	if maxResponse <= 0 {
		maxResponse = defaultMaxResponse
	}
	if maxResponse > absoluteMaxResponse {
		return nil, errors.New("http tool max_response_bytes exceeds 16 MiB")
	}
	timeout := spec.HTTP.Timeout
	if timeout <= 0 {
		timeout = 30 * time.Second
	}

	return func(ctx context.Context, call tool.Call) (tool.Result, error) {
		requestBody, err := json.Marshal(map[string]any{
			"run_id": call.RunID, "turn": call.Turn, "step": call.Step,
			"call_id": call.ID, "arguments": json.RawMessage(call.Arguments),
		})
		if err != nil {
			return tool.Result{}, fmt.Errorf("encode HTTP tool request: %w", err)
		}
		callCtx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()
		request, err := http.NewRequestWithContext(callCtx, http.MethodPost, endpoint.String(), bytes.NewReader(requestBody))
		if err != nil {
			return tool.Result{}, fmt.Errorf("create HTTP tool request: %w", err)
		}
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("Idempotency-Key", call.RunID+":"+call.ID)
		request.Header.Set("X-Agent-Run-ID", call.RunID)
		request.Header.Set("X-Agent-Tool-Call-ID", call.ID)
		for header, environment := range spec.HTTP.HeaderEnvironment {
			value, exists := lookup(environment)
			if !exists || strings.TrimSpace(value) == "" {
				return tool.Result{}, fmt.Errorf("required tool credential environment %q is unavailable", environment)
			}
			request.Header.Set(header, value)
		}
		response, err := client.Do(request)
		if err != nil {
			return tool.Result{}, fmt.Errorf("call HTTP tool: %w", err)
		}
		defer response.Body.Close()
		body, err := io.ReadAll(io.LimitReader(response.Body, maxResponse+1))
		if err != nil {
			return tool.Result{}, fmt.Errorf("read HTTP tool response: %w", err)
		}
		if int64(len(body)) > maxResponse {
			return tool.Result{}, fmt.Errorf("HTTP tool response exceeds %d bytes", maxResponse)
		}
		if response.StatusCode < 200 || response.StatusCode >= 300 {
			message := strings.TrimSpace(string(body))
			if len(message) > 4096 {
				message = message[:4096]
			}
			return tool.Result{}, fmt.Errorf("HTTP tool returned status %d: %s", response.StatusCode, message)
		}
		if !json.Valid(body) {
			return tool.Result{}, errors.New("HTTP tool response must be valid JSON")
		}
		return tool.Result{Content: json.RawMessage(body), Meta: map[string]string{
			"provider": "http", "status": response.Status,
		}}, nil
	}, nil
}

func cloneClient(client *http.Client) *http.Client {
	if client == nil {
		return &http.Client{}
	}
	clone := *client
	return &clone
}

func normalizeHosts(hosts []string) map[string]struct{} {
	result := make(map[string]struct{}, len(hosts))
	for _, host := range hosts {
		if normalized := strings.ToLower(strings.TrimSpace(host)); normalized != "" {
			result[normalized] = struct{}{}
		}
	}
	return result
}

func hostAllowed(endpoint *url.URL, allowed map[string]struct{}) bool {
	if len(allowed) == 0 {
		return false
	}
	host := strings.ToLower(endpoint.Host)
	hostname := strings.ToLower(endpoint.Hostname())
	_, exact := allowed[host]
	_, withoutPort := allowed[hostname]
	return exact || withoutPort
}
