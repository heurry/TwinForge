// Package sandbox executes workspace tools through an operator-owned isolated
// service. The Worker never receives a host path and cannot open workspace files.
package sandbox

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/resource"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

type Config struct {
	Endpoint   string
	Token      string
	HTTPClient *http.Client
}
type ExecuteRequest struct {
	Spec    resource.ToolSpec `json:"spec"`
	// Call is a deliberately narrow wire contract. Workflow correlation fields
	// are Worker-internal and must never be forwarded to the sandbox's strict
	// decoder as executable request fields.
	Call    WireCall          `json:"call"`
	Preview bool              `json:"preview,omitempty"`
}

type WireCall struct {
	RunID       string          `json:"run_id"`
	WorkspaceID string          `json:"workspace_id,omitempty"`
	PlanStepID  string          `json:"plan_step_id,omitempty"`
	Turn        int             `json:"turn"`
	Step        int             `json:"step"`
	ID          string          `json:"id"`
	Name        string          `json:"name"`
	Arguments   json.RawMessage `json:"arguments"`
}

func wireCall(call tool.Call) WireCall {
	return WireCall{RunID: call.RunID, WorkspaceID: call.WorkspaceID, PlanStepID: call.PlanStepID, Turn: call.Turn, Step: call.Step, ID: call.ID, Name: call.Name, Arguments: call.Arguments}
}

func (c WireCall) ToolCall() tool.Call {
	return tool.Call{RunID: c.RunID, WorkspaceID: c.WorkspaceID, PlanStepID: c.PlanStepID, Turn: c.Turn, Step: c.Step, ID: c.ID, Name: c.Name, Arguments: c.Arguments}
}
type ExecuteResponse struct {
	Result    tool.Result    `json:"result"`
	Artifacts []WireArtifact `json:"artifacts,omitempty"`
	Error     string         `json:"error,omitempty"`
}
type WireArtifact struct {
	Kind      string            `json:"kind"`
	Name      string            `json:"name"`
	MediaType string            `json:"media_type"`
	Content   []byte            `json:"content"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

type PromoteRequest struct {
	RunID                string `json:"run_id"`
	SourcePath           string `json:"source_path"`
	SourceHash           string `json:"source_hash"`
	TargetPath           string `json:"target_path"`
	ExpectedTargetSHA256 string `json:"expected_target_sha256,omitempty"`
}

type PromoteResponse struct {
	TargetPath           string `json:"target_path,omitempty"`
	PreviousTargetSHA256 string `json:"previous_target_sha256,omitempty"`
	ResultTargetSHA256   string `json:"result_target_sha256,omitempty"`
	Created              bool   `json:"created,omitempty"`
	Error                string `json:"error,omitempty"`
}

type Promoter struct {
	endpoint string
	token    string
	client   *http.Client
}

func NewPromoter(config Config) (*Promoter, error) {
	endpoint, err := url.Parse(strings.TrimSpace(config.Endpoint))
	if err != nil || endpoint.Scheme != "http" || endpoint.Host == "" {
		return nil, errors.New("sandbox endpoint must be an operator-owned absolute http URL")
	}
	if strings.TrimSpace(config.Token) == "" {
		return nil, errors.New("sandbox authentication token is required")
	}
	client := config.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	return &Promoter{endpoint: strings.TrimRight(config.Endpoint, "/"), token: config.Token, client: client}, nil
}

func (p *Promoter) Promote(ctx context.Context, input PromoteRequest) (PromoteResponse, error) {
	body, err := json.Marshal(input)
	if err != nil {
		return PromoteResponse{}, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint+"/v1/promote", bytes.NewReader(body))
	if err != nil {
		return PromoteResponse{}, err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+p.token)
	response, err := p.client.Do(request)
	if err != nil {
		return PromoteResponse{}, fmt.Errorf("sandbox promote: %w", err)
	}
	defer response.Body.Close()
	payload, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		return PromoteResponse{}, err
	}
	var decoded PromoteResponse
	if json.Unmarshal(payload, &decoded) != nil {
		return PromoteResponse{}, fmt.Errorf("sandbox returned invalid promotion JSON (HTTP %d)", response.StatusCode)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return decoded, errors.New(decoded.Error)
	}
	return decoded, nil
}

func NewHandler(spec resource.ToolSpec, config Config) (tool.Handler, error) {
	return newHandler(spec, config, false)
}

// NewPreviewHandler asks the isolated service to compute a write Diff without
// applying it. The Worker still never receives direct filesystem access.
func NewPreviewHandler(spec resource.ToolSpec, config Config) (tool.Handler, error) {
	return newHandler(spec, config, true)
}

func newHandler(spec resource.ToolSpec, config Config, preview bool) (tool.Handler, error) {
	if err := spec.Validate(); err != nil {
		return nil, err
	}
	if spec.ProviderType != "workspace" {
		return nil, errors.New("sandbox only executes workspace tools")
	}
	endpoint, err := url.Parse(strings.TrimSpace(config.Endpoint))
	if err != nil || endpoint.Scheme != "http" || endpoint.Host == "" {
		return nil, errors.New("sandbox endpoint must be an operator-owned absolute http URL")
	}
	if strings.TrimSpace(config.Token) == "" {
		return nil, errors.New("sandbox authentication token is required")
	}
	client := config.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	return func(ctx context.Context, call tool.Call) (tool.Result, error) {
		body, err := json.Marshal(ExecuteRequest{Spec: spec, Call: wireCall(call), Preview: preview})
		if err != nil {
			return tool.Result{}, err
		}
		request, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(config.Endpoint, "/")+"/v1/execute", bytes.NewReader(body))
		if err != nil {
			return tool.Result{}, err
		}
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("Authorization", "Bearer "+config.Token)
		response, err := client.Do(request)
		if err != nil {
			return tool.Result{}, fmt.Errorf("sandbox execute: %w", err)
		}
		defer response.Body.Close()
		payload, err := io.ReadAll(io.LimitReader(response.Body, 12<<20))
		if err != nil {
			return tool.Result{}, err
		}
		var decoded ExecuteResponse
		if json.Unmarshal(payload, &decoded) != nil {
			return tool.Result{}, fmt.Errorf("sandbox returned invalid JSON (HTTP %d)", response.StatusCode)
		}
		if response.StatusCode < 200 || response.StatusCode >= 300 {
			return tool.Result{}, fmt.Errorf("sandbox rejected execution: %s", decoded.Error)
		}
		for _, item := range decoded.Artifacts {
			decoded.Result.Artifacts = append(decoded.Result.Artifacts, tool.Artifact{Kind: item.Kind, Name: item.Name, MediaType: item.MediaType, Content: item.Content, Metadata: item.Metadata})
		}
		return decoded.Result, nil
	}, nil
}
