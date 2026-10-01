// Package dependency implements the structured, run-scoped dependency
// installation boundary. The model can name packages, but cannot provide a
// command, URL, target path, environment variable, or shell fragment.
package dependency

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"sort"
	"strings"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

const ToolName = "install_dependency"

var (
	packageNamePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)
	versionPattern     = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9.!+_-]{0,127}$`)
)

type Package struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type Request struct {
	RunID     string    `json:"run_id,omitempty"`
	Ecosystem string    `json:"ecosystem"`
	Packages  []Package `json:"packages"`
	Source    string    `json:"source"`
	Scope     string    `json:"scope"`
	Reason    string    `json:"reason"`
}

type Result struct {
	Ecosystem  string    `json:"ecosystem"`
	Packages   []Package `json:"packages"`
	Source     string    `json:"source"`
	Scope      string    `json:"scope"`
	Target     string    `json:"target"`
	Stdout     string    `json:"stdout,omitempty"`
	Stderr     string    `json:"stderr,omitempty"`
	DurationMS int64     `json:"duration_ms"`
}

type Response struct {
	Result Result `json:"result"`
	Error  string `json:"error,omitempty"`
}

func (r *Request) NormalizeAndValidate() error {
	r.Ecosystem = strings.ToLower(strings.TrimSpace(r.Ecosystem))
	r.Source = strings.ToLower(strings.TrimSpace(r.Source))
	r.Scope = strings.ToLower(strings.TrimSpace(r.Scope))
	r.Reason = strings.TrimSpace(r.Reason)
	if r.Ecosystem != "python" {
		return errors.New("ecosystem must be python")
	}
	if r.Scope != "run" {
		return errors.New("scope must be run")
	}
	if r.Source == "" || len(r.Source) > 64 || !packageNamePattern.MatchString(r.Source) {
		return errors.New("source must be a configured source name")
	}
	if len(r.Packages) == 0 || len(r.Packages) > 16 {
		return errors.New("packages must contain 1-16 exact dependencies")
	}
	if r.Reason == "" || len(r.Reason) > 500 {
		return errors.New("reason is required and must not exceed 500 characters")
	}
	seen := make(map[string]struct{}, len(r.Packages))
	for index := range r.Packages {
		r.Packages[index].Name = strings.TrimSpace(r.Packages[index].Name)
		r.Packages[index].Version = strings.TrimSpace(r.Packages[index].Version)
		if !packageNamePattern.MatchString(r.Packages[index].Name) {
			return fmt.Errorf("packages[%d].name is invalid", index)
		}
		if !versionPattern.MatchString(r.Packages[index].Version) {
			return fmt.Errorf("packages[%d].version must be an exact version", index)
		}
		key := strings.ToLower(strings.ReplaceAll(r.Packages[index].Name, "_", "-"))
		if _, exists := seen[key]; exists {
			return fmt.Errorf("package %q is duplicated", r.Packages[index].Name)
		}
		seen[key] = struct{}{}
	}
	sort.Slice(r.Packages, func(i, j int) bool {
		return strings.ToLower(r.Packages[i].Name) < strings.ToLower(r.Packages[j].Name)
	})
	return nil
}

// Definition is deliberately narrower than pip. In particular there is no
// command, index URL, filesystem target, editable/VCS dependency, or argument
// passthrough in the model-visible contract.
func Definition(sourceNames []string) tool.Definition {
	values := make([]string, 0, len(sourceNames))
	seen := map[string]struct{}{}
	for _, item := range sourceNames {
		item = strings.ToLower(strings.TrimSpace(item))
		if item != "" {
			if _, exists := seen[item]; !exists {
				values = append(values, item)
				seen[item] = struct{}{}
			}
		}
	}
	sort.Strings(values)
	if len(values) == 0 {
		values = []string{"pypi"}
	}
	schema, _ := json.Marshal(map[string]any{
		"type": "object", "additionalProperties": false,
		"required": []string{"ecosystem", "packages", "source", "scope", "reason"},
		"properties": map[string]any{
			"ecosystem": map[string]any{"const": "python"},
			"packages": map[string]any{"type": "array", "minItems": 1, "maxItems": 16, "items": map[string]any{
				"type": "object", "additionalProperties": false, "required": []string{"name", "version"},
				"properties": map[string]any{"name": map[string]any{"type": "string", "pattern": packageNamePattern.String()}, "version": map[string]any{"type": "string", "pattern": versionPattern.String()}},
			}},
			"source": map[string]any{"type": "string", "enum": values},
			"scope":  map[string]any{"const": "run"},
			"reason": map[string]any{"type": "string", "minLength": 1, "maxLength": 500},
		},
	})
	return tool.Definition{
		Name: ToolName, Version: "1", Risk: tool.RiskHigh, ExecutionMode: tool.ExecutionSerial,
		Description: "Use only when the current Run needs a Python package that is not already available in the prebuilt environment. Request exact pinned versions for this Run; the call always pauses for human approval. Do not use for shell commands, OS packages, URLs, local paths, editable/VCS installs, or unpinned versions. Before calling, explain why the package is needed and choose one offered source. Installation is isolated to this Run and does not modify the host or other Runs.",
		InputSchema: schema,
	}
}

type Config struct {
	Endpoint   string
	Token      string
	Sources    map[string]string
	HTTPClient *http.Client
}

type Client struct {
	endpoint string
	token    string
	sources  map[string]string
	client   *http.Client
}

func NewClient(config Config) (*Client, error) {
	endpoint, err := url.Parse(strings.TrimSpace(config.Endpoint))
	if err != nil || endpoint.Scheme != "http" || endpoint.Host == "" {
		return nil, errors.New("dependency installer endpoint must be an operator-owned absolute http URL")
	}
	if strings.TrimSpace(config.Token) == "" {
		return nil, errors.New("dependency installer authentication token is required")
	}
	if len(config.Sources) == 0 {
		return nil, errors.New("at least one dependency source is required")
	}
	sources := make(map[string]string, len(config.Sources))
	for name, rawURL := range config.Sources {
		name = strings.ToLower(strings.TrimSpace(name))
		parsed, parseErr := url.Parse(strings.TrimSpace(rawURL))
		if !packageNamePattern.MatchString(name) || parseErr != nil || parsed.Scheme != "https" || parsed.Host == "" || parsed.User != nil {
			return nil, fmt.Errorf("dependency source %q must be a named absolute https URL", name)
		}
		sources[name] = parsed.String()
	}
	client := config.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	return &Client{endpoint: strings.TrimRight(config.Endpoint, "/"), token: config.Token, sources: sources, client: client}, nil
}

func (c *Client) SourceNames() []string {
	result := make([]string, 0, len(c.sources))
	for name := range c.sources {
		result = append(result, name)
	}
	sort.Strings(result)
	return result
}

func (c *Client) Install(ctx context.Context, request Request) (Result, error) {
	if err := request.NormalizeAndValidate(); err != nil {
		return Result{}, err
	}
	if _, ok := c.sources[request.Source]; !ok {
		return Result{}, fmt.Errorf("dependency source %q is not configured", request.Source)
	}
	body, err := json.Marshal(request)
	if err != nil {
		return Result{}, err
	}
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint+"/v1/dependencies/install", bytes.NewReader(body))
	if err != nil {
		return Result{}, err
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Authorization", "Bearer "+c.token)
	response, err := c.client.Do(httpRequest)
	if err != nil {
		return Result{}, fmt.Errorf("dependency installer: %w", err)
	}
	defer response.Body.Close()
	payload, err := io.ReadAll(io.LimitReader(response.Body, 2<<20))
	if err != nil {
		return Result{}, err
	}
	var decoded Response
	if json.Unmarshal(payload, &decoded) != nil {
		return Result{}, fmt.Errorf("dependency installer returned invalid JSON (HTTP %d)", response.StatusCode)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return decoded.Result, errors.New(decoded.Error)
	}
	return decoded.Result, nil
}
