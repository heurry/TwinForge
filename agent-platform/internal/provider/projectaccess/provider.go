// Package projectaccess is the client for the trusted project file broker.
// The broker is isolated from model code and is the only component that sees
// the operator project mount.
package projectaccess

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path/filepath"
	"strings"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

type Config struct {
	Endpoint   string
	Token      string
	HTTPClient *http.Client
}

type ReadRequest struct {
	RunID     string `json:"run_id"`
	Path      string `json:"path"`
	Reason    string `json:"reason"`
	StartLine int    `json:"start_line,omitempty"`
	LineCount int    `json:"line_count,omitempty"`
}

func (r ReadRequest) ValidateModelInput() error {
	path := strings.TrimSpace(r.Path)
	if path == "" || filepath.IsAbs(path) {
		return errors.New("project path must be non-empty and relative")
	}
	clean := filepath.Clean(path)
	if clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return errors.New("project path escapes the bound project root")
	}
	internal := filepath.ToSlash(clean)
	if internal == ".agent-workspaces" || strings.HasPrefix(internal, ".agent-workspaces/") {
		return errors.New("project access cannot read internal Run workspaces")
	}
	if strings.TrimSpace(r.Reason) == "" {
		return errors.New("project file access reason is required")
	}
	if r.StartLine < 0 || r.LineCount <= 0 || r.LineCount > 200 {
		return errors.New("project file line range is invalid; line_count must be between 1 and 200")
	}
	return nil
}

type ReadResponse struct {
	Path            string `json:"path"`
	Bytes           int64  `json:"bytes"`
	SHA256          string `json:"sha256"`
	LineCount       int    `json:"line_count"`
	StartLine       int    `json:"start_line,omitempty"`
	Content         string `json:"content"`
	MediaType       string `json:"media_type"`
	SnapshotContent []byte `json:"snapshot_content"`
	Error           string `json:"error,omitempty"`
}

type Client struct {
	endpoint string
	token    string
	client   *http.Client
}

func NewClient(config Config) (*Client, error) {
	endpoint, err := url.Parse(strings.TrimSpace(config.Endpoint))
	if err != nil || endpoint.Scheme != "http" || endpoint.Host == "" {
		return nil, errors.New("project access endpoint must be an operator-owned absolute http URL")
	}
	if strings.TrimSpace(config.Token) == "" {
		return nil, errors.New("project access authentication token is required")
	}
	client := config.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	return &Client{endpoint: strings.TrimRight(config.Endpoint, "/"), token: config.Token, client: client}, nil
}

func (c *Client) Read(ctx context.Context, input ReadRequest) (tool.Result, error) {
	if err := input.ValidateModelInput(); err != nil {
		return tool.Result{}, err
	}
	body, err := json.Marshal(input)
	if err != nil {
		return tool.Result{}, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint+"/v1/read", bytes.NewReader(body))
	if err != nil {
		return tool.Result{}, err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+c.token)
	response, err := c.client.Do(request)
	if err != nil {
		return tool.Result{}, fmt.Errorf("project access broker: %w", err)
	}
	defer response.Body.Close()
	payload, err := io.ReadAll(io.LimitReader(response.Body, 3<<20))
	if err != nil {
		return tool.Result{}, err
	}
	var decoded ReadResponse
	if json.Unmarshal(payload, &decoded) != nil {
		return tool.Result{}, fmt.Errorf("project access broker returned invalid JSON (HTTP %d)", response.StatusCode)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return tool.Result{}, errors.New(decoded.Error)
	}
	artifactContent := append([]byte(nil), decoded.SnapshotContent...)
	decoded.SnapshotContent = nil
	content, err := json.Marshal(decoded)
	if err != nil {
		return tool.Result{}, err
	}
	return tool.Result{
		Content: content,
		Meta: map[string]string{
			"provider": "project_access", "scope": "approved_once", "project_path": decoded.Path,
		},
		Artifacts: []tool.Artifact{{
			Kind: "input_file_snapshot", Name: decoded.Path, MediaType: decoded.MediaType, Content: artifactContent,
			Metadata: map[string]string{"source": "project", "path": decoded.Path, "sha256": decoded.SHA256, "access_scope": "approved_once"},
		}},
	}, nil
}
