// Package mcp implements the MCP 2025-06-18 Streamable HTTP client contract.
package mcp

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const ProtocolVersion = "2025-06-18"

type ServerSpec struct {
	Transport         string            `json:"transport"`
	Endpoint          string            `json:"endpoint"`
	ProtocolVersion   string            `json:"protocol_version"`
	HeaderEnvironment map[string]string `json:"header_environment,omitempty"`
	Timeout           time.Duration     `json:"timeout"`
	MaxResponseBytes  int64             `json:"max_response_bytes,omitempty"`
}

func (s ServerSpec) Validate(allowedHosts []string) error {
	parsed, err := url.Parse(strings.TrimSpace(s.Endpoint))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return errors.New("MCP endpoint must be an absolute URL")
	}
	if parsed.Scheme != "https" && parsed.Scheme != "http" {
		return errors.New("MCP endpoint must use http or https")
	}
	if strings.ToLower(s.Transport) != "streamable_http" {
		return errors.New("only streamable_http MCP transport is enabled")
	}
	if version := s.EffectiveProtocolVersion(); version != ProtocolVersion {
		return fmt.Errorf("unsupported MCP protocol version %q", version)
	}
	if !hostAllowed(parsed.Host, allowedHosts) {
		return fmt.Errorf("MCP host %q is not operator-allowed", parsed.Host)
	}
	return nil
}
func (s ServerSpec) EffectiveProtocolVersion() string {
	if strings.TrimSpace(s.ProtocolVersion) == "" {
		return ProtocolVersion
	}
	return s.ProtocolVersion
}

type ToolSnapshot struct {
	ID              string          `json:"id,omitempty"`
	ServerVersionID string          `json:"server_version_id"`
	Name            string          `json:"name"`
	Description     string          `json:"description,omitempty"`
	InputSchema     json.RawMessage `json:"input_schema"`
	SchemaHash      string          `json:"schema_hash"`
	Risk            string          `json:"risk"`
	Enabled         bool            `json:"enabled"`
	SyncedAt        time.Time       `json:"synced_at"`
}
type ServerVersion struct {
	ID           string          `json:"id"`
	DefinitionID string          `json:"definition_id"`
	TenantID     string          `json:"tenant_id"`
	Key          string          `json:"key"`
	Name         string          `json:"name"`
	Version      int             `json:"version"`
	Spec         json.RawMessage `json:"spec"`
	SpecHash     string          `json:"spec_hash"`
	Status       string          `json:"status"`
	CreatedAt    time.Time       `json:"created_at"`
	Health       *Health         `json:"health,omitempty"`
	Tools        []ToolSnapshot  `json:"tools,omitempty"`
}
type Health struct {
	Status          string    `json:"status"`
	ProtocolVersion string    `json:"protocol_version,omitempty"`
	LatencyMS       int64     `json:"latency_ms,omitempty"`
	Error           string    `json:"error,omitempty"`
	CheckedAt       time.Time `json:"checked_at"`
}
type CreateServerVersion struct {
	TenantID  string
	Key       string
	Name      string
	Spec      ServerSpec
	CreatedBy *string
}

type Client struct {
	spec      ServerSpec
	client    *http.Client
	headers   http.Header
	mu        sync.Mutex
	sessionID string
	nextID    atomic.Int64
}

func NewClient(spec ServerSpec, allowedHosts []string, lookup func(string) (string, bool), client *http.Client) (*Client, error) {
	if err := spec.Validate(allowedHosts); err != nil {
		return nil, err
	}
	if lookup == nil {
		return nil, errors.New("MCP secret lookup is required")
	}
	headers := make(http.Header)
	for name, environment := range spec.HeaderEnvironment {
		value, ok := lookup(environment)
		if !ok || strings.TrimSpace(value) == "" {
			return nil, fmt.Errorf("MCP secret environment %q is unavailable", environment)
		}
		headers.Set(name, value)
	}
	if client == nil {
		client = http.DefaultClient
	}
	return &Client{spec: spec, client: client, headers: headers}, nil
}

func (c *Client) Initialize(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.sessionID != "" {
		return nil
	}
	var result struct {
		ProtocolVersion string `json:"protocolVersion"`
	}
	if err := c.callLocked(ctx, "initialize", map[string]any{"protocolVersion": c.spec.EffectiveProtocolVersion(), "capabilities": map[string]any{}, "clientInfo": map[string]string{"name": "TwinForge Agent Platform", "version": "0.8.0"}}, &result); err != nil {
		return err
	}
	if result.ProtocolVersion != c.spec.EffectiveProtocolVersion() {
		return fmt.Errorf("MCP negotiated unsupported protocol %q", result.ProtocolVersion)
	}
	return c.notifyLocked(ctx, "notifications/initialized", map[string]any{})
}
func (c *Client) ListTools(ctx context.Context) ([]ToolSnapshot, error) {
	if err := c.Initialize(ctx); err != nil {
		return nil, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	var response struct {
		Tools []struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			InputSchema json.RawMessage `json:"inputSchema"`
		} `json:"tools"`
	}
	if err := c.callLocked(ctx, "tools/list", map[string]any{}, &response); err != nil {
		return nil, err
	}
	now := time.Now().UTC()
	out := make([]ToolSnapshot, 0, len(response.Tools))
	for _, item := range response.Tools {
		sum := sha256.Sum256(item.InputSchema)
		out = append(out, ToolSnapshot{Name: item.Name, Description: item.Description, InputSchema: item.InputSchema, SchemaHash: "sha256:" + hex.EncodeToString(sum[:]), Risk: "READ", Enabled: true, SyncedAt: now})
	}
	return out, nil
}
func (c *Client) CallTool(ctx context.Context, name string, arguments json.RawMessage) (json.RawMessage, bool, error) {
	if err := c.Initialize(ctx); err != nil {
		return nil, false, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	var result struct {
		Content           json.RawMessage `json:"content"`
		StructuredContent json.RawMessage `json:"structuredContent"`
		IsError           bool            `json:"isError"`
	}
	if err := c.callLocked(ctx, "tools/call", map[string]any{"name": name, "arguments": json.RawMessage(arguments)}, &result); err != nil {
		return nil, false, err
	}
	if len(result.StructuredContent) > 0 && string(result.StructuredContent) != "null" {
		return result.StructuredContent, result.IsError, nil
	}
	return result.Content, result.IsError, nil
}
func (c *Client) callLocked(ctx context.Context, method string, params any, destination any) error {
	id := c.nextID.Add(1)
	payload := map[string]any{"jsonrpc": "2.0", "id": id, "method": method, "params": params}
	raw, err := c.postLocked(ctx, payload, false)
	if err != nil {
		return err
	}
	var envelope struct {
		ID     int64           `json:"id"`
		Result json.RawMessage `json:"result"`
		Error  *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return fmt.Errorf("decode MCP response: %w", err)
	}
	if envelope.Error != nil {
		return fmt.Errorf("MCP JSON-RPC %d: %s", envelope.Error.Code, envelope.Error.Message)
	}
	return json.Unmarshal(envelope.Result, destination)
}
func (c *Client) notifyLocked(ctx context.Context, method string, params any) error {
	_, err := c.postLocked(ctx, map[string]any{"jsonrpc": "2.0", "method": method, "params": params}, true)
	return err
}
func (c *Client) postLocked(ctx context.Context, payload any, notification bool) (json.RawMessage, error) {
	body, _ := json.Marshal(payload)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.spec.Endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header = c.headers.Clone()
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json, text/event-stream")
	request.Header.Set("MCP-Protocol-Version", c.spec.EffectiveProtocolVersion())
	if c.sessionID != "" {
		request.Header.Set("Mcp-Session-Id", c.sessionID)
	}
	response, err := c.client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if session := response.Header.Get("Mcp-Session-Id"); session != "" {
		c.sessionID = session
	}
	limit := c.spec.MaxResponseBytes
	if limit <= 0 || limit > 10<<20 {
		limit = 2 << 20
	}
	raw, err := io.ReadAll(io.LimitReader(response.Body, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(raw)) > limit {
		return nil, errors.New("MCP response exceeds configured limit")
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, fmt.Errorf("MCP endpoint returned HTTP %d", response.StatusCode)
	}
	if notification {
		return nil, nil
	}
	if strings.Contains(response.Header.Get("Content-Type"), "text/event-stream") {
		scanner := bufio.NewScanner(bytes.NewReader(raw))
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "data:") {
				return json.RawMessage(strings.TrimSpace(strings.TrimPrefix(line, "data:"))), nil
			}
		}
		return nil, errors.New("MCP SSE response contained no data")
	}
	return raw, nil
}
func hostAllowed(host string, allowed []string) bool {
	for _, candidate := range allowed {
		if strings.EqualFold(strings.TrimSpace(candidate), host) {
			return true
		}
	}
	return false
}
