// Package resource defines versioned Prompt, Tool, and ToolSet contracts.
package resource

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

var (
	// ErrPromptVersionNotFound reports an unknown tenant-scoped prompt version.
	ErrPromptVersionNotFound = errors.New("prompt version not found")
	// ErrToolVersionNotFound reports an unknown tenant-scoped tool version.
	ErrToolVersionNotFound = errors.New("tool version not found")
	// ErrToolSetVersionNotFound reports an unknown tenant-scoped tool-set version.
	ErrToolSetVersionNotFound = errors.New("tool set version not found")
)

// VersionRef identifies one immutable resource version.
type VersionRef struct {
	ID      string `json:"id"`
	Version string `json:"version"`
}

// PromptVersion is immutable prompt text selected by AgentSpec.PromptRef.
type PromptVersion struct {
	ID           string    `json:"id"`
	DefinitionID string    `json:"definition_id"`
	TenantID     string    `json:"tenant_id"`
	Key          string    `json:"key"`
	Name         string    `json:"name"`
	Version      int       `json:"version"`
	Content      string    `json:"content"`
	ContentHash  string    `json:"content_hash"`
	Status       string    `json:"status"`
	CreatedBy    *string   `json:"created_by,omitempty"`
	CreatedAt    time.Time `json:"created_at"`
}

// HTTPProvider contains non-secret execution configuration. Header values are
// environment-variable names, never credentials themselves.
type HTTPProvider struct {
	Endpoint          string            `json:"endpoint"`
	Method            string            `json:"method"`
	HeaderEnvironment map[string]string `json:"header_environment,omitempty"`
	Timeout           time.Duration     `json:"timeout"`
	MaxResponseBytes  int64             `json:"max_response_bytes,omitempty"`
}

// WorkspaceProvider binds one built-in filesystem operation. The actual root
// is deployment policy supplied to the Worker and can never be selected by a
// tenant-owned ToolVersion.
type WorkspaceProvider struct {
	Operation string `json:"operation"`
	MaxBytes  int64  `json:"max_bytes,omitempty"`
}

// MCPProvider pins a discovered tool schema to one immutable server version.
type MCPProvider struct {
	ServerVersionID string `json:"server_version_id"`
	ToolName        string `json:"tool_name"`
	SchemaHash      string `json:"schema_hash"`
}

// ToolSpec combines a model-visible definition with one provider binding.
type ToolSpec struct {
	Definition   tool.Definition    `json:"definition"`
	ProviderType string             `json:"provider_type"`
	HTTP         *HTTPProvider      `json:"http,omitempty"`
	Workspace    *WorkspaceProvider `json:"workspace,omitempty"`
	MCP          *MCPProvider       `json:"mcp,omitempty"`
}

// Validate rejects unsupported or unsafe provider declarations.
func (s ToolSpec) Validate() error {
	if err := tool.ValidateDefinition(s.Definition); err != nil {
		return err
	}
	switch s.ProviderType {
	case "workspace":
		if s.Workspace == nil {
			return errors.New("workspace provider configuration is required")
		}
		switch s.Workspace.Operation {
		case "read_file", "list_files", "search_files":
			if s.Definition.Risk != tool.RiskRead {
				return errors.New("read-only workspace operations must use READ risk")
			}
		case "write_file", "append_file", "edit_file", "promote_file", "create_directory":
			if s.Definition.Risk != tool.RiskLowWrite && s.Definition.Risk != tool.RiskHigh {
				return errors.New("mutating workspace operations must use LOW_WRITE or HIGH_RISK")
			}
		case "run_command":
			if s.Definition.Risk != tool.RiskHigh {
				return errors.New("workspace command execution must use HIGH_RISK")
			}
		default:
			return fmt.Errorf("unsupported workspace operation %q", s.Workspace.Operation)
		}
		if s.Workspace.MaxBytes < 0 || s.Workspace.MaxBytes > 1<<20 {
			return errors.New("workspace max_bytes must be between 0 and 1 MiB")
		}
		return nil
	case "http":
		if s.HTTP == nil {
			return errors.New("http provider configuration is required")
		}
	case "mcp":
		if s.MCP == nil || strings.TrimSpace(s.MCP.ServerVersionID) == "" || strings.TrimSpace(s.MCP.ToolName) == "" || strings.TrimSpace(s.MCP.SchemaHash) == "" {
			return errors.New("MCP server version, tool name and schema hash are required")
		}
		return nil
	default:
		return fmt.Errorf("unsupported tool provider %q", s.ProviderType)
	}
	parsed, err := url.Parse(strings.TrimSpace(s.HTTP.Endpoint))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return errors.New("http provider endpoint must be an absolute URL")
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return errors.New("http provider endpoint must use http or https")
	}
	if strings.ToUpper(strings.TrimSpace(s.HTTP.Method)) != "POST" {
		return errors.New("http tool provider currently requires POST")
	}
	for header, environment := range s.HTTP.HeaderEnvironment {
		if strings.TrimSpace(header) == "" || strings.TrimSpace(environment) == "" {
			return errors.New("http tool header and environment names must be non-empty")
		}
	}
	return nil
}

// ToolVersion is one immutable provider-bound Tool revision.
type ToolVersion struct {
	ID           string          `json:"id"`
	DefinitionID string          `json:"definition_id"`
	TenantID     string          `json:"tenant_id"`
	Key          string          `json:"key"`
	Name         string          `json:"name"`
	Version      int             `json:"version"`
	Spec         json.RawMessage `json:"spec"`
	SpecHash     string          `json:"spec_hash"`
	Status       string          `json:"status"`
	CreatedBy    *string         `json:"created_by,omitempty"`
	CreatedAt    time.Time       `json:"created_at"`
}

// ToolSetSpec pins exact ToolVersion identifiers.
type ToolSetSpec struct {
	Tools []VersionRef `json:"tools"`
}

// Validate rejects duplicate or incomplete Tool references.
func (s ToolSetSpec) Validate() error {
	seen := make(map[string]struct{}, len(s.Tools))
	for index, ref := range s.Tools {
		if strings.TrimSpace(ref.ID) == "" || strings.TrimSpace(ref.Version) == "" {
			return fmt.Errorf("tools[%d] id and version are required", index)
		}
		if _, exists := seen[ref.ID]; exists {
			return fmt.Errorf("tool version %q is duplicated", ref.ID)
		}
		seen[ref.ID] = struct{}{}
	}
	return nil
}

// ToolSetVersion is one immutable list of Tool versions.
type ToolSetVersion struct {
	ID           string          `json:"id"`
	DefinitionID string          `json:"definition_id"`
	TenantID     string          `json:"tenant_id"`
	Key          string          `json:"key"`
	Name         string          `json:"name"`
	Version      int             `json:"version"`
	Spec         json.RawMessage `json:"spec"`
	SpecHash     string          `json:"spec_hash"`
	Status       string          `json:"status"`
	CreatedBy    *string         `json:"created_by,omitempty"`
	CreatedAt    time.Time       `json:"created_at"`
}

// CreatePromptVersion appends a published version under a tenant key.
type CreatePromptVersion struct {
	TenantID  string
	Key       string
	Name      string
	Content   string
	CreatedBy *string
}

// CreateToolVersion appends a published provider-bound Tool version.
type CreateToolVersion struct {
	TenantID  string
	Key       string
	Name      string
	Spec      ToolSpec
	CreatedBy *string
}

// CreateToolSetVersion appends a published set of exact Tool versions.
type CreateToolSetVersion struct {
	TenantID  string
	Key       string
	Name      string
	Spec      ToolSetSpec
	CreatedBy *string
}
