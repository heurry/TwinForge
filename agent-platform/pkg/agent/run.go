package agent

import (
	"encoding/json"
	"time"
)

// Run is the durable execution record owned by the control plane.
type Run struct {
	ID string `json:"id"`
	// WorkflowID is the stable long-lived task identity. A Workflow may have
	// multiple Run attempts when a user continues a task or a Worker is replaced.
	WorkflowID        string           `json:"workflow_id"`
	TenantID          string           `json:"tenant_id"`
	SessionID         *string          `json:"session_id,omitempty"`
	AgentVersionID    string           `json:"agent_version_id"`
	Status            RunStatus        `json:"status"`
	TriggerType       string           `json:"trigger_type"`
	TraceParent       *string          `json:"traceparent,omitempty"`
	Input             json.RawMessage  `json:"input"`
	Output            json.RawMessage  `json:"output,omitempty"`
	BindingSnapshot   json.RawMessage  `json:"binding_snapshot"`
	ModelResolution   *ModelResolution `json:"model_resolution,omitempty"`
	CurrentTurn       int              `json:"current_turn"`
	CurrentStep       int              `json:"current_step"`
	Attempt           int              `json:"attempt"`
	LeaseOwner        *string          `json:"lease_owner,omitempty"`
	LeaseToken        int64            `json:"lease_token"`
	LeaseExpiresAt    *time.Time       `json:"lease_expires_at,omitempty"`
	NextWakeupAt      *time.Time       `json:"next_wakeup_at,omitempty"`
	CancelRequestedAt *time.Time       `json:"cancel_requested_at,omitempty"`
	StartedAt         *time.Time       `json:"started_at,omitempty"`
	FinishedAt        *time.Time       `json:"finished_at,omitempty"`
	ErrorCode         *string          `json:"error_code,omitempty"`
	ErrorMessage      *string          `json:"error_message,omitempty"`
	CreatedBy         *string          `json:"created_by,omitempty"`
	CreatedAt         time.Time        `json:"created_at"`
	UpdatedAt         time.Time        `json:"updated_at"`
	ParentRunID       *string          `json:"parent_run_id,omitempty"`
	RootRunID         *string          `json:"root_run_id,omitempty"`
	DelegationID      *string          `json:"delegation_id,omitempty"`
	DelegationDepth   int              `json:"delegation_depth"`
}

// ModelResolution is the immutable, Run-scoped result of model discovery.
// ServiceConfigHash prevents a resumed Run from silently switching endpoints.
type ModelResolution struct {
	SelectionPolicy     string    `json:"selection_policy"`
	Provider            string    `json:"provider"`
	ServiceRef          string    `json:"service_ref"`
	ModelID             string    `json:"model_id"`
	ModelVersion        string    `json:"model_version,omitempty"`
	ArtifactDigest      string    `json:"artifact_digest,omitempty"`
	ContextWindowTokens int       `json:"context_window_tokens,omitempty"`
	ServiceConfigHash   string    `json:"service_config_hash"`
	DiscoveredAt        time.Time `json:"discovered_at"`
}

// WorkerCapability is the last startup handshake observed by the control
// plane. It is operational metadata used to detect API/Worker image drift.
type WorkerCapability struct {
	WorkerID            string    `json:"worker_id"`
	RuntimeVersion      string    `json:"runtime_version"`
	ToolContractVersion string    `json:"tool_contract_version"`
	ProtocolVersion     string    `json:"protocol_version"`
	Capabilities        []string  `json:"capabilities"`
	CapabilityHash      string    `json:"capability_hash"`
	LastSeenAt          time.Time `json:"last_seen_at"`
}

// CreateRun contains immutable Run inputs resolved by agent-api.
type CreateRun struct {
	TenantID  string
	SessionID *string
	// WorkflowID is optional for backwards-compatible callers. When omitted,
	// the repository attaches the Run to the active resumable Workflow in the
	// Session or creates a new Workflow.
	WorkflowID *string
	// NewWorkflow forces a new long-lived task even when SessionID already has
	// an active Workflow. It is the explicit "新建任务" routing intent.
	NewWorkflow bool
	// RoutingIntent is an auditable user/UI intent: new_workflow, resume,
	// new_turn, or auto. It does not grant authority to bypass binding checks.
	RoutingIntent  string
	AgentVersionID string
	TriggerType    string
	// AllowDraft is set only by the authenticated Studio test-run path. Normal
	// API callers must execute published, immutable Agent versions.
	AllowDraft      bool
	Input           json.RawMessage
	CreatedBy       *string
	TraceParent     *string
	ParentRunID     *string
	RootRunID       *string
	DelegationID    *string
	DelegationDepth int
}

// Lease identifies a fenced Worker ownership generation.
type Lease struct {
	RunID  string
	Owner  string
	Token  int64
	Expiry time.Time
}
