// Package approval defines durable human decisions for risk-bearing tool calls.
package approval

import (
	"encoding/json"
	"errors"
	"time"
)

var (
	ErrRequired = errors.New("tool approval required")
	ErrRejected = errors.New("tool approval rejected")
)

type Approval struct {
	ID             string          `json:"id"`
	TenantID       string          `json:"tenant_id"`
	RunID          string          `json:"run_id"`
	CallID         string          `json:"call_id"`
	Turn           int             `json:"turn"`
	Step           int             `json:"step"`
	ToolVersionID  *string         `json:"tool_version_id,omitempty"`
	ToolName       string          `json:"tool_name"`
	Risk           string          `json:"risk"`
	RequestHash    string          `json:"request_hash"`
	Request        json.RawMessage `json:"request"`
	DiffArtifactID *string         `json:"diff_artifact_id,omitempty"`
	Status         string          `json:"status"`
	RequestedBy    *string         `json:"requested_by,omitempty"`
	DecidedBy      *string         `json:"decided_by,omitempty"`
	DecisionReason *string         `json:"decision_reason,omitempty"`
	ExpiresAt      *time.Time      `json:"expires_at,omitempty"`
	CreatedAt      time.Time       `json:"created_at"`
	DecidedAt      *time.Time      `json:"decided_at,omitempty"`
}

type Decision struct {
	Approved bool   `json:"approved"`
	Reason   string `json:"reason"`
	ActorID  string `json:"-"`
}
