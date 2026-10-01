package agent

import (
	"encoding/json"
	"time"
)

// Session groups multiple Runs into one tenant/user conversation.
type Session struct {
	ID             string          `json:"id"`
	TenantID       string          `json:"tenant_id"`
	AgentID        string          `json:"agent_id"`
	UserID         *string         `json:"user_id,omitempty"`
	Status         string          `json:"status"`
	Metadata       json.RawMessage `json:"metadata"`
	CreatedAt      time.Time       `json:"created_at"`
	UpdatedAt      time.Time       `json:"updated_at"`
	RunCount       int64           `json:"run_count,omitempty"`
	MessageCount   int64           `json:"message_count,omitempty"`
	ActiveRunCount int64           `json:"active_run_count,omitempty"`
	LastActivityAt *time.Time      `json:"last_activity_at,omitempty"`
}

// CreateSession contains immutable ownership plus optional metadata.
type CreateSession struct {
	TenantID string
	AgentID  string
	UserID   *string
	Metadata json.RawMessage
}
