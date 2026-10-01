package agent

import (
	"encoding/json"
	"time"
)

// Definition is the stable identity of an Agent across immutable versions.
type Definition struct {
	ID              string    `json:"id"`
	TenantID        string    `json:"tenant_id"`
	Key             string    `json:"key"`
	Name            string    `json:"name"`
	Description     *string   `json:"description,omitempty"`
	Owner           *string   `json:"owner,omitempty"`
	Status          string    `json:"status"`
	ActiveVersionID *string   `json:"active_version_id,omitempty"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
}

// Version stores one immutable AgentSpec revision after publication.
type Version struct {
	ID          string          `json:"id"`
	AgentID     string          `json:"agent_id"`
	Version     int             `json:"version"`
	Spec        json.RawMessage `json:"spec"`
	SpecHash    string          `json:"spec_hash"`
	Status      string          `json:"status"`
	CreatedBy   *string         `json:"created_by,omitempty"`
	CreatedAt   time.Time       `json:"created_at"`
	PublishedAt *time.Time      `json:"published_at,omitempty"`
}

// ExecutableVersion is a published Agent version presented to Run creators.
type ExecutableVersion struct {
	ID          string          `json:"id"`
	AgentID     string          `json:"agent_id"`
	AgentKey    string          `json:"agent_key"`
	AgentName   string          `json:"agent_name"`
	Version     int             `json:"version"`
	Spec        json.RawMessage `json:"spec"`
	PublishedAt *time.Time      `json:"published_at,omitempty"`
}

// CreateDefinition contains tenant-owned catalog input.
type CreateDefinition struct {
	TenantID    string
	Key         string
	Name        string
	Description *string
	Owner       *string
}

// CreateVersion contains a complete draft specification.
type CreateVersion struct {
	TenantID  string
	AgentID   string
	Spec      Spec
	CreatedBy *string
}
