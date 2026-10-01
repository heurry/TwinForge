// Package environment defines immutable execution templates and auditable,
// run-scoped dependency additions.
package environment

import (
	"encoding/json"
	"time"
)

type Template struct {
	ID           string          `json:"id"`
	Key          string          `json:"key"`
	Version      int             `json:"version"`
	Name         string          `json:"name"`
	Runtime      string          `json:"runtime"`
	ImageRef     string          `json:"image_ref"`
	SpecDigest   string          `json:"spec_digest"`
	Dependencies json.RawMessage `json:"dependencies"`
	Capabilities json.RawMessage `json:"capabilities"`
	Status       string          `json:"status"`
	CreatedAt    time.Time       `json:"created_at"`
}

type Install struct {
	ID                  string          `json:"id"`
	TenantID            string          `json:"tenant_id"`
	RunID               string          `json:"run_id"`
	CallID              string          `json:"call_id"`
	EnvironmentTemplate string          `json:"environment_template"`
	Ecosystem           string          `json:"ecosystem"`
	Packages            json.RawMessage `json:"packages"`
	Source              string          `json:"source"`
	Scope               string          `json:"scope"`
	Status              string          `json:"status"`
	Result              json.RawMessage `json:"result,omitempty"`
	Error               *string         `json:"error,omitempty"`
	CreatedAt           time.Time       `json:"created_at"`
	FinishedAt          *time.Time      `json:"finished_at,omitempty"`
}
