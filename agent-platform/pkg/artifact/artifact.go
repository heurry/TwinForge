// Package artifact defines bounded, tenant-owned Run outputs.
package artifact

import (
	"encoding/json"
	"time"
)

type Artifact struct {
	ID             string          `json:"id"`
	TenantID       string          `json:"tenant_id"`
	RunID          string          `json:"run_id"`
	WorkflowID     string          `json:"workflow_id"`
	CallID         string          `json:"call_id,omitempty"`
	Kind           string          `json:"kind"`
	Name           string          `json:"name"`
	MediaType      string          `json:"media_type"`
	ContentHash    string          `json:"content_hash"`
	SizeBytes      int64           `json:"size_bytes"`
	StorageBackend string          `json:"storage_backend"`
	StorageStatus  string          `json:"storage_status"`
	Metadata       json.RawMessage `json:"metadata"`
	CreatedAt      time.Time       `json:"created_at"`
}

// RunManifest is the canonical, replayable delivery projection for one Run.
// Artifact rows remain append-only history; CanonicalArtifacts contains only
// the latest artifact per logical name so a UI or completion guard never has
// to guess which store.py/store_v2.py variant is authoritative.
type RunManifest struct {
	RunID               string             `json:"run_id"`
	WorkflowID          string             `json:"workflow_id"`
	TenantID            string             `json:"tenant_id"`
	Status              string             `json:"status"`
	CanonicalArtifacts  []ManifestArtifact `json:"canonical_artifacts"`
	RequiredOutputs     []string           `json:"required_outputs,omitempty"`
	VerificationSummary json.RawMessage    `json:"verification_summary,omitempty"`
	ChildRuns           []string           `json:"child_runs,omitempty"`
	FinalArtifactIDs    []string           `json:"final_artifact_ids,omitempty"`
	FinalOutputHash     string             `json:"final_output_hash,omitempty"`
	FinalOutputPresent  bool               `json:"final_output_present"`
	UpdatedAt           time.Time          `json:"updated_at"`
}

type ManifestArtifact struct {
	ID          string          `json:"artifact_id"`
	Name        string          `json:"name"`
	Kind        string          `json:"kind"`
	MediaType   string          `json:"media_type"`
	ContentHash string          `json:"content_hash"`
	SizeBytes   int64           `json:"size_bytes"`
	Metadata    json.RawMessage `json:"metadata,omitempty"`
	Canonical   bool            `json:"canonical"`
	Phase       string          `json:"phase,omitempty"`
	Final       bool            `json:"final"`
	CreatedAt   time.Time       `json:"created_at"`
}

type PromotionRequest struct {
	TargetPath           string `json:"target_path"`
	ExpectedTargetSHA256 string `json:"expected_target_sha256,omitempty"`
}

type Promotion struct {
	ID                   string     `json:"id"`
	TenantID             string     `json:"tenant_id"`
	RunID                string     `json:"run_id"`
	ArtifactID           string     `json:"artifact_id"`
	TargetPath           string     `json:"target_path"`
	SourceHash           string     `json:"source_hash"`
	ExpectedTargetSHA256 string     `json:"expected_target_sha256,omitempty"`
	PreviousTargetSHA256 string     `json:"previous_target_sha256,omitempty"`
	ResultTargetSHA256   string     `json:"result_target_sha256,omitempty"`
	Status               string     `json:"status"`
	RequestedBy          string     `json:"requested_by"`
	Error                string     `json:"error,omitempty"`
	CreatedAt            time.Time  `json:"created_at"`
	FinishedAt           *time.Time `json:"finished_at,omitempty"`
}

type PromotionResult struct {
	TargetPath           string `json:"target_path"`
	PreviousTargetSHA256 string `json:"previous_target_sha256,omitempty"`
	ResultTargetSHA256   string `json:"result_target_sha256"`
	Created              bool   `json:"created"`
}
