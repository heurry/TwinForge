// Package verification defines the durable verification protocol independently
// from a planner's JSON representation and from any concrete Tool provider.
package verification

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

var ErrNoCompiler = errors.New("no verification compiler accepts the intent")

type Intent struct {
	ID                string          `json:"id,omitempty"`
	TenantID          string          `json:"tenant_id,omitempty"`
	RunID             string          `json:"run_id,omitempty"`
	PlanRevision      int             `json:"plan_revision"`
	PlanStepKey       string          `json:"plan_step_key"`
	CriterionKey      string          `json:"criterion_key"`
	Description       string          `json:"description"`
	Kind              string          `json:"kind,omitempty"`
	Enforcement       string          `json:"enforcement"`
	Origin            string          `json:"origin"`
	Parameters        json.RawMessage `json:"parameters"`
	Status            string          `json:"status"`
	DiagnosticCode    string          `json:"diagnostic_code,omitempty"`
	DiagnosticMessage string          `json:"diagnostic_message,omitempty"`
	CreatedAt         time.Time       `json:"created_at,omitempty"`
	UpdatedAt         time.Time       `json:"updated_at,omitempty"`
}

type ExecutableSpec struct {
	ID              string          `json:"id,omitempty"`
	IntentID        string          `json:"intent_id,omitempty"`
	ProviderKey     string          `json:"provider_key"`
	ProviderVersion string          `json:"provider_version"`
	Subject         json.RawMessage `json:"subject"`
	Execution       json.RawMessage `json:"execution"`
	Assertions      json.RawMessage `json:"assertions"`
	Digest          string          `json:"spec_digest"`
	Status          string          `json:"status"`
	CreatedAt       time.Time       `json:"created_at,omitempty"`
}

type Attempt struct {
	ID              string          `json:"id"`
	SpecID          string          `json:"spec_id"`
	ToolExecutionID string          `json:"tool_execution_id,omitempty"`
	Status          string          `json:"status"`
	ReasonCode      string          `json:"reason_code,omitempty"`
	Diagnostic      string          `json:"diagnostic,omitempty"`
	Result          json.RawMessage `json:"result"`
	StartedAt       time.Time       `json:"started_at"`
	FinishedAt      *time.Time      `json:"finished_at,omitempty"`
}

type Evidence struct {
	ID                string          `json:"id"`
	IntentID          string          `json:"intent_id"`
	SpecID            string          `json:"spec_id"`
	AttemptID         string          `json:"attempt_id"`
	ToolExecutionID   string          `json:"tool_execution_id,omitempty"`
	Verdict           string          `json:"verdict"`
	Payload           json.RawMessage `json:"evidence"`
	ResultDigest      string          `json:"result_digest,omitempty"`
	WorkspaceRevision string          `json:"workspace_revision,omitempty"`
	CreatedAt         time.Time       `json:"created_at"`
}

type Record struct {
	Intent   Intent          `json:"intent"`
	Spec     *ExecutableSpec `json:"spec,omitempty"`
	Attempts []Attempt       `json:"attempts"`
	Evidence []Evidence      `json:"evidence"`
}

// Compiler translates a declarative intent into an immutable executable
// contract. Match must be side-effect free; Compile must not execute a Tool.
type Compiler interface {
	Key() string
	Match(Intent) bool
	Compile(context.Context, Intent) (ExecutableSpec, error)
}

type Registry struct {
	mu        sync.RWMutex
	compilers map[string]Compiler
}

func NewRegistry() *Registry { return &Registry{compilers: make(map[string]Compiler)} }

func (r *Registry) Register(compiler Compiler) error {
	if compiler == nil || strings.TrimSpace(compiler.Key()) == "" {
		return errors.New("verification compiler and key are required")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.compilers[compiler.Key()]; exists {
		return fmt.Errorf("verification compiler %q is already registered", compiler.Key())
	}
	r.compilers[compiler.Key()] = compiler
	return nil
}

func (r *Registry) Compile(ctx context.Context, intent Intent) (ExecutableSpec, error) {
	r.mu.RLock()
	keys := make([]string, 0, len(r.compilers))
	for key := range r.compilers {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	compilers := make([]Compiler, 0, len(keys))
	for _, key := range keys {
		compilers = append(compilers, r.compilers[key])
	}
	r.mu.RUnlock()
	for _, compiler := range compilers {
		if !compiler.Match(intent) {
			continue
		}
		spec, err := compiler.Compile(ctx, intent)
		if err != nil {
			return ExecutableSpec{}, err
		}
		if spec.ProviderKey == "" {
			spec.ProviderKey = compiler.Key()
		}
		if spec.ProviderVersion == "" {
			spec.ProviderVersion = "v1"
		}
		if spec.Status == "" {
			spec.Status = "compiled"
		}
		spec.Digest, err = Digest(spec)
		return spec, err
	}
	return ExecutableSpec{}, ErrNoCompiler
}

func Digest(spec ExecutableSpec) (string, error) {
	value := struct {
		ProviderKey     string          `json:"provider_key"`
		ProviderVersion string          `json:"provider_version"`
		Subject         json.RawMessage `json:"subject"`
		Execution       json.RawMessage `json:"execution"`
		Assertions      json.RawMessage `json:"assertions"`
	}{spec.ProviderKey, spec.ProviderVersion, spec.Subject, spec.Execution, spec.Assertions}
	raw, err := json.Marshal(value)
	if err != nil {
		return "", fmt.Errorf("encode verification spec: %w", err)
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:]), nil
}

var DefaultRegistry = NewRegistry()
