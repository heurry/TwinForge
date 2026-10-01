// Package agent defines the stable, provider-neutral Agent Platform domain.
package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/contract"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/harness"
)

// VersionRef identifies an immutable versioned resource.
type VersionRef struct {
	ID      string `json:"id"`
	Version string `json:"version"`
}

// HarnessSpec selects a loop implementation and its execution limit.
type HarnessSpec struct {
	Name     string `json:"name"`
	MaxTurns int    `json:"max_turns"`
	MaxSteps int    `json:"max_steps"`
}

const (
	PlanningPolicyAuto     = "auto"
	PlanningPolicyRequired = "required"
	PlanningPolicyDisabled = "disabled"
)

// PlanningPolicy controls whether a Run may answer directly or must establish
// a durable Plan before doing work. Empty is treated as auto so AgentVersions
// published before this field was introduced retain adaptive behavior.
type PlanningPolicy struct {
	Policy string `json:"policy,omitempty"`
}

func (p PlanningPolicy) EffectivePolicy() string {
	policy := strings.ToLower(strings.TrimSpace(p.Policy))
	if policy == "" {
		return PlanningPolicyAuto
	}
	return policy
}

// ModelBinding selects a model-serving capability rather than a fixed Pod IP.
type ModelBinding struct {
	Provider          string   `json:"provider"`
	ServiceRef        string   `json:"service_ref"`
	ServiceCandidates []string `json:"service_candidates,omitempty"`
	Capability        string   `json:"capability"`
	SelectionPolicy   string   `json:"selection_policy,omitempty"`
	ModelID           string   `json:"model_id,omitempty"`
	ModelCandidates   []string `json:"model_candidates,omitempty"`
	ModelVersion      string   `json:"model_version,omitempty"`
	ArtifactDigest    string   `json:"artifact_digest,omitempty"`
}

const (
	ModelSelectionPinned = "pinned"
	ModelSelectionAuto   = "auto"
)

// EffectiveSelectionPolicy preserves the deterministic behavior of Agent
// versions published before selection_policy existed.
func (b ModelBinding) EffectiveSelectionPolicy() string {
	policy := strings.ToLower(strings.TrimSpace(b.SelectionPolicy))
	if policy == "" {
		return ModelSelectionPinned
	}
	return policy
}

// ContextPolicy controls section budgets, proactive read-time collapse, and
// the model-facing compaction strategy.
type ContextPolicy struct {
	// MaxInputTokens is an optional Agent-level safety cap. Zero means inherit
	// the context window discovered from the selected model at Run start.
	MaxInputTokens           int     `json:"max_input_tokens,omitempty"`
	ReserveOutputTokens      int     `json:"reserve_output_tokens"`
	RecentTurnTokens         int     `json:"recent_turn_tokens"`
	MemoryTokens             int     `json:"memory_tokens"`
	KnowledgeTokens          int     `json:"knowledge_tokens"`
	ToolResultTokens         int     `json:"tool_result_tokens"`
	SummaryTokens            int     `json:"summary_tokens,omitempty"`
	StaticInstructionTokens  int     `json:"static_instruction_tokens,omitempty"`
	Compaction               string  `json:"compaction"`
	CollapseTriggerRatio     float64 `json:"collapse_trigger_ratio,omitempty"`
	CollapseTargetRatio      float64 `json:"collapse_target_ratio,omitempty"`
	CollapseCooldownMessages int     `json:"collapse_cooldown_messages,omitempty"`
	CollapseCooldownTokens   int     `json:"collapse_cooldown_tokens,omitempty"`
}

// MemoryPolicy controls which ownership scopes, source layers and semantic
// types may be read and written by an Agent. ReadScopes remains for backward
// compatibility; ReadLayers/ReadTypes are the taxonomy-aware filters.
// ProjectKey and TeamID are optional hard filters for project/team isolation.
type MemoryPolicy struct {
	Enabled     bool     `json:"enabled"`
	ReadScopes  []string `json:"read_scopes,omitempty"`
	WriteScope  string   `json:"write_scope,omitempty"`
	ReadLayers  []string `json:"read_layers,omitempty"`
	WriteLayer  string   `json:"write_layer,omitempty"`
	ReadTypes   []string `json:"read_types,omitempty"`
	ProjectKey  string   `json:"project_key,omitempty"`
	TeamID      string   `json:"team_id,omitempty"`
	AutoExtract bool     `json:"auto_extract,omitempty"`
	// ExtractOnCollapse is an opt-in safety valve for deployments that cannot
	// retain complete Run events until Turn completion. Normal deployments keep
	// it false: Collapse is a read-time projection and should not launch an LLM
	// extraction job for every compaction generation.
	ExtractOnCollapse      bool          `json:"extract_on_collapse,omitempty"`
	TeamMemoryEnabled      bool          `json:"team_memory_enabled,omitempty"`
	TeamPromotion          string        `json:"team_promotion,omitempty"`
	DefaultTTL             time.Duration `json:"default_ttl,omitempty"`
	MaxRecall              int           `json:"max_recall,omitempty"`
	CandidateLimit         int           `json:"candidate_limit,omitempty"`
	RouterEnabled          bool          `json:"router_enabled,omitempty"`
	RouterTopK             int           `json:"router_top_k,omitempty"`
	MinimumScore           float64       `json:"minimum_score,omitempty"`
	ManifestTokens         int           `json:"manifest_tokens,omitempty"`
	RepeatSuppressionTurns int           `json:"repeat_suppression_turns,omitempty"`
	VerifyStaleMemory      bool          `json:"verify_stale_memory,omitempty"`
}

// RuntimePolicy bounds one Run independently from model-provider settings.
type RuntimePolicy struct {
	RunTimeout    time.Duration `json:"run_timeout"`
	ModelTimeout  time.Duration `json:"model_timeout"`
	ToolTimeout   time.Duration `json:"tool_timeout"`
	MaxModelCalls int           `json:"max_model_calls"`
	MaxToolCalls  int           `json:"max_tool_calls"`
}

// ApprovalPolicy determines whether a risk class requires a human decision.
type ApprovalPolicy struct {
	RequireFor                []string      `json:"require_for,omitempty"`
	AutoApproveSandboxCommand bool          `json:"auto_approve_sandbox_command,omitempty"`
	ExpiresIn                 time.Duration `json:"expires_in,omitempty"`
}

type CollaborationTarget struct {
	AgentID        string   `json:"agent_id"`
	AgentVersionID string   `json:"agent_version_id"`
	Modes          []string `json:"modes"`
}

type CollaborationBudget struct {
	MaxModelCalls int   `json:"max_model_calls"`
	MaxToolCalls  int   `json:"max_tool_calls"`
	MaxTokens     int64 `json:"max_tokens"`
}
type CollaborationPolicy struct {
	AllowedTargets        []CollaborationTarget `json:"allowed_targets,omitempty"`
	MaxDepth              int                   `json:"max_depth,omitempty"`
	MaxFanOut             int                   `json:"max_fan_out,omitempty"`
	MaxChildRuns          int                   `json:"max_child_runs,omitempty"`
	ShareSessionMemory    bool                  `json:"share_session_memory,omitempty"`
	PropagateUserIdentity bool                  `json:"propagate_user_identity,omitempty"`
	ChildTimeout          time.Duration         `json:"child_timeout,omitempty"`
	Budget                CollaborationBudget   `json:"budget,omitempty"`
}

// Spec is an immutable AgentVersion payload after publication.
type Spec struct {
	Name          string              `json:"name"`
	Description   string              `json:"description,omitempty"`
	Identity      Identity            `json:"identity,omitempty"`
	Harness       HarnessSpec         `json:"harness"`
	Planning      PlanningPolicy      `json:"planning,omitempty"`
	Model         ModelBinding        `json:"model"`
	PromptRef     VersionRef          `json:"prompt_ref"`
	ToolSetRef    VersionRef          `json:"toolset_ref"`
	SkillSetRef   *VersionRef         `json:"skillset_ref,omitempty"`
	InputSchema   json.RawMessage     `json:"input_schema"`
	OutputSchema  json.RawMessage     `json:"output_schema"`
	Context       ContextPolicy       `json:"context"`
	Memory        MemoryPolicy        `json:"memory"`
	Runtime       RuntimePolicy       `json:"runtime"`
	Approval      ApprovalPolicy      `json:"approval"`
	Collaboration CollaborationPolicy `json:"collaboration,omitempty"`
	Metadata      map[string]string   `json:"metadata,omitempty"`
}

// Validate rejects incomplete or internally inconsistent Agent definitions.
func (s Spec) Validate() error {
	var errs []error
	if strings.TrimSpace(s.Name) == "" {
		errs = append(errs, errors.New("name is required"))
	}
	if err := s.Identity.validate(); err != nil {
		errs = append(errs, err)
	}
	if strings.TrimSpace(s.Harness.Name) == "" {
		errs = append(errs, errors.New("harness.name is required"))
	} else if err := harness.ValidateName(s.Harness.Name); err != nil {
		errs = append(errs, err)
	}
	if s.Harness.MaxTurns <= 0 {
		errs = append(errs, errors.New("harness.max_turns must be positive"))
	}
	switch s.Planning.EffectivePolicy() {
	case PlanningPolicyAuto, PlanningPolicyRequired, PlanningPolicyDisabled:
	default:
		errs = append(errs, fmt.Errorf("planning.policy %q is unsupported", s.Planning.Policy))
	}
	if s.Harness.MaxSteps <= 0 {
		errs = append(errs, errors.New("harness.max_steps must be positive"))
	}
	if strings.TrimSpace(s.Model.Provider) == "" {
		errs = append(errs, errors.New("model.provider is required"))
	}
	if strings.TrimSpace(s.Model.ServiceRef) == "" {
		errs = append(errs, errors.New("model.service_ref is required"))
	}
	switch s.Model.EffectiveSelectionPolicy() {
	case ModelSelectionPinned:
		if strings.TrimSpace(s.Model.ModelID) == "" {
			errs = append(errs, errors.New("model.model_id is required for pinned selection"))
		}
	case ModelSelectionAuto:
	default:
		errs = append(errs, fmt.Errorf("model.selection_policy %q is unsupported", s.Model.SelectionPolicy))
	}
	if s.PromptRef.ID == "" || s.PromptRef.Version == "" {
		errs = append(errs, errors.New("prompt_ref id and version are required"))
	}
	if s.ToolSetRef.ID == "" || s.ToolSetRef.Version == "" {
		errs = append(errs, errors.New("toolset_ref id and version are required"))
	}
	if s.SkillSetRef != nil && (s.SkillSetRef.ID == "" || s.SkillSetRef.Version == "") {
		errs = append(errs, errors.New("skillset_ref id and version are required"))
	}
	if !validObjectSchema(s.InputSchema) {
		errs = append(errs, errors.New("input_schema must be a JSON object"))
	} else if _, err := contract.Compile(s.InputSchema); err != nil {
		errs = append(errs, fmt.Errorf("input_schema: %w", err))
	}
	if !validObjectSchema(s.OutputSchema) {
		errs = append(errs, errors.New("output_schema must be a JSON object"))
	} else if _, err := contract.Compile(s.OutputSchema); err != nil {
		errs = append(errs, fmt.Errorf("output_schema: %w", err))
	}
	if s.Context.MaxInputTokens < 0 {
		errs = append(errs, errors.New("context.max_input_tokens must not be negative"))
	}
	if s.Context.ReserveOutputTokens < 0 || (s.Context.MaxInputTokens > 0 && s.Context.ReserveOutputTokens >= s.Context.MaxInputTokens) {
		errs = append(errs, errors.New("context.reserve_output_tokens must be non-negative and below an explicit max_input_tokens"))
	}
	if s.Context.SummaryTokens < 0 || s.Context.StaticInstructionTokens < 0 {
		errs = append(errs, errors.New("context.summary_tokens and context.static_instruction_tokens must be non-negative"))
	}
	if s.Context.CollapseTriggerRatio < 0 || s.Context.CollapseTriggerRatio >= 1 {
		errs = append(errs, errors.New("context.collapse_trigger_ratio must be in [0,1); zero uses the runtime default"))
	}
	if s.Context.CollapseTargetRatio < 0 || s.Context.CollapseTargetRatio >= 1 {
		errs = append(errs, errors.New("context.collapse_target_ratio must be in [0,1); zero uses the runtime default"))
	}
	if s.Context.CollapseCooldownMessages < 0 || s.Context.CollapseCooldownTokens < 0 {
		errs = append(errs, errors.New("context collapse cooldown values must be non-negative"))
	}
	if s.Memory.Enabled {
		if s.Context.MemoryTokens <= 0 {
			errs = append(errs, errors.New("context.memory_tokens must be positive when memory is enabled"))
		}
		if s.Memory.MaxRecall <= 0 || s.Memory.MaxRecall > 20 {
			errs = append(errs, errors.New("memory.max_recall must be between 1 and 20"))
		}
		if s.Memory.MinimumScore < 0 || s.Memory.MinimumScore > 1 {
			errs = append(errs, errors.New("memory.minimum_score must be between 0 and 1"))
		}
		if len(s.Memory.ReadScopes) == 0 {
			errs = append(errs, errors.New("memory.read_scopes is required when memory is enabled"))
		}
		for _, scope := range append(append([]string(nil), s.Memory.ReadScopes...), s.Memory.WriteScope) {
			if scope != "" && scope != MemoryScopeTenant && scope != MemoryScopeAgent && scope != MemoryScopeUser && scope != MemoryScopeSession {
				errs = append(errs, fmt.Errorf("memory scope %q is unsupported", scope))
			}
		}
		for _, layer := range append(append([]string(nil), s.Memory.ReadLayers...), s.Memory.WriteLayer) {
			if layer != "" && !ValidMemoryLayer(layer) {
				errs = append(errs, fmt.Errorf("memory source layer %q is unsupported", layer))
			}
		}
		for _, semanticType := range s.Memory.ReadTypes {
			if !ValidMemoryType(semanticType) {
				errs = append(errs, fmt.Errorf("memory semantic type %q is unsupported", semanticType))
			}
		}
		if len([]rune(s.Memory.ProjectKey)) > 512 || len([]rune(s.Memory.TeamID)) > 128 {
			errs = append(errs, errors.New("memory project_key/team_id exceeds configured length"))
		}
		if s.Memory.TeamPromotion != "" && s.Memory.TeamPromotion != "disabled" && s.Memory.TeamPromotion != "approval" && s.Memory.TeamPromotion != "required-policy" {
			errs = append(errs, errors.New("memory.team_promotion must be disabled, approval or required-policy"))
		}
		if s.Memory.CandidateLimit < 0 || s.Memory.CandidateLimit > 100 {
			errs = append(errs, errors.New("memory.candidate_limit must be between 0 and 100"))
		}
		if s.Memory.RouterTopK < 0 || s.Memory.RouterTopK > 5 {
			errs = append(errs, errors.New("memory.router_top_k must be between 0 and 5"))
		}
		if s.Memory.ManifestTokens < 0 || s.Memory.ManifestTokens > 4096 {
			errs = append(errs, errors.New("memory.manifest_tokens must be between 0 and 4096"))
		}
		if s.Memory.RepeatSuppressionTurns < 0 || s.Memory.RepeatSuppressionTurns > 100 {
			errs = append(errs, errors.New("memory.repeat_suppression_turns must be between 0 and 100"))
		}
		if s.Memory.DefaultTTL < 0 {
			errs = append(errs, errors.New("memory.default_ttl must not be negative"))
		}
	}
	if s.Runtime.MaxModelCalls <= 0 || s.Runtime.MaxToolCalls <= 0 {
		errs = append(errs, errors.New("runtime model and tool call limits must be positive"))
	}
	if len(s.Collaboration.AllowedTargets) > 0 {
		if s.Collaboration.MaxDepth <= 0 || s.Collaboration.MaxDepth > 8 {
			errs = append(errs, errors.New("collaboration.max_depth must be between 1 and 8"))
		}
		if s.Collaboration.MaxFanOut <= 0 || s.Collaboration.MaxFanOut > 16 {
			errs = append(errs, errors.New("collaboration.max_fan_out must be between 1 and 16"))
		}
		if s.Collaboration.MaxChildRuns <= 0 || s.Collaboration.MaxChildRuns > 64 {
			errs = append(errs, errors.New("collaboration.max_child_runs must be between 1 and 64"))
		}
		if s.Collaboration.Budget.MaxModelCalls < 0 || s.Collaboration.Budget.MaxToolCalls < 0 || s.Collaboration.Budget.MaxTokens < 0 {
			errs = append(errs, errors.New("collaboration budget values must not be negative"))
		}
		seen := map[string]struct{}{}
		for index, target := range s.Collaboration.AllowedTargets {
			if strings.TrimSpace(target.AgentID) == "" || strings.TrimSpace(target.AgentVersionID) == "" {
				errs = append(errs, fmt.Errorf("collaboration.allowed_targets[%d] requires agent_id and agent_version_id", index))
			}
			if _, ok := seen[target.AgentVersionID]; ok {
				errs = append(errs, fmt.Errorf("collaboration target %q is duplicated", target.AgentVersionID))
			}
			seen[target.AgentVersionID] = struct{}{}
			for _, mode := range target.Modes {
				if mode != "sync" && mode != "async" {
					errs = append(errs, fmt.Errorf("collaboration mode %q is unsupported", mode))
				}
			}
		}
	}
	if descriptor, ok := harness.Lookup(s.Harness.Name); ok && !descriptor.SupportsMultiTurn && s.Harness.MaxTurns != 1 {
		errs = append(errs, fmt.Errorf("harness %q requires max_turns=1", s.Harness.Name))
	}
	if err := errors.Join(errs...); err != nil {
		return fmt.Errorf("invalid agent spec: %w", err)
	}
	return nil
}

func validObjectSchema(raw json.RawMessage) bool {
	if len(raw) == 0 || !json.Valid(raw) {
		return false
	}
	var value map[string]any
	return json.Unmarshal(raw, &value) == nil
}
