package agent

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"
)

var memorySecretPattern = regexp.MustCompile(`(?i)(api[_-]?key|secret|password|token|private[_-]?key)\s*[:=]\s*[^\s]+`)

const (
	MemoryScopeTenant  = "tenant"
	MemoryScopeAgent   = "agent"
	MemoryScopeUser    = "user"
	MemoryScopeSession = "session"
)

// Memory source layers describe provenance and authority. They are deliberately
// separate from MemoryScope, which describes ownership/visibility.
const (
	MemoryLayerManaged = "managed"
	MemoryLayerUser    = "user"
	MemoryLayerProject = "project"
	MemoryLayerLocal   = "local"
	MemoryLayerAuto    = "auto"
	MemoryLayerTeam    = "team"
)

// Memory semantic types describe what a memory means, not where it came from.
const (
	MemoryTypeUser      = "user"
	MemoryTypeFeedback  = "feedback"
	MemoryTypeProject   = "project"
	MemoryTypeReference = "reference"
)

const (
	MemoryStatusActive     = "active"
	MemoryStatusReview     = "review"
	MemoryStatusSuperseded = "superseded"
	MemoryStatusExpired    = "expired"
	MemoryStatusDeleted    = "deleted"
)

const (
	MemoryFreshnessStable   = "stable"
	MemoryFreshnessNormal   = "normal"
	MemoryFreshnessVolatile = "volatile"
)

func ValidMemoryLayer(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case MemoryLayerManaged, MemoryLayerUser, MemoryLayerProject, MemoryLayerLocal, MemoryLayerAuto, MemoryLayerTeam:
		return true
	default:
		return false
	}
}

func ValidMemoryType(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case MemoryTypeUser, MemoryTypeFeedback, MemoryTypeProject, MemoryTypeReference:
		return true
	default:
		return false
	}
}

func ValidMemoryStatus(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case MemoryStatusActive, MemoryStatusReview, MemoryStatusSuperseded, MemoryStatusExpired, MemoryStatusDeleted:
		return true
	default:
		return false
	}
}

func ValidMemoryFreshness(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case MemoryFreshnessStable, MemoryFreshnessNormal, MemoryFreshnessVolatile:
		return true
	default:
		return false
	}
}

// SemanticTypeForLegacyKind provides a reversible compatibility default for
// pre-taxonomy rows. It is intentionally conservative: callers can override
// the result with an explicit semantic type.
func SemanticTypeForLegacyKind(kind string) string {
	switch strings.ToLower(strings.TrimSpace(kind)) {
	case "preference":
		return MemoryTypeFeedback
	case "episodic":
		return MemoryTypeProject
	default:
		return MemoryTypeProject
	}
}

// Memory is an explicitly managed long-term fact. Scope ownership, expiry and
// deletion are enforced by persistence rather than left to prompt convention.
type Memory struct {
	ID               string          `json:"id"`
	TenantID         string          `json:"tenant_id"`
	Scope            string          `json:"scope"`
	AgentID          *string         `json:"agent_id,omitempty"`
	UserID           *string         `json:"user_id,omitempty"`
	SessionID        *string         `json:"session_id,omitempty"`
	Kind             string          `json:"kind"`
	Content          string          `json:"content"`
	Importance       float64         `json:"importance"`
	SourceRunID      *string         `json:"source_run_id,omitempty"`
	Metadata         json.RawMessage `json:"metadata"`
	ExpiresAt        *time.Time      `json:"expires_at,omitempty"`
	DeletedAt        *time.Time      `json:"deleted_at,omitempty"`
	CreatedBy        *string         `json:"created_by,omitempty"`
	CreatedAt        time.Time       `json:"created_at"`
	UpdatedAt        time.Time       `json:"updated_at"`
	RecallScore      float64         `json:"recall_score,omitempty"`
	EmbeddingStatus  string          `json:"embedding_status"`
	EmbeddingModel   *string         `json:"embedding_model,omitempty"`
	EmbeddingError   *string         `json:"embedding_error,omitempty"`
	EmbeddedAt       *time.Time      `json:"embedded_at,omitempty"`
	SourceID         *string         `json:"source_id,omitempty"`
	SourceLayer      string          `json:"source_layer"`
	SemanticType     string          `json:"semantic_type"`
	ProjectKey       *string         `json:"project_key,omitempty"`
	TeamID           *string         `json:"team_id,omitempty"`
	Title            string          `json:"title"`
	Description      string          `json:"description"`
	Body             string          `json:"body"`
	StructuredData   json.RawMessage `json:"structured_data"`
	Status           string          `json:"status"`
	Confidence       float64         `json:"confidence"`
	FreshnessClass   string          `json:"freshness_class"`
	ValidFrom        *time.Time      `json:"valid_from,omitempty"`
	ValidUntil       *time.Time      `json:"valid_until,omitempty"`
	LastVerifiedAt   *time.Time      `json:"last_verified_at,omitempty"`
	VerificationHint *string         `json:"verification_hint,omitempty"`
	CanonicalKey     string          `json:"canonical_key"`
	ContentHash      string          `json:"content_hash"`
	Pinned           bool            `json:"pinned"`
	SupersedesID     *string         `json:"supersedes_id,omitempty"`
}

type CreateMemory struct {
	TenantID         string
	Scope            string
	AgentID          *string
	UserID           *string
	SessionID        *string
	Kind             string
	Content          string
	Importance       float64
	SourceRunID      *string
	Metadata         json.RawMessage
	ExpiresAt        *time.Time
	CreatedBy        *string
	SourceID         *string
	SourceLayer      string
	SemanticType     string
	ProjectKey       *string
	TeamID           *string
	Title            string
	Description      string
	Body             string
	StructuredData   json.RawMessage
	Status           string
	Confidence       float64
	FreshnessClass   string
	ValidFrom        *time.Time
	ValidUntil       *time.Time
	LastVerifiedAt   *time.Time
	VerificationHint *string
	CanonicalKey     string
	Pinned           bool
	SupersedesID     *string
}

// MemoryPatch is the deliberately narrow human-edit surface. Source layer,
// scope, ownership and canonical identity are immutable here; changing them
// must go through the governance or extractor workflows.
type MemoryPatch struct {
	TenantID              string
	MemoryID              string
	Actor                 string
	IdempotencyKey        string
	ExpectedContentHash   string
	Title                 *string
	Description           *string
	Body                  *string
	StructuredData        *json.RawMessage
	Importance            *float64
	Confidence            *float64
	FreshnessClass        *string
	ExpiresAt             *time.Time
	ClearExpiresAt        bool
	ValidUntil            *time.Time
	ClearValidUntil       bool
	VerificationHint      *string
	ClearVerificationHint bool
	Pinned                *bool
}

type MemoryFilter struct {
	TenantID       string
	AgentID        string
	UserID         string
	SessionID      string
	Scope          string
	SourceLayer    string
	SemanticType   string
	ProjectKey     string
	TeamID         string
	ActorID        string
	Status         string
	IncludeExpired bool
	Limit          int
}

// MemoryManifestEntry is the bounded catalog used by retrieval routers. Excerpt
// is a small, read-only prefix used for model reranking; the complete Body is
// still loaded only after IDs have been selected and policy-validated.
type MemoryManifestEntry struct {
	ID             string     `json:"id"`
	RevisionKey    string     `json:"revision_key,omitempty"`
	Title          string     `json:"title"`
	Description    string     `json:"description"`
	Excerpt        string     `json:"excerpt,omitempty"`
	SourceLayer    string     `json:"source_layer"`
	SemanticType   string     `json:"semantic_type"`
	ProjectKey     *string    `json:"project_key,omitempty"`
	UpdatedAt      time.Time  `json:"updated_at"`
	LastVerifiedAt *time.Time `json:"last_verified_at,omitempty"`
	FreshnessClass string     `json:"freshness_class"`
	Confidence     float64    `json:"confidence"`
	Importance     float64    `json:"importance"`
	Pinned         bool       `json:"pinned"`
	RecallScore    float64    `json:"recall_score,omitempty"`
}

// MemoryRetrievalAudit is the durable decision projection for one memory
// retrieval. Bodies are intentionally absent; the event ledger may carry the
// model-visible projection while this record keeps the candidate/routing
// funnel and suppression rationale queryable for evaluation.
type MemoryRetrievalAudit struct {
	TurnID             string             `json:"turn_id"`
	QueryHash          string             `json:"query_hash"`
	CandidateIDs       []string           `json:"candidate_ids,omitempty"`
	RoutedIDs          []string           `json:"routed_ids,omitempty"`
	InjectedIDs        []string           `json:"injected_ids,omitempty"`
	SuppressedIDs      []string           `json:"suppressed_ids,omitempty"`
	Scores             map[string]float64 `json:"scores,omitempty"`
	SuppressionReasons map[string]string  `json:"suppression_reasons,omitempty"`
	RouterModel        string             `json:"router_model,omitempty"`
	ManifestTokens     int                `json:"manifest_tokens"`
	BodyTokens         int                `json:"body_tokens"`
	LatencyMS          int64              `json:"latency_ms"`
}

// MemoryRetrievalRecord is the durable, body-free retrieval funnel exposed to
// operators. It mirrors MemoryRetrievalAudit and adds the database identity
// and creation timestamp so Run Detail can reconstruct candidate → routed →
// injected → suppressed decisions without loading Memory bodies.
type MemoryRetrievalRecord struct {
	ID                 string             `json:"id"`
	TenantID           string             `json:"tenant_id"`
	RunID              string             `json:"run_id"`
	TurnID             string             `json:"turn_id"`
	QueryHash          string             `json:"query_hash"`
	CandidateIDs       []string           `json:"candidate_ids,omitempty"`
	RoutedIDs          []string           `json:"routed_ids,omitempty"`
	InjectedIDs        []string           `json:"injected_ids,omitempty"`
	SuppressedIDs      []string           `json:"suppressed_ids,omitempty"`
	Scores             map[string]float64 `json:"scores,omitempty"`
	SuppressionReasons map[string]string  `json:"suppression_reasons,omitempty"`
	RouterModel        *string            `json:"router_model,omitempty"`
	ManifestTokens     int                `json:"manifest_tokens"`
	BodyTokens         int                `json:"body_tokens"`
	LatencyMS          int64              `json:"latency_ms"`
	CreatedAt          time.Time          `json:"created_at"`
}

type MemoryFeedback struct {
	ID        string          `json:"id"`
	TenantID  string          `json:"tenant_id"`
	MemoryID  string          `json:"memory_id"`
	RunID     *string         `json:"run_id,omitempty"`
	TurnID    string          `json:"turn_id,omitempty"`
	Action    string          `json:"action"`
	Actor     *string         `json:"actor,omitempty"`
	Reason    *string         `json:"reason,omitempty"`
	Evidence  json.RawMessage `json:"evidence"`
	CreatedAt time.Time       `json:"created_at"`
}

type MemoryRevision struct {
	ID               string          `json:"id"`
	MemoryID         string          `json:"memory_id"`
	Revision         int64           `json:"revision"`
	Title            string          `json:"title"`
	Description      string          `json:"description"`
	Body             string          `json:"body"`
	StructuredData   json.RawMessage `json:"structured_data"`
	SourceMessageIDs []string        `json:"source_message_ids,omitempty"`
	SourceRunID      *string         `json:"source_run_id,omitempty"`
	SourceEventFrom  *int64          `json:"source_event_from,omitempty"`
	SourceEventTo    *int64          `json:"source_event_to,omitempty"`
	Reason           string          `json:"reason"`
	CreatedByType    string          `json:"created_by_type"`
	CreatedBy        *string         `json:"created_by,omitempty"`
	CreatedAt        time.Time       `json:"created_at"`
}

type MemoryLifecycleEvent struct {
	ID             string          `json:"id"`
	TenantID       string          `json:"tenant_id"`
	MemoryID       *string         `json:"memory_id,omitempty"`
	SourceID       *string         `json:"source_id,omitempty"`
	RunID          *string         `json:"run_id,omitempty"`
	EventType      string          `json:"event_type"`
	Actor          *string         `json:"actor,omitempty"`
	IdempotencyKey *string         `json:"idempotency_key,omitempty"`
	Payload        json.RawMessage `json:"payload"`
	CreatedAt      time.Time       `json:"created_at"`
}

// MemoryTimelineEntry is the normalized projection used by Run Detail. It
// keeps Run Events and Memory lifecycle rows distinguishable while exposing
// one chronological stream to operators.
type MemoryTimelineEntry struct {
	ID        string          `json:"id"`
	Source    string          `json:"source"` // run_event or memory_lifecycle
	Sequence  int64           `json:"sequence,omitempty"`
	RunID     string          `json:"run_id"`
	MemoryID  *string         `json:"memory_id,omitempty"`
	EventType string          `json:"event_type"`
	Actor     *string         `json:"actor,omitempty"`
	Payload   json.RawMessage `json:"payload"`
	CreatedAt time.Time       `json:"created_at"`
}

// MemoryWriteJob is the leased background extraction unit. The lease token is
// the fencing boundary for retries; a stale worker cannot complete a job it no
// longer owns.
type MemoryWriteJob struct {
	ID                      string          `json:"id"`
	TenantID                string          `json:"tenant_id"`
	RunID                   *string         `json:"run_id,omitempty"`
	SessionID               *string         `json:"session_id,omitempty"`
	TurnID                  string          `json:"turn_id"`
	Trigger                 string          `json:"trigger"`
	Status                  string          `json:"status"`
	Attempt                 int             `json:"attempt"`
	LeaseToken              *string         `json:"lease_token,omitempty"`
	AvailableAt             time.Time       `json:"available_at"`
	SourceEventFrom         *int64          `json:"source_event_from,omitempty"`
	SourceEventTo           *int64          `json:"source_event_to,omitempty"`
	LastMemoryWriteSequence *int64          `json:"last_memory_write_sequence,omitempty"`
	InputHash               string          `json:"input_hash"`
	ResultSummary           json.RawMessage `json:"result_summary"`
	LastError               *string         `json:"last_error,omitempty"`
	CreatedAt               time.Time       `json:"created_at"`
	StartedAt               *time.Time      `json:"started_at,omitempty"`
	FinishedAt              *time.Time      `json:"finished_at,omitempty"`
}

// StaticMemoryDocument is a revisioned instruction document discovered from
// the six source layers. Its content is captured once per Run; the runtime
// decides how much of it can enter the bounded static-context section.
type StaticMemoryDocument struct {
	ID          string    `json:"id"`
	SourceLayer string    `json:"source_layer"`
	Path        string    `json:"path"`
	ProjectKey  string    `json:"project_key,omitempty"`
	TeamID      string    `json:"team_id,omitempty"`
	ContentHash string    `json:"content_hash"`
	ObservedAt  time.Time `json:"observed_at"`
	Content     string    `json:"content"`
}

// MemoryExtractionCandidate is the only shape an asynchronous Extractor may
// propose. It is still untrusted until ValidateMemoryExtractionCandidates and
// the persistence writer apply scope, conflict and revision rules.
type MemoryExtractionCandidate struct {
	SemanticType     string          `json:"semantic_type"`
	Title            string          `json:"title"`
	Description      string          `json:"description"`
	StructuredData   json.RawMessage `json:"structured_payload"`
	Body             string          `json:"body"`
	SourceMessageIDs []string        `json:"source_message_ids"`
	Confidence       float64         `json:"confidence"`
	Importance       float64         `json:"importance"`
	FreshnessClass   string          `json:"freshness_class"`
	SuggestedAction  string          `json:"suggested_action"`
	MatchedMemoryID  *string         `json:"matched_memory_id,omitempty"`
}

func ValidateMemoryExtractionCandidates(candidates []MemoryExtractionCandidate) error {
	for index, candidate := range candidates {
		if !ValidMemoryType(candidate.SemanticType) {
			return fmt.Errorf("memory candidate %d has invalid semantic_type", index)
		}
		if strings.TrimSpace(candidate.Title) == "" || strings.TrimSpace(candidate.Description) == "" || strings.TrimSpace(candidate.Body) == "" {
			return fmt.Errorf("memory candidate %d requires title, description and body", index)
		}
		if len([]rune(candidate.Title)) > 256 || len([]rune(candidate.Description)) > 1200 || len([]rune(candidate.Body)) > 32000 {
			return fmt.Errorf("memory candidate %d exceeds text limits", index)
		}
		if candidate.Confidence < 0 || candidate.Confidence > 1 || candidate.Importance < 0 || candidate.Importance > 1 {
			return fmt.Errorf("memory candidate %d confidence/importance must be between 0 and 1", index)
		}
		if !ValidMemoryFreshness(candidate.FreshnessClass) {
			return fmt.Errorf("memory candidate %d has invalid freshness_class", index)
		}
		switch strings.ToLower(strings.TrimSpace(candidate.SuggestedAction)) {
		case "create", "update", "merge", "supersede", "ignore", "review":
		default:
			return fmt.Errorf("memory candidate %d has invalid suggested_action", index)
		}
		if memorySecretPattern.MatchString(candidate.Title) || memorySecretPattern.MatchString(candidate.Description) || memorySecretPattern.MatchString(candidate.Body) {
			return fmt.Errorf("memory candidate %d contains a probable secret", index)
		}
		if len(candidate.StructuredData) == 0 {
			candidate.StructuredData = json.RawMessage(`{}`)
		}
		var object map[string]any
		if json.Unmarshal(candidate.StructuredData, &object) != nil || object == nil {
			return fmt.Errorf("memory candidate %d structured_payload must be a JSON object", index)
		}
		if err := validateSemanticMemoryPayload(candidate.SemanticType, object, index); err != nil {
			return err
		}
	}
	return nil
}

// validateSemanticMemoryPayload keeps the four dynamic memory types distinct.
// The body remains human-readable, while structured_payload carries the
// fields that make feedback/project memories actionable instead of becoming
// context-free slogans.
func validateSemanticMemoryPayload(semanticType string, payload map[string]any, index int) error {
	require := func(fields ...string) error {
		for _, field := range fields {
			value, ok := payload[field].(string)
			if !ok || strings.TrimSpace(value) == "" {
				return fmt.Errorf("memory candidate %d %s structured_payload requires non-empty %q", index, semanticType, field)
			}
		}
		return nil
	}
	switch strings.ToLower(strings.TrimSpace(semanticType)) {
	case MemoryTypeFeedback:
		return require("rule", "why", "how_to_apply")
	case MemoryTypeProject:
		return require("fact", "why", "how_to_apply")
	case MemoryTypeReference:
		return require("system", "locator", "purpose")
	default:
		return nil
	}
}

// MemorySource is the provenance record for a static instruction document or
// an Auto/Team memory directory. Its revision is frozen into a Run when the
// source is resolved so historical executions remain replayable.
type MemorySource struct {
	ID          string    `json:"id"`
	TenantID    string    `json:"tenant_id"`
	SourceLayer string    `json:"source_layer"`
	AgentID     *string   `json:"agent_id,omitempty"`
	UserID      *string   `json:"user_id,omitempty"`
	ProjectKey  *string   `json:"project_key,omitempty"`
	TeamID      *string   `json:"team_id,omitempty"`
	URI         string    `json:"uri"`
	DisplayName string    `json:"display_name"`
	ContentHash string    `json:"content_hash"`
	Revision    int64     `json:"revision"`
	Authority   float64   `json:"authority"`
	WritableBy  string    `json:"writable_by"`
	Enabled     bool      `json:"enabled"`
	GitCommit   *string   `json:"git_commit,omitempty"`
	ObservedAt  time.Time `json:"observed_at"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// MemoryTeamMembership is the explicit ACL used for Team-layer memories.
type MemoryTeamMembership struct {
	TenantID  string    `json:"tenant_id"`
	TeamID    string    `json:"team_id"`
	UserID    string    `json:"user_id"`
	Role      string    `json:"role"`
	Enabled   bool      `json:"enabled"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}
