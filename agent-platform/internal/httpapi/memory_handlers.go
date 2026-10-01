package httpapi

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
)

type createMemoryRequest struct {
	Scope            string          `json:"scope"`
	AgentID          *string         `json:"agent_id"`
	UserID           *string         `json:"user_id"`
	SessionID        *string         `json:"session_id"`
	Kind             string          `json:"kind"`
	Content          string          `json:"content"`
	Importance       *float64        `json:"importance"`
	SourceRunID      *string         `json:"source_run_id"`
	Metadata         json.RawMessage `json:"metadata"`
	ExpiresAt        *time.Time      `json:"expires_at"`
	TTLSeconds       *int64          `json:"ttl_seconds"`
	SourceID         *string         `json:"source_id"`
	SourceLayer      string          `json:"source_layer"`
	SemanticType     string          `json:"semantic_type"`
	ProjectKey       *string         `json:"project_key"`
	TeamID           *string         `json:"team_id"`
	Title            string          `json:"title"`
	Description      string          `json:"description"`
	Body             string          `json:"body"`
	StructuredData   json.RawMessage `json:"structured_data"`
	Status           string          `json:"status"`
	Confidence       *float64        `json:"confidence"`
	FreshnessClass   string          `json:"freshness_class"`
	ValidFrom        *time.Time      `json:"valid_from"`
	ValidUntil       *time.Time      `json:"valid_until"`
	LastVerifiedAt   *time.Time      `json:"last_verified_at"`
	VerificationHint *string         `json:"verification_hint"`
	CanonicalKey     string          `json:"canonical_key"`
	Pinned           bool            `json:"pinned"`
	SupersedesID     *string         `json:"supersedes_id"`
}

type updateMemoryRequest struct {
	ExpectedContentHash   string           `json:"expected_content_hash"`
	Title                 *string          `json:"title"`
	Description           *string          `json:"description"`
	Body                  *string          `json:"body"`
	StructuredData        *json.RawMessage `json:"structured_data"`
	Importance            *float64         `json:"importance"`
	Confidence            *float64         `json:"confidence"`
	FreshnessClass        *string          `json:"freshness_class"`
	ExpiresAt             *time.Time       `json:"expires_at"`
	ClearExpiresAt        bool             `json:"clear_expires_at"`
	ValidUntil            *time.Time       `json:"valid_until"`
	ClearValidUntil       bool             `json:"clear_valid_until"`
	VerificationHint      *string          `json:"verification_hint"`
	ClearVerificationHint bool             `json:"clear_verification_hint"`
	Pinned                *bool            `json:"pinned"`
}

func (s *Server) createMemory(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	var request createMemoryRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	if request.ExpiresAt != nil && request.TTLSeconds != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "expires_at and ttl_seconds are mutually exclusive")
		return
	}
	for key, value := range map[string]*string{"agent_id": request.AgentID, "session_id": request.SessionID, "source_run_id": request.SourceRunID, "source_id": request.SourceID, "supersedes_id": request.SupersedesID} {
		if value != nil && !uuidPattern.MatchString(strings.TrimSpace(*value)) {
			writeError(w, http.StatusBadRequest, "invalid_request", key+" must be a UUID")
			return
		}
	}
	if request.TTLSeconds != nil {
		if *request.TTLSeconds <= 0 || *request.TTLSeconds > int64((5*365*24*time.Hour)/time.Second) {
			writeError(w, http.StatusBadRequest, "invalid_request", "ttl_seconds must be between 1 and five years")
			return
		}
		expires := time.Now().UTC().Add(time.Duration(*request.TTLSeconds) * time.Second)
		request.ExpiresAt = &expires
	}
	if request.SourceLayer == agent.MemoryLayerManaged || request.SourceLayer == agent.MemoryLayerTeam {
		writeError(w, http.StatusForbidden, "memory_layer_requires_governance", "managed and team memories require source management or promotion workflow")
		return
	}
	importance := 0.5
	if request.Importance != nil {
		importance = *request.Importance
	}
	var confidence float64
	if request.Confidence != nil {
		confidence = *request.Confidence
	}
	memory, err := s.memories.CreateMemory(r.Context(), agent.CreateMemory{
		TenantID: tenantID, Scope: request.Scope, AgentID: request.AgentID,
		UserID: request.UserID, SessionID: request.SessionID, Kind: request.Kind,
		Content: request.Content, Importance: importance, SourceRunID: request.SourceRunID,
		Metadata: request.Metadata, ExpiresAt: request.ExpiresAt, CreatedBy: optionalHeader(r, "X-Actor-ID"),
		SourceID: request.SourceID, SourceLayer: request.SourceLayer, SemanticType: request.SemanticType,
		ProjectKey: request.ProjectKey, TeamID: request.TeamID, Title: request.Title, Description: request.Description,
		Body: request.Body, StructuredData: request.StructuredData, Status: request.Status, Confidence: confidence,
		FreshnessClass: request.FreshnessClass, ValidFrom: request.ValidFrom, ValidUntil: request.ValidUntil,
		LastVerifiedAt: request.LastVerifiedAt, VerificationHint: request.VerificationHint, CanonicalKey: request.CanonicalKey,
		Pinned: request.Pinned, SupersedesID: request.SupersedesID,
	})
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, "invalid_memory", err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"data": memory})
}

func (s *Server) getMemory(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	memoryID := strings.TrimSpace(r.PathValue("memory_id"))
	if !uuidPattern.MatchString(memoryID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "memory_id must be a UUID")
		return
	}
	memory, err := s.memoryDetails.GetMemoryForTenant(r.Context(), tenantID, memoryID, strings.TrimSpace(r.Header.Get("X-Actor-ID")))
	if errors.Is(err, agent.ErrMemoryNotFound) {
		writeError(w, http.StatusNotFound, "memory_not_found", err.Error())
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "memory_get_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": memory})
}

func (s *Server) updateMemory(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	memoryID := strings.TrimSpace(r.PathValue("memory_id"))
	if !uuidPattern.MatchString(memoryID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "memory_id must be a UUID")
		return
	}
	actor := strings.TrimSpace(r.Header.Get("X-Actor-ID"))
	if actor == "" {
		writeError(w, http.StatusBadRequest, "missing_actor", "X-Actor-ID header is required")
		return
	}
	idempotencyKey := strings.TrimSpace(r.Header.Get("Idempotency-Key"))
	if idempotencyKey == "" {
		writeError(w, http.StatusBadRequest, "missing_idempotency_key", "Idempotency-Key header is required")
		return
	}
	var request updateMemoryRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	if request.ExpectedContentHash == "" {
		request.ExpectedContentHash = strings.Trim(strings.TrimSpace(r.Header.Get("If-Match")), "\"")
	}
	memory, err := s.memoryEdit.UpdateMemory(r.Context(), agent.MemoryPatch{
		TenantID: tenantID, MemoryID: memoryID, Actor: actor, IdempotencyKey: idempotencyKey,
		ExpectedContentHash: request.ExpectedContentHash, Title: request.Title, Description: request.Description,
		Body: request.Body, StructuredData: request.StructuredData, Importance: request.Importance,
		Confidence: request.Confidence, FreshnessClass: request.FreshnessClass, ExpiresAt: request.ExpiresAt,
		ClearExpiresAt: request.ClearExpiresAt, ValidUntil: request.ValidUntil, ClearValidUntil: request.ClearValidUntil,
		VerificationHint: request.VerificationHint, ClearVerificationHint: request.ClearVerificationHint, Pinned: request.Pinned,
	})
	if errors.Is(err, agent.ErrMemoryConflict) {
		writeError(w, http.StatusConflict, "memory_revision_conflict", err.Error())
		return
	}
	if errors.Is(err, agent.ErrMemoryNotFound) {
		writeError(w, http.StatusNotFound, "memory_not_found", err.Error())
		return
	}
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, "memory_update_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": memory})
}

func (s *Server) listMemories(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	for _, key := range []string{"agent_id", "session_id"} {
		value := strings.TrimSpace(r.URL.Query().Get(key))
		if value != "" && !uuidPattern.MatchString(value) {
			writeError(w, http.StatusBadRequest, "invalid_request", key+" must be a UUID")
			return
		}
	}
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	memories, err := s.memories.ListMemories(r.Context(), agent.MemoryFilter{
		TenantID: tenantID, AgentID: strings.TrimSpace(r.URL.Query().Get("agent_id")),
		UserID: strings.TrimSpace(r.URL.Query().Get("user_id")), SessionID: strings.TrimSpace(r.URL.Query().Get("session_id")),
		Scope: strings.TrimSpace(r.URL.Query().Get("scope")), SourceLayer: strings.TrimSpace(r.URL.Query().Get("source_layer")),
		SemanticType: strings.TrimSpace(r.URL.Query().Get("semantic_type")), ProjectKey: strings.TrimSpace(r.URL.Query().Get("project_key")),
		TeamID: strings.TrimSpace(r.URL.Query().Get("team_id")), Status: strings.TrimSpace(r.URL.Query().Get("status")),
		ActorID:        strings.TrimSpace(r.Header.Get("X-Actor-ID")),
		IncludeExpired: r.URL.Query().Get("include_expired") == "true", Limit: limit,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, "memory_list_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": memories})
}

func (s *Server) listMemoryManifest(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	for _, key := range []string{"agent_id", "session_id"} {
		value := strings.TrimSpace(r.URL.Query().Get(key))
		if value != "" && !uuidPattern.MatchString(value) {
			writeError(w, http.StatusBadRequest, "invalid_request", key+" must be a UUID")
			return
		}
	}
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	manifest, err := s.memoryManifest.ListMemoryManifest(r.Context(), agent.MemoryFilter{
		TenantID: tenantID, AgentID: strings.TrimSpace(r.URL.Query().Get("agent_id")),
		UserID: strings.TrimSpace(r.URL.Query().Get("user_id")), SessionID: strings.TrimSpace(r.URL.Query().Get("session_id")),
		Scope: strings.TrimSpace(r.URL.Query().Get("scope")), SourceLayer: strings.TrimSpace(r.URL.Query().Get("source_layer")),
		SemanticType: strings.TrimSpace(r.URL.Query().Get("semantic_type")), ProjectKey: strings.TrimSpace(r.URL.Query().Get("project_key")),
		TeamID: strings.TrimSpace(r.URL.Query().Get("team_id")), Status: strings.TrimSpace(r.URL.Query().Get("status")),
		ActorID:        strings.TrimSpace(r.Header.Get("X-Actor-ID")),
		IncludeExpired: false,
	}, limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "memory_manifest_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": manifest})
}

func (s *Server) deleteMemory(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	memoryID := r.PathValue("memory_id")
	if !uuidPattern.MatchString(memoryID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "memory_id must be a UUID")
		return
	}
	actor := strings.TrimSpace(r.Header.Get("X-Actor-ID"))
	if actor == "" {
		writeError(w, http.StatusBadRequest, "invalid_request", "X-Actor-ID is required")
		return
	}
	if err := s.memories.DeleteMemory(r.Context(), tenantID, memoryID, actor); err != nil {
		if errors.Is(err, agent.ErrMemoryNotFound) {
			writeError(w, http.StatusNotFound, "memory_not_found", err.Error())
			return
		}
		writeError(w, http.StatusInternalServerError, "memory_delete_failed", err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

type memoryTeamMembershipRequest struct {
	Role    string `json:"role"`
	Enabled *bool  `json:"enabled"`
}

func (s *Server) listMemoryTeamMemberships(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	if strings.TrimSpace(r.Header.Get("X-Actor-ID")) == "" {
		writeError(w, http.StatusBadRequest, "missing_actor", "X-Actor-ID header is required")
		return
	}
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	memberships, err := s.memoryTeams.ListMemoryTeamMemberships(r.Context(), tenantID, strings.TrimSpace(r.URL.Query().Get("team_id")), limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "memory_team_membership_list_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": memberships})
}

func (s *Server) upsertMemoryTeamMembership(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	if strings.TrimSpace(r.Header.Get("X-Actor-ID")) == "" {
		writeError(w, http.StatusBadRequest, "missing_actor", "X-Actor-ID header is required")
		return
	}
	var request memoryTeamMembershipRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	enabled := true
	if request.Enabled != nil {
		enabled = *request.Enabled
	}
	membership, err := s.memoryTeams.UpsertMemoryTeamMembership(r.Context(), tenantID, strings.TrimSpace(r.PathValue("team_id")), strings.TrimSpace(r.PathValue("user_id")), request.Role, enabled)
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, "invalid_memory_team_membership", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": membership})
}

func (s *Server) deleteMemoryTeamMembership(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	if strings.TrimSpace(r.Header.Get("X-Actor-ID")) == "" {
		writeError(w, http.StatusBadRequest, "missing_actor", "X-Actor-ID header is required")
		return
	}
	if err := s.memoryTeams.DeleteMemoryTeamMembership(r.Context(), tenantID, strings.TrimSpace(r.PathValue("team_id")), strings.TrimSpace(r.PathValue("user_id"))); err != nil {
		if errors.Is(err, agent.ErrMemoryNotFound) {
			writeError(w, http.StatusNotFound, "memory_team_membership_not_found", err.Error())
			return
		}
		writeError(w, http.StatusInternalServerError, "memory_team_membership_delete_failed", err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

type memoryFeedbackRequest struct {
	RunID    *string         `json:"run_id"`
	TurnID   string          `json:"turn_id"`
	Action   string          `json:"action"`
	Reason   *string         `json:"reason"`
	Evidence json.RawMessage `json:"evidence"`
}

func (s *Server) recordMemoryFeedback(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	memoryID := strings.TrimSpace(r.PathValue("memory_id"))
	if !uuidPattern.MatchString(memoryID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "memory_id must be a UUID")
		return
	}
	var request memoryFeedbackRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	if request.RunID != nil && strings.TrimSpace(*request.RunID) != "" && !uuidPattern.MatchString(strings.TrimSpace(*request.RunID)) {
		writeError(w, http.StatusBadRequest, "invalid_request", "run_id must be a UUID")
		return
	}
	actor := strings.TrimSpace(r.Header.Get("X-Actor-ID"))
	var actorPtr *string
	if actor != "" {
		actorPtr = &actor
	}
	feedback, err := s.memoryFeedback.RecordMemoryFeedback(r.Context(), agent.MemoryFeedback{
		TenantID: tenantID, MemoryID: memoryID, RunID: request.RunID, TurnID: strings.TrimSpace(request.TurnID),
		Action: request.Action, Actor: actorPtr, Reason: request.Reason, Evidence: request.Evidence,
	})
	if err != nil {
		if errors.Is(err, agent.ErrMemoryNotFound) {
			writeError(w, http.StatusNotFound, "memory_not_found", err.Error())
			return
		}
		writeError(w, http.StatusUnprocessableEntity, "invalid_memory_feedback", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": feedback})
}

func (s *Server) listMemoryFeedback(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	memoryID := strings.TrimSpace(r.PathValue("memory_id"))
	if !uuidPattern.MatchString(memoryID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "memory_id must be a UUID")
		return
	}
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	feedback, err := s.memoryFeedback.ListMemoryFeedback(r.Context(), tenantID, memoryID, limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "memory_feedback_list_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": feedback})
}

func (s *Server) listMemoryLifecycleEvents(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	memoryID := strings.TrimSpace(r.PathValue("memory_id"))
	if !uuidPattern.MatchString(memoryID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "memory_id must be a UUID")
		return
	}
	actor := strings.TrimSpace(r.Header.Get("X-Actor-ID"))
	if actor == "" {
		writeError(w, http.StatusBadRequest, "missing_actor", "X-Actor-ID header is required")
		return
	}
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	events, err := s.memoryLifecycle.ListMemoryLifecycleEvents(r.Context(), tenantID, memoryID, actor, limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "memory_lifecycle_list_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": events})
}

func (s *Server) listMemorySources(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	sources, err := s.memorySources.ListMemorySources(r.Context(), tenantID,
		strings.TrimSpace(r.URL.Query().Get("source_layer")), strings.TrimSpace(r.URL.Query().Get("project_key")),
		strings.TrimSpace(r.URL.Query().Get("team_id")), strings.TrimSpace(r.Header.Get("X-Actor-ID")), limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "memory_source_list_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": sources})
}

type syncMemorySourcesRequest struct {
	RunID     string                       `json:"run_id"`
	Documents []agent.StaticMemoryDocument `json:"documents"`
}

func (s *Server) syncMemorySources(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	if strings.TrimSpace(r.Header.Get("X-Actor-ID")) == "" {
		writeError(w, http.StatusBadRequest, "missing_actor", "X-Actor-ID header is required")
		return
	}
	var request syncMemorySourcesRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	request.RunID = strings.TrimSpace(request.RunID)
	if !uuidPattern.MatchString(request.RunID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "run_id must be a UUID")
		return
	}
	run, err := s.runs.GetRunForTenant(r.Context(), tenantID, request.RunID)
	if err != nil {
		if errors.Is(err, agent.ErrRunNotFound) {
			writeError(w, http.StatusNotFound, "run_not_found", err.Error())
			return
		}
		writeError(w, http.StatusInternalServerError, "run_lookup_failed", err.Error())
		return
	}
	if len(request.Documents) > 128 {
		writeError(w, http.StatusBadRequest, "invalid_request", "at most 128 static memory documents may be synced")
		return
	}
	for index := range request.Documents {
		if strings.TrimSpace(request.Documents[index].SourceLayer) == "" || strings.TrimSpace(request.Documents[index].Path) == "" {
			writeError(w, http.StatusBadRequest, "invalid_request", fmt.Sprintf("documents[%d] requires source_layer and path", index))
			return
		}
	}
	if err := s.memorySourceSync.SyncStaticMemorySources(r.Context(), run, request.Documents); err != nil {
		writeError(w, http.StatusUnprocessableEntity, "memory_source_sync_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{"data": map[string]any{"run_id": run.ID, "synced": len(request.Documents)}})
}

func (s *Server) listMemoryRetrievals(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	runID := strings.TrimSpace(r.PathValue("run_id"))
	if !uuidPattern.MatchString(runID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "run_id must be a UUID")
		return
	}
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	records, err := s.memoryRetrievals.ListMemoryRetrievals(r.Context(), tenantID, runID, limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "memory_retrieval_list_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": records})
}

func (s *Server) listMemoryTimeline(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	runID := strings.TrimSpace(r.PathValue("run_id"))
	if !uuidPattern.MatchString(runID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "run_id must be a UUID")
		return
	}
	actor := strings.TrimSpace(r.Header.Get("X-Actor-ID"))
	if actor == "" {
		writeError(w, http.StatusBadRequest, "missing_actor", "X-Actor-ID header is required")
		return
	}
	if _, err := s.runs.GetRunForTenant(r.Context(), tenantID, runID); err != nil {
		s.writeRunError(w, err)
		return
	}
	limit := parseBoundedInt(r.URL.Query().Get("limit"), 100, 500)
	// Read a bounded prefix from both ledgers, then merge. The response is
	// deliberately chronological; callers can keep the last entry as a UI
	// cursor until the two ledgers gain a shared sequence namespace.
	runEvents, err := s.runs.ListEventsForTenant(r.Context(), tenantID, runID, 0, limit*4)
	if err != nil {
		s.writeRunError(w, err)
		return
	}
	lifecycleEvents, err := s.memoryRunTimeline.ListMemoryLifecycleEventsForRun(r.Context(), tenantID, runID, actor, limit*4)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "memory_timeline_failed", err.Error())
		return
	}
	entries := make([]agent.MemoryTimelineEntry, 0, len(runEvents)+len(lifecycleEvents))
	for _, item := range runEvents {
		entries = append(entries, agent.MemoryTimelineEntry{
			ID: fmt.Sprintf("run-event:%d", item.Sequence), Source: "run_event", Sequence: item.Sequence,
			RunID: item.RunID, EventType: string(item.Type), Payload: item.Payload, CreatedAt: item.CreatedAt,
		})
	}
	for _, item := range lifecycleEvents {
		entries = append(entries, agent.MemoryTimelineEntry{
			ID: "memory-lifecycle:" + item.ID, Source: "memory_lifecycle", RunID: stringPointerValue(item.RunID),
			MemoryID: item.MemoryID, EventType: item.EventType, Actor: item.Actor, Payload: item.Payload, CreatedAt: item.CreatedAt,
		})
	}
	sort.SliceStable(entries, func(i, j int) bool {
		if entries[i].CreatedAt.Equal(entries[j].CreatedAt) {
			if entries[i].Source != entries[j].Source {
				return entries[i].Source < entries[j].Source
			}
			return entries[i].ID < entries[j].ID
		}
		return entries[i].CreatedAt.Before(entries[j].CreatedAt)
	})
	if len(entries) > limit {
		entries = entries[:limit]
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": entries})
}

func stringPointerValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func (s *Server) listMemoryRevisions(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	memoryID := strings.TrimSpace(r.PathValue("memory_id"))
	if !uuidPattern.MatchString(memoryID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "memory_id must be a UUID")
		return
	}
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	revisions, err := s.memoryGovernance.ListMemoryRevisions(r.Context(), tenantID, memoryID, limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "memory_revision_list_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": revisions})
}

type memoryGovernanceRequest struct {
	TeamID       string `json:"team_id"`
	SupersededBy string `json:"superseded_by"`
}

func (s *Server) memoryGovernanceAction(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	actor := strings.TrimSpace(r.Header.Get("X-Actor-ID"))
	if actor == "" {
		writeError(w, http.StatusBadRequest, "missing_actor", "X-Actor-ID header is required")
		return
	}
	value := strings.TrimSpace(r.PathValue("memory_action"))
	for _, action := range []string{"verify", "promote-to-team", "supersede"} {
		if id, matched := actionID(value, action); matched {
			if !uuidPattern.MatchString(id) {
				writeError(w, http.StatusBadRequest, "invalid_request", "memory action must contain a UUID")
				return
			}
			var request memoryGovernanceRequest
			if action != "verify" {
				if err := decodeJSON(w, r, &request); err != nil {
					return
				}
			}
			switch action {
			case "verify":
				memory, err := s.memoryGovernance.VerifyMemory(r.Context(), tenantID, id, actor)
				if err != nil {
					writeError(w, http.StatusNotFound, "memory_not_found", err.Error())
					return
				}
				writeJSON(w, http.StatusOK, map[string]any{"data": memory})
			case "promote-to-team":
				memory, err := s.memoryGovernance.PromoteMemoryToTeam(r.Context(), tenantID, id, strings.TrimSpace(request.TeamID), actor)
				if err != nil {
					writeError(w, http.StatusForbidden, "memory_team_promotion_denied", err.Error())
					return
				}
				writeJSON(w, http.StatusOK, map[string]any{"data": memory})
			case "supersede":
				if !uuidPattern.MatchString(strings.TrimSpace(request.SupersededBy)) {
					writeError(w, http.StatusBadRequest, "invalid_request", "superseded_by must be a UUID")
					return
				}
				if err := s.memoryGovernance.SupersedeMemory(r.Context(), tenantID, id, strings.TrimSpace(request.SupersededBy)); err != nil {
					writeError(w, http.StatusUnprocessableEntity, "memory_supersede_failed", err.Error())
					return
				}
				w.WriteHeader(http.StatusNoContent)
			}
			return
		}
	}
	writeError(w, http.StatusNotFound, "memory_action_not_found", "unsupported memory action")
}
