package postgres

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
)

var validMemoryScopes = map[string]bool{"tenant": true, "agent": true, "user": true, "session": true}
var validMemoryKinds = map[string]bool{"semantic": true, "episodic": true, "preference": true}
var validMemoryLifecycleEventTypes = map[string]bool{
	"MEMORY_EXTRACTION_REQUESTED": true, "MEMORY_EXTRACTION_STARTED": true,
	"MEMORY_CANDIDATE_PROPOSED": true, "MEMORY_CREATED": true,
	"MEMORY_UPDATED": true, "MEMORY_DELETED": true, "MEMORY_MERGED": true, "MEMORY_SUPERSEDED": true,
	"MEMORY_REVIEW_REQUIRED": true, "MEMORY_EXTRACTION_COMPLETED": true,
	"MEMORY_EXTRACTION_FAILED": true, "MEMORY_ROUTED": true,
	"MEMORY_INJECTED": true, "MEMORY_SUPPRESSED": true, "MEMORY_VERIFIED": true,
	"MEMORY_CONTRADICTED": true, "MEMORY_TEAM_PROMOTION_REQUESTED": true,
	"MEMORY_TEAM_PROMOTED": true, "MEMORY_SOURCE_REVISION_OBSERVED": true,
}

func (s *RunStore) CreateMemory(ctx context.Context, input agent.CreateMemory) (agent.Memory, error) {
	input.Scope = strings.ToLower(strings.TrimSpace(input.Scope))
	input.Kind = strings.ToLower(strings.TrimSpace(input.Kind))
	input.Content = strings.TrimSpace(input.Content)
	if !validMemoryScopes[input.Scope] {
		return agent.Memory{}, errors.New("memory scope must be tenant, agent, user or session")
	}
	if input.Kind == "" {
		input.Kind = "semantic"
	}
	if !validMemoryKinds[input.Kind] {
		return agent.Memory{}, errors.New("memory kind must be semantic, episodic or preference")
	}
	if input.Content == "" && strings.TrimSpace(input.Body) == "" {
		return agent.Memory{}, errors.New("memory content or body is required")
	}
	if len([]rune(input.Content)) > 8000 {
		return agent.Memory{}, errors.New("memory content must contain at most 8000 characters")
	}
	if input.Importance < 0 || input.Importance > 1 {
		return agent.Memory{}, errors.New("memory importance must be between 0 and 1")
	}
	if input.ExpiresAt != nil && !input.ExpiresAt.After(time.Now()) {
		return agent.Memory{}, errors.New("memory expires_at must be in the future")
	}
	metadata := input.Metadata
	if len(metadata) == 0 {
		metadata = json.RawMessage(`{}`)
	}
	if !validJSONObject(metadata) {
		return agent.Memory{}, errors.New("memory metadata must be a JSON object")
	}
	sourceLayer, semanticType, status, freshnessClass, title, description, body, structuredData, canonicalKey, err := normalizeMemoryTaxonomy(input, metadata)
	if err != nil {
		return agent.Memory{}, err
	}
	if sourceLayer != agent.MemoryLayerTeam {
		input.TeamID = nil
	}
	if input.ValidUntil != nil && !input.ValidUntil.After(time.Now()) {
		return agent.Memory{}, errors.New("memory valid_until must be in the future")
	}
	if input.ValidFrom != nil && input.ValidUntil != nil && !input.ValidUntil.After(*input.ValidFrom) {
		return agent.Memory{}, errors.New("memory valid_until must be after valid_from")
	}
	legacyContent := input.Content
	if legacyContent == "" {
		legacyContent = truncateRunes(body, 8000)
	}
	embeddingStatus := "pending"
	var embeddingValue, embeddingModel, embeddingError any
	var embeddedAt any
	retrievalText := strings.TrimSpace(strings.Join([]string{title, description, string(structuredData)}, "\n"))
	if retrievalText == "" {
		retrievalText = body
	}
	if s.embeddings != nil && s.embeddings.Enabled() {
		result, embedErr := s.embeddings.Embed(ctx, []string{retrievalText}, false)
		if embedErr == nil && len(result.Vectors) == 1 {
			embeddingStatus, embeddingValue, embeddingModel, embeddedAt = "ready", vectorLiteral(result.Vectors[0]), result.Model, time.Now().UTC()
		} else if embedErr != nil {
			embeddingError = truncateStorageError(embedErr.Error())
		}
	}
	row := s.pool.QueryRow(ctx, `
		INSERT INTO agent_platform.agent_memories
			(tenant_id,scope,agent_id,user_id,session_id,kind,content,importance,source_run_id,metadata,expires_at,created_by,
			 source_id,source_layer,semantic_type,project_key,team_id,title,description,body,structured_data,status,confidence,
			 freshness_class,valid_from,valid_until,last_verified_at,verification_hint,canonical_key,content_hash,pinned,supersedes_id,
			 embedding,embedding_model,embedding_status,embedding_error,embedded_at)
		SELECT $1::text,$2::text,
			CASE WHEN $2::text IN ('agent','user') THEN definition.id ELSE NULL END,
			CASE WHEN $2::text='user' THEN $4::text ELSE NULL END,
			CASE WHEN $2::text='session' THEN session.id ELSE NULL END,
			$6::text,$7::text,$8::double precision,$9::uuid,$10::jsonb,$11::timestamptz,$12::text,
			$13::uuid,$14::text,$15::text,$16::text,$17::text,$18::text,$19::text,$20::text,$21::jsonb,$22::text,$23::double precision,
			$24::text,$25::timestamptz,$26::timestamptz,$27::timestamptz,$28::text,$29::text,$30::text,$31::boolean,$32::uuid,
			$33::vector,$34::text,$35::text,$36::text,$37::timestamptz
		FROM (SELECT 1) seed
		LEFT JOIN agent_platform.agent_runs source_run
			ON source_run.id=$9::uuid AND source_run.tenant_id=$1::text
		LEFT JOIN agent_platform.agent_versions source_version
			ON source_version.id=source_run.agent_version_id
		LEFT JOIN agent_platform.agent_definitions definition
			ON definition.id=COALESCE($3::uuid,source_version.agent_id) AND definition.tenant_id=$1::text AND definition.status='active'
		LEFT JOIN agent_platform.agent_sessions session
			ON session.id=$5::uuid AND session.tenant_id=$1::text AND session.status='active'
		LEFT JOIN agent_platform.memory_sources source
			ON source.id=$13::uuid AND source.tenant_id=$1::text AND source.enabled
		WHERE (($2::text='tenant')
		   OR ($2::text='agent' AND definition.id IS NOT NULL)
		   OR ($2::text='user' AND definition.id IS NOT NULL AND NULLIF(btrim($4::text),'') IS NOT NULL)
		   OR ($2::text='session' AND session.id IS NOT NULL AND ($3::uuid IS NULL OR session.agent_id=$3::uuid)))
		  AND ($13::uuid IS NULL OR source.id IS NOT NULL)
		  AND ($9::uuid IS NULL OR source_run.id IS NOT NULL)
		RETURNING `+memoryColumns,
		input.TenantID, input.Scope, input.AgentID, input.UserID, input.SessionID,
		input.Kind, legacyContent, input.Importance, input.SourceRunID, metadata,
		input.ExpiresAt, input.CreatedBy, input.SourceID, sourceLayer, semanticType, input.ProjectKey, input.TeamID,
		title, description, body, structuredData, status, effectiveConfidence(input.Confidence, input.Importance), freshnessClass,
		input.ValidFrom, input.ValidUntil, input.LastVerifiedAt, input.VerificationHint, canonicalKey, memorySha256Hex(body), input.Pinned, input.SupersedesID,
		embeddingValue, embeddingModel, embeddingStatus, embeddingError, embeddedAt)
	memory, err := scanMemory(row)
	if errors.Is(err, pgx.ErrNoRows) {
		return agent.Memory{}, errors.New("memory owner does not exist in this tenant or does not match the selected scope")
	}
	if err != nil {
		return agent.Memory{}, fmt.Errorf("create memory: %w", err)
	}
	lifecyclePayload, _ := json.Marshal(map[string]any{
		"memory_id": memory.ID, "source_layer": memory.SourceLayer,
		"semantic_type": memory.SemanticType, "revision": 1,
	})
	if _, lifecycleErr := s.RecordMemoryLifecycleEvent(ctx, agent.MemoryLifecycleEvent{
		TenantID: memory.TenantID, MemoryID: &memory.ID, RunID: memory.SourceRunID,
		Actor: memory.CreatedBy, EventType: "MEMORY_CREATED",
		IdempotencyKey: optionalLifecycleKey("memory-created", memory.ID), Payload: lifecyclePayload,
	}); lifecycleErr != nil {
		return agent.Memory{}, lifecycleErr
	}
	if memory.Status == agent.MemoryStatusReview {
		reviewPayload, _ := json.Marshal(map[string]any{"memory_id": memory.ID, "status": memory.Status, "reason": "extractor_review"})
		if _, lifecycleErr := s.RecordMemoryLifecycleEvent(ctx, agent.MemoryLifecycleEvent{
			TenantID: memory.TenantID, MemoryID: &memory.ID, RunID: memory.SourceRunID,
			Actor: memory.CreatedBy, EventType: "MEMORY_REVIEW_REQUIRED",
			IdempotencyKey: optionalLifecycleKey("memory-review", memory.ID), Payload: reviewPayload,
		}); lifecycleErr != nil {
			return agent.Memory{}, lifecycleErr
		}
	}
	return memory, nil
}

// UpdateMemory applies the narrow human-edit surface. It deliberately excludes
// source layer, scope, ownership and canonical identity so a user cannot turn a
// private Memory edit into a Team/Managed instruction through a generic PATCH.
func (s *RunStore) UpdateMemory(ctx context.Context, patch agent.MemoryPatch) (agent.Memory, error) {
	patch.TenantID = strings.TrimSpace(patch.TenantID)
	patch.MemoryID = strings.TrimSpace(patch.MemoryID)
	patch.Actor = strings.TrimSpace(patch.Actor)
	patch.ExpectedContentHash = strings.TrimSpace(patch.ExpectedContentHash)
	if patch.TenantID == "" || patch.MemoryID == "" || patch.Actor == "" {
		return agent.Memory{}, errors.New("tenant_id, memory_id and actor are required")
	}
	if patch.Body != nil {
		body := strings.TrimSpace(*patch.Body)
		if body == "" || len([]rune(body)) > 32000 {
			return agent.Memory{}, errors.New("memory body must contain 1 to 32000 characters")
		}
		patch.Body = &body
	}
	if patch.Title != nil {
		value := truncateRunes(strings.TrimSpace(*patch.Title), 256)
		patch.Title = &value
	}
	if patch.Description != nil {
		value := truncateRunes(strings.TrimSpace(*patch.Description), 1200)
		patch.Description = &value
	}
	if patch.StructuredData != nil && !validJSONObject(*patch.StructuredData) {
		return agent.Memory{}, errors.New("memory structured_data must be a JSON object")
	}
	if patch.Importance != nil && (*patch.Importance < 0 || *patch.Importance > 1) {
		return agent.Memory{}, errors.New("memory importance must be between 0 and 1")
	}
	if patch.Confidence != nil && (*patch.Confidence < 0 || *patch.Confidence > 1) {
		return agent.Memory{}, errors.New("memory confidence must be between 0 and 1")
	}
	if patch.FreshnessClass != nil {
		value := strings.ToLower(strings.TrimSpace(*patch.FreshnessClass))
		if !agent.ValidMemoryFreshness(value) {
			return agent.Memory{}, errors.New("memory freshness_class is invalid")
		}
		patch.FreshnessClass = &value
	}
	if patch.ExpiresAt != nil && !patch.ExpiresAt.After(time.Now()) {
		return agent.Memory{}, errors.New("memory expires_at must be in the future")
	}
	if patch.ValidUntil != nil && !patch.ValidUntil.After(time.Now()) {
		return agent.Memory{}, errors.New("memory valid_until must be in the future")
	}
	if patch.Body == nil && patch.Title == nil && patch.Description == nil && patch.StructuredData == nil &&
		patch.Importance == nil && patch.Confidence == nil && patch.FreshnessClass == nil && patch.ExpiresAt == nil &&
		!patch.ClearExpiresAt && patch.ValidUntil == nil && !patch.ClearValidUntil && patch.VerificationHint == nil &&
		!patch.ClearVerificationHint && patch.Pinned == nil {
		return agent.Memory{}, errors.New("memory patch has no editable fields")
	}

	args := make([]any, 0, 16)
	arg := func(value any) string {
		args = append(args, value)
		return fmt.Sprintf("$%d", len(args))
	}
	sets := make([]string, 0, 12)
	contentChanged := patch.Body != nil || patch.Title != nil || patch.Description != nil || patch.StructuredData != nil
	if patch.Body != nil {
		bodyRef := arg(*patch.Body)
		sets = append(sets, "body="+bodyRef, "content=left("+bodyRef+",8000)", "content_hash="+arg(memorySha256Hex(*patch.Body)))
	}
	if patch.Title != nil {
		sets = append(sets, "title="+arg(*patch.Title))
	}
	if patch.Description != nil {
		sets = append(sets, "description="+arg(*patch.Description))
	}
	if patch.StructuredData != nil {
		sets = append(sets, "structured_data="+arg(*patch.StructuredData)+"::jsonb")
	}
	if patch.Importance != nil {
		sets = append(sets, "importance="+arg(*patch.Importance))
	}
	if patch.Confidence != nil {
		sets = append(sets, "confidence="+arg(*patch.Confidence))
	}
	if patch.FreshnessClass != nil {
		sets = append(sets, "freshness_class="+arg(*patch.FreshnessClass))
	}
	if patch.ExpiresAt != nil {
		sets = append(sets, "expires_at="+arg(*patch.ExpiresAt)+"::timestamptz")
	} else if patch.ClearExpiresAt {
		sets = append(sets, "expires_at=NULL")
	}
	if patch.ValidUntil != nil {
		sets = append(sets, "valid_until="+arg(*patch.ValidUntil)+"::timestamptz")
	} else if patch.ClearValidUntil {
		sets = append(sets, "valid_until=NULL")
	}
	if patch.VerificationHint != nil {
		sets = append(sets, "verification_hint="+arg(strings.TrimSpace(*patch.VerificationHint)))
	} else if patch.ClearVerificationHint {
		sets = append(sets, "verification_hint=NULL")
	}
	if patch.Pinned != nil {
		sets = append(sets, "pinned="+arg(*patch.Pinned))
	}
	if contentChanged {
		sets = append(sets, "last_verified_at=NULL", "embedding=NULL", "embedding_model=NULL", "embedding_status='pending'", "embedding_error=NULL", "embedded_at=NULL")
	}
	sets = append(sets, "updated_at=now()")
	idRef := arg(patch.MemoryID)
	tenantRef := arg(patch.TenantID)
	actorRef := arg(patch.Actor)
	where := []string{"id=" + idRef + "::uuid", "tenant_id=" + tenantRef, "deleted_at IS NULL", "status IN ('active','review')", "source_layer IN ('user','project','local')", "(created_by=" + actorRef + " OR (scope='user' AND user_id=" + actorRef + "))"}
	if patch.ExpectedContentHash != "" {
		where = append(where, "content_hash="+arg(patch.ExpectedContentHash))
	}
	row := s.pool.QueryRow(ctx, `UPDATE agent_platform.agent_memories SET `+strings.Join(sets, ",")+` WHERE `+strings.Join(where, " AND ")+` RETURNING `+memoryColumns, args...)
	memory, err := scanMemory(row)
	if errors.Is(err, pgx.ErrNoRows) {
		if patch.ExpectedContentHash != "" {
			return agent.Memory{}, agent.ErrMemoryConflict
		}
		return agent.Memory{}, agent.ErrMemoryNotFound
	}
	if err != nil {
		return agent.Memory{}, fmt.Errorf("update memory: %w", err)
	}
	if err := s.RecordManualMemoryRevision(ctx, memory, patch.Actor); err != nil {
		return agent.Memory{}, err
	}
	payload, _ := json.Marshal(map[string]any{"memory_id": memory.ID, "actor": patch.Actor, "content_hash": memory.ContentHash, "manual": true})
	key := patch.IdempotencyKey
	if strings.TrimSpace(key) == "" {
		key = memory.ID + ":" + memory.ContentHash
	}
	if _, eventErr := s.RecordMemoryLifecycleEvent(ctx, agent.MemoryLifecycleEvent{
		TenantID: memory.TenantID, MemoryID: &memory.ID, Actor: optionalStringPointer(patch.Actor), EventType: "MEMORY_UPDATED",
		IdempotencyKey: optionalLifecycleKey("manual-update", key), Payload: payload,
	}); eventErr != nil {
		return agent.Memory{}, eventErr
	}
	return memory, nil
}

func (s *RunStore) GetMemoryForTenant(ctx context.Context, tenantID, memoryID, actorID string) (agent.Memory, error) {
	row := s.pool.QueryRow(ctx, `
		SELECT `+memoryColumns+`
		FROM agent_platform.agent_memories memory
		WHERE memory.id=$1::uuid AND memory.tenant_id=$2 AND memory.deleted_at IS NULL
		  AND (memory.source_layer <> 'team' OR EXISTS (
			SELECT 1 FROM agent_platform.memory_team_memberships membership
			WHERE membership.tenant_id=memory.tenant_id AND membership.team_id=memory.team_id
			  AND membership.user_id=$3 AND membership.enabled
		  ))`, strings.TrimSpace(memoryID), strings.TrimSpace(tenantID), strings.TrimSpace(actorID))
	memory, err := scanMemory(row)
	if errors.Is(err, pgx.ErrNoRows) {
		return agent.Memory{}, agent.ErrMemoryNotFound
	}
	if err != nil {
		return agent.Memory{}, fmt.Errorf("get memory: %w", err)
	}
	return memory, nil
}

func normalizeMemoryTaxonomy(input agent.CreateMemory, metadata json.RawMessage) (sourceLayer, semanticType, status, freshnessClass, title, description, body string, structuredData json.RawMessage, canonicalKey string, err error) {
	sourceLayer = strings.ToLower(strings.TrimSpace(input.SourceLayer))
	if sourceLayer == "" {
		var metadataObject map[string]any
		if json.Unmarshal(metadata, &metadataObject) == nil {
			if value, ok := metadataObject["source_layer"].(string); ok {
				sourceLayer = strings.ToLower(strings.TrimSpace(value))
			}
		}
	}
	if sourceLayer == "" {
		sourceLayer = agent.MemoryLayerAuto
	}
	if !agent.ValidMemoryLayer(sourceLayer) {
		return "", "", "", "", "", "", "", nil, "", errors.New("memory source_layer must be managed, user, project, local, auto or team")
	}
	if sourceLayer == agent.MemoryLayerTeam && (input.TeamID == nil || strings.TrimSpace(*input.TeamID) == "") {
		return "", "", "", "", "", "", "", nil, "", errors.New("team memory requires team_id")
	}
	semanticType = strings.ToLower(strings.TrimSpace(input.SemanticType))
	if semanticType == "" {
		semanticType = agent.SemanticTypeForLegacyKind(input.Kind)
	}
	if !agent.ValidMemoryType(semanticType) {
		return "", "", "", "", "", "", "", nil, "", errors.New("memory semantic_type must be user, feedback, project or reference")
	}
	status = strings.ToLower(strings.TrimSpace(input.Status))
	if status == "" {
		status = agent.MemoryStatusActive
	}
	if !agent.ValidMemoryStatus(status) {
		return "", "", "", "", "", "", "", nil, "", errors.New("memory status is invalid")
	}
	freshnessClass = strings.ToLower(strings.TrimSpace(input.FreshnessClass))
	if freshnessClass == "" {
		freshnessClass = agent.MemoryFreshnessNormal
	}
	if !agent.ValidMemoryFreshness(freshnessClass) {
		return "", "", "", "", "", "", "", nil, "", errors.New("memory freshness_class must be stable, normal or volatile")
	}
	body = strings.TrimSpace(input.Body)
	if body == "" {
		body = input.Content
	}
	if body == "" || len([]rune(body)) > 32000 {
		return "", "", "", "", "", "", "", nil, "", errors.New("memory body must contain 1 to 32000 characters")
	}
	title = strings.TrimSpace(input.Title)
	if title == "" {
		title = strings.TrimSpace(strings.SplitN(body, "\n", 2)[0])
	}
	title = truncateRunes(title, 256)
	description = strings.TrimSpace(input.Description)
	if description == "" {
		description = truncateRunes(strings.ReplaceAll(body, "\n", " "), 1000)
	}
	description = truncateRunes(description, 1200)
	structuredData = input.StructuredData
	if len(structuredData) == 0 {
		structuredData = json.RawMessage(`{}`)
	}
	if !validJSONObject(structuredData) {
		return "", "", "", "", "", "", "", nil, "", errors.New("memory structured_data must be a JSON object")
	}
	canonicalKey = strings.TrimSpace(input.CanonicalKey)
	if canonicalKey == "" {
		canonicalKey = memorySha256Hex(strings.ToLower(strings.Join([]string{sourceLayer, semanticType, title, description}, "\n")))
	}
	canonicalKey = truncateRunes(canonicalKey, 512)
	return sourceLayer, semanticType, status, freshnessClass, title, description, body, structuredData, canonicalKey, nil
}

func effectiveConfidence(value, importance float64) float64 {
	if value <= 0 {
		value = importance
	}
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}

func truncateRunes(value string, limit int) string {
	runes := []rune(value)
	if limit <= 0 || len(runes) <= limit {
		return value
	}
	return string(runes[:limit])
}

func stringPointerValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func optionalLifecycleKey(prefix, value string) *string {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil
	}
	key := strings.TrimSpace(prefix) + ":" + value
	return &key
}

func memorySha256Hex(value string) string {
	digest := sha256.Sum256([]byte(value))
	return hex.EncodeToString(digest[:])
}

func (s *RunStore) ListMemories(ctx context.Context, filter agent.MemoryFilter) ([]agent.Memory, error) {
	if filter.Limit <= 0 || filter.Limit > 200 {
		filter.Limit = 100
	}
	if filter.SourceLayer != "" && !agent.ValidMemoryLayer(filter.SourceLayer) {
		return nil, errors.New("memory source_layer filter is invalid")
	}
	if filter.SemanticType != "" && !agent.ValidMemoryType(filter.SemanticType) {
		return nil, errors.New("memory semantic_type filter is invalid")
	}
	if filter.Status != "" && !agent.ValidMemoryStatus(filter.Status) {
		return nil, errors.New("memory status filter is invalid")
	}
	rows, err := s.pool.Query(ctx, `SELECT `+memoryColumns+`
		FROM agent_platform.agent_memories memory
		WHERE tenant_id=$1 AND deleted_at IS NULL
		  AND (($2='' AND $3='' AND $4='')
		    OR scope='tenant'
		    OR (scope='agent' AND agent_id=NULLIF($2,'')::uuid)
		    OR (scope='user' AND agent_id=NULLIF($2,'')::uuid AND user_id=$3)
		    OR (scope='session' AND session_id=NULLIF($4,'')::uuid))
		  AND ($5='' OR scope=$5)
		  AND ($6 OR expires_at IS NULL OR expires_at>now())
		  AND ($7='' OR source_layer=$7)
		  AND ($8='' OR semantic_type=$8)
		  AND ($9='' OR project_key=$9)
		  AND ($10='' OR team_id=$10)
		  AND ($11='' OR status=$11)
		  AND (source_layer <> 'team' OR EXISTS (
			SELECT 1 FROM agent_platform.memory_team_memberships membership
			WHERE membership.tenant_id=memory.tenant_id AND membership.team_id=memory.team_id
			  AND membership.user_id=$13 AND membership.enabled
		  ))
		ORDER BY pinned DESC, importance DESC, updated_at DESC LIMIT $12`, filter.TenantID, filter.AgentID,
		filter.UserID, filter.SessionID, filter.Scope, filter.IncludeExpired, filter.SourceLayer,
		filter.SemanticType, filter.ProjectKey, filter.TeamID, filter.Status, filter.Limit, strings.TrimSpace(filter.ActorID))
	if err != nil {
		return nil, fmt.Errorf("list memories: %w", err)
	}
	defer rows.Close()
	memories := make([]agent.Memory, 0)
	for rows.Next() {
		memory, scanErr := scanMemory(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan memory: %w", scanErr)
		}
		memories = append(memories, memory)
	}
	return memories, rows.Err()
}

// ListMemoryManifest returns the bounded catalog projection used by the
// retrieval router. It deliberately selects no legacy content or full body;
// callers must opt into LoadMemoriesForRun after selecting IDs.
func (s *RunStore) ListMemoryManifest(ctx context.Context, filter agent.MemoryFilter, limit int) ([]agent.MemoryManifestEntry, error) {
	if limit <= 0 || limit > 200 {
		limit = 50
	}
	filter.Limit = limit
	if strings.TrimSpace(filter.Status) == "" {
		filter.Status = agent.MemoryStatusActive
	}
	if filter.SourceLayer != "" && !agent.ValidMemoryLayer(filter.SourceLayer) {
		return nil, errors.New("memory source_layer filter is invalid")
	}
	if filter.SemanticType != "" && !agent.ValidMemoryType(filter.SemanticType) {
		return nil, errors.New("memory semantic_type filter is invalid")
	}
	if filter.Status != "" && !agent.ValidMemoryStatus(filter.Status) {
		return nil, errors.New("memory status filter is invalid")
	}
	rows, err := s.pool.Query(ctx, `SELECT `+memoryManifestColumns+`
		FROM agent_platform.agent_memories memory
		WHERE tenant_id=$1 AND deleted_at IS NULL
		  AND (($2='' AND $3='' AND $4='')
		    OR scope='tenant'
		    OR (scope='agent' AND agent_id=NULLIF($2,'')::uuid)
		    OR (scope='user' AND agent_id=NULLIF($2,'')::uuid AND user_id=$3)
		    OR (scope='session' AND session_id=NULLIF($4,'')::uuid))
		  AND ($5='' OR scope=$5)
		  AND ($6 OR expires_at IS NULL OR expires_at>now())
		  AND ($7='' OR source_layer=$7)
		  AND ($8='' OR semantic_type=$8)
		  AND ($9='' OR project_key=$9)
		  AND ($10='' OR team_id=$10)
		  AND ($11='' OR status=$11)
		  AND (source_layer <> 'team' OR EXISTS (
			SELECT 1 FROM agent_platform.memory_team_memberships membership
			WHERE membership.tenant_id=memory.tenant_id AND membership.team_id=memory.team_id
			  AND membership.user_id=$13 AND membership.enabled
		  ))
		ORDER BY updated_at DESC LIMIT $12`, filter.TenantID, filter.AgentID,
		filter.UserID, filter.SessionID, filter.Scope, filter.IncludeExpired, filter.SourceLayer,
		filter.SemanticType, filter.ProjectKey, filter.TeamID, filter.Status, filter.Limit, strings.TrimSpace(filter.ActorID))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	manifest := make([]agent.MemoryManifestEntry, 0, limit)
	for rows.Next() {
		var entry agent.MemoryManifestEntry
		if scanErr := rows.Scan(&entry.ID, &entry.RevisionKey, &entry.Title, &entry.Description,
			&entry.SourceLayer, &entry.SemanticType, &entry.ProjectKey, &entry.UpdatedAt,
			&entry.LastVerifiedAt, &entry.FreshnessClass, &entry.Confidence, &entry.Importance,
			&entry.Pinned); scanErr != nil {
			return nil, fmt.Errorf("scan memory manifest: %w", scanErr)
		}
		manifest = append(manifest, entry)
	}
	return manifest, rows.Err()
}

// ListVisibleMemoryManifest enumerates the complete policy-visible catalog
// without applying lexical/vector relevance thresholds. The returned excerpt
// is bounded and is safe to send to the model router; full bodies are loaded
// only after the router selects IDs.
func (s *RunStore) ListVisibleMemoryManifest(ctx context.Context, run agent.Run, policy agent.MemoryPolicy) ([]agent.MemoryManifestEntry, error) {
	_, layers, types, projectKey, teamID, _, limit, ok := memoryRecallSelection(policy)
	if !ok {
		return []agent.MemoryManifestEntry{}, nil
	}
	var agentID, userID, sessionID string
	if err := s.pool.QueryRow(ctx, `
		SELECT definition.id::text, COALESCE(session.user_id,''), COALESCE(session.id::text,'')
		FROM agent_platform.agent_runs run
		JOIN agent_platform.agent_versions version ON version.id=run.agent_version_id
		JOIN agent_platform.agent_definitions definition ON definition.id=version.agent_id
		LEFT JOIN agent_platform.agent_sessions session ON session.id=run.session_id
		WHERE run.id=$1::uuid AND run.tenant_id=$2`, run.ID, run.TenantID).Scan(&agentID, &userID, &sessionID); err != nil {
		return nil, fmt.Errorf("resolve memory visibility identity: %w", err)
	}
	filter := agent.MemoryFilter{
		TenantID: run.TenantID, AgentID: agentID, UserID: userID, SessionID: sessionID,
		ProjectKey: projectKey, TeamID: teamID, Status: agent.MemoryStatusActive,
	}
	manifest, err := s.ListMemoryManifest(ctx, filter, limit)
	if err != nil {
		return nil, fmt.Errorf("list visible memory catalog: %w", err)
	}
	// ListMemoryManifest applies visibility but accepts only scalar taxonomy
	// filters. Apply the policy's layer/type allowlists here as a final guard,
	// then attach bounded body previews for the model router.
	filtered := make([]agent.MemoryManifestEntry, 0, len(manifest))
	layerAllowed := make(map[string]struct{}, len(layers))
	for _, layer := range layers {
		layerAllowed[layer] = struct{}{}
	}
	typeAllowed := make(map[string]struct{}, len(types))
	for _, semanticType := range types {
		typeAllowed[semanticType] = struct{}{}
	}
	for _, entry := range manifest {
		if len(layerAllowed) != 0 {
			if _, exists := layerAllowed[entry.SourceLayer]; !exists {
				continue
			}
		}
		if len(typeAllowed) != 0 {
			if _, exists := typeAllowed[entry.SemanticType]; !exists {
				continue
			}
		}
		filtered = append(filtered, entry)
	}
	if len(filtered) == 0 {
		return filtered, nil
	}
	ids := make([]string, 0, len(filtered))
	for _, entry := range filtered {
		ids = append(ids, entry.ID)
	}
	memories, err := s.LoadMemoriesForRun(ctx, run, policy, ids)
	if err != nil {
		return nil, fmt.Errorf("load visible memory previews: %w", err)
	}
	byID := make(map[string]agent.Memory, len(memories))
	for _, memory := range memories {
		byID[memory.ID] = memory
	}
	for index := range filtered {
		if memory, exists := byID[filtered[index].ID]; exists {
			filtered[index].Excerpt = memoryPreview(memory.Body, memory.Content)
		}
	}
	return filtered, nil
}

func memoryPreview(body, content string) string {
	value := strings.TrimSpace(body)
	if value == "" {
		value = strings.TrimSpace(content)
	}
	if value == "" {
		return ""
	}
	lines := strings.Split(value, "\n")
	if len(lines) > 8 {
		lines = lines[:8]
	}
	value = strings.TrimSpace(strings.Join(lines, "\n"))
	runes := []rune(value)
	if len(runes) > 1600 {
		value = string(runes[:1600]) + "…"
	}
	return value
}

func memoryRevisionKey(memory agent.Memory) string {
	if strings.TrimSpace(memory.ContentHash) != "" {
		return memory.ContentHash
	}
	if !memory.UpdatedAt.IsZero() {
		return memory.UpdatedAt.UTC().Format(time.RFC3339Nano)
	}
	return memory.ID
}

func memoryRecallSelection(policy agent.MemoryPolicy) (scopes, layers, types []string, projectKey, teamID string, minimum float64, limit int, ok bool) {
	if !policy.Enabled || policy.MaxRecall <= 0 {
		return nil, nil, nil, "", "", 0, 0, false
	}
	allowed := make(map[string]bool, len(policy.ReadScopes))
	for _, scope := range policy.ReadScopes {
		if validMemoryScopes[scope] {
			allowed[scope] = true
		}
	}
	if len(allowed) == 0 {
		return nil, nil, nil, "", "", 0, 0, false
	}
	for _, scope := range []string{"tenant", "agent", "user", "session"} {
		if allowed[scope] {
			scopes = append(scopes, scope)
		}
	}
	for _, layer := range policy.ReadLayers {
		layer = strings.ToLower(strings.TrimSpace(layer))
		if agent.ValidMemoryLayer(layer) && (layer != agent.MemoryLayerTeam || policy.TeamMemoryEnabled) {
			layers = append(layers, layer)
		}
	}
	if len(policy.ReadLayers) == 0 {
		layers = []string{agent.MemoryLayerManaged, agent.MemoryLayerUser, agent.MemoryLayerProject, agent.MemoryLayerLocal, agent.MemoryLayerAuto}
		if policy.TeamMemoryEnabled {
			layers = append(layers, agent.MemoryLayerTeam)
		}
	} else if len(layers) == 0 {
		return nil, nil, nil, "", "", 0, 0, false
	}
	for _, semanticType := range policy.ReadTypes {
		semanticType = strings.ToLower(strings.TrimSpace(semanticType))
		if agent.ValidMemoryType(semanticType) {
			types = append(types, semanticType)
		}
	}
	limit = policy.MaxRecall
	if limit > 20 {
		limit = 20
	}
	minimum = policy.MinimumScore
	if minimum <= 0 {
		minimum = 0.15
	}
	return scopes, layers, types, strings.TrimSpace(policy.ProjectKey), strings.TrimSpace(policy.TeamID), minimum, limit, true
}

func (s *RunStore) DeleteMemory(ctx context.Context, tenantID, memoryID, actor string) error {
	command, err := s.pool.Exec(ctx, `UPDATE agent_platform.agent_memories
		SET deleted_at=now(),status='deleted',updated_at=now(),metadata=metadata || jsonb_build_object('deleted_by',$3::text)
		WHERE id=$1::uuid AND tenant_id=$2 AND deleted_at IS NULL`, memoryID, tenantID, actor)
	if err != nil {
		return fmt.Errorf("delete memory: %w", err)
	}
	if command.RowsAffected() == 0 {
		return agent.ErrMemoryNotFound
	}
	payload, _ := json.Marshal(map[string]any{"memory_id": strings.TrimSpace(memoryID), "deleted_by": strings.TrimSpace(actor)})
	if _, eventErr := s.RecordMemoryLifecycleEvent(ctx, agent.MemoryLifecycleEvent{
		TenantID: strings.TrimSpace(tenantID), MemoryID: optionalStringPointer(strings.TrimSpace(memoryID)), Actor: optionalStringPointer(actor),
		EventType: "MEMORY_DELETED", IdempotencyKey: optionalLifecycleKey("delete", strings.TrimSpace(memoryID)), Payload: payload,
	}); eventErr != nil {
		return eventErr
	}
	return nil
}

func (s *RunStore) RecordMemoryFeedback(ctx context.Context, feedback agent.MemoryFeedback) (agent.MemoryFeedback, error) {
	feedback.TenantID = strings.TrimSpace(feedback.TenantID)
	feedback.MemoryID = strings.TrimSpace(feedback.MemoryID)
	feedback.Action = strings.ToLower(strings.TrimSpace(feedback.Action))
	if feedback.TenantID == "" || feedback.MemoryID == "" {
		return agent.MemoryFeedback{}, errors.New("tenant_id and memory_id are required")
	}
	switch feedback.Action {
	case "used", "ignored", "contradicted", "verified", "updated":
	default:
		return agent.MemoryFeedback{}, errors.New("memory feedback action is invalid")
	}
	evidence := feedback.Evidence
	if len(evidence) == 0 {
		evidence = json.RawMessage(`{}`)
	}
	if !validJSONObject(evidence) {
		return agent.MemoryFeedback{}, errors.New("memory feedback evidence must be a JSON object")
	}
	row := s.pool.QueryRow(ctx, `
		INSERT INTO agent_platform.memory_feedback
			(tenant_id,memory_id,run_id,turn_id,action,actor,reason,evidence)
		SELECT $1,$2::uuid,NULLIF($3,'')::uuid,NULLIF($4,''),$5,$6,$7,$8::jsonb
		WHERE EXISTS (SELECT 1 FROM agent_platform.agent_memories memory WHERE memory.id=$2::uuid AND memory.tenant_id=$1)
		ON CONFLICT (tenant_id,memory_id,COALESCE(run_id,'00000000-0000-0000-0000-000000000000'::uuid),COALESCE(turn_id,''),action,COALESCE(actor,''))
		DO UPDATE SET reason=EXCLUDED.reason,evidence=EXCLUDED.evidence
		RETURNING id::text,tenant_id,memory_id::text,run_id::text,turn_id,action,actor,reason,evidence,created_at`,
		feedback.TenantID, feedback.MemoryID, stringPointerValue(feedback.RunID), feedback.TurnID, feedback.Action, stringPointerValue(feedback.Actor), stringPointerValue(feedback.Reason), evidence)
	var stored agent.MemoryFeedback
	if err := row.Scan(&stored.ID, &stored.TenantID, &stored.MemoryID, &stored.RunID, &stored.TurnID, &stored.Action, &stored.Actor, &stored.Reason, &stored.Evidence, &stored.CreatedAt); errors.Is(err, pgx.ErrNoRows) {
		return agent.MemoryFeedback{}, agent.ErrMemoryNotFound
	} else if err != nil {
		return agent.MemoryFeedback{}, fmt.Errorf("record memory feedback: %w", err)
	}
	eventType := map[string]string{"contradicted": "MEMORY_CONTRADICTED", "verified": "MEMORY_VERIFIED", "updated": "MEMORY_UPDATED", "used": "MEMORY_INJECTED", "ignored": "MEMORY_SUPPRESSED"}[feedback.Action]
	payload, _ := json.Marshal(map[string]any{"feedback_id": stored.ID, "action": stored.Action, "reason": stringPointerValue(stored.Reason)})
	if _, eventErr := s.RecordMemoryLifecycleEvent(ctx, agent.MemoryLifecycleEvent{
		TenantID: stored.TenantID, MemoryID: &stored.MemoryID, RunID: stored.RunID,
		Actor: stored.Actor, EventType: eventType, IdempotencyKey: optionalLifecycleKey("feedback", stored.ID), Payload: payload,
	}); eventErr != nil {
		return agent.MemoryFeedback{}, eventErr
	}
	return stored, nil
}

func (s *RunStore) ListMemoryFeedback(ctx context.Context, tenantID, memoryID string, limit int) ([]agent.MemoryFeedback, error) {
	if limit <= 0 || limit > 200 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `
		SELECT id::text,tenant_id,memory_id::text,run_id::text,turn_id,action,actor,reason,evidence,created_at
		FROM agent_platform.memory_feedback
		WHERE tenant_id=$1 AND memory_id=$2::uuid
		ORDER BY created_at DESC LIMIT $3`, strings.TrimSpace(tenantID), strings.TrimSpace(memoryID), limit)
	if err != nil {
		return nil, fmt.Errorf("list memory feedback: %w", err)
	}
	defer rows.Close()
	feedback := make([]agent.MemoryFeedback, 0)
	for rows.Next() {
		var item agent.MemoryFeedback
		if scanErr := rows.Scan(&item.ID, &item.TenantID, &item.MemoryID, &item.RunID, &item.TurnID, &item.Action, &item.Actor, &item.Reason, &item.Evidence, &item.CreatedAt); scanErr != nil {
			return nil, fmt.Errorf("scan memory feedback: %w", scanErr)
		}
		feedback = append(feedback, item)
	}
	return feedback, rows.Err()
}

func (s *RunStore) RecordMemoryLifecycleEvent(ctx context.Context, item agent.MemoryLifecycleEvent) (agent.MemoryLifecycleEvent, error) {
	item.TenantID = strings.TrimSpace(item.TenantID)
	item.EventType = strings.ToUpper(strings.TrimSpace(item.EventType))
	if item.TenantID == "" || !validMemoryLifecycleEventTypes[item.EventType] {
		return agent.MemoryLifecycleEvent{}, errors.New("tenant_id and a valid memory lifecycle event_type are required")
	}
	payload := item.Payload
	if len(payload) == 0 {
		payload = json.RawMessage(`{}`)
	}
	if !validJSONObject(payload) {
		return agent.MemoryLifecycleEvent{}, errors.New("memory lifecycle event payload must be a JSON object")
	}
	row := s.pool.QueryRow(ctx, `
		INSERT INTO agent_platform.memory_lifecycle_events
			(tenant_id,memory_id,source_id,run_id,event_type,actor,idempotency_key,payload)
		VALUES($1,NULLIF($2,'')::uuid,NULLIF($3,'')::uuid,NULLIF($4,'')::uuid,$5,$6,$7,$8::jsonb)
		ON CONFLICT (tenant_id,event_type,idempotency_key) WHERE idempotency_key IS NOT NULL
		DO UPDATE SET actor=EXCLUDED.actor,payload=EXCLUDED.payload
		RETURNING id::text,tenant_id,memory_id::text,source_id::text,run_id::text,event_type,actor,idempotency_key,payload,created_at`,
		item.TenantID, stringPointerValue(item.MemoryID), stringPointerValue(item.SourceID), stringPointerValue(item.RunID),
		item.EventType, stringPointerValue(item.Actor), stringPointerValue(item.IdempotencyKey), payload)
	var stored agent.MemoryLifecycleEvent
	if err := row.Scan(&stored.ID, &stored.TenantID, &stored.MemoryID, &stored.SourceID, &stored.RunID,
		&stored.EventType, &stored.Actor, &stored.IdempotencyKey, &stored.Payload, &stored.CreatedAt); err != nil {
		return agent.MemoryLifecycleEvent{}, fmt.Errorf("record memory lifecycle event: %w", err)
	}
	return stored, nil
}

func (s *RunStore) ListMemoryLifecycleEvents(ctx context.Context, tenantID, memoryID, actorID string, limit int) ([]agent.MemoryLifecycleEvent, error) {
	if limit <= 0 || limit > 200 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `
		SELECT id::text,tenant_id,memory_id::text,source_id::text,run_id::text,event_type,actor,idempotency_key,payload,created_at
		FROM agent_platform.memory_lifecycle_events lifecycle
		LEFT JOIN agent_platform.agent_memories memory ON memory.id=lifecycle.memory_id
		WHERE lifecycle.tenant_id=$1 AND lifecycle.memory_id=$2::uuid
		  AND (memory.source_layer IS NULL OR memory.source_layer <> 'team' OR EXISTS (
			SELECT 1 FROM agent_platform.memory_team_memberships membership
			WHERE membership.tenant_id=memory.tenant_id AND membership.team_id=memory.team_id
			  AND membership.user_id=$3 AND membership.enabled
		  ))
		ORDER BY lifecycle.created_at DESC,lifecycle.id DESC LIMIT $4`, strings.TrimSpace(tenantID), strings.TrimSpace(memoryID), strings.TrimSpace(actorID), limit)
	if err != nil {
		return nil, fmt.Errorf("list memory lifecycle events: %w", err)
	}
	defer rows.Close()
	events := make([]agent.MemoryLifecycleEvent, 0)
	for rows.Next() {
		var item agent.MemoryLifecycleEvent
		if scanErr := rows.Scan(&item.ID, &item.TenantID, &item.MemoryID, &item.SourceID, &item.RunID,
			&item.EventType, &item.Actor, &item.IdempotencyKey, &item.Payload, &item.CreatedAt); scanErr != nil {
			return nil, fmt.Errorf("scan memory lifecycle event: %w", scanErr)
		}
		events = append(events, item)
	}
	return events, rows.Err()
}

// ListMemoryLifecycleEventsForRun returns only lifecycle rows explicitly
// attributed to a Run. Team-scoped memories still require an active
// membership, matching the memory-detail audit endpoint.
func (s *RunStore) ListMemoryLifecycleEventsForRun(ctx context.Context, tenantID, runID, actorID string, limit int) ([]agent.MemoryLifecycleEvent, error) {
	if limit <= 0 || limit > 1000 {
		limit = 200
	}
	rows, err := s.pool.Query(ctx, `
		SELECT lifecycle.id::text,lifecycle.tenant_id,lifecycle.memory_id::text,lifecycle.source_id::text,lifecycle.run_id::text,
		       lifecycle.event_type,lifecycle.actor,lifecycle.idempotency_key,lifecycle.payload,lifecycle.created_at
		FROM agent_platform.memory_lifecycle_events lifecycle
		LEFT JOIN agent_platform.agent_memories memory ON memory.id=lifecycle.memory_id
		WHERE lifecycle.tenant_id=$1 AND lifecycle.run_id=$2::uuid
		  AND (memory.source_layer IS NULL OR memory.source_layer <> 'team' OR EXISTS (
			SELECT 1 FROM agent_platform.memory_team_memberships membership
			WHERE membership.tenant_id=memory.tenant_id AND membership.team_id=memory.team_id
			  AND membership.user_id=$3 AND membership.enabled
		  ))
		ORDER BY lifecycle.created_at,lifecycle.id LIMIT $4`, strings.TrimSpace(tenantID), strings.TrimSpace(runID), strings.TrimSpace(actorID), limit)
	if err != nil {
		return nil, fmt.Errorf("list memory lifecycle events for run: %w", err)
	}
	defer rows.Close()
	events := make([]agent.MemoryLifecycleEvent, 0)
	for rows.Next() {
		var item agent.MemoryLifecycleEvent
		if scanErr := rows.Scan(&item.ID, &item.TenantID, &item.MemoryID, &item.SourceID, &item.RunID,
			&item.EventType, &item.Actor, &item.IdempotencyKey, &item.Payload, &item.CreatedAt); scanErr != nil {
			return nil, fmt.Errorf("scan memory lifecycle event for run: %w", scanErr)
		}
		events = append(events, item)
	}
	return events, rows.Err()
}

func (s *RunStore) ListMemoryRevisions(ctx context.Context, tenantID, memoryID string, limit int) ([]agent.MemoryRevision, error) {
	if limit <= 0 || limit > 200 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `
		SELECT id::text,memory_id::text,revision,title,description,body,structured_data,source_message_ids,source_run_id::text,source_event_from,source_event_to,reason,created_by_type,created_by,created_at
		FROM agent_platform.memory_revisions
		WHERE tenant_id=$1 AND memory_id=$2::uuid
		ORDER BY revision DESC LIMIT $3`, strings.TrimSpace(tenantID), strings.TrimSpace(memoryID), limit)
	if err != nil {
		return nil, fmt.Errorf("list memory revisions: %w", err)
	}
	defer rows.Close()
	revisions := make([]agent.MemoryRevision, 0)
	for rows.Next() {
		var revision agent.MemoryRevision
		var sourceMessages json.RawMessage
		if scanErr := rows.Scan(&revision.ID, &revision.MemoryID, &revision.Revision, &revision.Title, &revision.Description, &revision.Body, &revision.StructuredData, &sourceMessages, &revision.SourceRunID, &revision.SourceEventFrom, &revision.SourceEventTo, &revision.Reason, &revision.CreatedByType, &revision.CreatedBy, &revision.CreatedAt); scanErr != nil {
			return nil, fmt.Errorf("scan memory revision: %w", scanErr)
		}
		if len(sourceMessages) != 0 && json.Unmarshal(sourceMessages, &revision.SourceMessageIDs) != nil {
			return nil, errors.New("memory revision source_message_ids is invalid JSON")
		}
		revisions = append(revisions, revision)
	}
	return revisions, rows.Err()
}

func (s *RunStore) VerifyMemory(ctx context.Context, tenantID, memoryID, actor string) (agent.Memory, error) {
	row := s.pool.QueryRow(ctx, `
		UPDATE agent_platform.agent_memories
		SET last_verified_at=now(),updated_at=now(),metadata=metadata || jsonb_build_object('last_verified_by',$3::text)
		WHERE id=$1::uuid AND tenant_id=$2 AND deleted_at IS NULL
		RETURNING `+memoryColumns, memoryID, tenantID, strings.TrimSpace(actor))
	memory, err := scanMemory(row)
	if errors.Is(err, pgx.ErrNoRows) {
		return agent.Memory{}, agent.ErrMemoryNotFound
	}
	if err != nil {
		return agent.Memory{}, fmt.Errorf("verify memory: %w", err)
	}
	payload, _ := json.Marshal(map[string]any{"memory_id": memory.ID, "verified_by": strings.TrimSpace(actor)})
	if _, eventErr := s.RecordMemoryLifecycleEvent(ctx, agent.MemoryLifecycleEvent{
		TenantID: memory.TenantID, MemoryID: &memory.ID, Actor: optionalStringPointer(actor),
		EventType: "MEMORY_VERIFIED", IdempotencyKey: optionalLifecycleKey("verify", memory.ID+":"+strings.TrimSpace(actor)), Payload: payload,
	}); eventErr != nil {
		return agent.Memory{}, eventErr
	}
	return memory, nil
}

func (s *RunStore) PromoteMemoryToTeam(ctx context.Context, tenantID, memoryID, teamID, actor string) (agent.Memory, error) {
	row := s.pool.QueryRow(ctx, `
		UPDATE agent_platform.agent_memories memory
		SET source_layer='team',team_id=$3,scope='tenant',agent_id=NULL,user_id=NULL,session_id=NULL,status='active',updated_at=now(),metadata=metadata || jsonb_build_object('promoted_by',$4::text)
		WHERE memory.id=$1::uuid AND memory.tenant_id=$2 AND memory.deleted_at IS NULL
		  AND memory.source_layer='auto'
		  AND EXISTS (
			SELECT 1 FROM agent_platform.memory_team_memberships membership
			WHERE membership.tenant_id=memory.tenant_id AND membership.team_id=$3
			  AND membership.user_id=$4 AND membership.role IN ('manager','owner') AND membership.enabled
		  )
		RETURNING `+memoryColumns, memoryID, tenantID, strings.TrimSpace(teamID), strings.TrimSpace(actor))
	memory, err := scanMemory(row)
	if errors.Is(err, pgx.ErrNoRows) {
		return agent.Memory{}, agent.ErrMemoryNotFound
	}
	if err != nil {
		return agent.Memory{}, fmt.Errorf("promote memory to team: %w", err)
	}
	requestPayload, _ := json.Marshal(map[string]any{"memory_id": memory.ID, "team_id": strings.TrimSpace(teamID), "requested_by": strings.TrimSpace(actor)})
	if _, eventErr := s.RecordMemoryLifecycleEvent(ctx, agent.MemoryLifecycleEvent{
		TenantID: memory.TenantID, MemoryID: &memory.ID, Actor: optionalStringPointer(actor),
		EventType: "MEMORY_TEAM_PROMOTION_REQUESTED", IdempotencyKey: optionalLifecycleKey("promote-request", memory.ID+":"+strings.TrimSpace(teamID)+":"+strings.TrimSpace(actor)), Payload: requestPayload,
	}); eventErr != nil {
		return agent.Memory{}, eventErr
	}
	payload, _ := json.Marshal(map[string]any{"memory_id": memory.ID, "team_id": strings.TrimSpace(teamID), "promoted_by": strings.TrimSpace(actor)})
	if _, eventErr := s.RecordMemoryLifecycleEvent(ctx, agent.MemoryLifecycleEvent{
		TenantID: memory.TenantID, MemoryID: &memory.ID, Actor: optionalStringPointer(actor),
		EventType: "MEMORY_TEAM_PROMOTED", IdempotencyKey: optionalLifecycleKey("promote-team", memory.ID+":"+strings.TrimSpace(teamID)), Payload: payload,
	}); eventErr != nil {
		return agent.Memory{}, eventErr
	}
	return memory, nil
}

func (s *RunStore) UpsertMemoryTeamMembership(ctx context.Context, tenantID, teamID, userID, role string, enabled bool) (agent.MemoryTeamMembership, error) {
	tenantID, teamID, userID, role = strings.TrimSpace(tenantID), strings.TrimSpace(teamID), strings.TrimSpace(userID), strings.ToLower(strings.TrimSpace(role))
	if tenantID == "" || teamID == "" || userID == "" {
		return agent.MemoryTeamMembership{}, errors.New("tenant_id, team_id and user_id are required")
	}
	if role == "" {
		role = "member"
	}
	if role != "member" && role != "manager" && role != "owner" {
		return agent.MemoryTeamMembership{}, errors.New("memory team membership role must be member, manager or owner")
	}
	row := s.pool.QueryRow(ctx, `
		INSERT INTO agent_platform.memory_team_memberships (tenant_id,team_id,user_id,role,enabled)
		VALUES($1,$2,$3,$4,$5)
		ON CONFLICT (tenant_id,team_id,user_id) DO UPDATE SET role=EXCLUDED.role,enabled=EXCLUDED.enabled,updated_at=now()
		RETURNING tenant_id,team_id,user_id,role,enabled,created_at,updated_at`, tenantID, teamID, userID, role, enabled)
	var membership agent.MemoryTeamMembership
	if err := row.Scan(&membership.TenantID, &membership.TeamID, &membership.UserID, &membership.Role, &membership.Enabled, &membership.CreatedAt, &membership.UpdatedAt); err != nil {
		return agent.MemoryTeamMembership{}, fmt.Errorf("upsert memory team membership: %w", err)
	}
	return membership, nil
}

func (s *RunStore) DeleteMemoryTeamMembership(ctx context.Context, tenantID, teamID, userID string) error {
	command, err := s.pool.Exec(ctx, `DELETE FROM agent_platform.memory_team_memberships WHERE tenant_id=$1 AND team_id=$2 AND user_id=$3`, strings.TrimSpace(tenantID), strings.TrimSpace(teamID), strings.TrimSpace(userID))
	if err != nil {
		return fmt.Errorf("delete memory team membership: %w", err)
	}
	if command.RowsAffected() == 0 {
		return agent.ErrMemoryNotFound
	}
	return nil
}

func (s *RunStore) ListMemoryTeamMemberships(ctx context.Context, tenantID, teamID string, limit int) ([]agent.MemoryTeamMembership, error) {
	if limit <= 0 || limit > 200 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `
		SELECT tenant_id,team_id,user_id,role,enabled,created_at,updated_at
		FROM agent_platform.memory_team_memberships
		WHERE tenant_id=$1 AND ($2='' OR team_id=$2)
		ORDER BY team_id,user_id LIMIT $3`, strings.TrimSpace(tenantID), strings.TrimSpace(teamID), limit)
	if err != nil {
		return nil, fmt.Errorf("list memory team memberships: %w", err)
	}
	defer rows.Close()
	memberships := make([]agent.MemoryTeamMembership, 0)
	for rows.Next() {
		var membership agent.MemoryTeamMembership
		if scanErr := rows.Scan(&membership.TenantID, &membership.TeamID, &membership.UserID, &membership.Role, &membership.Enabled, &membership.CreatedAt, &membership.UpdatedAt); scanErr != nil {
			return nil, fmt.Errorf("scan memory team membership: %w", scanErr)
		}
		memberships = append(memberships, membership)
	}
	return memberships, rows.Err()
}

// SyncStaticMemorySources persists immutable file observations without putting
// their full bodies into the database. A changed content hash creates the next
// source revision; an unchanged hash is idempotent. Team files without an
// explicit team ID are intentionally not persisted as governed sources.
func (s *RunStore) SyncStaticMemorySources(ctx context.Context, run agent.Run, documents []agent.StaticMemoryDocument) error {
	for _, document := range documents {
		if strings.TrimSpace(document.SourceLayer) == agent.MemoryLayerTeam && strings.TrimSpace(document.TeamID) == "" {
			continue
		}
		uri := "static://" + strings.TrimSpace(document.SourceLayer) + "/" + strings.TrimSpace(document.Path)
		if uri == "static:///" || strings.TrimSpace(document.ContentHash) == "" {
			continue
		}
		writableBy := "none"
		authority := 0.5
		switch document.SourceLayer {
		case agent.MemoryLayerManaged:
			writableBy, authority = "admin", 1.0
		case agent.MemoryLayerUser:
			writableBy, authority = "owner", 0.8
		case agent.MemoryLayerProject:
			writableBy, authority = "owner", 0.75
		case agent.MemoryLayerLocal:
			writableBy, authority = "owner", 0.7
		case agent.MemoryLayerAuto:
			writableBy, authority = "agent", 0.45
		case agent.MemoryLayerTeam:
			writableBy, authority = "team", 0.65
		default:
			continue
		}
		var sourceID string
		var revision int64
		err := s.pool.QueryRow(ctx, `
			WITH identity AS (
				SELECT definition.id AS agent_id, session.user_id
				FROM agent_platform.agent_runs run
				JOIN agent_platform.agent_versions version ON version.id=run.agent_version_id
				JOIN agent_platform.agent_definitions definition ON definition.id=version.agent_id
				LEFT JOIN agent_platform.agent_sessions session ON session.id=run.session_id
				WHERE run.id=$1::uuid AND run.tenant_id=$2
			), previous AS (
				SELECT
					COALESCE((SELECT revision FROM agent_platform.memory_sources WHERE tenant_id=$2 AND uri=$3 ORDER BY revision DESC LIMIT 1),0) AS revision,
					(SELECT content_hash FROM agent_platform.memory_sources WHERE tenant_id=$2 AND uri=$3 ORDER BY revision DESC LIMIT 1) AS content_hash
			)
			INSERT INTO agent_platform.memory_sources
				(tenant_id,source_layer,agent_id,user_id,project_key,team_id,uri,display_name,content_hash,revision,authority,writable_by,enabled,observed_at)
			SELECT $2,$4,identity.agent_id,identity.user_id,NULLIF($5,''),NULLIF($6,''),$3,$7,$8,previous.revision+1,$9,$10,TRUE,now()
			FROM identity CROSS JOIN previous
			WHERE previous.content_hash IS DISTINCT FROM $8
			ON CONFLICT (tenant_id,uri,revision) DO NOTHING
			RETURNING id::text,revision`,
			run.ID, run.TenantID, uri, document.SourceLayer, document.ProjectKey, document.TeamID,
			document.Path, document.ContentHash, authority, writableBy).Scan(&sourceID, &revision)
		if errors.Is(err, pgx.ErrNoRows) {
			continue
		}
		if err != nil {
			return fmt.Errorf("sync static memory source %s: %w", uri, err)
		}
		payload, _ := json.Marshal(map[string]any{"uri": uri, "path": document.Path, "content_hash": document.ContentHash, "revision": revision})
		if _, eventErr := s.RecordMemoryLifecycleEvent(ctx, agent.MemoryLifecycleEvent{
			TenantID: run.TenantID, SourceID: &sourceID, RunID: &run.ID,
			EventType:      "MEMORY_SOURCE_REVISION_OBSERVED",
			IdempotencyKey: optionalLifecycleKey("source-revision", sourceID), Payload: payload,
		}); eventErr != nil {
			return fmt.Errorf("record static memory source event %s: %w", uri, eventErr)
		}
	}
	return nil
}

func (s *RunStore) ListMemorySources(ctx context.Context, tenantID, sourceLayer, projectKey, teamID, actorID string, limit int) ([]agent.MemorySource, error) {
	if limit <= 0 || limit > 200 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `
		SELECT id::text,tenant_id,source_layer,agent_id::text,user_id,project_key,team_id,uri,display_name,content_hash,revision,authority,writable_by,enabled,git_commit,observed_at,created_at,updated_at
		FROM agent_platform.memory_sources
		WHERE tenant_id=$1 AND enabled
		  AND ($2='' OR source_layer=$2)
		  AND ($3='' OR project_key=$3)
		  AND ($4='' OR team_id=$4)
		  AND (source_layer <> 'team' OR EXISTS (
			SELECT 1 FROM agent_platform.memory_team_memberships membership
			WHERE membership.tenant_id=memory_sources.tenant_id AND membership.team_id=memory_sources.team_id
			  AND membership.user_id=$5 AND membership.enabled
		  ))
		ORDER BY updated_at DESC,uri,revision DESC LIMIT $6`, strings.TrimSpace(tenantID), strings.TrimSpace(sourceLayer), strings.TrimSpace(projectKey), strings.TrimSpace(teamID), strings.TrimSpace(actorID), limit)
	if err != nil {
		return nil, fmt.Errorf("list memory sources: %w", err)
	}
	defer rows.Close()
	sources := make([]agent.MemorySource, 0)
	for rows.Next() {
		var source agent.MemorySource
		if scanErr := rows.Scan(&source.ID, &source.TenantID, &source.SourceLayer, &source.AgentID, &source.UserID, &source.ProjectKey, &source.TeamID, &source.URI, &source.DisplayName, &source.ContentHash, &source.Revision, &source.Authority, &source.WritableBy, &source.Enabled, &source.GitCommit, &source.ObservedAt, &source.CreatedAt, &source.UpdatedAt); scanErr != nil {
			return nil, fmt.Errorf("scan memory source: %w", scanErr)
		}
		sources = append(sources, source)
	}
	return sources, rows.Err()
}

func (s *RunStore) FindActiveMemoryByCanonicalKey(ctx context.Context, tenantID, sourceLayer, canonicalKey string) (agent.Memory, bool, error) {
	memory, err := scanMemory(s.pool.QueryRow(ctx, `SELECT `+memoryColumns+`
		FROM agent_platform.agent_memories
		WHERE tenant_id=$1 AND source_layer=$2 AND canonical_key=$3
		  AND status IN ('active','review') AND deleted_at IS NULL
		ORDER BY updated_at DESC LIMIT 1`, tenantID, sourceLayer, canonicalKey))
	if errors.Is(err, pgx.ErrNoRows) {
		return agent.Memory{}, false, nil
	}
	if err != nil {
		return agent.Memory{}, false, fmt.Errorf("find memory canonical key: %w", err)
	}
	return memory, true, nil
}

func (s *RunStore) SupersedeMemory(ctx context.Context, tenantID, memoryID, supersededBy string) error {
	command, err := s.pool.Exec(ctx, `UPDATE agent_platform.agent_memories
		SET status='superseded',updated_at=now(),metadata=metadata || jsonb_build_object('superseded_by',$3::text)
		WHERE id=$1::uuid AND tenant_id=$2 AND deleted_at IS NULL AND status='active'`, memoryID, tenantID, supersededBy)
	if err != nil {
		return fmt.Errorf("supersede memory: %w", err)
	}
	if command.RowsAffected() == 0 {
		return agent.ErrMemoryNotFound
	}
	payload, _ := json.Marshal(map[string]any{"memory_id": strings.TrimSpace(memoryID), "superseded_by": strings.TrimSpace(supersededBy)})
	if _, eventErr := s.RecordMemoryLifecycleEvent(ctx, agent.MemoryLifecycleEvent{
		TenantID: strings.TrimSpace(tenantID), MemoryID: optionalStringPointer(strings.TrimSpace(memoryID)),
		EventType: "MEMORY_SUPERSEDED", IdempotencyKey: optionalLifecycleKey("supersede", strings.TrimSpace(memoryID)+":"+strings.TrimSpace(supersededBy)), Payload: payload,
	}); eventErr != nil {
		return eventErr
	}
	return nil
}

// UpdateAutoMemoryFromCandidate applies an extractor update/merge to the
// existing Auto Memory identity. Keeping the ID stable preserves references
// and lets the revision table describe the change; supersede remains the
// explicit path when a new identity is required.
func (s *RunStore) UpdateAutoMemoryFromCandidate(ctx context.Context, run agent.Run, memoryID string, candidate agent.MemoryExtractionCandidate, reason string, sourceFrom, sourceTo *int64) (agent.Memory, error) {
	if strings.TrimSpace(memoryID) == "" {
		return agent.Memory{}, errors.New("memory_id is required for auto memory update")
	}
	if reason != "update" && reason != "merge" {
		return agent.Memory{}, errors.New("auto memory update reason must be update or merge")
	}
	structured := candidate.StructuredData
	if len(structured) == 0 {
		structured = json.RawMessage(`{}`)
	}
	row := s.pool.QueryRow(ctx, `
		UPDATE agent_platform.agent_memories
		SET kind='semantic',content=$3,importance=$4,source_run_id=$5::uuid,
		    semantic_type=$6,title=$7,description=$8,body=$9,structured_data=$10::jsonb,
		    confidence=$11,freshness_class=$12,content_hash=$13,
		    embedding=NULL,embedding_model=NULL,embedding_status='pending',embedding_error=NULL,embedded_at=NULL,
		    status='active',updated_at=now()
		WHERE id=$1::uuid AND tenant_id=$2 AND source_layer='auto'
		  AND status IN ('active','review') AND deleted_at IS NULL
		RETURNING `+memoryColumns,
		memoryID, run.TenantID, candidate.Body, candidate.Importance, run.ID,
		candidate.SemanticType, candidate.Title, candidate.Description, candidate.Body, structured,
		candidate.Confidence, candidate.FreshnessClass, memorySha256Hex(candidate.Body))
	memory, err := scanMemory(row)
	if errors.Is(err, pgx.ErrNoRows) {
		return agent.Memory{}, agent.ErrMemoryNotFound
	}
	if err != nil {
		return agent.Memory{}, fmt.Errorf("update auto memory: %w", err)
	}
	if err := s.RecordMemoryRevision(ctx, memory, candidate, reason, "memory-extractor", sourceFrom, sourceTo); err != nil {
		return agent.Memory{}, err
	}
	eventType := "MEMORY_UPDATED"
	if reason == "merge" {
		eventType = "MEMORY_MERGED"
	}
	payload, _ := json.Marshal(map[string]any{"memory_id": memory.ID, "reason": reason, "source_run_id": run.ID})
	if _, eventErr := s.RecordMemoryLifecycleEvent(ctx, agent.MemoryLifecycleEvent{
		TenantID: memory.TenantID, MemoryID: &memory.ID, RunID: &run.ID,
		EventType: eventType, IdempotencyKey: optionalLifecycleKey("revision", memory.ID+":"+memory.ContentHash), Payload: payload,
	}); eventErr != nil {
		return agent.Memory{}, eventErr
	}
	return memory, nil
}

func (s *RunStore) RecordMemoryRevision(ctx context.Context, memory agent.Memory, candidate agent.MemoryExtractionCandidate, reason, createdBy string, sourceFrom, sourceTo *int64) error {
	sourceMessages := candidate.SourceMessageIDs
	if sourceMessages == nil {
		sourceMessages = []string{}
	}
	structured := candidate.StructuredData
	if len(structured) == 0 {
		structured = json.RawMessage(`{}`)
	}
	encodedSources, err := json.Marshal(sourceMessages)
	if err != nil {
		return fmt.Errorf("encode memory revision sources: %w", err)
	}
	_, err = s.pool.Exec(ctx, `
		INSERT INTO agent_platform.memory_revisions
			(tenant_id,memory_id,revision,title,description,body,structured_data,source_message_ids,source_run_id,source_event_from,source_event_to,reason,created_by_type,created_by)
		SELECT $1::varchar(128),$2::uuid,COALESCE(MAX(revision),0)+1,$3::varchar(256),$4::varchar(1200),$5,$6::jsonb,$7::jsonb,$8::uuid,$9::bigint,$10::bigint,$11::varchar(64),$12::varchar(32),$13::varchar(128)
		FROM agent_platform.memory_revisions WHERE tenant_id=$1::varchar(128) AND memory_id=$2::uuid`,
		memory.TenantID, memory.ID, memory.Title, memory.Description, memory.Body, structured,
		encodedSources, memory.SourceRunID, sourceFrom, sourceTo, reason, "agent", createdBy)
	if err != nil {
		return fmt.Errorf("record memory revision: %w", err)
	}
	return nil
}

func (s *RunStore) RecordManualMemoryRevision(ctx context.Context, memory agent.Memory, actor string) error {
	_, err := s.pool.Exec(ctx, `
		INSERT INTO agent_platform.memory_revisions
			(tenant_id,memory_id,revision,title,description,body,structured_data,source_message_ids,source_run_id,reason,created_by_type,created_by)
		SELECT $1,$2::uuid,COALESCE(MAX(revision),0)+1,$3,$4,$5,$6::jsonb,'[]'::jsonb,$7::uuid,'manual','user',$8
		FROM agent_platform.memory_revisions WHERE tenant_id=$1 AND memory_id=$2::uuid`,
		memory.TenantID, memory.ID, memory.Title, memory.Description, memory.Body, memory.StructuredData, memory.SourceRunID, strings.TrimSpace(actor))
	if err != nil {
		return fmt.Errorf("record manual memory revision: %w", err)
	}
	return nil
}

// ListMemoryIDsWrittenSince is the persistence half of hasMemoryWritesSince.
// Revisions carry source event bounds, so callers can feed only the identities
// already handled into a new Extractor request without loading memory bodies.
func (s *RunStore) ListMemoryIDsWrittenSince(ctx context.Context, run agent.Run, afterSequence int64) ([]string, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT DISTINCT memory_id::text
		FROM agent_platform.memory_revisions
		WHERE tenant_id=$1 AND source_run_id=$2::uuid
		  AND COALESCE(source_event_to,0)>$3
		ORDER BY memory_id::text`, run.TenantID, run.ID, afterSequence)
	if err != nil {
		return nil, fmt.Errorf("list memory ids written since: %w", err)
	}
	defer rows.Close()
	ids := make([]string, 0)
	for rows.Next() {
		var id string
		if scanErr := rows.Scan(&id); scanErr != nil {
			return nil, fmt.Errorf("scan memory id written since: %w", scanErr)
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

// RecallMemories returns only memories visible to this Run. Live embeddings
// use pgvector cosine distance plus lexical, importance and recency signals;
// provider failures explicitly fall back to the previous pg_trgm ranking.
func (s *RunStore) RecallMemories(ctx context.Context, run agent.Run, policy agent.MemoryPolicy, query string) ([]agent.Memory, error) {
	if !policy.Enabled || policy.MaxRecall <= 0 || strings.TrimSpace(query) == "" {
		return []agent.Memory{}, nil
	}
	allowed := make(map[string]bool, len(policy.ReadScopes))
	for _, scope := range policy.ReadScopes {
		if validMemoryScopes[scope] {
			allowed[scope] = true
		}
	}
	if len(allowed) == 0 {
		return []agent.Memory{}, nil
	}
	layers := make([]string, 0, len(policy.ReadLayers))
	for _, layer := range policy.ReadLayers {
		layer = strings.ToLower(strings.TrimSpace(layer))
		if agent.ValidMemoryLayer(layer) && (layer != agent.MemoryLayerTeam || policy.TeamMemoryEnabled) {
			layers = append(layers, layer)
		}
	}
	if len(policy.ReadLayers) == 0 {
		layers = []string{agent.MemoryLayerManaged, agent.MemoryLayerUser, agent.MemoryLayerProject, agent.MemoryLayerLocal, agent.MemoryLayerAuto}
		if policy.TeamMemoryEnabled {
			layers = append(layers, agent.MemoryLayerTeam)
		}
	} else if len(layers) == 0 {
		return []agent.Memory{}, nil
	}
	types := make([]string, 0, len(policy.ReadTypes))
	for _, semanticType := range policy.ReadTypes {
		semanticType = strings.ToLower(strings.TrimSpace(semanticType))
		if agent.ValidMemoryType(semanticType) {
			types = append(types, semanticType)
		}
	}
	scopes := make([]string, 0, len(allowed))
	for _, scope := range []string{"tenant", "agent", "user", "session"} {
		if allowed[scope] {
			scopes = append(scopes, scope)
		}
	}
	limit := policy.MaxRecall
	if limit > 20 {
		limit = 20
	}
	minimum := policy.MinimumScore
	if minimum <= 0 {
		minimum = 0.15
	}
	query = strings.TrimSpace(query)
	if s.embeddings != nil && s.embeddings.Enabled() {
		result, embedErr := s.embeddings.Embed(ctx, []string{query}, true)
		if embedErr == nil && len(result.Vectors) == 1 {
			return s.recallMemoriesHybrid(ctx, run, scopes, layers, types, strings.TrimSpace(policy.ProjectKey), strings.TrimSpace(policy.TeamID), vectorLiteral(result.Vectors[0]), result.Model, query, minimum, limit)
		}
	}
	return s.recallMemoriesLexical(ctx, run, scopes, layers, types, strings.TrimSpace(policy.ProjectKey), strings.TrimSpace(policy.TeamID), query, minimum, limit)
}

// RecallMemoryManifest performs the ranking stage for the runtime's two-stage
// memory path. The returned entries contain only a bounded excerpt; callers
// must use LoadMemoriesForRun with the selected IDs before injecting any full
// memory body. Relevance ranking is delegated to the model router.
func (s *RunStore) RecallMemoryManifest(ctx context.Context, run agent.Run, policy agent.MemoryPolicy, query string) ([]agent.MemoryManifestEntry, error) {
	query = strings.TrimSpace(query)
	scopes, layers, types, projectKey, teamID, _, limit, ok := memoryRecallSelection(policy)
	if !ok || query == "" {
		return []agent.MemoryManifestEntry{}, nil
	}
	// Relevance ranking is intentionally delegated to the already-resolved Run
	// model. PostgreSQL only applies visibility/lifecycle filters here; using
	// lexical or embedding thresholds before the model would recreate the false
	// negatives this path is designed to eliminate.
	manifest, err := s.recallMemoryManifestCatalog(ctx, run, scopes, layers, types, projectKey, teamID)
	if err != nil || len(manifest) != 0 {
		return manifest, err
	}
	// A manifest must not disappear merely because a visibility join or an old
	// session projection is temporarily incomplete. Fall back to a bounded,
	// policy-layered catalog ordered by pin/importance/recency; the runtime
	// still performs model routing and ID-validated full-body loading.
	return s.recallMemoryManifestFallback(ctx, run, layers, types, projectKey, teamID, limit)
}

func (s *RunStore) recallMemoryManifestFallback(ctx context.Context, run agent.Run, layers, types []string, projectKey, teamID string, limit int) ([]agent.MemoryManifestEntry, error) {
	if limit <= 0 || limit > 20 {
		limit = 8
	}
	rows, err := s.pool.Query(ctx, `
		WITH identity AS (
			SELECT definition.id AS agent_id, session.id AS session_id, session.user_id
			FROM agent_platform.agent_runs run
			JOIN agent_platform.agent_versions version ON version.id=run.agent_version_id
			JOIN agent_platform.agent_definitions definition ON definition.id=version.agent_id
			LEFT JOIN agent_platform.agent_sessions session ON session.id=run.session_id
			WHERE run.id=$1::uuid AND run.tenant_id=$2
		)
		SELECT `+memoryRouterManifestColumns+`, 0::float8 AS recall_score
		FROM agent_platform.agent_memories memory CROSS JOIN identity
		WHERE memory.tenant_id=$2 AND memory.deleted_at IS NULL AND memory.status='active'
		  AND (memory.expires_at IS NULL OR memory.expires_at>now())
		  AND (cardinality($3::text[])=0 OR memory.source_layer=ANY($3::text[]))
		  AND (cardinality($4::text[])=0 OR memory.semantic_type=ANY($4::text[]))
		  AND (memory.scope='tenant'
		    OR (memory.scope='agent' AND memory.agent_id=identity.agent_id)
		    OR (memory.scope='user' AND memory.agent_id=identity.agent_id AND memory.user_id=identity.user_id)
		    OR (memory.scope='session' AND memory.session_id=identity.session_id))
		  AND (memory.source_layer <> 'team' OR EXISTS (
			SELECT 1 FROM agent_platform.memory_team_memberships membership
			WHERE membership.tenant_id=memory.tenant_id AND membership.team_id=memory.team_id
			  AND membership.user_id=identity.user_id AND membership.enabled
		  ))
		  AND ($5='' OR memory.project_key=$5)
		  AND ($6='' OR memory.team_id=$6)
		ORDER BY memory.pinned DESC, memory.importance DESC, memory.updated_at DESC
		LIMIT $7`, run.ID, run.TenantID, layers, types, projectKey, teamID, limit)
	if err != nil {
		return nil, fmt.Errorf("fallback memory manifest: %w", err)
	}
	defer rows.Close()
	manifest := make([]agent.MemoryManifestEntry, 0, limit)
	for rows.Next() {
		entry, scanErr := scanMemoryManifestWithScore(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan fallback memory manifest: %w", scanErr)
		}
		manifest = append(manifest, entry)
	}
	return manifest, rows.Err()
}

// recallMemoryManifestCatalog returns the complete policy-filtered catalog
// without applying relevance thresholds. The model router batches this catalog
// before reading excerpts, so database ordering is not used as relevance.
func (s *RunStore) recallMemoryManifestCatalog(ctx context.Context, run agent.Run, scopes, layers, types []string, projectKey, teamID string) ([]agent.MemoryManifestEntry, error) {
	rows, err := s.pool.Query(ctx, `
		WITH identity AS (
			SELECT definition.id AS agent_id, session.id AS session_id, session.user_id
			FROM agent_platform.agent_runs run
			JOIN agent_platform.agent_versions version ON version.id=run.agent_version_id
			JOIN agent_platform.agent_definitions definition ON definition.id=version.agent_id
			LEFT JOIN agent_platform.agent_sessions session ON session.id=run.session_id
			WHERE run.id=$1::uuid AND run.tenant_id=$2
		)
		SELECT `+memoryRouterManifestColumns+`, 0::float8 AS recall_score
		FROM agent_platform.agent_memories memory CROSS JOIN identity
		WHERE memory.tenant_id=$2 AND memory.deleted_at IS NULL AND memory.status='active'
		  AND (memory.expires_at IS NULL OR memory.expires_at>now())
		  AND memory.scope=ANY($3::text[])
		  AND (cardinality($4::text[])=0 OR memory.source_layer=ANY($4::text[]))
		  AND (cardinality($5::text[])=0 OR memory.semantic_type=ANY($5::text[]))
		  AND ((memory.scope='tenant')
		    OR (memory.scope='agent' AND memory.agent_id=identity.agent_id)
		    OR (memory.scope='user' AND memory.agent_id=identity.agent_id AND memory.user_id=identity.user_id)
		    OR (memory.scope='session' AND memory.session_id=identity.session_id))
		  AND (memory.source_layer <> 'team' OR EXISTS (
			SELECT 1 FROM agent_platform.memory_team_memberships membership
			WHERE membership.tenant_id=memory.tenant_id AND membership.team_id=memory.team_id
			  AND membership.user_id=identity.user_id AND membership.enabled
		  ))
		  AND ($6='' OR memory.project_key=$6)
		  AND ($7='' OR memory.team_id=$7)
		ORDER BY memory.pinned DESC, memory.importance DESC, memory.updated_at DESC`, run.ID, run.TenantID, scopes, layers, types, projectKey, teamID)
	if err != nil {
		return nil, fmt.Errorf("catalog memory manifest: %w", err)
	}
	defer rows.Close()
	manifest := make([]agent.MemoryManifestEntry, 0)
	for rows.Next() {
		entry, scanErr := scanMemoryManifestWithScore(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan catalog memory manifest: %w", scanErr)
		}
		manifest = append(manifest, entry)
	}
	return manifest, rows.Err()
}

func (s *RunStore) recallMemoryManifestLexical(ctx context.Context, run agent.Run, scopes, layers, types []string, projectKey, teamID, query string, minimum float64, limit int) ([]agent.MemoryManifestEntry, error) {
	rows, err := s.pool.Query(ctx, `
		WITH identity AS (
			SELECT definition.id AS agent_id, session.id AS session_id, session.user_id
			FROM agent_platform.agent_runs run
			JOIN agent_platform.agent_versions version ON version.id=run.agent_version_id
			JOIN agent_platform.agent_definitions definition ON definition.id=version.agent_id
			LEFT JOIN agent_platform.agent_sessions session ON session.id=run.session_id
			WHERE run.id=$1::uuid AND run.tenant_id=$2
		), ranked AS (
			SELECT memory.*,
				similarity(lower(COALESCE(NULLIF(memory.title||E'\n'||memory.description,''),memory.content)),lower($3)) AS lexical_score,
				(0.85*similarity(lower(COALESCE(NULLIF(memory.title||E'\n'||memory.description,''),memory.content)),lower($3)) +
				 0.10*memory.importance +
				 0.05/(1+EXTRACT(EPOCH FROM (now()-memory.updated_at))/2592000.0)) AS recall_score
			FROM agent_platform.agent_memories memory CROSS JOIN identity
			WHERE memory.tenant_id=$2 AND memory.deleted_at IS NULL AND memory.status='active'
			  AND (memory.expires_at IS NULL OR memory.expires_at>now())
			  AND memory.scope=ANY($4::text[])
			  AND (cardinality($5::text[])=0 OR memory.source_layer=ANY($5::text[]))
			  AND (cardinality($6::text[])=0 OR memory.semantic_type=ANY($6::text[]))
			  AND ((memory.scope='tenant')
			    OR (memory.scope='agent' AND memory.agent_id=identity.agent_id)
			    OR (memory.scope='user' AND memory.agent_id=identity.agent_id AND memory.user_id=identity.user_id)
			    OR (memory.scope='session' AND memory.session_id=identity.session_id))
			  AND (memory.source_layer <> 'team' OR EXISTS (
				SELECT 1 FROM agent_platform.memory_team_memberships membership
				WHERE membership.tenant_id=memory.tenant_id AND membership.team_id=memory.team_id
				  AND membership.user_id=identity.user_id AND membership.enabled
			  ))
			  AND ($9='' OR memory.project_key=$9)
			  AND ($10='' OR memory.team_id=$10)
		)
		SELECT `+memoryRouterManifestColumns+`, recall_score FROM ranked memory
		WHERE lexical_score >= 0.03 AND recall_score >= $7
		ORDER BY recall_score DESC, importance DESC, updated_at DESC LIMIT $8`,
		run.ID, run.TenantID, query, scopes, layers, types, minimum, limit, projectKey, teamID)
	if err != nil {
		return nil, fmt.Errorf("recall memory manifest: %w", err)
	}
	defer rows.Close()
	manifest := make([]agent.MemoryManifestEntry, 0, limit)
	for rows.Next() {
		entry, scanErr := scanMemoryManifestWithScore(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan recalled memory manifest: %w", scanErr)
		}
		manifest = append(manifest, entry)
	}
	return manifest, rows.Err()
}

func (s *RunStore) recallMemoryManifestHybrid(ctx context.Context, run agent.Run, scopes, layers, types []string, projectKey, teamID, queryVector, queryModel, query string, minimum float64, limit int) ([]agent.MemoryManifestEntry, error) {
	candidateLimit := limit * 8
	if candidateLimit < 40 {
		candidateLimit = 40
	}
	rows, err := s.pool.Query(ctx, `
		WITH identity AS (
			SELECT definition.id AS agent_id, session.id AS session_id, session.user_id
			FROM agent_platform.agent_runs run
			JOIN agent_platform.agent_versions version ON version.id=run.agent_version_id
			JOIN agent_platform.agent_definitions definition ON definition.id=version.agent_id
			LEFT JOIN agent_platform.agent_sessions session ON session.id=run.session_id
			WHERE run.id=$1::uuid AND run.tenant_id=$2
		), eligible AS NOT MATERIALIZED (
			SELECT memory.id, memory.content, memory.body, memory.embedding, memory.embedding_model, memory.embedding_status,
				memory.content_hash, memory.title, memory.description, memory.source_layer,
				memory.semantic_type, memory.project_key, memory.updated_at, memory.last_verified_at,
				memory.freshness_class, memory.confidence, memory.importance, memory.pinned
			FROM agent_platform.agent_memories memory CROSS JOIN identity
			WHERE memory.tenant_id=$2 AND memory.deleted_at IS NULL AND memory.status='active'
			  AND (memory.expires_at IS NULL OR memory.expires_at>now())
			  AND memory.scope=ANY($3::text[])
			  AND (cardinality($4::text[])=0 OR memory.source_layer=ANY($4::text[]))
			  AND (cardinality($5::text[])=0 OR memory.semantic_type=ANY($5::text[]))
			  AND ((memory.scope='tenant')
			    OR (memory.scope='agent' AND memory.agent_id=identity.agent_id)
			    OR (memory.scope='user' AND memory.agent_id=identity.agent_id AND memory.user_id=identity.user_id)
			    OR (memory.scope='session' AND memory.session_id=identity.session_id))
			  AND (memory.source_layer <> 'team' OR EXISTS (
				SELECT 1 FROM agent_platform.memory_team_memberships membership
				WHERE membership.tenant_id=memory.tenant_id AND membership.team_id=memory.team_id
				  AND membership.user_id=identity.user_id AND membership.enabled
			  ))
			  AND ($12='' OR memory.project_key=$12)
			  AND ($13='' OR memory.team_id=$13)
		), candidates AS (
			(SELECT id FROM eligible WHERE embedding_status='ready' AND embedding_model=$7 ORDER BY embedding <=> $6::vector LIMIT $10)
			UNION
			(SELECT id FROM eligible WHERE similarity(lower(COALESCE(NULLIF(title||E'\n'||description,''),content)),lower($8))>=0.03 ORDER BY similarity(lower(COALESCE(NULLIF(title||E'\n'||description,''),content)),lower($8)) DESC LIMIT $10)
		), ranked AS (
			SELECT memory.*,
				GREATEST(0,1-(memory.embedding <=> $6::vector)) AS semantic_score,
				similarity(lower(COALESCE(NULLIF(memory.title||E'\n'||memory.description,''),memory.content)),lower($8)) AS lexical_score,
				(0.65*GREATEST(0,1-(memory.embedding <=> $6::vector)) +
				 0.25*similarity(lower(COALESCE(NULLIF(memory.title||E'\n'||memory.description,''),memory.content)),lower($8)) +
				 0.07*memory.importance +
				 0.03/(1+EXTRACT(EPOCH FROM (now()-memory.updated_at))/2592000.0)) AS recall_score
			FROM eligible memory JOIN candidates USING(id)
			WHERE memory.embedding_status='ready' AND memory.embedding_model=$7
		)
		SELECT `+memoryRouterManifestColumns+`, recall_score FROM ranked memory
		WHERE recall_score >= $9
		ORDER BY recall_score DESC,importance DESC,updated_at DESC LIMIT $11`,
		run.ID, run.TenantID, scopes, layers, types, queryVector, queryModel, query, minimum, candidateLimit, limit, projectKey, teamID)
	if err != nil {
		return nil, fmt.Errorf("hybrid recall memory manifest: %w", err)
	}
	defer rows.Close()
	manifest := make([]agent.MemoryManifestEntry, 0, limit)
	for rows.Next() {
		entry, scanErr := scanMemoryManifestWithScore(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan hybrid memory manifest: %w", scanErr)
		}
		manifest = append(manifest, entry)
	}
	return manifest, rows.Err()
}

// LoadMemoriesForRun loads full bodies only for IDs selected from a manifest.
// The SQL repeats Run visibility and policy filters so an ID cannot be used to
// bypass tenant, scope, layer or semantic-type isolation.
func (s *RunStore) LoadMemoriesForRun(ctx context.Context, run agent.Run, policy agent.MemoryPolicy, ids []string) ([]agent.Memory, error) {
	if len(ids) == 0 || !policy.Enabled {
		return []agent.Memory{}, nil
	}
	scopes := make([]string, 0, len(policy.ReadScopes))
	allowed := make(map[string]bool, len(policy.ReadScopes))
	for _, scope := range policy.ReadScopes {
		if validMemoryScopes[scope] {
			allowed[scope] = true
		}
	}
	for _, scope := range []string{"tenant", "agent", "user", "session"} {
		if allowed[scope] {
			scopes = append(scopes, scope)
		}
	}
	if len(scopes) == 0 {
		return []agent.Memory{}, nil
	}
	layers := make([]string, 0, len(policy.ReadLayers))
	for _, layer := range policy.ReadLayers {
		layer = strings.ToLower(strings.TrimSpace(layer))
		if agent.ValidMemoryLayer(layer) && (layer != agent.MemoryLayerTeam || policy.TeamMemoryEnabled) {
			layers = append(layers, layer)
		}
	}
	if len(policy.ReadLayers) == 0 {
		layers = []string{agent.MemoryLayerManaged, agent.MemoryLayerUser, agent.MemoryLayerProject, agent.MemoryLayerLocal, agent.MemoryLayerAuto}
		if policy.TeamMemoryEnabled {
			layers = append(layers, agent.MemoryLayerTeam)
		}
	}
	types := make([]string, 0, len(policy.ReadTypes))
	for _, semanticType := range policy.ReadTypes {
		semanticType = strings.ToLower(strings.TrimSpace(semanticType))
		if agent.ValidMemoryType(semanticType) {
			types = append(types, semanticType)
		}
	}
	rows, err := s.pool.Query(ctx, `
		WITH identity AS (
			SELECT definition.id AS agent_id, session.id AS session_id, session.user_id
			FROM agent_platform.agent_runs run
			JOIN agent_platform.agent_versions version ON version.id=run.agent_version_id
			JOIN agent_platform.agent_definitions definition ON definition.id=version.agent_id
			LEFT JOIN agent_platform.agent_sessions session ON session.id=run.session_id
			WHERE run.id=$1::uuid AND run.tenant_id=$2
		)
		SELECT `+memoryColumnsQualified+`
		FROM agent_platform.agent_memories memory CROSS JOIN identity
		WHERE memory.tenant_id=$2 AND memory.deleted_at IS NULL
		  AND memory.status='active'
		  AND (memory.expires_at IS NULL OR memory.expires_at>now())
		  AND memory.id::text=ANY($3::text[])
		  AND memory.scope=ANY($4::text[])
		  AND (cardinality($5::text[])=0 OR memory.source_layer=ANY($5::text[]))
		  AND (cardinality($6::text[])=0 OR memory.semantic_type=ANY($6::text[]))
		  AND ((memory.scope='tenant')
		    OR (memory.scope='agent' AND memory.agent_id=identity.agent_id)
		    OR (memory.scope='user' AND memory.agent_id=identity.agent_id AND memory.user_id=identity.user_id)
		    OR (memory.scope='session' AND memory.session_id=identity.session_id))
		  AND (memory.source_layer <> 'team' OR EXISTS (
			SELECT 1 FROM agent_platform.memory_team_memberships membership
			WHERE membership.tenant_id=memory.tenant_id AND membership.team_id=memory.team_id
			  AND membership.user_id=identity.user_id AND membership.enabled
		  ))
		  AND ($7='' OR memory.project_key=$7)
		  AND ($8='' OR memory.team_id=$8)
		ORDER BY array_position($3::text[], memory.id::text)`,
		run.ID, run.TenantID, ids, scopes, layers, types, strings.TrimSpace(policy.ProjectKey), strings.TrimSpace(policy.TeamID))
	if err != nil {
		return nil, fmt.Errorf("load selected memories: %w", err)
	}
	defer rows.Close()
	memories := make([]agent.Memory, 0, len(ids))
	for rows.Next() {
		memory, scanErr := scanMemory(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan selected memory: %w", scanErr)
		}
		memories = append(memories, memory)
	}
	return memories, rows.Err()
}

// RecordMemoryRetrieval persists the retrieval funnel without storing memory
// bodies. Query and IDs are supplied by the runtime after ACL and suppression
// decisions; the run/tenant pair remains the ownership boundary.
func (s *RunStore) RecordMemoryRetrieval(ctx context.Context, run agent.Run, audit agent.MemoryRetrievalAudit) error {
	if strings.TrimSpace(audit.TurnID) == "" {
		audit.TurnID = "1"
	}
	if strings.TrimSpace(audit.QueryHash) == "" {
		return errors.New("memory retrieval query_hash is required")
	}
	candidateValues := audit.CandidateIDs
	if candidateValues == nil {
		candidateValues = []string{}
	}
	routedValues := audit.RoutedIDs
	if routedValues == nil {
		routedValues = []string{}
	}
	injectedValues := audit.InjectedIDs
	if injectedValues == nil {
		injectedValues = []string{}
	}
	suppressedValues := audit.SuppressedIDs
	if suppressedValues == nil {
		suppressedValues = []string{}
	}
	scoreValues := audit.Scores
	if scoreValues == nil {
		scoreValues = map[string]float64{}
	}
	reasonValues := audit.SuppressionReasons
	if reasonValues == nil {
		reasonValues = map[string]string{}
	}
	candidateIDs, err := json.Marshal(candidateValues)
	if err != nil {
		return fmt.Errorf("encode memory retrieval candidate ids: %w", err)
	}
	routedIDs, err := json.Marshal(routedValues)
	if err != nil {
		return fmt.Errorf("encode memory retrieval routed ids: %w", err)
	}
	injectedIDs, err := json.Marshal(injectedValues)
	if err != nil {
		return fmt.Errorf("encode memory retrieval injected ids: %w", err)
	}
	suppressedIDs, err := json.Marshal(suppressedValues)
	if err != nil {
		return fmt.Errorf("encode memory retrieval suppressed ids: %w", err)
	}
	scores, err := json.Marshal(scoreValues)
	if err != nil {
		return fmt.Errorf("encode memory retrieval scores: %w", err)
	}
	reasons, err := json.Marshal(reasonValues)
	if err != nil {
		return fmt.Errorf("encode memory retrieval reasons: %w", err)
	}
	_, err = s.pool.Exec(ctx, `
		INSERT INTO agent_platform.memory_retrievals
			(tenant_id,run_id,turn_id,query_hash,candidate_ids,routed_ids,injected_ids,
			 suppressed_ids,scores,suppression_reasons,router_model,manifest_tokens,body_tokens,latency_ms)
		VALUES($1,$2::uuid,$3,$4,$5::jsonb,$6::jsonb,$7::jsonb,$8::jsonb,$9::jsonb,$10::jsonb,$11,$12,$13,$14)`,
		run.TenantID, run.ID, audit.TurnID, audit.QueryHash, candidateIDs, routedIDs, injectedIDs,
		suppressedIDs, scores, reasons, audit.RouterModel, audit.ManifestTokens, audit.BodyTokens, audit.LatencyMS)
	if err != nil {
		return fmt.Errorf("record memory retrieval: %w", err)
	}
	if err := s.recordMemoryRetrievalLifecycle(ctx, run, audit.TurnID, audit.QueryHash, routedValues, injectedValues, suppressedValues, reasonValues); err != nil {
		return err
	}
	return nil
}

func (s *RunStore) recordMemoryRetrievalLifecycle(ctx context.Context, run agent.Run, turnID, queryHash string, routedIDs, injectedIDs, suppressedIDs []string, reasons map[string]string) error {
	groups := []struct {
		eventType string
		ids       []string
	}{
		{eventType: "MEMORY_ROUTED", ids: routedIDs},
		{eventType: "MEMORY_INJECTED", ids: injectedIDs},
		{eventType: "MEMORY_SUPPRESSED", ids: suppressedIDs},
	}
	for _, group := range groups {
		for _, memoryID := range group.ids {
			memoryID = strings.TrimSpace(memoryID)
			if memoryID == "" {
				continue
			}
			payload := map[string]any{"turn_id": turnID, "query_hash": queryHash}
			if group.eventType == "MEMORY_SUPPRESSED" {
				payload["reason"] = reasons[memoryID]
			}
			encoded, err := json.Marshal(payload)
			if err != nil {
				return fmt.Errorf("encode memory retrieval lifecycle: %w", err)
			}
			_, err = s.RecordMemoryLifecycleEvent(ctx, agent.MemoryLifecycleEvent{
				TenantID: run.TenantID, MemoryID: optionalStringPointer(memoryID), RunID: &run.ID,
				EventType:      group.eventType,
				IdempotencyKey: optionalLifecycleKey("retrieval", run.ID+":"+turnID+":"+strings.ToLower(group.eventType)+":"+memoryID),
				Payload:        encoded,
			})
			if err != nil {
				return fmt.Errorf("record memory retrieval lifecycle %s: %w", group.eventType, err)
			}
		}
	}
	return nil
}

// ListMemoryRetrievals returns the bounded, body-free retrieval funnel for a
// Run. Tenant and Run are always applied together so the operator API cannot
// use a retrieval ID or Run ID to cross tenant boundaries.
func (s *RunStore) ListMemoryRetrievals(ctx context.Context, tenantID, runID string, limit int) ([]agent.MemoryRetrievalRecord, error) {
	if limit <= 0 || limit > 200 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `
		SELECT id::text,tenant_id,run_id::text,turn_id,query_hash,candidate_ids,
		       routed_ids,injected_ids,suppressed_ids,scores,suppression_reasons,
		       router_model,manifest_tokens,body_tokens,latency_ms,created_at
		FROM agent_platform.memory_retrievals
		WHERE tenant_id=$1 AND run_id=$2::uuid
		ORDER BY created_at DESC,id DESC LIMIT $3`, strings.TrimSpace(tenantID), strings.TrimSpace(runID), limit)
	if err != nil {
		return nil, fmt.Errorf("list memory retrievals: %w", err)
	}
	defer rows.Close()
	records := make([]agent.MemoryRetrievalRecord, 0)
	for rows.Next() {
		var record agent.MemoryRetrievalRecord
		var candidateIDs, routedIDs, injectedIDs, suppressedIDs json.RawMessage
		var scores, reasons json.RawMessage
		if scanErr := rows.Scan(&record.ID, &record.TenantID, &record.RunID, &record.TurnID, &record.QueryHash,
			&candidateIDs, &routedIDs, &injectedIDs, &suppressedIDs, &scores, &reasons, &record.RouterModel,
			&record.ManifestTokens, &record.BodyTokens, &record.LatencyMS, &record.CreatedAt); scanErr != nil {
			return nil, fmt.Errorf("scan memory retrieval: %w", scanErr)
		}
		if err := json.Unmarshal(candidateIDs, &record.CandidateIDs); err != nil {
			return nil, fmt.Errorf("decode memory retrieval candidate ids: %w", err)
		}
		if err := json.Unmarshal(routedIDs, &record.RoutedIDs); err != nil {
			return nil, fmt.Errorf("decode memory retrieval routed ids: %w", err)
		}
		if err := json.Unmarshal(injectedIDs, &record.InjectedIDs); err != nil {
			return nil, fmt.Errorf("decode memory retrieval injected ids: %w", err)
		}
		if err := json.Unmarshal(suppressedIDs, &record.SuppressedIDs); err != nil {
			return nil, fmt.Errorf("decode memory retrieval suppressed ids: %w", err)
		}
		if err := json.Unmarshal(scores, &record.Scores); err != nil {
			return nil, fmt.Errorf("decode memory retrieval scores: %w", err)
		}
		if err := json.Unmarshal(reasons, &record.SuppressionReasons); err != nil {
			return nil, fmt.Errorf("decode memory retrieval suppression reasons: %w", err)
		}
		records = append(records, record)
	}
	return records, rows.Err()
}

// EnqueueMemoryWriteJob creates an idempotent asynchronous extraction request.
// The unique constraint on (run, turn, trigger, input_hash) makes retries and
// Collapse re-entry safe without requiring Redis as a source of truth.
func (s *RunStore) EnqueueMemoryWriteJob(ctx context.Context, run agent.Run, turnID, trigger, inputHash string, sourceFrom, sourceTo int64) error {
	trigger = strings.ToLower(strings.TrimSpace(trigger))
	switch trigger {
	case "turn_complete", "collapse_barrier", "manual", "consolidation":
	default:
		return fmt.Errorf("unsupported memory write trigger %q", trigger)
	}
	if strings.TrimSpace(turnID) == "" || strings.TrimSpace(inputHash) == "" {
		return errors.New("memory write job turn_id and input_hash are required")
	}
	if sourceFrom < 0 || sourceTo < 0 || (sourceFrom > 0 && sourceTo > 0 && sourceFrom > sourceTo) {
		return fmt.Errorf("memory write job event range is invalid: %d..%d", sourceFrom, sourceTo)
	}
	_, err := s.pool.Exec(ctx, `
		INSERT INTO agent_platform.memory_write_jobs
			(tenant_id,run_id,session_id,turn_id,trigger,input_hash,source_event_from,source_event_to)
		VALUES($1,$2::uuid,NULLIF($3,'')::uuid,$4,$5,$6,NULLIF($7::bigint,0),NULLIF($8::bigint,0))
		ON CONFLICT (tenant_id,run_id,turn_id,trigger,input_hash) DO NOTHING`,
		run.TenantID, run.ID, sessionIDOf(run.SessionID), turnID, trigger, inputHash, sourceFrom, sourceTo)
	if err != nil {
		return fmt.Errorf("enqueue memory write job: %w", err)
	}
	return nil
}

// ClaimMemoryWriteJob leases one pending (or stale-running) extraction job.
// FOR UPDATE SKIP LOCKED allows multiple memory workers to share the queue
// without duplicate claims.
func (s *RunStore) ClaimMemoryWriteJob(ctx context.Context, leaseSeconds int64) (agent.MemoryWriteJob, bool, error) {
	if leaseSeconds <= 0 {
		leaseSeconds = 300
	}
	row := s.pool.QueryRow(ctx, `
		WITH candidate AS (
			SELECT id FROM agent_platform.memory_write_jobs
			WHERE (status='pending' AND available_at<=now())
			   OR (status='running' AND started_at IS NOT NULL AND started_at <= now()-($1::bigint*interval '1 second'))
			ORDER BY available_at ASC, created_at ASC
			FOR UPDATE SKIP LOCKED LIMIT 1
		)
		UPDATE agent_platform.memory_write_jobs job
		SET status='running', lease_token=gen_random_uuid(), attempt=job.attempt+1,
			started_at=now(), finished_at=NULL, last_error=NULL
		FROM candidate
		WHERE job.id=candidate.id
		RETURNING job.id::text,job.tenant_id,job.run_id::text,job.session_id::text,job.turn_id,
			job.trigger,job.status,job.attempt,job.lease_token::text,job.available_at,
			job.source_event_from,job.source_event_to,job.last_memory_write_sequence,
			job.input_hash,job.result_summary,job.last_error,job.created_at,job.started_at,job.finished_at`, leaseSeconds)
	var job agent.MemoryWriteJob
	err := scanMemoryWriteJob(row, &job)
	if errors.Is(err, pgx.ErrNoRows) {
		return agent.MemoryWriteJob{}, false, nil
	}
	if err != nil {
		return agent.MemoryWriteJob{}, false, fmt.Errorf("claim memory write job: %w", err)
	}
	return job, true, nil
}

func (s *RunStore) CompleteMemoryWriteJob(ctx context.Context, tenantID string, jobID, leaseToken string, resultSummary json.RawMessage) error {
	if len(resultSummary) == 0 {
		resultSummary = json.RawMessage(`{}`)
	}
	if !validJSONObject(resultSummary) {
		return errors.New("memory write job result_summary must be a JSON object")
	}
	command, err := s.pool.Exec(ctx, `
		UPDATE agent_platform.memory_write_jobs
		SET status='completed',finished_at=now(),lease_token=NULL,result_summary=$4::jsonb,last_error=NULL,
			last_memory_write_sequence=COALESCE(source_event_to,last_memory_write_sequence)
		WHERE id=$1::uuid AND tenant_id=$2 AND lease_token=$3::uuid AND status='running'`, jobID, tenantID, leaseToken, resultSummary)
	if err != nil {
		return fmt.Errorf("complete memory write job: %w", err)
	}
	if command.RowsAffected() == 0 {
		return agent.ErrLeaseLost
	}
	return nil
}

func (s *RunStore) FailMemoryWriteJob(ctx context.Context, tenantID string, jobID, leaseToken, message string, retryAt *time.Time, resultSummary json.RawMessage) error {
	if len(resultSummary) == 0 {
		resultSummary = json.RawMessage(`{}`)
	}
	if !validJSONObject(resultSummary) {
		return errors.New("memory write job result_summary must be a JSON object")
	}
	status := "failed"
	if retryAt != nil {
		status = "pending"
	}
	command, err := s.pool.Exec(ctx, `
		UPDATE agent_platform.memory_write_jobs
		SET status=$5::text,available_at=COALESCE($6::timestamptz,available_at),finished_at=CASE WHEN $5::text='failed' THEN now() ELSE NULL END,
			lease_token=NULL,last_error=$4,result_summary=$7::jsonb
		WHERE id=$1::uuid AND tenant_id=$2 AND lease_token=$3::uuid AND status='running'`, jobID, tenantID, leaseToken, message, status, retryAt, resultSummary)
	if err != nil {
		return fmt.Errorf("fail memory write job: %w", err)
	}
	if command.RowsAffected() == 0 {
		return agent.ErrLeaseLost
	}
	return nil
}

func (s *RunStore) LastMemoryWriteSequence(ctx context.Context, run agent.Run) (int64, error) {
	var sequence int64
	err := s.pool.QueryRow(ctx, `
		SELECT COALESCE(MAX(last_memory_write_sequence),0)
		FROM agent_platform.memory_write_jobs
		WHERE tenant_id=$1 AND run_id=$2::uuid AND status='completed'`, run.TenantID, run.ID).Scan(&sequence)
	if err != nil {
		return 0, fmt.Errorf("load last memory write sequence: %w", err)
	}
	return sequence, nil
}

func (s *RunStore) recallMemoriesLexical(ctx context.Context, run agent.Run, scopes, layers, types []string, projectKey, teamID, query string, minimum float64, limit int) ([]agent.Memory, error) {
	rows, err := s.pool.Query(ctx, `
		WITH identity AS (
			SELECT definition.id AS agent_id, session.id AS session_id, session.user_id
			FROM agent_platform.agent_runs run
			JOIN agent_platform.agent_versions version ON version.id=run.agent_version_id
			JOIN agent_platform.agent_definitions definition ON definition.id=version.agent_id
			LEFT JOIN agent_platform.agent_sessions session ON session.id=run.session_id
			WHERE run.id=$1::uuid AND run.tenant_id=$2
		), ranked AS (
			SELECT memory.*, similarity(lower(memory.content),lower($3)) AS lexical_score,
				(0.85*similarity(lower(memory.content),lower($3)) +
				 0.10*memory.importance +
				 0.05/(1+EXTRACT(EPOCH FROM (now()-memory.updated_at))/2592000.0)) AS recall_score
			FROM agent_platform.agent_memories memory CROSS JOIN identity
			WHERE memory.tenant_id=$2 AND memory.deleted_at IS NULL
			  AND memory.status='active'
			  AND (memory.expires_at IS NULL OR memory.expires_at>now())
			  AND memory.scope=ANY($4::text[])
			  AND (cardinality($5::text[])=0 OR memory.source_layer=ANY($5::text[]))
			  AND (cardinality($6::text[])=0 OR memory.semantic_type=ANY($6::text[]))
			  AND ((memory.scope='tenant')
			    OR (memory.scope='agent' AND memory.agent_id=identity.agent_id)
			    OR (memory.scope='user' AND memory.agent_id=identity.agent_id AND memory.user_id=identity.user_id)
			    OR (memory.scope='session' AND memory.session_id=identity.session_id))
			  AND (memory.source_layer <> 'team' OR EXISTS (
				SELECT 1 FROM agent_platform.memory_team_memberships membership
				WHERE membership.tenant_id=memory.tenant_id AND membership.team_id=memory.team_id
				  AND membership.user_id=identity.user_id AND membership.enabled
			  ))
			  AND ($9='' OR memory.project_key=$9)
			  AND ($10='' OR memory.team_id=$10)
		)
		SELECT `+memoryColumns+`, recall_score FROM ranked memory
		WHERE lexical_score >= 0.03 AND recall_score >= $7
		ORDER BY recall_score DESC, importance DESC, updated_at DESC LIMIT $8`,
		run.ID, run.TenantID, query, scopes, layers, types, minimum, limit, projectKey, teamID)
	if err != nil {
		return nil, fmt.Errorf("recall memories: %w", err)
	}
	defer rows.Close()
	memories := make([]agent.Memory, 0, limit)
	for rows.Next() {
		memory, scanErr := scanMemoryWithScore(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan recalled memory: %w", scanErr)
		}
		memories = append(memories, memory)
	}
	return memories, rows.Err()
}

func (s *RunStore) recallMemoriesHybrid(ctx context.Context, run agent.Run, scopes, layers, types []string, projectKey, teamID, queryVector, queryModel, query string, minimum float64, limit int) ([]agent.Memory, error) {
	candidateLimit := limit * 8
	if candidateLimit < 40 {
		candidateLimit = 40
	}
	rows, err := s.pool.Query(ctx, `
		WITH identity AS (
			SELECT definition.id AS agent_id, session.id AS session_id, session.user_id
			FROM agent_platform.agent_runs run
			JOIN agent_platform.agent_versions version ON version.id=run.agent_version_id
			JOIN agent_platform.agent_definitions definition ON definition.id=version.agent_id
			LEFT JOIN agent_platform.agent_sessions session ON session.id=run.session_id
			WHERE run.id=$1::uuid AND run.tenant_id=$2
		), eligible AS NOT MATERIALIZED (
			SELECT memory.* FROM agent_platform.agent_memories memory CROSS JOIN identity
			WHERE memory.tenant_id=$2 AND memory.deleted_at IS NULL
			  AND memory.status='active'
			  AND (memory.expires_at IS NULL OR memory.expires_at>now())
			  AND memory.scope=ANY($3::text[])
			  AND (cardinality($4::text[])=0 OR memory.source_layer=ANY($4::text[]))
			  AND (cardinality($5::text[])=0 OR memory.semantic_type=ANY($5::text[]))
			  AND ((memory.scope='tenant')
			    OR (memory.scope='agent' AND memory.agent_id=identity.agent_id)
			    OR (memory.scope='user' AND memory.agent_id=identity.agent_id AND memory.user_id=identity.user_id)
			    OR (memory.scope='session' AND memory.session_id=identity.session_id))
			  AND (memory.source_layer <> 'team' OR EXISTS (
				SELECT 1 FROM agent_platform.memory_team_memberships membership
				WHERE membership.tenant_id=memory.tenant_id AND membership.team_id=memory.team_id
				  AND membership.user_id=identity.user_id AND membership.enabled
			  ))
			  AND ($12='' OR memory.project_key=$12)
			  AND ($13='' OR memory.team_id=$13)
		), candidates AS (
			(SELECT id FROM eligible WHERE embedding_status='ready' AND embedding_model=$7 ORDER BY embedding <=> $6::vector LIMIT $10)
			UNION
			(SELECT id FROM eligible WHERE similarity(lower(content),lower($8))>=0.03 ORDER BY similarity(lower(content),lower($8)) DESC LIMIT $10)
		), ranked AS (
			SELECT memory.*,
				GREATEST(0,1-(memory.embedding <=> $6::vector)) AS semantic_score,
				similarity(lower(memory.content),lower($8)) AS lexical_score,
				(0.65*GREATEST(0,1-(memory.embedding <=> $6::vector)) +
				 0.25*similarity(lower(memory.content),lower($8)) +
				 0.07*memory.importance +
				 0.03/(1+EXTRACT(EPOCH FROM (now()-memory.updated_at))/2592000.0)) AS recall_score
			FROM eligible memory JOIN candidates USING(id)
			WHERE memory.embedding_status='ready' AND memory.embedding_model=$7
		)
		SELECT `+memoryColumns+`,recall_score FROM ranked memory
		WHERE recall_score >= $9
		ORDER BY recall_score DESC,importance DESC,updated_at DESC LIMIT $11`,
		run.ID, run.TenantID, scopes, layers, types, queryVector, queryModel, query, minimum, candidateLimit, limit, projectKey, teamID)
	if err != nil {
		return nil, fmt.Errorf("hybrid recall memories: %w", err)
	}
	defer rows.Close()
	memories := make([]agent.Memory, 0, limit)
	for rows.Next() {
		memory, scanErr := scanMemoryWithScore(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		memories = append(memories, memory)
	}
	return memories, rows.Err()
}

type MemoryEmbeddingReconcileResult struct{ Scanned, Embedded, Failed int }

func (s *RunStore) ReconcileMemoryEmbeddings(ctx context.Context, limit int) (MemoryEmbeddingReconcileResult, error) {
	if s.embeddings == nil || !s.embeddings.Enabled() {
		return MemoryEmbeddingReconcileResult{}, nil
	}
	if limit <= 0 || limit > 200 {
		limit = 50
	}
	activeModel := s.embeddings.Status().Model
	rows, err := s.pool.Query(ctx, `SELECT id::text,COALESCE(NULLIF(btrim(title||E'\n'||description||E'\n'||structured_data::text),''),body,content) FROM agent_platform.agent_memories WHERE deleted_at IS NULL AND status='active' AND (embedding_status<>'ready' OR ($2<>'' AND embedding_model<>$2)) ORDER BY updated_at,id LIMIT $1`, limit, activeModel)
	if err != nil {
		return MemoryEmbeddingReconcileResult{}, err
	}
	type item struct{ id, content string }
	var items []item
	for rows.Next() {
		var value item
		if err := rows.Scan(&value.id, &value.content); err != nil {
			rows.Close()
			return MemoryEmbeddingReconcileResult{}, err
		}
		items = append(items, value)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return MemoryEmbeddingReconcileResult{}, err
	}
	rows.Close()
	result := MemoryEmbeddingReconcileResult{Scanned: len(items)}
	texts := make([]string, len(items))
	for index := range items {
		texts[index] = items[index].content
	}
	embedded, embedErr := s.embeddings.Embed(ctx, texts, false)
	if embedErr != nil || len(embedded.Vectors) != len(items) {
		message := "embedding provider returned an incomplete batch"
		if embedErr != nil {
			message = embedErr.Error()
		}
		for _, value := range items {
			_, _ = s.pool.Exec(ctx, `UPDATE agent_platform.agent_memories SET embedding=NULL,embedding_model=NULL,embedding_status='failed',embedding_error=$2,embedded_at=NULL,updated_at=now() WHERE id=$1::uuid AND (embedding_status<>'ready' OR embedding_model<>$3)`, value.id, truncateStorageError(message), activeModel)
			result.Failed++
		}
		return result, nil
	}
	for index, value := range items {
		tag, updateErr := s.pool.Exec(ctx, `UPDATE agent_platform.agent_memories SET embedding=$2::vector,embedding_model=$3,embedding_status='ready',embedding_error=NULL,embedded_at=now(),updated_at=now() WHERE id=$1::uuid AND (embedding_status<>'ready' OR embedding_model<>$3)`, value.id, vectorLiteral(embedded.Vectors[index]), embedded.Model)
		if updateErr != nil {
			return result, updateErr
		}
		result.Embedded += int(tag.RowsAffected())
	}
	return result, nil
}

const memoryColumns = `
id::text,tenant_id,scope,agent_id::text,user_id,session_id::text,kind,content,
importance,source_run_id::text,metadata,expires_at,deleted_at,created_by,created_at,updated_at,
embedding_status,embedding_model,embedding_error,embedded_at,
source_id::text,source_layer,semantic_type,project_key,team_id,title,description,body,structured_data,status,
confidence,freshness_class,valid_from,valid_until,last_verified_at,verification_hint,canonical_key,content_hash,pinned,supersedes_id::text`

// memoryColumnsQualified is used when the memory row is joined with the Run
// identity CTE. Every column must be qualified because identity also exposes
// agent_id, user_id and session_id.
const memoryColumnsQualified = `
memory.id::text,memory.tenant_id,memory.scope,memory.agent_id::text,memory.user_id,memory.session_id::text,memory.kind,memory.content,
memory.importance,memory.source_run_id::text,memory.metadata,memory.expires_at,memory.deleted_at,memory.created_by,memory.created_at,memory.updated_at,
memory.embedding_status,memory.embedding_model,memory.embedding_error,memory.embedded_at,
memory.source_id::text,memory.source_layer,memory.semantic_type,memory.project_key,memory.team_id,memory.title,memory.description,memory.body,memory.structured_data,memory.status,
memory.confidence,memory.freshness_class,memory.valid_from,memory.valid_until,memory.last_verified_at,memory.verification_hint,memory.canonical_key,memory.content_hash,memory.pinned,memory.supersedes_id::text`

const memoryManifestColumns = `
id::text,content_hash,title,description,source_layer,semantic_type,project_key,
updated_at,last_verified_at,freshness_class,confidence,importance,pinned`

// Only the router path receives a bounded body prefix. ListMemoryManifest
// remains body-free for operator/catalog callers.
const memoryRouterManifestColumns = memoryManifestColumns + `,
left(COALESCE(NULLIF(memory.body,''), NULLIF(memory.content,'')), 1600)`

func scanMemory(row rowScanner) (agent.Memory, error) {
	var memory agent.Memory
	err := row.Scan(&memory.ID, &memory.TenantID, &memory.Scope, &memory.AgentID,
		&memory.UserID, &memory.SessionID, &memory.Kind, &memory.Content,
		&memory.Importance, &memory.SourceRunID, &memory.Metadata, &memory.ExpiresAt,
		&memory.DeletedAt, &memory.CreatedBy, &memory.CreatedAt, &memory.UpdatedAt,
		&memory.EmbeddingStatus, &memory.EmbeddingModel, &memory.EmbeddingError, &memory.EmbeddedAt,
		&memory.SourceID, &memory.SourceLayer, &memory.SemanticType, &memory.ProjectKey, &memory.TeamID,
		&memory.Title, &memory.Description, &memory.Body, &memory.StructuredData, &memory.Status,
		&memory.Confidence, &memory.FreshnessClass, &memory.ValidFrom, &memory.ValidUntil, &memory.LastVerifiedAt,
		&memory.VerificationHint, &memory.CanonicalKey, &memory.ContentHash, &memory.Pinned, &memory.SupersedesID)
	return memory, err
}

func scanMemoryWithScore(row rowScanner) (agent.Memory, error) {
	var memory agent.Memory
	err := row.Scan(&memory.ID, &memory.TenantID, &memory.Scope, &memory.AgentID,
		&memory.UserID, &memory.SessionID, &memory.Kind, &memory.Content,
		&memory.Importance, &memory.SourceRunID, &memory.Metadata, &memory.ExpiresAt,
		&memory.DeletedAt, &memory.CreatedBy, &memory.CreatedAt, &memory.UpdatedAt,
		&memory.EmbeddingStatus, &memory.EmbeddingModel, &memory.EmbeddingError, &memory.EmbeddedAt,
		&memory.SourceID, &memory.SourceLayer, &memory.SemanticType, &memory.ProjectKey, &memory.TeamID,
		&memory.Title, &memory.Description, &memory.Body, &memory.StructuredData, &memory.Status,
		&memory.Confidence, &memory.FreshnessClass, &memory.ValidFrom, &memory.ValidUntil, &memory.LastVerifiedAt,
		&memory.VerificationHint, &memory.CanonicalKey, &memory.ContentHash, &memory.Pinned, &memory.SupersedesID,
		&memory.RecallScore)
	return memory, err
}

func scanMemoryManifestWithScore(row rowScanner) (agent.MemoryManifestEntry, error) {
	var entry agent.MemoryManifestEntry
	err := row.Scan(&entry.ID, &entry.RevisionKey, &entry.Title, &entry.Description,
		&entry.SourceLayer, &entry.SemanticType, &entry.ProjectKey, &entry.UpdatedAt,
		&entry.LastVerifiedAt, &entry.FreshnessClass, &entry.Confidence, &entry.Importance,
		&entry.Pinned, &entry.Excerpt, &entry.RecallScore)
	return entry, err
}

func scanMemoryWriteJob(row rowScanner, job *agent.MemoryWriteJob) error {
	return row.Scan(&job.ID, &job.TenantID, &job.RunID, &job.SessionID, &job.TurnID,
		&job.Trigger, &job.Status, &job.Attempt, &job.LeaseToken, &job.AvailableAt,
		&job.SourceEventFrom, &job.SourceEventTo, &job.LastMemoryWriteSequence,
		&job.InputHash, &job.ResultSummary, &job.LastError, &job.CreatedAt,
		&job.StartedAt, &job.FinishedAt)
}

func vectorLiteral(vector []float32) string {
	var builder strings.Builder
	builder.Grow(len(vector) * 10)
	builder.WriteByte('[')
	for index, value := range vector {
		if index > 0 {
			builder.WriteByte(',')
		}
		builder.WriteString(strconv.FormatFloat(float64(value), 'g', -1, 32))
	}
	builder.WriteByte(']')
	return builder.String()
}

func truncateStorageError(value string) string {
	const limit = 1000
	if len(value) <= limit {
		return value
	}
	return value[:limit]
}
