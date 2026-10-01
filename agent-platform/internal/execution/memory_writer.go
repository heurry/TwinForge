package execution

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
)

// Persist implements the runtime MemoryCandidateWriter contract. Automatic
// extraction is intentionally restricted to Auto + session/tenant scope; it
// can never create Managed, Project, Local or Team records directly.
func (r *Resolver) Persist(ctx context.Context, job agent.MemoryWriteJob, candidates []agent.MemoryExtractionCandidate) (json.RawMessage, error) {
	if job.RunID == nil || strings.TrimSpace(*job.RunID) == "" {
		return nil, fmt.Errorf("memory write job has no run_id")
	}
	run, err := r.store.GetRun(ctx, *job.RunID)
	if err != nil {
		return nil, fmt.Errorf("load memory write run: %w", err)
	}
	if err := agent.ValidateMemoryExtractionCandidates(candidates); err != nil {
		return nil, err
	}
	var binding struct {
		Spec agent.Spec `json:"spec"`
	}
	if err := json.Unmarshal(run.BindingSnapshot, &binding); err != nil {
		return nil, fmt.Errorf("decode memory writer binding: %w", err)
	}
	projectKey := strings.TrimSpace(binding.Spec.Memory.ProjectKey)
	createdIDs := make([]string, 0, len(candidates))
	updatedIDs := make([]string, 0, len(candidates))
	reviewCount, ignoredCount := 0, 0
	for _, candidate := range candidates {
		scope, sessionID := automaticMemoryScope(binding.Spec.Memory, run, candidate)
		action := strings.ToLower(strings.TrimSpace(candidate.SuggestedAction))
		if action == "ignore" {
			ignoredCount++
			continue
		}
		structured := candidate.StructuredData
		if len(structured) == 0 {
			structured = json.RawMessage(`{}`)
		}
		canonicalKey := memoryCandidateCanonicalKey(candidate)
		_, found, findErr := r.store.FindActiveMemoryByCanonicalKey(ctx, run.TenantID, agent.MemoryLayerAuto, canonicalKey)
		if findErr != nil {
			return nil, findErr
		}
		matchedID := ""
		if candidate.MatchedMemoryID != nil {
			matchedID = strings.TrimSpace(*candidate.MatchedMemoryID)
		}
		// A matched update/merge is allowed to revise the existing identity even
		// when its canonical key is unchanged. Create/review candidates with an
		// existing key remain idempotent no-ops.
		if found && matchedID == "" {
			ignoredCount++
			continue
		}
		if matchedID != "" && (action == "update" || action == "merge") {
			updated, updateErr := r.store.UpdateAutoMemoryFromCandidate(ctx, run, matchedID, candidate, action, job.SourceEventFrom, job.SourceEventTo)
			if updateErr != nil {
				if updateErr == agent.ErrMemoryNotFound {
					reviewCount++
					continue
				}
				return nil, updateErr
			}
			updatedIDs = append(updatedIDs, updated.ID)
			continue
		}
		status := agent.MemoryStatusActive
		if action == "review" {
			status = agent.MemoryStatusReview
			reviewCount++
		}
		if (action == "update" || action == "merge" || action == "supersede") && (candidate.MatchedMemoryID == nil || strings.TrimSpace(*candidate.MatchedMemoryID) == "") {
			status = agent.MemoryStatusReview
			reviewCount++
		}
		created, createErr := r.store.CreateMemory(ctx, agent.CreateMemory{
			TenantID: run.TenantID, Scope: scope, SessionID: sessionID, Kind: "semantic",
			Content: candidate.Body, Importance: candidate.Importance, SourceRunID: &run.ID,
			SourceLayer: agent.MemoryLayerAuto, SemanticType: candidate.SemanticType,
			ProjectKey: optionalString(projectKey),
			Title:      candidate.Title, Description: candidate.Description, Body: candidate.Body,
			StructuredData: structured, Status: status, Confidence: candidate.Confidence,
			FreshnessClass: candidate.FreshnessClass, CanonicalKey: canonicalKey,
		})
		if createErr != nil {
			return nil, createErr
		}
		revisionReason := action
		if revisionReason == "review" {
			revisionReason = "create"
		}
		if revisionErr := r.store.RecordMemoryRevision(ctx, created, candidate, revisionReason, "memory-extractor", job.SourceEventFrom, job.SourceEventTo); revisionErr != nil {
			return nil, revisionErr
		}
		createdIDs = append(createdIDs, created.ID)
		if matchedID != "" && action == "supersede" {
			if supersedeErr := r.store.SupersedeMemory(ctx, run.TenantID, matchedID, created.ID); supersedeErr != nil && supersedeErr != agent.ErrMemoryNotFound {
				return nil, supersedeErr
			}
		}
	}
	result, _ := json.Marshal(map[string]any{"created_ids": createdIDs, "updated_ids": updatedIDs, "review_count": reviewCount, "ignored_count": ignoredCount})
	return result, nil
}

// automaticMemoryScope keeps ordinary conversation-derived facts scoped to
// the policy's default (normally the current Session), while promoting
// deterministic tool-contract feedback to the owning Agent. A schema contract
// such as write_file.content maxLength is reusable in another Session of the
// same Agent and must not disappear merely because the prior task ended.
func automaticMemoryScope(policy agent.MemoryPolicy, run agent.Run, candidate agent.MemoryExtractionCandidate) (string, *string) {
	if candidate.SemanticType == agent.MemoryTypeFeedback && isToolContractMemory(candidate) {
		return agent.MemoryScopeAgent, nil
	}
	scope := strings.ToLower(strings.TrimSpace(policy.WriteScope))
	if scope == "" {
		if run.SessionID != nil && strings.TrimSpace(*run.SessionID) != "" {
			scope = agent.MemoryScopeSession
		} else {
			scope = agent.MemoryScopeTenant
		}
	}
	if scope == agent.MemoryScopeSession && run.SessionID != nil && strings.TrimSpace(*run.SessionID) != "" {
		return scope, run.SessionID
	}
	return scope, nil
}

func isToolContractMemory(candidate agent.MemoryExtractionCandidate) bool {
	if len(candidate.StructuredData) == 0 {
		return false
	}
	var structured map[string]any
	if json.Unmarshal(candidate.StructuredData, &structured) != nil {
		return false
	}
	return strings.TrimSpace(stringValue(structured["tool_name"])) != "" &&
		(strings.TrimSpace(stringValue(structured["error_code"])) != "" ||
			strings.TrimSpace(stringValue(structured["field"])) != "" ||
			strings.TrimSpace(stringValue(structured["constraint"])) != "")
}

func memoryCandidateCanonicalKey(candidate agent.MemoryExtractionCandidate) string {
	// Tool-contract memories must be keyed by their stable coordinates rather
	// than by model prose. Otherwise each extraction can create a new memory
	// for the same rule (for example, several differently worded write_file
	// maxLength reminders), drowning exact recovery memories in duplicates.
	if len(candidate.StructuredData) != 0 {
		var structured map[string]any
		if json.Unmarshal(candidate.StructuredData, &structured) == nil {
			toolName := strings.ToLower(strings.TrimSpace(stringValue(structured["tool_name"])))
			errorCode := strings.ToLower(strings.TrimSpace(stringValue(structured["error_code"])))
			field := strings.ToLower(strings.TrimSpace(stringValue(structured["field"])))
			constraint := strings.ToLower(strings.TrimSpace(stringValue(structured["constraint"])))
			if toolName != "" && (errorCode != "" || field != "" || constraint != "") {
				value := strings.Join([]string{agent.MemoryLayerAuto, candidate.SemanticType, "tool-contract", toolName, errorCode, field, constraint}, "\n")
				digest := sha256.Sum256([]byte(value))
				return hex.EncodeToString(digest[:])
			}
		}
	}
	value := strings.ToLower(strings.Join([]string{agent.MemoryLayerAuto, candidate.SemanticType, strings.TrimSpace(candidate.Title), strings.TrimSpace(candidate.Description)}, "\n"))
	digest := sha256.Sum256([]byte(value))
	return hex.EncodeToString(digest[:])
}

func stringValue(value any) string {
	switch typed := value.(type) {
	case string:
		return typed
	case json.Number:
		return typed.String()
	default:
		return ""
	}
}

func optionalString(value string) *string {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	value = strings.TrimSpace(value)
	return &value
}
