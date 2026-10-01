package execution

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

// registerMemoryTools adds only policy-approved, bounded Memory operations to
// a Run-local registry. The handlers never expose Managed/Team writes and do
// not let a model mutate source layer, scope or canonical identity directly.
func registerMemoryTools(ctx context.Context, resolver *Resolver, run agent.Run, spec agent.Spec, registry *tool.Registry) error {
	if !spec.Memory.Enabled {
		return nil
	}
	writeLayer := strings.ToLower(strings.TrimSpace(spec.Memory.WriteLayer))
	if writeLayer == "" {
		writeLayer = agent.MemoryLayerAuto
	}
	if writeLayer == agent.MemoryLayerAuto {
		if err := registerRememberTool(resolver, run, spec, registry); err != nil {
			return err
		}
	}
	if err := registerForgetTool(resolver, run, spec, registry); err != nil {
		return err
	}
	if err := registerVerifyTool(resolver, run, registry); err != nil {
		return err
	}
	if spec.Memory.TeamMemoryEnabled && strings.ToLower(strings.TrimSpace(spec.Memory.TeamPromotion)) != "disabled" {
		if err := registerTeamProposalTool(resolver, run, spec, registry); err != nil {
			return err
		}
	}
	return nil
}

type rememberToolRequest struct {
	SemanticType      string          `json:"semantic_type"`
	Title             string          `json:"title"`
	Description       string          `json:"description"`
	Body              string          `json:"body"`
	StructuredPayload json.RawMessage `json:"structured_payload"`
	SourceMessageIDs  []string        `json:"source_message_ids"`
	Confidence        float64         `json:"confidence"`
	Importance        float64         `json:"importance"`
	FreshnessClass    string          `json:"freshness_class"`
}

func registerRememberTool(resolver *Resolver, run agent.Run, spec agent.Spec, registry *tool.Registry) error {
	definition := tool.Definition{
		Name: "remember", Version: "1", Risk: tool.RiskLowWrite, ExecutionMode: tool.ExecutionSerial,
		Description: "Use when the conversation establishes a reusable, non-secret fact, user preference, project decision, or verified tool-contract lesson that should affect a future Run. Do not store transient file state, raw tool output, credentials, one-off paths, or an unverified guess. Choose semantic_type=user, feedback, project, or reference; write a concise rule with why/how it applies and include exact source_message_ids. This writes only Auto Memory; it cannot modify Managed, Project, Local, or Team policy.",
		InputSchema: json.RawMessage(`{"type":"object","required":["semantic_type","title","description","body","source_message_ids"],"properties":{"semantic_type":{"enum":["user","feedback","project","reference"]},"title":{"type":"string","minLength":1,"maxLength":256},"description":{"type":"string","minLength":1,"maxLength":1200},"body":{"type":"string","minLength":1,"maxLength":32000},"structured_payload":{"type":"object"},"source_message_ids":{"type":"array","minItems":1,"maxItems":32,"items":{"type":"string","minLength":1,"maxLength":256}},"confidence":{"type":"number","minimum":0,"maximum":1},"importance":{"type":"number","minimum":0,"maximum":1},"freshness_class":{"enum":["stable","normal","volatile"]}},"additionalProperties":false}`),
	}
	return registry.Register(definition, func(callCtx context.Context, call tool.Call) (tool.Result, error) {
		var request rememberToolRequest
		if err := json.Unmarshal(call.Arguments, &request); err != nil {
			return tool.Result{}, fmt.Errorf("decode remember request: %w", err)
		}
		if strings.TrimSpace(request.FreshnessClass) == "" {
			request.FreshnessClass = agent.MemoryFreshnessNormal
		}
		if request.Confidence == 0 {
			request.Confidence = 0.8
		}
		if request.Importance == 0 {
			request.Importance = 0.6
		}
		structured := request.StructuredPayload
		if len(structured) == 0 {
			structured = json.RawMessage(`{}`)
		}
		candidate := agent.MemoryExtractionCandidate{
			SemanticType: request.SemanticType, Title: request.Title, Description: request.Description, Body: request.Body,
			StructuredData: structured, SourceMessageIDs: request.SourceMessageIDs, Confidence: request.Confidence,
			Importance: request.Importance, FreshnessClass: request.FreshnessClass, SuggestedAction: "create",
		}
		if err := agent.ValidateMemoryExtractionCandidates([]agent.MemoryExtractionCandidate{candidate}); err != nil {
			return tool.Result{}, err
		}
		scope := strings.ToLower(strings.TrimSpace(spec.Memory.WriteScope))
		if scope == "" {
			if run.SessionID != nil && strings.TrimSpace(*run.SessionID) != "" {
				scope = agent.MemoryScopeSession
			} else {
				scope = agent.MemoryScopeTenant
			}
		}
		if scope != agent.MemoryScopeSession && scope != agent.MemoryScopeTenant {
			return tool.Result{}, errors.New("remember only permits session or tenant scope")
		}
		createdBy := "agent:" + run.ID
		memory, err := resolver.store.CreateMemory(callCtx, agent.CreateMemory{
			TenantID: run.TenantID, Scope: scope, SessionID: run.SessionID, Kind: "semantic", Content: candidate.Body,
			Importance: candidate.Importance, SourceRunID: &run.ID, CreatedBy: &createdBy, SourceLayer: agent.MemoryLayerAuto,
			SemanticType: candidate.SemanticType, Title: candidate.Title, Description: candidate.Description, Body: candidate.Body,
			StructuredData: candidate.StructuredData, Confidence: candidate.Confidence, FreshnessClass: candidate.FreshnessClass,
			CanonicalKey: "",
		})
		if err != nil {
			return tool.Result{}, err
		}
		if err := resolver.store.RecordMemoryRevision(callCtx, memory, candidate, "create", "memory-tool", nil, nil); err != nil {
			return tool.Result{}, err
		}
		content, _ := json.Marshal(memory)
		return tool.Result{Content: content, Meta: map[string]string{"provider": "platform", "resource": "memory", "source_layer": agent.MemoryLayerAuto}}, nil
	})
}

type memoryIDToolRequest struct {
	MemoryID string `json:"memory_id"`
	Reason   string `json:"reason"`
}

func registerForgetTool(resolver *Resolver, run agent.Run, spec agent.Spec, registry *tool.Registry) error {
	definition := tool.Definition{
		Name: "forget_memory", Version: "1", Risk: tool.RiskLowWrite, ExecutionMode: tool.ExecutionSerial,
		Description: "Use only when a visible Auto Memory is demonstrably obsolete, incorrect, duplicated, or explicitly requested for removal. First verify the memory ID and explain the reason; do not use this to hide uncertainty or remove Managed, Project, Local, Team, or another Run's memory. This is a soft delete and does not erase the audit history.",
		InputSchema: json.RawMessage(`{"type":"object","required":["memory_id","reason"],"properties":{"memory_id":{"type":"string","minLength":1},"reason":{"type":"string","minLength":1,"maxLength":1000}},"additionalProperties":false}`),
	}
	return registry.Register(definition, func(callCtx context.Context, call tool.Call) (tool.Result, error) {
		var request memoryIDToolRequest
		if err := json.Unmarshal(call.Arguments, &request); err != nil {
			return tool.Result{}, err
		}
		memory, err := resolver.store.GetMemoryForTenant(callCtx, run.TenantID, strings.TrimSpace(request.MemoryID), "agent:"+run.ID)
		if err != nil {
			return tool.Result{}, err
		}
		if memory.SourceLayer != agent.MemoryLayerAuto {
			return tool.Result{}, errors.New("forget_memory only permits Auto Memory")
		}
		if memory.Scope == agent.MemoryScopeSession && (run.SessionID == nil || memory.SessionID == nil || *memory.SessionID != *run.SessionID) {
			return tool.Result{}, errors.New("memory does not belong to this Run session")
		}
		if err := resolver.store.DeleteMemory(callCtx, run.TenantID, memory.ID, "agent:"+run.ID); err != nil {
			return tool.Result{}, err
		}
		return tool.Result{Content: json.RawMessage(`{"deleted":true}`), Meta: map[string]string{"provider": "platform", "resource": "memory"}}, nil
	})
}

func registerVerifyTool(resolver *Resolver, run agent.Run, registry *tool.Registry) error {
	definition := tool.Definition{
		Name: "verify_memory", Version: "1", Risk: tool.RiskLowWrite, ExecutionMode: tool.ExecutionSerial,
		Description: "Use after checking the current workspace or project state and confirming that a visible Auto or reference Memory is still accurate. Provide the memory_id and the concrete verification reason. Do not verify from memory alone, and do not use this tool to rewrite the memory body; verification records freshness without changing content.",
		InputSchema: json.RawMessage(`{"type":"object","required":["memory_id","reason"],"properties":{"memory_id":{"type":"string","minLength":1},"reason":{"type":"string","minLength":1,"maxLength":1000}},"additionalProperties":false}`),
	}
	return registry.Register(definition, func(callCtx context.Context, call tool.Call) (tool.Result, error) {
		var request memoryIDToolRequest
		if err := json.Unmarshal(call.Arguments, &request); err != nil {
			return tool.Result{}, err
		}
		memory, err := resolver.store.GetMemoryForTenant(callCtx, run.TenantID, strings.TrimSpace(request.MemoryID), "agent:"+run.ID)
		if err != nil {
			return tool.Result{}, err
		}
		if memory.SourceLayer != agent.MemoryLayerAuto && memory.SemanticType != agent.MemoryTypeReference {
			return tool.Result{}, errors.New("verify_memory only permits Auto or reference Memory")
		}
		verified, err := resolver.store.VerifyMemory(callCtx, run.TenantID, memory.ID, "agent:"+run.ID)
		if err != nil {
			return tool.Result{}, err
		}
		content, _ := json.Marshal(verified)
		return tool.Result{Content: content, Meta: map[string]string{"provider": "platform", "resource": "memory"}}, nil
	})
}

type teamProposalToolRequest struct {
	MemoryID string `json:"memory_id"`
	TeamID   string `json:"team_id"`
	Reason   string `json:"reason"`
}

func registerTeamProposalTool(resolver *Resolver, run agent.Run, spec agent.Spec, registry *tool.Registry) error {
	definition := tool.Definition{
		Name: "propose_team_memory", Version: "1", Risk: tool.RiskLowWrite, ExecutionMode: tool.ExecutionSerial,
		Description: "Use when an Auto Memory appears broadly reusable for the team and should be reviewed for promotion. Include the visible memory_id, exact team_id, and a concise reason. This creates a reviewable proposal only; it never writes, promotes, or bypasses approval for Team memory.",
		InputSchema: json.RawMessage(`{"type":"object","required":["memory_id","team_id","reason"],"properties":{"memory_id":{"type":"string","minLength":1},"team_id":{"type":"string","minLength":1,"maxLength":128},"reason":{"type":"string","minLength":1,"maxLength":1000}},"additionalProperties":false}`),
	}
	return registry.Register(definition, func(callCtx context.Context, call tool.Call) (tool.Result, error) {
		var request teamProposalToolRequest
		if err := json.Unmarshal(call.Arguments, &request); err != nil {
			return tool.Result{}, err
		}
		memory, err := resolver.store.GetMemoryForTenant(callCtx, run.TenantID, strings.TrimSpace(request.MemoryID), "agent:"+run.ID)
		if err != nil {
			return tool.Result{}, err
		}
		if memory.SourceLayer != agent.MemoryLayerAuto {
			return tool.Result{}, errors.New("propose_team_memory only permits Auto Memory")
		}
		payload, _ := json.Marshal(map[string]any{"memory_id": memory.ID, "team_id": strings.TrimSpace(request.TeamID), "reason": strings.TrimSpace(request.Reason), "run_id": run.ID})
		if _, err := resolver.store.RecordMemoryLifecycleEvent(callCtx, agent.MemoryLifecycleEvent{
			TenantID: run.TenantID, MemoryID: &memory.ID, RunID: &run.ID, Actor: stringPtr("agent:" + run.ID),
			EventType: string(event.MemoryTeamPromotionRequested), IdempotencyKey: stringPtr("team-proposal:" + memory.ID + ":" + strings.TrimSpace(request.TeamID)), Payload: payload,
		}); err != nil {
			return tool.Result{}, err
		}
		return tool.Result{Content: json.RawMessage(`{"status":"review_required"}`), Meta: map[string]string{"provider": "platform", "resource": "memory_team_proposal"}}, nil
	})
}

func stringPtr(value string) *string { return &value }
