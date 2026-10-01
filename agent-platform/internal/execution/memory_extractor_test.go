package execution

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
)

func TestBuildMemoryExtractionMessagesKeepsSystemAtBeginning(t *testing.T) {
	messages := buildMemoryExtractionMessages([]string{
		"Existing memory manifest: [{\"id\":\"m-1\"}]",
		"Structured tool failure evidence: update_plan schema correction",
		"[assistant message]\nThe tool failed",
	})
	if len(messages) != 2 {
		t.Fatalf("messages=%d, want one system and one evidence message", len(messages))
	}
	if messages[0].Role != model.RoleSystem || messages[1].Role != model.RoleUser {
		t.Fatalf("roles=%q,%q; system instruction must be first", messages[0].Role, messages[1].Role)
	}
	if messages[0].TextContent() == "" || messages[1].TextContent() == "" {
		t.Fatal("extraction instruction/evidence must not be empty")
	}
}

func TestMemoryExtractorInstructionKeepsExactToolConstraints(t *testing.T) {
	for _, required := range []string{
		"exact tool name, error_code, constrained field",
		"write_file or append_file content exceeding 8192",
		"matched_memory_id",
	} {
		if !strings.Contains(memoryExtractorInstruction, required) {
			t.Fatalf("instruction missing %q", required)
		}
	}
}

func TestToolFailureEvidenceKeepsSchemaErrorAndGroupsStableContract(t *testing.T) {
	makeFailure := func(errorText string) event.Event {
		payload, err := json.Marshal(map[string]any{
			"name": "write_file",
			"result": map[string]any{
				"error": errorText,
				"meta":  map[string]any{"error_code": "TOOL_SCHEMA_INVALID"},
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		return event.Event{Input: event.Input{Type: event.ToolFailed, Payload: payload}}
	}
	evidence := collectToolFailureEvidence([]event.Event{
		makeFailure("schema validation failed: '/content' does not validate with maxLength: length must be <= 8192, but got 9213"),
		makeFailure("schema validation failed: '/content' does not validate with maxLength: length must be <= 8192, but got 10258"),
	})
	if len(evidence) != 1 || evidence[0].Count != 2 {
		t.Fatalf("evidence=%+v, want one grouped maxLength contract", evidence)
	}
	if !strings.Contains(summarizeToolFailureEvidence(evidence), "8192") {
		t.Fatalf("schema error was omitted from extraction evidence: %s", summarizeToolFailureEvidence(evidence))
	}
}

func TestMergeDeterministicToolFailureCandidatesAddsExactContracts(t *testing.T) {
	raw := json.RawMessage(`{"candidates":[]}`)
	merged, err := mergeDeterministicToolFailureCandidates(raw, []toolFailureEvidence{
		{Tool: "write_file", ErrorCode: "TOOL_SCHEMA_INVALID", Error: "missing properties: 'path'", Count: 1},
		{Tool: "write_file", ErrorCode: "TOOL_SCHEMA_INVALID", Error: "'/content' maxLength: length must be <= 8192", Count: 3},
		{Tool: "run_command", ErrorCode: "TOOL_SCHEMA_INVALID", Error: "missing properties: 'args'", Count: 1},
		{Tool: "run_command", ErrorCode: "DETERMINISTIC_RETRY_BLOCKED", Error: "workspace has not changed", Count: 1},
		{Tool: "revise_verification", ErrorCode: "TOOL_SCHEMA_INVALID", Error: "additionalProperties 'tool_hints' not allowed", Count: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	var envelope struct {
		Candidates []agent.MemoryExtractionCandidate `json:"candidates"`
	}
	if err := json.Unmarshal(merged, &envelope); err != nil {
		t.Fatal(err)
	}
	if len(envelope.Candidates) != 5 {
		t.Fatalf("candidate count=%d, want 5: %s", len(envelope.Candidates), merged)
	}
	if err := agent.ValidateMemoryExtractionCandidates(envelope.Candidates); err != nil {
		t.Fatalf("deterministic candidates must pass normal validation: %v", err)
	}
	for _, candidate := range envelope.Candidates {
		if candidate.SemanticType != agent.MemoryTypeFeedback || candidate.SuggestedAction != "create" {
			t.Fatalf("invalid deterministic candidate: %+v", candidate)
		}
	}
}
