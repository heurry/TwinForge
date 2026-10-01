package execution

import (
	"encoding/json"
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
)

func TestMemoryCandidateCanonicalKeyUsesToolContractCoordinates(t *testing.T) {
	structured := json.RawMessage(`{"tool_name":"write_file","error_code":"TOOL_SCHEMA_INVALID","field":"content","constraint":"maxLength 4096","rule":"keep chunks small"}`)
	a := agent.MemoryExtractionCandidate{
		SemanticType: "feedback", Title: "write_file content maxLength", Description: "Use bounded chunks", StructuredData: structured,
	}
	b := a
	b.Title = "File writing guidance"
	b.Description = "Split source files when necessary"
	if got, want := memoryCandidateCanonicalKey(a), memoryCandidateCanonicalKey(b); got != want {
		t.Fatalf("same tool contract produced different keys: %s != %s", got, want)
	}
	b.StructuredData = json.RawMessage(`{"tool_name":"write_file","error_code":"TOOL_SCHEMA_INVALID","field":"path","constraint":"required","rule":"include path"}`)
	if memoryCandidateCanonicalKey(a) == memoryCandidateCanonicalKey(b) {
		t.Fatal("different tool-contract fields must not share a canonical key")
	}
}

func TestAutomaticMemoryScopePromotesToolContractsToAgent(t *testing.T) {
	policy := agent.MemoryPolicy{WriteScope: agent.MemoryScopeSession}
	run := agent.Run{SessionID: stringPointer("session-1")}
	toolContract := agent.MemoryExtractionCandidate{SemanticType: agent.MemoryTypeFeedback, StructuredData: json.RawMessage(`{"tool_name":"write_file","error_code":"TOOL_SCHEMA_INVALID","field":"content","constraint":"maxLength 8192"}`)}
	scope, sessionID := automaticMemoryScope(policy, run, toolContract)
	if scope != agent.MemoryScopeAgent || sessionID != nil {
		t.Fatalf("tool-contract scope=%q session=%v, want agent/no-session", scope, sessionID)
	}
	normal := agent.MemoryExtractionCandidate{SemanticType: agent.MemoryTypeProject}
	scope, sessionID = automaticMemoryScope(policy, run, normal)
	if scope != agent.MemoryScopeSession || sessionID == nil || *sessionID != "session-1" {
		t.Fatalf("ordinary memory scope=%q session=%v, want session/session-1", scope, sessionID)
	}
}

func stringPointer(value string) *string { return &value }
