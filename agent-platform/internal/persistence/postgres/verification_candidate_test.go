package postgres

import (
	"encoding/json"
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/taskplan"
)

func TestSelectVerificationCandidateRejectsUnrelatedReceiptsWithoutAttempts(t *testing.T) {
	candidates := []successfulToolEvidence{
		{CallID: "db-edit", Name: "edit_file", Arguments: json.RawMessage(`{"path":"pkg/db.py"}`)},
		{CallID: "models-read", Name: "read_file", Arguments: json.RawMessage(`{"path":"pkg/models.py"}`)},
		{CallID: "target-read", Name: "read_file", Arguments: json.RawMessage(`{"path":"pkg/__init__.py"}`)},
		{CallID: "older-target-write", Name: "write_file", Arguments: json.RawMessage(`{"path":"pkg/__init__.py"}`)},
	}
	selected, ok, rejected := selectVerificationCandidate(taskplan.VerificationSpec{Kind: "file_exists", Target: "pkg/__init__.py"}, candidates)
	if !ok || selected.CallID != "target-read" || rejected != 2 {
		t.Fatalf("selected=%+v ok=%v rejected=%d", selected, ok, rejected)
	}
}

func TestSelectVerificationCandidateReturnsNoAttemptForUnrelatedReceipts(t *testing.T) {
	candidates := []successfulToolEvidence{
		{CallID: "db-edit", Name: "edit_file", Arguments: json.RawMessage(`{"path":"pkg/db.py"}`)},
		{CallID: "models-read", Name: "read_file", Arguments: json.RawMessage(`{"path":"pkg/models.py"}`)},
	}
	_, ok, rejected := selectVerificationCandidate(taskplan.VerificationSpec{Kind: "file_exists", Target: "pkg/__init__.py"}, candidates)
	if ok || rejected != len(candidates) {
		t.Fatalf("ok=%v rejected=%d", ok, rejected)
	}
}
