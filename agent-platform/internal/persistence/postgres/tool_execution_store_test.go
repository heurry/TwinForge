package postgres

import (
	"encoding/json"
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

func TestHashToolRequestCanonicalizesArgumentOrder(t *testing.T) {
	first, _, err := hashToolRequest("tool-v1", tool.Call{Name: "run_command", Arguments: json.RawMessage(`{"command":"python3","args":["game.py"],"timeout_seconds":10}`)})
	if err != nil {
		t.Fatal(err)
	}
	second, _, err := hashToolRequest("tool-v1", tool.Call{Name: "run_command", Arguments: json.RawMessage(`{"timeout_seconds":10,"args":["game.py"],"command":"python3"}`)})
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Fatalf("semantic equivalents produced different hashes: %s != %s", first, second)
	}
}
