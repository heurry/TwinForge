package resource

import (
	"bytes"
	"encoding/json"
	"testing"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

func TestHTTPProviderDurationJSON(t *testing.T) {
	t.Parallel()
	var provider HTTPProvider
	if err := json.Unmarshal([]byte(`{"endpoint":"http://tool.local/call","method":"POST","timeout":"30s"}`), &provider); err != nil {
		t.Fatal(err)
	}
	if provider.Timeout != 30*time.Second {
		t.Fatalf("timeout = %s", provider.Timeout)
	}
	encoded, err := json.Marshal(provider)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(encoded, []byte(`"timeout":"30s"`)) {
		t.Fatalf("encoded provider = %s", encoded)
	}
}

func TestWorkspaceCommandRequiresHighRisk(t *testing.T) {
	base := ToolSpec{
		Definition:   tool.Definition{Name: "run_command", Version: "1", Description: "run", InputSchema: json.RawMessage(`{"type":"object"}`), ExecutionMode: tool.ExecutionSerial},
		ProviderType: "workspace", Workspace: &WorkspaceProvider{Operation: "run_command"},
	}
	base.Definition.Risk = tool.RiskLowWrite
	if err := base.Validate(); err == nil {
		t.Fatal("expected low-write command tool to be rejected")
	}
	base.Definition.Risk = tool.RiskHigh
	if err := base.Validate(); err != nil {
		t.Fatalf("expected high-risk command tool to be valid: %v", err)
	}
}
