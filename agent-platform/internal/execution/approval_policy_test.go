package execution

import (
	"encoding/json"
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

func TestSandboxCommandGrantDoesNotDisableHighRiskApproval(t *testing.T) {
	run := agent.Run{BindingSnapshot: json.RawMessage(`{"spec":{"approval":{"require_for":["HIGH_RISK"],"auto_approve_sandbox_command":true}}}`)}
	if !runRequiresApproval(run, tool.RiskHigh) {
		t.Fatal("HIGH_RISK approval must remain enabled for unrelated tools")
	}
	if !runAutoApprovesSandboxCommand(run) {
		t.Fatal("sandbox command-specific grant was not resolved")
	}

	legacy := agent.Run{BindingSnapshot: json.RawMessage(`{"spec":{"approval":{"require_for":["HIGH_RISK"]}}}`)}
	if runAutoApprovesSandboxCommand(legacy) {
		t.Fatal("legacy versions must not gain autonomous command permission")
	}
}
