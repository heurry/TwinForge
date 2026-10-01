package taskplan

import (
	"context"
	"encoding/json"
	"strings"

	verificationdomain "github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/verification"
)

// contractCompiler adapts the current Task Plan contract to the standalone
// verification protocol. Adding a language/framework provider remains a
// registry operation; persistence and UI do not need a new branch.
type contractCompiler struct{}

func (contractCompiler) Key() string { return "taskplan-contract" }
func (contractCompiler) Match(intent verificationdomain.Intent) bool {
	_, exists := verificationProvider(strings.TrimSpace(intent.Kind))
	return exists
}
func (contractCompiler) Compile(_ context.Context, intent verificationdomain.Intent) (verificationdomain.ExecutableSpec, error) {
	var spec VerificationSpec
	if err := json.Unmarshal(intent.Parameters, &spec); err != nil {
		return verificationdomain.ExecutableSpec{}, err
	}
	if err := ValidateVerification(spec); err != nil {
		return verificationdomain.ExecutableSpec{}, err
	}
	subject, _ := json.Marshal(map[string]any{"kind": spec.Kind, "target": spec.Target, "match": spec.Match})
	execution, _ := json.Marshal(map[string]any{"tool": spec.Tool, "arguments": json.RawMessage(spec.Arguments), "evidence_tools": spec.EvidenceTools()})
	assertions, _ := json.Marshal(spec.Assertions)
	return verificationdomain.ExecutableSpec{
		ProviderKey: spec.Kind, ProviderVersion: "v1", Subject: subject,
		Execution: execution, Assertions: assertions, Status: "compiled",
	}, nil
}

func init() {
	if err := verificationdomain.DefaultRegistry.Register(contractCompiler{}); err != nil {
		panic(err)
	}
}
