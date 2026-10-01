package skill

import (
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
)

func TestSkillValidation(t *testing.T) {
	valid := Spec{Instructions: []InstructionBlock{{Name: "diagnose", Content: "Collect evidence first."}}, RequiredTools: []agent.VersionRef{{ID: "tool-version", Version: "1"}}}
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid skill: %v", err)
	}
	if err := (Spec{}).Validate(); err == nil {
		t.Fatal("empty skill must fail")
	}
	if err := (SetSpec{Skills: []agent.VersionRef{{ID: "one", Version: "1"}, {ID: "one", Version: "1"}}}).Validate(); err == nil {
		t.Fatal("duplicate skill must fail")
	}
}
