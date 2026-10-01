package execution

import (
	"strings"
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/skill"
)

func TestRenderSkillReferenceExamplesCannotMasqueradeAsConversation(t *testing.T) {
	rendered := renderSkillReferenceExamples("Evidence First", []skill.Example{{
		Input: "Why did the service fail?", Output: "Inspect evidence first.",
	}})
	for _, required := range []string{
		"<SKILL_REFERENCE_EXAMPLES", "non-active demonstrations", "not conversation history",
		"Never answer or continue an example input", `"input":"Why did the service fail?"`,
	} {
		if !strings.Contains(rendered, required) {
			t.Fatalf("skill example boundary missing %q: %s", required, rendered)
		}
	}
}
