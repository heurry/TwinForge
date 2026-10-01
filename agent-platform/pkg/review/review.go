// Package review defines the platform-owned contract between a read-only
// Reviewer child Agent and the parent Workflow scheduler.
package review

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/taskplan"
)

const (
	VerdictPass            = "pass"
	VerdictChangesRequired = "changes_required"
	VerdictBlocked         = "blocked"
)

type Finding struct {
	Severity       string `json:"severity"`
	Summary        string `json:"summary"`
	Evidence       string `json:"evidence"`
	Path           string `json:"path,omitempty"`
	Line           int    `json:"line,omitempty"`
	Recommendation string `json:"recommendation,omitempty"`
}

type PlanChange struct {
	Operation          string                    `json:"operation"`
	StepID             string                    `json:"step_id,omitempty"`
	CriterionID        string                    `json:"criterion_id,omitempty"`
	Description        string                    `json:"description,omitempty"`
	Reason             string                    `json:"reason"`
	DependsOn          []string                  `json:"depends_on,omitempty"`
	ToolHints          []string                  `json:"tool_hints,omitempty"`
	AcceptanceCriteria []PlanCriterion           `json:"acceptance_criteria,omitempty"`
	Verification       taskplan.VerificationSpec `json:"verification,omitempty"`
}

type PlanCriterion struct {
	ID           string                    `json:"id"`
	Description  string                    `json:"description"`
	Verification taskplan.VerificationSpec `json:"verification"`
}

type Result struct {
	Verdict                string       `json:"verdict"`
	Summary                string       `json:"summary"`
	Findings               []Finding    `json:"findings"`
	RecommendedPlanChanges []PlanChange `json:"recommended_plan_changes"`
}

var outputSchema = json.RawMessage(`{
  "type":"object",
  "required":["verdict","summary","findings","recommended_plan_changes"],
  "properties":{
    "verdict":{"type":"string","enum":["pass","changes_required","blocked"]},
    "summary":{"type":"string","minLength":1,"maxLength":1200},
    "findings":{"type":"array","maxItems":12,"items":{
      "type":"object","required":["severity","summary","evidence"],
      "properties":{
        "severity":{"type":"string","enum":["critical","high","medium","low","info"]},
        "summary":{"type":"string","minLength":1,"maxLength":500},
        "evidence":{"type":"string","minLength":1,"maxLength":1200},
        "path":{"type":"string","maxLength":500},
        "line":{"type":"integer","minimum":1},
        "recommendation":{"type":"string","maxLength":1000}
      },"additionalProperties":false
    }},
    "recommended_plan_changes":{"type":"array","maxItems":8,"items":{
      "type":"object","required":["operation","reason"],
      "properties":{
        "operation":{"type":"string","enum":["add_step","modify_step","retire_step","revise_verification"]},
        "step_id":{"type":"string","maxLength":128},
        "criterion_id":{"type":"string","maxLength":128},
        "description":{"type":"string","maxLength":800},
        "reason":{"type":"string","minLength":1,"maxLength":800},
        "depends_on":{"type":"array","maxItems":16,"items":{"type":"string","maxLength":128}},
        "tool_hints":{"type":"array","maxItems":12,"uniqueItems":true,"items":{"type":"string","minLength":1,"maxLength":128}},
        "acceptance_criteria":{"type":"array","maxItems":8,"items":{
          "type":"object","required":["id","description","verification"],
          "properties":{
            "id":{"type":"string","minLength":1,"maxLength":128},
            "description":{"type":"string","minLength":1,"maxLength":1000},
            "verification":{"$ref":"#/$defs/verification"}
          },"additionalProperties":false
        }},
        "verification":{"$ref":"#/$defs/verification"}
      },"additionalProperties":false
    }}
  },
  "$defs":{"verification":{
    "type":"object","required":["kind"],
    "properties":{
      "kind":{"type":"string","minLength":1,"maxLength":128},
      "target":{"type":"string","maxLength":500},
      "match":{"type":"string","maxLength":500},
      "tool":{"type":"string","maxLength":128},
      "arguments":{"type":"object"},
      "assertions":{"type":"array","maxItems":16,"items":{
        "type":"object","required":["path","operator"],
        "properties":{"path":{"type":"string","minLength":1,"maxLength":256},"operator":{"type":"string","enum":["equals","contains","nonempty"]},"value":{}},
        "additionalProperties":false
      }}
    },"additionalProperties":false
  }},
  "additionalProperties":false
}`)

func OutputSchema() json.RawMessage {
	return append(json.RawMessage(nil), outputSchema...)
}

func Parse(raw json.RawMessage) (Result, error) {
	var result Result
	if len(raw) == 0 {
		return result, errors.New("Reviewer output is empty")
	}
	if err := json.Unmarshal(raw, &result); err != nil {
		return result, fmt.Errorf("decode Reviewer output: %w", err)
	}
	if err := result.Validate(); err != nil {
		return result, err
	}
	return result, nil
}

func (r Result) Validate() error {
	switch r.Verdict {
	case VerdictPass, VerdictChangesRequired, VerdictBlocked:
	default:
		return fmt.Errorf("unsupported Reviewer verdict %q", r.Verdict)
	}
	if strings.TrimSpace(r.Summary) == "" {
		return errors.New("Reviewer summary is required")
	}
	if len(r.Findings) > 12 || len(r.RecommendedPlanChanges) > 8 {
		return errors.New("Reviewer output exceeds bounded collection limits")
	}
	if r.Verdict == VerdictPass && len(r.Findings) != 0 {
		for _, finding := range r.Findings {
			if finding.Severity != "info" && finding.Severity != "low" {
				return errors.New("pass verdict cannot contain medium, high, or critical findings")
			}
		}
	}
	if r.Verdict == VerdictChangesRequired && len(r.Findings) == 0 {
		return errors.New("changes_required verdict needs at least one finding")
	}
	for index, finding := range r.Findings {
		switch finding.Severity {
		case "critical", "high", "medium", "low", "info":
		default:
			return fmt.Errorf("finding %d has unsupported severity %q", index, finding.Severity)
		}
		if strings.TrimSpace(finding.Summary) == "" || strings.TrimSpace(finding.Evidence) == "" {
			return fmt.Errorf("finding %d requires summary and evidence", index)
		}
	}
	for index, change := range r.RecommendedPlanChanges {
		switch change.Operation {
		case "add_step", "modify_step", "retire_step", "revise_verification":
		default:
			return fmt.Errorf("recommended Plan change %d has unsupported operation %q", index, change.Operation)
		}
		if strings.TrimSpace(change.Reason) == "" {
			return fmt.Errorf("recommended Plan change %d requires a reason", index)
		}
		stepID := strings.TrimSpace(change.StepID)
		switch change.Operation {
		case "add_step":
			if stepID == "" || strings.TrimSpace(change.Description) == "" || len(change.AcceptanceCriteria) == 0 {
				return fmt.Errorf("recommended add_step %d requires step_id, description, and acceptance_criteria", index)
			}
			for criterionIndex, criterion := range change.AcceptanceCriteria {
				if strings.TrimSpace(criterion.ID) == "" || strings.TrimSpace(criterion.Description) == "" {
					return fmt.Errorf("recommended add_step %d criterion %d requires id and description", index, criterionIndex)
				}
				if err := criterion.Verification.Validate(); err != nil {
					return fmt.Errorf("recommended add_step %d criterion %d: %w", index, criterionIndex, err)
				}
			}
		case "modify_step":
			if stepID == "" || (strings.TrimSpace(change.Description) == "" && change.DependsOn == nil && change.ToolHints == nil) {
				return fmt.Errorf("recommended modify_step %d requires step_id and at least one changed field", index)
			}
		case "retire_step":
			if stepID == "" {
				return fmt.Errorf("recommended retire_step %d requires step_id", index)
			}
		case "revise_verification":
			if stepID == "" || strings.TrimSpace(change.CriterionID) == "" {
				return fmt.Errorf("recommended revise_verification %d requires step_id and criterion_id", index)
			}
			if err := change.Verification.Validate(); err != nil {
				return fmt.Errorf("recommended revise_verification %d: %w", index, err)
			}
		}
	}
	return nil
}
