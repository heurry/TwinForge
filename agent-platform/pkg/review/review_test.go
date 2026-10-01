package review

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/contract"
)

func TestOutputSchemaAvoidsUnsupportedGuidedDecodingKeywords(t *testing.T) {
	if strings.Contains(string(OutputSchema()), "uniqueItems") {
		t.Fatal("Reviewer schema contains vLLM-incompatible uniqueItems")
	}
}

func TestOutputSchemaAndParserAgree(t *testing.T) {
	raw := json.RawMessage(`{"verdict":"changes_required","summary":"接口不一致","findings":[{"severity":"high","summary":"签名不一致","evidence":"db.py:20 的参数与调用方不同","path":"db.py","line":20,"recommendation":"统一接口"}],"recommended_plan_changes":[{"operation":"modify_step","step_id":"implement-db","description":"统一数据库接口","reason":"先统一接口"}]}`)
	if err := contract.Validate(OutputSchema(), raw); err != nil {
		t.Fatal(err)
	}
	result, err := Parse(raw)
	if err != nil || result.Verdict != VerdictChangesRequired || len(result.Findings) != 1 {
		t.Fatalf("result=%+v error=%v", result, err)
	}
}

func TestPassRejectsSeriousFinding(t *testing.T) {
	_, err := Parse(json.RawMessage(`{"verdict":"pass","summary":"ok","findings":[{"severity":"high","summary":"broken","evidence":"line 1"}],"recommended_plan_changes":[]}`))
	if err == nil {
		t.Fatal("pass verdict accepted a high-severity finding")
	}
}

func TestOutputSchemaAcceptsExecutableAddAndVerificationChanges(t *testing.T) {
	for _, raw := range []json.RawMessage{
		json.RawMessage(`{"verdict":"changes_required","summary":"missing step","findings":[{"severity":"medium","summary":"no verification phase","evidence":"plan ends after build"}],"recommended_plan_changes":[{"operation":"add_step","step_id":"verify","description":"verify output","reason":"missing phase","depends_on":["build"],"tool_hints":["read_file"],"acceptance_criteria":[{"id":"exists","description":"output exists","verification":{"kind":"file_exists","target":"output.bin"}}]}]}`),
		json.RawMessage(`{"verdict":"changes_required","summary":"wrong target","findings":[{"severity":"medium","summary":"target moved","evidence":"new.go exists"}],"recommended_plan_changes":[{"operation":"revise_verification","step_id":"build","criterion_id":"exists","reason":"target moved","verification":{"kind":"file_exists","target":"new.go"}}]}`),
	} {
		if err := contract.Validate(OutputSchema(), raw); err != nil {
			t.Fatalf("schema rejected %s: %v", raw, err)
		}
		if _, err := Parse(raw); err != nil {
			t.Fatalf("parser rejected %s: %v", raw, err)
		}
	}
}
