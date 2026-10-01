package contract

import (
	"encoding/json"
	"testing"
)

func TestCompileAndValidateDraft2020Schema(t *testing.T) {
	schema, err := Compile(json.RawMessage(`{"type":"object","required":["question"],"properties":{"question":{"type":"string","minLength":1}},"additionalProperties":false}`))
	if err != nil {
		t.Fatal(err)
	}
	if err := schema.ValidateJSON(json.RawMessage(`{"question":"hello"}`)); err != nil {
		t.Fatalf("valid instance rejected: %v", err)
	}
	if err := schema.ValidateJSON(json.RawMessage(`{"question":"","extra":true}`)); err == nil {
		t.Fatal("invalid instance must be rejected")
	}
}

func TestCompileRejectsInvalidSchema(t *testing.T) {
	if _, err := Compile(json.RawMessage(`{"type":"not-a-json-schema-type"}`)); err == nil {
		t.Fatal("invalid schema must be rejected")
	}
}
