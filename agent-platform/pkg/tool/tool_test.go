package tool

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestRegistryIsDeterministicAndRejectsDuplicates(t *testing.T) {
	t.Parallel()

	registry := NewRegistry()
	for _, name := range []string{"z.last", "a.first"} {
		if err := registry.Register(Definition{
			Name: name, Version: "1", InputSchema: json.RawMessage(`{"type":"object"}`),
			Risk: RiskRead, ExecutionMode: ExecutionSerial,
		}, func(context.Context, Call) (Result, error) {
			return Result{Content: json.RawMessage(`{"ok":true}`)}, nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	definitions := registry.Definitions()
	if definitions[0].Name != "a.first" || definitions[1].Name != "z.last" {
		t.Fatalf("definitions are not sorted: %+v", definitions)
	}
	if err := registry.Register(definitions[0], func(context.Context, Call) (Result, error) {
		return Result{}, nil
	}); err == nil {
		t.Fatal("duplicate tool name must be rejected")
	}
}

func TestRegistryRejectsInvalidArguments(t *testing.T) {
	t.Parallel()

	registry := NewRegistry()
	if err := registry.Register(Definition{
		Name: "read", Version: "1", InputSchema: json.RawMessage(`{"type":"object"}`),
		Risk: RiskRead, ExecutionMode: ExecutionSerial,
	}, func(context.Context, Call) (Result, error) {
		return Result{}, nil
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Execute(context.Background(), Call{Name: "read", Arguments: json.RawMessage(`{`)}); err == nil {
		t.Fatal("invalid JSON arguments must be rejected")
	}
}

func TestRegistryNormalizesHandlerFailureForModelRepair(t *testing.T) {
	t.Parallel()
	registry := NewRegistry()
	if err := registry.Register(Definition{
		Name: "read_file", Version: "1", InputSchema: json.RawMessage(`{"type":"object"}`),
		Risk: RiskRead, ExecutionMode: ExecutionSerial,
	}, func(context.Context, Call) (Result, error) {
		return Result{}, errors.New("open workspace file: no such file or directory")
	}); err != nil {
		t.Fatal(err)
	}
	_, err := registry.Execute(context.Background(), Call{Name: "read_file", Arguments: json.RawMessage(`{}`)})
	contract, ok := AsContractError(err)
	if !ok || contract.Code != "PATH_NOT_FOUND" || !contract.Retryable || contract.Correction == "" || FailureKind(contract.Code) != "path_not_found" {
		t.Fatalf("unstructured handler error: %#v", err)
	}
}

func TestNormalizeExecutionErrorDoesNotRelabelSchemaAsSandbox(t *testing.T) {
	err := NormalizeExecutionError(Call{Name: "edit_file"}, errors.New(`inner (TOOL_SCHEMA_INVALID): additionalProperties "content" not allowed`))
	contract, ok := AsContractError(err)
	if !ok || contract.Code != "TOOL_SCHEMA_INVALID" || FailureKind(contract.Code) != "schema_invalid" {
		t.Fatalf("normalized=%#v", err)
	}
}

func TestRegistryNormalizesResultFailureForModelRepair(t *testing.T) {
	t.Parallel()
	registry := NewRegistry()
	if err := registry.Register(Definition{
		Name: "run_command", Version: "1", InputSchema: json.RawMessage(`{"type":"object"}`),
		Risk: RiskHigh, ExecutionMode: ExecutionSerial,
	}, func(context.Context, Call) (Result, error) {
		return Result{
			Content: json.RawMessage(`{"exit_code":1,"stderr":"broken"}`),
			IsError: true,
			Error:   "command exited with code 1",
			Meta:    map[string]string{"provider": "workspace"},
		}, nil
	}); err != nil {
		t.Fatal(err)
	}
	result, err := registry.Execute(context.Background(), Call{Name: "run_command", Arguments: json.RawMessage(`{}`)})
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(result.Content, &payload); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"error_code", "retryable", "correction", "retry_template"} {
		if _, ok := payload[key]; !ok {
			t.Fatalf("normalized result missing %q: %s", key, result.Content)
		}
	}
	if payload["error_code"] != "TOOL_EXECUTION_FAILED" || payload["retryable"] != true {
		t.Fatalf("unexpected normalized failure: %s", result.Content)
	}
}

func TestRegistryEnforcesInputAndOutputSchemas(t *testing.T) {
	t.Parallel()
	registry := NewRegistry()
	err := registry.Register(Definition{
		Name: "lookup", Version: "1", Description: "lookup",
		InputSchema:  json.RawMessage(`{"type":"object","required":["id"],"properties":{"id":{"type":"string"}},"additionalProperties":false}`),
		OutputSchema: json.RawMessage(`{"type":"object","required":["found"],"properties":{"found":{"type":"boolean"}},"additionalProperties":false}`),
		Risk:         RiskRead, ExecutionMode: ExecutionSerial,
	}, func(context.Context, Call) (Result, error) {
		return Result{Content: json.RawMessage(`{"unexpected":true}`)}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Execute(context.Background(), Call{Name: "lookup", Arguments: json.RawMessage(`{"extra":true}`)}); err == nil {
		t.Fatal("invalid tool arguments must be rejected")
	}
	if _, err := registry.Execute(context.Background(), Call{Name: "lookup", Arguments: json.RawMessage(`{"id":"42"}`)}); err == nil {
		t.Fatal("invalid tool result must be rejected")
	}
}

func TestDeniedToolIsHiddenAndNeverExecutes(t *testing.T) {
	t.Parallel()
	registry := NewRegistry()
	executed := false
	if err := registry.Register(Definition{
		Name: "destroy", Version: "1", InputSchema: json.RawMessage(`{"type":"object"}`),
		Risk: RiskDenied, ExecutionMode: ExecutionSerial,
	}, func(context.Context, Call) (Result, error) {
		executed = true
		return Result{}, nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(registry.Definitions()) != 0 {
		t.Fatal("denied tool must not be exposed to the model")
	}
	if _, err := registry.Execute(context.Background(), Call{Name: "destroy", Arguments: json.RawMessage(`{}`)}); !errors.Is(err, ErrDenied) {
		t.Fatalf("denied execution error=%v", err)
	}
	if executed {
		t.Fatal("denied tool handler must never execute")
	}
}

func TestLargeResultUsesBoundedModelProjectionWithoutDiscardingContent(t *testing.T) {
	content := json.RawMessage(`{"content":"` + strings.Repeat("x", InlineResultLimit) + `"}`)
	result := Result{Content: content, Meta: map[string]string{"content_artifact_id": "artifact-1"}}
	result.ApplyStoredModelProjection()
	if len(result.Content) != len(content) {
		t.Fatalf("durable content was changed: got=%d want=%d", len(result.Content), len(content))
	}
	visible := result.ModelVisible()
	if len(visible.Content) >= len(content) || visible.ModelContent != nil {
		t.Fatalf("model projection was not bounded: visible=%d durable=%d", len(visible.Content), len(content))
	}
	var receipt map[string]any
	if err := json.Unmarshal(visible.Content, &receipt); err != nil {
		t.Fatalf("receipt is not JSON: %v", err)
	}
	if receipt["artifact_id"] != "artifact-1" || receipt["status"] != "offloaded" {
		t.Fatalf("unexpected receipt: %s", visible.Content)
	}
	encoded, err := json.Marshal(result)
	if err != nil || string(encoded) == "" || string(encoded) == string(visible.Content) {
		t.Fatalf("durable result JSON was not preserved: %s (err=%v)", encoded, err)
	}
}
