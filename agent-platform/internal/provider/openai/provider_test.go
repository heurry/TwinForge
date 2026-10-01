package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
)

func TestCompleteMapsToolCallsAndUsage(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer secret" {
			t.Errorf("authorization = %q", r.Header.Get("Authorization"))
		}
		var request chatRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if request.Model != "qwen" || len(request.Tools) != 1 || len(request.Messages) != 2 {
			t.Errorf("request = %+v", request)
		}
		if request.ToolChoice != "auto" {
			t.Errorf("tool_choice = %q, want auto", request.ToolChoice)
		}
		if request.ChatTemplateKwargs == nil || request.ChatTemplateKwargs.EnableThinking {
			t.Errorf("chat_template_kwargs = %+v", request.ChatTemplateKwargs)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"model":"qwen-runtime","choices":[{"finish_reason":"tool_calls","message":{
				"role":"assistant","tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{\"id\":7}"}}]
			}}],"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14}
		}`))
	}))
	defer server.Close()

	provider, err := New(Config{
		Endpoint: server.URL, APIKey: "secret", Model: "qwen",
		DisableThinking: true, HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	response, err := provider.Complete(context.Background(), model.Request{
		Messages: []model.Message{
			{Role: model.RoleUser, Content: "find 7"},
			{Role: model.RoleTool, ToolCallID: "previous", Content: `{"ok":true}`},
		},
		Tools: []model.ToolSchema{{Name: "lookup", Parameters: json.RawMessage(`{"type":"object"}`)}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if response.ModelID != "qwen-runtime" || response.Usage.TotalTokens != 14 || len(response.Message.ToolCalls) != 1 {
		t.Fatalf("response = %+v", response)
	}
	call := response.Message.ToolCalls[0]
	if call.ID != "call-1" || call.Name != "lookup" || string(call.Arguments) != `{"id":7}` {
		t.Fatalf("tool call = %+v", call)
	}
}

func TestCompleteExplicitlyDisablesToolsWhenNoneAreOffered(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request chatRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		if request.ToolChoice != "none" || len(request.Tools) != 0 {
			t.Fatalf("request tool policy = %q tools=%d", request.ToolChoice, len(request.Tools))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"qwen","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"done"}}]}`))
	}))
	defer server.Close()
	provider, err := New(Config{Endpoint: server.URL, Model: "qwen", HTTPClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := provider.Complete(context.Background(), model.Request{Messages: []model.Message{{Role: model.RoleUser, Content: "finish"}}}); err != nil {
		t.Fatal(err)
	}
}

func TestCompleteRemovesUnsupportedGrammarKeywordsFromToolProjection(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request chatRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		if request.ToolChoice != "required" || len(request.Tools) != 1 {
			t.Fatalf("request tool policy = %q tools=%d", request.ToolChoice, len(request.Tools))
		}
		var projected map[string]any
		if err := json.Unmarshal(request.Tools[0].Function.Parameters, &projected); err != nil {
			t.Fatal(err)
		}
		encoded, _ := json.Marshal(projected)
		if strings.Contains(string(encoded), "uniqueItems") {
			t.Fatalf("unsupported keyword reached provider: %s", encoded)
		}
		properties := projected["properties"].(map[string]any)
		items := properties["tool_hints"].(map[string]any)
		if items["maxItems"] != float64(12) {
			t.Fatalf("supported constraints were lost: %#v", items)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"qwen","choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[{"id":"plan-1","type":"function","function":{"name":"update_plan","arguments":"{}"}}]}}]}`))
	}))
	defer server.Close()
	provider, err := New(Config{Endpoint: server.URL, Model: "qwen", HTTPClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	original := json.RawMessage(`{"type":"object","properties":{"tool_hints":{"type":"array","maxItems":12,"uniqueItems":true,"items":{"type":"string"}}}}`)
	if _, err := provider.Complete(context.Background(), model.Request{
		Messages:   []model.Message{{Role: model.RoleUser, Content: "replan"}},
		Tools:      []model.ToolSchema{{Name: "update_plan", Parameters: original}},
		ToolChoice: "required",
	}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(original), "uniqueItems") {
		t.Fatal("provider mutated the authoritative Runtime schema")
	}
}

func TestNewPreservesCompleteEndpoint(t *testing.T) {
	t.Parallel()
	provider, err := New(Config{Endpoint: "http://model.local/v1/chat/completions", Model: "qwen"})
	if err != nil {
		t.Fatal(err)
	}
	if provider.endpoint != "http://model.local/v1/chat/completions" {
		t.Fatalf("endpoint = %q", provider.endpoint)
	}
}

func TestCompleteRejectsFabricatedToolReceipt(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"qwen","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"{\"status\":\"completed\",\"receipt\":{\"tool_name\":\"run_command\"}}"}}]}`))
	}))
	defer server.Close()
	provider, err := New(Config{Endpoint: server.URL, Model: "qwen", HTTPClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	_, err = provider.Complete(context.Background(), model.Request{Messages: []model.Message{{Role: model.RoleUser, Content: "run it"}}})
	if err == nil || !strings.Contains(err.Error(), "fabricated tool receipt") {
		t.Fatalf("error = %v", err)
	}
}

func TestCompleteAdaptsLegacyToolMarkupAtProviderBoundary(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"qwen","choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":"<toolcall><function=runcommand><parameter=command>python3 -c \\\"print(1)\\\"</parameter></function></toolcall>"}}]}`))
	}))
	defer server.Close()
	provider, err := New(Config{Endpoint: server.URL, Model: "qwen", HTTPClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	response, err := provider.Complete(context.Background(), model.Request{Messages: []model.Message{{Role: model.RoleUser, Content: "run"}}, Tools: []model.ToolSchema{{Name: "run_command", Parameters: json.RawMessage(`{"type":"object"}`)}}})
	if err != nil || len(response.Message.ToolCalls) != 1 || response.Message.ToolCalls[0].Name != "run_command" {
		t.Fatalf("adapted response=%+v err=%v", response, err)
	}
}
