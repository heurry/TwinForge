package execution

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
)

func TestDiscoverModelUsesExplicitChatProbeForAIBrix(t *testing.T) {
	t.Parallel()
	var requested []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		var body struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		requested = append(requested, body.Model)
		if body.Model == "missing-model" {
			http.Error(w, `{"error":"model not found"}`, http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"live-model","choices":[{"message":{"role":"assistant","content":"pong"},"finish_reason":"length"}],"usage":{"total_tokens":1}}`))
	}))
	defer server.Close()

	resolver := &Resolver{
		modelServices: map[string]ModelService{
			"aibrix": {Endpoint: server.URL + "/v1", ContextWindowTokens: 4096, DiscoveryStrategy: "chat_probe", Capabilities: []string{"chat"}},
		},
		httpClient: server.Client(), lookupEnv: func(string) (string, bool) { return "", false },
	}
	resolution, err := resolver.discoverModel(context.Background(), agent.ModelBinding{
		Provider: "aibrix", ServiceRef: "aibrix", Capability: "chat",
		SelectionPolicy: agent.ModelSelectionAuto, ModelCandidates: []string{"missing-model", "live-model"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resolution.ModelID != "live-model" || len(requested) != 2 || requested[0] != "missing-model" {
		t.Fatalf("resolution = %+v, requested = %v", resolution, requested)
	}
}

func TestDiscoverModelUsesCandidateOrderAndMetadata(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"model":"preferred-model","choices":[{"message":{"role":"assistant","tool_calls":[{"id":"probe-1","type":"function","function":{"name":"health_probe","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"total_tokens":1}}`))
			return
		}
		if r.URL.Path != "/v1/models" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[
			{"id":"fallback-model"},
			{"id":"preferred-model","revision":"r17","digest":"sha256:abc","max_model_len":16384,"capabilities":["chat","tool_calling"]}
		]}`))
	}))
	defer server.Close()

	resolver := &Resolver{
		modelServices: map[string]ModelService{
			"primary": {Endpoint: server.URL + "/v1", Capabilities: []string{"chat", "tool_calling"}},
		},
		httpClient: server.Client(),
		lookupEnv:  func(string) (string, bool) { return "", false },
	}
	resolution, err := resolver.discoverModel(context.Background(), agent.ModelBinding{
		Provider: "vllm", ServiceRef: "primary", Capability: "tool_calling",
		SelectionPolicy: agent.ModelSelectionAuto,
		ModelCandidates: []string{"missing-model", "preferred-model", "fallback-model"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resolution.ModelID != "preferred-model" || resolution.ModelVersion != "r17" ||
		resolution.ArtifactDigest != "sha256:abc" || resolution.ContextWindowTokens != 16384 || resolution.ServiceConfigHash == "" {
		t.Fatalf("resolution = %+v", resolution)
	}
}

func TestDiscoverModelFallsBackAcrossConfiguredServices(t *testing.T) {
	t.Parallel()
	secondary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"data":[{"id":"target-model","max_model_len":8192}]}`))
	}))
	defer secondary.Close()

	resolver := &Resolver{
		modelServices: map[string]ModelService{
			"primary":   {Endpoint: "http://127.0.0.1:1/v1", Capabilities: []string{"chat"}},
			"secondary": {Endpoint: secondary.URL + "/v1", Capabilities: []string{"chat"}},
		},
		httpClient: secondary.Client(),
		lookupEnv:  func(string) (string, bool) { return "", false },
	}
	resolution, err := resolver.discoverModel(context.Background(), agent.ModelBinding{
		Provider: "openai-compatible", ServiceRef: "primary", ServiceCandidates: []string{"secondary"},
		Capability: "chat", SelectionPolicy: agent.ModelSelectionAuto,
		ModelCandidates: []string{"target-model"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resolution.ServiceRef != "secondary" || resolution.ModelID != "target-model" {
		t.Fatalf("resolution = %+v", resolution)
	}
}

func TestHydratePinnedModelMetadataReadsContextWindow(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"data":[{"id":"pinned-model","revision":"r2","max_model_len":24576}]}`))
	}))
	defer server.Close()
	resolver := &Resolver{
		modelServices: map[string]ModelService{"primary": {Endpoint: server.URL + "/v1"}},
		httpClient:    server.Client(), lookupEnv: func(string) (string, bool) { return "", false },
	}
	base, err := resolver.resolvePinnedModel(agent.ModelBinding{Provider: "vllm", ServiceRef: "primary", ModelID: "pinned-model"})
	if err != nil {
		t.Fatal(err)
	}
	resolution, err := resolver.hydratePinnedModelMetadata(context.Background(), agent.ModelBinding{}, base)
	if err != nil {
		t.Fatal(err)
	}
	if resolution.ContextWindowTokens != 24576 || resolution.ModelVersion != "r2" {
		t.Fatalf("resolution = %+v", resolution)
	}
}

func TestPinnedResolutionRejectsUndeclaredCapability(t *testing.T) {
	t.Parallel()
	resolver := &Resolver{modelServices: map[string]ModelService{
		"chat-only": {Endpoint: "http://model.invalid/v1", Capabilities: []string{"chat"}},
	}}
	_, err := resolver.resolvePinnedModel(agent.ModelBinding{
		Provider: "vllm", ServiceRef: "chat-only", Capability: "tool_calling",
		SelectionPolicy: agent.ModelSelectionPinned, ModelID: "model-a",
	})
	if err == nil {
		t.Fatal("undeclared tool_calling capability must be rejected")
	}
}
