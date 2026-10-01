package httptool

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/resource"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

func TestHTTPToolUsesIdempotencyAndEnvironmentHeader(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Idempotency-Key") != "run-1:call-1" || r.Header.Get("Authorization") != "Bearer test" {
			t.Errorf("headers = %+v", r.Header)
		}
		var body struct {
			Arguments map[string]int `json:"arguments"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Arguments["id"] != 7 {
			t.Errorf("request body = %+v, error = %v", body, err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"name":"record-7"}`))
	}))
	defer server.Close()

	spec := validHTTPSpec(server.URL)
	handler, err := NewHandler(spec, Config{
		AllowedHosts: []string{server.Listener.Addr().String()}, HTTPClient: server.Client(),
		LookupEnv: func(name string) (string, bool) {
			return "Bearer test", name == "TEST_TOOL_TOKEN"
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := handler(context.Background(), tool.Call{
		RunID: "run-1", ID: "call-1", Name: "lookup", Arguments: json.RawMessage(`{"id":7}`),
	})
	if err != nil || string(result.Content) != `{"name":"record-7"}` {
		t.Fatalf("result = %+v, error = %v", result, err)
	}
}

func TestHTTPToolRejectsUnapprovedHost(t *testing.T) {
	t.Parallel()
	if _, err := NewHandler(validHTTPSpec("http://metadata.internal/tool"), Config{AllowedHosts: []string{"api.internal"}}); err == nil {
		t.Fatal("unapproved endpoint must be rejected")
	}
}

func validHTTPSpec(endpoint string) resource.ToolSpec {
	return resource.ToolSpec{
		Definition: tool.Definition{
			Name: "lookup", Version: "1", Description: "look up one record",
			InputSchema: json.RawMessage(`{"type":"object"}`), Risk: tool.RiskRead,
			ExecutionMode: tool.ExecutionSerial,
		},
		ProviderType: "http",
		HTTP: &resource.HTTPProvider{
			Endpoint: endpoint, Method: "POST",
			HeaderEnvironment: map[string]string{"Authorization": "TEST_TOOL_TOKEN"},
		},
	}
}
