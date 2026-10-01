package mcp

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestClientNegotiatesSessionListsAndCallsTools(t *testing.T) {
	var initialized, notified bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("MCP-Protocol-Version") != ProtocolVersion || r.Header.Get("Accept") != "application/json, text/event-stream" {
			t.Errorf("missing MCP transport headers: %v", r.Header)
		}
		var request struct {
			ID     int64           `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		switch request.Method {
		case "initialize":
			initialized = true
			w.Header().Set("Mcp-Session-Id", "session-1")
			_ = json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": request.ID, "result": map[string]any{"protocolVersion": ProtocolVersion}})
		case "notifications/initialized":
			if r.Header.Get("Mcp-Session-Id") != "session-1" {
				t.Error("initialized notification did not carry session")
			}
			notified = true
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			_ = json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": request.ID, "result": map[string]any{"tools": []any{map[string]any{"name": "lookup", "description": "lookup", "inputSchema": map[string]any{"type": "object"}}}}})
		case "tools/call":
			_ = json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": request.ID, "result": map[string]any{"structuredContent": map[string]any{"ok": true}}})
		default:
			t.Fatalf("unexpected method %q", request.Method)
		}
	}))
	defer server.Close()

	client, err := NewClient(ServerSpec{Transport: "streamable_http", Endpoint: server.URL, ProtocolVersion: ProtocolVersion}, []string{server.Listener.Addr().String()}, func(string) (string, bool) { return "", false }, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	tools, err := client.ListTools(context.Background())
	if err != nil || len(tools) != 1 || tools[0].SchemaHash == "" {
		t.Fatalf("tools=%+v err=%v", tools, err)
	}
	result, isError, err := client.CallTool(context.Background(), "lookup", json.RawMessage(`{"q":"x"}`))
	if err != nil || isError || string(result) != `{"ok":true}` {
		t.Fatalf("result=%s isError=%v err=%v", result, isError, err)
	}
	if !initialized || !notified {
		t.Fatalf("initialize=%v notified=%v", initialized, notified)
	}
}
