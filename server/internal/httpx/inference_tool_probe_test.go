package httpx

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestProbeToolCallingSendsToolSchema(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		if r.URL.Path != "/v1/chat/completions" || request["tool_choice"] != "auto" {
			t.Fatalf("path=%s request=%+v", r.URL.Path, request)
		}
		tools, _ := request["tools"].([]any)
		if len(tools) != 1 {
			t.Fatalf("tools=%+v", request["tools"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","tool_calls":[]}}]}`))
	}))
	defer server.Close()

	code, _, err := probeToolCalling(context.Background(), server.URL+"/v1", "model-a")
	if err != nil || code != http.StatusOK {
		t.Fatalf("code=%d err=%v", code, err)
	}
}

func TestProbeToolCallingReportsMissingParser(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, `{"error":"auto tool choice requires tool-call-parser"}`, http.StatusBadRequest)
	}))
	defer server.Close()

	code, detail, err := probeToolCalling(context.Background(), server.URL+"/v1", "model-a")
	if err == nil || code != http.StatusBadRequest || !strings.Contains(detail, "tool-call-parser") {
		t.Fatalf("code=%d detail=%q err=%v", code, detail, err)
	}
}
