package embedding

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestLiveEmbeddingContractAndStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddings":[[0.5,0.5]],"model":"test-embed","dim":2,"mode":"live"}`))
	}))
	defer server.Close()
	client := New(server.URL, 2, true, nil)
	result, err := client.Embed(context.Background(), []string{"memory"}, false)
	if err != nil {
		t.Fatal(err)
	}
	if result.Model != "test-embed" || len(result.Vectors[0]) != 2 {
		t.Fatalf("unexpected result: %+v", result)
	}
	if status := client.Status(); !status.Ready || status.Mode != "live" {
		t.Fatalf("unexpected status: %+v", status)
	}
}

func TestStubEmbeddingDoesNotMasqueradeAsProductionSemanticMemory(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddings":[[0.5,0.5]],"model":"test-embed","dim":2,"mode":"stub"}`))
	}))
	defer server.Close()
	client := New(server.URL, 2, true, nil)
	if _, err := client.Embed(context.Background(), []string{"memory"}, false); err == nil {
		t.Fatal("expected stub rejection")
	}
	if status := client.Status(); status.Ready || status.Mode != "stub" || status.LastError == "" {
		t.Fatalf("unexpected status: %+v", status)
	}
}
