package objectstore

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func TestS3RoundTripAndBucketBootstrap(t *testing.T) {
	var mu sync.Mutex
	bucketReady := false
	objects := map[string][]byte{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.Header.Get("Authorization"), "AWS4-HMAC-SHA256 ") || r.Header.Get("X-Amz-Content-Sha256") == "" {
			http.Error(w, "missing signature", http.StatusForbidden)
			return
		}
		mu.Lock()
		defer mu.Unlock()
		if r.URL.Path == "/agent-artifacts" {
			if r.Method == http.MethodHead && !bucketReady {
				w.WriteHeader(http.StatusNotFound)
				return
			}
			if r.Method == http.MethodPut {
				bucketReady = true
				w.WriteHeader(http.StatusOK)
				return
			}
			w.WriteHeader(http.StatusOK)
			return
		}
		if !bucketReady {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		switch r.Method {
		case http.MethodPut:
			objects[r.URL.Path], _ = io.ReadAll(r.Body)
			w.WriteHeader(http.StatusOK)
		case http.MethodGet:
			value, ok := objects[r.URL.Path]
			if !ok {
				w.WriteHeader(http.StatusNotFound)
				return
			}
			_, _ = w.Write(value)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	}))
	defer server.Close()
	client, err := New(Config{Endpoint: server.URL, AccessKey: "access", SecretKey: "secret"})
	if err != nil {
		t.Fatal(err)
	}
	content := []byte("durable artifact")
	if err := client.Put(context.Background(), "sha256/ab/value", content, "text/plain"); err != nil {
		t.Fatal(err)
	}
	got, err := client.Get(context.Background(), "sha256/ab/value")
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(content) {
		t.Fatalf("got %q", got)
	}
	if err := client.Ping(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestDisabledS3IsExplicit(t *testing.T) {
	client, err := New(Config{})
	if err != nil {
		t.Fatal(err)
	}
	if client.Enabled() {
		t.Fatal("empty endpoint must be disabled")
	}
	if err := client.Ping(context.Background()); err != ErrDisabled {
		t.Fatalf("got %v", err)
	}
}
