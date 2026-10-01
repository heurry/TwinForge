package auth

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestMiddlewareDerivesTenantAndActorAndGatesViewer(t *testing.T) {
	config := Config{Enabled: true, JWTSecret: "secret"}
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]string{
			"tenant": r.Header.Get("X-Tenant-ID"), "actor": r.Header.Get("X-Actor-ID"),
		})
	})
	handler := config.Middleware(next)

	request := httptest.NewRequest(http.MethodGet, "/api/v1/runs", nil)
	request.Header.Set("Authorization", "Bearer "+testToken("secret", "alice", "viewer", "tenant-a"))
	request.Header.Set("X-Tenant-ID", "spoofed")
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK || recorder.Body.String() != "{\"actor\":\"alice\",\"tenant\":\"tenant-a\"}\n" {
		t.Fatalf("derived identity status=%d body=%s", recorder.Code, recorder.Body.String())
	}

	write := httptest.NewRequest(http.MethodPost, "/api/v1/runs", nil)
	write.Header.Set("Authorization", "Bearer "+testToken("secret", "alice", "viewer", "tenant-a"))
	writeRecorder := httptest.NewRecorder()
	handler.ServeHTTP(writeRecorder, write)
	if writeRecorder.Code != http.StatusForbidden {
		t.Fatalf("viewer write status=%d", writeRecorder.Code)
	}
}

func TestMiddlewareRejectsMissingTokenAndTenant(t *testing.T) {
	handler := (Config{Enabled: true, JWTSecret: "secret"}).Middleware(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	missing := httptest.NewRecorder()
	handler.ServeHTTP(missing, httptest.NewRequest(http.MethodGet, "/api/v1/runs", nil))
	if missing.Code != http.StatusUnauthorized {
		t.Fatalf("missing token status=%d", missing.Code)
	}
	withoutTenant := httptest.NewRequest(http.MethodGet, "/api/v1/runs", nil)
	withoutTenant.Header.Set("Authorization", "Bearer "+testToken("secret", "alice", "operator", ""))
	noTenant := httptest.NewRecorder()
	handler.ServeHTTP(noTenant, withoutTenant)
	if noTenant.Code != http.StatusForbidden {
		t.Fatalf("missing tenant status=%d", noTenant.Code)
	}
}

func TestMiddlewareAllowsPublicA2AAgentCardDiscovery(t *testing.T) {
	called := false
	handler := (Config{Enabled: true, JWTSecret: "secret"}).Middleware(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	}))
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/.well-known/agent-card.json", nil))
	if recorder.Code != http.StatusOK || !called {
		t.Fatalf("well-known status=%d called=%v", recorder.Code, called)
	}
}

func testToken(secret, subject, role, tenant string) string {
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"HS256","typ":"JWT"}`))
	payload, _ := json.Marshal(claims{Subject: subject, Role: role, TenantID: tenant, Expires: time.Now().Add(time.Hour).Unix()})
	encodedPayload := base64.RawURLEncoding.EncodeToString(payload)
	input := header + "." + encodedPayload
	mac := hmac.New(sha256.New, []byte(secret))
	_, _ = mac.Write([]byte(input))
	return input + "." + base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}
