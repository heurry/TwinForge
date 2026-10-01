// Package auth enforces the identity boundary for the standalone Agent API.
package auth

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"
)

type contextKey int

const identityKey contextKey = iota

// Identity is a verified caller. TenantID and Subject, never client-provided
// forwarding headers, become the effective ownership boundary.
type Identity struct {
	Subject  string `json:"sub"`
	Role     string `json:"role"`
	TenantID string `json:"tenant_id"`
}

type claims struct {
	Subject  string `json:"sub"`
	Role     string `json:"role"`
	TenantID string `json:"tenant_id"`
	Expires  int64  `json:"exp"`
}

// Config controls compatibility with the platform's HS256 login tokens.
// DefaultTenant is only a migration bridge for older tokens without a tenant claim.
type Config struct {
	Enabled       bool
	JWTSecret     string
	DefaultTenant string
}

// Middleware authenticates every Agent API route except health probes. When
// enabled it also applies coarse viewer/operator authorization and overwrites
// spoofable tenant/actor headers with verified claims.
func (c Config) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !c.Enabled || strings.HasPrefix(r.URL.Path, "/health/") || r.URL.Path == "/.well-known/agent-card.json" {
			next.ServeHTTP(w, r)
			return
		}
		identity, err := c.verifyBearer(r.Header.Get("Authorization"))
		if err != nil {
			writeError(w, http.StatusUnauthorized, "unauthenticated", "valid bearer token required")
			return
		}
		if identity.TenantID == "" {
			identity.TenantID = strings.TrimSpace(c.DefaultTenant)
		}
		if identity.TenantID == "" {
			writeError(w, http.StatusForbidden, "tenant_required", "authenticated identity has no tenant")
			return
		}
		if isWrite(r.Method) && identity.Role != "operator" && identity.Role != "admin" {
			writeError(w, http.StatusForbidden, "forbidden", "operator role required")
			return
		}
		r.Header.Set("X-Tenant-ID", identity.TenantID)
		r.Header.Set("X-Actor-ID", identity.Subject)
		ctx := context.WithValue(r.Context(), identityKey, identity)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func (c Config) verifyBearer(header string) (Identity, error) {
	if c.JWTSecret == "" || !strings.HasPrefix(header, "Bearer ") {
		return Identity{}, errors.New("missing credentials")
	}
	token := strings.TrimSpace(strings.TrimPrefix(header, "Bearer "))
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return Identity{}, errors.New("invalid token")
	}
	signingInput := parts[0] + "." + parts[1]
	mac := hmac.New(sha256.New, []byte(c.JWTSecret))
	_, _ = mac.Write([]byte(signingInput))
	want := base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
	if subtle.ConstantTimeCompare([]byte(want), []byte(parts[2])) != 1 {
		return Identity{}, errors.New("invalid token")
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return Identity{}, errors.New("invalid token")
	}
	var value claims
	if json.Unmarshal(payload, &value) != nil || value.Subject == "" || value.Expires <= time.Now().Unix() {
		return Identity{}, errors.New("invalid or expired token")
	}
	role := strings.ToLower(strings.TrimSpace(value.Role))
	if role != "viewer" && role != "operator" && role != "admin" {
		role = "viewer"
	}
	return Identity{Subject: value.Subject, Role: role, TenantID: strings.TrimSpace(value.TenantID)}, nil
}

func isWrite(method string) bool {
	return method == http.MethodPost || method == http.MethodPut || method == http.MethodPatch || method == http.MethodDelete
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{"code": code, "message": message}})
}
