package httpx

import (
	"context"
	"net/http"
	"strings"

	"github.com/heurry/cloudnative-infra-platform/server/internal/auth"
)

// D2：认证授权中间件 + 登录/whoami。默认关闭（a.AuthEnabled=false）时全透传，现有开放演示不受影响。
// 开启后：读（GET）需任一登录态，写（POST/PUT/PATCH/DELETE）需 operator+；/health 与 /auth/* 放行。

const userContextKey ctxKey = 100

func userFromContext(ctx context.Context) *auth.User {
	if u, ok := ctx.Value(userContextKey).(*auth.User); ok {
		return u
	}
	return nil
}

func isWriteMethod(m string) bool {
	switch m {
	case http.MethodPost, http.MethodPut, http.MethodPatch, http.MethodDelete:
		return true
	default:
		return false
	}
}

// Authn 解析 Bearer JWT，校验通过则把主体写入 context（始终透传——无/坏 token 也继续，由 Authz 决定是否拦）。
func (a *API) Authn(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if a.Auth != nil {
			if h := r.Header.Get("Authorization"); strings.HasPrefix(h, "Bearer ") {
				if u, err := a.Auth.Verify(strings.TrimSpace(strings.TrimPrefix(h, "Bearer "))); err == nil {
					next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), userContextKey, u)))
					return
				}
			}
		}
		next.ServeHTTP(w, r)
	})
}

// Authz 在 AuthEnabled 时按方法+角色门禁；关闭时透传。/health 与 /auth/* 永远放行。
func (a *API) Authz(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !a.AuthEnabled {
			next.ServeHTTP(w, r)
			return
		}
		p := r.URL.Path
		if strings.HasSuffix(p, "/health") || strings.Contains(p, "/auth/") {
			next.ServeHTTP(w, r)
			return
		}
		u := userFromContext(r.Context())
		if u == nil {
			WriteError(w, r, http.StatusUnauthorized, "unauthenticated", "authentication required")
			return
		}
		if isWriteMethod(r.Method) && !auth.CanWrite(u.Role) {
			WriteError(w, r, http.StatusForbidden, "forbidden", "operator role required for write operations")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// actor 返回审计用的真实操作者：认证态优先用 JWT 主体，否则回退请求体 operator / 默认。
func (a *API) actor(r *http.Request, fallback string) string {
	if u := userFromContext(r.Context()); u != nil {
		return u.Subject
	}
	return orDefault(fallback, defaultOperator)
}

// ===== /api/auth/* =====

type loginReq struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

// POST /api/auth/login —— 校验用户名口令，签发 JWT。
func (a *API) login(w http.ResponseWriter, r *http.Request) {
	if a.Auth == nil {
		WriteError(w, r, http.StatusServiceUnavailable, "auth_unavailable", "auth not configured")
		return
	}
	var req loginReq
	if err := decodeBody(r, &req); err != nil {
		a.badRequest(w, r, "invalid body")
		return
	}
	u, ok := auth.Authenticate(a.Users, req.Username, req.Password)
	if !ok {
		WriteError(w, r, http.StatusUnauthorized, "invalid_credentials", "用户名或口令错误")
		return
	}
	token, exp := a.Auth.IssueForTenant(u.Subject, u.Role, u.TenantID)
	a.Store.Audit(r.Context(), u.Subject, u.Role, "auth.login", "user", u.Subject, map[string]any{})
	WriteJSON(w, http.StatusOK, map[string]any{
		"token":      token,
		"subject":    u.Subject,
		"role":       u.Role,
		"tenant_id":  u.TenantID,
		"expires_at": exp.UTC().Format("2006-01-02T15:04:05Z07:00"),
	})
}

// GET /api/auth/me —— 当前身份与认证开关状态（始终 200，便于前端引导是否需要登录）。
func (a *API) me(w http.ResponseWriter, r *http.Request) {
	u := userFromContext(r.Context())
	resp := map[string]any{"auth_enabled": a.AuthEnabled, "authenticated": u != nil}
	if u != nil {
		resp["subject"] = u.Subject
		resp["role"] = u.Role
		resp["tenant_id"] = u.TenantID
	}
	WriteJSON(w, http.StatusOK, resp)
}
