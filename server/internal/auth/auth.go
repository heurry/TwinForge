// Package auth 提供 D2 的认证授权基建：HS256 JWT（stdlib，无第三方依赖）+ 角色模型。
// 角色：viewer（只读）< operator（读 + 写操作）< admin（全部）。
// 设计：默认关闭（由 config.AuthEnabled 控制）；开启后读需任一登录态、写需 operator+。
package auth

import (
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

const (
	RoleViewer   = "viewer"
	RoleOperator = "operator"
	RoleAdmin    = "admin"
)

// ErrInvalidToken 统一的令牌无效错误（签名错/过期/格式坏）。
var ErrInvalidToken = errors.New("invalid token")

func roleRank(r string) int {
	switch r {
	case RoleAdmin:
		return 3
	case RoleOperator:
		return 2
	case RoleViewer:
		return 1
	default:
		return 0
	}
}

// CanWrite 判断角色是否可执行写操作（operator 及以上）。
func CanWrite(role string) bool { return roleRank(role) >= roleRank(RoleOperator) }

// IsAdmin 判断是否管理员。
func IsAdmin(role string) bool { return roleRank(role) >= roleRank(RoleAdmin) }

// normalizeRole 把任意输入归一到已知角色，未知一律降级 viewer（最小权限）。
func normalizeRole(r string) string {
	switch strings.ToLower(strings.TrimSpace(r)) {
	case RoleAdmin:
		return RoleAdmin
	case RoleOperator:
		return RoleOperator
	default:
		return RoleViewer
	}
}

// User 是认证后的主体（写进请求 context，供 authz + 审计用）。
type User struct {
	Subject  string `json:"sub"`
	Role     string `json:"role"`
	TenantID string `json:"tenant_id,omitempty"`
}

type claims struct {
	Sub      string `json:"sub"`
	Role     string `json:"role"`
	TenantID string `json:"tenant_id,omitempty"`
	Exp      int64  `json:"exp"`
	Iat      int64  `json:"iat"`
}

// Issuer 签发/校验 HS256 JWT。
type Issuer struct {
	secret []byte
	ttl    time.Duration
}

func NewIssuer(secret string, ttl time.Duration) *Issuer {
	if ttl <= 0 {
		ttl = 12 * time.Hour
	}
	return &Issuer{secret: []byte(secret), ttl: ttl}
}

func b64(b []byte) string { return base64.RawURLEncoding.EncodeToString(b) }

func (i *Issuer) sign(input string) string {
	m := hmac.New(sha256.New, i.secret)
	m.Write([]byte(input))
	return b64(m.Sum(nil))
}

// Issue 签发一个 subject+role 的 JWT，返回 token 与过期时刻。
func (i *Issuer) Issue(subject, role string) (string, time.Time) {
	return i.IssueForTenant(subject, role, "")
}

// IssueForTenant binds the authenticated identity to one tenant so downstream
// services never have to trust a client-supplied tenant forwarding header.
func (i *Issuer) IssueForTenant(subject, role, tenantID string) (string, time.Time) {
	now := time.Now()
	exp := now.Add(i.ttl)
	header := b64([]byte(`{"alg":"HS256","typ":"JWT"}`))
	payload, _ := json.Marshal(claims{Sub: subject, Role: normalizeRole(role), TenantID: strings.TrimSpace(tenantID), Exp: exp.Unix(), Iat: now.Unix()})
	signingInput := header + "." + b64(payload)
	return signingInput + "." + i.sign(signingInput), exp
}

// Verify 校验签名与过期，返回主体。
func (i *Issuer) Verify(token string) (*User, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return nil, ErrInvalidToken
	}
	signingInput := parts[0] + "." + parts[1]
	if subtle.ConstantTimeCompare([]byte(i.sign(signingInput)), []byte(parts[2])) != 1 {
		return nil, ErrInvalidToken
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, ErrInvalidToken
	}
	var c claims
	if err := json.Unmarshal(payload, &c); err != nil {
		return nil, ErrInvalidToken
	}
	if c.Sub == "" {
		return nil, ErrInvalidToken
	}
	if c.Exp > 0 && time.Now().Unix() > c.Exp {
		return nil, fmt.Errorf("%w: expired", ErrInvalidToken)
	}
	return &User{Subject: c.Sub, Role: normalizeRole(c.Role), TenantID: strings.TrimSpace(c.TenantID)}, nil
}
