// Package store 是领域层：包裹 sqlc 生成代码与 pgx 连接池，对外暴露领域方法、
// 处理多语句事务。handler 只调本层 + 做 DTO 映射，前端改线上格式时领域层不动。
// （云原生平台/11-go迁移计划.md Phase 2 分层决策）
package store

import (
	"context"
	"encoding/json"
	"log/slog"

	"github.com/heurry/cloudnative-infra-platform/server/internal/db/sqlcgen"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Store struct {
	pool *pgxpool.Pool
	q    *sqlcgen.Queries
}

func New(pool *pgxpool.Pool) *Store {
	return &Store{pool: pool, q: sqlcgen.New(pool)}
}

// Audit writes an audit event and surfaces persistence failures. Critical
// multi-statement mutations additionally write their audit row in-transaction.
func (s *Store) Audit(ctx context.Context, actorID, actorRole, action, resourceType, resourceID string, metadata map[string]any) error {
	meta, err := json.Marshal(metadata)
	if err != nil {
		meta = []byte("{}")
	}
	err = s.q.InsertAuditEvent(ctx, sqlcgen.InsertAuditEventParams{
		ActorID:      ptr(actorID),
		ActorRole:    ptr(actorRole),
		Action:       action,
		ResourceType: ptr(resourceType),
		ResourceID:   ptr(resourceID),
		Metadata:     meta,
	})
	if err != nil {
		slog.Error("audit persistence failed", "action", action, "resource_type", resourceType, "resource_id", resourceID, "err", err)
	}
	return err
}

func ptr(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}
