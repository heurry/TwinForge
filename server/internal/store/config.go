package store

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/heurry/cloudnative-infra-platform/server/internal/db/sqlcgen"
)

// ConfigItems 返回全部配置项（含 version_count）。
func (s *Store) ConfigItems(ctx context.Context) ([]sqlcgen.ListConfigItemsRow, error) {
	return s.q.ListConfigItems(ctx)
}

// ConfigVersions 返回某配置项的版本列表 + 当前 active_version。
func (s *Store) ConfigVersions(ctx context.Context, id string) (int32, []sqlcgen.ListConfigVersionsRow, error) {
	active, err := s.q.GetConfigActiveVersion(ctx, id)
	if err != nil {
		return 0, nil, err
	}
	versions, err := s.q.ListConfigVersions(ctx, id)
	if err != nil {
		return 0, nil, err
	}
	return active, versions, nil
}

// CreateConfigItem 建项 + v1（事务），返回新 id。
func (s *Store) CreateConfigItem(ctx context.Context, env, namespace, configKey, configType, content, reason, operator string) (string, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return "", err
	}
	defer tx.Rollback(ctx)
	q := s.q.WithTx(tx)

	id, err := q.CreateConfigItem(ctx, sqlcgen.CreateConfigItemParams{
		Env: env, Namespace: ptr(namespace), ConfigKey: configKey,
		ConfigType: configType, CreatedBy: ptr(operator),
	})
	if err != nil {
		return "", err
	}
	if err := q.InsertConfigVersion(ctx, sqlcgen.InsertConfigVersionParams{
		ConfigItemID: id, Version: 1, Content: ptr(content),
		ChangeReason: ptr(reason), Operator: ptr(operator), Status: "active",
	}); err != nil {
		return "", err
	}
	meta, _ := json.Marshal(map[string]any{"env": env, "namespace": namespace, "version": 1})
	if err := q.InsertAuditEvent(ctx, sqlcgen.InsertAuditEventParams{
		ActorID: ptr(operator), ActorRole: ptr("operator"), Action: "config.create",
		ResourceType: ptr("config_item"), ResourceID: ptr(configKey), Metadata: meta,
	}); err != nil {
		return "", err
	}
	if err := tx.Commit(ctx); err != nil {
		return "", err
	}
	return id, nil
}

// PublishConfigVersion archive 全部→插入新 active 版本→更新 active_version（事务）。
// 返回新版本号与 config_key（供审计）。
func (s *Store) PublishConfigVersion(ctx context.Context, id, content, reason, operator string) (int32, string, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return 0, "", err
	}
	defer tx.Rollback(ctx)
	q := s.q.WithTx(tx)

	maxV, err := q.MaxConfigVersion(ctx, id)
	if err != nil {
		return 0, "", err
	}
	next := maxV + 1
	if err := q.ArchiveConfigVersions(ctx, id); err != nil {
		return 0, "", err
	}
	if err := q.InsertConfigVersion(ctx, sqlcgen.InsertConfigVersionParams{
		ConfigItemID: id, Version: next, Content: ptr(content),
		ChangeReason: ptr(reason), Operator: ptr(operator), Status: "active",
	}); err != nil {
		return 0, "", err
	}
	if err := q.SetConfigItemActiveVersion(ctx, sqlcgen.SetConfigItemActiveVersionParams{
		ActiveVersion: next, ID: id,
	}); err != nil {
		return 0, "", err
	}
	configKey, err := q.GetConfigKey(ctx, id)
	if err != nil {
		return 0, "", err
	}
	meta, _ := json.Marshal(map[string]any{"version": next, "reason": reason})
	if err := q.InsertAuditEvent(ctx, sqlcgen.InsertAuditEventParams{
		ActorID: ptr(operator), ActorRole: ptr("operator"), Action: "config.publish",
		ResourceType: ptr("config_item"), ResourceID: ptr(configKey), Metadata: meta,
	}); err != nil {
		return 0, "", err
	}
	if err := tx.Commit(ctx); err != nil {
		return 0, "", err
	}
	return next, configKey, nil
}

// ErrVersionNotFound 表示回滚目标版本不存在（handler 映射为 400）。
var ErrVersionNotFound = fmt.Errorf("version not found")

// RollbackConfigVersion 校验目标版本存在→archive 全部→激活目标→更新 active_version（事务）。
func (s *Store) RollbackConfigVersion(ctx context.Context, id string, target int32, operator string) (string, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return "", err
	}
	defer tx.Rollback(ctx)
	q := s.q.WithTx(tx)

	n, err := q.CountConfigVersion(ctx, sqlcgen.CountConfigVersionParams{ConfigItemID: id, Version: target})
	if err != nil {
		return "", err
	}
	if n == 0 {
		return "", ErrVersionNotFound
	}
	if err := q.ArchiveConfigVersions(ctx, id); err != nil {
		return "", err
	}
	if err := q.ActivateConfigVersion(ctx, sqlcgen.ActivateConfigVersionParams{ConfigItemID: id, Version: target}); err != nil {
		return "", err
	}
	if err := q.SetConfigItemActiveVersion(ctx, sqlcgen.SetConfigItemActiveVersionParams{ActiveVersion: target, ID: id}); err != nil {
		return "", err
	}
	configKey, err := q.GetConfigKey(ctx, id)
	if err != nil {
		return "", err
	}
	meta, _ := json.Marshal(map[string]any{"rolled_back_to": target})
	if err := q.InsertAuditEvent(ctx, sqlcgen.InsertAuditEventParams{
		ActorID: ptr(operator), ActorRole: ptr("operator"), Action: "config.rollback",
		ResourceType: ptr("config_item"), ResourceID: ptr(configKey), Metadata: meta,
	}); err != nil {
		return "", err
	}
	if err := tx.Commit(ctx); err != nil {
		return "", err
	}
	return configKey, nil
}
