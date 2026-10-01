package postgres

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/mcp"
)

func (s *RunStore) CreateMCPServerVersion(ctx context.Context, input mcp.CreateServerVersion) (mcp.ServerVersion, error) {
	raw, err := json.Marshal(input.Spec)
	if err != nil {
		return mcp.ServerVersion{}, err
	}
	sum := sha256.Sum256(raw)
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return mcp.ServerVersion{}, err
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	var definitionID string
	err = tx.QueryRow(ctx, `
		INSERT INTO agent_platform.mcp_server_definitions(tenant_id,server_key,name)
		VALUES($1::text,$2::text,$3::text)
		ON CONFLICT(tenant_id,server_key) DO UPDATE SET name=EXCLUDED.name,updated_at=now()
		RETURNING id::text`, input.TenantID, input.Key, input.Name).Scan(&definitionID)
	if err != nil {
		return mcp.ServerVersion{}, err
	}
	var item mcp.ServerVersion
	err = tx.QueryRow(ctx, `
		INSERT INTO agent_platform.mcp_server_versions(definition_id,version,spec,spec_hash,created_by)
		SELECT $1::uuid,COALESCE(max(version),0)+1,$2::jsonb,$3::text,$4::text
		FROM agent_platform.mcp_server_versions WHERE definition_id=$1::uuid
		RETURNING id::text,definition_id::text,
		 (SELECT tenant_id FROM agent_platform.mcp_server_definitions WHERE id=$1::uuid),
		 (SELECT server_key FROM agent_platform.mcp_server_definitions WHERE id=$1::uuid),
		 (SELECT name FROM agent_platform.mcp_server_definitions WHERE id=$1::uuid),
		 version,spec,spec_hash,status,created_at`, definitionID, raw, "sha256:"+hex.EncodeToString(sum[:]), input.CreatedBy).Scan(
		&item.ID, &item.DefinitionID, &item.TenantID, &item.Key, &item.Name, &item.Version, &item.Spec, &item.SpecHash, &item.Status, &item.CreatedAt)
	if err != nil {
		return mcp.ServerVersion{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return mcp.ServerVersion{}, err
	}
	return item, nil
}

func (s *RunStore) ListMCPServersForTenant(ctx context.Context, tenantID string, limit int) ([]mcp.ServerVersion, error) {
	if limit <= 0 || limit > 200 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `
		SELECT version.id::text,definition.id::text,definition.tenant_id,definition.server_key,definition.name,
		 version.version,version.spec,version.spec_hash,version.status,version.created_at,
		 health.status,health.protocol_version,health.latency_ms,health.error,health.checked_at
		FROM agent_platform.mcp_server_versions version
		JOIN agent_platform.mcp_server_definitions definition ON definition.id=version.definition_id
		LEFT JOIN agent_platform.mcp_connection_health health ON health.server_version_id=version.id
		WHERE definition.tenant_id=$1::text ORDER BY version.created_at DESC LIMIT $2`, tenantID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var items []mcp.ServerVersion
	for rows.Next() {
		var item mcp.ServerVersion
		var healthStatus, protocol, healthError *string
		var latency *int64
		var checkedAt *time.Time
		if err := rows.Scan(&item.ID, &item.DefinitionID, &item.TenantID, &item.Key, &item.Name, &item.Version, &item.Spec, &item.SpecHash, &item.Status, &item.CreatedAt, &healthStatus, &protocol, &latency, &healthError, &checkedAt); err != nil {
			return nil, err
		}
		if healthStatus != nil {
			item.Health = &mcp.Health{Status: *healthStatus}
			if protocol != nil {
				item.Health.ProtocolVersion = *protocol
			}
			if latency != nil {
				item.Health.LatencyMS = *latency
			}
			if healthError != nil {
				item.Health.Error = *healthError
			}
			if checkedAt != nil {
				item.Health.CheckedAt = *checkedAt
			}
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	rows.Close()
	for index := range items {
		toolRows, err := s.pool.Query(ctx, `SELECT id::text,server_version_id::text,tool_name,COALESCE(description,''),input_schema,schema_hash,risk,enabled,synced_at FROM agent_platform.mcp_tool_snapshots WHERE tenant_id=$1::text AND server_version_id=$2::uuid ORDER BY tool_name`, tenantID, items[index].ID)
		if err != nil {
			return nil, err
		}
		for toolRows.Next() {
			var item mcp.ToolSnapshot
			if err := toolRows.Scan(&item.ID, &item.ServerVersionID, &item.Name, &item.Description, &item.InputSchema, &item.SchemaHash, &item.Risk, &item.Enabled, &item.SyncedAt); err != nil {
				toolRows.Close()
				return nil, err
			}
			items[index].Tools = append(items[index].Tools, item)
		}
		toolRows.Close()
	}
	return items, nil
}

func (s *RunStore) GetMCPServerVersionForTenant(ctx context.Context, tenantID, id string) (mcp.ServerVersion, error) {
	var item mcp.ServerVersion
	err := s.pool.QueryRow(ctx, `
		SELECT version.id::text,definition.id::text,definition.tenant_id,definition.server_key,definition.name,
		 version.version,version.spec,version.spec_hash,version.status,version.created_at
		FROM agent_platform.mcp_server_versions version JOIN agent_platform.mcp_server_definitions definition ON definition.id=version.definition_id
		WHERE version.id=$1::uuid AND definition.tenant_id=$2::text`, id, tenantID).Scan(&item.ID, &item.DefinitionID, &item.TenantID, &item.Key, &item.Name, &item.Version, &item.Spec, &item.SpecHash, &item.Status, &item.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return item, errors.New("MCP server version not found")
	}
	return item, err
}

func (s *RunStore) ReplaceMCPToolSnapshots(ctx context.Context, tenantID, serverVersionID string, tools []mcp.ToolSnapshot) error {
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	if _, err := tx.Exec(ctx, `DELETE FROM agent_platform.mcp_tool_snapshots WHERE server_version_id=$1::uuid AND tenant_id=$2::text`, serverVersionID, tenantID); err != nil {
		return err
	}
	for _, item := range tools {
		_, err = tx.Exec(ctx, `INSERT INTO agent_platform.mcp_tool_snapshots(tenant_id,server_version_id,tool_name,description,input_schema,schema_hash,risk,enabled,synced_at) VALUES($1::text,$2::uuid,$3::text,$4::text,$5::jsonb,$6::text,$7::text,$8,$9)`, tenantID, serverVersionID, item.Name, item.Description, item.InputSchema, item.SchemaHash, item.Risk, item.Enabled, item.SyncedAt)
		if err != nil {
			return err
		}
	}
	return tx.Commit(ctx)
}

func (s *RunStore) SaveMCPHealth(ctx context.Context, serverVersionID string, health mcp.Health) error {
	_, err := s.pool.Exec(ctx, `INSERT INTO agent_platform.mcp_connection_health(server_version_id,status,protocol_version,latency_ms,error,checked_at) VALUES($1::uuid,$2::text,$3::text,$4,$5::text,$6) ON CONFLICT(server_version_id) DO UPDATE SET status=EXCLUDED.status,protocol_version=EXCLUDED.protocol_version,latency_ms=EXCLUDED.latency_ms,error=EXCLUDED.error,checked_at=EXCLUDED.checked_at`, serverVersionID, health.Status, health.ProtocolVersion, health.LatencyMS, health.Error, health.CheckedAt)
	return err
}

func (s *RunStore) GetMCPBinding(ctx context.Context, tenantID, serverVersionID, toolName, schemaHash string) (mcp.ServerSpec, mcp.ToolSnapshot, error) {
	version, err := s.GetMCPServerVersionForTenant(ctx, tenantID, serverVersionID)
	if err != nil {
		return mcp.ServerSpec{}, mcp.ToolSnapshot{}, err
	}
	var spec mcp.ServerSpec
	if err := json.Unmarshal(version.Spec, &spec); err != nil {
		return spec, mcp.ToolSnapshot{}, err
	}
	var item mcp.ToolSnapshot
	err = s.pool.QueryRow(ctx, `SELECT id::text,server_version_id::text,tool_name,COALESCE(description,''),input_schema,schema_hash,risk,enabled,synced_at FROM agent_platform.mcp_tool_snapshots WHERE tenant_id=$1::text AND server_version_id=$2::uuid AND tool_name=$3::text AND schema_hash=$4::text AND enabled=true`, tenantID, serverVersionID, toolName, schemaHash).Scan(&item.ID, &item.ServerVersionID, &item.Name, &item.Description, &item.InputSchema, &item.SchemaHash, &item.Risk, &item.Enabled, &item.SyncedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return spec, item, errors.New("pinned MCP tool snapshot not found or schema drifted")
	}
	if err != nil {
		return spec, item, fmt.Errorf("get MCP binding: %w", err)
	}
	return spec, item, nil
}
