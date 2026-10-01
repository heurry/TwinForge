package postgres

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/skill"
)

var ErrSkillVersionNotFound = errors.New("skill version not found")
var ErrSkillSetVersionNotFound = errors.New("skill set version not found")

func (s *RunStore) CreateSkillVersion(ctx context.Context, input skill.CreateVersion) (skill.Version, error) {
	if strings.TrimSpace(input.TenantID) == "" || strings.TrimSpace(input.Key) == "" || strings.TrimSpace(input.Name) == "" {
		return skill.Version{}, errors.New("tenant_id, key and name are required")
	}
	if err := input.Spec.Validate(); err != nil {
		return skill.Version{}, err
	}
	encoded, hash, err := marshalHashed(input.Spec)
	if err != nil {
		return skill.Version{}, err
	}
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return skill.Version{}, err
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	for _, ref := range input.Spec.RequiredTools {
		var actual int
		err = tx.QueryRow(ctx, `SELECT v.version FROM agent_platform.tool_versions v JOIN agent_platform.tool_definitions d ON d.id=v.definition_id WHERE v.id=$1::uuid AND d.tenant_id=$2 AND v.status='published'`, ref.ID, input.TenantID).Scan(&actual)
		if errors.Is(err, pgx.ErrNoRows) {
			return skill.Version{}, ErrSkillVersionNotFound
		}
		if err != nil {
			return skill.Version{}, fmt.Errorf("validate skill tool: %w", err)
		}
		if strconv.Itoa(actual) != ref.Version {
			return skill.Version{}, fmt.Errorf("tool %s revision mismatch", ref.ID)
		}
	}
	var definitionID string
	err = tx.QueryRow(ctx, `INSERT INTO agent_platform.skill_definitions(tenant_id,skill_key,name) VALUES($1,$2,$3) ON CONFLICT(tenant_id,skill_key) DO UPDATE SET name=EXCLUDED.name,updated_at=now() RETURNING id::text`, input.TenantID, input.Key, input.Name).Scan(&definitionID)
	if err != nil {
		return skill.Version{}, fmt.Errorf("upsert skill: %w", err)
	}
	v, err := scanSkillVersion(tx.QueryRow(ctx, `INSERT INTO agent_platform.skill_versions(definition_id,version,spec,spec_hash,created_by) SELECT $1::uuid,COALESCE(MAX(version),0)+1,$2::jsonb,$3,$4 FROM agent_platform.skill_versions WHERE definition_id=$1::uuid RETURNING id::text,definition_id::text,version,spec,spec_hash,status,created_by,created_at`, definitionID, encoded, hash, input.CreatedBy))
	if err != nil {
		return skill.Version{}, fmt.Errorf("insert skill version: %w", err)
	}
	v.TenantID, v.Key, v.Name = input.TenantID, input.Key, input.Name
	if err = tx.Commit(ctx); err != nil {
		return skill.Version{}, err
	}
	return v, nil
}

func (s *RunStore) GetSkillVersion(ctx context.Context, tenantID, id string) (skill.Version, error) {
	v, err := scanSkillVersionWithDefinition(s.pool.QueryRow(ctx, `SELECT v.id::text,v.definition_id::text,d.tenant_id,d.skill_key,d.name,v.version,v.spec,v.spec_hash,v.status,v.created_by,v.created_at FROM agent_platform.skill_versions v JOIN agent_platform.skill_definitions d ON d.id=v.definition_id WHERE v.id=$1::uuid AND d.tenant_id=$2 AND v.status='published'`, id, tenantID))
	if errors.Is(err, pgx.ErrNoRows) {
		return v, ErrSkillVersionNotFound
	}
	if err != nil {
		return v, fmt.Errorf("get skill: %w", err)
	}
	return v, nil
}

func (s *RunStore) ListSkillVersions(ctx context.Context, tenantID string, limit int) ([]skill.Version, error) {
	rows, err := s.pool.Query(ctx, `SELECT v.id::text,v.definition_id::text,d.tenant_id,d.skill_key,d.name,v.version,v.spec,v.spec_hash,v.status,v.created_by,v.created_at FROM agent_platform.skill_versions v JOIN agent_platform.skill_definitions d ON d.id=v.definition_id WHERE d.tenant_id=$1 AND v.status='published' ORDER BY v.created_at DESC,v.version DESC LIMIT $2`, tenantID, normalizeResourceLimit(limit))
	if err != nil {
		return nil, fmt.Errorf("list skills: %w", err)
	}
	defer rows.Close()
	versions := make([]skill.Version, 0)
	for rows.Next() {
		version, scanErr := scanSkillVersionWithDefinition(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan skill: %w", scanErr)
		}
		versions = append(versions, version)
	}
	return versions, rows.Err()
}

func (s *RunStore) CreateSkillSetVersion(ctx context.Context, input skill.CreateSetVersion) (skill.Version, error) {
	if strings.TrimSpace(input.TenantID) == "" || strings.TrimSpace(input.Key) == "" || strings.TrimSpace(input.Name) == "" {
		return skill.Version{}, errors.New("tenant_id, key and name are required")
	}
	if err := input.Spec.Validate(); err != nil {
		return skill.Version{}, err
	}
	encoded, hash, err := marshalHashed(input.Spec)
	if err != nil {
		return skill.Version{}, err
	}
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return skill.Version{}, err
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	for _, ref := range input.Spec.Skills {
		var actual int
		err = tx.QueryRow(ctx, `SELECT v.version FROM agent_platform.skill_versions v JOIN agent_platform.skill_definitions d ON d.id=v.definition_id WHERE v.id=$1::uuid AND d.tenant_id=$2 AND v.status='published'`, ref.ID, input.TenantID).Scan(&actual)
		if errors.Is(err, pgx.ErrNoRows) {
			return skill.Version{}, ErrSkillVersionNotFound
		}
		if err != nil {
			return skill.Version{}, err
		}
		if strconv.Itoa(actual) != ref.Version {
			return skill.Version{}, fmt.Errorf("skill %s revision mismatch", ref.ID)
		}
	}
	var definitionID string
	err = tx.QueryRow(ctx, `INSERT INTO agent_platform.skillset_definitions(tenant_id,skillset_key,name) VALUES($1,$2,$3) ON CONFLICT(tenant_id,skillset_key) DO UPDATE SET name=EXCLUDED.name,updated_at=now() RETURNING id::text`, input.TenantID, input.Key, input.Name).Scan(&definitionID)
	if err != nil {
		return skill.Version{}, err
	}
	v, err := scanSkillVersion(tx.QueryRow(ctx, `INSERT INTO agent_platform.skillset_versions(definition_id,version,spec,spec_hash,created_by) SELECT $1::uuid,COALESCE(MAX(version),0)+1,$2::jsonb,$3,$4 FROM agent_platform.skillset_versions WHERE definition_id=$1::uuid RETURNING id::text,definition_id::text,version,spec,spec_hash,status,created_by,created_at`, definitionID, encoded, hash, input.CreatedBy))
	if err != nil {
		return skill.Version{}, err
	}
	v.TenantID, v.Key, v.Name = input.TenantID, input.Key, input.Name
	if err = tx.Commit(ctx); err != nil {
		return skill.Version{}, err
	}
	return v, nil
}

func (s *RunStore) GetSkillSetVersion(ctx context.Context, tenantID, id string) (skill.Version, error) {
	v, err := scanSkillVersionWithDefinition(s.pool.QueryRow(ctx, `SELECT v.id::text,v.definition_id::text,d.tenant_id,d.skillset_key,d.name,v.version,v.spec,v.spec_hash,v.status,v.created_by,v.created_at FROM agent_platform.skillset_versions v JOIN agent_platform.skillset_definitions d ON d.id=v.definition_id WHERE v.id=$1::uuid AND d.tenant_id=$2 AND v.status='published'`, id, tenantID))
	if errors.Is(err, pgx.ErrNoRows) {
		return v, ErrSkillSetVersionNotFound
	}
	return v, err
}

func (s *RunStore) ListSkillSetVersions(ctx context.Context, tenantID string, limit int) ([]skill.Version, error) {
	rows, err := s.pool.Query(ctx, `SELECT v.id::text,v.definition_id::text,d.tenant_id,d.skillset_key,d.name,v.version,v.spec,v.spec_hash,v.status,v.created_by,v.created_at FROM agent_platform.skillset_versions v JOIN agent_platform.skillset_definitions d ON d.id=v.definition_id WHERE d.tenant_id=$1 AND v.status='published' ORDER BY v.created_at DESC,v.version DESC LIMIT $2`, tenantID, normalizeResourceLimit(limit))
	if err != nil {
		return nil, fmt.Errorf("list skill sets: %w", err)
	}
	defer rows.Close()
	versions := make([]skill.Version, 0)
	for rows.Next() {
		version, scanErr := scanSkillVersionWithDefinition(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan skill set: %w", scanErr)
		}
		versions = append(versions, version)
	}
	return versions, rows.Err()
}

func scanSkillVersion(row rowScanner) (skill.Version, error) {
	var v skill.Version
	err := row.Scan(&v.ID, &v.DefinitionID, &v.Version, &v.Spec, &v.SpecHash, &v.Status, &v.CreatedBy, &v.CreatedAt)
	return v, err
}
func scanSkillVersionWithDefinition(row rowScanner) (skill.Version, error) {
	var v skill.Version
	err := row.Scan(&v.ID, &v.DefinitionID, &v.TenantID, &v.Key, &v.Name, &v.Version, &v.Spec, &v.SpecHash, &v.Status, &v.CreatedBy, &v.CreatedAt)
	return v, err
}
