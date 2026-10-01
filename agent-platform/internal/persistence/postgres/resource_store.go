package postgres

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/resource"
)

// CreatePromptVersion atomically upserts a prompt identity and appends an
// immutable published version.
func (s *RunStore) CreatePromptVersion(ctx context.Context, input resource.CreatePromptVersion) (resource.PromptVersion, error) {
	if strings.TrimSpace(input.TenantID) == "" || strings.TrimSpace(input.Key) == "" ||
		strings.TrimSpace(input.Name) == "" || strings.TrimSpace(input.Content) == "" {
		return resource.PromptVersion{}, errors.New("tenant_id, key, name and content are required")
	}
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return resource.PromptVersion{}, fmt.Errorf("begin prompt version: %w", err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	var definitionID string
	if err := tx.QueryRow(ctx, `
		INSERT INTO agent_platform.prompt_definitions (tenant_id, prompt_key, name)
		VALUES ($1::text, $2::text, $3::text)
		ON CONFLICT (tenant_id, prompt_key) DO UPDATE SET name=EXCLUDED.name, updated_at=now()
		RETURNING id::text`, input.TenantID, input.Key, input.Name).Scan(&definitionID); err != nil {
		return resource.PromptVersion{}, fmt.Errorf("upsert prompt definition: %w", err)
	}
	digest := sha256.Sum256([]byte(input.Content))
	version, err := scanPromptVersion(tx.QueryRow(ctx, `
		INSERT INTO agent_platform.prompt_versions (definition_id, version, content, content_hash, created_by)
		SELECT $1::uuid, COALESCE(MAX(version), 0)+1, $2::text, $3::text, $4::text
		FROM agent_platform.prompt_versions WHERE definition_id=$1::uuid
		RETURNING id::text, definition_id::text, version, content, content_hash, status, created_by, created_at`,
		definitionID, input.Content, hex.EncodeToString(digest[:]), input.CreatedBy))
	if err != nil {
		return resource.PromptVersion{}, fmt.Errorf("insert prompt version: %w", err)
	}
	version.TenantID, version.Key, version.Name = input.TenantID, input.Key, input.Name
	if err := tx.Commit(ctx); err != nil {
		return resource.PromptVersion{}, fmt.Errorf("commit prompt version: %w", err)
	}
	return version, nil
}

// GetPromptVersion resolves one exact tenant-owned prompt version.
func (s *RunStore) GetPromptVersion(ctx context.Context, tenantID, versionID string) (resource.PromptVersion, error) {
	version, err := scanPromptVersionWithDefinition(s.pool.QueryRow(ctx, `
		SELECT version.id::text, version.definition_id::text, definition.tenant_id,
			definition.prompt_key, definition.name, version.version, version.content,
			version.content_hash, version.status, version.created_by, version.created_at
		FROM agent_platform.prompt_versions AS version
		JOIN agent_platform.prompt_definitions AS definition ON definition.id=version.definition_id
		WHERE version.id=$1::uuid AND definition.tenant_id=$2::text AND version.status='published'`,
		versionID, tenantID))
	if errors.Is(err, pgx.ErrNoRows) {
		return resource.PromptVersion{}, resource.ErrPromptVersionNotFound
	}
	if err != nil {
		return resource.PromptVersion{}, fmt.Errorf("get prompt version: %w", err)
	}
	return version, nil
}

// ListPromptVersions returns published prompt revisions newest first. Keeping
// every revision visible is required by the authoring UI and release audit.
func (s *RunStore) ListPromptVersions(ctx context.Context, tenantID string, limit int) ([]resource.PromptVersion, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT version.id::text, version.definition_id::text, definition.tenant_id,
			definition.prompt_key, definition.name, version.version, version.content,
			version.content_hash, version.status, version.created_by, version.created_at
		FROM agent_platform.prompt_versions AS version
		JOIN agent_platform.prompt_definitions AS definition ON definition.id=version.definition_id
		WHERE definition.tenant_id=$1::text AND version.status='published'
		ORDER BY version.created_at DESC, version.version DESC
		LIMIT $2`, tenantID, normalizeResourceLimit(limit))
	if err != nil {
		return nil, fmt.Errorf("list prompt versions: %w", err)
	}
	defer rows.Close()
	versions := make([]resource.PromptVersion, 0)
	for rows.Next() {
		version, scanErr := scanPromptVersionWithDefinition(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan prompt version: %w", scanErr)
		}
		versions = append(versions, version)
	}
	return versions, rows.Err()
}

// CreateToolVersion appends an immutable provider-bound Tool version.
func (s *RunStore) CreateToolVersion(ctx context.Context, input resource.CreateToolVersion) (resource.ToolVersion, error) {
	if strings.TrimSpace(input.TenantID) == "" || strings.TrimSpace(input.Key) == "" || strings.TrimSpace(input.Name) == "" {
		return resource.ToolVersion{}, errors.New("tenant_id, key and name are required")
	}
	// The store owns the monotonically increasing revision; callers cannot forge
	// the model-visible Tool definition version.
	input.Spec.Definition.Version = "pending"
	if err := input.Spec.Validate(); err != nil {
		return resource.ToolVersion{}, err
	}
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return resource.ToolVersion{}, fmt.Errorf("begin tool version: %w", err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	var definitionID string
	if err := tx.QueryRow(ctx, `
		INSERT INTO agent_platform.tool_definitions (tenant_id, tool_key, name)
		VALUES ($1::text, $2::text, $3::text)
		ON CONFLICT (tenant_id, tool_key) DO UPDATE SET name=EXCLUDED.name, updated_at=now()
		RETURNING id::text`, input.TenantID, input.Key, input.Name).Scan(&definitionID); err != nil {
		return resource.ToolVersion{}, fmt.Errorf("upsert tool definition: %w", err)
	}
	var nextVersion int
	if err := tx.QueryRow(ctx, `
		SELECT COALESCE(MAX(version), 0)+1 FROM agent_platform.tool_versions
		WHERE definition_id=$1::uuid`, definitionID).Scan(&nextVersion); err != nil {
		return resource.ToolVersion{}, fmt.Errorf("allocate tool version: %w", err)
	}
	input.Spec.Definition.Version = strconv.Itoa(nextVersion)
	spec, hash, err := marshalHashed(input.Spec)
	if err != nil {
		return resource.ToolVersion{}, err
	}
	version, err := scanToolVersion(tx.QueryRow(ctx, `
		INSERT INTO agent_platform.tool_versions (definition_id, version, spec, spec_hash, created_by)
		VALUES ($1::uuid, $2, $3::jsonb, $4::text, $5::text)
		RETURNING id::text, definition_id::text, version, spec, spec_hash, status, created_by, created_at`,
		definitionID, nextVersion, spec, hash, input.CreatedBy))
	if err != nil {
		return resource.ToolVersion{}, fmt.Errorf("insert tool version: %w", err)
	}
	version.TenantID, version.Key, version.Name = input.TenantID, input.Key, input.Name
	if err := tx.Commit(ctx); err != nil {
		return resource.ToolVersion{}, fmt.Errorf("commit tool version: %w", err)
	}
	return version, nil
}

// GetToolVersion resolves one exact tenant-owned Tool version.
func (s *RunStore) GetToolVersion(ctx context.Context, tenantID, versionID string) (resource.ToolVersion, error) {
	version, err := scanToolVersionWithDefinition(s.pool.QueryRow(ctx, `
		SELECT version.id::text, version.definition_id::text, definition.tenant_id,
			definition.tool_key, definition.name, version.version, version.spec,
			version.spec_hash, version.status, version.created_by, version.created_at
		FROM agent_platform.tool_versions AS version
		JOIN agent_platform.tool_definitions AS definition ON definition.id=version.definition_id
		WHERE version.id=$1::uuid AND definition.tenant_id=$2::text AND version.status='published'`,
		versionID, tenantID))
	if errors.Is(err, pgx.ErrNoRows) {
		return resource.ToolVersion{}, resource.ErrToolVersionNotFound
	}
	if err != nil {
		return resource.ToolVersion{}, fmt.Errorf("get tool version: %w", err)
	}
	return version, nil
}

// ListToolVersions returns published provider-bound Tool revisions newest first.
func (s *RunStore) ListToolVersions(ctx context.Context, tenantID string, limit int) ([]resource.ToolVersion, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT version.id::text, version.definition_id::text, definition.tenant_id,
			definition.tool_key, definition.name, version.version, version.spec,
			version.spec_hash, version.status, version.created_by, version.created_at
		FROM agent_platform.tool_versions AS version
		JOIN agent_platform.tool_definitions AS definition ON definition.id=version.definition_id
		WHERE definition.tenant_id=$1::text AND version.status='published'
		ORDER BY version.created_at DESC, version.version DESC
		LIMIT $2`, tenantID, normalizeResourceLimit(limit))
	if err != nil {
		return nil, fmt.Errorf("list tool versions: %w", err)
	}
	defer rows.Close()
	versions := make([]resource.ToolVersion, 0)
	for rows.Next() {
		version, scanErr := scanToolVersionWithDefinition(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan tool version: %w", scanErr)
		}
		versions = append(versions, version)
	}
	return versions, rows.Err()
}

// CreateToolSetVersion validates all referenced versions within the tenant and
// appends an immutable ToolSet version.
func (s *RunStore) CreateToolSetVersion(ctx context.Context, input resource.CreateToolSetVersion) (resource.ToolSetVersion, error) {
	if strings.TrimSpace(input.TenantID) == "" || strings.TrimSpace(input.Key) == "" || strings.TrimSpace(input.Name) == "" {
		return resource.ToolSetVersion{}, errors.New("tenant_id, key and name are required")
	}
	if err := input.Spec.Validate(); err != nil {
		return resource.ToolSetVersion{}, err
	}
	spec, hash, err := marshalHashed(input.Spec)
	if err != nil {
		return resource.ToolSetVersion{}, err
	}
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return resource.ToolSetVersion{}, fmt.Errorf("begin tool set version: %w", err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	for _, ref := range input.Spec.Tools {
		var actualVersion int
		if err := tx.QueryRow(ctx, `
			SELECT version.version FROM agent_platform.tool_versions AS version
			JOIN agent_platform.tool_definitions AS definition ON definition.id=version.definition_id
			WHERE version.id=$1::uuid AND definition.tenant_id=$2::text AND version.status='published'`,
			ref.ID, input.TenantID).Scan(&actualVersion); errors.Is(err, pgx.ErrNoRows) {
			return resource.ToolSetVersion{}, resource.ErrToolVersionNotFound
		} else if err != nil {
			return resource.ToolSetVersion{}, fmt.Errorf("validate tool set member: %w", err)
		}
		if strconv.Itoa(actualVersion) != ref.Version {
			return resource.ToolSetVersion{}, fmt.Errorf("tool version %s expected revision %s, got %d", ref.ID, ref.Version, actualVersion)
		}
	}
	var definitionID string
	if err := tx.QueryRow(ctx, `
		INSERT INTO agent_platform.toolset_definitions (tenant_id, toolset_key, name)
		VALUES ($1::text, $2::text, $3::text)
		ON CONFLICT (tenant_id, toolset_key) DO UPDATE SET name=EXCLUDED.name, updated_at=now()
		RETURNING id::text`, input.TenantID, input.Key, input.Name).Scan(&definitionID); err != nil {
		return resource.ToolSetVersion{}, fmt.Errorf("upsert tool set definition: %w", err)
	}
	version, err := scanToolSetVersion(tx.QueryRow(ctx, `
		INSERT INTO agent_platform.toolset_versions (definition_id, version, spec, spec_hash, created_by)
		SELECT $1::uuid, COALESCE(MAX(version), 0)+1, $2::jsonb, $3::text, $4::text
		FROM agent_platform.toolset_versions WHERE definition_id=$1::uuid
		RETURNING id::text, definition_id::text, version, spec, spec_hash, status, created_by, created_at`,
		definitionID, spec, hash, input.CreatedBy))
	if err != nil {
		return resource.ToolSetVersion{}, fmt.Errorf("insert tool set version: %w", err)
	}
	version.TenantID, version.Key, version.Name = input.TenantID, input.Key, input.Name
	if err := tx.Commit(ctx); err != nil {
		return resource.ToolSetVersion{}, fmt.Errorf("commit tool set version: %w", err)
	}
	return version, nil
}

// GetToolSetVersion resolves one exact tenant-owned ToolSet version.
func (s *RunStore) GetToolSetVersion(ctx context.Context, tenantID, versionID string) (resource.ToolSetVersion, error) {
	version, err := scanToolSetVersionWithDefinition(s.pool.QueryRow(ctx, `
		SELECT version.id::text, version.definition_id::text, definition.tenant_id,
			definition.toolset_key, definition.name, version.version, version.spec,
			version.spec_hash, version.status, version.created_by, version.created_at
		FROM agent_platform.toolset_versions AS version
		JOIN agent_platform.toolset_definitions AS definition ON definition.id=version.definition_id
		WHERE version.id=$1::uuid AND definition.tenant_id=$2::text AND version.status='published'`,
		versionID, tenantID))
	if errors.Is(err, pgx.ErrNoRows) {
		return resource.ToolSetVersion{}, resource.ErrToolSetVersionNotFound
	}
	if err != nil {
		return resource.ToolSetVersion{}, fmt.Errorf("get tool set version: %w", err)
	}
	return version, nil
}

// ListToolSetVersions returns published ToolSet revisions newest first.
func (s *RunStore) ListToolSetVersions(ctx context.Context, tenantID string, limit int) ([]resource.ToolSetVersion, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT version.id::text, version.definition_id::text, definition.tenant_id,
			definition.toolset_key, definition.name, version.version, version.spec,
			version.spec_hash, version.status, version.created_by, version.created_at
		FROM agent_platform.toolset_versions AS version
		JOIN agent_platform.toolset_definitions AS definition ON definition.id=version.definition_id
		WHERE definition.tenant_id=$1::text AND version.status='published'
		ORDER BY version.created_at DESC, version.version DESC
		LIMIT $2`, tenantID, normalizeResourceLimit(limit))
	if err != nil {
		return nil, fmt.Errorf("list tool set versions: %w", err)
	}
	defer rows.Close()
	versions := make([]resource.ToolSetVersion, 0)
	for rows.Next() {
		version, scanErr := scanToolSetVersionWithDefinition(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan tool set version: %w", scanErr)
		}
		versions = append(versions, version)
	}
	return versions, rows.Err()
}

func normalizeResourceLimit(limit int) int {
	if limit <= 0 {
		return 100
	}
	if limit > 500 {
		return 500
	}
	return limit
}

func marshalHashed(value any) (json.RawMessage, string, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, "", fmt.Errorf("marshal versioned resource: %w", err)
	}
	digest := sha256.Sum256(encoded)
	return encoded, hex.EncodeToString(digest[:]), nil
}

func scanPromptVersion(row rowScanner) (resource.PromptVersion, error) {
	var version resource.PromptVersion
	err := row.Scan(&version.ID, &version.DefinitionID, &version.Version, &version.Content,
		&version.ContentHash, &version.Status, &version.CreatedBy, &version.CreatedAt)
	return version, err
}

func scanPromptVersionWithDefinition(row rowScanner) (resource.PromptVersion, error) {
	var version resource.PromptVersion
	err := row.Scan(&version.ID, &version.DefinitionID, &version.TenantID, &version.Key, &version.Name,
		&version.Version, &version.Content, &version.ContentHash, &version.Status, &version.CreatedBy, &version.CreatedAt)
	return version, err
}

func scanToolVersion(row rowScanner) (resource.ToolVersion, error) {
	var version resource.ToolVersion
	err := row.Scan(&version.ID, &version.DefinitionID, &version.Version, &version.Spec,
		&version.SpecHash, &version.Status, &version.CreatedBy, &version.CreatedAt)
	return version, err
}

func scanToolVersionWithDefinition(row rowScanner) (resource.ToolVersion, error) {
	var version resource.ToolVersion
	err := row.Scan(&version.ID, &version.DefinitionID, &version.TenantID, &version.Key, &version.Name,
		&version.Version, &version.Spec, &version.SpecHash, &version.Status, &version.CreatedBy, &version.CreatedAt)
	return version, err
}

func scanToolSetVersion(row rowScanner) (resource.ToolSetVersion, error) {
	var version resource.ToolSetVersion
	err := row.Scan(&version.ID, &version.DefinitionID, &version.Version, &version.Spec,
		&version.SpecHash, &version.Status, &version.CreatedBy, &version.CreatedAt)
	return version, err
}

func scanToolSetVersionWithDefinition(row rowScanner) (resource.ToolSetVersion, error) {
	var version resource.ToolSetVersion
	err := row.Scan(&version.ID, &version.DefinitionID, &version.TenantID, &version.Key, &version.Name,
		&version.Version, &version.Spec, &version.SpecHash, &version.Status, &version.CreatedBy, &version.CreatedAt)
	return version, err
}
