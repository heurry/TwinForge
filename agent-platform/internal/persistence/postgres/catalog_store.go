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
	"github.com/jackc/pgx/v5/pgconn"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/resource"
)

// CreateDefinition adds a tenant-scoped Agent identity.
func (s *RunStore) CreateDefinition(ctx context.Context, input agent.CreateDefinition) (agent.Definition, error) {
	if strings.TrimSpace(input.TenantID) == "" || strings.TrimSpace(input.Key) == "" || strings.TrimSpace(input.Name) == "" {
		return agent.Definition{}, errors.New("tenant_id, key and name are required")
	}
	row := s.pool.QueryRow(ctx, `
		INSERT INTO agent_platform.agent_definitions (tenant_id, agent_key, name, description, owner)
		VALUES ($1::text, $2::text, $3::text, $4::text, $5::text)
		RETURNING `+definitionColumns,
		input.TenantID, input.Key, input.Name, input.Description, input.Owner)
	definition, err := scanDefinition(row)
	var databaseError *pgconn.PgError
	if errors.As(err, &databaseError) && databaseError.Code == "23505" {
		return agent.Definition{}, agent.ErrDefinitionConflict
	}
	if err != nil {
		return agent.Definition{}, fmt.Errorf("create agent definition: %w", err)
	}
	return definition, nil
}

// ListDefinitions returns a bounded tenant catalog ordered by creation time.
func (s *RunStore) ListDefinitions(ctx context.Context, tenantID string, limit int) ([]agent.Definition, error) {
	if limit <= 0 || limit > 200 {
		limit = 50
	}
	rows, err := s.pool.Query(ctx, `
		SELECT `+definitionColumns+` FROM agent_platform.agent_definitions
		WHERE tenant_id=$1::text ORDER BY created_at DESC, id DESC LIMIT $2`, tenantID, limit)
	if err != nil {
		return nil, fmt.Errorf("list agent definitions: %w", err)
	}
	defer rows.Close()
	definitions := make([]agent.Definition, 0)
	for rows.Next() {
		definition, err := scanDefinition(rows)
		if err != nil {
			return nil, fmt.Errorf("scan agent definition: %w", err)
		}
		definitions = append(definitions, definition)
	}
	return definitions, rows.Err()
}

// GetDefinition returns one tenant-scoped Agent identity.
func (s *RunStore) GetDefinition(ctx context.Context, tenantID, definitionID string) (agent.Definition, error) {
	definition, err := scanDefinition(s.pool.QueryRow(ctx, `
		SELECT `+definitionColumns+` FROM agent_platform.agent_definitions
		WHERE id=$1::uuid AND tenant_id=$2::text`, definitionID, tenantID))
	if errors.Is(err, pgx.ErrNoRows) {
		return agent.Definition{}, agent.ErrDefinitionNotFound
	}
	if err != nil {
		return agent.Definition{}, fmt.Errorf("get agent definition: %w", err)
	}
	return definition, nil
}

// CreateVersion serializes version-number allocation by locking its definition.
func (s *RunStore) CreateVersion(ctx context.Context, input agent.CreateVersion) (agent.Version, error) {
	if err := input.Spec.Validate(); err != nil {
		return agent.Version{}, err
	}
	spec, err := json.Marshal(input.Spec)
	if err != nil {
		return agent.Version{}, fmt.Errorf("marshal agent spec: %w", err)
	}
	digest := sha256.Sum256(spec)
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return agent.Version{}, fmt.Errorf("begin create agent version: %w", err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	var exists bool
	if err := tx.QueryRow(ctx, `
		SELECT true FROM agent_platform.agent_definitions
		WHERE id=$1::uuid AND tenant_id=$2::text FOR UPDATE`, input.AgentID, input.TenantID).Scan(&exists); errors.Is(err, pgx.ErrNoRows) {
		return agent.Version{}, agent.ErrDefinitionNotFound
	} else if err != nil {
		return agent.Version{}, fmt.Errorf("lock agent definition: %w", err)
	}
	if err := validateAgentResourceRef(ctx, tx, input.TenantID, "prompt", input.Spec.PromptRef); err != nil {
		return agent.Version{}, err
	}
	if err := validateAgentResourceRef(ctx, tx, input.TenantID, "toolset", input.Spec.ToolSetRef); err != nil {
		return agent.Version{}, err
	}
	if input.Spec.SkillSetRef != nil {
		if err := validateAgentResourceRef(ctx, tx, input.TenantID, "skillset", *input.Spec.SkillSetRef); err != nil {
			return agent.Version{}, err
		}
	}
	version, err := scanVersion(tx.QueryRow(ctx, `
		INSERT INTO agent_platform.agent_versions (agent_id, version, spec, spec_hash, created_by)
		SELECT $1::uuid, COALESCE(MAX(version), 0)+1, $2::jsonb, $3::text, $4::text
		FROM agent_platform.agent_versions WHERE agent_id=$1::uuid
		RETURNING `+versionColumns,
		input.AgentID, spec, hex.EncodeToString(digest[:]), input.CreatedBy))
	if err != nil {
		return agent.Version{}, fmt.Errorf("create agent version: %w", err)
	}
	if err := tx.Commit(ctx); err != nil {
		return agent.Version{}, fmt.Errorf("commit agent version: %w", err)
	}
	return version, nil
}

func validateAgentResourceRef(
	ctx context.Context, tx pgx.Tx, tenantID, kind string, ref agent.VersionRef,
) error {
	var query string
	var notFound error
	switch kind {
	case "prompt":
		query = `
			SELECT version.version FROM agent_platform.prompt_versions AS version
			JOIN agent_platform.prompt_definitions AS definition ON definition.id=version.definition_id
			WHERE version.id=$1::uuid AND definition.tenant_id=$2::text AND version.status='published'`
		notFound = resource.ErrPromptVersionNotFound
	case "toolset":
		query = `
			SELECT version.version FROM agent_platform.toolset_versions AS version
			JOIN agent_platform.toolset_definitions AS definition ON definition.id=version.definition_id
			WHERE version.id=$1::uuid AND definition.tenant_id=$2::text AND version.status='published'`
		notFound = resource.ErrToolSetVersionNotFound
	case "skillset":
		query = `
			SELECT version.version FROM agent_platform.skillset_versions AS version
			JOIN agent_platform.skillset_definitions AS definition ON definition.id=version.definition_id
			WHERE version.id=$1::uuid AND definition.tenant_id=$2::text AND version.status='published'`
		notFound = ErrSkillSetVersionNotFound
	default:
		return fmt.Errorf("unknown agent resource kind %q", kind)
	}
	var actual int
	if err := tx.QueryRow(ctx, query, ref.ID, tenantID).Scan(&actual); errors.Is(err, pgx.ErrNoRows) {
		return notFound
	} else if err != nil {
		return fmt.Errorf("validate agent %s ref: %w", kind, err)
	}
	if strconv.Itoa(actual) != ref.Version {
		return fmt.Errorf("%s version %s expected revision %s, got %d", kind, ref.ID, ref.Version, actual)
	}
	return nil
}

// ListExecutableVersions returns published versions with stable Agent labels.
func (s *RunStore) ListExecutableVersions(ctx context.Context, tenantID string, limit int) ([]agent.ExecutableVersion, error) {
	if limit <= 0 || limit > 200 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `SELECT version.id::text,definition.id::text,definition.agent_key,definition.name,
		version.version,version.spec,version.published_at FROM agent_platform.agent_versions version
		JOIN agent_platform.agent_definitions definition ON definition.id=version.agent_id
		WHERE definition.tenant_id=$1 AND version.status='published'
		ORDER BY version.published_at DESC NULLS LAST,version.created_at DESC LIMIT $2`, tenantID, limit)
	if err != nil {
		return nil, fmt.Errorf("list executable agent versions: %w", err)
	}
	defer rows.Close()
	items := make([]agent.ExecutableVersion, 0)
	for rows.Next() {
		var item agent.ExecutableVersion
		if err := rows.Scan(&item.ID, &item.AgentID, &item.AgentKey, &item.AgentName, &item.Version, &item.Spec, &item.PublishedAt); err != nil {
			return nil, fmt.Errorf("scan executable agent version: %w", err)
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

// ListVersions returns all versions of a tenant-scoped Agent.
func (s *RunStore) ListVersions(ctx context.Context, tenantID, definitionID string) ([]agent.Version, error) {
	if _, err := s.GetDefinition(ctx, tenantID, definitionID); err != nil {
		return nil, err
	}
	rows, err := s.pool.Query(ctx, `
		SELECT `+versionColumns+` FROM agent_platform.agent_versions
		WHERE agent_id=$1::uuid ORDER BY version DESC`, definitionID)
	if err != nil {
		return nil, fmt.Errorf("list agent versions: %w", err)
	}
	defer rows.Close()
	versions := make([]agent.Version, 0)
	for rows.Next() {
		version, err := scanVersion(rows)
		if err != nil {
			return nil, fmt.Errorf("scan agent version: %w", err)
		}
		versions = append(versions, version)
	}
	return versions, rows.Err()
}

// ReleaseVersion validates and atomically makes a version active.
func (s *RunStore) ReleaseVersion(ctx context.Context, tenantID, versionID string) (agent.Version, error) {
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return agent.Version{}, fmt.Errorf("begin release agent version: %w", err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	version, err := scanVersion(tx.QueryRow(ctx, `
		SELECT `+qualifiedVersionColumns("version")+`
		FROM agent_platform.agent_versions AS version
		JOIN agent_platform.agent_definitions AS definition ON definition.id=version.agent_id
		WHERE version.id=$1::uuid AND definition.tenant_id=$2::text FOR UPDATE OF version, definition`,
		versionID, tenantID))
	if errors.Is(err, pgx.ErrNoRows) {
		return agent.Version{}, agent.ErrVersionNotFound
	}
	if err != nil {
		return agent.Version{}, fmt.Errorf("lock agent version: %w", err)
	}
	var spec agent.Spec
	if err := json.Unmarshal(version.Spec, &spec); err != nil {
		return agent.Version{}, fmt.Errorf("decode stored agent spec: %w", err)
	}
	if err := spec.Validate(); err != nil {
		return agent.Version{}, err
	}
	if _, err := tx.Exec(ctx, `
		UPDATE agent_platform.agent_versions SET status='published', published_at=COALESCE(published_at, now())
		WHERE id=$1::uuid`, version.ID); err != nil {
		return agent.Version{}, fmt.Errorf("release agent version: %w", err)
	}
	if _, err := tx.Exec(ctx, `
		UPDATE agent_platform.agent_definitions SET active_version_id=$1::uuid, updated_at=now()
		WHERE id=$2::uuid`, version.ID, version.AgentID); err != nil {
		return agent.Version{}, fmt.Errorf("activate agent version: %w", err)
	}
	version, err = scanVersion(tx.QueryRow(ctx,
		`SELECT `+versionColumns+` FROM agent_platform.agent_versions WHERE id=$1::uuid`, version.ID))
	if err != nil {
		return agent.Version{}, fmt.Errorf("read released agent version: %w", err)
	}
	if err := tx.Commit(ctx); err != nil {
		return agent.Version{}, fmt.Errorf("commit agent version release: %w", err)
	}
	return version, nil
}

const definitionColumns = `
	id::text, tenant_id, agent_key, name, description, owner, status,
	active_version_id::text, created_at, updated_at`

const versionColumns = `
	id::text, agent_id::text, version, spec, spec_hash, status,
	created_by, created_at, published_at`

func qualifiedVersionColumns(alias string) string {
	return alias + `.id::text, ` + alias + `.agent_id::text, ` + alias + `.version, ` +
		alias + `.spec, ` + alias + `.spec_hash, ` + alias + `.status, ` +
		alias + `.created_by, ` + alias + `.created_at, ` + alias + `.published_at`
}

func scanDefinition(row rowScanner) (agent.Definition, error) {
	var definition agent.Definition
	err := row.Scan(&definition.ID, &definition.TenantID, &definition.Key, &definition.Name,
		&definition.Description, &definition.Owner, &definition.Status, &definition.ActiveVersionID,
		&definition.CreatedAt, &definition.UpdatedAt)
	return definition, err
}

func scanVersion(row rowScanner) (agent.Version, error) {
	var version agent.Version
	err := row.Scan(&version.ID, &version.AgentID, &version.Version, &version.Spec,
		&version.SpecHash, &version.Status, &version.CreatedBy, &version.CreatedAt, &version.PublishedAt)
	return version, err
}
