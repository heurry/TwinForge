package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
)

// CreateSession creates a tenant-owned conversation for one Agent definition.
func (s *RunStore) CreateSession(ctx context.Context, input agent.CreateSession) (agent.Session, error) {
	metadata := input.Metadata
	if len(metadata) == 0 {
		metadata = json.RawMessage(`{}`)
	}
	if !validJSONObject(metadata) {
		return agent.Session{}, errors.New("session metadata must be a JSON object")
	}
	session, err := scanSession(s.pool.QueryRow(ctx, `
		INSERT INTO agent_platform.agent_sessions (tenant_id, agent_id, user_id, metadata)
		SELECT $1::text, definition.id, $3::text, $4::jsonb
		FROM agent_platform.agent_definitions AS definition
		WHERE definition.id=$2::uuid AND definition.tenant_id=$1::text AND definition.status='active'
		RETURNING id::text, tenant_id, agent_id::text, user_id, status, metadata, created_at, updated_at`,
		input.TenantID, input.AgentID, input.UserID, metadata))
	if errors.Is(err, pgx.ErrNoRows) {
		return agent.Session{}, agent.ErrDefinitionNotFound
	}
	if err != nil {
		return agent.Session{}, fmt.Errorf("create session: %w", err)
	}
	return session, nil
}

// GetSession returns one tenant-scoped Session.
func (s *RunStore) GetSession(ctx context.Context, tenantID, sessionID string) (agent.Session, error) {
	session, err := scanSession(s.pool.QueryRow(ctx, `
		SELECT id::text, tenant_id, agent_id::text, user_id, status, metadata, created_at, updated_at
		FROM agent_platform.agent_sessions WHERE id=$1::uuid AND tenant_id=$2::text`,
		sessionID, tenantID))
	if errors.Is(err, pgx.ErrNoRows) {
		return agent.Session{}, agent.ErrSessionNotFound
	}
	if err != nil {
		return agent.Session{}, fmt.Errorf("get session: %w", err)
	}
	return session, nil
}

// ListSessions returns recent conversations with persisted Run activity facts.
func (s *RunStore) ListSessions(ctx context.Context, tenantID, agentID string, limit int) ([]agent.Session, error) {
	if limit <= 0 || limit > 100 {
		limit = 50
	}
	rows, err := s.pool.Query(ctx, `
		SELECT session.id::text, session.tenant_id, session.agent_id::text, session.user_id,
			session.status, session.metadata, session.created_at, session.updated_at,
			count(run.id),
			(SELECT count(*) FROM agent_platform.agent_session_messages AS message WHERE message.session_id=session.id),
			count(run.id) FILTER (WHERE run.status IN ('queued','running','waiting_tool','waiting_approval','waiting_input','waiting_external')),
			max(run.created_at)
		FROM agent_platform.agent_sessions AS session
		LEFT JOIN agent_platform.agent_runs AS run ON run.session_id=session.id
		WHERE session.tenant_id=$1::text AND ($2::text='' OR session.agent_id=$2::uuid)
		GROUP BY session.id
		ORDER BY COALESCE(max(run.created_at), session.updated_at) DESC
		LIMIT $3`, tenantID, agentID, limit)
	if err != nil {
		return nil, fmt.Errorf("list sessions: %w", err)
	}
	defer rows.Close()
	sessions := make([]agent.Session, 0)
	for rows.Next() {
		var session agent.Session
		if err := rows.Scan(&session.ID, &session.TenantID, &session.AgentID, &session.UserID,
			&session.Status, &session.Metadata, &session.CreatedAt, &session.UpdatedAt,
			&session.RunCount, &session.MessageCount, &session.ActiveRunCount, &session.LastActivityAt); err != nil {
			return nil, fmt.Errorf("scan session: %w", err)
		}
		sessions = append(sessions, session)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate sessions: %w", err)
	}
	return sessions, nil
}

// ListSessionRuns returns the complete visible conversation in chronological order.
func (s *RunStore) ListSessionRuns(ctx context.Context, tenantID, sessionID string, limit int) ([]agent.Run, error) {
	if limit <= 0 || limit > 200 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `SELECT `+runColumns+` FROM (
		SELECT * FROM agent_platform.agent_runs
		WHERE tenant_id=$1::text AND session_id=$2::uuid
		ORDER BY created_at DESC LIMIT $3
	) AS recent ORDER BY created_at`, tenantID, sessionID, limit)
	if err != nil {
		return nil, fmt.Errorf("list session runs: %w", err)
	}
	defer rows.Close()
	runs := make([]agent.Run, 0)
	for rows.Next() {
		run, scanErr := scanRun(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan session run: %w", scanErr)
		}
		runs = append(runs, run)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate session runs: %w", err)
	}
	if len(runs) == 0 {
		if _, err := s.GetSession(ctx, tenantID, sessionID); err != nil {
			return nil, err
		}
	}
	return runs, nil
}

// ListSessionMessages returns prior user inputs plus completed Assistant outputs
// in model-visible order. Failed/cancelled Runs therefore preserve the user's
// task without inventing an Assistant response.
func (s *RunStore) ListSessionMessages(
	ctx context.Context, tenantID, sessionID, currentRunID string, limit int,
) ([]model.Message, error) {
	if limit <= 0 || limit > 100 {
		limit = 20
	}
	// Read-repair closes the rolling-release window where an older binary can
	// commit a Run after the one-shot storage backfill without writing the new
	// Session message projection. The advisory/session locks also give this
	// query a stable boundary against concurrent sequence allocation.
	if _, err := s.ReconcileSessionMessages(ctx, tenantID, sessionID); err != nil {
		return nil, fmt.Errorf("repair session messages before read: %w", err)
	}
	rows, err := s.pool.Query(ctx, `
		SELECT history.role, history.content
		FROM (
			SELECT message.role, message.content, message.sequence
			FROM agent_platform.agent_session_messages AS message
			JOIN agent_platform.agent_runs AS run ON run.id=message.run_id
			WHERE message.tenant_id=$1::text AND message.session_id=$2::uuid
			  AND message.run_id<>$3::uuid
			  AND (message.role='user' OR run.status='completed')
			ORDER BY message.sequence DESC LIMIT ($4 * 2)
		) AS history ORDER BY history.sequence`, tenantID, sessionID, currentRunID, limit)
	if err != nil {
		return nil, fmt.Errorf("list session messages: %w", err)
	}
	defer rows.Close()
	messages := make([]model.Message, 0, limit*2)
	for rows.Next() {
		var role string
		var content json.RawMessage
		if err := rows.Scan(&role, &content); err != nil {
			return nil, fmt.Errorf("scan session messages: %w", err)
		}
		messages = append(messages, model.Message{Role: model.Role(role), Content: string(content)})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate session messages: %w", err)
	}
	if len(messages) == 0 {
		if _, err := s.GetSession(ctx, tenantID, sessionID); err != nil {
			return nil, err
		}
	}
	return messages, nil
}

func appendSessionMessageTx(ctx context.Context, tx pgx.Tx, tenantID, sessionID, runID, role, kind string, content json.RawMessage, createdAt time.Time) error {
	if len(content) == 0 {
		return errors.New("session message content is required")
	}
	if err := lockSessionMessagesTx(ctx, tx, tenantID, sessionID); err != nil {
		return err
	}
	var exists bool
	if err := tx.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM agent_platform.agent_sessions WHERE id=$1::uuid AND tenant_id=$2::text)`, sessionID, tenantID).Scan(&exists); err != nil {
		return fmt.Errorf("verify session for message append: %w", err)
	}
	if !exists {
		return agent.ErrSessionNotFound
	}
	if createdAt.IsZero() {
		createdAt = time.Now().UTC()
	}
	if _, err := tx.Exec(ctx, `
		INSERT INTO agent_platform.agent_session_messages
			(tenant_id,session_id,sequence,run_id,role,message_kind,content,content_hash,created_at)
		VALUES ($1::text,$2::uuid,
			(SELECT COALESCE(MAX(sequence),0)+1 FROM agent_platform.agent_session_messages WHERE session_id=$2::uuid),
			$3::uuid,$4::text,$5::text,$6::jsonb,
			encode(sha256(convert_to(($6::jsonb)::text,'UTF8')),'hex'),$7)
		ON CONFLICT DO NOTHING`, tenantID, sessionID, runID, role, kind, content, createdAt); err != nil {
		return fmt.Errorf("append session message: %w", err)
	}
	if _, err := tx.Exec(ctx, `UPDATE agent_platform.agent_sessions SET updated_at=GREATEST(updated_at,$3) WHERE id=$1::uuid AND tenant_id=$2::text`, sessionID, tenantID, createdAt); err != nil {
		return fmt.Errorf("touch session after message append: %w", err)
	}
	return nil
}

// lockSessionMessagesTx serializes sequence allocation without upgrading the
// foreign-key KEY SHARE lock held by a newly inserted Run. A row-level
// SELECT ... FOR UPDATE here can deadlock when two transactions create Runs
// for the same Session concurrently; the transaction-scoped advisory lock is
// also shared by rolling-release reconciliation.
func lockSessionMessagesTx(ctx context.Context, tx pgx.Tx, tenantID, sessionID string) error {
	if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1::text || ':' || $2::text, 0))`, tenantID, sessionID); err != nil {
		return fmt.Errorf("lock session message sequence: %w", err)
	}
	return nil
}

func scanSession(row rowScanner) (agent.Session, error) {
	var session agent.Session
	err := row.Scan(&session.ID, &session.TenantID, &session.AgentID, &session.UserID,
		&session.Status, &session.Metadata, &session.CreatedAt, &session.UpdatedAt)
	return session, err
}
