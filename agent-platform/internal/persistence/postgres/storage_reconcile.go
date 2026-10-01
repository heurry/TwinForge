package postgres

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
)

const storageReconcileSessionBatchSize = 256

// StorageReconcileResult describes durable projection repairs performed from
// canonical Run and checkpoint facts. The counters can be logged or exported
// as reconciliation metrics.
type StorageReconcileResult struct {
	SessionsScanned         int   `json:"sessions_scanned"`
	SessionMessagesInserted int64 `json:"session_messages_inserted"`
	RunStatesUpserted       int64 `json:"run_states_upserted"`
}

type storageSessionIdentity struct {
	tenantID  string
	sessionID string
}

// ReconcileStorageFoundation repairs projections that an older binary may
// have omitted during a rolling release. It is safe for multiple replicas to
// call concurrently: state repair is monotonic by event_seq and Session
// message repair uses the same transaction-scoped advisory lock as the normal
// append path.
//
// This method is deliberately repeatable rather than migration-only. An old
// writer can commit after a migration's statement snapshot, so callers should
// run it at startup and periodically until a rollout has converged.
func (s *RunStore) ReconcileStorageFoundation(ctx context.Context) (StorageReconcileResult, error) {
	result := StorageReconcileResult{}

	states, err := s.reconcileRunStates(ctx)
	if err != nil {
		return result, err
	}
	result.RunStatesUpserted = states

	for {
		sessions, err := s.listSessionsMissingMessages(ctx, storageReconcileSessionBatchSize)
		if err != nil {
			return result, err
		}
		if len(sessions) == 0 {
			return result, nil
		}

		insertedThisBatch := int64(0)
		for _, session := range sessions {
			inserted, err := s.ReconcileSessionMessages(ctx, session.tenantID, session.sessionID)
			if err != nil {
				return result, fmt.Errorf("reconcile session %s: %w", session.sessionID, err)
			}
			result.SessionsScanned++
			result.SessionMessagesInserted += inserted
			insertedThisBatch += inserted
		}

		// Another replica may have repaired every Session between candidate
		// discovery and lock acquisition. Re-query before deciding whether the
		// zero-progress batch is benign or indicates malformed persistent data.
		if insertedThisBatch == 0 {
			remaining, err := s.listSessionsMissingMessages(ctx, 1)
			if err != nil {
				return result, err
			}
			if len(remaining) == 0 {
				return result, nil
			}
			return result, fmt.Errorf("session message reconciliation made no progress for session %s", remaining[0].sessionID)
		}
	}
}

// ReconcileSessionMessages performs read-repair for one tenant-scoped
// Session. Missing Run inputs and completed outputs are inserted, then the
// Session sequence is normalized to canonical Run order. Reordering matters
// when an old transaction began before the storage migration but committed
// after a newer Run had already been projected.
func (s *RunStore) ReconcileSessionMessages(ctx context.Context, tenantID, sessionID string) (int64, error) {
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return 0, fmt.Errorf("begin session message reconciliation: %w", err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()

	if err := lockSessionMessagesTx(ctx, tx, tenantID, sessionID); err != nil {
		return 0, err
	}
	// New writers acquire the advisory lock before their Run INSERT. Taking
	// the Session row lock second also waits for a pre-upgrade binary that used
	// SELECT ... FOR UPDATE, without reintroducing advisory/FK lock inversion.
	var locked bool
	if err := tx.QueryRow(ctx, `
		SELECT true FROM agent_platform.agent_sessions
		WHERE id=$1::uuid AND tenant_id=$2::text
		FOR UPDATE`, sessionID, tenantID).Scan(&locked); errors.Is(err, pgx.ErrNoRows) {
		return 0, agent.ErrSessionNotFound
	} else if err != nil {
		return 0, fmt.Errorf("lock session for message reconciliation: %w", err)
	}

	tag, err := tx.Exec(ctx, `
		WITH missing AS (
			SELECT run.tenant_id, run.session_id, run.id AS run_id,
				'user'::text AS role, 'run_input'::text AS message_kind,
				run.input AS content, run.created_at AS message_at,
				run.created_at AS sort_at, 0 AS role_position
			FROM agent_platform.agent_runs AS run
			WHERE run.tenant_id=$1::text AND run.session_id=$2::uuid
			  AND NOT EXISTS (
				SELECT 1 FROM agent_platform.agent_session_messages AS message
				WHERE message.run_id=run.id AND message.message_kind='run_input'
			  )
			UNION ALL
			SELECT run.tenant_id, run.session_id, run.id AS run_id,
				'assistant'::text AS role, 'run_output'::text AS message_kind,
				run.output AS content, COALESCE(run.finished_at, run.updated_at) AS message_at,
				run.created_at AS sort_at, 2 AS role_position
			FROM agent_platform.agent_runs AS run
			WHERE run.tenant_id=$1::text AND run.session_id=$2::uuid
			  AND run.status='completed' AND run.output IS NOT NULL
			  AND NOT EXISTS (
				SELECT 1 FROM agent_platform.agent_session_messages AS message
				WHERE message.run_id=run.id AND message.message_kind='run_output'
			  )
		), numbered AS (
			SELECT missing.*,
				row_number() OVER (ORDER BY sort_at, run_id, role_position) AS sequence_offset
			FROM missing
		), boundary AS (
			SELECT COALESCE(MAX(sequence), 0) AS max_sequence
			FROM agent_platform.agent_session_messages
			WHERE session_id=$2::uuid
		)
		INSERT INTO agent_platform.agent_session_messages
			(tenant_id,session_id,sequence,run_id,role,message_kind,content,content_hash,created_at)
		SELECT numbered.tenant_id, numbered.session_id,
			boundary.max_sequence + numbered.sequence_offset,
			numbered.run_id, numbered.role, numbered.message_kind, numbered.content,
			encode(sha256(convert_to(numbered.content::text,'UTF8')),'hex'), numbered.message_at
		FROM numbered CROSS JOIN boundary
		ON CONFLICT DO NOTHING`, tenantID, sessionID)
	if err != nil {
		return 0, fmt.Errorf("insert missing session messages: %w", err)
	}
	inserted := tag.RowsAffected()

	if inserted > 0 {
		// Move current sequences above the occupied range first. That keeps
		// UNIQUE(session_id, sequence) valid while the second UPDATE assigns
		// dense canonical positions.
		if _, err := tx.Exec(ctx, `
			WITH boundary AS (
				SELECT COALESCE(MAX(sequence),0) + count(*) + 1 AS shift
				FROM agent_platform.agent_session_messages
				WHERE session_id=$1::uuid
			)
			UPDATE agent_platform.agent_session_messages AS message
			SET sequence=message.sequence + boundary.shift
			FROM boundary
			WHERE message.session_id=$1::uuid`, sessionID); err != nil {
			return 0, fmt.Errorf("stage session message resequencing: %w", err)
		}
		if _, err := tx.Exec(ctx, `
			WITH ordered AS (
				SELECT message.id,
					row_number() OVER (
						ORDER BY run.created_at, run.id,
							CASE message.message_kind
								WHEN 'run_input' THEN 0
								WHEN 'user_input' THEN 1
								ELSE 2
							END,
							message.created_at, message.id
					) AS canonical_sequence
				FROM agent_platform.agent_session_messages AS message
				JOIN agent_platform.agent_runs AS run ON run.id=message.run_id
				WHERE message.session_id=$1::uuid
			)
			UPDATE agent_platform.agent_session_messages AS message
			SET sequence=ordered.canonical_sequence
			FROM ordered
			WHERE message.id=ordered.id`, sessionID); err != nil {
			return 0, fmt.Errorf("normalize session message sequence: %w", err)
		}
		if _, err := tx.Exec(ctx, `
			UPDATE agent_platform.agent_sessions AS session
			SET updated_at=GREATEST(session.updated_at, latest.message_at)
			FROM (
				SELECT MAX(created_at) AS message_at
				FROM agent_platform.agent_session_messages
				WHERE session_id=$1::uuid
			) AS latest
			WHERE session.id=$1::uuid AND latest.message_at IS NOT NULL`, sessionID); err != nil {
			return 0, fmt.Errorf("touch reconciled session: %w", err)
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return 0, fmt.Errorf("commit session message reconciliation: %w", err)
	}
	return inserted, nil
}

func (s *RunStore) listSessionsMissingMessages(ctx context.Context, limit int) ([]storageSessionIdentity, error) {
	if limit <= 0 {
		limit = storageReconcileSessionBatchSize
	}
	rows, err := s.pool.Query(ctx, `
		SELECT session.tenant_id, session.id::text
		FROM agent_platform.agent_sessions AS session
		WHERE EXISTS (
			SELECT 1
			FROM agent_platform.agent_runs AS run
			WHERE run.session_id=session.id
			  AND (
				NOT EXISTS (
					SELECT 1 FROM agent_platform.agent_session_messages AS input_message
					WHERE input_message.run_id=run.id AND input_message.message_kind='run_input'
				)
				OR (
					run.status='completed' AND run.output IS NOT NULL
					AND NOT EXISTS (
						SELECT 1 FROM agent_platform.agent_session_messages AS output_message
						WHERE output_message.run_id=run.id AND output_message.message_kind='run_output'
					)
				)
			  )
		)
		ORDER BY session.id
		LIMIT $1`, limit)
	if err != nil {
		return nil, fmt.Errorf("list sessions missing message projections: %w", err)
	}
	defer rows.Close()

	sessions := make([]storageSessionIdentity, 0, limit)
	for rows.Next() {
		var session storageSessionIdentity
		if err := rows.Scan(&session.tenantID, &session.sessionID); err != nil {
			return nil, fmt.Errorf("scan session missing message projection: %w", err)
		}
		sessions = append(sessions, session)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate sessions missing message projections: %w", err)
	}
	return sessions, nil
}

func (s *RunStore) reconcileRunStates(ctx context.Context) (int64, error) {
	rootTag, err := s.pool.Exec(ctx, `
		INSERT INTO agent_platform.agent_run_states
			(run_id,workflow_id,event_seq,state_seq,lease_token,state,context_summary,state_hash,scope_run_id,updated_at)
		SELECT DISTINCT ON (checkpoint.workflow_id)
			checkpoint.run_id, run.workflow_id, checkpoint.event_seq,
			COALESCE(NULLIF(checkpoint.state_seq,0),checkpoint.event_seq), checkpoint.lease_token,
			checkpoint.state, checkpoint.context_summary,
			encode(sha256(convert_to(checkpoint.state::text,'UTF8')),'hex'), NULL, now()
		FROM agent_platform.agent_checkpoints AS checkpoint
		JOIN agent_platform.agent_runs AS run ON run.id=checkpoint.run_id
		LEFT JOIN agent_platform.agent_run_states AS current_state
			ON current_state.workflow_id=checkpoint.workflow_id AND current_state.scope_run_id IS NULL
		WHERE run.parent_run_id IS NULL
		  AND (current_state.run_id IS NULL
		   OR COALESCE(NULLIF(checkpoint.state_seq,0),checkpoint.event_seq) > current_state.state_seq
		  )
		ORDER BY checkpoint.workflow_id,
			COALESCE(NULLIF(checkpoint.state_seq,0),checkpoint.event_seq) DESC,
			checkpoint.event_seq DESC, checkpoint.run_id
		ON CONFLICT(workflow_id) WHERE scope_run_id IS NULL DO UPDATE SET
			run_id=EXCLUDED.run_id,
			event_seq=EXCLUDED.event_seq,
			state_seq=EXCLUDED.state_seq,
			lease_token=EXCLUDED.lease_token,
			state=EXCLUDED.state,
			context_summary=EXCLUDED.context_summary,
			state_hash=EXCLUDED.state_hash,
			updated_at=EXCLUDED.updated_at
		WHERE agent_run_states.state_seq < EXCLUDED.state_seq`)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return 0, err
		}
		return 0, fmt.Errorf("reconcile current run states: %w", err)
	}
	childTag, err := s.pool.Exec(ctx, `
		INSERT INTO agent_platform.agent_run_states
			(run_id,workflow_id,event_seq,state_seq,lease_token,state,context_summary,state_hash,scope_run_id,updated_at)
		SELECT DISTINCT ON (checkpoint.run_id)
			checkpoint.run_id, run.workflow_id, checkpoint.event_seq,
			COALESCE(NULLIF(checkpoint.state_seq,0),checkpoint.event_seq), checkpoint.lease_token,
			checkpoint.state, checkpoint.context_summary,
			encode(sha256(convert_to(checkpoint.state::text,'UTF8')),'hex'), checkpoint.run_id, now()
		FROM agent_platform.agent_checkpoints AS checkpoint
		JOIN agent_platform.agent_runs AS run ON run.id=checkpoint.run_id
		LEFT JOIN agent_platform.agent_run_states AS current_state ON current_state.run_id=checkpoint.run_id
		WHERE run.parent_run_id IS NOT NULL
		  AND (current_state.run_id IS NULL
		   OR COALESCE(NULLIF(checkpoint.state_seq,0),checkpoint.event_seq) > current_state.state_seq)
		ORDER BY checkpoint.run_id,
			COALESCE(NULLIF(checkpoint.state_seq,0),checkpoint.event_seq) DESC,
			checkpoint.event_seq DESC
		ON CONFLICT(run_id) DO UPDATE SET
			workflow_id=EXCLUDED.workflow_id,
			event_seq=EXCLUDED.event_seq,
			state_seq=EXCLUDED.state_seq,
			lease_token=EXCLUDED.lease_token,
			state=EXCLUDED.state,
			context_summary=EXCLUDED.context_summary,
			state_hash=EXCLUDED.state_hash,
			scope_run_id=EXCLUDED.scope_run_id,
			updated_at=EXCLUDED.updated_at
		WHERE agent_run_states.state_seq < EXCLUDED.state_seq`)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return 0, err
		}
		return 0, fmt.Errorf("reconcile delegated run states: %w", err)
	}
	return rootTag.RowsAffected() + childTag.RowsAffected(), nil
}
