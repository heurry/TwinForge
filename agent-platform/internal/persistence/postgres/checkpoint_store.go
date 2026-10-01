package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/harness"
)

// SaveCheckpointFenced atomically appends CHECKPOINT_CREATED and persists its
// model-visible recovery state under the current Worker lease.
func (s *RunStore) SaveCheckpointFenced(
	ctx context.Context, lease agent.Lease, checkpoint harness.Checkpoint,
) error {
	if checkpoint.RunID != lease.RunID {
		return errors.New("checkpoint run_id does not match lease")
	}
	// Persist the attempt that wrote this generation. A checkpoint loaded from
	// an older Run is deliberately rebound by the runtime; the next write must
	// become same-attempt resumable for the new owner.
	checkpoint.SourceRunID = checkpoint.RunID
	checkpoint = harness.ProjectCheckpointForStorage(checkpoint)
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return fmt.Errorf("begin checkpoint: %w", err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	var tenantID, workflowID string
	var parentRunID *string
	if err := tx.QueryRow(ctx, `
		SELECT tenant_id,workflow_id::text,parent_run_id::text FROM agent_platform.agent_runs
		WHERE id=$1::uuid AND lease_owner=$2::text AND lease_token=$3 AND status='running'
		FOR UPDATE`, lease.RunID, lease.Owner, lease.Token).Scan(&tenantID, &workflowID, &parentRunID); errors.Is(err, pgx.ErrNoRows) {
		return agent.ErrLeaseLost
	} else if err != nil {
		return fmt.Errorf("verify checkpoint lease: %w", err)
	}
	// event_seq is local to the current Run Attempt. Serialize on the
	// Workflow row and allocate a monotonic state_seq across all attempts.
	var stateSeq int64
	if err := tx.QueryRow(ctx, `
		SELECT COALESCE(GREATEST(
			(SELECT MAX(state_seq) FROM agent_platform.agent_run_states WHERE workflow_id=$1::uuid),
			(SELECT MAX(state_seq) FROM agent_platform.agent_checkpoints WHERE workflow_id=$1::uuid),
			(SELECT MAX(event_seq) FROM agent_platform.agent_run_states WHERE workflow_id=$1::uuid),
			(SELECT MAX(event_seq) FROM agent_platform.agent_checkpoints WHERE workflow_id=$1::uuid)
		),0)+1
		FROM agent_platform.agent_workflows
		WHERE id=$1::uuid
		FOR UPDATE`, workflowID).Scan(&stateSeq); err != nil {
		return fmt.Errorf("allocate workflow checkpoint sequence: %w", err)
	}
	checkpoint.StateSeq = stateSeq
	state, err := json.Marshal(checkpoint)
	if err != nil {
		return fmt.Errorf("marshal checkpoint: %w", err)
	}
	committed, err := s.appendEventTx(ctx, tx, tenantID, event.Input{
		RunID: lease.RunID, WorkflowID: workflowID, CheckpointSeq: stateSeq, Type: event.CheckpointCreated,
		Turn: checkpoint.Turn,
		Payload: mustJSON(map[string]any{
			"next_step": checkpoint.NextStep, "completed": checkpoint.Completed, "workflow_id": workflowID, "state_seq": stateSeq,
		}),
	})
	if err != nil {
		return err
	}
	contextSummary := checkpointContextSummary(checkpoint)
	stateArgs := []any{lease.RunID, workflowID, committed.Sequence, stateSeq, lease.Token, state, contextSummary}
	var stateErr error
	if parentRunID == nil {
		_, stateErr = tx.Exec(ctx, `
			INSERT INTO agent_platform.agent_run_states
				(run_id,workflow_id,event_seq,state_seq,lease_token,state,context_summary,state_hash,scope_run_id)
			VALUES ($1::uuid,$2::uuid,$3,$4,$5,$6::jsonb,NULLIF($7::text,''),
				encode(sha256(convert_to(($6::jsonb)::text,'UTF8')),'hex'),NULL)
			ON CONFLICT(workflow_id) WHERE scope_run_id IS NULL DO UPDATE SET
				run_id=EXCLUDED.run_id,event_seq=EXCLUDED.event_seq,state_seq=EXCLUDED.state_seq,lease_token=EXCLUDED.lease_token,
				state=EXCLUDED.state,context_summary=EXCLUDED.context_summary,
				state_hash=EXCLUDED.state_hash,updated_at=now()
			WHERE agent_run_states.state_seq < EXCLUDED.state_seq`, stateArgs...)
	} else {
		_, stateErr = tx.Exec(ctx, `
			INSERT INTO agent_platform.agent_run_states
				(run_id,workflow_id,event_seq,state_seq,lease_token,state,context_summary,state_hash,scope_run_id)
			VALUES ($1::uuid,$2::uuid,$3,$4,$5,$6::jsonb,NULLIF($7::text,''),
				encode(sha256(convert_to(($6::jsonb)::text,'UTF8')),'hex'),$1::uuid)
			ON CONFLICT(run_id) DO UPDATE SET
				event_seq=EXCLUDED.event_seq,state_seq=EXCLUDED.state_seq,lease_token=EXCLUDED.lease_token,
				state=EXCLUDED.state,context_summary=EXCLUDED.context_summary,
				state_hash=EXCLUDED.state_hash,scope_run_id=EXCLUDED.scope_run_id,updated_at=now()
			WHERE agent_run_states.state_seq < EXCLUDED.state_seq`, stateArgs...)
	}
	if stateErr != nil {
		return fmt.Errorf("upsert current run state: %w", stateErr)
	}
	if parentRunID == nil {
		if _, err := tx.Exec(ctx, `
			UPDATE agent_platform.agent_workflows
			SET latest_state_seq=GREATEST(latest_state_seq,$2), updated_at=now()
			WHERE id=$1::uuid AND tenant_id=$3::text`, workflowID, stateSeq, tenantID); err != nil {
			return fmt.Errorf("project workflow state sequence: %w", err)
		}
	}
	// The current state row is sufficient for crash recovery. Preserve a full
	// historical snapshot only at terminal completion; the append-only
	// CHECKPOINT_CREATED event retains the intermediate audit trail without
	// duplicating the full message context after every model/tool boundary.
	if checkpoint.Completed {
		var checkpointID string
		if err := tx.QueryRow(ctx, `
			INSERT INTO agent_platform.agent_checkpoints (run_id,workflow_id,event_seq,state_seq,lease_token,state,context_summary)
			VALUES ($1::uuid,$2::uuid,$3,$4,$5,$6::jsonb,NULLIF($7::text,''))
			RETURNING id::text`,
			lease.RunID, workflowID, committed.Sequence, stateSeq, lease.Token, state, contextSummary).Scan(&checkpointID); err != nil {
			return fmt.Errorf("insert terminal checkpoint: %w", err)
		}
		if parentRunID == nil {
			if _, err := tx.Exec(ctx, `
				UPDATE agent_platform.agent_workflows
				SET latest_checkpoint_id=$2::uuid, latest_state_seq=GREATEST(latest_state_seq,$3), updated_at=now()
				WHERE id=$1::uuid AND tenant_id=$4::text`, workflowID, checkpointID, stateSeq, tenantID); err != nil {
				return fmt.Errorf("project latest checkpoint: %w", err)
			}
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit checkpoint: %w", err)
	}
	return nil
}

// LoadLatestCheckpoint returns the newest durable recovery snapshot.
func (s *RunStore) LoadLatestCheckpoint(ctx context.Context, runID string) (harness.Checkpoint, bool, error) {
	var workflowID string
	if err := s.pool.QueryRow(ctx, `SELECT workflow_id::text FROM agent_platform.agent_runs WHERE id=$1::uuid`, runID).Scan(&workflowID); errors.Is(err, pgx.ErrNoRows) {
		return harness.Checkpoint{}, false, nil
	} else if err != nil {
		return harness.Checkpoint{}, false, fmt.Errorf("load checkpoint workflow: %w", err)
	}
	return s.LoadLatestCheckpointForWorkflow(ctx, workflowID, runID)
}

// LoadLatestCheckpointForWorkflow restores root state across continuation Run
// attempts. Delegated children are deliberately scoped to their own Run so a
// Reviewer cannot overwrite the parent's pending delegation/tool checkpoint.
// The returned RunID is rebound only for a root continuation attempt.
func (s *RunStore) LoadLatestCheckpointForWorkflow(ctx context.Context, workflowID, targetRunID string) (harness.Checkpoint, bool, error) {
	var delegated bool
	if err := s.pool.QueryRow(ctx, `SELECT parent_run_id IS NOT NULL FROM agent_platform.agent_runs WHERE id=$1::uuid AND workflow_id=$2::uuid`, targetRunID, workflowID).Scan(&delegated); errors.Is(err, pgx.ErrNoRows) {
		return harness.Checkpoint{}, false, nil
	} else if err != nil {
		return harness.Checkpoint{}, false, fmt.Errorf("load checkpoint target scope: %w", err)
	}
	var raw json.RawMessage
	var sourceRunID string
	var stateSeq int64
	err := s.pool.QueryRow(ctx, `
		SELECT candidate.state,candidate.run_id::text,candidate.state_seq
		FROM (
			SELECT state.state, state.run_id, COALESCE(NULLIF(state.state_seq,0),state.event_seq) AS state_seq, 1 AS source_priority
			FROM agent_platform.agent_run_states AS state
			JOIN agent_platform.agent_runs AS state_run ON state_run.id=state.run_id
			WHERE state.workflow_id=$1::uuid
			  AND (($3::boolean AND state.run_id=$2::uuid)
			       OR (NOT $3::boolean AND state_run.parent_run_id IS NULL))
			UNION ALL
			SELECT checkpoint.state, checkpoint.run_id, COALESCE(NULLIF(checkpoint.state_seq,0),checkpoint.event_seq) AS state_seq, 0 AS source_priority
			FROM agent_platform.agent_checkpoints AS checkpoint
			JOIN agent_platform.agent_runs AS checkpoint_run ON checkpoint_run.id=checkpoint.run_id
			WHERE checkpoint.workflow_id=$1::uuid
			  AND (($3::boolean AND checkpoint.run_id=$2::uuid)
			       OR (NOT $3::boolean AND checkpoint_run.parent_run_id IS NULL))
		) AS candidate
		ORDER BY candidate.state_seq DESC, candidate.source_priority DESC
		LIMIT 1`, workflowID, targetRunID, delegated).Scan(&raw, &sourceRunID, &stateSeq)
	if errors.Is(err, pgx.ErrNoRows) {
		return harness.Checkpoint{}, false, nil
	}
	if err != nil {
		return harness.Checkpoint{}, false, fmt.Errorf("load latest checkpoint: %w", err)
	}
	var checkpoint harness.Checkpoint
	if err := json.Unmarshal(raw, &checkpoint); err != nil {
		return harness.Checkpoint{}, false, fmt.Errorf("decode latest checkpoint: %w", err)
	}
	checkpoint = harness.RestoreCheckpointFromStorage(checkpoint)
	checkpoint.SourceRunID = sourceRunID
	checkpoint.RunID = targetRunID
	if checkpoint.StateSeq == 0 {
		checkpoint.StateSeq = stateSeq
	}
	return checkpoint, true, nil
}

func checkpointContextSummary(checkpoint harness.Checkpoint) string {
	if len(checkpoint.ContextState) != 0 {
		var state struct {
			Summary string `json:"summary"`
		}
		if json.Unmarshal(checkpoint.ContextState, &state) == nil && strings.TrimSpace(state.Summary) != "" {
			return strings.TrimSpace(state.Summary)
		}
	}
	for index := len(checkpoint.Messages) - 1; index >= 0; index-- {
		content := strings.TrimSpace(checkpoint.Messages[index].TextContent())
		lower := strings.ToLower(content)
		if strings.HasPrefix(lower, "<context_summary>") || strings.HasPrefix(lower, "<context_collapse") {
			return content
		}
	}
	return ""
}

// FencedCheckpointSink binds snapshots to one Worker ownership generation.
type FencedCheckpointSink struct {
	store *RunStore
	lease agent.Lease
}

// NewFencedCheckpointSink creates a lease-guarded checkpoint sink.
func NewFencedCheckpointSink(store *RunStore, lease agent.Lease) *FencedCheckpointSink {
	return &FencedCheckpointSink{store: store, lease: lease}
}

// Save implements harness.CheckpointSink.
func (s *FencedCheckpointSink) Save(ctx context.Context, checkpoint harness.Checkpoint) error {
	return s.store.SaveCheckpointFenced(ctx, s.lease, checkpoint)
}
