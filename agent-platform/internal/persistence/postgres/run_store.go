package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"sort"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/embedding"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/contract"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/taskplan"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/workflow"
)

type ArtifactObjectStore interface {
	Enabled() bool
	Bucket() string
	Put(context.Context, string, []byte, string) error
	Get(context.Context, string) ([]byte, error)
}

type MemoryEmbeddingProvider interface {
	Enabled() bool
	Embed(context.Context, []string, bool) (embedding.Result, error)
	Status() embedding.Status
}

// RunStore persists Run ownership and append-only events.
type RunStore struct {
	pool          *pgxpool.Pool
	outboxEnabled bool
	artifactStore ArtifactObjectStore
	embeddings    MemoryEmbeddingProvider
}

// RegisterWorkerCapabilities records the executable contract of a Worker
// before it can claim work. The row is an operational handshake, not a source
// of truth for Agent configuration; it lets the API/ops console detect an API
// and Worker image mismatch instead of silently omitting a runtime tool.
func (s *RunStore) RegisterWorkerCapabilities(ctx context.Context, workerID, runtimeVersion, toolContractVersion, protocolVersion, capabilityHash string, capabilities []string) error {
	if strings.TrimSpace(workerID) == "" || strings.TrimSpace(runtimeVersion) == "" || strings.TrimSpace(toolContractVersion) == "" || strings.TrimSpace(protocolVersion) == "" || strings.TrimSpace(capabilityHash) == "" {
		return errors.New("worker capability identity is required")
	}
	encoded, err := json.Marshal(capabilities)
	if err != nil {
		return fmt.Errorf("encode worker capabilities: %w", err)
	}
	_, err = s.pool.Exec(ctx, `
		INSERT INTO agent_platform.agent_worker_capabilities
			(worker_id,runtime_version,tool_contract_version,protocol_version,capabilities,capability_hash,last_seen_at)
		VALUES($1,$2,$3,$4,$5::jsonb,$6,now())
		ON CONFLICT(worker_id) DO UPDATE SET
			runtime_version=EXCLUDED.runtime_version,
			tool_contract_version=EXCLUDED.tool_contract_version,
			protocol_version=EXCLUDED.protocol_version,
			capabilities=EXCLUDED.capabilities,
			capability_hash=EXCLUDED.capability_hash,
			last_seen_at=now()`, workerID, runtimeVersion, toolContractVersion, protocolVersion, encoded, capabilityHash)
	if err != nil {
		return fmt.Errorf("register worker capabilities: %w", err)
	}
	return nil
}

func (s *RunStore) ListWorkerCapabilities(ctx context.Context) ([]agent.WorkerCapability, error) {
	rows, err := s.pool.Query(ctx, `SELECT worker_id,runtime_version,tool_contract_version,protocol_version,capabilities,capability_hash,last_seen_at FROM agent_platform.agent_worker_capabilities ORDER BY worker_id`)
	if err != nil {
		return nil, fmt.Errorf("list worker capabilities: %w", err)
	}
	defer rows.Close()
	result := make([]agent.WorkerCapability, 0)
	for rows.Next() {
		var item agent.WorkerCapability
		var raw []byte
		if err := rows.Scan(&item.WorkerID, &item.RuntimeVersion, &item.ToolContractVersion, &item.ProtocolVersion, &raw, &item.CapabilityHash, &item.LastSeenAt); err != nil {
			return nil, fmt.Errorf("scan worker capability: %w", err)
		}
		if err := json.Unmarshal(raw, &item.Capabilities); err != nil {
			return nil, fmt.Errorf("decode worker capabilities: %w", err)
		}
		result = append(result, item)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate worker capabilities: %w", err)
	}
	return result, nil
}

// NewRunStore creates a PostgreSQL Run repository.
func NewRunStore(pool *pgxpool.Pool) *RunStore {
	return &RunStore{pool: pool}
}

// SetOutboxEnabled controls whether durable event transactions also enqueue a
// Redis wake-up. It must be configured once, before the store is shared with
// request handlers or workers. PostgreSQL events remain authoritative either
// way; disabling the outbox prevents unbounded pending rows when no relay is
// configured.
func (s *RunStore) SetOutboxEnabled(enabled bool) {
	s.outboxEnabled = enabled
}

// SetArtifactObjectStore switches new Artifact payloads from PostgreSQL bytea
// to immutable content-addressed objects. Metadata remains transactional in PG.
func (s *RunStore) SetArtifactObjectStore(store ArtifactObjectStore) {
	s.artifactStore = store
}

// SetMemoryEmbeddingProvider enables vector writes and hybrid recall. Lexical
// recall remains an explicit degradation path when the provider is unavailable.
func (s *RunStore) SetMemoryEmbeddingProvider(provider MemoryEmbeddingProvider) {
	s.embeddings = provider
}

// CreateRun atomically creates a queued Run and its first RUN_CREATED event.
func (s *RunStore) CreateRun(ctx context.Context, input agent.CreateRun) (agent.Run, error) {
	if strings.TrimSpace(input.TenantID) == "" || strings.TrimSpace(input.AgentVersionID) == "" {
		return agent.Run{}, errors.New("tenant_id and agent_version_id are required")
	}
	if !validJSONObject(input.Input) {
		return agent.Run{}, errors.New("input must be a JSON object")
	}
	if input.TriggerType == "" {
		input.TriggerType = "api"
	}
	routingIntent, err := normalizeWorkflowRoutingIntent(input.RoutingIntent, input.NewWorkflow)
	if err != nil {
		return agent.Run{}, err
	}
	if routingIntent == "new_task" {
		if input.WorkflowID != nil && strings.TrimSpace(*input.WorkflowID) != "" {
			return agent.Run{}, fmt.Errorf("%w: new_task cannot specify workflow_id", agent.ErrWorkflowRoutingInvalid)
		}
		input.NewWorkflow = true
	}
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return agent.Run{}, fmt.Errorf("begin create run: %w", err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	// Acquire the Session sequence lock before the Run INSERT obtains its
	// foreign-key KEY SHARE lock. Keeping this order consistent with rolling
	// reconciliation prevents advisory/FK lock inversion while still allowing
	// old binaries that use a Session row lock to finish safely.
	if input.SessionID != nil {
		if err := lockSessionMessagesTx(ctx, tx, input.TenantID, *input.SessionID); err != nil {
			return agent.Run{}, err
		}
	}
	workflowID := ""
	workflowCreated := false
	if input.WorkflowID != nil {
		workflowID = strings.TrimSpace(*input.WorkflowID)
	}
	if workflowID != "" {
		var valid bool
		err := tx.QueryRow(ctx, `
			SELECT true FROM agent_platform.agent_workflows
			WHERE id=$1::uuid AND tenant_id=$2::text
			  AND ($3::uuid IS NULL OR session_id=$3::uuid)
			FOR UPDATE`, workflowID, input.TenantID, input.SessionID).Scan(&valid)
		if errors.Is(err, pgx.ErrNoRows) {
			return agent.Run{}, agent.ErrRunBindingInvalid
		}
		if err != nil {
			return agent.Run{}, fmt.Errorf("validate workflow: %w", err)
		}
	} else if input.SessionID != nil && !input.NewWorkflow {
		// A normal follow-up can auto-bind only when there is exactly one
		// resumable Workflow. Choosing the newest one when several exist makes
		// a continuation silently mutate the wrong task.
		rows, err := tx.Query(ctx, `
			SELECT id::text,COALESCE(goal,''),status
			FROM agent_platform.agent_workflows
			WHERE tenant_id=$1::text AND session_id=$2::uuid
			  AND status NOT IN ('completed','cancelled','archived')
			ORDER BY updated_at DESC, id DESC
			FOR UPDATE`, input.TenantID, *input.SessionID)
		if err != nil {
			return agent.Run{}, fmt.Errorf("find session workflows: %w", err)
		}
		candidates := make([]agent.WorkflowCandidate, 0, 2)
		for rows.Next() {
			var candidate agent.WorkflowCandidate
			if err := rows.Scan(&candidate.ID, &candidate.Goal, &candidate.Status); err != nil {
				rows.Close()
				return agent.Run{}, fmt.Errorf("scan session workflow: %w", err)
			}
			candidates = append(candidates, candidate)
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return agent.Run{}, fmt.Errorf("iterate session workflows: %w", err)
		}
		rows.Close()
		if len(candidates) > 1 {
			return agent.Run{}, &agent.WorkflowAmbiguousError{Candidates: candidates}
		}
		if len(candidates) == 1 {
			workflowID = candidates[0].ID
		}
	}
	if workflowID == "" {
		if routingIntent == "continue" || routingIntent == "extend" || routingIntent == "replan" {
			return agent.Run{}, fmt.Errorf("%w: %s requires an existing workflow_id or one unambiguous active Workflow", agent.ErrWorkflowRoutingInvalid, routingIntent)
		}
		goal := workflowGoal(input.Input)
		if err := tx.QueryRow(ctx, `
			INSERT INTO agent_platform.agent_workflows
				(tenant_id,session_id,agent_version_id,status,phase,goal,workspace_id)
			VALUES($1::text,$2::uuid,$3::uuid,'active','created',NULLIF($4::text,''),gen_random_uuid()::text)
			RETURNING id::text`, input.TenantID, input.SessionID, input.AgentVersionID, goal).Scan(&workflowID); err != nil {
			return agent.Run{}, fmt.Errorf("create workflow: %w", err)
		}
		workflowCreated = true
	}
	if routingIntent == "auto" {
		if workflowCreated {
			routingIntent = "new_task"
		} else {
			routingIntent = "continue"
		}
	}

	row := tx.QueryRow(ctx, `
		INSERT INTO agent_platform.agent_runs (
			tenant_id, session_id, workflow_id, agent_version_id, trigger_type,
			input, binding_snapshot, created_by, traceparent
		)
		SELECT $1::text, $2::uuid, $3::uuid, version.id, $5::text, $6::jsonb,
			jsonb_build_object(
				'agent_version_id', version.id::text,
				'version', version.version,
				'spec_hash', version.spec_hash,
				'test_run', $9::boolean,
				'routing_intent', $10::text,
				'spec', version.spec
			),
			$7::text, $8::text
		FROM agent_platform.agent_versions AS version
		JOIN agent_platform.agent_definitions AS definition ON definition.id=version.agent_id
		WHERE version.id=$4::uuid
		  AND (version.status='published' OR ($9::boolean AND version.status='draft'))
		  AND definition.tenant_id=$1::text
		  AND ($2::uuid IS NULL OR EXISTS (
			SELECT 1 FROM agent_platform.agent_sessions AS session
			WHERE session.id=$2::uuid AND session.tenant_id=$1::text AND session.agent_id=version.agent_id
		  ))
		RETURNING `+runColumns,
		input.TenantID, input.SessionID, workflowID, input.AgentVersionID, input.TriggerType,
		input.Input, input.CreatedBy, input.TraceParent, input.AllowDraft, routingIntent,
	)
	run, err := scanRun(row)
	var constraintErr *pgconn.PgError
	if errors.As(err, &constraintErr) && constraintErr.Code == "23505" && constraintErr.ConstraintName == "uq_agent_runs_workflow_active" {
		return agent.Run{}, agent.ErrWorkflowBusy
	}
	if errors.Is(err, pgx.ErrNoRows) {
		return agent.Run{}, agent.ErrRunBindingInvalid
	}
	if err != nil {
		return agent.Run{}, fmt.Errorf("insert run: %w", err)
	}
	if _, err := tx.Exec(ctx, `
		UPDATE agent_platform.agent_workflows
		SET active_run_id=$2::uuid, status='active', phase='ready',
			execution_generation=execution_generation+CASE WHEN $4::boolean THEN 0 ELSE 1 END,
			updated_at=now()
		WHERE id=$1::uuid AND tenant_id=$3::text`, workflowID, run.ID, input.TenantID, workflowCreated); err != nil {
		return agent.Run{}, fmt.Errorf("activate workflow run: %w", err)
	}
	var snapshot struct {
		Spec agent.Spec `json:"spec"`
	}
	_ = json.Unmarshal(run.BindingSnapshot, &snapshot)
	if err := contract.Validate(snapshot.Spec.InputSchema, run.Input); err != nil {
		return agent.Run{}, fmt.Errorf("%w: %v", agent.ErrRunInputInvalid, err)
	}
	createdPayload, err := json.Marshal(map[string]any{
		"agent_name":   snapshot.Spec.Name,
		"description":  snapshot.Spec.Description,
		"trigger_type": run.TriggerType,
		"input":        json.RawMessage(run.Input),
	})
	if err != nil {
		return agent.Run{}, fmt.Errorf("encode run created event: %w", err)
	}
	if _, err := s.appendEventTx(ctx, tx, run.TenantID, event.Input{RunID: run.ID, WorkflowID: workflowID, SessionID: sessionIDOf(run.SessionID), Type: event.RunCreated, Payload: createdPayload}); err != nil {
		return agent.Run{}, err
	}
	if workflowCreated {
		if _, err := s.appendEventTx(ctx, tx, run.TenantID, event.Input{
			RunID: run.ID, WorkflowID: workflowID, SessionID: sessionIDOf(run.SessionID), Type: event.WorkflowCreated,
			Payload: mustJSON(map[string]any{"workflow_id": workflowID, "goal": workflowGoal(input.Input)}),
		}); err != nil {
			return agent.Run{}, err
		}
	}
	if _, err := s.appendEventTx(ctx, tx, run.TenantID, event.Input{
		RunID: run.ID, WorkflowID: workflowID, SessionID: sessionIDOf(run.SessionID), Type: event.RunAttemptCreated,
		Payload: mustJSON(map[string]any{"workflow_id": workflowID, "attempt": run.Attempt, "trigger_type": run.TriggerType}),
	}); err != nil {
		return agent.Run{}, err
	}
	if _, err := s.appendEventTx(ctx, tx, run.TenantID, event.Input{
		RunID: run.ID, WorkflowID: workflowID, SessionID: sessionIDOf(run.SessionID), Type: event.WorkflowRoutingDecided,
		Payload: mustJSON(map[string]any{
			"workflow_id": workflowID, "routing_intent": routingIntent,
			"explicit_workflow_id": input.WorkflowID != nil,
			"new_workflow":         input.NewWorkflow,
		}),
	}); err != nil {
		return agent.Run{}, err
	}
	if run.SessionID != nil {
		if err := appendSessionMessageTx(ctx, tx, run.TenantID, *run.SessionID, run.ID, string(model.RoleUser), "run_input", run.Input, run.CreatedAt); err != nil {
			return agent.Run{}, err
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return agent.Run{}, fmt.Errorf("commit create run: %w", err)
	}
	return run, nil
}

func normalizeWorkflowRoutingIntent(value string, newWorkflow bool) (string, error) {
	if newWorkflow {
		return "new_task", nil
	}
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "auto":
		return "auto", nil
	case "resume", "new_turn", "continue", "continuation":
		return "continue", nil
	case "extend":
		return "extend", nil
	case "replan":
		return "replan", nil
	case "new_workflow", "new_task":
		return "new_task", nil
	default:
		return "", fmt.Errorf("%w: unsupported routing_intent %q", agent.ErrWorkflowRoutingInvalid, value)
	}
}

func workflowGoal(input json.RawMessage) string {
	var payload map[string]any
	if json.Unmarshal(input, &payload) != nil {
		return ""
	}
	for _, key := range []string{"goal", "question", "task", "prompt"} {
		if value, ok := payload[key].(string); ok && strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func sessionIDOf(sessionID *string) string {
	if sessionID == nil {
		return ""
	}
	return strings.TrimSpace(*sessionID)
}

// GetRunForTenant returns a Run only within the caller's tenant boundary.
func (s *RunStore) GetRunForTenant(ctx context.Context, tenantID, runID string) (agent.Run, error) {
	run, err := scanRun(s.pool.QueryRow(ctx,
		`SELECT `+runColumns+` FROM agent_platform.agent_runs WHERE id=$1::uuid AND tenant_id=$2`,
		runID, tenantID))
	if errors.Is(err, pgx.ErrNoRows) {
		return agent.Run{}, agent.ErrRunNotFound
	}
	if err != nil {
		return agent.Run{}, fmt.Errorf("get tenant run: %w", err)
	}
	return run, nil
}

// ListEventsForTenant returns a Run timeline after an exclusive sequence.
func (s *RunStore) ListEventsForTenant(
	ctx context.Context, tenantID, runID string, after int64, limit int,
) ([]event.Event, error) {
	if limit <= 0 || limit > 1000 {
		limit = 200
	}
	rows, err := s.pool.Query(ctx, `
		SELECT event.run_id::text, event.event_type, event.schema_version,
			COALESCE(event.turn_no, 0), COALESCE(event.step_no, 0), COALESCE(event.call_id, ''),
			event.payload, event.seq, event.workflow_seq, event.created_at, event.workflow_id::text
		FROM agent_platform.agent_events AS event
		JOIN agent_platform.agent_runs AS run ON run.id=event.run_id
		WHERE event.run_id=$1::uuid AND run.tenant_id=$2::text AND event.seq>$3
		ORDER BY event.seq LIMIT $4`, runID, tenantID, after, limit)
	if err != nil {
		return nil, fmt.Errorf("list run events: %w", err)
	}
	defer rows.Close()
	events := make([]event.Event, 0)
	for rows.Next() {
		var committed event.Event
		var authoritativeWorkflowID string
		if err := rows.Scan(
			&committed.RunID, &committed.Type, &committed.SchemaVersion,
			&committed.Turn, &committed.Step, &committed.CallID, &committed.Payload,
			&committed.Sequence, &committed.WorkflowSequence, &committed.CreatedAt, &authoritativeWorkflowID,
		); err != nil {
			return nil, fmt.Errorf("scan run event: %w", err)
		}
		_, committed.PlanNodeID, committed.DecisionCycle, committed.ActionID = event.SemanticFromPayload(committed.Payload)
		// Run ownership is authoritative for both pre-Workflow events and events
		// written during the migration window with workflow_id=run_id. Preserve
		// append-only payload bytes and repair only the read projection.
		committed.WorkflowID = authoritativeWorkflowID
		if committed.DecisionCycle == 0 {
			committed.DecisionCycle = committed.Turn
		}
		if committed.ActionID == "" {
			committed.ActionID = committed.CallID
		}
		events = append(events, committed)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate run events: %w", err)
	}
	if len(events) == 0 {
		if _, err := s.GetRunForTenant(ctx, tenantID, runID); err != nil {
			return nil, err
		}
	}
	return events, nil
}

// ListWorkflowEventsForTenant returns the durable cross-Run timeline for one
// long-lived task. after is a Workflow cursor, not a Run-local event sequence.
func (s *RunStore) ListWorkflowEventsForTenant(
	ctx context.Context, tenantID, workflowID string, after int64, limit int,
) ([]event.Event, error) {
	if limit <= 0 || limit > 1000 {
		limit = 200
	}
	rows, err := s.pool.Query(ctx, `
		SELECT event.run_id::text,event.event_type,event.schema_version,
		       COALESCE(event.turn_no,0),COALESCE(event.step_no,0),COALESCE(event.call_id,''),
		       event.payload,event.seq,event.workflow_seq,event.created_at,event.workflow_id::text
		FROM agent_platform.agent_events AS event
		JOIN agent_platform.agent_workflows AS workflow ON workflow.id=event.workflow_id
		WHERE event.workflow_id=$1::uuid AND workflow.tenant_id=$2::text
		  AND event.workflow_seq>$3
		ORDER BY event.workflow_seq LIMIT $4`, workflowID, tenantID, after, limit)
	if err != nil {
		return nil, fmt.Errorf("list workflow events: %w", err)
	}
	defer rows.Close()
	events := make([]event.Event, 0)
	for rows.Next() {
		var committed event.Event
		if err := rows.Scan(
			&committed.RunID, &committed.Type, &committed.SchemaVersion,
			&committed.Turn, &committed.Step, &committed.CallID, &committed.Payload,
			&committed.Sequence, &committed.WorkflowSequence, &committed.CreatedAt, &committed.WorkflowID,
		); err != nil {
			return nil, fmt.Errorf("scan workflow event: %w", err)
		}
		_, committed.PlanNodeID, committed.DecisionCycle, committed.ActionID = event.SemanticFromPayload(committed.Payload)
		if committed.DecisionCycle == 0 {
			committed.DecisionCycle = committed.Turn
		}
		if committed.ActionID == "" {
			committed.ActionID = committed.CallID
		}
		events = append(events, committed)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate workflow events: %w", err)
	}
	if len(events) == 0 {
		var exists bool
		if err := s.pool.QueryRow(ctx, `SELECT true FROM agent_platform.agent_workflows WHERE id=$1::uuid AND tenant_id=$2::text`, workflowID, tenantID).Scan(&exists); errors.Is(err, pgx.ErrNoRows) {
			return nil, agent.ErrRunNotFound
		} else if err != nil {
			return nil, fmt.Errorf("validate workflow timeline: %w", err)
		}
	}
	return events, nil
}

// ValidateSuccessfulToolEvidence verifies that every platform-bound receipt
// refers to a committed TOOL_COMPLETED event in the same tenant-scoped Run.
// Plan evidence can therefore be rendered as a clickable trace reference
// instead of trusting an ungrounded text claim.
func (s *RunStore) ValidateSuccessfulToolEvidence(ctx context.Context, tenantID, runID string, callIDs []string) error {
	if len(callIDs) == 0 {
		return errors.New("passed acceptance criterion requires at least one evidence_call_id")
	}
	wanted := make(map[string]struct{}, len(callIDs))
	evidenceRecords := make([]successfulToolEvidence, 0, len(callIDs))
	for _, callID := range callIDs {
		callID = strings.TrimSpace(callID)
		if callID == "" {
			return errors.New("evidence_call_id must not be empty")
		}
		wanted[callID] = struct{}{}
	}
	rows, err := s.pool.Query(ctx, `
		SELECT DISTINCT event.call_id, event.seq, COALESCE(event.payload->>'name',''),
			COALESCE(event.payload->'arguments','{}'::jsonb)
		FROM agent_platform.agent_events AS event
		JOIN agent_platform.agent_runs AS run ON run.id=event.run_id
		LEFT JOIN agent_platform.agent_tool_executions AS execution
		  ON execution.run_id=event.run_id AND execution.call_id=event.call_id
		WHERE event.run_id=$1::uuid AND run.tenant_id=$2::text
		  AND event.event_type=$3::text AND event.call_id=ANY($4::text[])
		  AND (execution.id IS NULL OR execution.application_status='applied')
		  AND COALESCE(event.payload->>'name','') NOT IN ('update_plan','update_plan_step','ask_user')`,
		runID, tenantID, string(event.ToolCompleted), callIDs)
	if err != nil {
		return fmt.Errorf("validate Tool evidence: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var evidence successfulToolEvidence
		if err := rows.Scan(&evidence.CallID, &evidence.Sequence, &evidence.Name, &evidence.Arguments); err != nil {
			return fmt.Errorf("scan Tool evidence: %w", err)
		}
		delete(wanted, evidence.CallID)
		evidenceRecords = append(evidenceRecords, evidence)
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("iterate Tool evidence: %w", err)
	}
	if len(wanted) != 0 {
		missing := make([]string, 0, len(wanted))
		for callID := range wanted {
			missing = append(missing, callID)
		}
		sort.Strings(missing)
		return fmt.Errorf("evidence_call_ids are not successful Tool events in this Run: %s", strings.Join(missing, ", "))
	}
	mutationRows, err := s.pool.Query(ctx, `
		SELECT event.seq, COALESCE(event.payload->>'name',''),
			COALESCE(event.payload->'arguments'->>'path', event.payload->'arguments'->>'target_path','')
		FROM agent_platform.agent_events AS event
		JOIN agent_platform.agent_runs AS run ON run.id=event.run_id
		WHERE event.run_id=$1::uuid AND run.tenant_id=$2::text
		  AND event.event_type=$3::text
		  AND COALESCE(event.payload->>'name','') IN ('write_file','append_file','edit_file','promote_file')
		ORDER BY event.seq`, runID, tenantID, string(event.ToolCompleted))
	if err != nil {
		return fmt.Errorf("load workspace mutations for evidence freshness: %w", err)
	}
	defer mutationRows.Close()
	latestMutation := make(map[string]int64)
	for mutationRows.Next() {
		var sequence int64
		var name, path string
		if err := mutationRows.Scan(&sequence, &name, &path); err != nil {
			return fmt.Errorf("scan workspace mutation: %w", err)
		}
		if path = canonicalEvidencePath(path); path != "" {
			latestMutation[path] = sequence
		}
	}
	if err := mutationRows.Err(); err != nil {
		return fmt.Errorf("iterate workspace mutations: %w", err)
	}
	if err := validateEvidenceFreshness(evidenceRecords, latestMutation); err != nil {
		return err
	}
	return nil
}

type successfulToolEvidence struct {
	ExecutionID string
	CallID      string
	Sequence    int64
	Name        string
	Arguments   json.RawMessage
	Result      json.RawMessage
}

// ResolveAcceptanceCriterionEvidence selects a platform-recorded successful
// Tool receipt from the current durable Todo. The model never supplies or
// copies call IDs: it only asks to close a criterion, while the platform owns
// provenance, semantic matching and freshness.
func (s *RunStore) ResolveAcceptanceCriterionEvidence(ctx context.Context, tenantID, runID, planStepID string, criterion taskplan.AcceptanceCriterion) (taskplan.AcceptanceCriterion, error) {
	verification, err := taskplan.NormalizeCriterionVerification(criterion.Description, criterion.Verification)
	if err != nil {
		return criterion, err
	}
	criterion.Verification = verification
	eligibleTools := verification.EvidenceTools()
	if eligibleTools == nil {
		eligibleTools = []string{}
	}
	activationSequence, err := s.planStepActivationSequence(ctx, tenantID, runID, planStepID)
	if err != nil {
		return criterion, err
	}
	rows, err := s.pool.Query(ctx, `
		SELECT COALESCE(execution.id::text,''),event.call_id,event.seq,
			COALESCE(event.payload->>'name',''),
			COALESCE(event.payload->'arguments','{}'::jsonb),
			COALESCE(event.payload->'result'->'content','{}'::jsonb)
		FROM agent_platform.agent_events AS event
		JOIN agent_platform.agent_runs AS run ON run.id=event.run_id
		LEFT JOIN agent_platform.agent_tool_executions AS execution
		  ON execution.run_id=event.run_id AND execution.call_id=event.call_id
		 AND execution.status='succeeded' AND execution.application_status='applied'
		WHERE run.tenant_id=$1::text AND event.run_id=$2::uuid
		  AND event.event_type=$5::text
		  AND COALESCE(event.payload->>'name','') NOT IN ('update_plan','update_plan_step','ask_user')
		  AND (execution.id IS NOT NULL OR NOT EXISTS (
			SELECT 1 FROM agent_platform.agent_tool_executions AS recorded
			WHERE recorded.run_id=event.run_id AND recorded.call_id=event.call_id
		  ))
		  AND (COALESCE(execution.plan_step_key,event.payload->>'plan_node_id')=$3::text OR
		       (COALESCE(execution.plan_step_key,event.payload->>'plan_node_id') IS NULL
		        AND $6::bigint>0 AND event.seq>$6::bigint))
		  AND (cardinality($4::text[])=0 OR COALESCE(event.payload->>'name','')=ANY($4::text[]))
		ORDER BY (
			COALESCE(NULLIF(event.payload->'arguments'->>'target_path',''),
			         event.payload->'arguments'->>'path','')=$7::text
		) DESC,(COALESCE(execution.plan_step_key,event.payload->>'plan_node_id')=$3::text) DESC NULLS LAST,event.seq DESC`, tenantID, runID, planStepID, eligibleTools, string(event.ToolCompleted), activationSequence, verification.Target)
	if err != nil {
		return criterion, fmt.Errorf("load platform-managed evidence: %w", err)
	}
	defer rows.Close()
	var candidates []successfulToolEvidence
	for rows.Next() {
		var candidate successfulToolEvidence
		if err := rows.Scan(&candidate.ExecutionID, &candidate.CallID, &candidate.Sequence, &candidate.Name, &candidate.Arguments, &candidate.Result); err != nil {
			return criterion, fmt.Errorf("scan platform-managed evidence: %w", err)
		}
		candidates = append(candidates, candidate)
	}
	if err := rows.Err(); err != nil {
		return criterion, fmt.Errorf("iterate platform-managed evidence: %w", err)
	}
	candidate, matched, rejected := selectVerificationCandidate(verification, candidates)
	action := taskplan.VerificationActionHint(verification)
	if !matched {
		return criterion, taskplan.NewVerificationFailure(taskplan.VerificationReasonEvidenceMissing,
			fmt.Errorf("no exact receipt matched criterion %q in plan step %q: inspected=%d candidate_rejected=%d eligible_tools=%s; %s", criterion.ID, planStepID, len(candidates), rejected, strings.Join(eligibleTools, "/"), action))
	}
	resolved := criterion
	resolved.Status = taskplan.CriterionPassed
	resolved.EvidenceCallIDs = []string{candidate.CallID}
	resolved.Evidence = fmt.Sprintf("平台自动验证：%s 成功（call %s）", candidate.Name, candidate.CallID)
	if err := s.ValidateAcceptanceCriterionEvidence(ctx, tenantID, runID, resolved); err == nil {
		return resolved, nil
	} else {
		// Only an exact, structurally matching receipt becomes a formal attempt.
		// Unrelated candidates above are selection diagnostics, not failures.
		if persistErr := s.recordVerificationFailure(ctx, tenantID, runID, planStepID, criterion, candidate, err); persistErr != nil {
			return criterion, fmt.Errorf("record failed verification attempt: %w", persistErr)
		}
		return criterion, taskplan.NewVerificationFailure(taskplan.VerificationReasonAssertionFailed,
			fmt.Errorf("latest exact %s receipt for criterion %q failed verification after rejecting %d unrelated candidate(s): %v; %s", candidate.Name, criterion.ID, rejected, err, action))
	}
}

// selectVerificationCandidate separates evidence discovery from a formal
// attempt. Candidates arrive newest-first (with exact raw paths prioritized),
// and only the newest structurally matching receipt is eligible for verdict
// evaluation. Older success cannot override a newer failure for the same
// contract, and unrelated receipts never become failed attempts.
func selectVerificationCandidate(verification taskplan.VerificationSpec, candidates []successfulToolEvidence) (successfulToolEvidence, bool, int) {
	rejected := 0
	for _, candidate := range candidates {
		receipt := taskplan.EvidenceReceipt{Tool: candidate.Name, Arguments: candidate.Arguments, Result: candidate.Result}
		if taskplan.VerificationCandidateMatches(verification, receipt) {
			return candidate, true, rejected
		}
		rejected++
	}
	return successfulToolEvidence{}, false, rejected
}

// planStepActivationSequence bounds rolling-upgrade compatibility. Tool rows
// written by an older Worker have no plan_step_key, so they are eligible only
// when their committed event followed the current Todo's latest transition to
// an active state. New Workers always use the explicit key.
func (s *RunStore) planStepActivationSequence(ctx context.Context, tenantID, runID, planStepID string) (int64, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT event.seq,event.payload
		FROM agent_platform.agent_events AS event
		JOIN agent_platform.agent_runs AS run ON run.id=event.run_id
		WHERE event.run_id=$1::uuid AND run.tenant_id=$2::text
		  AND event.event_type=ANY($3::text[])
		ORDER BY event.seq`, runID, tenantID, []string{string(event.PlanCreated), string(event.PlanUpdated)})
	if err != nil {
		return 0, fmt.Errorf("load plan activation history: %w", err)
	}
	defer rows.Close()
	var activation int64
	wasActive := false
	for rows.Next() {
		var sequence int64
		var payload struct {
			Steps []taskplan.Step `json:"steps"`
		}
		var raw json.RawMessage
		if err := rows.Scan(&sequence, &raw); err != nil {
			return 0, fmt.Errorf("scan plan activation history: %w", err)
		}
		if err := json.Unmarshal(raw, &payload); err != nil {
			return 0, fmt.Errorf("decode plan activation history: %w", err)
		}
		isActive := false
		for _, step := range payload.Steps {
			if step.ID == planStepID && (step.Status == taskplan.StatusInProgress || step.Status == taskplan.StatusBlocked) {
				isActive = true
				break
			}
		}
		if isActive && !wasActive {
			activation = sequence
		}
		wasActive = isActive
	}
	if err := rows.Err(); err != nil {
		return 0, fmt.Errorf("iterate plan activation history: %w", err)
	}
	if !wasActive {
		return 0, nil
	}
	return activation, nil
}

// ValidateAcceptanceCriterionEvidence validates both receipt provenance and
// semantic compatibility. A successful call is insufficient when it proves a
// different property (for example py_compile cannot prove that a GUI starts).
func (s *RunStore) ValidateAcceptanceCriterionEvidence(ctx context.Context, tenantID, runID string, criterion taskplan.AcceptanceCriterion) error {
	if err := s.ValidateSuccessfulToolEvidence(ctx, tenantID, runID, criterion.EvidenceCallIDs); err != nil {
		return err
	}
	rows, err := s.pool.Query(ctx, `
		SELECT DISTINCT event.call_id, event.seq, COALESCE(event.payload->>'name',''),
			COALESCE(event.payload->'arguments','{}'::jsonb),
			COALESCE(event.payload->'result'->'content','{}'::jsonb)
		FROM agent_platform.agent_events AS event
		JOIN agent_platform.agent_runs AS run ON run.id=event.run_id
		WHERE event.run_id=$1::uuid AND run.tenant_id=$2::text
		  AND event.event_type=$3::text AND event.call_id=ANY($4::text[])`,
		runID, tenantID, string(event.ToolCompleted), criterion.EvidenceCallIDs)
	if err != nil {
		return fmt.Errorf("load typed Tool evidence: %w", err)
	}
	defer rows.Close()
	records := make([]successfulToolEvidence, 0, len(criterion.EvidenceCallIDs))
	for rows.Next() {
		var record successfulToolEvidence
		if err := rows.Scan(&record.CallID, &record.Sequence, &record.Name, &record.Arguments, &record.Result); err != nil {
			return fmt.Errorf("scan typed Tool evidence: %w", err)
		}
		records = append(records, record)
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("iterate typed Tool evidence: %w", err)
	}
	return validateEvidenceSemantics(criterion, records)
}

func validateEvidenceSemantics(criterion taskplan.AcceptanceCriterion, records []successfulToolEvidence) error {
	verification, err := taskplan.NormalizeCriterionVerification(criterion.Description, criterion.Verification)
	if err != nil {
		return err
	}
	if verification.Kind == "" {
		// Legacy plans predate typed contracts. Retain provenance/freshness checks,
		// but close the two known unsafe equivalences from production traces.
		lower := strings.ToLower(criterion.Description)
		if strings.Contains(lower, "可运行") || strings.Contains(lower, "直接运行") || strings.Contains(lower, "runnable") || strings.Contains(lower, "run successfully") {
			verification.Kind = "command_exit_zero"
			verification.Target = firstEvidencePath(records)
		} else {
			return nil
		}
	}
	receipts := make([]taskplan.EvidenceReceipt, 0, len(records))
	for _, record := range records {
		receipts = append(receipts, taskplan.EvidenceReceipt{
			Tool: record.Name, Arguments: record.Arguments, Result: record.Result,
		})
	}
	if err := taskplan.ValidateVerification(verification); err != nil {
		return err
	}
	if err := taskplan.VerifyEvidence(verification, receipts); err != nil {
		return err
	}
	return nil
}

func firstEvidencePath(records []successfulToolEvidence) string {
	for _, record := range records {
		paths := evidencePaths(record.Name, record.Arguments)
		if len(paths) != 0 {
			return paths[0]
		}
	}
	return ""
}

func evidencePathMatches(name string, arguments map[string]any, target string) bool {
	value := canonicalEvidencePath(stringValue(arguments["path"]))
	if name == "promote_file" && value == "" {
		value = canonicalEvidencePath(stringValue(arguments["target_path"]))
	}
	if target == "" {
		return value != ""
	}
	return value == target
}

func isFileEvidence(name string) bool {
	switch name {
	case "read_file", "write_file", "append_file", "edit_file", "promote_file":
		return true
	default:
		return false
	}
}

func commandTargets(arguments map[string]any, target string) bool {
	if target == "" {
		return false
	}
	working := canonicalEvidencePath(stringValue(arguments["working_directory"]))
	for _, argument := range stringSlice(arguments["args"]) {
		candidate := canonicalEvidencePath(argument)
		if working != "" && working != "." {
			candidate = canonicalEvidencePath(working + "/" + candidate)
		}
		if candidate == target {
			return true
		}
	}
	return false
}

func isPythonCompile(arguments map[string]any) bool {
	args := stringSlice(arguments["args"])
	return len(args) >= 2 && args[0] == "-m" && (args[1] == "py_compile" || args[1] == "compileall")
}

func isTestCommand(arguments map[string]any) bool {
	command := strings.ToLower(path.Base(strings.TrimSpace(stringValue(arguments["command"]))))
	args := stringSlice(arguments["args"])
	if len(args) == 0 {
		return false
	}
	if (command == "python" || command == "python3") && len(args) >= 2 && args[0] == "-m" && (args[1] == "unittest" || args[1] == "pytest") {
		return true
	}
	if command == "pytest" || (command == "go" && args[0] == "test") ||
		(command == "cargo" && args[0] == "test") ||
		((command == "npm" || command == "pnpm" || command == "yarn") && args[0] == "test") {
		return true
	}
	if command != "python" && command != "python3" {
		return false
	}
	entry := strings.ToLower(path.Base(strings.TrimSpace(args[0])))
	return strings.HasSuffix(entry, ".py") &&
		(strings.Contains(entry, "test") || strings.Contains(entry, "smoke") || strings.Contains(entry, "verify"))
}

func stringValue(value any) string {
	text, _ := value.(string)
	return text
}

func stringSlice(value any) []string {
	values, _ := value.([]any)
	result := make([]string, 0, len(values))
	for _, item := range values {
		if text, ok := item.(string); ok {
			result = append(result, text)
		}
	}
	return result
}

func sliceLength(value any) int {
	values, _ := value.([]any)
	return len(values)
}

func validateEvidenceFreshness(records []successfulToolEvidence, latestMutation map[string]int64) error {
	for _, record := range records {
		for _, path := range evidencePaths(record.Name, record.Arguments) {
			if latest := latestMutation[path]; latest > record.Sequence {
				return fmt.Errorf("evidence_call_id %s is stale: %q changed after that Tool result", record.CallID, path)
			}
		}
	}
	return nil
}

func evidencePaths(name string, arguments json.RawMessage) []string {
	var input struct {
		Path       string   `json:"path"`
		TargetPath string   `json:"target_path"`
		Args       []string `json:"args"`
		WorkingDir string   `json:"working_directory"`
	}
	if json.Unmarshal(arguments, &input) != nil {
		return nil
	}
	switch name {
	case "read_file", "write_file", "append_file", "edit_file", "promote_file":
		if name == "promote_file" && input.Path == "" {
			input.Path = input.TargetPath
		}
		if path := canonicalEvidencePath(input.Path); path != "" {
			return []string{path}
		}
	case "run_command":
		seen := make(map[string]struct{})
		var result []string
		for _, argument := range input.Args {
			if strings.HasPrefix(argument, "-") {
				continue
			}
			candidate := canonicalEvidencePath(argument)
			if candidate == "" || (!strings.Contains(candidate, "/") && !strings.Contains(candidate, ".")) {
				continue
			}
			if working := canonicalEvidencePath(input.WorkingDir); working != "" && working != "." {
				candidate = canonicalEvidencePath(working + "/" + candidate)
			}
			if _, exists := seen[candidate]; !exists {
				seen[candidate] = struct{}{}
				result = append(result, candidate)
			}
		}
		return result
	}
	return nil
}

func canonicalEvidencePath(value string) string {
	value = strings.TrimSpace(strings.ReplaceAll(value, "\\", "/"))
	value = strings.TrimPrefix(value, "/workspace/")
	value = strings.TrimPrefix(value, "./")
	value = strings.TrimPrefix(value, "/")
	if value == "" {
		return ""
	}
	cleaned := path.Clean(value)
	if cleaned == "." || cleaned == ".." || strings.HasPrefix(cleaned, "../") {
		return ""
	}
	return cleaned
}

// ListRunsForTenant returns recent Runs for the Agent operations console.
func (s *RunStore) ListRunsForTenant(ctx context.Context, tenantID string, limit int) ([]agent.Run, error) {
	if limit <= 0 || limit > 200 {
		limit = 50
	}
	rows, err := s.pool.Query(ctx, `SELECT `+runColumns+` FROM agent_platform.agent_runs
		WHERE tenant_id=$1::text ORDER BY created_at DESC LIMIT $2`, tenantID, limit)
	if err != nil {
		return nil, fmt.Errorf("list tenant runs: %w", err)
	}
	defer rows.Close()
	runs := make([]agent.Run, 0)
	for rows.Next() {
		run, err := scanRun(rows)
		if err != nil {
			return nil, fmt.Errorf("scan tenant run: %w", err)
		}
		runs = append(runs, run)
	}
	return runs, rows.Err()
}

// ListWorkflowsForTenant returns task-level projections rather than flattening
// every Run attempt. A Session may contain multiple independent Workflows.
func (s *RunStore) ListWorkflowsForTenant(ctx context.Context, tenantID, sessionID string, limit int) ([]workflow.Workflow, error) {
	if limit <= 0 || limit > 200 {
		limit = 50
	}
	rows, err := s.pool.Query(ctx, `
		SELECT id::text,COALESCE(session_id::text,''),status,COALESCE(goal,''),
			workspace_id,COALESCE(active_run_id::text,''),COALESCE(active_plan_id::text,''),
			COALESCE(latest_checkpoint_id::text,''),latest_state_seq,created_at,updated_at
		FROM agent_platform.agent_workflows
		WHERE tenant_id=$1::text AND ($2::text='' OR session_id=NULLIF($2::text,'')::uuid)
		ORDER BY updated_at DESC,id DESC LIMIT $3`, tenantID, strings.TrimSpace(sessionID), limit)
	if err != nil {
		return nil, fmt.Errorf("list tenant workflows: %w", err)
	}
	defer rows.Close()
	items := make([]workflow.Workflow, 0)
	for rows.Next() {
		var item workflow.Workflow
		if err := rows.Scan(&item.ID, &item.SessionID, &item.Status, &item.Goal, &item.WorkspaceID, &item.ActiveRunID, &item.ActivePlanID, &item.LatestCheckpointID, &item.LatestStateSeq, &item.CreatedAt, &item.UpdatedAt); err != nil {
			return nil, fmt.Errorf("scan tenant workflow: %w", err)
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

// ListWorkflowRunsForTenant returns every Run attempt belonging to one
// Workflow in chronological order. Run IDs remain useful provenance, while
// the Workflow is the durable parent used for resume and history rendering.
func (s *RunStore) ListWorkflowRunsForTenant(ctx context.Context, tenantID, workflowID string, limit int) ([]agent.Run, error) {
	if limit <= 0 || limit > 200 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `SELECT `+runColumns+` FROM (
		SELECT * FROM agent_platform.agent_runs
		WHERE tenant_id=$1::text AND workflow_id=$2::uuid
		ORDER BY created_at DESC LIMIT $3
	) AS recent ORDER BY created_at`, tenantID, workflowID, limit)
	if err != nil {
		return nil, fmt.Errorf("list workflow runs: %w", err)
	}
	defer rows.Close()
	runs := make([]agent.Run, 0)
	for rows.Next() {
		run, scanErr := scanRun(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan workflow run: %w", scanErr)
		}
		runs = append(runs, run)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate workflow runs: %w", err)
	}
	return runs, nil
}

// ObservabilitySummary returns persisted execution telemetry, not process-local counters.
func (s *RunStore) ObservabilitySummary(ctx context.Context, tenantID string) (agent.ObservabilitySummary, error) {
	var summary agent.ObservabilitySummary
	err := s.pool.QueryRow(ctx, `
		SELECT count(*), count(*) FILTER (WHERE status='completed'),
			count(*) FILTER (WHERE status='failed'), count(*) FILTER (WHERE status IN ('queued','running','waiting_tool','waiting_approval','waiting_input','waiting_external')),
			COALESCE(avg(EXTRACT(EPOCH FROM (finished_at-started_at))*1000)
				FILTER (WHERE finished_at IS NOT NULL AND started_at IS NOT NULL), 0)
		FROM agent_platform.agent_runs WHERE tenant_id=$1::text`, tenantID).Scan(
		&summary.TotalRuns, &summary.CompletedRuns, &summary.FailedRuns, &summary.ActiveRuns, &summary.AverageRunLatencyMS,
	)
	if err != nil {
		return summary, fmt.Errorf("summarize runs: %w", err)
	}
	err = s.pool.QueryRow(ctx, `
		SELECT count(*), COALESCE(sum(input_tokens),0), COALESCE(sum(output_tokens),0),
			COALESCE(avg(latency_ms),0)
		FROM agent_platform.agent_model_calls AS call
		JOIN agent_platform.agent_runs AS run ON run.id=call.run_id
		WHERE run.tenant_id=$1::text`, tenantID).Scan(
		&summary.ModelCalls, &summary.InputTokens, &summary.OutputTokens, &summary.AverageModelLatencyMS,
	)
	if err != nil {
		return summary, fmt.Errorf("summarize model calls: %w", err)
	}
	err = s.pool.QueryRow(ctx, `
		SELECT count(*), count(*) FILTER (WHERE execution.status='failed')
		FROM agent_platform.agent_tool_executions AS execution
		WHERE execution.tenant_id=$1::text`, tenantID).Scan(&summary.ToolCalls, &summary.FailedToolCalls)
	if err != nil {
		return summary, fmt.Errorf("summarize tool calls: %w", err)
	}
	summary.ToolFailuresByCode = make(map[string]int64)
	rows, err := s.pool.Query(ctx, `
		SELECT COALESCE(NULLIF(error->>'error_code',''),'TOOL_EXECUTION_FAILED'), count(*)
		FROM agent_platform.agent_tool_executions
		WHERE tenant_id=$1::text AND status='failed'
		GROUP BY 1 ORDER BY count(*) DESC`, tenantID)
	if err != nil {
		return summary, fmt.Errorf("summarize tool failure codes: %w", err)
	}
	for rows.Next() {
		var code string
		var count int64
		if err := rows.Scan(&code, &count); err != nil {
			rows.Close()
			return summary, fmt.Errorf("scan tool failure code: %w", err)
		}
		summary.ToolFailuresByCode[code] = count
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return summary, fmt.Errorf("iterate tool failure codes: %w", err)
	}
	rows.Close()
	var protocol, verification, plan, compactions, beforeTokens, afterTokens int64
	err = s.pool.QueryRow(ctx, `
		SELECT
			count(*) FILTER (WHERE event_type IN ('MODEL_FAILED','MODEL_PROTOCOL_ERROR') AND lower(COALESCE(payload->>'error_code',payload->>'error','')) LIKE '%protocol%'),
			count(*) FILTER (WHERE event_type LIKE 'VERIFICATION%' OR event_type IN ('COMPLETION_BLOCKED','COMPLETION_GUARD_FAILED')),
			count(*) FILTER (WHERE event_type LIKE 'PLAN%' AND (payload->>'status'='failed' OR payload->>'error_code' IS NOT NULL)),
			count(*) FILTER (WHERE event_type='CONTEXT_COMPACTED'),
			COALESCE(sum(CASE WHEN payload->>'before_tokens' ~ '^[0-9]+$' THEN (payload->>'before_tokens')::bigint ELSE 0 END) FILTER (WHERE event_type='CONTEXT_COMPACTED'),0),
			COALESCE(sum(CASE WHEN payload->>'after_tokens' ~ '^[0-9]+$' THEN (payload->>'after_tokens')::bigint ELSE 0 END) FILTER (WHERE event_type='CONTEXT_COMPACTED'),0)
		FROM agent_platform.agent_events WHERE tenant_id=$1::text`, tenantID).Scan(&protocol, &verification, &plan, &compactions, &beforeTokens, &afterTokens)
	if err != nil {
		return summary, fmt.Errorf("summarize failure and compaction events: %w", err)
	}
	summary.ModelProtocolFailures, summary.VerificationFailures, summary.PlanFailures = protocol, verification, plan
	summary.CompactionCount, summary.CompactionBeforeTokens, summary.CompactionAfterTokens = compactions, beforeTokens, afterTokens
	return summary, nil
}

// AppendEventFenced serializes an execution event with the Run row and rejects
// a stale Worker generation before allocating the next event sequence.
func (s *RunStore) AppendEventFenced(ctx context.Context, lease agent.Lease, input event.Input) (event.Event, error) {
	if input.RunID != "" && input.RunID != lease.RunID {
		return event.Event{}, errors.New("event run_id does not match lease")
	}
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return event.Event{}, fmt.Errorf("begin fenced event: %w", err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	var tenantID string
	if err := tx.QueryRow(ctx, `
		SELECT tenant_id FROM agent_platform.agent_runs
		WHERE id=$1::uuid AND lease_owner=$2::text AND lease_token=$3 AND status='running'
		FOR UPDATE`, lease.RunID, lease.Owner, lease.Token).Scan(&tenantID); errors.Is(err, pgx.ErrNoRows) {
		return event.Event{}, agent.ErrLeaseLost
	} else if err != nil {
		return event.Event{}, fmt.Errorf("verify fenced event lease: %w", err)
	}
	input.RunID = lease.RunID
	committed, err := s.appendEventTx(ctx, tx, tenantID, input)
	if err != nil {
		return event.Event{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return event.Event{}, fmt.Errorf("commit fenced event: %w", err)
	}
	return committed, nil
}

// FencedEventSink binds event writes to one Worker lease generation.
type FencedEventSink struct {
	store *RunStore
	lease agent.Lease
}

// NewFencedEventSink creates a lease-guarded event sink for one claimed Run.
func NewFencedEventSink(store *RunStore, lease agent.Lease) *FencedEventSink {
	return &FencedEventSink{store: store, lease: lease}
}

// Append implements event.Sink.
func (s *FencedEventSink) Append(ctx context.Context, input event.Input) (event.Event, error) {
	return s.store.AppendEventFenced(ctx, s.lease, input)
}

// GetRun returns one durable Run.
func (s *RunStore) GetRun(ctx context.Context, runID string) (agent.Run, error) {
	run, err := scanRun(s.pool.QueryRow(ctx,
		`SELECT `+runColumns+` FROM agent_platform.agent_runs WHERE id=$1::uuid`, runID))
	if errors.Is(err, pgx.ErrNoRows) {
		return agent.Run{}, agent.ErrRunNotFound
	}
	if err != nil {
		return agent.Run{}, fmt.Errorf("get run: %w", err)
	}
	return run, nil
}

// FreezeModelResolution persists the first successful discovery result under
// the active Worker lease. A resumed or racing Worker receives the existing
// value and can never silently move the Run to another model deployment.
func (s *RunStore) FreezeModelResolution(
	ctx context.Context, lease agent.Lease, resolution agent.ModelResolution,
) (agent.ModelResolution, error) {
	if strings.TrimSpace(resolution.ServiceRef) == "" || strings.TrimSpace(resolution.ModelID) == "" ||
		strings.TrimSpace(resolution.ServiceConfigHash) == "" {
		return agent.ModelResolution{}, errors.New("resolved model service, model and config hash are required")
	}
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return agent.ModelResolution{}, fmt.Errorf("begin model resolution freeze: %w", err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()

	var tenantID string
	var existingJSON json.RawMessage
	err = tx.QueryRow(ctx, `
		SELECT tenant_id, COALESCE(binding_snapshot->'resolved_model', 'null'::jsonb)
		FROM agent_platform.agent_runs
		WHERE id=$1::uuid AND lease_owner=$2::text AND lease_token=$3 AND status='running'
		FOR UPDATE`, lease.RunID, lease.Owner, lease.Token).Scan(&tenantID, &existingJSON)
	if errors.Is(err, pgx.ErrNoRows) {
		return agent.ModelResolution{}, agent.ErrLeaseLost
	}
	if err != nil {
		return agent.ModelResolution{}, fmt.Errorf("lock run for model resolution: %w", err)
	}
	if string(existingJSON) != "null" {
		var existing agent.ModelResolution
		if err := json.Unmarshal(existingJSON, &existing); err != nil {
			return agent.ModelResolution{}, fmt.Errorf("decode frozen model resolution: %w", err)
		}
		if err := tx.Commit(ctx); err != nil {
			return agent.ModelResolution{}, fmt.Errorf("commit existing model resolution: %w", err)
		}
		return existing, nil
	}
	if resolution.DiscoveredAt.IsZero() {
		resolution.DiscoveredAt = time.Now().UTC()
	}
	encoded, err := json.Marshal(resolution)
	if err != nil {
		return agent.ModelResolution{}, fmt.Errorf("encode model resolution: %w", err)
	}
	if _, err := tx.Exec(ctx, `
		UPDATE agent_platform.agent_runs
		SET binding_snapshot=jsonb_set(binding_snapshot, '{resolved_model}', $4::jsonb, true), updated_at=now()
		WHERE id=$1::uuid AND lease_owner=$2::text AND lease_token=$3 AND status='running'`,
		lease.RunID, lease.Owner, lease.Token, encoded); err != nil {
		return agent.ModelResolution{}, fmt.Errorf("freeze model resolution: %w", err)
	}
	eventPayload := mustJSON(map[string]any{
		"selection_policy":      resolution.SelectionPolicy,
		"provider":              resolution.Provider,
		"service_ref":           resolution.ServiceRef,
		"model_id":              resolution.ModelID,
		"model_version":         resolution.ModelVersion,
		"artifact_digest":       resolution.ArtifactDigest,
		"context_window_tokens": resolution.ContextWindowTokens,
		"discovered_at":         resolution.DiscoveredAt,
	})
	if _, err := s.appendEventTx(ctx, tx, tenantID, event.Input{
		RunID: lease.RunID, Type: event.ModelResolved, Payload: eventPayload,
	}); err != nil {
		return agent.ModelResolution{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return agent.ModelResolution{}, fmt.Errorf("commit model resolution: %w", err)
	}
	return resolution, nil
}

// ClaimNext obtains the oldest runnable Run and increments its fencing token.
func (s *RunStore) ClaimNext(ctx context.Context, workerID string, leaseDuration time.Duration) (agent.Run, error) {
	if strings.TrimSpace(workerID) == "" || leaseDuration <= 0 {
		return agent.Run{}, errors.New("worker id and positive lease duration are required")
	}
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return agent.Run{}, fmt.Errorf("begin claim run: %w", err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()

	row := tx.QueryRow(ctx, `
		WITH candidate AS (
			SELECT id, status AS previous_status
			FROM agent_platform.agent_runs
			WHERE status = 'queued'
			   OR (status IN ('running', 'waiting_tool') AND lease_expires_at < now())
			   OR (status = 'waiting_external' AND next_wakeup_at <= now()
			       AND (lease_expires_at IS NULL OR lease_expires_at < now()))
			ORDER BY created_at
			FOR UPDATE SKIP LOCKED
			LIMIT 1
		)
		UPDATE agent_platform.agent_runs AS run
		SET status='running', lease_owner=$1, lease_token=run.lease_token+1,
			lease_expires_at=now()+($2::bigint * interval '1 millisecond'),
			attempt=run.attempt+1, started_at=COALESCE(run.started_at, now()), updated_at=now()
		FROM candidate
		WHERE run.id=candidate.id
		RETURNING `+qualifiedRunColumns("run")+`, candidate.previous_status`,
		workerID, leaseDuration.Milliseconds(),
	)
	run, previous, err := scanClaimedRun(row)
	if errors.Is(err, pgx.ErrNoRows) {
		return agent.Run{}, agent.ErrNoRunnableRun
	}
	if err != nil {
		return agent.Run{}, fmt.Errorf("claim run: %w", err)
	}
	eventType := event.RunClaimed
	if previous != agent.RunQueued {
		eventType = event.RunResumed
	}
	if eventType == event.RunResumed {
		if _, err := s.appendEventTx(ctx, tx, run.TenantID, event.Input{
			RunID: run.ID, WorkflowID: run.WorkflowID, SessionID: sessionIDOf(run.SessionID), Type: event.WorkflowResumed,
			Payload: mustJSON(map[string]any{"workflow_id": run.WorkflowID, "worker_id": workerID, "lease_token": run.LeaseToken, "previous_status": previous}),
		}); err != nil {
			return agent.Run{}, err
		}
	}
	if _, err := s.appendEventTx(ctx, tx, run.TenantID, event.Input{
		RunID: run.ID, WorkflowID: run.WorkflowID, SessionID: sessionIDOf(run.SessionID), Type: eventType,
		Payload: mustJSON(map[string]any{"worker_id": workerID, "lease_token": run.LeaseToken, "previous_status": previous}),
	}); err != nil {
		return agent.Run{}, err
	}
	if run.ParentRunID == nil {
		if err := projectWorkflowPhaseTx(ctx, tx, run.TenantID, run.WorkflowID, run.ID, "active", workflow.StatusRunning); err != nil {
			return agent.Run{}, err
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return agent.Run{}, fmt.Errorf("commit claim run: %w", err)
	}
	return run, nil
}

// ContinueRunAfterDeadline atomically fences a timed-out root attempt and
// creates a queued successor in the same Workflow. Checkpoint, Plan, workspace,
// and Artifact ownership are Workflow-scoped, so the successor can resume
// without replaying committed Tool calls. The bounded retry count prevents a
// permanently unavailable model from creating an unbounded Run chain.
func (s *RunStore) ContinueRunAfterDeadline(ctx context.Context, lease agent.Lease, failure event.Input) (agent.Run, error) {
	const maxAutomaticContinuations = 3
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return agent.Run{}, fmt.Errorf("begin deadline continuation: %w", err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	row := tx.QueryRow(ctx, `SELECT `+qualifiedRunColumns("run")+` FROM agent_platform.agent_runs run
		WHERE run.id=$1::uuid AND run.lease_owner=$2::text AND run.lease_token=$3 AND run.status='running'
		FOR UPDATE`, lease.RunID, lease.Owner, lease.Token)
	current, err := scanRun(row)
	if errors.Is(err, pgx.ErrNoRows) {
		return agent.Run{}, agent.ErrLeaseLost
	}
	if err != nil {
		return agent.Run{}, fmt.Errorf("lock timed-out run: %w", err)
	}
	if current.ParentRunID != nil {
		return agent.Run{}, errors.New("delegated child Run deadlines are resolved by the parent delegation contract")
	}
	var prior int
	if err := tx.QueryRow(ctx, `SELECT count(*) FROM agent_platform.agent_runs WHERE workflow_id=$1::uuid AND trigger_type='automatic_retry'`, current.WorkflowID).Scan(&prior); err != nil {
		return agent.Run{}, fmt.Errorf("count automatic continuations: %w", err)
	}
	if prior >= maxAutomaticContinuations {
		return agent.Run{}, fmt.Errorf("automatic continuation limit %d reached", maxAutomaticContinuations)
	}
	payload := failure.Payload
	if len(payload) == 0 {
		payload = processFailureJSON("run attempt deadline exceeded")
	}
	closed, err := tx.Exec(ctx, `UPDATE agent_platform.agent_runs
		SET status='failed',finished_at=now(),updated_at=now(),
			error_code=COALESCE(($5::jsonb)->>'error_code','RUN_ATTEMPT_DEADLINE_EXCEEDED'),
			error_message=COALESCE(($5::jsonb)->>'error','run attempt deadline exceeded'),
			lease_owner=NULL,lease_expires_at=NULL
		WHERE id=$1::uuid AND lease_owner=$2::text AND lease_token=$3 AND status=$4`,
		lease.RunID, lease.Owner, lease.Token, agent.RunRunning, payload)
	if err != nil {
		return agent.Run{}, fmt.Errorf("close timed-out run: %w", err)
	}
	if closed.RowsAffected() != 1 {
		return agent.Run{}, agent.ErrLeaseLost
	}
	if _, err := s.appendEventTx(ctx, tx, current.TenantID, event.Input{RunID: current.ID, WorkflowID: current.WorkflowID, SessionID: sessionIDOf(current.SessionID), Type: event.RunFailed, Turn: failure.Turn, Step: failure.Step, Payload: payload}); err != nil {
		return agent.Run{}, err
	}
	nextRow := tx.QueryRow(ctx, `INSERT INTO agent_platform.agent_runs
		(tenant_id,session_id,workflow_id,agent_version_id,status,trigger_type,input,binding_snapshot,created_by,traceparent)
		SELECT tenant_id,session_id,workflow_id,agent_version_id,'queued','automatic_retry',input,binding_snapshot,created_by,traceparent
		FROM agent_platform.agent_runs WHERE id=$1::uuid RETURNING `+runColumns, current.ID)
	next, err := scanRun(nextRow)
	if err != nil {
		return agent.Run{}, fmt.Errorf("create automatic continuation: %w", err)
	}
	activated, err := tx.Exec(ctx, `UPDATE agent_platform.agent_workflows
		SET active_run_id=$2::uuid,status='active',phase='ready',
			execution_generation=execution_generation+1,updated_at=now()
		WHERE id=$1::uuid AND tenant_id=$3::text`, current.WorkflowID, next.ID, current.TenantID)
	if err != nil {
		return agent.Run{}, fmt.Errorf("activate automatic continuation: %w", err)
	}
	if activated.RowsAffected() != 1 {
		return agent.Run{}, errors.New("deadline continuation workflow disappeared")
	}
	recoveryPayload := mustJSON(map[string]any{"trigger_type": "automatic_retry", "recovery_from_run_id": current.ID, "workflow_id": current.WorkflowID, "automatic_retry": prior + 1})
	for _, input := range []event.Input{
		{RunID: next.ID, WorkflowID: next.WorkflowID, SessionID: sessionIDOf(next.SessionID), Type: event.RunCreated, Payload: recoveryPayload},
		{RunID: next.ID, WorkflowID: next.WorkflowID, SessionID: sessionIDOf(next.SessionID), Type: event.RunAttemptCreated, Payload: recoveryPayload},
		{RunID: next.ID, WorkflowID: next.WorkflowID, SessionID: sessionIDOf(next.SessionID), Type: event.WorkflowResumed, Payload: recoveryPayload},
		{RunID: next.ID, WorkflowID: next.WorkflowID, SessionID: sessionIDOf(next.SessionID), Type: event.WorkflowRoutingDecided, Payload: mustJSON(map[string]any{"workflow_id": next.WorkflowID, "routing_intent": "automatic_retry", "new_workflow": false, "recovery_from_run_id": current.ID})},
	} {
		if _, err := s.appendEventTx(ctx, tx, current.TenantID, input); err != nil {
			return agent.Run{}, err
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return agent.Run{}, fmt.Errorf("commit deadline continuation: %w", err)
	}
	return next, nil
}

func processFailureJSON(message string) json.RawMessage {
	return mustJSON(map[string]any{"error": message, "error_code": "RUN_ATTEMPT_DEADLINE_EXCEEDED", "retryable": true})
}

// RenewLease extends ownership only when owner and fencing token still match.
func (s *RunStore) RenewLease(ctx context.Context, lease agent.Lease, duration time.Duration) (agent.Lease, error) {
	if duration <= 0 {
		return agent.Lease{}, errors.New("lease duration must be positive")
	}
	var expiry time.Time
	tag, err := s.pool.Exec(ctx, `
		UPDATE agent_platform.agent_runs
		SET lease_expires_at=now()+($4::bigint * interval '1 millisecond'), updated_at=now()
		WHERE id=$1::uuid AND lease_owner=$2 AND lease_token=$3 AND status='running'`,
		lease.RunID, lease.Owner, lease.Token, duration.Milliseconds())
	if err != nil {
		return agent.Lease{}, fmt.Errorf("renew run lease: %w", err)
	}
	if tag.RowsAffected() != 1 {
		return agent.Lease{}, agent.ErrLeaseLost
	}
	if err := s.pool.QueryRow(ctx,
		`SELECT lease_expires_at FROM agent_platform.agent_runs WHERE id=$1::uuid`, lease.RunID,
	).Scan(&expiry); err != nil {
		return agent.Lease{}, fmt.Errorf("read renewed lease: %w", err)
	}
	lease.Expiry = expiry
	return lease, nil
}

// Transition atomically applies a fenced status change and appends its event.
func (s *RunStore) Transition(
	ctx context.Context,
	lease agent.Lease,
	expected, next agent.RunStatus,
	transitionEvent event.Input,
) (agent.Run, error) {
	if err := agent.ValidateTransition(expected, next); err != nil {
		return agent.Run{}, err
	}
	payload := transitionEvent.Payload
	if len(payload) == 0 {
		payload = json.RawMessage(`{}`)
	}
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return agent.Run{}, fmt.Errorf("begin run transition: %w", err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()

	row := tx.QueryRow(ctx, `
		UPDATE agent_platform.agent_runs
		SET status=$5::text, updated_at=now(),
			finished_at=CASE WHEN $5::text IN ('completed','failed','cancelled') THEN now() ELSE finished_at END,
			output=CASE WHEN $5::text='completed' THEN COALESCE(($6::jsonb)->'output', output) ELSE output END,
			error_message=CASE WHEN $5::text='failed' THEN COALESCE(($6::jsonb)->>'error', error_message) ELSE error_message END,
			lease_owner=CASE WHEN $5::text IN ('completed','failed','cancelled','suspended','waiting_approval','waiting_input','waiting_external') THEN NULL ELSE lease_owner END,
			lease_expires_at=CASE WHEN $5::text IN ('completed','failed','cancelled','suspended','waiting_approval','waiting_input','waiting_external') THEN NULL ELSE lease_expires_at END
		WHERE id=$1::uuid AND lease_owner=$2 AND lease_token=$3 AND status=$4
		RETURNING `+runColumns,
		lease.RunID, lease.Owner, lease.Token, expected, next, payload,
	)
	run, err := scanRun(row)
	if errors.Is(err, pgx.ErrNoRows) {
		return agent.Run{}, agent.ErrLeaseLost
	}
	if err != nil {
		return agent.Run{}, fmt.Errorf("transition run: %w", err)
	}
	transitionEvent.RunID = run.ID
	transitionEvent.WorkflowID = run.WorkflowID
	transitionEvent.SessionID = sessionIDOf(run.SessionID)
	// A completed Run is only one attempt. A direct status/diagnostic answer
	// must not close its Workflow while the Workflow-owned durable Plan still
	// has pending, active, blocked, or mandatory-unverified work. Child Agent
	// Runs are also internal attempts and never own Workflow completion.
	workflowCompleted := next == agent.RunCompleted && run.ParentRunID == nil
	if workflowCompleted {
		hasOpenWork, planErr := workflowPlanHasOpenWorkTx(ctx, tx, run.TenantID, run.WorkflowID)
		if planErr != nil {
			return agent.Run{}, planErr
		}
		workflowCompleted = !hasOpenWork
	}
	if workflowCompleted {
		if _, err := s.appendEventTx(ctx, tx, run.TenantID, event.Input{
			RunID: run.ID, WorkflowID: run.WorkflowID, SessionID: sessionIDOf(run.SessionID), Type: event.WorkflowCompleted,
			Payload: mustJSON(map[string]any{"workflow_id": run.WorkflowID, "run_id": run.ID}),
		}); err != nil {
			return agent.Run{}, err
		}
	}
	if next == agent.RunWaitingApproval || next == agent.RunWaitingInput || next == agent.RunWaitingExternal || next == agent.RunSuspended {
		if _, err := s.appendEventTx(ctx, tx, run.TenantID, event.Input{
			RunID: run.ID, WorkflowID: run.WorkflowID, SessionID: sessionIDOf(run.SessionID), Type: event.WorkflowSuspended,
			Payload: mustJSON(map[string]any{"workflow_id": run.WorkflowID, "run_id": run.ID, "status": next}),
		}); err != nil {
			return agent.Run{}, err
		}
	}
	if _, err := s.appendEventTx(ctx, tx, run.TenantID, transitionEvent); err != nil {
		return agent.Run{}, err
	}
	// Keep the long-lived Workflow projection in sync with the current Run
	// state. This is a projection only; Run status remains the authoritative
	// attempt-level state and can be replayed from events.
	workflowStatus := workflowStatusForRun(next)
	workflowPhase := workflowPhaseForRun(next)
	if next == agent.RunCompleted && !workflowCompleted {
		workflowStatus = "active"
		workflowPhase = workflow.StatusReady
	}
	// A delegated child never owns the Workflow cursor. Projecting its running,
	// waiting or terminal state would race with an async parent and can even
	// resurrect an already-completed Workflow. Synchronous completion below
	// explicitly reactivates the parent only when it was actually waiting.
	if run.ParentRunID == nil {
		if err := projectWorkflowPhaseTx(ctx, tx, run.TenantID, run.WorkflowID, run.ID, workflowStatus, workflowPhase); err != nil {
			return agent.Run{}, err
		}
	}
	if next == agent.RunCompleted && run.SessionID != nil && len(run.Output) != 0 {
		createdAt := time.Now().UTC()
		if run.FinishedAt != nil {
			createdAt = *run.FinishedAt
		}
		if err := appendSessionMessageTx(ctx, tx, run.TenantID, *run.SessionID, run.ID, string(model.RoleAssistant), "run_output", run.Output, createdAt); err != nil {
			return agent.Run{}, err
		}
	}
	if run.ParentRunID != nil && run.DelegationID != nil && next.Terminal() {
		delegationStatus := string(next)
		if next == agent.RunCancelled {
			delegationStatus = "cancelled"
		}
		var parentID, callID string
		var turn, step int
		errText := ""
		if run.ErrorMessage != nil {
			errText = *run.ErrorMessage
		}
		if err := tx.QueryRow(ctx, `UPDATE agent_platform.agent_delegations SET status=$2::text,output=$3::jsonb,error=NULLIF($4::text,''),updated_at=now() WHERE id=$1::uuid RETURNING parent_run_id::text,call_id`, *run.DelegationID, delegationStatus, run.Output, errText).Scan(&parentID, &callID); err != nil {
			return agent.Run{}, fmt.Errorf("finish delegation: %w", err)
		}
		_ = tx.QueryRow(ctx, `SELECT turn_no,step_no FROM agent_platform.agent_delegations WHERE id=$1::uuid`, *run.DelegationID).Scan(&turn, &step)
		requeued, err := tx.Exec(ctx, `UPDATE agent_platform.agent_runs SET status='queued',next_wakeup_at=now(),updated_at=now() WHERE id=$1::uuid AND tenant_id=$2::text AND status='waiting_external'`, parentID, run.TenantID)
		if err != nil {
			return agent.Run{}, err
		}
		var parentWorkflowID string
		if err := tx.QueryRow(ctx, `SELECT workflow_id::text FROM agent_platform.agent_runs WHERE id=$1::uuid AND tenant_id=$2::text`, parentID, run.TenantID).Scan(&parentWorkflowID); err != nil {
			return agent.Run{}, fmt.Errorf("resolve parent workflow: %w", err)
		}
		// Only a synchronous parent waiting on this child is resumed here. An
		// async parent may already be terminal; a late child completion must not
		// resurrect that Workflow or replace its terminal projection.
		if requeued.RowsAffected() == 1 {
			if err := projectWorkflowTx(ctx, tx, run.TenantID, parentWorkflowID, parentID, "active"); err != nil {
				return agent.Run{}, err
			}
		}
		eventType := event.DelegationCompleted
		if next != agent.RunCompleted {
			eventType = event.DelegationFailed
		}
		if _, err := s.appendEventTx(ctx, tx, run.TenantID, event.Input{RunID: parentID, WorkflowID: parentWorkflowID, Type: eventType, Turn: turn, DecisionCycle: turn, Step: step, ActionID: callID, CallID: callID, Payload: mustJSON(map[string]any{"delegation_id": *run.DelegationID, "child_run_id": run.ID, "status": delegationStatus, "error": errText, "workflow_id": parentWorkflowID, "action_id": callID, "action_kind": "agent"})}); err != nil {
			return agent.Run{}, err
		}
	}
	// Rebuild the canonical delivery projection in the same transaction as the
	// Run transition. A terminal Run therefore cannot be exposed without a
	// manifest that reflects its final status and latest artifact versions.
	if err := s.rebuildRunManifestTx(ctx, tx, run.TenantID, run.ID); err != nil {
		return agent.Run{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return agent.Run{}, fmt.Errorf("commit run transition: %w", err)
	}
	return run, nil
}

// workflowPlanHasOpenWorkTx uses the same taskplan completion semantics as
// the runtime guard. The row lock prevents a concurrent Plan update from
// racing a Run completion into an incorrect WORKFLOW_COMPLETED projection.
func workflowPlanHasOpenWorkTx(ctx context.Context, tx pgx.Tx, tenantID, workflowID string) (bool, error) {
	var rawSteps []byte
	var planRevision int
	err := tx.QueryRow(ctx, `
		SELECT steps,revision FROM agent_platform.agent_task_plans
		WHERE workflow_id=$1::uuid AND tenant_id=$2::text
		FOR UPDATE`, workflowID, tenantID).Scan(&rawSteps, &planRevision)
	if errors.Is(err, pgx.ErrNoRows) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("load workflow plan before completion: %w", err)
	}
	var steps []taskplan.Step
	if err := json.Unmarshal(rawSteps, &steps); err != nil {
		return false, fmt.Errorf("decode workflow plan before completion: %w", err)
	}
	if (taskplan.Plan{Steps: steps}).HasOpenWork() {
		return true, nil
	}
	// Node and verification projections are updated independently from the Plan
	// JSON so a workspace mutation can invalidate previously completed work
	// without rewriting model-authored history. Completion must therefore consult
	// both projections, otherwise a stale verification could still close the
	// Workflow based on the older JSON snapshot.
	var projectedOpen bool
	err = tx.QueryRow(ctx, `
		SELECT
			EXISTS (
				SELECT 1 FROM agent_platform.agent_plan_node_states AS node
				WHERE node.workflow_id=$1::uuid
				  AND node.status NOT IN ('completed','skipped')
			)
			OR EXISTS (
				SELECT 1 FROM agent_platform.agent_verification_intents AS intent
				JOIN agent_platform.agent_task_plans AS plan
				  ON plan.workflow_id=$1::uuid AND plan.revision=$3
				 AND plan.last_modified_run_id=intent.run_id
				WHERE intent.plan_revision=$3
				  AND intent.tenant_id=$2::text
				  AND intent.enforcement IN ('required','release_gate')
				  AND intent.status <> 'passed'
			)`, workflowID, tenantID, planRevision).Scan(&projectedOpen)
	if err != nil {
		return false, fmt.Errorf("load workflow plan projections before completion: %w", err)
	}
	return projectedOpen, nil
}

// RequestCancel records a durable cancellation request without taking Worker ownership.
func (s *RunStore) RequestCancel(ctx context.Context, runID, requestedBy string) error {
	return s.requestCancel(ctx, "", runID, requestedBy)
}

// RequestCancelForTenant records cancellation within a tenant boundary.
func (s *RunStore) RequestCancelForTenant(ctx context.Context, tenantID, runID, requestedBy string) error {
	return s.requestCancel(ctx, tenantID, runID, requestedBy)
}

func (s *RunStore) requestCancel(ctx context.Context, tenantID, runID, requestedBy string) error {
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return fmt.Errorf("begin cancel request: %w", err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	var storedTenantID, storedWorkflowID string
	var status agent.RunStatus
	if err := tx.QueryRow(ctx, `
		SELECT tenant_id, workflow_id::text, status FROM agent_platform.agent_runs
		WHERE id=$1::uuid AND ($2='' OR tenant_id=$2) FOR UPDATE`, runID, tenantID,
	).Scan(&storedTenantID, &storedWorkflowID, &status); errors.Is(err, pgx.ErrNoRows) {
		return agent.ErrRunNotFound
	} else if err != nil {
		return fmt.Errorf("lock run for cancellation: %w", err)
	}
	if status.Terminal() {
		return fmt.Errorf("%w: %s", agent.ErrRunTerminal, status)
	}
	if _, err := tx.Exec(ctx, `
		UPDATE agent_platform.agent_runs
		SET cancel_requested_at=COALESCE(cancel_requested_at, now()),
			status=CASE WHEN status='running' THEN status ELSE 'cancelled' END,
			finished_at=CASE WHEN status='running' THEN finished_at ELSE now() END, updated_at=now()
		WHERE id=$1::uuid`, runID); err != nil {
		return fmt.Errorf("request run cancellation: %w", err)
	}
	if _, err := s.appendEventTx(ctx, tx, storedTenantID, event.Input{
		RunID: runID, Type: event.RunCancelRequested,
		Payload: mustJSON(map[string]string{"requested_by": requestedBy}),
	}); err != nil {
		return err
	}
	if status != agent.RunRunning {
		if err := projectWorkflowTx(ctx, tx, storedTenantID, storedWorkflowID, runID, "cancelled"); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `UPDATE agent_platform.agent_user_questions SET status='cancelled' WHERE run_id=$1::uuid AND status='pending'`, runID); err != nil {
			return err
		}
		if _, err := s.appendEventTx(ctx, tx, storedTenantID, event.Input{RunID: runID, Type: event.RunCancelled, Payload: mustJSON(map[string]string{"reason": "cancelled before execution or while waiting"})}); err != nil {
			return err
		}
	}
	rows, err := tx.Query(ctx, `WITH RECURSIVE descendants AS (SELECT id,status FROM agent_platform.agent_runs WHERE parent_run_id=$1::uuid UNION ALL SELECT child.id,child.status FROM agent_platform.agent_runs child JOIN descendants parent ON child.parent_run_id=parent.id) SELECT id::text,status FROM descendants WHERE status NOT IN ('completed','failed','cancelled') FOR UPDATE`, runID)
	if err != nil {
		return err
	}
	type descendant struct {
		id     string
		status agent.RunStatus
	}
	var descendants []descendant
	for rows.Next() {
		var item descendant
		if err := rows.Scan(&item.id, &item.status); err != nil {
			rows.Close()
			return err
		}
		descendants = append(descendants, item)
	}
	rows.Close()
	for _, child := range descendants {
		if _, err := tx.Exec(ctx, `UPDATE agent_platform.agent_runs SET cancel_requested_at=COALESCE(cancel_requested_at,now()),status=CASE WHEN status='running' THEN status ELSE 'cancelled' END,finished_at=CASE WHEN status='running' THEN finished_at ELSE now() END,updated_at=now() WHERE id=$1::uuid`, child.id); err != nil {
			return err
		}
		if child.status != agent.RunRunning {
			// Descendants share the parent's Workflow but never own its cursor or
			// lifecycle. The root cancellation above is the single projection write;
			// projecting each child would leave active_run_id pointing at the last
			// cancelled child.
			if _, err := s.appendEventTx(ctx, tx, storedTenantID, event.Input{RunID: child.id, Type: event.RunCancelled, Payload: mustJSON(map[string]string{"reason": "parent run cancelled"})}); err != nil {
				return err
			}
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit cancel request: %w", err)
	}
	return nil
}

func (s *RunStore) appendEventTx(ctx context.Context, tx pgx.Tx, tenantID string, input event.Input) (event.Event, error) {
	// Older call sites only supplied RunID. Resolve the durable Workflow and
	// Session boundary here so every committed event carries the same
	// structured identity envelope, including legacy tool/approval paths.
	if strings.TrimSpace(input.WorkflowID) == "" || strings.TrimSpace(input.SessionID) == "" {
		var workflowID, sessionID string
		if err := tx.QueryRow(ctx, `
			SELECT workflow_id::text, COALESCE(session_id::text,'')
			FROM agent_platform.agent_runs
			WHERE id=$1::uuid AND tenant_id=$2::text`, input.RunID, tenantID).Scan(&workflowID, &sessionID); err != nil {
			return event.Event{}, fmt.Errorf("resolve event workflow/session: %w", err)
		}
		if strings.TrimSpace(input.WorkflowID) == "" {
			input.WorkflowID = workflowID
		}
		if strings.TrimSpace(input.SessionID) == "" {
			input.SessionID = sessionID
		}
	}
	// Persist the canonical workflow vocabulary inside the payload while the
	// legacy turn_no/step_no columns remain available to old projections.
	input = event.NormalizeSemantic(input)
	if input.SchemaVersion == 0 {
		input.SchemaVersion = 1
	}
	payload := input.Payload
	if len(payload) == 0 {
		payload = json.RawMessage(`{}`)
	}
	// Updating the Workflow cursor takes a row lock and serializes all event
	// appenders for the same long-lived task. This makes workflow_seq monotonic
	// across continuation Runs and also removes the MAX(seq)+1 race for events
	// concurrently appended to one Run.
	var workflowSequence int64
	if err := tx.QueryRow(ctx, `
		UPDATE agent_platform.agent_workflows
		SET latest_workflow_seq=latest_workflow_seq+1,updated_at=now()
		WHERE id=$1::uuid AND tenant_id=$2::text
		RETURNING latest_workflow_seq`, input.WorkflowID, tenantID).Scan(&workflowSequence); errors.Is(err, pgx.ErrNoRows) {
		return event.Event{}, errors.New("event workflow does not exist in tenant")
	} else if err != nil {
		return event.Event{}, fmt.Errorf("allocate workflow event sequence: %w", err)
	}
	var committed event.Event
	err := tx.QueryRow(ctx, `
		INSERT INTO agent_platform.agent_events (
			tenant_id, run_id, workflow_id, workflow_seq, seq, event_type, schema_version,
			turn_no, step_no, call_id, payload
		) VALUES (
			$1, $2::uuid, $3::uuid, $4,
			(SELECT COALESCE(MAX(seq), 0) + 1 FROM agent_platform.agent_events WHERE run_id=$2::uuid),
			$5, $6, NULLIF($7, 0), NULLIF($8, 0), NULLIF($9, ''), $10::jsonb
		)
		RETURNING seq, created_at`,
		tenantID, input.RunID, input.WorkflowID, workflowSequence, input.Type, input.SchemaVersion,
		input.Turn, input.Step, input.CallID, payload,
	).Scan(&committed.Sequence, &committed.CreatedAt)
	if err != nil {
		return event.Event{}, fmt.Errorf("append %s event: %w", input.Type, err)
	}
	committed.Input = input
	committed.Payload = payload
	committed.WorkflowSequence = workflowSequence
	if err := projectExecutionEvent(ctx, tx, input, payload); err != nil {
		return event.Event{}, err
	}
	if err := projectWorkflowExecutionTx(ctx, tx, tenantID, committed); err != nil {
		return event.Event{}, err
	}
	if err := projectWorkflowPhaseEventTx(ctx, tx, tenantID, committed); err != nil {
		return event.Event{}, err
	}
	if err := s.projectObservationTx(ctx, tx, tenantID, committed); err != nil {
		return event.Event{}, err
	}
	if !s.outboxEnabled {
		return committed, nil
	}
	// The Event Ledger already owns the complete payload. The outbox carries
	// only the cursor needed to wake readers, avoiding a second copy of prompts,
	// model output, tool results, or other potentially sensitive data.
	outboxPayload, err := json.Marshal(map[string]int64{"sequence": committed.Sequence})
	if err != nil {
		return event.Event{}, fmt.Errorf("encode %s outbox event: %w", input.Type, err)
	}
	if _, err := tx.Exec(ctx, `
		INSERT INTO agent_platform.agent_outbox
			(aggregate_type,aggregate_id,event_type,payload)
		VALUES ('agent_run',$1::uuid,$2::text,$3::jsonb)`, input.RunID, input.Type, outboxPayload); err != nil {
		return event.Event{}, fmt.Errorf("enqueue %s outbox event: %w", input.Type, err)
	}
	return committed, nil
}

func projectExecutionEvent(ctx context.Context, tx pgx.Tx, input event.Input, payload json.RawMessage) error {
	var err error
	switch input.Type {
	case event.TurnStarted:
		_, err = tx.Exec(ctx, `INSERT INTO agent_platform.agent_turns (run_id, turn_no, status)
			VALUES ($1::uuid,$2,'running') ON CONFLICT (run_id,turn_no) DO UPDATE SET status='running'`, input.RunID, input.Turn)
	case event.TurnCompleted:
		_, err = tx.Exec(ctx, `UPDATE agent_platform.agent_turns SET status='completed', finished_at=now()
			WHERE run_id=$1::uuid AND turn_no=$2`, input.RunID, input.Turn)
	case event.StepStarted:
		_, err = tx.Exec(ctx, `INSERT INTO agent_platform.agent_steps (run_id,turn_no,step_no,status)
			VALUES ($1::uuid,$2,$3,'running') ON CONFLICT (run_id,turn_no,step_no)
			DO UPDATE SET status='running', started_at=now(), finished_at=NULL, error=NULL`, input.RunID, input.Turn, input.Step)
		if err == nil {
			_, err = tx.Exec(ctx, `UPDATE agent_platform.agent_runs SET current_turn=$2,current_step=$3 WHERE id=$1::uuid`, input.RunID, input.Turn, input.Step)
		}
	case event.StepCompleted:
		_, err = tx.Exec(ctx, `UPDATE agent_platform.agent_steps SET status=CASE WHEN EXISTS(
			SELECT 1 FROM agent_platform.agent_events WHERE run_id=$1::uuid AND turn_no=$2 AND step_no=$3
			AND event_type IN ('MODEL_FAILED','TOOL_FAILED')) THEN 'failed' ELSE 'completed' END,finished_at=now()
			WHERE run_id=$1::uuid AND turn_no=$2 AND step_no=$3`, input.RunID, input.Turn, input.Step)
	case event.ModelCompleted:
		_, err = tx.Exec(ctx, `INSERT INTO agent_platform.agent_model_calls
			(run_id,turn_no,step_no,provider,model_id,model_version,service_ref,artifact_digest,selection_strategy,context_manifest,input_tokens,output_tokens,latency_ms,status)
			VALUES ($1::uuid,$2,$3,COALESCE(NULLIF($4::jsonb->>'provider',''),'unknown'),
			COALESCE(NULLIF($4::jsonb->>'model_id',''),'unknown'),NULLIF($4::jsonb->>'model_version',''),
			NULLIF($4::jsonb->>'service',''),NULLIF($4::jsonb->>'artifact_digest',''),NULLIF($4::jsonb->>'selection_policy',''),
			COALESCE($4::jsonb->'context_manifest','{}'::jsonb),
			COALESCE(($4::jsonb->'usage'->>'input_tokens')::bigint,0),
			COALESCE(($4::jsonb->'usage'->>'output_tokens')::bigint,0),
			COALESCE(($4::jsonb->>'latency_ms')::bigint,0),'completed')`, input.RunID, input.Turn, input.Step, payload)
	case event.ModelFailed:
		_, err = tx.Exec(ctx, `INSERT INTO agent_platform.agent_model_calls
			(run_id,turn_no,step_no,provider,model_id,model_version,service_ref,artifact_digest,selection_strategy,context_manifest,latency_ms,status,error)
			VALUES ($1::uuid,$2,$3,COALESCE(NULLIF($4::jsonb->>'provider',''),'unknown'),
			COALESCE(NULLIF($4::jsonb->>'model_id',''),'unknown'),NULLIF($4::jsonb->>'model_version',''),
			NULLIF($4::jsonb->>'service',''),NULLIF($4::jsonb->>'artifact_digest',''),NULLIF($4::jsonb->>'selection_policy',''),
			COALESCE($4::jsonb->'context_manifest','{}'::jsonb),
			COALESCE(($4::jsonb->>'latency_ms')::bigint,0),'failed',jsonb_build_object('message',$4::jsonb->>'error'))`,
			input.RunID, input.Turn, input.Step, payload)
	}
	if err != nil {
		return fmt.Errorf("project %s event: %w", input.Type, err)
	}
	return nil
}

const runColumns = `
	id::text, tenant_id, session_id::text, workflow_id::text, agent_version_id::text, status,
	trigger_type, traceparent, input, output, binding_snapshot, current_turn, current_step,
	attempt, lease_owner, lease_token, lease_expires_at, next_wakeup_at,
	cancel_requested_at, started_at, finished_at, error_code, error_message,
	created_by, created_at, updated_at, parent_run_id::text, root_run_id::text, delegation_id::text, delegation_depth`

func qualifiedRunColumns(alias string) string {
	parts := strings.Split(runColumns, ",")
	for index, part := range parts {
		trimmed := strings.TrimSpace(part)
		if trimmed == "" {
			continue
		}
		if strings.Contains(trimmed, "::") {
			pieces := strings.SplitN(trimmed, "::", 2)
			parts[index] = alias + "." + pieces[0] + "::" + pieces[1]
		} else {
			parts[index] = alias + "." + trimmed
		}
	}
	return strings.Join(parts, ",")
}

type rowScanner interface {
	Scan(dest ...any) error
}

func scanRun(row rowScanner) (agent.Run, error) {
	var run agent.Run
	err := row.Scan(
		&run.ID, &run.TenantID, &run.SessionID, &run.WorkflowID, &run.AgentVersionID, &run.Status,
		&run.TriggerType, &run.TraceParent, &run.Input, &run.Output, &run.BindingSnapshot,
		&run.CurrentTurn, &run.CurrentStep, &run.Attempt, &run.LeaseOwner,
		&run.LeaseToken, &run.LeaseExpiresAt, &run.NextWakeupAt,
		&run.CancelRequestedAt, &run.StartedAt, &run.FinishedAt,
		&run.ErrorCode, &run.ErrorMessage, &run.CreatedBy, &run.CreatedAt, &run.UpdatedAt,
		&run.ParentRunID, &run.RootRunID, &run.DelegationID, &run.DelegationDepth,
	)
	if err == nil {
		if strings.TrimSpace(run.WorkflowID) == "" {
			run.WorkflowID = run.ID
		}
		hydrateModelResolution(&run)
	}
	return run, err
}

func scanClaimedRun(row rowScanner) (agent.Run, agent.RunStatus, error) {
	var run agent.Run
	var previous agent.RunStatus
	err := row.Scan(
		&run.ID, &run.TenantID, &run.SessionID, &run.WorkflowID, &run.AgentVersionID, &run.Status,
		&run.TriggerType, &run.TraceParent, &run.Input, &run.Output, &run.BindingSnapshot,
		&run.CurrentTurn, &run.CurrentStep, &run.Attempt, &run.LeaseOwner,
		&run.LeaseToken, &run.LeaseExpiresAt, &run.NextWakeupAt,
		&run.CancelRequestedAt, &run.StartedAt, &run.FinishedAt,
		&run.ErrorCode, &run.ErrorMessage, &run.CreatedBy, &run.CreatedAt, &run.UpdatedAt,
		&run.ParentRunID, &run.RootRunID, &run.DelegationID, &run.DelegationDepth,
		&previous,
	)
	if err == nil {
		if strings.TrimSpace(run.WorkflowID) == "" {
			run.WorkflowID = run.ID
		}
		hydrateModelResolution(&run)
	}
	return run, previous, err
}

func hydrateModelResolution(run *agent.Run) {
	var snapshot struct {
		ResolvedModel *agent.ModelResolution `json:"resolved_model"`
	}
	if json.Unmarshal(run.BindingSnapshot, &snapshot) == nil {
		run.ModelResolution = snapshot.ResolvedModel
	}
}

func validJSONObject(raw json.RawMessage) bool {
	if !json.Valid(raw) {
		return false
	}
	var object map[string]any
	return json.Unmarshal(raw, &object) == nil
}

func mustJSON(value any) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err != nil {
		panic(fmt.Sprintf("marshal internal persistence payload: %v", err))
	}
	return encoded
}

func workflowStatusForRun(status agent.RunStatus) string {
	switch status {
	case agent.RunWaitingTool, agent.RunWaitingApproval, agent.RunWaitingInput, agent.RunWaitingExternal, agent.RunSuspended:
		return "waiting"
	case agent.RunCompleted:
		return "completed"
	case agent.RunFailed:
		return "failed"
	case agent.RunCancelled:
		return "cancelled"
	default:
		return "active"
	}
}

func workflowPhaseForRun(status agent.RunStatus) workflow.Status {
	switch status {
	case agent.RunQueued:
		return workflow.StatusReady
	case agent.RunRunning:
		return workflow.StatusRunning
	case agent.RunWaitingTool, agent.RunWaitingExternal:
		return workflow.StatusWaitingTool
	case agent.RunWaitingApproval:
		return workflow.StatusWaitingApproval
	case agent.RunWaitingInput:
		return workflow.StatusWaitingUser
	case agent.RunSuspended:
		return workflow.StatusPaused
	case agent.RunCompleted:
		return workflow.StatusSucceeded
	case agent.RunFailed:
		return workflow.StatusFailed
	case agent.RunCancelled:
		return workflow.StatusCancelled
	default:
		return workflow.StatusCreated
	}
}

// projectWorkflowTx keeps the task-level projection synchronized with direct
// queue operations (approval/user-answer/delegation) as well as fenced Run
// transitions. Without this, a requeued Run could execute correctly while the
// UI still showed the Workflow as waiting forever.
func projectWorkflowTx(ctx context.Context, tx pgx.Tx, tenantID, workflowID, runID, status string) error {
	phase := workflow.StatusReady
	switch status {
	case "waiting":
		phase = workflow.StatusPaused
	case "completed":
		phase = workflow.StatusSucceeded
	case "failed":
		phase = workflow.StatusFailed
	case "cancelled":
		phase = workflow.StatusCancelled
	}
	return projectWorkflowPhaseTx(ctx, tx, tenantID, workflowID, runID, status, phase)
}

func projectWorkflowPhaseTx(ctx context.Context, tx pgx.Tx, tenantID, workflowID, runID, status string, phase workflow.Status) error {
	if strings.TrimSpace(workflowID) == "" {
		return nil
	}
	if _, err := tx.Exec(ctx, `
		UPDATE agent_platform.agent_workflows
		SET active_run_id=$2::uuid,status=$3::text,phase=$4::text,updated_at=now()
		WHERE id=$1::uuid AND tenant_id=$5::text`, workflowID, runID, status, phase, tenantID); err != nil {
		return fmt.Errorf("project workflow status: %w", err)
	}
	return nil
}
