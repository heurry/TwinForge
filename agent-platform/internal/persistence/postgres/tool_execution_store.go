package postgres

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

// ExecuteToolIdempotent claims one durable Tool Call, returns an already stored
// result when present, and fences stale Workers from committing a result.
func (s *RunStore) ExecuteToolIdempotent(
	ctx context.Context,
	lease agent.Lease,
	tenantID, toolVersionID, provider string,
	call tool.Call,
	handler tool.Handler,
) (tool.Result, error) {
	if handler == nil {
		return tool.Result{}, errors.New("tool handler is required")
	}
	if call.RunID != lease.RunID || call.ID == "" {
		return tool.Result{}, errors.New("tool call does not match lease")
	}
	requestHash, request, err := hashToolRequest(toolVersionID, call)
	if err != nil {
		return tool.Result{}, err
	}
	idempotencyKey := call.RunID + ":" + call.ID
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return tool.Result{}, fmt.Errorf("begin tool execution claim: %w", err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	var leaseValid bool
	var workflowID string
	var workflowGeneration, workspaceRevision, nodeRevision int64
	var planRevision int
	planNodeID := firstNonEmpty(call.PlanNodeID, call.PlanStepID)
	if err := tx.QueryRow(ctx, `
		SELECT true,run.workflow_id::text,workflow.execution_generation,workflow.workspace_revision,
		       COALESCE(plan.revision,0),COALESCE(node.node_revision,0)
		FROM agent_platform.agent_runs AS run
		JOIN agent_platform.agent_workflows AS workflow ON workflow.id=run.workflow_id
		LEFT JOIN agent_platform.agent_task_plans AS plan ON plan.workflow_id=run.workflow_id
		LEFT JOIN agent_platform.agent_plan_node_states AS node
		  ON node.workflow_id=run.workflow_id AND node.node_id=NULLIF($5::text,'')
		WHERE run.id=$1::uuid AND run.tenant_id=$2::text AND run.lease_owner=$3::text
		  AND run.lease_token=$4 AND run.status='running'
		  AND (run.parent_run_id IS NOT NULL OR workflow.active_run_id=run.id)
		FOR UPDATE OF run,workflow`,
		lease.RunID, tenantID, lease.Owner, lease.Token, planNodeID).Scan(
		&leaseValid, &workflowID, &workflowGeneration, &workspaceRevision, &planRevision, &nodeRevision,
	); errors.Is(err, pgx.ErrNoRows) {
		return tool.Result{}, agent.ErrLeaseLost
	} else if err != nil {
		return tool.Result{}, fmt.Errorf("verify tool execution lease: %w", err)
	}
	if call.WorkflowGeneration != 0 && call.WorkflowGeneration != workflowGeneration {
		return tool.Result{}, staleToolFenceError(call.Name, "workflow generation changed before execution")
	}
	if call.PlanRevision != 0 && call.PlanRevision != planRevision {
		return tool.Result{}, staleToolFenceError(call.Name, "plan revision changed before execution")
	}
	if call.NodeRevision != 0 && call.NodeRevision != nodeRevision {
		return tool.Result{}, staleToolFenceError(call.Name, "plan node revision changed before execution")
	}
	if call.WorkspaceRevision != 0 && call.WorkspaceRevision != workspaceRevision {
		return tool.Result{}, staleToolFenceError(call.Name, "workspace revision changed before execution")
	}
	call.WorkflowGeneration = workflowGeneration
	call.WorkspaceRevision = workspaceRevision
	if call.PlanRevision == 0 && planNodeID != "" {
		call.PlanRevision = planRevision
	}
	if call.NodeRevision == 0 && planNodeID != "" {
		call.NodeRevision = nodeRevision
	}
	tag, err := tx.Exec(ctx, `
		INSERT INTO agent_platform.agent_tool_executions (
			tenant_id, run_id, call_id, tool_name, tool_version, provider,
			request_hash, idempotency_key, status, attempt, request,
			started_at, lease_owner, lease_token, plan_step_key,
			workflow_id, plan_node_id, decision_cycle, action_id, action_kind,
			workflow_generation,plan_revision,node_revision,expected_workspace_revision
		) VALUES ($1::text, $2::uuid, $3::text, $4::text, $5::text, $6::text,
			$7::text, $8::text, 'running', 1, $9::jsonb, now(), $10::text, $11,
			NULLIF($12::text,''), NULLIF($13::text,'')::uuid, NULLIF($14::text,''), NULLIF($15::int,0),
			NULLIF($16::text,''), $17::text,$18,$19,$20,$21)
		ON CONFLICT (tenant_id, idempotency_key) DO NOTHING`,
		tenantID, call.RunID, call.ID, call.Name, toolVersionID, provider,
		requestHash, idempotencyKey, request, lease.Owner, lease.Token,
		firstNonEmpty(call.PlanStepID, call.PlanNodeID), workflowID, planNodeID, call.DecisionCycle,
		firstNonEmpty(call.ActionID, call.ID), actionKind(provider), workflowGeneration,
		call.PlanRevision, call.NodeRevision, workspaceRevision)
	if err != nil {
		return tool.Result{}, fmt.Errorf("claim tool execution: %w", err)
	}
	shouldExecute := tag.RowsAffected() == 1
	if !shouldExecute {
		var status, storedHash, applicationStatus string
		var storedResult, storedError json.RawMessage
		var owner *string
		var token, storedGeneration, storedNodeRevision, storedWorkspaceRevision int64
		var storedPlanRevision int
		if err := tx.QueryRow(ctx, `
			SELECT status, request_hash, result, error, lease_owner, lease_token,
			       workflow_generation,plan_revision,node_revision,
			       COALESCE(finished_workspace_revision,expected_workspace_revision),
			       application_status
			FROM agent_platform.agent_tool_executions
			WHERE tenant_id=$1::text AND idempotency_key=$2::text FOR UPDATE`,
			tenantID, idempotencyKey).Scan(
			&status, &storedHash, &storedResult, &storedError, &owner, &token,
			&storedGeneration, &storedPlanRevision, &storedNodeRevision, &storedWorkspaceRevision,
			&applicationStatus,
		); err != nil {
			return tool.Result{}, fmt.Errorf("read existing tool execution: %w", err)
		}
		if storedHash != requestHash {
			return tool.Result{}, tool.ErrCallIDConflict
		}
		storedFenceCurrent := applicationStatus == "applied" &&
			(storedGeneration == 0 || storedGeneration == workflowGeneration) &&
			(storedPlanRevision == 0 || storedPlanRevision == planRevision) &&
			(storedNodeRevision == 0 || storedNodeRevision == nodeRevision) &&
			storedWorkspaceRevision == workspaceRevision
		if status != "running" && !storedFenceCurrent {
			return tool.Result{}, staleToolFenceError(call.Name, "stored tool receipt belongs to an older workflow, plan, node, or workspace generation")
		}
		switch status {
		case "succeeded":
			var result tool.Result
			if err := json.Unmarshal(storedResult, &result); err != nil {
				return tool.Result{}, fmt.Errorf("decode stored tool result: %w", err)
			}
			result.ApplyStoredModelProjection()
			if err := tx.Commit(ctx); err != nil {
				return tool.Result{}, fmt.Errorf("commit stored tool result read: %w", err)
			}
			return result, nil
		case "failed":
			var failure struct {
				Message       string          `json:"message"`
				ErrorCode     string          `json:"error_code"`
				Retryable     bool            `json:"retryable"`
				Correction    string          `json:"correction"`
				RetryTemplate json.RawMessage `json:"retry_template"`
			}
			_ = json.Unmarshal(storedError, &failure)
			if err := tx.Commit(ctx); err != nil {
				return tool.Result{}, fmt.Errorf("commit stored tool failure read: %w", err)
			}
			if failure.ErrorCode != "" {
				return tool.Result{}, tool.NewContractErrorWithRepair(failure.ErrorCode, "", "", "", "", failure.Message, failure.Correction, failure.RetryTemplate, failure.Retryable)
			}
			return tool.Result{}, fmt.Errorf("stored tool execution failed: %s", failure.Message)
		case "running":
			if owner != nil && *owner == lease.Owner && token == lease.Token {
				return tool.Result{}, tool.ErrExecutionInProgress
			}
			if _, err := tx.Exec(ctx, `
				UPDATE agent_platform.agent_tool_executions
				SET lease_owner=$3::text, lease_token=$4, attempt=attempt+1, started_at=now(),
					workflow_generation=$5,plan_revision=$6,node_revision=$7,
					expected_workspace_revision=$8,application_status='applied',
					workflow_id=COALESCE(workflow_id,$9::uuid),
					plan_node_id=COALESCE(plan_node_id,NULLIF($10::text,'')),
					plan_step_key=COALESCE(plan_step_key,NULLIF($10::text,''))
				WHERE tenant_id=$1::text AND idempotency_key=$2::text`,
				tenantID, idempotencyKey, lease.Owner, lease.Token, workflowGeneration,
				call.PlanRevision, call.NodeRevision, workspaceRevision, workflowID, planNodeID); err != nil {
				return tool.Result{}, fmt.Errorf("take over tool execution: %w", err)
			}
			shouldExecute = true
		default:
			return tool.Result{}, fmt.Errorf("unknown stored tool status %q", status)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return tool.Result{}, fmt.Errorf("commit tool execution claim: %w", err)
	}
	if !shouldExecute {
		return tool.Result{}, tool.ErrExecutionInProgress
	}
	result, executeErr := handler(ctx, call)
	if executeErr != nil {
		executeErr = tool.NormalizeExecutionError(call, executeErr)
	}
	persistedResult, finishErr := s.finishToolExecution(
		context.WithoutCancel(ctx), lease, tenantID, idempotencyKey, result, executeErr,
	)
	if finishErr != nil {
		if executeErr != nil {
			return tool.Result{}, errors.Join(executeErr, finishErr)
		}
		return tool.Result{}, finishErr
	}
	return persistedResult, executeErr
}

// OffloadToolResult persists a large result for generic Executors that do not
// pass through ExecuteToolIdempotent. The artifact row and payload are still
// tenant/run scoped and use the same content-addressed storage path.
func (s *RunStore) OffloadToolResult(ctx context.Context, run agent.Run, call tool.Call, content []byte) (tool.ArtifactRef, error) {
	if len(content) <= tool.InlineResultLimit {
		return tool.ArtifactRef{}, errors.New("tool result does not require offload")
	}
	if len(content) > 10<<20 {
		return tool.ArtifactRef{}, errors.New("tool result exceeds 10 MiB artifact limit")
	}
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return tool.ArtifactRef{}, fmt.Errorf("begin generic tool result offload: %w", err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	var exists bool
	if err := tx.QueryRow(ctx, `SELECT true FROM agent_platform.agent_runs WHERE id=$1::uuid AND tenant_id=$2::text`, run.ID, run.TenantID).Scan(&exists); err != nil {
		return tool.ArtifactRef{}, fmt.Errorf("verify generic tool result run: %w", err)
	}
	id, err := s.persistArtifactTx(ctx, tx, run.TenantID, run.ID, call.ID, artifactWrite{
		Kind: "tool_result", Name: "tool-result-" + call.ID, MediaType: "application/json",
		Content: content, Metadata: map[string]string{"source": "generic_tool_result_offload", "tool": call.Name},
	})
	if err != nil {
		return tool.ArtifactRef{}, fmt.Errorf("persist generic tool result artifact: %w", err)
	}
	if err := tx.Commit(ctx); err != nil {
		return tool.ArtifactRef{}, fmt.Errorf("commit generic tool result artifact: %w", err)
	}
	digest := sha256.Sum256(content)
	return tool.ArtifactRef{ID: id, URI: "/api/v1/artifacts/" + id + "/content", SHA256: hex.EncodeToString(digest[:]), SizeBytes: len(content)}, nil
}

func (s *RunStore) finishToolExecution(
	ctx context.Context,
	lease agent.Lease,
	tenantID, idempotencyKey string,
	result tool.Result,
	executeErr error,
) (tool.Result, error) {
	status := "succeeded"
	if executeErr == nil && result.IsError {
		// A provider may report a process/MCP failure in the result envelope
		// instead of returning a Go error. Normalize before persisting so replay
		// and observability see the same contract and failed executions are not
		// incorrectly counted as succeeded.
		result = tool.NormalizeResultFailure(tool.Call{}, result)
		status = "failed"
	}
	if len(result.Artifacts) > 8 {
		return tool.Result{}, errors.New("tool returned more than 8 artifacts")
	}
	for _, item := range result.Artifacts {
		if len(item.Content) > 10<<20 {
			return tool.Result{}, fmt.Errorf("artifact %q exceeds 10 MiB", item.Name)
		}
	}
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return tool.Result{}, fmt.Errorf("begin tool execution finish: %w", err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	var artifactURI *string
	var runID, workflowID, executionID, toolName, actionID string
	var executionGeneration, currentGeneration, expectedWorkspaceRevision, currentWorkspaceRevision, executionNodeRevision, currentNodeRevision int64
	var executionPlanRevision, currentPlanRevision int
	var runFenceCurrent bool
	if err := tx.QueryRow(ctx, `
		SELECT execution.run_id::text,execution.workflow_id::text,execution.id::text,
		       execution.tool_name,COALESCE(NULLIF(execution.action_id,''),execution.call_id),
		       run.lease_owner=$3::text AND run.lease_token=$4 AND run.status='running'
		         AND (run.parent_run_id IS NOT NULL OR workflow.active_run_id=run.id),
		       execution.workflow_generation,workflow.execution_generation,
		       execution.plan_revision,COALESCE(plan.revision,0),
		       execution.node_revision,COALESCE(node.node_revision,0),
		       execution.expected_workspace_revision,workflow.workspace_revision
		FROM agent_platform.agent_tool_executions AS execution
		JOIN agent_platform.agent_runs AS run ON run.id=execution.run_id
		JOIN agent_platform.agent_workflows AS workflow ON workflow.id=execution.workflow_id
		LEFT JOIN agent_platform.agent_task_plans AS plan ON plan.workflow_id=execution.workflow_id
		LEFT JOIN agent_platform.agent_plan_node_states AS node
		  ON node.workflow_id=execution.workflow_id AND node.node_id=execution.plan_node_id
		WHERE execution.tenant_id=$1::text AND execution.idempotency_key=$2::text
		  AND execution.lease_owner=$3::text AND execution.lease_token=$4
		FOR UPDATE OF execution`, tenantID, idempotencyKey, lease.Owner, lease.Token).Scan(
		&runID, &workflowID, &executionID, &toolName, &actionID,
		&runFenceCurrent, &executionGeneration, &currentGeneration,
		&executionPlanRevision, &currentPlanRevision, &executionNodeRevision, &currentNodeRevision,
		&expectedWorkspaceRevision, &currentWorkspaceRevision,
	); errors.Is(err, pgx.ErrNoRows) {
		return tool.Result{}, agent.ErrLeaseLost
	} else if err != nil {
		return tool.Result{}, fmt.Errorf("resolve tool artifact run: %w", err)
	}
	fenceCurrent := runFenceCurrent && executionGeneration == currentGeneration &&
		(executionPlanRevision == 0 || executionPlanRevision == currentPlanRevision) &&
		(executionNodeRevision == 0 || executionNodeRevision == currentNodeRevision) &&
		expectedWorkspaceRevision == currentWorkspaceRevision
	for index, item := range result.Artifacts {
		id, err := s.persistArtifactTx(ctx, tx, tenantID, runID, callIDFromKey(idempotencyKey), artifactWrite{
			Kind: item.Kind, Name: item.Name, MediaType: item.MediaType, Content: item.Content, Metadata: item.Metadata,
		})
		if err != nil {
			return tool.Result{}, fmt.Errorf("persist tool artifact: %w", err)
		}
		uri := "/api/v1/artifacts/" + id + "/content"
		if artifactURI == nil {
			artifactURI = &uri
		}
		if result.Meta == nil {
			result.Meta = make(map[string]string)
		}
		result.Meta[fmt.Sprintf("artifact_%d", index)] = id
		// Keep the hash of the persisted Artifact separate from the workspace
		// file hash returned by the provider. They may be equal in the common
		// workspace snapshot case, but they represent different integrity
		// domains and must not share an ambiguous `sha256` field.
		result.Meta[fmt.Sprintf("artifact_%d_sha256", index)] = sha256Hex(item.Content)
		result.Meta[fmt.Sprintf("artifact_%d_kind", index)] = item.Kind
	}
	// Keep the complete result in the execution record for audit and replay,
	// but persist a bounded Artifact and attach a model-only receipt whenever a
	// result (including a provider failure payload) would exceed the 24k model's
	// inline budget.
	if len(result.Content) > tool.InlineResultLimit {
		if len(result.Artifacts) >= 8 {
			return tool.Result{}, errors.New("tool result is too large and artifact limit is exhausted")
		}
		if result.Meta == nil {
			result.Meta = make(map[string]string)
		}
		id, err := s.persistArtifactTx(ctx, tx, tenantID, runID, callIDFromKey(idempotencyKey), artifactWrite{
			Kind: "tool_result", Name: "tool-result-" + callIDFromKey(idempotencyKey), MediaType: "application/json",
			Content: result.Content, Metadata: map[string]string{"source": "automatic_tool_result_offload"},
		})
		if err != nil {
			return tool.Result{}, fmt.Errorf("persist oversized tool result artifact: %w", err)
		}
		result.Meta["content_artifact_id"] = id
		result.Meta["content_sha256"] = sha256Hex(result.Content)
		result.Meta["content_bytes"] = fmt.Sprintf("%d", len(result.Content))
		result.ModelContent = tool.BuildResultReceipt(result.Content, id)
		if artifactURI == nil {
			uri := "/api/v1/artifacts/" + id + "/content"
			artifactURI = &uri
		}
	}
	result.Artifacts = nil
	resultJSON, err := json.Marshal(result)
	if err != nil {
		return tool.Result{}, fmt.Errorf("encode tool result: %w", err)
	}
	errorJSON := json.RawMessage(`null`)
	if executeErr != nil {
		status = "failed"
		failure := map[string]any{"message": executeErr.Error(), "error_code": "TOOL_EXECUTION_FAILED", "retryable": true}
		if typed, ok := tool.AsContractError(executeErr); ok {
			failure["error_code"] = typed.Code
			failure["retryable"] = typed.Retryable
			if typed.Correction != "" {
				failure["correction"] = typed.Correction
			}
			if len(typed.RetryTemplate) != 0 {
				failure["retry_template"] = json.RawMessage(typed.RetryTemplate)
			}
		}
		errorJSON, _ = json.Marshal(failure)
	} else if result.IsError {
		failure := map[string]any{
			"message":    result.Error,
			"error_code": result.Meta["error_code"],
			"retryable":  result.Meta["retryable"] == "true",
			"correction": result.Meta["correction"],
		}
		var payload map[string]any
		if json.Unmarshal(result.Content, &payload) == nil {
			for _, key := range []string{"error", "error_code", "retryable", "correction", "retry_template"} {
				if value, ok := payload[key]; ok {
					if key == "error" {
						failure["message"] = value
					} else {
						failure[key] = value
					}
				}
			}
		}
		errorJSON, _ = json.Marshal(failure)
	}
	applicationStatus := "applied"
	if !fenceCurrent {
		applicationStatus = "stale_ignored"
	}
	if fenceCurrent && status == "succeeded" && toolMutatesWorkspace(toolName) {
		if err := tx.QueryRow(ctx, `
			UPDATE agent_platform.agent_workflows
			SET workspace_revision=workspace_revision+1,updated_at=now()
			WHERE id=$1::uuid AND tenant_id=$2::text AND execution_generation=$3
			RETURNING workspace_revision`, workflowID, tenantID, executionGeneration).Scan(&currentWorkspaceRevision); err != nil {
			return tool.Result{}, fmt.Errorf("advance workspace revision: %w", err)
		}
		path, previousHash, currentHash := workspaceMutationIdentity(result)
		metadata := result.Content
		if len(metadata) == 0 || !json.Valid(metadata) {
			metadata = json.RawMessage(`{}`)
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO agent_platform.agent_workspace_mutations(
				tenant_id,workflow_id,run_id,tool_execution_id,action_id,workspace_revision,
				operation,path,previous_file_sha256,file_sha256,metadata)
			VALUES($1,$2::uuid,$3::uuid,$4::uuid,$5,$6,$7,NULLIF($8,''),NULLIF($9,''),NULLIF($10,''),$11::jsonb)`,
			tenantID, workflowID, runID, executionID, actionID, currentWorkspaceRevision,
			toolName, path, previousHash, currentHash, metadata); err != nil {
			return tool.Result{}, fmt.Errorf("record workspace mutation: %w", err)
		}
		// Required/release-gate evidence proves a particular workspace snapshot.
		// A later mutation makes global checks stale; path-scoped checks become stale
		// only when their declared target overlaps the changed path.
		if _, err := tx.Exec(ctx, `
			UPDATE agent_platform.agent_evidence_records AS evidence
			SET verdict='stale',workspace_revision=COALESCE(NULLIF(workspace_revision,''),$3::text)
			FROM agent_platform.agent_verification_intents AS intent
			JOIN agent_platform.agent_task_plans AS plan
			  ON plan.workflow_id=$1::uuid AND plan.revision=intent.plan_revision
			 AND plan.last_modified_run_id=intent.run_id
			WHERE evidence.intent_id=intent.id AND evidence.verdict='passed'
			  AND intent.enforcement IN ('required','release_gate')
			  AND (
				$2::text=''
				OR COALESCE(intent.kind,'') NOT IN ('file_exists','file_contains','python_syntax','list_nonempty','search_nonempty')
				OR COALESCE(intent.parameters->>'target','')=$2::text
				OR COALESCE(intent.parameters->>'target','') LIKE $2::text || '/%'
				OR $2::text LIKE COALESCE(intent.parameters->>'target','') || '/%'
			  )`, workflowID, path, expectedWorkspaceRevision); err != nil {
			return tool.Result{}, fmt.Errorf("invalidate verification evidence: %w", err)
		}
		if _, err := tx.Exec(ctx, `
			UPDATE agent_platform.agent_plan_node_states AS node
			SET status='stale',blocked_reason='workspace_changed_after_verification',updated_at=now()
			WHERE node.workflow_id=$1::uuid AND node.status='completed'
			  AND EXISTS (
				SELECT 1
				FROM agent_platform.agent_verification_intents AS intent
				JOIN agent_platform.agent_task_plans AS plan
				  ON plan.workflow_id=$1::uuid AND plan.revision=intent.plan_revision
				 AND plan.last_modified_run_id=intent.run_id
				WHERE intent.plan_step_key=node.node_id
				  AND intent.enforcement IN ('required','release_gate')
				  AND (
					$2::text=''
					OR COALESCE(intent.kind,'') NOT IN ('file_exists','file_contains','python_syntax','list_nonempty','search_nonempty')
					OR COALESCE(intent.parameters->>'target','')=$2::text
					OR COALESCE(intent.parameters->>'target','') LIKE $2::text || '/%'
					OR $2::text LIKE COALESCE(intent.parameters->>'target','') || '/%'
				  )
			  )`, workflowID, path); err != nil {
			return tool.Result{}, fmt.Errorf("invalidate verified plan nodes: %w", err)
		}
		if _, err := tx.Exec(ctx, `
			UPDATE agent_platform.agent_verification_intents AS intent
			SET status='stale',diagnostic_code='workspace_changed',
				diagnostic_message='workspace changed after verification',updated_at=now()
			FROM agent_platform.agent_task_plans AS plan
			WHERE plan.workflow_id=$1::uuid AND plan.revision=intent.plan_revision
			  AND plan.last_modified_run_id=intent.run_id
			  AND intent.enforcement IN ('required','release_gate')
			  AND intent.status='passed'
			  AND (
				$2::text=''
				OR COALESCE(intent.kind,'') NOT IN ('file_exists','file_contains','python_syntax','list_nonempty','search_nonempty')
				OR COALESCE(intent.parameters->>'target','')=$2::text
				OR COALESCE(intent.parameters->>'target','') LIKE $2::text || '/%'
				OR $2::text LIKE COALESCE(intent.parameters->>'target','') || '/%'
			  )`, workflowID, path); err != nil {
			return tool.Result{}, fmt.Errorf("invalidate verification intents: %w", err)
		}
	}
	tag, err := tx.Exec(ctx, `
		UPDATE agent_platform.agent_tool_executions AS execution
		SET status=$5::text, result=$6::jsonb, error=$7::jsonb, result_artifact_uri=$8::text,
			application_status=$9::text,finished_workspace_revision=$10,finished_at=now()
		WHERE execution.tenant_id=$1::text AND execution.idempotency_key=$2::text
		  AND execution.lease_owner=$3::text AND execution.lease_token=$4
		  AND execution.status='running'`,
		tenantID, idempotencyKey, lease.Owner, lease.Token, status, resultJSON, errorJSON,
		artifactURI, applicationStatus, currentWorkspaceRevision)
	if err != nil {
		return tool.Result{}, fmt.Errorf("finish tool execution: %w", err)
	}
	if tag.RowsAffected() != 1 {
		return tool.Result{}, agent.ErrLeaseLost
	}
	if err := tx.Commit(ctx); err != nil {
		return tool.Result{}, fmt.Errorf("commit tool execution finish: %w", err)
	}
	if !fenceCurrent {
		return tool.Result{}, staleToolFenceError("", "tool result belongs to an older workflow, plan, node, or workspace generation")
	}
	return result, nil
}

func staleToolFenceError(toolName, reason string) error {
	return tool.NewContractErrorWithRepair(
		"TOOL_RESULT_STALE",
		toolName,
		"/runtime_fence",
		"the current workflow, plan node, and workspace generation",
		"stale execution generation",
		reason,
		"Reload the durable Plan and workspace state. Do not reuse this receipt; issue a new action only if the current active node still requires it.",
		nil,
		true,
	)
}

func toolMutatesWorkspace(name string) bool {
	switch strings.TrimSpace(name) {
	case "write_file", "append_file", "edit_file", "promote_file", "create_directory":
		return true
	default:
		return false
	}
}

func workspaceMutationIdentity(result tool.Result) (path, previousHash, currentHash string) {
	var content map[string]any
	if json.Unmarshal(result.Content, &content) == nil {
		path, _ = content["path"].(string)
		previousHash, _ = content["previous_file_sha256"].(string)
		currentHash, _ = content["file_sha256"].(string)
	}
	if path == "" && result.Meta != nil {
		path = result.Meta["path"]
	}
	if previousHash == "" && result.Meta != nil {
		previousHash = result.Meta["previous_file_sha256"]
	}
	if currentHash == "" && result.Meta != nil {
		currentHash = result.Meta["file_sha256"]
	}
	return strings.TrimSpace(path), strings.TrimSpace(previousHash), strings.TrimSpace(currentHash)
}

func sha256Hex(content []byte) string {
	digest := sha256.Sum256(content)
	return hex.EncodeToString(digest[:])
}

func callIDFromKey(idempotencyKey string) string {
	if index := strings.IndexByte(idempotencyKey, ':'); index >= 0 {
		return idempotencyKey[index+1:]
	}
	return idempotencyKey
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func actionKind(provider string) string {
	switch strings.ToLower(strings.TrimSpace(provider)) {
	case "mcp":
		return "mcp"
	case "internal_agent", "a2a", "delegation":
		return "agent"
	case "platform":
		return "control"
	default:
		return "tool"
	}
}

func hashToolRequest(toolVersionID string, call tool.Call) (string, json.RawMessage, error) {
	arguments, err := canonicalToolArguments(call.Arguments)
	if err != nil {
		return "", nil, err
	}
	request, err := json.Marshal(map[string]any{
		"tool_version_id": toolVersionID, "name": call.Name,
		"arguments": arguments,
	})
	if err != nil {
		return "", nil, fmt.Errorf("encode tool execution request: %w", err)
	}
	digest := sha256.Sum256(request)
	return hex.EncodeToString(digest[:]), request, nil
}

func canonicalToolArguments(raw json.RawMessage) (any, error) {
	if len(raw) == 0 {
		return map[string]any{}, nil
	}
	var arguments any
	if err := json.Unmarshal(raw, &arguments); err != nil {
		return nil, fmt.Errorf("decode tool arguments for canonical hash: %w", err)
	}
	return arguments, nil
}
