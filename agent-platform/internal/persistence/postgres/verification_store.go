package postgres

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/taskplan"
	verificationdomain "github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/verification"
)

// persistVerificationPlanTx materializes a Plan revision into independent,
// queryable verification facts. Plan JSON remains the rolling-upgrade input;
// these tables are the authoritative audit projection for new revisions.
func (s *RunStore) persistVerificationPlanTx(ctx context.Context, tx pgx.Tx, plan taskplan.Plan, turn, step int, callID string) error {
	for _, planStep := range plan.Steps {
		for _, criterion := range planStep.AcceptanceCriteria {
			parameters, err := json.Marshal(criterion.Verification)
			if err != nil {
				return fmt.Errorf("encode verification intent: %w", err)
			}
			enforcement := strings.TrimSpace(criterion.Enforcement)
			if enforcement == "" {
				enforcement = taskplan.EnforcementAdvisory
			}
			origin := strings.TrimSpace(criterion.Origin)
			if origin == "" {
				origin = taskplan.OriginAgentInferred
			}
			var intentID string
			err = tx.QueryRow(ctx, `
				INSERT INTO agent_platform.agent_verification_intents(
					tenant_id,run_id,plan_revision,plan_step_key,criterion_key,description,
					kind,enforcement,origin,parameters,status,diagnostic_code,diagnostic_message)
				VALUES($1,$2::uuid,$3,$4,$5,$6,NULLIF($7,''),$8,$9,$10::jsonb,$11,NULLIF($12,''),NULLIF($13,''))
				ON CONFLICT(run_id,plan_revision,plan_step_key,criterion_key) DO UPDATE SET
					description=EXCLUDED.description,kind=EXCLUDED.kind,enforcement=EXCLUDED.enforcement,
					origin=EXCLUDED.origin,parameters=EXCLUDED.parameters,status=EXCLUDED.status,
					diagnostic_code=EXCLUDED.diagnostic_code,diagnostic_message=EXCLUDED.diagnostic_message,updated_at=now()
				RETURNING id::text`, plan.TenantID, plan.EffectiveRunID(), plan.Revision, planStep.ID,
				criterion.ID, criterion.Description, criterion.Verification.Kind, enforcement, origin,
				parameters, criterion.Status, criterion.VerificationReason, criterion.VerificationMessage).Scan(&intentID)
			if err != nil {
				return fmt.Errorf("persist verification intent: %w", err)
			}
			if _, err := s.appendEventTx(ctx, tx, plan.TenantID, event.Input{RunID: plan.EffectiveRunID(), WorkflowID: plan.WorkflowID, Type: event.VerificationIntentCreated, Turn: turn, Step: step, CallID: callID, Payload: mustJSON(map[string]any{
				"intent_id": intentID, "plan_revision": plan.Revision, "plan_step_key": planStep.ID,
				"criterion_key": criterion.ID, "kind": criterion.Verification.Kind,
				"enforcement": enforcement, "origin": origin, "status": criterion.Status,
			})}); err != nil {
				return err
			}

			if criterion.Verification.Kind == "" || criterion.Status == taskplan.CriterionInvalid || criterion.Status == taskplan.CriterionUnsupported {
				continue
			}
			intent := verificationdomain.Intent{ID: intentID, TenantID: plan.TenantID, RunID: plan.EffectiveRunID(),
				PlanRevision: plan.Revision, PlanStepKey: planStep.ID, CriterionKey: criterion.ID,
				Description: criterion.Description, Kind: criterion.Verification.Kind,
				Enforcement: enforcement, Origin: origin, Parameters: parameters, Status: criterion.Status}
			spec, compileErr := verificationdomain.DefaultRegistry.Compile(ctx, intent)
			if compileErr != nil {
				if _, err := tx.Exec(ctx, `UPDATE agent_platform.agent_verification_intents SET status='invalid',diagnostic_code='spec_invalid',diagnostic_message=$2,updated_at=now() WHERE id=$1::uuid`, intentID, compileErr.Error()); err != nil {
					return err
				}
				continue
			}
			var specID string
			if err := tx.QueryRow(ctx, `
				INSERT INTO agent_platform.agent_verification_specs(
					tenant_id,run_id,intent_id,provider_key,provider_version,subject,execution,assertions,spec_digest,status)
				VALUES($1,$2::uuid,$3::uuid,$4,$5,$6::jsonb,$7::jsonb,$8::jsonb,$9,$10)
				ON CONFLICT(intent_id) DO UPDATE SET provider_key=EXCLUDED.provider_key,
					provider_version=EXCLUDED.provider_version,subject=EXCLUDED.subject,
					execution=EXCLUDED.execution,assertions=EXCLUDED.assertions,
					spec_digest=EXCLUDED.spec_digest,status=EXCLUDED.status
				RETURNING id::text`, plan.TenantID, plan.EffectiveRunID(), intentID, spec.ProviderKey,
				spec.ProviderVersion, jsonObject(spec.Subject), jsonObject(spec.Execution), jsonArray(spec.Assertions),
				spec.Digest, spec.Status).Scan(&specID); err != nil {
				return fmt.Errorf("persist verification spec: %w", err)
			}
			if _, err := s.appendEventTx(ctx, tx, plan.TenantID, event.Input{RunID: plan.EffectiveRunID(), WorkflowID: plan.WorkflowID, Type: event.VerificationSpecCompiled, Turn: turn, Step: step, CallID: callID, Payload: mustJSON(map[string]any{
				"intent_id": intentID, "spec_id": specID, "provider_key": spec.ProviderKey,
				"provider_version": spec.ProviderVersion, "spec_digest": spec.Digest,
			})}); err != nil {
				return err
			}
			for _, evidenceCallID := range criterion.EvidenceCallIDs {
				if err := s.persistVerificationReceiptTx(ctx, tx, plan, planStep.ID, criterion, intentID, specID, evidenceCallID, turn, step, callID); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func (s *RunStore) persistVerificationReceiptTx(ctx context.Context, tx pgx.Tx, plan taskplan.Plan, planStepID string, criterion taskplan.AcceptanceCriterion, intentID, specID, evidenceCallID string, turn, step int, causationCallID string) error {
	var toolExecutionID, toolName string
	var arguments, result json.RawMessage
	if err := tx.QueryRow(ctx, `SELECT id::text,tool_name,COALESCE(request->'arguments','{}'::jsonb),COALESCE(result->'content','{}'::jsonb) FROM agent_platform.agent_tool_executions WHERE tenant_id=$1 AND run_id=$2::uuid AND call_id=$3 AND status='succeeded' AND application_status='applied'`, plan.TenantID, plan.EffectiveRunID(), evidenceCallID).Scan(&toolExecutionID, &toolName, &arguments, &result); errors.Is(err, pgx.ErrNoRows) {
		// Runtime-owned tools such as delegate_agent produce the same committed
		// TOOL_COMPLETED fact as Sandbox tools, but intentionally have no
		// agent_tool_executions row. Materialize that event as auditable evidence
		// with a nullable tool_execution_id instead of silently dropping it.
		if eventErr := tx.QueryRow(ctx, `
			SELECT COALESCE(payload->>'name',''),COALESCE(payload->'arguments','{}'::jsonb),
			       COALESCE(payload->'result'->'content','{}'::jsonb)
			FROM agent_platform.agent_events AS event
			WHERE event.run_id=$1::uuid AND event.call_id=$2 AND event.event_type=$3
			  AND NOT EXISTS (
				SELECT 1 FROM agent_platform.agent_tool_executions AS execution
				WHERE execution.run_id=event.run_id AND execution.call_id=event.call_id
			  )
			ORDER BY seq DESC LIMIT 1`, plan.EffectiveRunID(), evidenceCallID, string(event.ToolCompleted)).Scan(&toolName, &arguments, &result); errors.Is(eventErr, pgx.ErrNoRows) {
			return nil // A rolling-upgrade Plan may still reference pruned evidence.
		} else if eventErr != nil {
			return fmt.Errorf("load event-backed verification receipt: %w", eventErr)
		}
	} else if err != nil {
		return fmt.Errorf("load verification receipt: %w", err)
	}
	resultRaw, _ := json.Marshal(map[string]any{"tool": toolName, "arguments": json.RawMessage(arguments), "result": json.RawMessage(result)})
	var attemptID string
	if err := tx.QueryRow(ctx, `
		INSERT INTO agent_platform.agent_verification_attempts(tenant_id,run_id,spec_id,tool_execution_id,status,result,finished_at)
		VALUES($1,$2::uuid,$3::uuid,NULLIF($4,'')::uuid,'passed',$5::jsonb,now())
		ON CONFLICT(spec_id,tool_execution_id) WHERE tool_execution_id IS NOT NULL DO UPDATE SET status='passed',result=EXCLUDED.result,finished_at=now()
		RETURNING id::text`, plan.TenantID, plan.EffectiveRunID(), specID, toolExecutionID, resultRaw).Scan(&attemptID); err != nil {
		return fmt.Errorf("persist verification attempt: %w", err)
	}
	digest := sha256.Sum256(resultRaw)
	resultDigest := hex.EncodeToString(digest[:])
	evidenceRaw, _ := json.Marshal(map[string]any{"call_id": evidenceCallID, "tool": toolName, "summary": criterion.Evidence})
	var evidenceID string
	if err := tx.QueryRow(ctx, `
		INSERT INTO agent_platform.agent_evidence_records(tenant_id,run_id,intent_id,spec_id,attempt_id,tool_execution_id,verdict,evidence,result_digest,workspace_revision)
		SELECT $1,$2::uuid,$3::uuid,$4::uuid,$5::uuid,NULLIF($6,'')::uuid,'passed',$7::jsonb,$8,workflow.workspace_revision::text
		FROM agent_platform.agent_workflows AS workflow WHERE workflow.id=$9::uuid
		ON CONFLICT(attempt_id) DO UPDATE SET verdict='passed',evidence=EXCLUDED.evidence,
			result_digest=EXCLUDED.result_digest,workspace_revision=EXCLUDED.workspace_revision
		RETURNING id::text`, plan.TenantID, plan.EffectiveRunID(), intentID, specID, attemptID, toolExecutionID, evidenceRaw, resultDigest, plan.WorkflowID).Scan(&evidenceID); err != nil {
		return fmt.Errorf("persist verification evidence: %w", err)
	}
	if _, err := s.appendEventTx(ctx, tx, plan.TenantID, event.Input{RunID: plan.EffectiveRunID(), WorkflowID: plan.WorkflowID, Type: event.VerificationCompleted, Turn: turn, Step: step, CallID: causationCallID, Payload: mustJSON(map[string]any{
		"intent_id": intentID, "spec_id": specID, "attempt_id": attemptID, "evidence_id": evidenceID,
		"plan_step_key": planStepID, "criterion_key": criterion.ID, "tool_call_id": evidenceCallID,
		"tool_execution_id": toolExecutionID, "verdict": "passed",
	})}); err != nil {
		return err
	}
	return nil
}

func (s *RunStore) recordVerificationFailure(ctx context.Context, tenantID, runID, planStepID string, criterion taskplan.AcceptanceCriterion, receipt successfulToolEvidence, cause error) error {
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	var intentID, specID, workflowID string
	err = tx.QueryRow(ctx, `
		SELECT intent.id::text,spec.id::text,plan.workflow_id::text
		FROM agent_platform.agent_task_plans AS plan
		JOIN agent_platform.agent_verification_intents AS intent
		  ON intent.run_id=plan.run_id AND intent.plan_revision=plan.revision
		 AND intent.plan_step_key=$3 AND intent.criterion_key=$4
		JOIN agent_platform.agent_verification_specs AS spec ON spec.intent_id=intent.id
		WHERE plan.tenant_id=$1 AND plan.run_id=$2::uuid`, tenantID, runID, planStepID, criterion.ID).Scan(&intentID, &specID, &workflowID)
	if errors.Is(err, pgx.ErrNoRows) {
		return tx.Commit(ctx)
	}
	if err != nil {
		return err
	}
	resultRaw, _ := json.Marshal(map[string]any{"tool": receipt.Name, "arguments": json.RawMessage(receipt.Arguments), "result": json.RawMessage(receipt.Result)})
	var attemptID string
	if err := tx.QueryRow(ctx, `
		INSERT INTO agent_platform.agent_verification_attempts(tenant_id,run_id,spec_id,tool_execution_id,status,reason_code,diagnostic,result,finished_at)
		VALUES($1,$2::uuid,$3::uuid,NULLIF($4,'')::uuid,'failed',$5,$6,$7::jsonb,now())
		ON CONFLICT(spec_id,tool_execution_id) WHERE tool_execution_id IS NOT NULL DO UPDATE SET
			status='failed',reason_code=EXCLUDED.reason_code,diagnostic=EXCLUDED.diagnostic,result=EXCLUDED.result,finished_at=now()
		RETURNING id::text`, tenantID, runID, specID, receipt.ExecutionID,
		taskplan.VerificationReasonAssertionFailed, cause.Error(), resultRaw).Scan(&attemptID); err != nil {
		return err
	}
	digest := sha256.Sum256(resultRaw)
	evidenceRaw, _ := json.Marshal(map[string]any{"call_id": receipt.CallID, "tool": receipt.Name, "diagnostic": cause.Error()})
	var evidenceID string
	if err := tx.QueryRow(ctx, `
		INSERT INTO agent_platform.agent_evidence_records(tenant_id,run_id,intent_id,spec_id,attempt_id,tool_execution_id,verdict,evidence,result_digest,workspace_revision)
		SELECT $1,$2::uuid,$3::uuid,$4::uuid,$5::uuid,NULLIF($6,'')::uuid,'failed',$7::jsonb,$8,workflow.workspace_revision::text
		FROM agent_platform.agent_workflows AS workflow WHERE workflow.id=$9::uuid
		ON CONFLICT(attempt_id) DO UPDATE SET verdict='failed',evidence=EXCLUDED.evidence,
			result_digest=EXCLUDED.result_digest,workspace_revision=EXCLUDED.workspace_revision
		RETURNING id::text`, tenantID, runID, intentID, specID, attemptID, receipt.ExecutionID, evidenceRaw, hex.EncodeToString(digest[:]), workflowID).Scan(&evidenceID); err != nil {
		return err
	}
	if _, err := s.appendEventTx(ctx, tx, tenantID, event.Input{RunID: runID, WorkflowID: workflowID, Type: event.VerificationFailed, CallID: receipt.CallID, Payload: mustJSON(map[string]any{
		"intent_id": intentID, "spec_id": specID, "attempt_id": attemptID, "evidence_id": evidenceID,
		"plan_step_key": planStepID, "criterion_key": criterion.ID, "tool_call_id": receipt.CallID,
		"tool_execution_id": receipt.ExecutionID, "verdict": "failed", "reason_code": taskplan.VerificationReasonAssertionFailed,
	})}); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

func jsonObject(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 || string(raw) == "null" {
		return json.RawMessage(`{}`)
	}
	return raw
}
func jsonArray(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 || string(raw) == "null" {
		return json.RawMessage(`[]`)
	}
	return raw
}

func (s *RunStore) ListVerificationRecordsForTenant(ctx context.Context, tenantID, runID string) ([]verificationdomain.Record, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT intent.id::text,intent.plan_revision,intent.plan_step_key,intent.criterion_key,
			intent.description,COALESCE(intent.kind,''),intent.enforcement,intent.origin,intent.parameters,
			intent.status,COALESCE(intent.diagnostic_code,''),COALESCE(intent.diagnostic_message,''),intent.created_at,intent.updated_at,
			spec.id::text,COALESCE(spec.provider_key,''),COALESCE(spec.provider_version,''),
			COALESCE(spec.subject,'{}'::jsonb),COALESCE(spec.execution,'{}'::jsonb),COALESCE(spec.assertions,'[]'::jsonb),
			COALESCE(spec.spec_digest,''),COALESCE(spec.status,''),spec.created_at
		FROM agent_platform.agent_verification_intents AS intent
		LEFT JOIN agent_platform.agent_verification_specs AS spec ON spec.intent_id=intent.id
		WHERE intent.tenant_id=$1 AND intent.run_id=$2::uuid
		ORDER BY intent.plan_revision,intent.created_at,intent.id`, tenantID, runID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	records := make([]verificationdomain.Record, 0)
	bySpec := make(map[string]int)
	for rows.Next() {
		var record verificationdomain.Record
		record.Intent.TenantID, record.Intent.RunID = tenantID, runID
		var specID *string
		var providerKey, providerVersion string
		var subject, execution, assertions json.RawMessage
		var digest, specStatus string
		var specCreated *time.Time
		if err := rows.Scan(&record.Intent.ID, &record.Intent.PlanRevision, &record.Intent.PlanStepKey, &record.Intent.CriterionKey,
			&record.Intent.Description, &record.Intent.Kind, &record.Intent.Enforcement, &record.Intent.Origin, &record.Intent.Parameters,
			&record.Intent.Status, &record.Intent.DiagnosticCode, &record.Intent.DiagnosticMessage, &record.Intent.CreatedAt, &record.Intent.UpdatedAt,
			&specID, &providerKey, &providerVersion, &subject, &execution, &assertions, &digest, &specStatus, &specCreated); err != nil {
			return nil, err
		}
		if specID != nil {
			record.Spec = &verificationdomain.ExecutableSpec{ID: *specID, IntentID: record.Intent.ID, ProviderKey: providerKey, ProviderVersion: providerVersion, Subject: subject, Execution: execution, Assertions: assertions, Digest: digest, Status: specStatus}
			if specCreated != nil {
				record.Spec.CreatedAt = *specCreated
			}
			bySpec[*specID] = len(records)
		}
		record.Attempts = []verificationdomain.Attempt{}
		record.Evidence = []verificationdomain.Evidence{}
		records = append(records, record)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	attempts, err := s.pool.Query(ctx, `SELECT attempt.id::text,attempt.spec_id::text,COALESCE(attempt.tool_execution_id::text,''),attempt.status,COALESCE(attempt.reason_code,''),COALESCE(attempt.diagnostic,''),attempt.result,attempt.started_at,attempt.finished_at FROM agent_platform.agent_verification_attempts AS attempt WHERE attempt.tenant_id=$1 AND attempt.run_id=$2::uuid ORDER BY attempt.started_at,attempt.id`, tenantID, runID)
	if err != nil {
		return nil, err
	}
	for attempts.Next() {
		var value verificationdomain.Attempt
		if err := attempts.Scan(&value.ID, &value.SpecID, &value.ToolExecutionID, &value.Status, &value.ReasonCode, &value.Diagnostic, &value.Result, &value.StartedAt, &value.FinishedAt); err != nil {
			attempts.Close()
			return nil, err
		}
		if index, ok := bySpec[value.SpecID]; ok {
			records[index].Attempts = append(records[index].Attempts, value)
		}
	}
	if err := attempts.Err(); err != nil {
		attempts.Close()
		return nil, err
	}
	attempts.Close()
	evidenceRows, err := s.pool.Query(ctx, `SELECT evidence.id::text,evidence.intent_id::text,evidence.spec_id::text,evidence.attempt_id::text,COALESCE(evidence.tool_execution_id::text,''),evidence.verdict,evidence.evidence,COALESCE(evidence.result_digest,''),COALESCE(evidence.workspace_revision,''),evidence.created_at FROM agent_platform.agent_evidence_records AS evidence WHERE evidence.tenant_id=$1 AND evidence.run_id=$2::uuid ORDER BY evidence.created_at,evidence.id`, tenantID, runID)
	if err != nil {
		return nil, err
	}
	defer evidenceRows.Close()
	for evidenceRows.Next() {
		var value verificationdomain.Evidence
		if err := evidenceRows.Scan(&value.ID, &value.IntentID, &value.SpecID, &value.AttemptID, &value.ToolExecutionID, &value.Verdict, &value.Payload, &value.ResultDigest, &value.WorkspaceRevision, &value.CreatedAt); err != nil {
			return nil, err
		}
		if index, ok := bySpec[value.SpecID]; ok {
			records[index].Evidence = append(records[index].Evidence, value)
		}
	}
	return records, evidenceRows.Err()
}
