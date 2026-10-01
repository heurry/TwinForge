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

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/approval"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

func (s *RunStore) RequireToolApproval(ctx context.Context, lease agent.Lease, tenantID, toolVersionID string, risk tool.Risk, call tool.Call, expiresIn time.Duration, preview *tool.Artifact) error {
	arguments, err := canonicalToolArguments(call.Arguments)
	if err != nil {
		return err
	}
	request, err := json.Marshal(map[string]any{"name": call.Name, "arguments": arguments})
	if err != nil {
		return err
	}
	hashInput := append([]byte(toolVersionID+"\x00"), request...)
	if preview != nil {
		previewDigest := sha256.Sum256(preview.Content)
		hashInput = append(hashInput, previewDigest[:]...)
	}
	digest := sha256.Sum256(hashInput)
	requestHash := hex.EncodeToString(digest[:])
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	var valid bool
	if err := tx.QueryRow(ctx, `SELECT true FROM agent_platform.agent_runs WHERE id=$1::uuid AND tenant_id=$2::text AND lease_owner=$3::text AND lease_token=$4 AND status='running' FOR UPDATE`, lease.RunID, tenantID, lease.Owner, lease.Token).Scan(&valid); errors.Is(err, pgx.ErrNoRows) {
		return agent.ErrLeaseLost
	} else if err != nil {
		return err
	}
	_, _ = tx.Exec(ctx, `UPDATE agent_platform.agent_tool_approvals SET status='expired', decided_at=now(), decision_reason='approval expired' WHERE run_id=$1::uuid AND call_id=$2::text AND status='pending' AND expires_at IS NOT NULL AND expires_at<=now()`, call.RunID, call.ID)
	var status, storedHash string
	err = tx.QueryRow(ctx, `SELECT status, request_hash FROM agent_platform.agent_tool_approvals WHERE tenant_id=$1::text AND run_id=$2::uuid AND call_id=$3::text FOR UPDATE`, tenantID, call.RunID, call.ID).Scan(&status, &storedHash)
	if err == nil {
		if storedHash != requestHash {
			return tool.ErrCallIDConflict
		}
		if commitErr := tx.Commit(ctx); commitErr != nil {
			return commitErr
		}
		switch status {
		case "approved":
			return nil
		case "rejected", "expired":
			return approval.ErrRejected
		default:
			return approval.ErrRequired
		}
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return err
	}
	var expiresAt any
	if expiresIn > 0 {
		expiresAt = time.Now().UTC().Add(expiresIn)
	}
	var diffArtifactID *string
	if preview != nil {
		if len(preview.Content) > 10<<20 {
			return errors.New("approval preview artifact exceeds 10 MiB")
		}
		id, err := s.persistArtifactTx(ctx, tx, tenantID, call.RunID, call.ID, artifactWrite{
			Kind: preview.Kind, Name: preview.Name, MediaType: preview.MediaType, Content: preview.Content, Metadata: preview.Metadata,
		})
		if err != nil {
			return err
		}
		diffArtifactID = &id
	}
	var approvalID string
	if err := tx.QueryRow(ctx, `INSERT INTO agent_platform.agent_tool_approvals (tenant_id,run_id,call_id,turn_no,step_no,tool_version_id,tool_name,risk,request_hash,request,diff_artifact_id,expires_at) VALUES ($1::text,$2::uuid,$3::text,$4,$5,NULLIF($6::text,'')::uuid,$7::text,$8::text,$9::text,$10::jsonb,$11::uuid,$12) RETURNING id::text`, tenantID, call.RunID, call.ID, call.Turn, call.Step, toolVersionID, call.Name, risk, requestHash, request, diffArtifactID, expiresAt).Scan(&approvalID); err != nil {
		return err
	}
	if _, err := s.appendEventTx(ctx, tx, tenantID, event.Input{RunID: call.RunID, Type: event.ToolApprovalRequested, Turn: call.Turn, Step: call.Step, CallID: call.ID, Payload: mustJSON(map[string]any{"approval_id": approvalID, "tool_name": call.Name, "risk": risk, "diff_artifact_id": diffArtifactID, "expires_at": expiresAt})}); err != nil {
		return err
	}
	if err := tx.Commit(ctx); err != nil {
		return err
	}
	return approval.ErrRequired
}

func (s *RunStore) ListApprovalsForTenant(ctx context.Context, tenantID, status string, limit int) ([]approval.Approval, error) {
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `SELECT id::text,tenant_id,run_id::text,call_id,turn_no,step_no,tool_version_id::text,tool_name,risk,request_hash,request,diff_artifact_id::text,status,requested_by,decided_by,decision_reason,expires_at,created_at,decided_at FROM agent_platform.agent_tool_approvals WHERE tenant_id=$1::text AND ($2::text='' OR status=$2::text) ORDER BY created_at DESC LIMIT $3`, tenantID, status, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var items []approval.Approval
	for rows.Next() {
		var item approval.Approval
		if err := rows.Scan(&item.ID, &item.TenantID, &item.RunID, &item.CallID, &item.Turn, &item.Step, &item.ToolVersionID, &item.ToolName, &item.Risk, &item.RequestHash, &item.Request, &item.DiffArtifactID, &item.Status, &item.RequestedBy, &item.DecidedBy, &item.DecisionReason, &item.ExpiresAt, &item.CreatedAt, &item.DecidedAt); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

func (s *RunStore) DecideApproval(ctx context.Context, tenantID, approvalID string, decision approval.Decision) (approval.Approval, error) {
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return approval.Approval{}, err
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	var item approval.Approval
	err = tx.QueryRow(ctx, `SELECT id::text,tenant_id,run_id::text,call_id,turn_no,step_no,tool_version_id::text,tool_name,risk,request_hash,request,diff_artifact_id::text,status,requested_by,decided_by,decision_reason,expires_at,created_at,decided_at FROM agent_platform.agent_tool_approvals WHERE id=$1::uuid AND tenant_id=$2::text FOR UPDATE`, approvalID, tenantID).Scan(&item.ID, &item.TenantID, &item.RunID, &item.CallID, &item.Turn, &item.Step, &item.ToolVersionID, &item.ToolName, &item.Risk, &item.RequestHash, &item.Request, &item.DiffArtifactID, &item.Status, &item.RequestedBy, &item.DecidedBy, &item.DecisionReason, &item.ExpiresAt, &item.CreatedAt, &item.DecidedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return approval.Approval{}, errors.New("approval not found")
	}
	if err != nil {
		return approval.Approval{}, err
	}
	if item.Status != "pending" {
		return approval.Approval{}, fmt.Errorf("approval is already %s", item.Status)
	}
	status := "rejected"
	if decision.Approved {
		status = "approved"
	}
	if _, err := tx.Exec(ctx, `UPDATE agent_platform.agent_tool_approvals SET status=$3::text,decided_by=NULLIF($4::text,''),decision_reason=NULLIF($5::text,''),decided_at=now() WHERE id=$1::uuid AND tenant_id=$2::text`, approvalID, tenantID, status, decision.ActorID, decision.Reason); err != nil {
		return approval.Approval{}, err
	}
	// Approval can arrive after the approval row commits but before Worker has
	// changed running -> waiting_approval. Requeue both states and fence the old
	// lease, matching the user-input resume path. If the Run is already terminal,
	// roll back the approval decision instead of resurrecting its Workflow.
	command, err := tx.Exec(ctx, `UPDATE agent_platform.agent_runs SET status='queued',next_wakeup_at=now(),lease_owner=NULL,lease_expires_at=NULL,updated_at=now() WHERE id=$1::uuid AND tenant_id=$2::text AND status IN ('running','waiting_approval')`, item.RunID, tenantID)
	if err != nil {
		return approval.Approval{}, err
	}
	if command.RowsAffected() != 1 {
		return approval.Approval{}, errors.New("approval run is no longer resumable")
	}
	var workflowID string
	if err := tx.QueryRow(ctx, `SELECT workflow_id::text FROM agent_platform.agent_runs WHERE id=$1::uuid AND tenant_id=$2::text`, item.RunID, tenantID).Scan(&workflowID); err != nil {
		return approval.Approval{}, err
	}
	if err := projectWorkflowTx(ctx, tx, tenantID, workflowID, item.RunID, "active"); err != nil {
		return approval.Approval{}, err
	}
	if _, err := s.appendEventTx(ctx, tx, tenantID, event.Input{RunID: item.RunID, Type: event.ToolApprovalResolved, Turn: item.Turn, Step: item.Step, CallID: item.CallID, Payload: mustJSON(map[string]any{"approval_id": item.ID, "status": status, "decided_by": decision.ActorID, "reason": decision.Reason})}); err != nil {
		return approval.Approval{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return approval.Approval{}, err
	}
	item.Status = status
	item.DecidedBy = &decision.ActorID
	item.DecisionReason = &decision.Reason
	now := time.Now().UTC()
	item.DecidedAt = &now
	return item, nil
}
