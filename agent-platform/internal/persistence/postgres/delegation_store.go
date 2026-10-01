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

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/delegation"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

// DescribeDelegationTargets resolves the human-readable labels for an
// immutable collaboration allowlist. Missing or unpublished targets remain in
// the result as unavailable entries so callers can explain the policy rather
// than silently hiding the capability.
func (s *RunStore) DescribeDelegationTargets(ctx context.Context, tenantID string, allowlist []agent.CollaborationTarget) ([]delegation.Target, error) {
	result := make([]delegation.Target, 0, len(allowlist))
	seen := make(map[string]struct{}, len(allowlist))
	for _, item := range allowlist {
		versionID := strings.TrimSpace(item.AgentVersionID)
		if versionID == "" {
			continue
		}
		if _, ok := seen[versionID]; ok {
			continue
		}
		seen[versionID] = struct{}{}
		modes := append([]string(nil), item.Modes...)
		info := delegation.Target{AgentID: item.AgentID, AgentVersionID: versionID, Modes: modes}
		var status string
		err := s.pool.QueryRow(ctx, `
			SELECT definition.agent_key, definition.name, version.version, version.status
			FROM agent_platform.agent_versions AS version
			JOIN agent_platform.agent_definitions AS definition ON definition.id=version.agent_id
			WHERE version.id=$1::uuid AND definition.id=$2::uuid AND definition.tenant_id=$3::text`,
			versionID, item.AgentID, tenantID).Scan(&info.AgentKey, &info.AgentName, &info.Version, &status)
		if errors.Is(err, pgx.ErrNoRows) {
			result = append(result, info)
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("describe delegation target %s: %w", versionID, err)
		}
		info.Available = status == "published"
		result = append(result, info)
	}
	return result, nil
}

// ResolveDelegationTarget canonicalizes a model-provided UUID, Agent key, or
// display name against the published parent's allowlist. It never searches
// the tenant catalog outside that allowlist, preventing alias-based privilege
// escalation.
func (s *RunStore) ResolveDelegationTarget(ctx context.Context, tenantID string, allowlist []agent.CollaborationTarget, requested string) (string, []delegation.Target, error) {
	targets, err := s.DescribeDelegationTargets(ctx, tenantID, allowlist)
	if err != nil {
		return strings.TrimSpace(requested), nil, err
	}
	requested = strings.TrimSpace(requested)
	for _, target := range targets {
		if target.AgentVersionID == requested {
			return target.AgentVersionID, targets, nil
		}
	}
	normalized := strings.ToLower(requested)
	var match string
	for _, target := range targets {
		if !target.Available {
			continue
		}
		if strings.ToLower(strings.TrimSpace(target.AgentKey)) == normalized || strings.ToLower(strings.TrimSpace(target.AgentName)) == normalized {
			if match != "" && match != target.AgentVersionID {
				return requested, targets, nil
			}
			match = target.AgentVersionID
		}
	}
	if match != "" {
		return match, targets, nil
	}
	return requested, targets, nil
}

func (s *RunStore) ResolveDelegation(ctx context.Context, lease agent.Lease, parent agent.Run, call tool.Call, request delegation.Request, policy agent.CollaborationPolicy) (delegation.Outcome, error) {
	if request.Mode == "" {
		request.Mode = "sync"
	}
	if request.Mode != "sync" && request.Mode != "async" {
		return delegation.Outcome{}, errors.New("delegation mode must be sync or async")
	}
	requestedTarget := strings.TrimSpace(request.TargetAgentVersionID)
	if requestedTarget == "" {
		requestedTarget = strings.TrimSpace(request.TargetAgent)
	}
	canonicalTarget, targets, err := s.ResolveDelegationTarget(ctx, parent.TenantID, policy.AllowedTargets, requestedTarget)
	if err != nil {
		return delegation.Outcome{}, err
	}
	request.TargetAgentVersionID = canonicalTarget
	allowed := false
	allowedAgentID := ""
	for _, target := range policy.AllowedTargets {
		if target.AgentVersionID == request.TargetAgentVersionID {
			for _, mode := range target.Modes {
				if mode == request.Mode {
					allowed = true
					allowedAgentID = target.AgentID
				}
			}
		}
	}
	if !allowed {
		return delegation.Outcome{}, &delegation.TargetNotAllowedError{
			RequestedTarget: requestedTarget,
			RequestedMode:   request.Mode,
			AllowedTargets:  targets,
			Reason:          "target alias, AgentVersion ID, or requested mode is not in the published allowlist",
		}
	}
	if parent.DelegationDepth >= policy.MaxDepth {
		return delegation.Outcome{}, fmt.Errorf("delegation max depth %d reached", policy.MaxDepth)
	}
	payload, _ := json.Marshal(request)
	sum := sha256.Sum256(payload)
	requestHash := hex.EncodeToString(sum[:])
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return delegation.Outcome{}, err
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	var valid bool
	if err := tx.QueryRow(ctx, `SELECT true FROM agent_platform.agent_runs WHERE id=$1::uuid AND tenant_id=$2::text AND lease_owner=$3::text AND lease_token=$4 AND status='running' FOR UPDATE`, parent.ID, parent.TenantID, lease.Owner, lease.Token).Scan(&valid); errors.Is(err, pgx.ErrNoRows) {
		return delegation.Outcome{}, agent.ErrLeaseLost
	} else if err != nil {
		return delegation.Outcome{}, err
	}
	var existing delegation.Outcome
	var storedHash string
	err = tx.QueryRow(ctx, `SELECT id::text,COALESCE(child_run_id::text,''),status,output,COALESCE(error,''),request_hash FROM agent_platform.agent_delegations WHERE tenant_id=$1::text AND parent_run_id=$2::uuid AND call_id=$3::text`, parent.TenantID, parent.ID, call.ID).Scan(&existing.DelegationID, &existing.ChildRunID, &existing.Status, &existing.Output, &existing.Error, &storedHash)
	if err == nil {
		if storedHash != requestHash {
			return existing, tool.ErrCallIDConflict
		}
		if commitErr := tx.Commit(ctx); commitErr != nil {
			return existing, commitErr
		}
		switch existing.Status {
		case "completed":
			return existing, nil
		case "failed", "cancelled":
			return existing, fmt.Errorf("child agent %s: %s", existing.Status, existing.Error)
		default:
			if request.Mode == "async" {
				return existing, nil
			}
			return existing, delegation.ErrPending
		}
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return existing, err
	}
	var targetSpecRaw json.RawMessage
	var targetAgentID string
	if err := tx.QueryRow(ctx, `SELECT version.spec,version.agent_id::text FROM agent_platform.agent_versions version JOIN agent_platform.agent_definitions definition ON definition.id=version.agent_id WHERE version.id=$1::uuid AND version.status='published' AND definition.tenant_id=$2::text`, request.TargetAgentVersionID, parent.TenantID).Scan(&targetSpecRaw, &targetAgentID); errors.Is(err, pgx.ErrNoRows) {
		return existing, errors.New("target AgentVersion is not published in this tenant")
	} else if err != nil {
		return existing, err
	}
	if targetAgentID != allowedAgentID {
		return existing, errors.New("delegation target Agent identity does not match the published allowlist")
	}
	var targetSpec agent.Spec
	if err := json.Unmarshal(targetSpecRaw, &targetSpec); err != nil {
		return existing, fmt.Errorf("decode target AgentVersion: %w", err)
	}
	reservedModelCalls := targetSpec.Runtime.MaxModelCalls
	reservedToolCalls := targetSpec.Runtime.MaxToolCalls
	reservedTokens := int64(targetSpec.Context.MaxInputTokens) * int64(targetSpec.Runtime.MaxModelCalls)
	var usedModelCalls, usedToolCalls int
	var usedTokens int64
	if err := tx.QueryRow(ctx, `SELECT COALESCE(sum((version.spec->'runtime'->>'max_model_calls')::integer),0),COALESCE(sum((version.spec->'runtime'->>'max_tool_calls')::integer),0),COALESCE(sum((version.spec->'runtime'->>'max_model_calls')::bigint*(version.spec->'context'->>'max_input_tokens')::bigint),0) FROM agent_platform.agent_delegations delegation JOIN agent_platform.agent_versions version ON version.id=delegation.target_agent_version_id WHERE delegation.parent_run_id=$1::uuid AND delegation.status NOT IN ('failed','cancelled')`, parent.ID).Scan(&usedModelCalls, &usedToolCalls, &usedTokens); err != nil {
		return existing, err
	}
	if limit := policy.Budget.MaxModelCalls; limit > 0 && usedModelCalls+reservedModelCalls > limit {
		return existing, fmt.Errorf("delegation model-call budget %d would be exceeded", limit)
	}
	if limit := policy.Budget.MaxToolCalls; limit > 0 && usedToolCalls+reservedToolCalls > limit {
		return existing, fmt.Errorf("delegation tool-call budget %d would be exceeded", limit)
	}
	if limit := policy.Budget.MaxTokens; limit > 0 && usedTokens+reservedTokens > limit {
		return existing, fmt.Errorf("delegation token reservation budget %d would be exceeded", limit)
	}
	var childCount int
	if err := tx.QueryRow(ctx, `SELECT count(*) FROM agent_platform.agent_delegations WHERE parent_run_id=$1::uuid`, parent.ID).Scan(&childCount); err != nil {
		return existing, err
	}
	if childCount >= policy.MaxChildRuns {
		return existing, fmt.Errorf("delegation child-run limit %d reached", policy.MaxChildRuns)
	}
	var fanOut int
	if err := tx.QueryRow(ctx, `SELECT count(*) FROM agent_platform.agent_delegations WHERE parent_run_id=$1::uuid AND turn_no=$2 AND step_no=$3 AND status NOT IN ('failed','cancelled')`, parent.ID, call.Turn, call.Step).Scan(&fanOut); err != nil {
		return existing, err
	}
	if fanOut >= policy.MaxFanOut {
		return existing, fmt.Errorf("delegation fan-out limit %d reached for turn %d step %d", policy.MaxFanOut, call.Turn, call.Step)
	}
	var cycle bool
	if err := tx.QueryRow(ctx, `WITH RECURSIVE ancestry AS (SELECT id,parent_run_id,agent_version_id FROM agent_platform.agent_runs WHERE id=$1::uuid UNION ALL SELECT run.id,run.parent_run_id,run.agent_version_id FROM agent_platform.agent_runs run JOIN ancestry ON run.id=ancestry.parent_run_id) SELECT EXISTS(SELECT 1 FROM ancestry WHERE agent_version_id=$2::uuid)`, parent.ID, request.TargetAgentVersionID).Scan(&cycle); err != nil {
		return existing, err
	}
	if cycle {
		return existing, errors.New("delegation cycle detected")
	}
	deadline := time.Now().UTC().Add(policy.ChildTimeout)
	if policy.ChildTimeout <= 0 {
		deadline = time.Now().UTC().Add(2 * time.Minute)
	}
	if err := tx.QueryRow(ctx, `INSERT INTO agent_platform.agent_delegations(tenant_id,parent_run_id,source_agent_version_id,target_agent_version_id,call_id,turn_no,step_no,mode,request_hash,input,deadline) VALUES($1::text,$2::uuid,$3::uuid,$4::uuid,$5::text,$6,$7,$8::text,$9::text,$10::jsonb,$11) RETURNING id::text`, parent.TenantID, parent.ID, parent.AgentVersionID, request.TargetAgentVersionID, call.ID, call.Turn, call.Step, request.Mode, requestHash, request.Input, deadline).Scan(&existing.DelegationID); err != nil {
		return existing, err
	}
	rootID := parent.ID
	if parent.RootRunID != nil {
		rootID = *parent.RootRunID
	}
	err = tx.QueryRow(ctx, `INSERT INTO agent_platform.agent_runs(tenant_id,workflow_id,agent_version_id,status,trigger_type,input,binding_snapshot,created_by,parent_run_id,root_run_id,delegation_id,delegation_depth) SELECT $1::text,$3::uuid,version.id,'queued','delegation',$4::jsonb,jsonb_build_object('agent_version_id',version.id::text,'version',version.version,'spec_hash',version.spec_hash,'spec',version.spec),$5::text,$6::uuid,$7::uuid,$8::uuid,$9 FROM agent_platform.agent_versions version JOIN agent_platform.agent_definitions definition ON definition.id=version.agent_id WHERE version.id=$2::uuid AND version.status='published' AND definition.tenant_id=$1::text RETURNING id::text`, parent.TenantID, request.TargetAgentVersionID, parent.WorkflowID, request.Input, parent.CreatedBy, parent.ID, rootID, existing.DelegationID, parent.DelegationDepth+1).Scan(&existing.ChildRunID)
	if errors.Is(err, pgx.ErrNoRows) {
		return existing, errors.New("target AgentVersion is not published in this tenant")
	}
	if err != nil {
		return existing, err
	}
	if _, err := tx.Exec(ctx, `UPDATE agent_platform.agent_delegations SET child_run_id=$2::uuid WHERE id=$1::uuid`, existing.DelegationID, existing.ChildRunID); err != nil {
		return existing, err
	}
	createdPayload := mustJSON(map[string]any{"trigger_type": "delegation", "parent_run_id": parent.ID, "delegation_id": existing.DelegationID, "input": json.RawMessage(request.Input)})
	if _, err := s.appendEventTx(ctx, tx, parent.TenantID, event.Input{RunID: existing.ChildRunID, WorkflowID: parent.WorkflowID, Type: event.RunCreated, Payload: createdPayload}); err != nil {
		return existing, err
	}
	if _, err := s.appendEventTx(ctx, tx, parent.TenantID, event.Input{RunID: parent.ID, WorkflowID: parent.WorkflowID, Type: event.DelegationRequested, Turn: call.Turn, DecisionCycle: call.DecisionCycle, Step: call.Step, ActionID: call.ActionID, CallID: call.ID, Payload: mustJSON(map[string]any{"delegation_id": existing.DelegationID, "child_run_id": existing.ChildRunID, "target_agent_version_id": request.TargetAgentVersionID, "mode": request.Mode, "workflow_id": parent.WorkflowID, "action_id": firstNonEmpty(call.ActionID, call.ID), "action_kind": "agent"})}); err != nil {
		return existing, err
	}
	if err := tx.Commit(ctx); err != nil {
		return existing, err
	}
	existing.Status = "queued"
	if request.Mode == "sync" {
		return existing, delegation.ErrPending
	}
	return existing, nil
}

func (s *RunStore) ListChildRunsForTenant(ctx context.Context, tenantID, parentRunID string) ([]agent.Run, error) {
	rows, err := s.pool.Query(ctx, `SELECT `+qualifiedRunColumns("run")+` FROM agent_platform.agent_runs run WHERE run.tenant_id=$1::text AND run.parent_run_id=$2::uuid ORDER BY run.created_at`, tenantID, parentRunID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var result []agent.Run
	for rows.Next() {
		item, err := scanRun(rows)
		if err != nil {
			return nil, err
		}
		result = append(result, item)
	}
	return result, rows.Err()
}
