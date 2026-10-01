package postgres

import (
	"context"
	"fmt"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
)

// GetSessionAudit derives one conversation's usage and reliability metrics
// from durable execution projections. No process-local counter is required,
// and tenant ownership is applied before any usage row is aggregated.
func (s *RunStore) GetSessionAudit(ctx context.Context, tenantID, sessionID string) (agent.SessionAudit, error) {
	if _, err := s.GetSession(ctx, tenantID, sessionID); err != nil {
		return agent.SessionAudit{}, err
	}

	rows, err := s.pool.Query(ctx, `
		WITH session_runs AS MATERIALIZED (
			SELECT id, workflow_id, status, started_at, finished_at, created_at, updated_at
			FROM agent_platform.agent_runs
			WHERE tenant_id=$1::text AND session_id=$2::uuid
		), model_tokens AS (
			SELECT call.run_id,
				COALESCE(sum(call.input_tokens),0) AS input_tokens,
				COALESCE(sum(call.output_tokens),0) AS output_tokens
			FROM agent_platform.agent_model_calls AS call
			JOIN session_runs AS run ON run.id=call.run_id
			GROUP BY call.run_id
		), lifecycle_stats AS (
			SELECT lifecycle.run_id,
				count(*) FILTER (WHERE lifecycle.event_type='MODEL_REQUESTED') AS model_calls,
				count(*) FILTER (WHERE lifecycle.event_type='MODEL_COMPLETED') AS successful_model_calls,
				count(*) FILTER (WHERE lifecycle.event_type='MODEL_FAILED') AS failed_model_calls,
				count(*) FILTER (WHERE lifecycle.event_type='TOOL_CALLED') AS tool_calls,
				count(*) FILTER (WHERE lifecycle.event_type='TOOL_COMPLETED') AS successful_tool_calls,
				count(*) FILTER (WHERE lifecycle.event_type='TOOL_FAILED') AS failed_tool_calls
			FROM agent_platform.agent_events AS lifecycle
			JOIN session_runs AS run ON run.id=lifecycle.run_id
			WHERE lifecycle.event_type IN ('MODEL_REQUESTED','MODEL_COMPLETED','MODEL_FAILED','TOOL_CALLED','TOOL_COMPLETED','TOOL_FAILED')
			GROUP BY lifecycle.run_id
		)
		SELECT run.id::text, run.workflow_id::text, run.status,
			run.started_at, run.finished_at,
			CASE WHEN run.started_at IS NULL THEN 0 ELSE GREATEST(0,
				EXTRACT(EPOCH FROM (COALESCE(run.finished_at,
					CASE WHEN run.status IN ('queued','running','waiting_tool','waiting_approval','waiting_input','waiting_external') THEN now() ELSE run.updated_at END)
					- run.started_at))*1000)::bigint END AS duration_ms,
			COALESCE(lifecycle.model_calls,0), COALESCE(lifecycle.successful_model_calls,0), COALESCE(lifecycle.failed_model_calls,0),
			COALESCE(tokens.input_tokens,0), COALESCE(tokens.output_tokens,0),
			COALESCE(lifecycle.tool_calls,0), COALESCE(lifecycle.successful_tool_calls,0), COALESCE(lifecycle.failed_tool_calls,0),
			COALESCE(run.started_at,run.created_at), COALESCE(run.finished_at,run.updated_at)
		FROM session_runs AS run
		LEFT JOIN model_tokens AS tokens ON tokens.run_id=run.id
		LEFT JOIN lifecycle_stats AS lifecycle ON lifecycle.run_id=run.id
		ORDER BY run.created_at`, tenantID, sessionID)
	if err != nil {
		return agent.SessionAudit{}, fmt.Errorf("query session audit: %w", err)
	}
	defer rows.Close()

	audit := agent.SessionAudit{SessionID: sessionID, Runs: make([]agent.RunAudit, 0)}
	for rows.Next() {
		var item agent.RunAudit
		var firstAt, lastAt time.Time
		if err := rows.Scan(
			&item.RunID, &item.WorkflowID, &item.Status, &item.StartedAt, &item.FinishedAt, &item.DurationMS,
			&item.ModelCalls, &item.SuccessfulModelCalls, &item.FailedModelCalls, &item.InputTokens, &item.OutputTokens,
			&item.ToolCalls, &item.SuccessfulToolCalls, &item.FailedToolCalls, &firstAt, &lastAt,
		); err != nil {
			return agent.SessionAudit{}, fmt.Errorf("scan session audit: %w", err)
		}
		item.TotalTokens = item.InputTokens + item.OutputTokens
		item.TerminalToolCalls = item.SuccessfulToolCalls + item.FailedToolCalls
		item.ToolSuccessRatePercent = successRate(item.SuccessfulToolCalls, item.TerminalToolCalls)
		audit.Runs = append(audit.Runs, item)
		audit.RunCount++
		switch item.Status {
		case agent.RunCompleted:
			audit.CompletedRuns++
		case agent.RunFailed:
			audit.FailedRuns++
		case agent.RunQueued, agent.RunRunning, agent.RunWaitingTool, agent.RunWaitingApproval, agent.RunWaitingInput, agent.RunWaitingExternal:
			audit.ActiveRuns++
		}
		audit.ExecutionDurationMS += item.DurationMS
		audit.ModelCalls += item.ModelCalls
		audit.SuccessfulModelCalls += item.SuccessfulModelCalls
		audit.FailedModelCalls += item.FailedModelCalls
		audit.InputTokens += item.InputTokens
		audit.OutputTokens += item.OutputTokens
		audit.ToolCalls += item.ToolCalls
		audit.SuccessfulToolCalls += item.SuccessfulToolCalls
		audit.FailedToolCalls += item.FailedToolCalls
		if audit.FirstStartedAt == nil || firstAt.Before(*audit.FirstStartedAt) {
			value := firstAt
			audit.FirstStartedAt = &value
		}
		if audit.LastActivityAt == nil || lastAt.After(*audit.LastActivityAt) {
			value := lastAt
			audit.LastActivityAt = &value
		}
	}
	if err := rows.Err(); err != nil {
		return agent.SessionAudit{}, fmt.Errorf("iterate session audit: %w", err)
	}
	audit.TotalTokens = audit.InputTokens + audit.OutputTokens
	audit.TerminalToolCalls = audit.SuccessfulToolCalls + audit.FailedToolCalls
	audit.ToolSuccessRatePercent = successRate(audit.SuccessfulToolCalls, audit.TerminalToolCalls)
	if audit.FirstStartedAt != nil && audit.LastActivityAt != nil && audit.LastActivityAt.After(*audit.FirstStartedAt) {
		audit.WallDurationMS = audit.LastActivityAt.Sub(*audit.FirstStartedAt).Milliseconds()
	}
	return audit, nil
}

func successRate(successful, terminal int64) float64 {
	if terminal <= 0 {
		return 0
	}
	return float64(successful) * 100 / float64(terminal)
}
