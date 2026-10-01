package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/score"
)

func (s *RunStore) CreateScore(ctx context.Context, input score.Create) (score.Score, error) {
	if err := input.Validate(); err != nil {
		return score.Score{}, err
	}
	metadata, err := json.Marshal(input.Metadata)
	if err != nil {
		return score.Score{}, fmt.Errorf("encode score metadata: %w", err)
	}
	var result score.Score
	err = s.pool.QueryRow(ctx, `INSERT INTO agent_platform.agent_scores
		(tenant_id,run_id,observation_id,session_id,dataset_run_id,name,score_type,value,string_value,source,evaluator_version,agent_version_id,model_resolution_id,prompt_version_id,toolset_version_id,skillset_version_id,metadata,created_by)
		VALUES($1,NULLIF($2,'')::uuid,NULLIF($3,'')::uuid,NULLIF($4,'')::uuid,NULLIF($5,''),$6,$7,$8,$9,$10,NULLIF($11,''),NULLIF($12,''),NULLIF($13,''),NULLIF($14,''),NULLIF($15,''),NULLIF($16,''),$17::jsonb,NULLIF($18,''))
		RETURNING id::text,tenant_id,COALESCE(run_id::text,''),COALESCE(observation_id::text,''),COALESCE(session_id::text,''),COALESCE(dataset_run_id,''),name,score_type,value,string_value,source,COALESCE(evaluator_version,''),COALESCE(agent_version_id,''),COALESCE(model_resolution_id,''),COALESCE(prompt_version_id,''),COALESCE(toolset_version_id,''),COALESCE(skillset_version_id,''),metadata,COALESCE(created_by,''),created_at`,
		input.TenantID, input.RunID, input.ObservationID, input.SessionID, input.DatasetRunID, input.Name, input.ScoreType, input.Value, input.StringValue, input.Source, input.EvaluatorVersion, input.AgentVersionID, input.ModelResolutionID, input.PromptVersionID, input.ToolSetVersionID, input.SkillSetVersionID, metadata, input.CreatedBy).Scan(
		&result.ID, &result.TenantID, &result.RunID, &result.ObservationID, &result.SessionID, &result.DatasetRunID, &result.Name, &result.ScoreType, &result.Value, &result.StringValue, &result.Source, &result.EvaluatorVersion, &result.AgentVersionID, &result.ModelResolutionID, &result.PromptVersionID, &result.ToolSetVersionID, &result.SkillSetVersionID, &result.Metadata, &result.CreatedBy, &result.CreatedAt)
	if err != nil {
		return score.Score{}, fmt.Errorf("create Agent score: %w", err)
	}
	return result, nil
}

func (s *RunStore) ListScoresForTenant(ctx context.Context, tenantID, runID string, limit int) ([]score.Score, error) {
	if strings.TrimSpace(tenantID) == "" || strings.TrimSpace(runID) == "" {
		return nil, errors.New("tenant_id and run_id are required")
	}
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `SELECT id::text,tenant_id,COALESCE(run_id::text,''),COALESCE(observation_id::text,''),COALESCE(session_id::text,''),COALESCE(dataset_run_id,''),name,score_type,value,string_value,source,COALESCE(evaluator_version,''),COALESCE(agent_version_id,''),COALESCE(model_resolution_id,''),COALESCE(prompt_version_id,''),COALESCE(toolset_version_id,''),COALESCE(skillset_version_id,''),metadata,COALESCE(created_by,''),created_at FROM agent_platform.agent_scores WHERE tenant_id=$1 AND run_id=$2::uuid ORDER BY created_at DESC,id LIMIT $3`, tenantID, runID, limit)
	if err != nil {
		return nil, fmt.Errorf("list Agent scores: %w", err)
	}
	defer rows.Close()
	result := make([]score.Score, 0)
	for rows.Next() {
		var item score.Score
		if err := rows.Scan(&item.ID, &item.TenantID, &item.RunID, &item.ObservationID, &item.SessionID, &item.DatasetRunID, &item.Name, &item.ScoreType, &item.Value, &item.StringValue, &item.Source, &item.EvaluatorVersion, &item.AgentVersionID, &item.ModelResolutionID, &item.PromptVersionID, &item.ToolSetVersionID, &item.SkillSetVersionID, &item.Metadata, &item.CreatedBy, &item.CreatedAt); err != nil {
			return nil, err
		}
		result = append(result, item)
	}
	return result, rows.Err()
}
