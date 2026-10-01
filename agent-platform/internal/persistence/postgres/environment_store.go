package postgres

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/provider/dependency"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/environment"
)

func (s *RunStore) ListEnvironmentTemplates(ctx context.Context) ([]environment.Template, error) {
	rows, err := s.pool.Query(ctx, `SELECT id::text,template_key,version,name,runtime,image_ref,spec_digest,dependencies,capabilities,status,created_at FROM agent_platform.agent_environment_templates WHERE status<>'retired' ORDER BY template_key,version DESC`)
	if err != nil {
		return nil, fmt.Errorf("list environment templates: %w", err)
	}
	defer rows.Close()
	items := make([]environment.Template, 0)
	for rows.Next() {
		var item environment.Template
		if err := rows.Scan(&item.ID, &item.Key, &item.Version, &item.Name, &item.Runtime, &item.ImageRef, &item.SpecDigest, &item.Dependencies, &item.Capabilities, &item.Status, &item.CreatedAt); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

func (s *RunStore) RecordDependencyInstall(ctx context.Context, tenantID, runID, callID, template string, request dependency.Request, result dependency.Result, installErr error) error {
	packages, err := json.Marshal(request.Packages)
	if err != nil {
		return err
	}
	resultJSON, err := json.Marshal(result)
	if err != nil {
		return err
	}
	status := "installed"
	var errorText *string
	if installErr != nil {
		status = "failed"
		value := installErr.Error()
		errorText = &value
	}
	var finishedAt *time.Time
	now := time.Now().UTC()
	finishedAt = &now
	_, err = s.pool.Exec(ctx, `INSERT INTO agent_platform.agent_dependency_installs(tenant_id,run_id,call_id,environment_template,ecosystem,packages,source,scope,status,result,error,finished_at) VALUES($1,$2::uuid,$3,$4,$5,$6::jsonb,$7,$8,$9,$10::jsonb,$11,$12) ON CONFLICT(tenant_id,run_id,call_id) DO UPDATE SET status=EXCLUDED.status,result=EXCLUDED.result,error=EXCLUDED.error,finished_at=EXCLUDED.finished_at`, tenantID, runID, callID, template, request.Ecosystem, packages, request.Source, request.Scope, status, resultJSON, errorText, finishedAt)
	if err != nil {
		return fmt.Errorf("record dependency install: %w", err)
	}
	return nil
}

func (s *RunStore) ListDependencyInstalls(ctx context.Context, tenantID, runID string, limit int) ([]environment.Install, error) {
	if limit <= 0 || limit > 200 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `SELECT id::text,tenant_id,run_id::text,call_id,environment_template,ecosystem,packages,source,scope,status,result,error,created_at,finished_at FROM agent_platform.agent_dependency_installs WHERE tenant_id=$1 AND ($2='' OR run_id=$2::uuid) ORDER BY created_at DESC LIMIT $3`, tenantID, runID, limit)
	if err != nil {
		return nil, fmt.Errorf("list dependency installs: %w", err)
	}
	defer rows.Close()
	items := make([]environment.Install, 0)
	for rows.Next() {
		var item environment.Install
		if err := rows.Scan(&item.ID, &item.TenantID, &item.RunID, &item.CallID, &item.EnvironmentTemplate, &item.Ecosystem, &item.Packages, &item.Source, &item.Scope, &item.Status, &item.Result, &item.Error, &item.CreatedAt, &item.FinishedAt); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}
