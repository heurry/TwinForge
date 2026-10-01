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

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/artifact"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/taskplan"
)

func (s *RunStore) ListArtifactsForTenant(ctx context.Context, tenantID, runID string, limit int) ([]artifact.Artifact, error) {
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `
		SELECT id::text, tenant_id, run_id::text, workflow_id::text, COALESCE(call_id,''), kind, name,
			media_type, content_hash, size_bytes, storage_backend, storage_status, metadata, created_at
		FROM agent_platform.agent_artifacts
		WHERE tenant_id=$1::text AND run_id=$2::uuid ORDER BY created_at, id LIMIT $3`, tenantID, runID, limit)
	if err != nil {
		return nil, fmt.Errorf("list run artifacts: %w", err)
	}
	defer rows.Close()
	var result []artifact.Artifact
	for rows.Next() {
		var item artifact.Artifact
		if err := rows.Scan(&item.ID, &item.TenantID, &item.RunID, &item.WorkflowID, &item.CallID, &item.Kind, &item.Name, &item.MediaType, &item.ContentHash, &item.SizeBytes, &item.StorageBackend, &item.StorageStatus, &item.Metadata, &item.CreatedAt); err != nil {
			return nil, err
		}
		result = append(result, item)
	}
	return result, rows.Err()
}

// GetRunManifestForTenant returns the canonical delivery projection for one
// Run. It is intentionally keyed by run_id; Workflow history is exposed by a
// separate endpoint and must never be substituted implicitly.
func (s *RunStore) GetRunManifestForTenant(ctx context.Context, tenantID, runID string) (artifact.RunManifest, error) {
	var item artifact.RunManifest
	var canonical, required, verification, children, finalArtifacts json.RawMessage
	err := s.pool.QueryRow(ctx, `
		SELECT run_id::text,tenant_id,workflow_id::text,status,canonical_artifacts,
			required_outputs,verification_summary,child_runs,final_artifacts,COALESCE(final_output_hash,''),
			final_output_present,updated_at
		FROM agent_platform.agent_run_manifests
		WHERE tenant_id=$1::text AND run_id=$2::uuid`, tenantID, runID).Scan(
		&item.RunID, &item.TenantID, &item.WorkflowID, &item.Status, &canonical,
		&required, &verification, &children, &finalArtifacts, &item.FinalOutputHash, &item.FinalOutputPresent, &item.UpdatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return artifact.RunManifest{}, errors.New("run manifest not found")
	}
	if err != nil {
		return artifact.RunManifest{}, fmt.Errorf("get Run manifest: %w", err)
	}
	if err := json.Unmarshal(canonical, &item.CanonicalArtifacts); err != nil {
		return artifact.RunManifest{}, fmt.Errorf("decode canonical artifacts: %w", err)
	}
	if err := json.Unmarshal(required, &item.RequiredOutputs); err != nil {
		return artifact.RunManifest{}, fmt.Errorf("decode required outputs: %w", err)
	}
	item.VerificationSummary = append(json.RawMessage(nil), verification...)
	if err := json.Unmarshal(children, &item.ChildRuns); err != nil {
		return artifact.RunManifest{}, fmt.Errorf("decode child runs: %w", err)
	}
	if err := json.Unmarshal(finalArtifacts, &item.FinalArtifactIDs); err != nil {
		return artifact.RunManifest{}, fmt.Errorf("decode final artifacts: %w", err)
	}
	// JSONB defaults to arrays for new rows, but old or externally repaired
	// rows may still contain null. Keep the HTTP contract stable so clients can
	// safely use length/map without carrying database null semantics.
	if item.CanonicalArtifacts == nil {
		item.CanonicalArtifacts = []artifact.ManifestArtifact{}
	}
	if item.RequiredOutputs == nil {
		item.RequiredOutputs = []string{}
	}
	if item.ChildRuns == nil {
		item.ChildRuns = []string{}
	}
	if item.FinalArtifactIDs == nil {
		item.FinalArtifactIDs = []string{}
	}
	return item, nil
}

func (s *RunStore) RebuildRunManifest(ctx context.Context, tenantID, runID string) error {
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return fmt.Errorf("begin Run manifest rebuild: %w", err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	if err := s.rebuildRunManifestTx(ctx, tx, tenantID, runID); err != nil {
		return err
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit Run manifest rebuild: %w", err)
	}
	return nil
}

func (s *RunStore) rebuildRunManifestTx(ctx context.Context, tx pgx.Tx, tenantID, runID string) error {
	var workflowID, status string
	var output json.RawMessage
	if err := tx.QueryRow(ctx, `SELECT workflow_id::text,status,COALESCE(output,'null'::jsonb) FROM agent_platform.agent_runs WHERE id=$1::uuid AND tenant_id=$2::text`, runID, tenantID).Scan(&workflowID, &status, &output); err != nil {
		return fmt.Errorf("resolve Run for manifest: %w", err)
	}
	canonicalQuery := `COALESCE((SELECT jsonb_agg(to_jsonb(latest) ORDER BY latest.name) FROM (
		SELECT DISTINCT ON (name) id::text AS artifact_id,name,kind,media_type,content_hash,size_bytes,metadata,created_at,true AS canonical,COALESCE(metadata->>'phase','intermediate') AS phase,(COALESCE(metadata->>'phase','')='final' OR (kind='workspace_file' AND EXISTS (SELECT 1 FROM agent_platform.agent_runs completed_run WHERE completed_run.id=$1::uuid AND completed_run.status='completed'))) AS final
		FROM agent_platform.agent_artifacts
		WHERE tenant_id=$2::text AND run_id=$1::uuid
		ORDER BY name,created_at DESC,id DESC
	) latest),'[]'::jsonb)`
	var childRuns json.RawMessage
	if err := tx.QueryRow(ctx, `SELECT COALESCE(jsonb_agg(child.id::text ORDER BY child.created_at),'[]'::jsonb) FROM agent_platform.agent_runs child WHERE child.tenant_id=$1::text AND child.parent_run_id=$2::uuid`, tenantID, runID).Scan(&childRuns); err != nil {
		return fmt.Errorf("resolve child runs for manifest: %w", err)
	}
	var finalArtifacts json.RawMessage
	if err := tx.QueryRow(ctx, `SELECT COALESCE(jsonb_agg(id::text ORDER BY created_at,id),'[]'::jsonb) FROM agent_platform.agent_artifacts WHERE tenant_id=$1::text AND run_id=$2::uuid AND (COALESCE(metadata->>'phase','')='final' OR (EXISTS (SELECT 1 FROM agent_platform.agent_runs completed_run WHERE completed_run.id=$2::uuid AND completed_run.status='completed') AND kind='workspace_file'))`, tenantID, runID).Scan(&finalArtifacts); err != nil {
		return fmt.Errorf("resolve final artifacts for manifest: %w", err)
	}
	// Compile a bounded, replayable verification projection from the durable
	// Plan. The model's final prose is deliberately absent: statuses, targets
	// and evidence call IDs are facts owned by the Plan projector.
	requiredOutputs := make([]string, 0)
	verificationSummary := map[string]any{"status": status, "plan_revision": 0, "steps": []any{}}
	var planRevision int
	var rawSteps json.RawMessage
	planErr := tx.QueryRow(ctx, `SELECT revision,steps FROM agent_platform.agent_task_plans WHERE workflow_id=$1::uuid AND tenant_id=$2::text ORDER BY revision DESC LIMIT 1`, workflowID, tenantID).Scan(&planRevision, &rawSteps)
	if planErr == nil {
		verificationSummary["plan_revision"] = planRevision
		var steps []taskplan.Step
		if err := json.Unmarshal(rawSteps, &steps); err == nil {
			projectedSteps := make([]any, 0, len(steps))
			seenOutputs := make(map[string]struct{})
			for _, step := range steps {
				projectedCriteria := make([]any, 0, len(step.AcceptanceCriteria))
				for _, criterion := range step.AcceptanceCriteria {
					projectedCriteria = append(projectedCriteria, map[string]any{
						"id": criterion.ID, "status": criterion.Status, "enforcement": criterion.Enforcement,
						"verification_kind": criterion.Verification.Kind, "target": criterion.Verification.Target,
						"verification":      criterion.Verification,
						"evidence_call_ids": criterion.EvidenceCallIDs,
					})
					if criterion.BlocksCompletion() && strings.TrimSpace(criterion.Verification.Target) != "" {
						output := strings.TrimSpace(criterion.Verification.Target)
						if _, seen := seenOutputs[output]; !seen {
							seenOutputs[output] = struct{}{}
							requiredOutputs = append(requiredOutputs, output)
						}
					}
				}
				projectedSteps = append(projectedSteps, map[string]any{
					"id": step.ID, "status": step.Status, "state": step.State, "criteria": projectedCriteria,
				})
			}
			verificationSummary["steps"] = projectedSteps
		}
	}
	requiredJSON, err := json.Marshal(requiredOutputs)
	if err != nil {
		return fmt.Errorf("encode manifest required outputs: %w", err)
	}
	verificationJSON, err := json.Marshal(verificationSummary)
	if err != nil {
		return fmt.Errorf("encode manifest verification summary: %w", err)
	}
	finalHash := ""
	finalPresent := len(output) != 0 && string(output) != "null" && string(output) != `""`
	if finalPresent {
		finalHash = sha256Hex(output)
	}
	if _, err := tx.Exec(ctx, `
		INSERT INTO agent_platform.agent_run_manifests
			(run_id,tenant_id,workflow_id,status,canonical_artifacts,required_outputs,verification_summary,child_runs,final_artifacts,final_output_hash,final_output_present,updated_at)
		VALUES($1::uuid,$2::text,$3::uuid,$4::text,`+canonicalQuery+`,$5::jsonb,$6::jsonb,$7::jsonb,$8::jsonb,NULLIF($9::text,''),$10,now())
		ON CONFLICT(run_id) DO UPDATE SET tenant_id=EXCLUDED.tenant_id,workflow_id=EXCLUDED.workflow_id,status=EXCLUDED.status,
			canonical_artifacts=EXCLUDED.canonical_artifacts,required_outputs=EXCLUDED.required_outputs,verification_summary=EXCLUDED.verification_summary,child_runs=EXCLUDED.child_runs,final_artifacts=EXCLUDED.final_artifacts,
			final_output_hash=EXCLUDED.final_output_hash,final_output_present=EXCLUDED.final_output_present,updated_at=now()`,
		runID, tenantID, workflowID, status, requiredJSON, verificationJSON, childRuns, finalArtifacts, finalHash, finalPresent); err != nil {
		return fmt.Errorf("upsert Run manifest: %w", err)
	}
	return nil
}

// ListWorkflowArtifactsForTenant returns the complete artifact history of a
// Workflow, including artifacts produced by previous Run attempts.  The
// run_id is retained on each item so callers can still show provenance.
func (s *RunStore) ListWorkflowArtifactsForTenant(ctx context.Context, tenantID, workflowID string, limit int) ([]artifact.Artifact, error) {
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `
		SELECT id::text, tenant_id, run_id::text, workflow_id::text, COALESCE(call_id,''), kind, name,
			media_type, content_hash, size_bytes, storage_backend, storage_status, metadata, created_at
		FROM agent_platform.agent_artifacts
		WHERE tenant_id=$1::text AND workflow_id=$2::uuid
		ORDER BY created_at, id LIMIT $3`, tenantID, workflowID, limit)
	if err != nil {
		return nil, fmt.Errorf("list workflow artifacts: %w", err)
	}
	defer rows.Close()
	var result []artifact.Artifact
	for rows.Next() {
		var item artifact.Artifact
		if err := rows.Scan(&item.ID, &item.TenantID, &item.RunID, &item.WorkflowID, &item.CallID, &item.Kind, &item.Name, &item.MediaType, &item.ContentHash, &item.SizeBytes, &item.StorageBackend, &item.StorageStatus, &item.Metadata, &item.CreatedAt); err != nil {
			return nil, err
		}
		result = append(result, item)
	}
	return result, rows.Err()
}

func (s *RunStore) BeginArtifactPromotion(ctx context.Context, tenantID, artifactID string, request artifact.PromotionRequest, actor string) (artifact.Promotion, error) {
	target := strings.TrimSpace(request.TargetPath)
	if target == "" {
		return artifact.Promotion{}, errors.New("target_path is required")
	}
	var result artifact.Promotion
	err := s.pool.QueryRow(ctx, `
		INSERT INTO agent_platform.agent_artifact_promotions
			(tenant_id,run_id,artifact_id,target_path,source_hash,expected_target_sha256,requested_by)
		SELECT $1::text,a.run_id,a.id,$3::text,a.content_hash,NULLIF($4::text,''),$5::text
		FROM agent_platform.agent_artifacts AS a
		JOIN agent_platform.agent_runs AS r ON r.id=a.run_id
		WHERE a.id=$2::uuid AND a.tenant_id=$1::text AND a.kind='workspace_file' AND r.status='completed'
		  AND NOT EXISTS (
			SELECT 1 FROM agent_platform.agent_artifacts newer
			WHERE newer.run_id=a.run_id AND newer.tenant_id=a.tenant_id
			  AND newer.kind='workspace_file' AND newer.name=a.name AND newer.created_at>a.created_at
		  )
		RETURNING id::text,tenant_id,run_id::text,artifact_id::text,target_path,source_hash,
			COALESCE(expected_target_sha256,''),status,requested_by,created_at`,
		tenantID, artifactID, target, request.ExpectedTargetSHA256, actor).Scan(
		&result.ID, &result.TenantID, &result.RunID, &result.ArtifactID, &result.TargetPath,
		&result.SourceHash, &result.ExpectedTargetSHA256, &result.Status, &result.RequestedBy, &result.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return artifact.Promotion{}, errors.New("only the latest workspace_file artifact of a completed Run can be promoted")
	}
	if err != nil {
		return artifact.Promotion{}, fmt.Errorf("begin artifact promotion: %w", err)
	}
	return result, nil
}

func (s *RunStore) FinishArtifactPromotion(ctx context.Context, tenantID, promotionID string, outcome artifact.PromotionResult, promoteErr error) (artifact.Promotion, error) {
	status := "promoted"
	errText := ""
	if promoteErr != nil {
		status = "failed"
		errText = promoteErr.Error()
	}
	var result artifact.Promotion
	err := s.pool.QueryRow(ctx, `
		UPDATE agent_platform.agent_artifact_promotions
		SET status=$3::text,previous_target_sha256=NULLIF($4::text,''),result_target_sha256=NULLIF($5::text,''),
			error=NULLIF($6::text,''),finished_at=now()
		WHERE id=$1::uuid AND tenant_id=$2::text AND status='requested'
		RETURNING id::text,tenant_id,run_id::text,artifact_id::text,target_path,source_hash,
			COALESCE(expected_target_sha256,''),COALESCE(previous_target_sha256,''),
			COALESCE(result_target_sha256,''),status,requested_by,COALESCE(error,''),created_at,finished_at`,
		promotionID, tenantID, status, outcome.PreviousTargetSHA256, outcome.ResultTargetSHA256, errText).Scan(
		&result.ID, &result.TenantID, &result.RunID, &result.ArtifactID, &result.TargetPath,
		&result.SourceHash, &result.ExpectedTargetSHA256, &result.PreviousTargetSHA256,
		&result.ResultTargetSHA256, &result.Status, &result.RequestedBy, &result.Error,
		&result.CreatedAt, &result.FinishedAt)
	if err != nil {
		return artifact.Promotion{}, fmt.Errorf("finish artifact promotion: %w", err)
	}
	return result, nil
}

func (s *RunStore) GetArtifactContentForTenant(ctx context.Context, tenantID, artifactID string) (artifact.Artifact, []byte, error) {
	var item artifact.Artifact
	var content []byte
	var objectKey *string
	err := s.pool.QueryRow(ctx, `
		SELECT id::text, tenant_id, run_id::text, workflow_id::text, COALESCE(call_id,''), kind, name,
			media_type, content_hash, size_bytes, storage_backend, storage_status,
			metadata, created_at, content, object_key
		FROM agent_platform.agent_artifacts WHERE id=$1::uuid AND tenant_id=$2::text`, artifactID, tenantID).Scan(
		&item.ID, &item.TenantID, &item.RunID, &item.WorkflowID, &item.CallID, &item.Kind, &item.Name, &item.MediaType,
		&item.ContentHash, &item.SizeBytes, &item.StorageBackend, &item.StorageStatus,
		&item.Metadata, &item.CreatedAt, &content, &objectKey)
	if errors.Is(err, pgx.ErrNoRows) {
		return artifact.Artifact{}, nil, errors.New("artifact not found")
	}
	if err != nil {
		return artifact.Artifact{}, nil, fmt.Errorf("get artifact: %w", err)
	}
	if item.StorageBackend == "minio" {
		if s.artifactStore == nil || !s.artifactStore.Enabled() || objectKey == nil {
			return artifact.Artifact{}, nil, errors.New("artifact object storage is unavailable")
		}
		content, err = s.artifactStore.Get(ctx, *objectKey)
		if err != nil {
			return artifact.Artifact{}, nil, fmt.Errorf("read artifact object: %w", err)
		}
	}
	if int64(len(content)) != item.SizeBytes {
		return artifact.Artifact{}, nil, fmt.Errorf("artifact integrity check failed: size=%d, expected=%d", len(content), item.SizeBytes)
	}
	digest := sha256.Sum256(content)
	if hex.EncodeToString(digest[:]) != item.ContentHash {
		return artifact.Artifact{}, nil, errors.New("artifact integrity check failed: sha256 mismatch")
	}
	return item, content, nil
}

type artifactWrite struct {
	Kind      string
	Name      string
	MediaType string
	Content   []byte
	Metadata  map[string]string
}

func (s *RunStore) persistArtifactTx(ctx context.Context, tx pgx.Tx, tenantID, runID, callID string, input artifactWrite) (string, error) {
	if len(input.Content) > 10<<20 {
		return "", fmt.Errorf("artifact %q exceeds 10 MiB", input.Name)
	}
	metadata, err := json.Marshal(input.Metadata)
	if err != nil {
		return "", fmt.Errorf("encode artifact metadata: %w", err)
	}
	digest := sha256.Sum256(input.Content)
	hash := hex.EncodeToString(digest[:])
	var workflowID string
	if err := tx.QueryRow(ctx, `
		SELECT workflow_id::text FROM agent_platform.agent_runs
		WHERE id=$1::uuid AND tenant_id=$2::text`, runID, tenantID).Scan(&workflowID); err != nil {
		return "", fmt.Errorf("resolve artifact workflow: %w", err)
	}
	backend := "inline"
	status := "ready"
	var objectKey *string
	var inline any = input.Content
	if s.artifactStore != nil && s.artifactStore.Enabled() {
		key := "sha256/" + hash[:2] + "/" + hash
		if err := s.artifactStore.Put(ctx, key, input.Content, input.MediaType); err != nil {
			return "", fmt.Errorf("store artifact object: %w", err)
		}
		backend, objectKey, inline = "minio", &key, nil
	}
	var id string
	err = tx.QueryRow(ctx, `
		INSERT INTO agent_platform.agent_artifacts
			(tenant_id,run_id,workflow_id,call_id,kind,name,media_type,content,content_hash,size_bytes,metadata,storage_backend,object_key,storage_status)
		VALUES($1::text,$2::uuid,$3::uuid,NULLIF($4::text,''),$5::text,$6::text,$7::text,$8::bytea,$9::text,$10,$11::jsonb,$12::text,$13::text,$14::text)
		RETURNING id::text`, tenantID, runID, workflowID, callID, input.Kind, input.Name, input.MediaType,
		inline, hash, len(input.Content), metadata, backend, objectKey, status).Scan(&id)
	if err != nil {
		return "", fmt.Errorf("persist artifact metadata: %w", err)
	}
	if err := s.rebuildRunManifestTx(ctx, tx, tenantID, runID); err != nil {
		return "", err
	}
	return id, nil
}

type ArtifactReconcileResult struct {
	Scanned  int `json:"scanned"`
	Migrated int `json:"migrated"`
}

// ReconcileArtifactObjects incrementally drains legacy bytea payloads into
// content-addressed objects. Conditional updates make retries and concurrent
// reconcilers safe; an interrupted upload can only leave a reclaimable orphan.
func (s *RunStore) ReconcileArtifactObjects(ctx context.Context, limit int) (ArtifactReconcileResult, error) {
	if s.artifactStore == nil || !s.artifactStore.Enabled() {
		return ArtifactReconcileResult{}, nil
	}
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `SELECT id::text,content,content_hash,media_type FROM agent_platform.agent_artifacts WHERE storage_backend='inline' ORDER BY created_at,id LIMIT $1`, limit)
	if err != nil {
		return ArtifactReconcileResult{}, fmt.Errorf("list inline artifacts: %w", err)
	}
	type legacy struct {
		id              string
		content         []byte
		hash, mediaType string
	}
	var items []legacy
	for rows.Next() {
		var item legacy
		if err := rows.Scan(&item.id, &item.content, &item.hash, &item.mediaType); err != nil {
			rows.Close()
			return ArtifactReconcileResult{}, err
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return ArtifactReconcileResult{}, err
	}
	rows.Close()
	result := ArtifactReconcileResult{Scanned: len(items)}
	for _, item := range items {
		if len(item.hash) != 64 {
			return result, fmt.Errorf("legacy artifact %s has invalid sha256 metadata", item.id)
		}
		digest := sha256.Sum256(item.content)
		if hex.EncodeToString(digest[:]) != item.hash {
			return result, fmt.Errorf("legacy artifact %s sha256 mismatch", item.id)
		}
		key := "sha256/" + item.hash[:2] + "/" + item.hash
		if err := s.artifactStore.Put(ctx, key, item.content, item.mediaType); err != nil {
			return result, err
		}
		tag, err := s.pool.Exec(ctx, `UPDATE agent_platform.agent_artifacts SET storage_backend='minio',object_key=$2,content=NULL,storage_status='ready',migrated_at=$3 WHERE id=$1::uuid AND storage_backend='inline' AND content_hash=$4`, item.id, key, time.Now().UTC(), item.hash)
		if err != nil {
			return result, err
		}
		result.Migrated += int(tag.RowsAffected())
	}
	return result, nil
}
