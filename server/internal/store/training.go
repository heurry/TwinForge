package store

import (
	"context"
	"encoding/json"
	"time"
)

// Phase F / F2：分布式训练任务领域层（training_jobs 表，000012）。走 hand-pgx（与 knowledge/chat 一致）。
// 结构化列在此读写；runner 用裸 SQL 增量写 metadata（见 httpx/training_runner.go），故本层不含 metadata 写方法。

// TrainingJob 是一个分布式训练任务的控制面台账记录。
type TrainingJob struct {
	ID                string         `json:"id"`
	Name              string         `json:"name"`
	Framework         string         `json:"framework"`
	Namespace         string         `json:"namespace"`
	BaseModel         string         `json:"base_model"`
	DatasetURI        *string        `json:"dataset_uri"`
	Image             string         `json:"image"`
	Workers           int32          `json:"workers"`
	GPUsPerWorker     int32          `json:"gpus_per_worker"`
	Hyperparams       map[string]any `json:"hyperparams"`
	Status            string         `json:"status"`
	K8sJobRef         *string        `json:"k8s_job_ref"`
	OutputArtifactURI *string        `json:"output_artifact_uri"`
	ModelVersionID    *string        `json:"model_version_id"`
	Metadata          map[string]any `json:"metadata"`
	CreatedBy         *string        `json:"created_by"`
	CreatedAt         time.Time      `json:"created_at"`
	UpdatedAt         time.Time      `json:"updated_at"`
}

// CreateTrainingJobParams 是提交训练任务的入参（handler→store 边界）。
type CreateTrainingJobParams struct {
	Name          string
	Framework     string
	Namespace     string
	BaseModel     string
	DatasetURI    *string
	Image         string
	Workers       int32
	GPUsPerWorker int32
	Hyperparams   map[string]any
	K8sJobRef     *string
	CreatedBy     string
}

const trainingJobColumns = `id, name, framework, namespace, base_model, dataset_uri, image, workers, gpus_per_worker,
	hyperparams, status, k8s_job_ref, output_artifact_uri, model_version_id, metadata, created_by, created_at, updated_at`

func (s *Store) CreateTrainingJob(ctx context.Context, p CreateTrainingJobParams) (string, error) {
	hp, err := json.Marshal(p.Hyperparams)
	if err != nil || len(hp) == 0 {
		hp = []byte("{}")
	}
	var createdBy *string
	if p.CreatedBy != "" {
		createdBy = &p.CreatedBy
	}
	framework := p.Framework
	if framework == "" {
		framework = "pytorch"
	}
	var id string
	err = s.pool.QueryRow(ctx, `
		INSERT INTO training_jobs (name, framework, namespace, base_model, dataset_uri, image, workers, gpus_per_worker, hyperparams, status, k8s_job_ref, created_by)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,'pending',$10,$11)
		RETURNING id`,
		p.Name, framework, p.Namespace, p.BaseModel, p.DatasetURI, p.Image, p.Workers, p.GPUsPerWorker, hp, p.K8sJobRef, createdBy,
	).Scan(&id)
	return id, err
}

func (s *Store) ListTrainingJobs(ctx context.Context) ([]TrainingJob, error) {
	rows, err := s.pool.Query(ctx, `SELECT `+trainingJobColumns+` FROM training_jobs ORDER BY created_at DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []TrainingJob{}
	for rows.Next() {
		j, err := scanTrainingJob(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, j)
	}
	return out, rows.Err()
}

func (s *Store) GetTrainingJob(ctx context.Context, id string) (TrainingJob, error) {
	return scanTrainingJob(s.pool.QueryRow(ctx, `SELECT `+trainingJobColumns+` FROM training_jobs WHERE id = $1::uuid`, id))
}

func (s *Store) UpdateTrainingJobStatus(ctx context.Context, id, status string) error {
	_, err := s.pool.Exec(ctx, `UPDATE training_jobs SET status = $2, updated_at = now() WHERE id = $1::uuid`, id, status)
	return err
}

// SetTrainingJobOutput 回填成功产物与注册版本（F3：训练→注册闭环）。空串存 NULL。
func (s *Store) SetTrainingJobOutput(ctx context.Context, id, artifactURI, modelVersionID string) error {
	var art, mv *string
	if artifactURI != "" {
		art = &artifactURI
	}
	if modelVersionID != "" {
		mv = &modelVersionID
	}
	_, err := s.pool.Exec(ctx,
		`UPDATE training_jobs SET output_artifact_uri = $2, model_version_id = $3, updated_at = now() WHERE id = $1::uuid`,
		id, art, mv)
	return err
}

// trainingRowScanner 抽象 pgx.Row / pgx.Rows 的 Scan（Get/List 共用）。
type trainingRowScanner interface{ Scan(dest ...any) error }

func scanTrainingJob(row trainingRowScanner) (TrainingJob, error) {
	var j TrainingJob
	var hp, meta []byte
	if err := row.Scan(&j.ID, &j.Name, &j.Framework, &j.Namespace, &j.BaseModel, &j.DatasetURI, &j.Image,
		&j.Workers, &j.GPUsPerWorker, &hp, &j.Status, &j.K8sJobRef, &j.OutputArtifactURI, &j.ModelVersionID,
		&meta, &j.CreatedBy, &j.CreatedAt, &j.UpdatedAt); err != nil {
		return TrainingJob{}, err
	}
	j.Hyperparams = jsonToMap(hp)
	j.Metadata = jsonToMap(meta)
	return j, nil
}

func jsonToMap(b []byte) map[string]any {
	m := map[string]any{}
	if len(b) > 0 {
		_ = json.Unmarshal(b, &m)
	}
	return m
}
