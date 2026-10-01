package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/heurry/cloudnative-infra-platform/server/internal/db/sqlcgen"
	"github.com/jackc/pgx/v5"
)

// Incidents 返回故障事件列表（status 为空则不过滤）。
func (s *Store) Incidents(ctx context.Context, status string) ([]sqlcgen.ListIncidentsRow, error) {
	if status != "" {
		byStatus, err := s.q.ListIncidentsByStatus(ctx, status)
		if err != nil {
			return nil, err
		}
		// ListIncidentsByStatusRow 与 ListIncidentsRow 同构，转换以统一返回类型。
		out := make([]sqlcgen.ListIncidentsRow, 0, len(byStatus))
		for _, r := range byStatus {
			out = append(out, sqlcgen.ListIncidentsRow(r))
		}
		return out, nil
	}
	return s.q.ListIncidents(ctx)
}

// IncidentDetail 返回事件 + 时间线。
func (s *Store) IncidentDetail(ctx context.Context, id string) (sqlcgen.GetIncidentRow, []sqlcgen.ListIncidentEventsRow, error) {
	inc, err := s.q.GetIncident(ctx, id)
	if err != nil {
		return sqlcgen.GetIncidentRow{}, nil, err
	}
	events, err := s.q.ListIncidentEvents(ctx, id)
	if err != nil {
		return sqlcgen.GetIncidentRow{}, nil, err
	}
	return inc, events, nil
}

// CreateIncident 新建（open）+ created 时间线事件（事务），返回新 id。
func (s *Store) CreateIncident(ctx context.Context, title, severity, summary string) (string, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return "", err
	}
	defer tx.Rollback(ctx)
	q := s.q.WithTx(tx)

	id, err := q.CreateIncident(ctx, sqlcgen.CreateIncidentParams{Title: title, Severity: severity, Summary: ptr(summary)})
	if err != nil {
		return "", err
	}
	payload, _ := json.Marshal(map[string]any{"severity": severity})
	if err := q.InsertIncidentEvent(ctx, sqlcgen.InsertIncidentEventParams{
		IncidentID: id, EventType: ptr("created"), Payload: payload,
	}); err != nil {
		return "", err
	}
	if err := tx.Commit(ctx); err != nil {
		return "", err
	}
	return id, nil
}

// AckIncident 受理 + acknowledged 时间线事件（事务）。
func (s *Store) AckIncident(ctx context.Context, id, operator string) error {
	return s.incidentTransition(ctx, id, "acknowledged", operator, func(q *sqlcgen.Queries) error {
		return q.AckIncident(ctx, id)
	})
}

// ResolveIncident 解决 + resolved 时间线事件（事务）。
func (s *Store) ResolveIncident(ctx context.Context, id, operator string) error {
	return s.incidentTransition(ctx, id, "resolved", operator, func(q *sqlcgen.Queries) error {
		return q.ResolveIncident(ctx, id)
	})
}

func (s *Store) incidentTransition(ctx context.Context, id, eventType, operator string, mutate func(*sqlcgen.Queries) error) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	q := s.q.WithTx(tx)

	if err := mutate(q); err != nil {
		return err
	}
	payload, _ := json.Marshal(map[string]any{"operator": operator})
	if err := q.InsertIncidentEvent(ctx, sqlcgen.InsertIncidentEventParams{
		IncidentID: id, EventType: ptr(eventType), Payload: payload,
	}); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

// LinkIncidentDiagnosis 把结构化诊断回链到 Incident，并记录可审计时间线事件。
func (s *Store) LinkIncidentDiagnosis(ctx context.Context, incidentID, diagnosisID string, metadata map[string]any) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	tag, err := tx.Exec(ctx, `UPDATE incidents SET diagnosis_id=$2::uuid WHERE id=$1::uuid`, incidentID, diagnosisID)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return pgx.ErrNoRows
	}
	metadata["diagnosis_id"] = diagnosisID
	payload, _ := json.Marshal(metadata)
	if _, err := tx.Exec(ctx, `
		INSERT INTO incident_events (incident_id, event_type, payload)
		VALUES ($1::uuid, 'diagnosis_linked', $2::jsonb)`, incidentID, payload); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

// EnsureWorkloadIncident 为同一训练或推理运行记录复用未解决 Incident，避免重复点击产生事件风暴。
func (s *Store) EnsureWorkloadIncident(
	ctx context.Context, resourceKey, resourceID, title, severity, rootCause, diagnosisID string,
) (string, bool, error) {
	if resourceKey == "" || resourceID == "" {
		return "", false, errors.New("workload resource key and id are required for incident")
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return "", false, err
	}
	defer tx.Rollback(ctx)

	marker := resourceKey + "=" + resourceID
	var incidentID string
	created := false
	err = tx.QueryRow(ctx, `
		SELECT id FROM incidents
		WHERE status<>'resolved' AND COALESCE(summary,'') LIKE '%' || $1 || '%'
		ORDER BY created_at DESC LIMIT 1`, marker).Scan(&incidentID)
	if errors.Is(err, pgx.ErrNoRows) {
		summary := fmt.Sprintf("%s; %s", marker, rootCause)
		err = tx.QueryRow(ctx, `
			INSERT INTO incidents (title, severity, status, summary, diagnosis_id)
			VALUES ($1, $2, 'open', $3, $4::uuid) RETURNING id`,
			title, severity, summary, diagnosisID).Scan(&incidentID)
		created = true
	}
	if err != nil {
		return "", false, err
	}
	if !created {
		summary := fmt.Sprintf("%s; %s", marker, rootCause)
		if _, err := tx.Exec(ctx, `
			UPDATE incidents SET diagnosis_id=$2::uuid, title=$3, severity=$4, summary=$5
			WHERE id=$1::uuid`, incidentID, diagnosisID, title, severity, summary); err != nil {
			return "", false, err
		}
	}
	payload, _ := json.Marshal(map[string]any{
		"diagnosis_id": diagnosisID, resourceKey: resourceID,
		"severity": severity, "created_incident": created,
	})
	if _, err := tx.Exec(ctx, `
		INSERT INTO incident_events (incident_id, event_type, payload)
		VALUES ($1::uuid, 'diagnosis_linked', $2::jsonb)`, incidentID, payload); err != nil {
		return "", false, err
	}
	if err := tx.Commit(ctx); err != nil {
		return "", false, err
	}
	return incidentID, created, nil
}
