package postgres

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
)

// LoadEventForObservationArchive reads the authoritative event after the
// Redis queue has delivered only its run/sequence cursor.
func (s *RunStore) LoadEventForObservationArchive(ctx context.Context, runID string, sequence int64) (string, event.Event, error) {
	var tenant string
	var committed event.Event
	err := s.pool.QueryRow(ctx, `
		SELECT tenant_id,run_id::text,event_type,schema_version,COALESCE(turn_no,0),
			COALESCE(step_no,0),COALESCE(call_id,''),payload,seq,created_at
		FROM agent_platform.agent_events
		WHERE run_id=$1::uuid AND seq=$2`, runID, sequence).Scan(
		&tenant, &committed.RunID, &committed.Type, &committed.SchemaVersion,
		&committed.Turn, &committed.Step, &committed.CallID, &committed.Payload,
		&committed.Sequence, &committed.CreatedAt)
	if err != nil {
		return "", event.Event{}, fmt.Errorf("load observation event: %w", err)
	}
	return tenant, committed, nil
}

func (s *RunStore) RecordObservationArchive(ctx context.Context, tenantID string, committed event.Event, objectKey, contentHash string, size int64) error {
	if size < 0 {
		size = 0
	}
	_, err := s.pool.Exec(ctx, `
		INSERT INTO agent_platform.agent_observation_archives
			(tenant_id,run_id,sequence,event_type,object_key,content_hash,size_bytes,status)
		VALUES($1,$2::uuid,$3,$4,$5,$6,$7,'ready')
		ON CONFLICT(run_id,sequence) DO UPDATE SET
			object_key=EXCLUDED.object_key,content_hash=EXCLUDED.content_hash,
			size_bytes=EXCLUDED.size_bytes,status='ready'`,
		tenantID, committed.RunID, committed.Sequence, committed.Type, objectKey, contentHash, size)
	if err != nil {
		return fmt.Errorf("record observation archive: %w", err)
	}
	return nil
}

// MarshalObservationEvent is kept here so all archive producers use the same
// stable envelope and content hash input.
func MarshalObservationEvent(committed event.Event) ([]byte, error) {
	return json.Marshal(struct {
		RunID         string          `json:"run_id"`
		Type          event.Type      `json:"type"`
		SchemaVersion int             `json:"schema_version"`
		Turn          int             `json:"turn,omitempty"`
		Step          int             `json:"step,omitempty"`
		CallID        string          `json:"call_id,omitempty"`
		Payload       json.RawMessage `json:"payload"`
		Sequence      int64           `json:"sequence"`
		CreatedAt     time.Time       `json:"created_at"`
	}{committed.RunID, committed.Type, committed.SchemaVersion, committed.Turn, committed.Step, committed.CallID, committed.Payload, committed.Sequence, committed.CreatedAt})
}
