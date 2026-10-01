package observability

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/objectstore"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/realtime"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
)

type ObservationArchiveStore interface {
	LoadEventForObservationArchive(context.Context, string, int64) (string, event.Event, error)
	RecordObservationArchive(context.Context, string, event.Event, string, string, int64) error
}

// ObservationArchiver drains Redis cursors, re-reads event facts from PG and
// writes immutable raw envelopes to MinIO. Failures leave messages pending for
// retry; acknowledgement happens only after both object and PG metadata exist.
type ObservationArchiver struct {
	Queue    *realtime.RedisBus
	Store    ObservationArchiveStore
	Objects  *objectstore.Client
	Consumer string
	Poll     time.Duration
}

func (a *ObservationArchiver) Run(ctx context.Context) error {
	if a == nil || a.Queue == nil || a.Store == nil || a.Objects == nil || !a.Objects.Enabled() {
		return nil
	}
	if a.Consumer == "" {
		a.Consumer = "agent-worker"
	}
	if a.Poll <= 0 {
		a.Poll = 2 * time.Second
	}
	if err := a.Queue.EnsureObservationQueue(ctx); err != nil {
		return fmt.Errorf("ensure observation queue: %w", err)
	}
	ticker := time.NewTicker(a.Poll)
	defer ticker.Stop()
	for {
		if err := a.drain(ctx); err != nil && !errors.Is(err, context.Canceled) {
			slog.Warn("Agent observation archive degraded", "err", err)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

func (a *ObservationArchiver) drain(ctx context.Context) error {
	items, err := a.Queue.ReclaimObservationQueue(ctx, a.Consumer, 50, time.Minute)
	if err != nil {
		return err
	}
	if len(items) == 0 {
		items, err = a.Queue.ReadObservationQueue(ctx, a.Consumer, 50, 750*time.Millisecond)
		if err != nil {
			return err
		}
	}
	for _, item := range items {
		if item.RunID == "" || item.Sequence <= 0 {
			_ = a.Queue.AckObservationQueue(ctx, item.ID)
			continue
		}
		if err := a.archiveOne(ctx, item); err != nil {
			return err
		}
	}
	return nil
}

func (a *ObservationArchiver) archiveOne(ctx context.Context, item realtime.ObservationQueueItem) error {
	tenant, committed, err := a.Store.LoadEventForObservationArchive(ctx, item.RunID, item.Sequence)
	if err != nil {
		return err
	}
	body, err := json.Marshal(struct {
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
	if err != nil {
		return err
	}
	hash := sha256.Sum256(body)
	digest := hex.EncodeToString(hash[:])
	key := fmt.Sprintf("observations/%s/%s/%020d-%s.json", tenant, committed.RunID, committed.Sequence, digest[:16])
	if err := a.Objects.Put(ctx, key, body, "application/json"); err != nil {
		return err
	}
	if err := a.Store.RecordObservationArchive(ctx, tenant, committed, key, digest, int64(len(body))); err != nil {
		return err
	}
	return a.Queue.AckObservationQueue(ctx, item.ID)
}
