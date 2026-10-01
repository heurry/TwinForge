package runtime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
)

type memoryWriteQueue interface {
	ClaimMemoryWriteJob(context.Context, int64) (agent.MemoryWriteJob, bool, error)
	CompleteMemoryWriteJob(context.Context, string, string, string, json.RawMessage) error
	FailMemoryWriteJob(context.Context, string, string, string, string, *time.Time, json.RawMessage) error
}

type memoryWriteLifecycleRecorder interface {
	RecordMemoryLifecycleEvent(context.Context, agent.MemoryLifecycleEvent) (agent.MemoryLifecycleEvent, error)
}

// MemoryExtractor is intentionally separate from queue ownership. A provider
// adapter can load the job's Run evidence and return only the structured
// candidate envelope; the worker still validates it before marking success.
type MemoryExtractor interface {
	Extract(context.Context, agent.MemoryWriteJob) (json.RawMessage, error)
}

type MemoryCandidateWriter interface {
	Persist(context.Context, agent.MemoryWriteJob, []agent.MemoryExtractionCandidate) (json.RawMessage, error)
}

type MemoryWriteWorkerConfig struct {
	LeaseSeconds int64
	PollInterval time.Duration
	MaxAttempts  int
	RetryBackoff time.Duration
}

type MemoryWriteWorker struct {
	queue     memoryWriteQueue
	extractor MemoryExtractor
	writer    MemoryCandidateWriter
	config    MemoryWriteWorkerConfig
}

func NewMemoryWriteWorker(queue memoryWriteQueue, extractor MemoryExtractor, writer MemoryCandidateWriter, config MemoryWriteWorkerConfig) (*MemoryWriteWorker, error) {
	if queue == nil || extractor == nil || writer == nil {
		return nil, errors.New("memory write queue, extractor and writer are required")
	}
	if config.LeaseSeconds <= 0 {
		config.LeaseSeconds = 300
	}
	if config.PollInterval <= 0 {
		config.PollInterval = 2 * time.Second
	}
	if config.MaxAttempts <= 0 {
		config.MaxAttempts = 5
	}
	if config.RetryBackoff <= 0 {
		config.RetryBackoff = 5 * time.Second
	}
	return &MemoryWriteWorker{queue: queue, extractor: extractor, writer: writer, config: config}, nil
}

// RunOnce claims and processes at most one job. It is useful both for a
// dedicated worker loop and for deterministic tests.
func (w *MemoryWriteWorker) RunOnce(ctx context.Context) (bool, error) {
	job, found, err := w.queue.ClaimMemoryWriteJob(ctx, w.config.LeaseSeconds)
	if err != nil || !found {
		return found, err
	}
	w.recordLifecycle(ctx, job, "MEMORY_EXTRACTION_STARTED", map[string]any{"trigger": job.Trigger, "attempt": job.Attempt}, "started")
	result, extractErr := w.extractor.Extract(ctx, job)
	var candidates []agent.MemoryExtractionCandidate
	var persistedSummary json.RawMessage
	if extractErr == nil {
		candidates, extractErr = decodeMemoryExtractionEnvelope(result)
		if extractErr == nil {
			w.recordLifecycle(ctx, job, "MEMORY_CANDIDATE_PROPOSED", map[string]any{"candidate_count": len(candidates)}, "candidates")
		}
	}
	if extractErr == nil {
		persistedSummary, extractErr = w.writer.Persist(ctx, job, candidates)
	}
	if extractErr == nil {
		if len(persistedSummary) == 0 {
			persistedSummary = json.RawMessage(`{"candidate_count":0}`)
		}
		if err := w.queue.CompleteMemoryWriteJob(ctx, job.TenantID, job.ID, stringValue(job.LeaseToken), persistedSummary); err != nil {
			return true, err
		}
		w.recordLifecycle(ctx, job, "MEMORY_EXTRACTION_COMPLETED", map[string]any{"candidate_count": len(candidates)}, "completed")
		return true, nil
	}
	retry := job.Attempt < w.config.MaxAttempts
	var retryAt *time.Time
	if retry {
		next := time.Now().UTC().Add(w.config.RetryBackoff * time.Duration(job.Attempt))
		retryAt = &next
	}
	if err := w.queue.FailMemoryWriteJob(ctx, job.TenantID, job.ID, stringValue(job.LeaseToken), extractErr.Error(), retryAt, json.RawMessage(`{"error":"extraction_failed"}`)); err != nil {
		return true, err
	}
	w.recordLifecycle(ctx, job, "MEMORY_EXTRACTION_FAILED", map[string]any{"error": extractErr.Error(), "retry": retry}, "failed")
	return true, nil
}

func (w *MemoryWriteWorker) recordLifecycle(ctx context.Context, job agent.MemoryWriteJob, eventType string, payload map[string]any, suffix string) {
	recorder, ok := w.queue.(memoryWriteLifecycleRecorder)
	if !ok || strings.TrimSpace(job.TenantID) == "" {
		return
	}
	if payload == nil {
		payload = map[string]any{}
	}
	payload["job_id"] = job.ID
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return
	}
	var runID *string
	if job.RunID != nil && strings.TrimSpace(*job.RunID) != "" {
		runID = job.RunID
	}
	_, _ = recorder.RecordMemoryLifecycleEvent(ctx, agent.MemoryLifecycleEvent{
		TenantID: job.TenantID, RunID: runID, EventType: eventType,
		IdempotencyKey: lifecycleKey("memory-job", job.ID+":"+suffix), Payload: payloadBytes,
	})
}

func lifecycleKey(prefix, value string) *string {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil
	}
	key := strings.TrimSpace(prefix) + ":" + value
	return &key
}

func (w *MemoryWriteWorker) Run(ctx context.Context) error {
	ticker := time.NewTicker(w.config.PollInterval)
	defer ticker.Stop()
	for {
		worked, err := w.RunOnce(ctx)
		if err != nil {
			return err
		}
		if worked {
			continue
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

func decodeMemoryExtractionEnvelope(raw json.RawMessage) ([]agent.MemoryExtractionCandidate, error) {
	if len(raw) == 0 {
		return nil, errors.New("memory extractor returned an empty result")
	}
	var envelope struct {
		Candidates []agent.MemoryExtractionCandidate `json:"candidates"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return nil, fmt.Errorf("decode memory extraction result: %w", err)
	}
	if err := agent.ValidateMemoryExtractionCandidates(envelope.Candidates); err != nil {
		return nil, err
	}
	return envelope.Candidates, nil
}

func stringValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}
