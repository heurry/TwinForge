package runtime

import (
	"context"
	"encoding/json"
	"strconv"
	"testing"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
)

type fakeMemoryWriteQueue struct {
	job       agent.MemoryWriteJob
	completed bool
	failed    bool
	retryAt   *time.Time
	events    []agent.MemoryLifecycleEvent
}

func (q *fakeMemoryWriteQueue) ClaimMemoryWriteJob(context.Context, int64) (agent.MemoryWriteJob, bool, error) {
	if q.completed || q.failed {
		return agent.MemoryWriteJob{}, false, nil
	}
	return q.job, true, nil
}
func (q *fakeMemoryWriteQueue) CompleteMemoryWriteJob(context.Context, string, string, string, json.RawMessage) error {
	q.completed = true
	return nil
}
func (q *fakeMemoryWriteQueue) FailMemoryWriteJob(_ context.Context, _ string, _ string, _ string, _ string, retryAt *time.Time, _ json.RawMessage) error {
	q.failed = true
	q.retryAt = retryAt
	return nil
}
func (q *fakeMemoryWriteQueue) RecordMemoryLifecycleEvent(_ context.Context, item agent.MemoryLifecycleEvent) (agent.MemoryLifecycleEvent, error) {
	q.events = append(q.events, item)
	return item, nil
}

type fakeMemoryExtractor struct {
	result json.RawMessage
	err    error
}

func (e fakeMemoryExtractor) Extract(context.Context, agent.MemoryWriteJob) (json.RawMessage, error) {
	return e.result, e.err
}

type fakeMemoryWriter struct{}

func (fakeMemoryWriter) Persist(_ context.Context, _ agent.MemoryWriteJob, candidates []agent.MemoryExtractionCandidate) (json.RawMessage, error) {
	return json.RawMessage(`{"candidate_count":` + strconv.Itoa(len(candidates)) + `}`), nil
}

func TestMemoryWriteWorkerValidatesBeforeCompleting(t *testing.T) {
	lease := "lease-1"
	queue := &fakeMemoryWriteQueue{job: agent.MemoryWriteJob{ID: "job-1", TenantID: "tenant-1", Attempt: 1, LeaseToken: &lease}}
	worker, err := NewMemoryWriteWorker(queue, fakeMemoryExtractor{result: json.RawMessage(`{"candidates":[]}`)}, fakeMemoryWriter{}, MemoryWriteWorkerConfig{})
	if err != nil {
		t.Fatal(err)
	}
	worked, err := worker.RunOnce(context.Background())
	if err != nil || !worked || !queue.completed || queue.failed {
		t.Fatalf("worked=%v err=%v completed=%v failed=%v", worked, err, queue.completed, queue.failed)
	}
	if len(queue.events) != 3 || queue.events[0].EventType != "MEMORY_EXTRACTION_STARTED" || queue.events[1].EventType != "MEMORY_CANDIDATE_PROPOSED" || queue.events[2].EventType != "MEMORY_EXTRACTION_COMPLETED" {
		t.Fatalf("unexpected lifecycle events: %+v", queue.events)
	}
}

func TestMemoryWriteWorkerRetriesInvalidExtraction(t *testing.T) {
	lease := "lease-1"
	queue := &fakeMemoryWriteQueue{job: agent.MemoryWriteJob{ID: "job-1", TenantID: "tenant-1", Attempt: 1, LeaseToken: &lease}}
	worker, err := NewMemoryWriteWorker(queue, fakeMemoryExtractor{result: json.RawMessage(`{"candidates":[{"semantic_type":"project"}]}`)}, fakeMemoryWriter{}, MemoryWriteWorkerConfig{RetryBackoff: time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := worker.RunOnce(context.Background()); err != nil || !queue.failed || queue.retryAt == nil {
		t.Fatalf("err=%v failed=%v retry_at=%v", err, queue.failed, queue.retryAt)
	}
	if len(queue.events) != 2 || queue.events[0].EventType != "MEMORY_EXTRACTION_STARTED" || queue.events[1].EventType != "MEMORY_EXTRACTION_FAILED" {
		t.Fatalf("unexpected lifecycle events: %+v", queue.events)
	}
}
