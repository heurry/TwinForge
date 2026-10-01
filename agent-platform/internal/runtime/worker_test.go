package runtime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
)

type fakeRepository struct {
	mu          sync.Mutex
	run         agent.Run
	claimErr    error
	renewErr    error
	transitions []agent.RunStatus
	events      []event.Input
}

type continuingFakeRepository struct {
	*fakeRepository
	continuations int
	lastFailure   event.Input
}

func (f *continuingFakeRepository) ContinueRunAfterDeadline(_ context.Context, _ agent.Lease, failure event.Input) (agent.Run, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.continuations++
	f.lastFailure = failure
	f.run.Status = agent.RunFailed
	f.transitions = append(f.transitions, agent.RunFailed)
	f.events = append(f.events, failure)
	return agent.Run{ID: "run-2", WorkflowID: f.run.WorkflowID, Status: agent.RunQueued, TriggerType: "automatic_retry"}, nil
}

func (f *fakeRepository) ClaimNext(_ context.Context, workerID string, leaseDuration time.Duration) (agent.Run, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.claimErr != nil {
		return agent.Run{}, f.claimErr
	}
	owner := workerID
	expiry := time.Now().Add(leaseDuration)
	f.run.Status = agent.RunRunning
	f.run.LeaseOwner = &owner
	f.run.LeaseToken++
	f.run.LeaseExpiresAt = &expiry
	return f.run, nil
}

func (f *fakeRepository) RenewLease(_ context.Context, lease agent.Lease, duration time.Duration) (agent.Lease, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.renewErr != nil {
		return agent.Lease{}, f.renewErr
	}
	lease.Expiry = time.Now().Add(duration)
	return lease, nil
}

func (f *fakeRepository) GetRun(_ context.Context, _ string) (agent.Run, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.run, nil
}

func (f *fakeRepository) Transition(
	_ context.Context,
	_ agent.Lease,
	_, next agent.RunStatus,
	input event.Input,
) (agent.Run, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.run.Status = next
	f.transitions = append(f.transitions, next)
	f.events = append(f.events, input)
	return f.run, nil
}

func TestWorkerMarksDeadlineFailureAsResumableAttempt(t *testing.T) {
	repository := &fakeRepository{run: agent.Run{ID: "run-1", Status: agent.RunQueued}}
	worker := mustWorker(t, repository, processorFunc(func(context.Context, agent.Run) (json.RawMessage, error) {
		return nil, fmt.Errorf("model call at step 7: %w", context.DeadlineExceeded)
	}))
	run, err := repository.ClaimNext(context.Background(), "worker-1", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if err := worker.process(context.Background(), run); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error = %v", err)
	}
	if len(repository.events) != 1 {
		t.Fatalf("transition events = %+v", repository.events)
	}
	var payload map[string]any
	if err := json.Unmarshal(repository.events[0].Payload, &payload); err != nil {
		t.Fatal(err)
	}
	if payload["error_code"] != "RUN_ATTEMPT_DEADLINE_EXCEEDED" || payload["retryable"] != true {
		t.Fatalf("deadline failure payload = %+v", payload)
	}
}

func TestWorkerAutomaticallyContinuesRootRunAfterDeadline(t *testing.T) {
	base := &fakeRepository{run: agent.Run{ID: "run-1", WorkflowID: "workflow-1", Status: agent.RunQueued}}
	repository := &continuingFakeRepository{fakeRepository: base}
	worker := mustWorker(t, repository, processorFunc(func(context.Context, agent.Run) (json.RawMessage, error) {
		return nil, fmt.Errorf("model call: %w", context.DeadlineExceeded)
	}))
	run, err := repository.ClaimNext(context.Background(), "worker-1", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if err := worker.process(context.Background(), run); err != nil {
		t.Fatalf("handled deadline continuation returned error: %v", err)
	}
	if repository.continuations != 1 || len(repository.transitions) != 1 || repository.transitions[0] != agent.RunFailed {
		t.Fatalf("continuation state = %+v", repository)
	}
	var payload map[string]any
	if err := json.Unmarshal(repository.lastFailure.Payload, &payload); err != nil {
		t.Fatal(err)
	}
	if payload["error_code"] != "RUN_ATTEMPT_DEADLINE_EXCEEDED" || payload["retryable"] != true {
		t.Fatalf("failure payload = %+v", payload)
	}
}

type processorFunc func(context.Context, agent.Run) (json.RawMessage, error)

func (f processorFunc) Process(ctx context.Context, run agent.Run) (json.RawMessage, error) {
	return f(ctx, run)
}

func TestWorkerCompletesClaimedRun(t *testing.T) {
	t.Parallel()

	repository := &fakeRepository{run: agent.Run{ID: "run-1", Status: agent.RunQueued}}
	worker := mustWorker(t, repository, processorFunc(func(context.Context, agent.Run) (json.RawMessage, error) {
		return json.RawMessage(`{"answer":"ok"}`), nil
	}))
	run, err := repository.ClaimNext(context.Background(), "worker-1", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if err := worker.process(context.Background(), run); err != nil {
		t.Fatal(err)
	}
	if got := repository.transitions; len(got) != 1 || got[0] != agent.RunCompleted {
		t.Fatalf("transitions = %v, want completed", got)
	}
}

func TestWorkerFailsClaimedRun(t *testing.T) {
	t.Parallel()

	repository := &fakeRepository{run: agent.Run{ID: "run-1", Status: agent.RunQueued}}
	processErr := errors.New("model unavailable")
	worker := mustWorker(t, repository, processorFunc(func(context.Context, agent.Run) (json.RawMessage, error) {
		return nil, processErr
	}))
	run, err := repository.ClaimNext(context.Background(), "worker-1", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if err := worker.process(context.Background(), run); !errors.Is(err, processErr) {
		t.Fatalf("error = %v, want %v", err, processErr)
	}
	if got := repository.transitions; len(got) != 1 || got[0] != agent.RunFailed {
		t.Fatalf("transitions = %v, want failed", got)
	}
}

func TestWorkerHonorsCancellationBeforeExecution(t *testing.T) {
	t.Parallel()

	requested := time.Now()
	repository := &fakeRepository{run: agent.Run{ID: "run-1", Status: agent.RunQueued, CancelRequestedAt: &requested}}
	called := false
	worker := mustWorker(t, repository, processorFunc(func(context.Context, agent.Run) (json.RawMessage, error) {
		called = true
		return nil, nil
	}))
	run, err := repository.ClaimNext(context.Background(), "worker-1", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if err := worker.process(context.Background(), run); err != nil {
		t.Fatal(err)
	}
	if called {
		t.Fatal("processor must not execute a cancelled run")
	}
	if got := repository.transitions; len(got) != 1 || got[0] != agent.RunCancelled {
		t.Fatalf("transitions = %v, want cancelled", got)
	}
}

func TestWorkerStopsRunningProcessWhenCancellationIsRequested(t *testing.T) {
	t.Parallel()

	repository := &fakeRepository{run: agent.Run{ID: "run-1", Status: agent.RunQueued}}
	started := make(chan struct{})
	worker := mustWorker(t, repository, processorFunc(func(ctx context.Context, _ agent.Run) (json.RawMessage, error) {
		close(started)
		<-ctx.Done()
		return nil, context.Cause(ctx)
	}))
	run, err := repository.ClaimNext(context.Background(), "worker-1", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- worker.process(context.Background(), run) }()
	<-started

	repository.mu.Lock()
	requested := time.Now()
	repository.run.CancelRequestedAt = &requested
	repository.mu.Unlock()

	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("running process did not stop after cancellation was requested")
	}
	if got := repository.transitions; len(got) != 1 || got[0] != agent.RunCancelled {
		t.Fatalf("transitions = %v, want cancelled", got)
	}
}

func TestWorkerStopsWhenLeaseIsLost(t *testing.T) {
	t.Parallel()

	repository := &fakeRepository{
		run:      agent.Run{ID: "run-1", Status: agent.RunQueued},
		renewErr: agent.ErrLeaseLost,
	}
	worker := mustWorker(t, repository, processorFunc(func(ctx context.Context, _ agent.Run) (json.RawMessage, error) {
		<-ctx.Done()
		return nil, context.Cause(ctx)
	}))
	run, err := repository.ClaimNext(context.Background(), "worker-1", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if err := worker.process(context.Background(), run); !errors.Is(err, agent.ErrLeaseLost) {
		t.Fatalf("error = %v, want lease lost", err)
	}
	if len(repository.transitions) != 0 {
		t.Fatalf("stale worker must not transition run: %v", repository.transitions)
	}
}

func mustWorker(t *testing.T, repository Repository, processor Processor) *Worker {
	t.Helper()
	worker, err := NewWorker(repository, processor, Config{
		WorkerID: "worker-1", PollInterval: time.Millisecond,
		LeaseDuration: 50 * time.Millisecond, HeartbeatInterval: time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	return worker
}
