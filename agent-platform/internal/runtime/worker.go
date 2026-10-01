// Package runtime owns distributed Worker scheduling and fenced Run execution.
package runtime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/approval"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/delegation"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/interaction"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
)

// Repository is the durable scheduling surface required by a Worker.
type Repository interface {
	ClaimNext(ctx context.Context, workerID string, leaseDuration time.Duration) (agent.Run, error)
	RenewLease(ctx context.Context, lease agent.Lease, duration time.Duration) (agent.Lease, error)
	GetRun(ctx context.Context, runID string) (agent.Run, error)
	Transition(
		ctx context.Context,
		lease agent.Lease,
		expected, next agent.RunStatus,
		transitionEvent event.Input,
	) (agent.Run, error)
}

// Processor executes the version-pinned Agent attached to one claimed Run.
type Processor interface {
	Process(ctx context.Context, run agent.Run) (json.RawMessage, error)
}

// Config bounds polling and ownership renewal.
type Config struct {
	WorkerID            string
	RuntimeVersion      string
	ToolContractVersion string
	ProtocolVersion     string
	CapabilityHash      string
	Capabilities        []string
	PollInterval        time.Duration
	LeaseDuration       time.Duration
	HeartbeatInterval   time.Duration
}

type capabilityRegistrar interface {
	RegisterWorkerCapabilities(context.Context, string, string, string, string, string, []string) error
}

// deadlineAttemptContinuator atomically closes one timed-out root Run attempt
// and queues its replacement under the same Workflow. Implementations must
// fence the old lease and create the continuation in one transaction.
type deadlineAttemptContinuator interface {
	ContinueRunAfterDeadline(context.Context, agent.Lease, event.Input) (agent.Run, error)
}

// Worker claims and processes Runs one at a time. Horizontal concurrency is
// achieved by multiple Worker processes; per-process concurrency is added only
// after durable invariants are exercised under load.
type Worker struct {
	repository Repository
	processor  Processor
	config     Config
}

// NewWorker validates and creates a durable Worker.
func NewWorker(repository Repository, processor Processor, config Config) (*Worker, error) {
	if repository == nil || processor == nil {
		return nil, errors.New("repository and processor are required")
	}
	if config.WorkerID == "" {
		return nil, errors.New("worker id is required")
	}
	if config.PollInterval <= 0 || config.LeaseDuration <= 0 || config.HeartbeatInterval <= 0 {
		return nil, errors.New("poll, lease, and heartbeat durations must be positive")
	}
	if config.HeartbeatInterval >= config.LeaseDuration {
		return nil, errors.New("heartbeat interval must be shorter than lease duration")
	}
	return &Worker{repository: repository, processor: processor, config: config}, nil
}

// Run polls until the parent context is cancelled.
func (w *Worker) Run(ctx context.Context) error {
	if registrar, ok := w.repository.(capabilityRegistrar); ok {
		if err := registrar.RegisterWorkerCapabilities(ctx, w.config.WorkerID, w.config.RuntimeVersion, w.config.ToolContractVersion, w.config.ProtocolVersion, w.config.CapabilityHash, append([]string(nil), w.config.Capabilities...)); err != nil {
			return fmt.Errorf("worker capability handshake: %w", err)
		}
	}
	timer := time.NewTimer(0)
	defer timer.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-timer.C:
		}
		run, err := w.repository.ClaimNext(ctx, w.config.WorkerID, w.config.LeaseDuration)
		if errors.Is(err, agent.ErrNoRunnableRun) {
			timer.Reset(w.config.PollInterval)
			continue
		}
		if err != nil {
			return fmt.Errorf("claim run: %w", err)
		}
		if err := w.process(ctx, run); err != nil && !errors.Is(err, agent.ErrLeaseLost) {
			slog.Error("process agent run", "run_id", run.ID, "worker_id", w.config.WorkerID, "err", err)
		}
		timer.Reset(0)
	}
}

func (w *Worker) process(parent context.Context, run agent.Run) error {
	if run.TraceParent != nil {
		headers := map[string][]string{"traceparent": {*run.TraceParent}}
		parent = otel.GetTextMapPropagator().Extract(parent, propagation.HeaderCarrier(headers))
	}
	lease := agent.Lease{RunID: run.ID, Owner: w.config.WorkerID, Token: run.LeaseToken}
	if run.LeaseExpiresAt != nil {
		lease.Expiry = *run.LeaseExpiresAt
	}
	if run.CancelRequestedAt != nil {
		_, err := w.repository.Transition(parent, lease, agent.RunRunning, agent.RunCancelled, event.Input{
			Type: event.RunCancelled, Payload: jsonPayload(map[string]string{"reason": "cancel requested before execution"}),
		})
		return err
	}

	runCtx, cancel := context.WithCancelCause(parent)
	defer cancel(nil)
	renewDone := make(chan error, 1)
	go func() { renewDone <- w.renew(runCtx, cancel, lease) }()

	output, processErr := w.processor.Process(runCtx, run)
	cancel(errProcessingFinished)
	renewErr := <-renewDone
	if errors.Is(renewErr, agent.ErrLeaseLost) {
		return agent.ErrLeaseLost
	}
	if renewErr != nil && !errors.Is(renewErr, context.Canceled) && !errors.Is(renewErr, errProcessingFinished) && !errors.Is(renewErr, errCancellationRequested) {
		return fmt.Errorf("renew lease: %w", renewErr)
	}
	if cause := context.Cause(runCtx); errors.Is(cause, errCancellationRequested) {
		_, err := w.repository.Transition(context.WithoutCancel(parent), lease,
			agent.RunRunning, agent.RunCancelled, event.Input{
				Type: event.RunCancelled, Payload: jsonPayload(map[string]string{"reason": "cancel requested"}),
			})
		return err
	}
	if parent.Err() != nil {
		// Leave the Run owned until lease expiry so another Worker resumes it.
		return parent.Err()
	}
	if processErr != nil {
		if errors.Is(processErr, approval.ErrRequired) {
			_, err := w.repository.Transition(parent, lease, agent.RunRunning, agent.RunWaitingApproval, event.Input{
				Type: event.RunSuspended, Payload: jsonPayload(map[string]string{"reason": "tool_approval_required"}),
			})
			return err
		}
		if errors.Is(processErr, delegation.ErrPending) {
			_, err := w.repository.Transition(parent, lease, agent.RunRunning, agent.RunWaitingExternal, event.Input{Type: event.RunSuspended, Payload: jsonPayload(map[string]string{"reason": "agent_delegation_pending"})})
			return err
		}
		if errors.Is(processErr, interaction.ErrInputRequired) {
			_, err := w.repository.Transition(parent, lease, agent.RunRunning, agent.RunWaitingInput, event.Input{Type: event.RunSuspended, Payload: jsonPayload(map[string]string{"reason": "user_input_required"})})
			return err
		}
		if errors.Is(processErr, context.DeadlineExceeded) && run.ParentRunID == nil {
			if continuator, ok := w.repository.(deadlineAttemptContinuator); ok {
				failure := event.Input{Type: event.RunFailed, Payload: processFailurePayload(processErr)}
				if _, continueErr := continuator.ContinueRunAfterDeadline(context.WithoutCancel(parent), lease, failure); continueErr == nil {
					// The attempt failed, but the Workflow remains active and has a
					// queued successor. Returning nil prevents the Worker from logging
					// a handled recovery as an unhandled processing failure.
					return nil
				}
			}
		}
		_, err := w.repository.Transition(parent, lease, agent.RunRunning, agent.RunFailed, event.Input{
			Type: event.RunFailed, Payload: processFailurePayload(processErr),
		})
		if err != nil {
			return errors.Join(processErr, err)
		}
		return processErr
	}
	_, err := w.repository.Transition(parent, lease, agent.RunRunning, agent.RunCompleted, event.Input{
		Type: event.RunCompleted, Payload: jsonPayload(map[string]any{"output": output}),
	})
	return err
}

func processFailurePayload(processErr error) json.RawMessage {
	payload := map[string]any{
		"error":      processErr.Error(),
		"error_code": "RUN_EXECUTION_FAILED",
		"retryable":  false,
	}
	if errors.Is(processErr, context.DeadlineExceeded) {
		payload["error_code"] = "RUN_ATTEMPT_DEADLINE_EXCEEDED"
		payload["retryable"] = true
		payload["recovery"] = "create a continuation Run for the same Workflow; restore the latest checkpoint and do not replay committed Tool calls"
	}
	return jsonPayload(payload)
}

func (w *Worker) renew(ctx context.Context, cancel context.CancelCauseFunc, lease agent.Lease) error {
	ticker := time.NewTicker(w.config.HeartbeatInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return context.Cause(ctx)
		case <-ticker.C:
		}
		next, err := w.repository.RenewLease(ctx, lease, w.config.LeaseDuration)
		if err != nil {
			cancel(err)
			return err
		}
		lease = next
		current, err := w.repository.GetRun(ctx, lease.RunID)
		if err != nil {
			cancel(err)
			return err
		}
		if current.CancelRequestedAt != nil {
			cancel(errCancellationRequested)
			return errCancellationRequested
		}
	}
}

var (
	errProcessingFinished    = errors.New("processor finished")
	errCancellationRequested = errors.New("run cancellation requested")
)

func jsonPayload(value any) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err != nil {
		panic(fmt.Sprintf("marshal runtime event payload: %v", err))
	}
	return encoded
}
