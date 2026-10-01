// Package react implements the generic ReAct v1 Harness.
package react

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/approval"
	contextpkg "github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/context"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/contract"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/delegation"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/harness"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/interaction"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/protocol"
	reviewcontract "github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/review"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/taskplan"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

// ErrStepLimit reports a Run that did not produce a final answer in time.
var ErrStepLimit = errors.New("react step limit reached")

const completionBlockFingerprintMetadata = "runtime_completion_block_fingerprint"

// ToolFailureRecoveryMetadata identifies a short, deterministic recovery
// contract. It is intentionally separate from retrieved memory: memory is
// historical guidance, while this block is a current runtime fact derived
// from a just-recorded Tool result.
const ToolFailureRecoveryMetadata = "runtime.tool_failure_recovery"

// ToolFailureRecoveryToolMetadata identifies which Tool must satisfy a
// recovery contract. It lets a successful call retire only its own reminder,
// without discarding still-relevant corrections for another Tool.
const ToolFailureRecoveryToolMetadata = "runtime.tool_failure_recovery_tool"

// FileChunkRecoveryMetadata marks a recovery contract created after a file
// payload approached or exceeded the hard Tool Schema limit. The following
// call to the same file tool is checked by the runtime instead of relying on
// the model to remember a prose-only correction.
const FileChunkRecoveryMetadata = "runtime.file_chunk_recovery"

// FileChunkRecoveryPathMetadata pins a chunk recovery to the path from the
// rejected call when that path was present and valid JSON.
const FileChunkRecoveryPathMetadata = "runtime.file_chunk_recovery_path"

// FileChunkRecoverySafeChars leaves room inside a 2048-token model response
// for the tool name, JSON envelope, escaping, and path. The Tool Schema keeps
// its 8192-character hard ceiling for compatibility; recovery deliberately
// targets a lower operational ceiling so calls do not ride that boundary.
const FileChunkRecoverySafeChars = 6000

// ToolFailureMemoryMetadata distinguishes a failure-specific historical
// memory lookup from the Turn's initial memory block. Only one such lookup is
// active at a time; otherwise every failed call appends another Top-K block.
const ToolFailureMemoryMetadata = "runtime.tool_failure_memory"

const maxActiveToolFailureRecoveryContracts = 2

// Config bounds one synchronous Agent decision cycle. MaxSteps is retained as
// a compatibility field for the old harness; it must not be confused with a
// PlanNode or an ActionAttempt.
type Config struct {
	// WorkflowID is the stable task identity shared by continuation Runs.
	WorkflowID string
	// TurnID is the stable Workflow-level identity. Turn remains the physical
	// per-Run lifecycle counter for compatibility with the event validator.
	TurnID   string
	MaxSteps int
	// DisableStepLimit temporarily removes the synchronous ReAct step cap.
	// Context cancellation, model/tool timeouts, tool-call policy limits and
	// durable worker leases remain active; this is not an unbounded process.
	DisableStepLimit     bool
	MaxTokens            int
	Temperature          float64
	ModelProvider        string
	ModelService         string
	ModelID              string
	ModelVersion         string
	ModelArtifactDigest  string
	ModelSelectionPolicy string
	AgentVersionID       string
	PromptVersionID      string
	SkillSetVersionID    string
	ToolSetVersionID     string
	ModelTimeout         time.Duration
	// ModelTimeoutRetries retries transient provider deadlines inside the same
	// decision boundary. Run-level deadlines are not retried here; they are
	// recovered by a new fenced Run attempt from the saved checkpoint.
	ModelTimeoutRetries int
	PlanningPolicy      string
	// PlanMutationMode is the Run routing decision. Existing Plans expose the
	// full update_plan surface only for explicit extend/replan Runs.
	PlanMutationMode            string
	ContextInputTokens          int
	ContextWindowTokens         int
	ContextTurn                 int
	RecentTurnTokens            int
	MemoryTokens                int
	KnowledgeTokens             int
	ToolResultTokens            int
	SummaryTokens               int
	StaticInstructionTokens     int
	CollapseTriggerRatio        float64
	CollapseTargetRatio         float64
	MinMessagesBetweenCollapses int
	MinTokensBetweenCollapses   int
	// ContextState is durable read-time projection metadata. Immutable events
	// keep the complete audit history; Checkpoints may adopt the bounded view.
	ContextState         json.RawMessage
	MemoryRetrieved      json.RawMessage
	ExternalRunLifecycle bool
	Checkpoints          harness.CheckpointSink
	ValidateAnswer       func(model.Message) error
	// FinalAnswerGuard returns a non-empty correction instruction when a
	// candidate answer is transport-valid but not safe to persist as the Run's
	// final output (for example an echoed context-compaction envelope).
	FinalAnswerGuard func(model.Message) (string, error)
	// ContextSummarizer is an optional semantic summarizer. Returning an error
	// activates the deterministic, loss-bounded fallback.
	ContextSummarizer func(context.Context, contextpkg.SummaryInput) (string, error)
	// MemoryCollapseBarrier runs before a new message range leaves the
	// model-facing projection, allowing a durable extraction job to be queued.
	MemoryCollapseBarrier func(context.Context, contextpkg.CollapseBarrierInput) error
	// MemoryFailureRecovery performs one bounded, error-specific memory lookup
	// after a failed tool call. It must return only model-visible reminder
	// messages; the callback owns retrieval/audit/policy enforcement.
	MemoryFailureRecovery func(context.Context, ToolFailure) ([]model.Message, error)
	// MergeMemoryContextState merges the latest memory-owned state into the
	// current context projection. The callback must preserve collapse-owned
	// generation/summary/coverage fields; it is intentionally a reducer rather
	// than a last-writer-wins state getter.
	MergeMemoryContextState func(json.RawMessage) json.RawMessage
	// MemoryReinject reloads surfaced memory revisions after Collapse removed
	// their model-facing projection. The durable message history remains the
	// source of truth; this callback only rebuilds the bounded reminder.
	MemoryReinject func(context.Context, []contextpkg.SurfacedMemory) ([]model.Message, error)
	// ToolResultOffloader persists a complete large result for executors that
	// do not already return an Artifact reference.
	ToolResultOffloader func(context.Context, tool.Call, []byte) (tool.ArtifactRef, error)
	// PlanContext returns the latest durable, runtime-rendered plan directive.
	// It is refreshed before every model request and replaces the previous
	// directive instead of accumulating stale plan revisions in the context.
	PlanContext func(context.Context) (string, error)
	// PlanSnapshot returns the complete durable Plan for deterministic scheduler
	// mutations. Unlike PlanContext, it is never truncated for model input.
	PlanSnapshot func(context.Context) (taskplan.Plan, error)
	// PlanProgress folds a committed successful Tool receipt into the durable
	// Plan before the next model decision. It is intentionally called only
	// after TOOL_COMPLETED was appended, so automatic criterion/node progress
	// can never be based on a model claim or an uncommitted provider response.
	PlanProgress func(context.Context, tool.Call) error
	// CompletionGuard returns an actionable block when a model attempts to
	// finish while durable plan work or valid completion evidence is missing.
	CompletionGuard func(context.Context) (CompletionBlock, error)
	// PlanState supplies durable Plan state to the projection and completion
	// layers. It also selects the active execution-phase tool surface;
	// substantive calls are still checked by the execution wrapper.
	PlanState func(context.Context) (exists bool, open bool, err error)
	// PlanNeedsUserInput is the explicit gate for exposing ask_user while an
	// active Plan is running. A blocked node may request external information;
	// ordinary execution and loop recovery should keep progressing autonomously.
	PlanNeedsUserInput func(context.Context) (bool, error)
	// PlanTools returns the runtime-projected capability closure for the active
	// Todo. Model-authored tool_hints guide execution; they are not an
	// authorization boundary. AgentVersion policy, risk approval, and Sandbox
	// confinement remain the actual enforcement layers.
	PlanTools func(context.Context) ([]string, error)
	// PlanContinuationRequiresTools tells the projection layer that a new user
	// instruction is an explicit continuation/repair of an existing Plan. A
	// closed Plan is a runtime state, not a reason to erase all tool schemas;
	// this callback makes that distinction explicit for the current Run.
	PlanContinuationRequiresTools func(context.Context) (bool, error)
	// NormalizeToolArguments applies narrowly scoped, auditable compatibility
	// aliases before validating against the exact schema offered this step.
	NormalizeToolArguments func(string, json.RawMessage) (json.RawMessage, error)
}

// ToolFailure is the stable, model-independent input to memory recovery.
type ToolFailure struct {
	ToolName    string
	ErrorCode   string
	FailureKind string
	Correction  string
	Error       string
	Diagnostic  string
	StdoutTail  string
	StderrTail  string
	ExitCode    *int
	TimedOut    bool
	HasStdout   bool
	HasStderr   bool
	Arguments   json.RawMessage
}

// CompletionBlock is persisted in PLAN_COMPLETION_BLOCKED and its Instruction
// is appended to the next model request as a user message. Structured fields
// keep the UI and the model-facing recovery instruction aligned.
type CompletionBlock struct {
	Fingerprint       string   `json:"fingerprint,omitempty"`
	Reason            string   `json:"reason"`
	Instruction       string   `json:"instruction"`
	StepID            string   `json:"step_id,omitempty"`
	CriterionID       string   `json:"criterion_id,omitempty"`
	Criterion         string   `json:"criterion,omitempty"`
	VerificationKind  string   `json:"verification_kind,omitempty"`
	Target            string   `json:"target,omitempty"`
	RequiredTools     []string `json:"required_tools,omitempty"`
	DeclaredTools     []string `json:"declared_tools,omitempty"`
	AutoInjectedTools []string `json:"auto_injected_tools,omitempty"`
	AvailableTools    []string `json:"available_tools,omitempty"`
	RequiredAction    string   `json:"required_action,omitempty"`
}

// Runner executes model requests and Tool Calls while emitting structural Run
// events. Durable scheduling, leases, and retry policy are runtime concerns.
type Runner struct {
	provider            model.Provider
	tools               tool.Executor
	events              event.Sink
	config              Config
	failureRecoverySeen map[string]int
}

// New creates a ReAct v1 Runner.
func New(provider model.Provider, tools tool.Executor, events event.Sink, config Config) (*Runner, error) {
	if provider == nil {
		return nil, errors.New("model provider is required")
	}
	if tools == nil {
		return nil, errors.New("tool executor is required")
	}
	if events == nil {
		return nil, errors.New("event sink is required")
	}
	if config.MaxSteps <= 0 && !config.DisableStepLimit {
		return nil, errors.New("max steps must be positive")
	}
	if effectivePlanningPolicy(config.PlanningPolicy) == "required" && config.PlanState == nil {
		return nil, errors.New("required planning policy needs durable PlanState")
	}
	return &Runner{provider: provider, tools: tools, events: events, config: config, failureRecoverySeen: make(map[string]int)}, nil
}

// Name returns the stable Harness implementation identifier.
func (r *Runner) Name() string { return "react-v1" }

// Run executes a bounded DecisionCycle. Model-visible private chain-of-thought is neither
// requested nor persisted; messages, Tool Calls, results, and usage are logged.
func (r *Runner) Run(ctx context.Context, request harness.Request) (result harness.Result, runErr error) {
	if request.RunID == "" {
		return harness.Result{}, errors.New("run id is required")
	}
	if len(request.Messages) == 0 {
		return harness.Result{}, errors.New("at least one input message is required")
	}
	result.Usage = request.Usage
	if !r.config.ExternalRunLifecycle {
		if err := r.append(ctx, event.Input{RunID: request.RunID, Type: event.RunCreated}); err != nil {
			return harness.Result{}, err
		}
	}
	turn := request.Turn
	if turn <= 0 {
		turn = 1
	}
	startStep := request.StartStep
	if startStep <= 0 {
		startStep = 1
	}
	if !request.Resume {
		if err := r.append(ctx, event.Input{RunID: request.RunID, Type: event.TurnCreated, Turn: turn}); err != nil {
			return harness.Result{}, err
		}
		if err := r.append(ctx, event.Input{RunID: request.RunID, Type: event.TurnStarted, Turn: turn}); err != nil {
			return harness.Result{}, err
		}
	}
	turnOpen := true
	stepOpen := false
	terminal := false
	paused := false
	currentStep := 0
	compactionGeneration := 0
	protocolRecoveryCount := 0
	lastRecordedToolSchemaDigest := ""

	messages := restoreExecutionLedger(mergeSystemMessages(request.Messages), request.ExecutionLedger)
	messages = contextpkg.EnsureMessageIDs(messages, request.RunID)
	if len(request.ContextState) != 0 {
		r.config.ContextState = append(json.RawMessage(nil), request.ContextState...)
	}
	definitions := r.tools.Definitions()
	schemas := make([]model.ToolSchema, 0, len(definitions))
	for _, definition := range definitions {
		schemas = append(schemas, model.ToolSchema{
			Name: definition.Name, Description: definition.Description, Parameters: definition.InputSchema,
			Risk: string(definition.Risk), CapabilityAvailable: true, ExecutionAllowed: true, RuntimeState: "ready",
		})
	}
	defer func() {
		if runErr == nil || paused {
			return
		}
		cleanupCtx := context.WithoutCancel(ctx)
		var cleanupErrs []error
		if stepOpen {
			if err := r.append(cleanupCtx, event.Input{
				RunID: request.RunID, Type: event.StepFailed, Turn: turn, Step: currentStep,
				Payload: jsonPayload(map[string]any{"status": "failed", "error": runErr.Error()}),
			}); err != nil {
				cleanupErrs = append(cleanupErrs, err)
			} else {
				stepOpen = false
			}
		}
		if turnOpen && !stepOpen {
			if err := r.append(cleanupCtx, event.Input{RunID: request.RunID, Type: event.TurnCompleted, Turn: turn}); err != nil {
				cleanupErrs = append(cleanupErrs, err)
			} else {
				turnOpen = false
			}
		}
		if !r.config.ExternalRunLifecycle && !terminal && !turnOpen && !stepOpen {
			if err := r.append(cleanupCtx, event.Input{
				RunID: request.RunID, Type: event.RunFailed, Payload: jsonPayload(map[string]string{"error": runErr.Error()}),
			}); err != nil {
				cleanupErrs = append(cleanupErrs, err)
			}
		}
		if len(cleanupErrs) != 0 {
			runErr = errors.Join(runErr, errors.Join(cleanupErrs...))
		}
	}()

	if request.Resume && len(request.PendingToolCalls) > 0 {
		currentStep = startStep
		stepOpen = true
		toolFailed, err := r.executeToolCalls(ctx, request.RunID, request.WorkspaceID, turn, startStep, &messages, result.Usage, request.PendingToolCalls, request.ActiveToolCallID, nil, "")
		if err != nil {
			if errors.Is(err, approval.ErrRequired) || errors.Is(err, delegation.ErrPending) || errors.Is(err, interaction.ErrInputRequired) {
				paused = true
			}
			return harness.Result{}, err
		}
		if toolFailed && containsAutomaticReviewerCall(messages, request.PendingToolCalls) {
			var fallback *progressReview
			messages, fallback = fallbackReviewerToReplan(messages, "the runtime-created Reviewer delegation failed; revise the Plan before further workspace mutation")
			if fallback != nil {
				if err := r.append(ctx, event.Input{RunID: request.RunID, Type: event.ProgressReviewCreated, Turn: turn, Step: startStep, Payload: jsonPayload(fallback)}); err != nil {
					return harness.Result{}, err
				}
			}
		}
		if !toolFailed {
			if decision := automaticReviewerDecision(messages, request.PendingToolCalls); decision != nil {
				if err := r.appendReviewerDecision(ctx, request.RunID, turn, startStep, *decision); err != nil {
					return harness.Result{}, err
				}
				attempted, patchFailed, patchErr := r.tryApplyReviewerPlanPatch(ctx, request.RunID, request.WorkspaceID, turn, startStep, &messages, result.Usage, *decision, schemas)
				if patchErr != nil {
					if errors.Is(patchErr, approval.ErrRequired) || errors.Is(patchErr, delegation.ErrPending) || errors.Is(patchErr, interaction.ErrInputRequired) {
						paused = true
					}
					return harness.Result{}, patchErr
				}
				if attempted && patchFailed {
					toolFailed = true
				}
			}
			if !toolFailed {
				if followUp := automaticReviewerFollowUp(messages, request.PendingToolCalls); followUp != nil {
					if err := r.append(ctx, event.Input{RunID: request.RunID, Type: event.ProgressReviewCreated, Turn: turn, Step: startStep, Payload: jsonPayload(followUp)}); err != nil {
						return harness.Result{}, err
					}
				}
			}
		}
		stepType := event.StepCompleted
		stepStatus := "completed"
		if toolFailed {
			stepType = event.StepFailed
			stepStatus = "failed"
		}
		position := event.Input{RunID: request.RunID, Turn: turn, Step: startStep, Type: stepType, Payload: jsonPayload(map[string]any{"status": stepStatus, "resumed": true})}
		if err := r.append(ctx, position); err != nil {
			return harness.Result{}, err
		}
		stepOpen = false
		if err := r.saveCheckpoint(ctx, harness.Checkpoint{RunID: request.RunID, Turn: turn, NextStep: startStep + 1, Messages: messages, Usage: result.Usage}); err != nil {
			return harness.Result{}, err
		}
		startStep++
	}

	for step := startStep; r.config.DisableStepLimit || step <= r.config.MaxSteps; step++ {
		stepStarted := time.Now()
		if err := ctx.Err(); err != nil {
			return harness.Result{}, err
		}
		position := event.Input{RunID: request.RunID, Turn: turn, Step: step}
		position.Type = event.StepStarted
		if err := r.append(ctx, position); err != nil {
			return harness.Result{}, err
		}
		stepOpen = true
		currentStep = step
		planContext := ""
		if r.config.PlanContext != nil {
			var planErr error
			planContext, planErr = r.config.PlanContext(ctx)
			if planErr != nil {
				return harness.Result{}, fmt.Errorf("load durable plan context: %w", planErr)
			}
			messages = replaceRuntimePlanContext(messages, planContext)
		}
		// Turn-local ReAct loops otherwise only see individual tool errors. Raise
		// a bounded, durable checkpoint before the next model request when facts
		// show repeated failure, repeated observation, or no workspace progress.
		// The checkpoint is deterministic and auditable; it asks the model for a
		// new action without storing private chain-of-thought.
		var progressReview *progressReview
		messages, progressReview = ensureProgressReview(messages, false)
		if progressReview != nil {
			position.Type = event.ProgressReviewCreated
			position.Payload = jsonPayload(progressReview)
			if err := r.append(ctx, position); err != nil {
				return harness.Result{}, err
			}
		}
		stepSchemas := append([]model.ToolSchema(nil), schemas...)
		planExistsAtStep := false
		planHasOpenWorkAtStep := false
		planNeedsUserInputAtStep := false
		continuationRequiresTools := false
		planningGoverned := r.config.PlanState != nil || effectivePlanningPolicy(r.config.PlanningPolicy) == "disabled"
		if effectivePlanningPolicy(r.config.PlanningPolicy) == "disabled" {
			// Conversational-only is one of the explicit permanent visibility
			// boundaries. ask_user remains visible because it does not perform
			// substantive work.
			stepSchemas = onlyToolSchemas(stepSchemas, "ask_user")
		} else if r.config.PlanState != nil {
			planExists, planHasOpenWork, planErr := r.config.PlanState(ctx)
			if planErr != nil {
				return harness.Result{}, fmt.Errorf("load durable plan state: %w", planErr)
			}
			planExistsAtStep = planExists
			planHasOpenWorkAtStep = planHasOpenWork
			if r.config.PlanContinuationRequiresTools != nil {
				continuationRequiresTools, planErr = r.config.PlanContinuationRequiresTools(ctx)
				if planErr != nil {
					return harness.Result{}, fmt.Errorf("resolve Plan continuation intent: %w", planErr)
				}
			}
			if planExists {
				mutationMode := strings.ToLower(strings.TrimSpace(r.config.PlanMutationMode))
				if mutationMode != "extend" && mutationMode != "replan" {
					stepSchemas = withoutToolSchema(stepSchemas, "update_plan")
				}
				if planHasOpenWork && r.config.PlanTools != nil {
					if _, toolsErr := r.config.PlanTools(ctx); toolsErr != nil {
						return harness.Result{}, fmt.Errorf("load active Plan capabilities: %w", toolsErr)
					}
				}
				if planHasOpenWork {
					// The durable graph is already authoritative. Normal execution
					// uses update_plan_step; full graph replacement and user blocking
					// are reopened only by an explicit runtime condition.
					if mutationMode != "extend" && mutationMode != "replan" {
						stepSchemas = withoutToolSchema(stepSchemas, "update_plan")
					}
					if r.config.PlanNeedsUserInput != nil {
						planNeedsUserInputAtStep, planErr = r.config.PlanNeedsUserInput(ctx)
						if planErr != nil {
							return harness.Result{}, fmt.Errorf("resolve Plan user-input state: %w", planErr)
						}
					}
					if r.config.PlanNeedsUserInput != nil && !planNeedsUserInputAtStep {
						stepSchemas = withoutToolSchema(stepSchemas, "ask_user")
					}
					if review := loadExecutionLedger(messages).ProgressReview; review != nil {
						switch review.Phase {
						case "implement", "repair":
							stepSchemas = withoutToolSchemas(stepSchemas, "list_files", "search_files")
						}
					}
				}
			} else {
				// Before a Plan exists, substantive execution will be rejected by
				// planRequiredToolExecutor, but keeping its schemas visible lets the
				// model recover from that structured result instead of treating the
				// tool as nonexistent.
			}
		}
		// Progress Review is a scheduler decision, not optional prose. Narrow the
		// next model request to the action selected from durable observations. The
		// full ToolSet remains the capability ceiling and is never expanded beyond
		// schemas resolved for this AgentVersion.
		stepSchemas = applyProgressReviewToolProjection(messages, stepSchemas, schemas)
		stepToolChoice := toolChoiceForMessages(messages, stepSchemas)
		// The resolved ToolSet is the capability ceiling. The active Plan and
		// recovery phase expose a smaller decision surface without changing the
		// executor's authorization or approval policy.
		if err := r.append(ctx, event.Input{RunID: request.RunID, Turn: turn, Step: step, Type: event.ToolSchemaProjected, Payload: jsonPayload(toolProjectionPayload(schemas, stepSchemas, effectivePlanningPolicy(r.config.PlanningPolicy), planExistsAtStep, planHasOpenWorkAtStep, continuationRequiresTools))}); err != nil {
			return harness.Result{}, err
		}
		// Reviewer escalation is a scheduler action, not another model decision.
		// Build the bounded sync delegation from the immutable allowlist metadata
		// already carried by the offered schema, then execute it through the same
		// Tool/Checkpoint/Event path as a model-authored call. This removes target,
		// mode and input-shape errors without bypassing policy or idempotency.
		if review := loadExecutionLedger(messages).ProgressReview; review != nil && review.Action == "review" {
			workflowID := strings.TrimSpace(r.config.WorkflowID)
			if workflowID == "" {
				workflowID = request.RunID
			}
			if reviewerCall, ok := automaticReviewerCall(*review, stepSchemas, workflowID, turn, step, planContext); ok {
				messages = append(messages, automaticReviewerMessage(request.RunID, turn, step, reviewerCall))
				toolFailed, executeErr := r.executeToolCalls(ctx, request.RunID, request.WorkspaceID, turn, step, &messages, result.Usage, []model.ToolCall{reviewerCall}, "", stepSchemas, "runtime_scheduler")
				if executeErr != nil {
					if errors.Is(executeErr, approval.ErrRequired) || errors.Is(executeErr, delegation.ErrPending) || errors.Is(executeErr, interaction.ErrInputRequired) {
						paused = true
					}
					return harness.Result{}, executeErr
				}
				stepType := event.StepCompleted
				stepStatus := "completed"
				stepReason := "runtime_reviewer_completed"
				if toolFailed {
					stepType = event.StepFailed
					stepStatus = "failed"
					stepReason = "runtime_reviewer_failed"
					updatedMessages, fallback := fallbackReviewerToReplan(messages, "the runtime-created Reviewer delegation failed; revise the Plan before further workspace mutation")
					messages = updatedMessages
					if fallback != nil {
						if err := r.append(ctx, event.Input{RunID: request.RunID, Type: event.ProgressReviewCreated, Turn: turn, Step: step, Payload: jsonPayload(fallback)}); err != nil {
							return harness.Result{}, err
						}
					}
				} else {
					if decision := automaticReviewerDecision(messages, []model.ToolCall{reviewerCall}); decision != nil {
						if err := r.appendReviewerDecision(ctx, request.RunID, turn, step, *decision); err != nil {
							return harness.Result{}, err
						}
						attempted, patchFailed, patchErr := r.tryApplyReviewerPlanPatch(ctx, request.RunID, request.WorkspaceID, turn, step, &messages, result.Usage, *decision, schemas)
						if patchErr != nil {
							if errors.Is(patchErr, approval.ErrRequired) || errors.Is(patchErr, delegation.ErrPending) || errors.Is(patchErr, interaction.ErrInputRequired) {
								paused = true
							}
							return harness.Result{}, patchErr
						}
						if attempted {
							if patchFailed {
								stepType = event.StepFailed
								stepStatus = "failed"
								stepReason = "runtime_reviewer_plan_patch_failed"
							} else {
								stepReason = "runtime_reviewer_plan_patch_applied"
							}
						}
					}
					if stepType != event.StepFailed {
						if followUp := automaticReviewerFollowUp(messages, []model.ToolCall{reviewerCall}); followUp != nil {
							stepReason = "runtime_reviewer_requires_" + followUp.Action
							if err := r.append(ctx, event.Input{RunID: request.RunID, Type: event.ProgressReviewCreated, Turn: turn, Step: step, Payload: jsonPayload(followUp)}); err != nil {
								return harness.Result{}, err
							}
						}
					}
				}
				position.Type = stepType
				position.Payload = jsonPayload(map[string]any{"latency_ms": time.Since(stepStarted).Milliseconds(), "status": stepStatus, "reason": stepReason})
				if err := r.append(ctx, position); err != nil {
					return harness.Result{}, err
				}
				stepOpen = false
				if err := r.saveCheckpoint(ctx, harness.Checkpoint{RunID: request.RunID, Turn: turn, NextStep: step + 1, Messages: messages, Usage: result.Usage}); err != nil {
					return harness.Result{}, err
				}
				continue
			}
		}
		messageBudget := r.config.ContextInputTokens
		toolSchemaTokens := contextpkg.EstimateToolSchemas(stepSchemas)
		if r.config.ContextWindowTokens > 0 {
			var budgetErr error
			messageBudget, toolSchemaTokens, budgetErr = contextpkg.ModelMessageBudget(r.config.ContextWindowTokens, r.config.MaxTokens, stepSchemas)
			if budgetErr != nil {
				return harness.Result{}, fmt.Errorf("calculate model context budget: %w", budgetErr)
			}
			if r.config.ContextInputTokens > 0 && r.config.ContextInputTokens < messageBudget {
				messageBudget = r.config.ContextInputTokens
			}
		}
		if !request.Resume && turn == 1 && step == startStep && len(r.config.MemoryRetrieved) != 0 {
			position.Type = event.MemoryRetrieved
			position.Payload = r.config.MemoryRetrieved
			if err := r.append(ctx, position); err != nil {
				return harness.Result{}, err
			}
		}
		// Runtime state is durable and may be verbose, but it must not become an
		// ever-growing mandatory system prompt. Events preserve the full audit
		// history; the live/Checkpoint transcript is a bounded continuation view.
		messages = contextpkg.EnsureMessageIDs(messages, request.RunID)
		fullExecutionLedger := executionLedgerJSON(messages)
		// Build a non-executable read-time view on every decision, not only after
		// a worker resume or Context Collapse. Complete Tool arguments/results stay
		// in the live transcript and Event Ledger, while the model sees at most the
		// recent completed interactions and bounded Tool History observations.
		modelCheckpoint := harness.ProjectCheckpointForStorage(harness.Checkpoint{
			RunID:        request.RunID,
			Messages:     messages,
			ContextState: r.config.ContextState,
		})
		modelCheckpoint = harness.RestoreCheckpointFromStorage(modelCheckpoint)
		modelMessages := projectExecutionLedgerForModel(modelCheckpoint.Messages)
		if strings.TrimSpace(planContext) != "" {
			modelMessages = replaceRuntimePlanContext(modelMessages, planContext)
		}
		// The provider wire format intentionally forwards only the standard
		// function schema. Put the runtime capability layers in a bounded,
		// provider-neutral system block so the model can distinguish a missing
		// capability from a temporarily hidden, approval-gated, or waiting tool.
		// This is a read-time projection and never becomes durable conversation
		// history.
		// On very small legacy windows the mandatory prompt/ledger itself is
		// more important than repeating a compact projection; the complete
		// layer remains in the durable TOOL_SCHEMA_PROJECTED event. Normal
		// production windows (4K+) receive the model-visible projection.
		runtimeProjectionEnabled := r.config.ContextWindowTokens <= 0 || r.config.ContextWindowTokens >= 4096
		runtimeToolPayload := modelToolProjectionPayload(schemas, stepSchemas, effectivePlanningPolicy(r.config.PlanningPolicy), planExistsAtStep, planHasOpenWorkAtStep, continuationRequiresTools)
		runtimeToolProjectionEnabled := runtimeProjectionEnabled && (hasSubstantiveToolSchema(schemas) || hasSubstantiveToolSchema(stepSchemas))
		if runtimeToolProjectionEnabled {
			modelMessages = injectRuntimeToolProjection(modelMessages, runtimeToolPayload)
		}
		runtimeProjectionTokens := estimateMessages(modelMessages) - estimateMessages(contextpkg.StripRebuildableRuntimeBlocks(modelMessages))
		stepMaxTokens := r.config.MaxTokens
		stepMessageBudget := messageBudget
		var collapseState contextpkg.CollapseState
		if len(r.config.ContextState) != 0 {
			_ = json.Unmarshal(r.config.ContextState, &collapseState)
			if collapseState.Generation > compactionGeneration {
				compactionGeneration = collapseState.Generation
			}
		}
		if messageBudget > 0 {
			sectionBudget := contextpkg.AllocateSectionBudgetsWithStatic(stepMessageBudget, r.config.RecentTurnTokens, r.config.MemoryTokens, r.config.KnowledgeTokens, r.config.ToolResultTokens, r.config.SummaryTokens, r.config.StaticInstructionTokens)
			projectionOptions := contextpkg.ProjectionOptions{
				TriggerRatio:                r.config.CollapseTriggerRatio,
				TargetRatio:                 r.config.CollapseTargetRatio,
				MinMessagesBetweenCollapses: r.config.MinMessagesBetweenCollapses,
				MinTokensBetweenCollapses:   r.config.MinTokensBetweenCollapses,
				RecentTurnTokens:            sectionBudget.RecentTurnTokens,
				MemoryTokens:                sectionBudget.MemoryTokens,
				KnowledgeTokens:             sectionBudget.KnowledgeTokens,
				ToolResultTokens:            sectionBudget.ToolResultTokens,
				SummaryTokens:               sectionBudget.SummaryTokens,
				Summarize:                   r.config.ContextSummarizer,
				BeforeCollapse:              r.config.MemoryCollapseBarrier,
			}
			collapsed, nextState, report, compactErr := projectRuntimeAwareMessages(ctx, modelMessages, stepMessageBudget, collapseState, projectionOptions, fullExecutionLedger, planContext, runtimeToolPayload, runtimeToolProjectionEnabled)
			for (errors.Is(compactErr, contextpkg.ErrBudgetTooSmall) || errors.Is(compactErr, contextpkg.ErrProjectionBudgetTooSmall)) && r.config.ContextWindowTokens > 0 && stepMaxTokens > 128 {
				stepMaxTokens /= 2
				if stepMaxTokens < 128 {
					stepMaxTokens = 128
				}
				adjustedBudget, _, budgetErr := contextpkg.ModelMessageBudget(r.config.ContextWindowTokens, stepMaxTokens, stepSchemas)
				if budgetErr != nil {
					break
				}
				if r.config.ContextInputTokens > 0 && r.config.ContextInputTokens < adjustedBudget {
					adjustedBudget = r.config.ContextInputTokens
				}
				stepMessageBudget = adjustedBudget
				sectionBudget = contextpkg.AllocateSectionBudgetsWithStatic(stepMessageBudget, r.config.RecentTurnTokens, r.config.MemoryTokens, r.config.KnowledgeTokens, r.config.ToolResultTokens, r.config.SummaryTokens, r.config.StaticInstructionTokens)
				projectionOptions.RecentTurnTokens = sectionBudget.RecentTurnTokens
				projectionOptions.MemoryTokens = sectionBudget.MemoryTokens
				projectionOptions.KnowledgeTokens = sectionBudget.KnowledgeTokens
				projectionOptions.ToolResultTokens = sectionBudget.ToolResultTokens
				projectionOptions.SummaryTokens = sectionBudget.SummaryTokens
				collapsed, nextState, report, compactErr = projectRuntimeAwareMessages(ctx, modelMessages, stepMessageBudget, collapseState, projectionOptions, fullExecutionLedger, planContext, runtimeToolPayload, runtimeToolProjectionEnabled)
			}
			if compactErr != nil {
				return harness.Result{}, fmt.Errorf("compact model context: %w", compactErr)
			}
			modelMessages = collapsed
			collapseState = nextState
			if report.Compacted && r.config.MemoryReinject != nil && len(collapseState.SurfacedMemories) != 0 {
				rebuilt, reinjectErr := r.config.MemoryReinject(ctx, append([]contextpkg.SurfacedMemory(nil), collapseState.SurfacedMemories...))
				if reinjectErr != nil {
					return harness.Result{}, fmt.Errorf("reinject surfaced memories after collapse: %w", reinjectErr)
				}
				modelMessages = replaceContextSection(modelMessages, contextpkg.ContextSectionMemory, rebuilt)
			}
			if encodedState, stateErr := json.Marshal(collapseState); stateErr == nil {
				if r.config.MergeMemoryContextState != nil {
					encodedState = r.config.MergeMemoryContextState(encodedState)
					_ = json.Unmarshal(encodedState, &collapseState)
				}
				r.config.ContextState = encodedState
			}
			if report.Compacted {
				// Adopt the compacted projection as the live continuation state.
				// Immutable events retain full calls/results, so keeping a second
				// unbounded transcript in every subsequent Checkpoint only causes
				// the same history to be collapsed again.
				messages = restoreExecutionLedger(contextpkg.StripRebuildableRuntimeBlocks(modelMessages), fullExecutionLedger)
				compactionGeneration++
				position.Type = event.ContextCompacted
				position.Payload = jsonPayload(map[string]any{"generation": compactionGeneration, "collapse_generation": report.Generation, "mode": "read_time_projection", "summary_mode": report.SummaryMode, "before_tokens": report.BeforeTokens, "durable_history_tokens": report.BeforeTokens, "working_before_tokens": report.WorkingBeforeTokens, "after_tokens": report.AfterTokens, "final_model_input_tokens": report.AfterTokens, "removed_messages": report.RemovedMessages, "runtime_projection_tokens": runtimeProjectionTokens, "protected_tokens": report.ProtectedTokens, "protected_failure_tokens": report.ProtectedFailureTokens, "message_budget_tokens": stepMessageBudget, "tool_schema_tokens": toolSchemaTokens, "context_window_tokens": r.config.ContextWindowTokens, "configured_output_tokens": r.config.MaxTokens, "reserve_output_tokens": stepMaxTokens, "collapse_trigger_ratio": r.config.CollapseTriggerRatio, "collapse_target_ratio": r.config.CollapseTargetRatio, "collapse_cooldown_messages": r.config.MinMessagesBetweenCollapses, "collapse_cooldown_tokens": r.config.MinTokensBetweenCollapses, "recent_turn_tokens": sectionBudget.RecentTurnTokens, "memory_tokens": sectionBudget.MemoryTokens, "knowledge_tokens": sectionBudget.KnowledgeTokens, "tool_result_tokens": sectionBudget.ToolResultTokens, "summary_tokens": r.config.SummaryTokens, "static_instruction_tokens": r.config.StaticInstructionTokens})
				if err := r.append(ctx, position); err != nil {
					return harness.Result{}, err
				}
			}
		}
		position.Type = event.ContextBuilt
		currentManifest := refreshContextManifest(request.ContextManifest, modelMessages, compactionGeneration)
		if len(currentManifest) != 0 {
			position.Payload = currentManifest
		} else {
			position.Payload = jsonPayload(map[string]int{"message_count": len(modelMessages)})
		}
		if err := r.append(ctx, position); err != nil {
			return harness.Result{}, err
		}
		// Persist the exact decision boundary before invoking the provider. Tool
		// boundaries were already checkpointed, but a new continuation Run could
		// time out on its very first model call before its user input ever reached
		// durable Workflow state. This snapshot also lets a lease takeover retry
		// the model request without replaying a committed Tool call.
		if err := r.saveCheckpoint(ctx, harness.Checkpoint{
			RunID: request.RunID, Turn: turn, NextStep: step,
			Messages: messages, Usage: result.Usage,
		}); err != nil {
			return harness.Result{}, err
		}
		position.Type = event.ModelRequested
		toolSchemaDigest := toolSchemasDigest(stepSchemas)
		modelRequestEvent := map[string]any{
			"messages":           modelMessages,
			"tool_choice":        stepToolChoice,
			"tool_schema_digest": toolSchemaDigest,
			"context_manifest":   json.RawMessage(currentManifest),
			"max_tokens":         stepMaxTokens, "temperature": r.config.Temperature,
			"context_window_tokens": r.config.ContextWindowTokens, "message_budget_tokens": stepMessageBudget,
			"tool_schema_tokens": toolSchemaTokens,
			"provider":           r.config.ModelProvider, "service": r.config.ModelService,
			"model_id": r.config.ModelID, "model_version": r.config.ModelVersion,
			"artifact_digest": r.config.ModelArtifactDigest, "selection_policy": r.config.ModelSelectionPolicy,
			"agent_version_id": r.config.AgentVersionID, "prompt_version_id": r.config.PromptVersionID,
			"skillset_version_id": r.config.SkillSetVersionID, "toolset_version_id": r.config.ToolSetVersionID,
			"timeout_ms": r.config.ModelTimeout.Milliseconds(),
			"attempt":    1,
		}
		if toolSchemaDigest != lastRecordedToolSchemaDigest {
			modelRequestEvent["tools"] = stepSchemas
			lastRecordedToolSchemaDigest = toolSchemaDigest
		} else {
			modelRequestEvent["tool_schema_ref"] = "previous_model_request"
		}
		position.Payload = jsonPayload(modelRequestEvent)
		if err := r.append(ctx, position); err != nil {
			return harness.Result{}, err
		}

		modelAttempt := 1
		modelStarted := time.Now()
		initialModelRequestPayload := append([]byte(nil), position.Payload...)
		// Retries are separate model observations. Persist both the failed
		// attempt and the follow-up request so protocol recovery is visible in
		// the trajectory instead of appearing as one opaque failure.
		recordModelFailure := func(cause error, recoverable bool) error {
			position.Type = event.ModelFailed
			position.Payload = jsonPayload(map[string]any{
				"error": cause.Error(), "attempt": modelAttempt, "recoverable": recoverable,
				"latency_ms":       time.Since(modelStarted).Milliseconds(),
				"context_manifest": json.RawMessage(currentManifest),
				"provider":         r.config.ModelProvider, "service": r.config.ModelService,
				"model_id": r.config.ModelID, "model_version": r.config.ModelVersion,
				"artifact_digest": r.config.ModelArtifactDigest, "selection_policy": r.config.ModelSelectionPolicy,
				"agent_version_id": r.config.AgentVersionID, "prompt_version_id": r.config.PromptVersionID,
				"skillset_version_id": r.config.SkillSetVersionID, "toolset_version_id": r.config.ToolSetVersionID,
				"timeout_ms":   r.config.ModelTimeout.Milliseconds(),
				"failed_stage": "model_request", "error_kind": modelErrorKind(cause),
			})
			return r.append(context.WithoutCancel(ctx), position)
		}
		recordModelRetryRequest := func(cause error, maxTokens int, mode string) error {
			modelAttempt++
			var payload map[string]any
			if err := json.Unmarshal(initialModelRequestPayload, &payload); err != nil {
				payload = map[string]any{}
			}
			payload["messages"] = modelMessages
			delete(payload, "tools")
			payload["tool_schema_digest"] = toolSchemaDigest
			payload["tool_schema_ref"] = "previous_model_request"
			payload["tool_choice"] = stepToolChoice
			payload["attempt"] = modelAttempt
			payload["retry_reason"] = cause.Error()
			payload["max_tokens"] = maxTokens
			if mode != "" {
				payload["retry_mode"] = mode
			}
			position.Type = event.ModelRequested
			position.Payload = jsonPayload(payload)
			if err := r.append(ctx, position); err != nil {
				return err
			}
			modelStarted = time.Now()
			return nil
		}
		response, err := r.provider.Complete(ctx, model.Request{
			RunID: request.RunID, Messages: modelMessages, Tools: stepSchemas, ToolChoice: stepToolChoice,
			MaxTokens: stepMaxTokens, Temperature: r.config.Temperature,
		})
		// Provider tokenizers/chat templates can differ by a few tokens from our
		// deterministic estimator. If the endpoint rejects the request at the
		// hard context boundary, compact the same step once and retry instead of
		// failing a long-running task permanently.
		if err != nil && modelErrorKind(err) == "context_window_exceeded" && len(modelMessages) > 1 {
			retryBudget := estimateMessages(modelMessages) * 85 / 100
			if retryBudget < 512 {
				retryBudget = 512
			}
			sectionBudget := contextpkg.AllocateSectionBudgetsWithStatic(retryBudget, r.config.RecentTurnTokens, r.config.MemoryTokens, r.config.KnowledgeTokens, r.config.ToolResultTokens, r.config.SummaryTokens, r.config.StaticInstructionTokens)
			projectionOptions := contextpkg.ProjectionOptions{
				TriggerRatio:                r.config.CollapseTriggerRatio,
				TargetRatio:                 r.config.CollapseTargetRatio,
				MinMessagesBetweenCollapses: r.config.MinMessagesBetweenCollapses,
				MinTokensBetweenCollapses:   r.config.MinTokensBetweenCollapses,
				RecentTurnTokens:            sectionBudget.RecentTurnTokens,
				MemoryTokens:                sectionBudget.MemoryTokens,
				KnowledgeTokens:             sectionBudget.KnowledgeTokens,
				ToolResultTokens:            sectionBudget.ToolResultTokens,
				SummaryTokens:               sectionBudget.SummaryTokens,
				Summarize:                   r.config.ContextSummarizer,
				BeforeCollapse:              r.config.MemoryCollapseBarrier,
			}
			compacted, retryState, _, compactErr := projectRuntimeAwareMessages(ctx, modelMessages, retryBudget, collapseState, projectionOptions, fullExecutionLedger, planContext, runtimeToolPayload, runtimeToolProjectionEnabled)
			if compactErr != nil {
				// The legacy deterministic compactor remains the last transport
				// fallback for pathological protected-message layouts. Normal retry
				// compaction must preserve every exact user constraint.
				retryState = collapseState
				fallbackBefore := estimateMessages(modelMessages)
				fallbackInput := contextpkg.StripRebuildableRuntimeBlocks(modelMessages)
				fallbackRuntimeTokens := fallbackBefore - estimateMessages(fallbackInput)
				fallbackBudget := retryBudget - fallbackRuntimeTokens
				if fallbackBudget > 0 {
					compacted, _, compactErr = contextpkg.CompactMessages(fallbackInput, fallbackBudget)
					if compactErr == nil {
						compacted = rebuildRuntimeProjection(compacted, fullExecutionLedger, planContext, runtimeToolPayload, runtimeToolProjectionEnabled)
						if estimateMessages(compacted) > retryBudget {
							compactErr = contextpkg.ErrProjectionBudgetTooSmall
						}
					}
				}
			}
			if compactErr == nil && len(compacted) != 0 {
				if encodedState, stateErr := json.Marshal(retryState); stateErr == nil {
					r.config.ContextState = encodedState
				}
				if appendErr := recordModelFailure(err, true); appendErr != nil {
					return harness.Result{}, appendErr
				}
				modelMessages = compacted
				if appendErr := recordModelRetryRequest(err, stepMaxTokens, "context_compaction"); appendErr != nil {
					return harness.Result{}, appendErr
				}
				response, err = r.provider.Complete(ctx, model.Request{
					RunID: request.RunID, Messages: modelMessages, Tools: stepSchemas, ToolChoice: stepToolChoice,
					MaxTokens: stepMaxTokens, Temperature: r.config.Temperature,
				})
			}
		}
		// A provider-local deadline is often a transient queue/serving event. Retry
		// it with bounded backoff and a smaller output budget, but only while the
		// enclosing Run still has time. Every attempt is separately observable and
		// the decision-boundary checkpoint above remains the recovery source.
		for timeoutRetry := 0; err != nil && errors.Is(err, context.DeadlineExceeded) && ctx.Err() == nil && timeoutRetry < r.config.ModelTimeoutRetries; timeoutRetry++ {
			if appendErr := recordModelFailure(err, true); appendErr != nil {
				return harness.Result{}, appendErr
			}
			backoff := time.Duration(250*(1<<timeoutRetry)) * time.Millisecond
			timer := time.NewTimer(backoff)
			select {
			case <-ctx.Done():
				timer.Stop()
				break
			case <-timer.C:
			}
			if ctx.Err() != nil {
				break
			}
			degradedMaxTokens := stepMaxTokens * 3 / 4
			if degradedMaxTokens < 256 {
				degradedMaxTokens = stepMaxTokens
			}
			if appendErr := recordModelRetryRequest(err, degradedMaxTokens, "deadline_backoff_degraded_output"); appendErr != nil {
				return harness.Result{}, appendErr
			}
			response, err = r.provider.Complete(ctx, model.Request{
				RunID: request.RunID, Messages: modelMessages, Tools: stepSchemas, ToolChoice: stepToolChoice,
				MaxTokens: degradedMaxTokens, Temperature: r.config.Temperature,
			})
		}
		// A few instruct checkpoints emit legacy XML-ish tool markup in text
		// instead of structured tool_calls. Keep the step and its tool projection
		// intact, add one concise protocol correction, and retry once.
		if err != nil && strings.Contains(err.Error(), "model_protocol_error") {
			if appendErr := recordModelFailure(err, true); appendErr != nil {
				return harness.Result{}, appendErr
			}
			recoveryInstruction := "TOOL_PROTOCOL_ERROR: use the provided structured tool_calls schema. Do not emit <toolcall>, <function=>, or <parameter=> text. Re-emit one valid JSON tool call or a normal answer."
			if len(stepSchemas) == 0 {
				// Finalizer path: the Plan is complete and there is no executable
				// capability in this request. Ask for plain text only and keep the
				// correction out of the user conversation.
				recoveryInstruction = "FINALIZER_PROTOCOL_ERROR: this request offers no tools. Return only a concise user-facing final answer as plain text or Markdown. Never emit <toolcall>, <tool_call>, <function=>, <parameter=>, JSON tool-call envelopes, or tool receipts. Do not propose another tool call."
				modelMessages = append(modelMessages, model.TextMessage(model.RoleSystem, recoveryInstruction))
			} else {
				modelMessages = append(modelMessages, runtimeControlMessage(recoveryInstruction))
			}
			if appendErr := recordModelRetryRequest(err, stepMaxTokens, "protocol_correction"); appendErr != nil {
				return harness.Result{}, appendErr
			}
			response, err = r.provider.Complete(ctx, model.Request{
				RunID: request.RunID, Messages: modelMessages, Tools: stepSchemas, ToolChoice: stepToolChoice,
				MaxTokens: stepMaxTokens, Temperature: r.config.Temperature,
			})
		}
		if err != nil {
			if appendErr := recordModelFailure(err, false); appendErr != nil {
				_ = r.append(context.WithoutCancel(ctx), position)
			}
			position.Type = event.StepFailed
			position.Payload = jsonPayload(map[string]any{"latency_ms": time.Since(stepStarted).Milliseconds(), "status": "failed"})
			if appendErr := r.append(context.WithoutCancel(ctx), position); appendErr == nil {
				stepOpen = false
			}
			return harness.Result{}, fmt.Errorf("model call at step %d: %w", step, err)
		}
		if strings.TrimSpace(response.Message.ID) == "" {
			response.Message.ID = fmt.Sprintf("%s:turn:%d:step:%d:assistant", request.RunID, turn, step)
		}
		position.Type = event.ModelCompleted
		position.Payload = jsonPayload(map[string]any{
			"provider": response.Provider, "model_id": response.ModelID,
			"model_version": r.config.ModelVersion, "service": r.config.ModelService,
			"artifact_digest": r.config.ModelArtifactDigest, "selection_policy": r.config.ModelSelectionPolicy,
			"agent_version_id": r.config.AgentVersionID, "prompt_version_id": r.config.PromptVersionID,
			"skillset_version_id": r.config.SkillSetVersionID, "toolset_version_id": r.config.ToolSetVersionID,
			"usage":         response.Usage,
			"finish_reason": response.FinishReason, "latency_ms": time.Since(modelStarted).Milliseconds(),
			"context_manifest": json.RawMessage(currentManifest), "message": response.Message,
		})
		if err := r.append(ctx, position); err != nil {
			return harness.Result{}, err
		}
		// Rehydrate only the durable execution ledger. modelMessages is a read-time
		// projection and must never replace the complete durable message history.
		messages = restoreExecutionLedger(messages, fullExecutionLedger)
		messages = contextpkg.EnsureMessageIDs(messages, request.RunID)
		if planContext != "" {
			messages = replaceRuntimePlanContext(messages, planContext)
		}
		messages = append(messages, response.Message)
		addUsage(&result.Usage, response.Usage)
		decision, protocolErr := protocol.Decode(response.Message)
		if protocolErr != nil {
			// Transport succeeded, but the model did not produce a Runtime
			// decision. Treat this as a bounded protocol-recovery cycle rather than
			// failing the whole long task. The invalid text is retained for audit,
			// never interpreted as a tool call, and the next model request gets a
			// concise correction. A second violation is terminal and explicit.
			protocolRecoveryCount++
			position.Type = event.ModelFailed
			position.Payload = jsonPayload(map[string]any{"error": protocolErr.Error(), "error_kind": "decision_protocol", "recoverable": protocolRecoveryCount < 2, "attempt": protocolRecoveryCount})
			if appendErr := r.append(ctx, position); appendErr != nil {
				return harness.Result{}, appendErr
			}
			if protocolRecoveryCount >= 2 {
				return harness.Result{}, fmt.Errorf("model decision protocol error after recovery: %w", protocolErr)
			}
			recoveryInstruction := "RUNTIME_PROTOCOL_ERROR: output one valid structured tool call using the provided schema, or plain final text. Never emit <toolcall>, <tool_call>, <function=>, or <parameter=> markup."
			if len(stepSchemas) == 0 {
				recoveryInstruction = "RUNTIME_PROTOCOL_ERROR: no tools are offered in this model request. Return plain final text only; never emit tool markup."
			}
			messages = append(messages, runtimeControlMessage(recoveryInstruction))
			position.Type = event.StepCompleted
			position.Payload = jsonPayload(map[string]any{"latency_ms": time.Since(stepStarted).Milliseconds(), "status": "continued", "reason": "decision_protocol"})
			if err := r.append(ctx, position); err != nil {
				return harness.Result{}, err
			}
			stepOpen = false
			if err := r.saveCheckpoint(ctx, harness.Checkpoint{RunID: request.RunID, Turn: turn, NextStep: step + 1, Messages: messages, Usage: result.Usage}); err != nil {
				return harness.Result{}, err
			}
			continue
		}
		protocolRecoveryCount = 0
		// A length-terminated response is materially different from a normal
		// schema violation. Keep the provider's finish reason in the durable
		// event, and also give the model an explicit recovery instruction in the
		// next context instead of making it infer truncation from broken JSON.
		if strings.EqualFold(strings.TrimSpace(response.FinishReason), "length") {
			messages = append(messages, runtimeControlMessage("RUNTIME_PROTOCOL_ERROR\nThe previous model response reached max_tokens and may have been truncated. Do not retry the same large tool payload. Re-emit one complete, compact JSON tool call; for update_plan use 3-8 short steps and one minimal acceptance criterion per step."))
		}

		if len(decision.ToolCalls) == 0 {
			if r.config.CompletionGuard != nil {
				block, guardErr := r.config.CompletionGuard(ctx)
				if guardErr != nil {
					return harness.Result{}, fmt.Errorf("check durable plan completion: %w", guardErr)
				}
				if strings.TrimSpace(block.Instruction) != "" {
					fingerprint := completionBlockFingerprint(block)
					block.Fingerprint = fingerprint
					position.Type = event.PlanCompletionBlocked
					position.Payload = jsonPayload(block)
					if err := r.append(ctx, position); err != nil {
						return harness.Result{}, err
					}
					if countUnresolvedCompletionBlocks(messages, fingerprint) >= 1 {
						position.Type = event.VerificationLoopDetected
						position.Payload = jsonPayload(map[string]any{"fingerprint": fingerprint, "reason": block.Reason, "step_id": block.StepID, "criterion_id": block.CriterionID, "action": "run_failed_before_budget_exhaustion"})
						if err := r.append(ctx, position); err != nil {
							return harness.Result{}, err
						}
						return harness.Result{}, fmt.Errorf("verification recovery loop detected for step %q criterion %q; the model ignored the same deterministic recovery contract twice", block.StepID, block.CriterionID)
					}
					message := model.TextMessage(model.RoleUser, block.Instruction)
					message.Metadata = map[string]string{
						completionBlockFingerprintMetadata:   fingerprint,
						contextpkg.ContextSectionMetadataKey: contextpkg.ContextSectionRuntime,
						contextpkg.RuntimeControlMetadataKey: "true",
					}
					messages = append(messages, message)
					position.Type = event.StepCompleted
					position.Payload = jsonPayload(map[string]any{"latency_ms": time.Since(stepStarted).Milliseconds(), "status": "continued", "reason": block.Reason})
					if err := r.append(ctx, position); err != nil {
						return harness.Result{}, err
					}
					stepOpen = false
					if err := r.saveCheckpoint(ctx, harness.Checkpoint{RunID: request.RunID, Turn: turn, NextStep: step + 1, Messages: messages, Usage: result.Usage}); err != nil {
						return harness.Result{}, err
					}
					continue
				}
			}
			if r.config.FinalAnswerGuard != nil {
				continuation, guardErr := r.config.FinalAnswerGuard(response.Message)
				if guardErr != nil {
					return harness.Result{}, fmt.Errorf("check final answer integrity: %w", guardErr)
				}
				if strings.TrimSpace(continuation) != "" {
					position.Type = event.FinalOutputRejected
					position.Payload = jsonPayload(map[string]any{"reason": "invalid_final_output", "instruction": continuation})
					if err := r.append(ctx, position); err != nil {
						return harness.Result{}, err
					}
					messages = append(messages, runtimeControlMessage(continuation))
					position.Type = event.StepCompleted
					position.Payload = jsonPayload(map[string]any{"latency_ms": time.Since(stepStarted).Milliseconds(), "status": "continued", "reason": "invalid_final_output"})
					if err := r.append(ctx, position); err != nil {
						return harness.Result{}, err
					}
					stepOpen = false
					if err := r.saveCheckpoint(ctx, harness.Checkpoint{RunID: request.RunID, Turn: turn, NextStep: step + 1, Messages: messages, Usage: result.Usage}); err != nil {
						return harness.Result{}, err
					}
					continue
				}
			}
			if r.config.ValidateAnswer != nil {
				if err := r.config.ValidateAnswer(response.Message); err != nil {
					return harness.Result{}, fmt.Errorf("validate final answer: %w", err)
				}
			}
			if planningGoverned && !planExistsAtStep {
				position.Type = event.ExecutionModeSelected
				position.Payload = jsonPayload(map[string]any{
					"mode": "conversational", "policy": effectivePlanningPolicy(r.config.PlanningPolicy),
					"source": "runtime", "reason": "completed_without_plan",
				})
				if err := r.append(ctx, position); err != nil {
					return harness.Result{}, err
				}
			}
			position.Type = event.StepCompleted
			position.Payload = jsonPayload(map[string]any{"latency_ms": time.Since(stepStarted).Milliseconds(), "status": "completed"})
			if err := r.append(ctx, position); err != nil {
				return harness.Result{}, err
			}
			stepOpen = false
			if err := r.append(ctx, event.Input{RunID: request.RunID, Type: event.TurnCompleted, Turn: turn}); err != nil {
				return harness.Result{}, err
			}
			turnOpen = false
			if err := r.saveCheckpoint(ctx, harness.Checkpoint{
				RunID: request.RunID, Turn: turn, NextStep: step + 1,
				Messages: messages, Usage: result.Usage, Completed: true, Answer: response.Message,
			}); err != nil {
				return harness.Result{}, err
			}
			if !r.config.ExternalRunLifecycle {
				if err := r.append(ctx, event.Input{RunID: request.RunID, Type: event.RunCompleted}); err != nil {
					return harness.Result{}, err
				}
			}
			terminal = true
			return harness.Result{Answer: response.Message, Messages: messages, Steps: step, Usage: result.Usage}, nil
		}

		toolFailed, err := r.executeToolCalls(ctx, request.RunID, request.WorkspaceID, turn, step, &messages, result.Usage, decision.ToolCalls, "", stepSchemas, response.FinishReason)
		if err != nil {
			if errors.Is(err, approval.ErrRequired) || errors.Is(err, delegation.ErrPending) || errors.Is(err, interaction.ErrInputRequired) {
				paused = true
			}
			return harness.Result{}, err
		}
		position.Type = event.StepCompleted
		stepStatus := "completed"
		stepReason := ""
		if toolFailed {
			// A failed Tool is still returned to the model as an observation so it
			// can repair the workspace on the next decision cycle. Close this
			// decision step as failed, rather than falsely recording it as a
			// successful step; the next step remains the recovery attempt.
			position.Type = event.StepFailed
			stepStatus = "failed"
			stepReason = "tool_failure"
		}
		payload := map[string]any{"latency_ms": time.Since(stepStarted).Milliseconds(), "status": stepStatus}
		if stepReason != "" {
			payload["reason"] = stepReason
		}
		position.Payload = jsonPayload(payload)
		if err := r.append(ctx, position); err != nil {
			return harness.Result{}, err
		}
		stepOpen = false
		if err := r.saveCheckpoint(ctx, harness.Checkpoint{
			RunID: request.RunID, Turn: turn, NextStep: step + 1,
			Messages: messages, Usage: result.Usage,
		}); err != nil {
			return harness.Result{}, err
		}
	}

	if err := r.append(ctx, event.Input{RunID: request.RunID, Type: event.TurnCompleted, Turn: turn}); err != nil {
		return harness.Result{}, err
	}
	turnOpen = false
	// Every rollover gets one persisted review, even if it did not meet an
	// early-stagnation threshold. The next physical Turn then starts from an
	// explicit progress baseline rather than replaying the prior action loop.
	var rolloverReview *progressReview
	messages, rolloverReview = ensureProgressReview(messages, true)
	if rolloverReview != nil {
		if err := r.append(ctx, event.Input{RunID: request.RunID, Type: event.ProgressReviewCreated, Turn: turn, Payload: jsonPayload(rolloverReview)}); err != nil {
			return harness.Result{}, err
		}
	}
	if err := r.saveCheckpoint(ctx, harness.Checkpoint{RunID: request.RunID, Turn: turn, NextStep: r.config.MaxSteps + 1, Messages: messages, Usage: result.Usage}); err != nil {
		return harness.Result{}, err
	}
	return harness.Result{}, ErrStepLimit
}

func replaceContextSection(messages []model.Message, section string, replacements []model.Message) []model.Message {
	filtered := make([]model.Message, 0, len(messages)+len(replacements))
	for _, message := range messages {
		if message.Metadata != nil && message.Metadata[contextpkg.ContextSectionMetadataKey] == section {
			continue
		}
		filtered = append(filtered, message)
	}
	if len(replacements) == 0 {
		return filtered
	}
	insertAt := len(filtered)
	for index, message := range filtered {
		if message.Metadata != nil && message.Metadata[contextpkg.TaskAnchorMetadataKey] == "true" {
			insertAt = index
			break
		}
	}
	result := make([]model.Message, 0, len(filtered)+len(replacements))
	result = append(result, filtered[:insertAt]...)
	result = append(result, replacements...)
	result = append(result, filtered[insertAt:]...)
	return result
}

func runtimeControlMessage(content string) model.Message {
	message := model.TextMessage(model.RoleUser, content)
	message.Metadata = map[string]string{
		contextpkg.ContextSectionMetadataKey: contextpkg.ContextSectionRuntime,
		contextpkg.RuntimeControlMetadataKey: "true",
	}
	return message
}

func effectivePlanningPolicy(policy string) string {
	switch strings.ToLower(strings.TrimSpace(policy)) {
	case "required":
		return "required"
	case "disabled":
		return "disabled"
	default:
		return "auto"
	}
}

func withoutToolSchema(schemas []model.ToolSchema, name string) []model.ToolSchema {
	filtered := make([]model.ToolSchema, 0, len(schemas))
	for _, schema := range schemas {
		if schema.Name != name {
			filtered = append(filtered, schema)
		}
	}
	return filtered
}

func withoutToolSchemas(schemas []model.ToolSchema, names ...string) []model.ToolSchema {
	for _, name := range names {
		schemas = withoutToolSchema(schemas, name)
	}
	return schemas
}

func isRuntimeControlTool(name string) bool {
	switch name {
	case "update_plan", "update_plan_step", "revise_verification", "ask_user":
		return true
	default:
		return false
	}
}

func hasSubstantiveToolSchema(schemas []model.ToolSchema) bool {
	for _, schema := range schemas {
		if !isRuntimeControlTool(schema.Name) {
			return true
		}
	}
	return false
}

func onlyToolSchemas(schemas []model.ToolSchema, names ...string) []model.ToolSchema {
	allowed := make(map[string]struct{}, len(names))
	for _, name := range names {
		allowed[name] = struct{}{}
	}
	filtered := make([]model.ToolSchema, 0, len(schemas))
	for _, schema := range schemas {
		if _, ok := allowed[schema.Name]; ok {
			filtered = append(filtered, schema)
		}
	}
	return filtered
}

func applyProgressReviewToolProjection(messages []model.Message, projected, all []model.ToolSchema) []model.ToolSchema {
	review := loadExecutionLedger(messages).ProgressReview
	if review == nil {
		return projected
	}
	switch review.Action {
	case "replan":
		projected = ensureToolSchemas(projected, all, "update_plan")
		return onlyToolSchemas(projected, "update_plan")
	case "review":
		projected = ensureToolSchemas(projected, all, "delegate_agent")
		if containsToolSchema(projected, "delegate_agent") {
			return onlyToolSchemas(projected, "delegate_agent")
		}
		// A version without Reviewer capability cannot satisfy the review
		// boundary. Deterministically fall back to changing the Plan instead of
		// reopening unconstrained local edits.
		projected = ensureToolSchemas(projected, all, "update_plan")
		return onlyToolSchemas(projected, "update_plan")
	case "ask_user":
		projected = ensureToolSchemas(projected, all, "ask_user")
		return onlyToolSchemas(projected, "ask_user")
	case "retry", "execute":
		return withoutToolSchemas(projected, "list_files", "search_files")
	default:
		return projected
	}
}

func ensureToolSchemas(projected, all []model.ToolSchema, names ...string) []model.ToolSchema {
	present := make(map[string]struct{}, len(projected))
	for _, schema := range projected {
		present[schema.Name] = struct{}{}
	}
	for _, name := range names {
		if _, ok := present[name]; ok {
			continue
		}
		for _, schema := range all {
			if schema.Name == name {
				projected = append(projected, schema)
				present[name] = struct{}{}
				break
			}
		}
	}
	return projected
}

func containsToolSchema(schemas []model.ToolSchema, name string) bool {
	for _, schema := range schemas {
		if schema.Name == name {
			return true
		}
	}
	return false
}

// toolProjectionPayload makes the visibility/execution split observable. The
// available list is already the result of upstream Worker capability,
// AgentVersion ToolSet, version and permanent security checks. The offered
// list is only what this model request received. If a future projection layer
// temporarily narrows that list, the omitted names are reported with a reason
// instead of being silently indistinguishable from an unavailable tool.
func toolProjectionPayload(available, offered []model.ToolSchema, policy string, planExists, planOpen, continuation bool) map[string]any {
	availableNames := schemaNames(available)
	offeredNames := schemaNames(offered)
	offeredSet := make(map[string]struct{}, len(offeredNames))
	for _, name := range offeredNames {
		offeredSet[name] = struct{}{}
	}
	hidden := make(map[string]string)
	layers := make([]map[string]any, 0, len(available))
	reason := "runtime_projection"
	if policy == "disabled" {
		reason = "conversational_only"
	}
	for _, name := range availableNames {
		if _, ok := offeredSet[name]; !ok {
			hidden[name] = reason
		}
	}
	for _, schema := range available {
		_, offeredNow := offeredSet[schema.Name]
		state := "ready"
		projectionReason := "offered"
		if !offeredNow {
			state = "projected_out"
			projectionReason = reason
		}
		approvalRequired := schema.ApprovalRequired
		if !approvalRequired && (schema.Risk == "LOW_WRITE" || schema.Risk == "HIGH_RISK") {
			// Risk is a conservative capability signal. The final decision still
			// belongs to the approval policy at execution time.
			approvalRequired = true
		}
		layers = append(layers, map[string]any{
			"name":                 schema.Name,
			"capability_available": true,
			"schema_visible":       offeredNow,
			"execution_allowed":    offeredNow && schema.ExecutionAllowed,
			"approval_required":    approvalRequired,
			"runtime_state":        state,
			"reason":               projectionReason,
		})
	}
	return map[string]any{
		"layer":                           "tool_projection",
		"permanent_availability_boundary": "resolved_registry",
		"available_tools":                 availableNames,
		"offered_tools":                   offeredNames,
		"temporarily_projected_out":       hidden,
		"layers":                          layers,
		"planning_policy":                 policy,
		"plan_exists":                     planExists,
		"plan_open":                       planOpen,
		"continuation_requires_tools":     continuation,
		"execution_authority":             "executor_and_policy",
	}
}

// modelToolProjectionPayload is the compact form placed in the model context.
// The full projection is retained in TOOL_SCHEMA_PROJECTED for the UI and
// replay; the model only needs per-tool state and a few booleans. Keeping this
// payload compact is important for small-context providers where the durable
// ledger is already a protected system block.
func modelToolProjectionPayload(available, offered []model.ToolSchema, policy string, planExists, planOpen, continuation bool) map[string]any {
	offeredSet := make(map[string]struct{}, len(offered))
	for _, schema := range offered {
		offeredSet[schema.Name] = struct{}{}
	}
	layers := make([]map[string]any, 0, len(available))
	for _, schema := range available {
		_, visible := offeredSet[schema.Name]
		state := "ready"
		reason := "offered"
		if !visible {
			state = "projected_out"
			if policy == "disabled" {
				reason = "conversational_only"
			} else {
				reason = "runtime_projection"
			}
		}
		approval := schema.ApprovalRequired || schema.Risk == "LOW_WRITE" || schema.Risk == "HIGH_RISK"
		layers = append(layers, map[string]any{
			"name": schema.Name, "capability_available": true, "schema_visible": visible,
			"execution_allowed": visible && schema.ExecutionAllowed, "approval_required": approval,
			"runtime_state": state, "reason": reason,
		})
	}
	return map[string]any{
		"tools": layers, "planning_policy": policy, "plan_exists": planExists,
		"plan_open": planOpen, "continuation_requires_tools": continuation,
	}
}

func schemaNames(schemas []model.ToolSchema) []string {
	names := make([]string, 0, len(schemas))
	for _, schema := range schemas {
		names = append(names, schema.Name)
	}
	sort.Strings(names)
	return names
}

func toolChoiceForSchemas(schemas []model.ToolSchema) string {
	if len(schemas) == 0 {
		return "none"
	}
	return "auto"
}

func toolChoiceForMessages(messages []model.Message, schemas []model.ToolSchema) string {
	if len(schemas) == 1 {
		if review := loadExecutionLedger(messages).ProgressReview; review != nil {
			switch review.Action {
			case "replan":
				if schemas[0].Name == "update_plan" {
					return "required"
				}
			case "review":
				if schemas[0].Name == "delegate_agent" || schemas[0].Name == "update_plan" {
					return "required"
				}
			case "ask_user":
				if schemas[0].Name == "ask_user" {
					return "required"
				}
			}
		}
	}
	return toolChoiceForSchemas(schemas)
}

const automaticReviewerCallPrefix = "runtime-review-"

func automaticReviewerCall(review progressReview, schemas []model.ToolSchema, workflowID string, turn, step int, planContext string) (model.ToolCall, bool) {
	var delegateSchema *model.ToolSchema
	for index := range schemas {
		if schemas[index].Name == "delegate_agent" {
			delegateSchema = &schemas[index]
			break
		}
	}
	if delegateSchema == nil {
		return model.ToolCall{}, false
	}
	var schema struct {
		AllowedTargets []delegation.Target `json:"x-allowed-targets"`
	}
	if json.Unmarshal(delegateSchema.Parameters, &schema) != nil {
		return model.ToolCall{}, false
	}
	var selected delegation.Target
	for _, target := range schema.AllowedTargets {
		if !target.Available || !containsString(target.Modes, "sync") {
			continue
		}
		key := strings.ToLower(strings.TrimSpace(target.AgentKey))
		name := strings.ToLower(strings.TrimSpace(target.AgentName))
		if key == "reviewer-agent" || strings.Contains(key, "reviewer") || strings.Contains(name, "reviewer") || strings.Contains(name, "审查") {
			selected = target
			break
		}
	}
	if strings.TrimSpace(selected.AgentVersionID) == "" {
		return model.ToolCall{}, false
	}
	if len(planContext) > 4000 {
		planContext = planContext[:4000]
	}
	input := map[string]any{
		"task":                  "Independently review the current workspace and active Plan after repeated stalled execution. Return evidence-based findings and the concrete next corrective action; do not modify files.",
		"workflow_id":           workflowID,
		"reason":                review.Reason,
		"review_generation":     review.Generation,
		"tool_calls":            review.ToolCalls,
		"failed_tool_calls":     review.FailedToolCalls,
		"workspace_mutations":   review.WorkspaceMutations,
		"plan_control_calls":    review.PlanControlCalls,
		"base_plan_revision":    planRevisionFromContext(planContext),
		"current_plan_snapshot": strings.TrimSpace(planContext),
	}
	arguments, err := json.Marshal(delegation.Request{TargetAgentVersionID: selected.AgentVersionID, Mode: "sync", Input: jsonPayload(input)})
	if err != nil {
		return model.ToolCall{}, false
	}
	return model.ToolCall{
		ID:        fmt.Sprintf("%sg%d-t%d-s%d", automaticReviewerCallPrefix, review.Generation, turn, step),
		Name:      "delegate_agent",
		Arguments: arguments,
	}, true
}

func planRevisionFromContext(planContext string) int {
	start := strings.Index(planContext, "{")
	if start < 0 {
		return 0
	}
	var payload struct {
		Revision int `json:"revision"`
	}
	decoder := json.NewDecoder(strings.NewReader(planContext[start:]))
	if decoder.Decode(&payload) != nil {
		return 0
	}
	return payload.Revision
}

func automaticReviewerMessage(runID string, turn, step int, call model.ToolCall) model.Message {
	return runtimeSchedulerMessage(runID, turn, step, "review", call)
}

func runtimeSchedulerMessage(runID string, turn, step int, action string, call model.ToolCall) model.Message {
	return model.Message{
		ID:        fmt.Sprintf("%s:turn:%d:step:%d:runtime-%s", runID, turn, step, action),
		Role:      model.RoleAssistant,
		ToolCalls: []model.ToolCall{call},
		Metadata: map[string]string{
			contextpkg.ContextSectionMetadataKey: contextpkg.ContextSectionRuntime,
			contextpkg.RuntimeControlMetadataKey: "true",
			"runtime.scheduler_action":           action,
		},
	}
}

func containsAutomaticReviewerCall(messages []model.Message, calls []model.ToolCall) bool {
	for _, call := range calls {
		if isAutomaticReviewerCall(messages, call) {
			return true
		}
	}
	return false
}

func automaticReviewerFollowUp(messages []model.Message, calls []model.ToolCall) *progressReview {
	ledger := loadExecutionLedger(messages)
	decision := automaticReviewerDecision(messages, calls)
	if ledger.ProgressReview == nil || decision == nil {
		return nil
	}
	copy := *ledger.ProgressReview
	return &copy
}

func automaticReviewerDecision(messages []model.Message, calls []model.ToolCall) *reviewerDecision {
	if !containsAutomaticReviewerCall(messages, calls) {
		return nil
	}
	decision := loadExecutionLedger(messages).ReviewerDecision
	if decision == nil {
		return nil
	}
	for _, call := range calls {
		if decision.CallID == call.ID {
			copy := *decision
			copy.Findings = append(copy.Findings[:0:0], decision.Findings...)
			copy.RecommendedPlanChanges = append(copy.RecommendedPlanChanges[:0:0], decision.RecommendedPlanChanges...)
			return &copy
		}
	}
	return nil
}

func isAutomaticReviewerCall(messages []model.Message, call model.ToolCall) bool {
	if call.Name != "delegate_agent" || !strings.HasPrefix(call.ID, automaticReviewerCallPrefix) {
		return false
	}
	return runtimeSchedulerAction(messages, call) == "review"
}

func runtimeSchedulerAction(messages []model.Message, call model.ToolCall) string {
	for _, message := range messages {
		if message.Metadata == nil || message.Metadata[contextpkg.RuntimeControlMetadataKey] != "true" {
			continue
		}
		for _, recorded := range message.ToolCalls {
			if recorded.ID == call.ID && recorded.Name == call.Name && string(normalizeArguments(recorded.Arguments)) == string(normalizeArguments(call.Arguments)) {
				return message.Metadata["runtime.scheduler_action"]
			}
		}
	}
	return ""
}

func containsString(values []string, expected string) bool {
	for _, value := range values {
		if value == expected {
			return true
		}
	}
	return false
}

func completionBlockFingerprint(block CompletionBlock) string {
	raw, _ := json.Marshal(struct {
		Reason           string `json:"reason"`
		StepID           string `json:"step_id"`
		CriterionID      string `json:"criterion_id"`
		VerificationKind string `json:"verification_kind"`
		Target           string `json:"target"`
		RequiredAction   string `json:"required_action"`
	}{block.Reason, block.StepID, block.CriterionID, block.VerificationKind, block.Target, block.RequiredAction})
	digest := sha256.Sum256(raw)
	return hex.EncodeToString(digest[:])
}

func latestCompletionBlockFingerprint(messages []model.Message) string {
	if len(messages) == 0 {
		return ""
	}
	return messages[len(messages)-1].Metadata[completionBlockFingerprintMetadata]
}

func countUnresolvedCompletionBlocks(messages []model.Message, fingerprint string) int {
	count := 0
	for index := len(messages) - 1; index >= 0; index-- {
		message := messages[index]
		if message.Role == model.RoleTool && isRuntimeControlTool(message.Name) {
			break
		}
		if message.Metadata[completionBlockFingerprintMetadata] == fingerprint {
			count++
		}
	}
	return count
}

const (
	runtimePlanStart = "\n\n<RUNTIME_DURABLE_PLAN>\n"
	runtimePlanEnd   = "\n</RUNTIME_DURABLE_PLAN>"
)

func replaceRuntimePlanContext(messages []model.Message, planContext string) []model.Message {
	updated := append([]model.Message(nil), messages...)
	if len(updated) == 0 || updated[0].Role != model.RoleSystem {
		if strings.TrimSpace(planContext) == "" {
			return updated
		}
		return append([]model.Message{model.TextMessage(model.RoleSystem, runtimePlanStart+planContext+runtimePlanEnd)}, updated...)
	}
	content := updated[0].TextContent()
	if start := strings.Index(content, runtimePlanStart); start >= 0 {
		end := strings.Index(content[start+len(runtimePlanStart):], runtimePlanEnd)
		if end >= 0 {
			end += start + len(runtimePlanStart) + len(runtimePlanEnd)
			content = content[:start] + content[end:]
		}
	}
	if strings.TrimSpace(planContext) != "" {
		content += runtimePlanStart + planContext + runtimePlanEnd
	}
	updated[0] = model.TextMessage(model.RoleSystem, content)
	return updated
}

func (r *Runner) executeToolCalls(ctx context.Context, runID, workspaceID string, turn, step int, messages *[]model.Message, usage model.Usage, calls []model.ToolCall, activeCallID string, offered []model.ToolSchema, finishReason string) (bool, error) {
	toolFailed := false
	var allowed map[string]model.ToolSchema
	var offeredNames []string
	if offered != nil {
		allowed = make(map[string]model.ToolSchema, len(offered))
		offeredNames = make([]string, 0, len(offered))
		for _, schema := range offered {
			allowed[schema.Name] = schema
			offeredNames = append(offeredNames, schema.Name)
		}
	}
	for index, modelCall := range calls {
		if modelCall.ID == "" || modelCall.Name == "" {
			return toolFailed, errors.New("model returned a tool call without id or name")
		}
		rawArguments := normalizeArguments(modelCall.Arguments)
		var normalizeErr error
		if r.config.NormalizeToolArguments != nil {
			var normalized json.RawMessage
			normalized, normalizeErr = r.config.NormalizeToolArguments(modelCall.Name, rawArguments)
			if normalizeErr == nil {
				modelCall.Arguments = normalized
			} else {
				// Preserve the model's exact request for audit and recovery. A
				// normalizer error must not erase it into `{}`.
				modelCall.Arguments = rawArguments
			}
		}
		workflowID := r.config.WorkflowID
		if strings.TrimSpace(workflowID) == "" {
			workflowID = runID
		}
		toolCalledPayload := map[string]any{"name": modelCall.Name, "arguments": normalizeArguments(modelCall.Arguments), "toolset_version_id": r.config.ToolSetVersionID, "workflow_id": workflowID, "action_id": modelCall.ID, "decision_cycle": turn, "decision_source": "model"}
		schedulerAction := runtimeSchedulerAction(*messages, modelCall)
		if finishReason == "runtime_scheduler" || schedulerAction != "" {
			toolCalledPayload["decision_source"] = "runtime_scheduler"
			if schedulerAction == "" {
				schedulerAction = "runtime"
			}
			toolCalledPayload["scheduler_action"] = schedulerAction
		}
		toolPosition := event.Input{RunID: runID, WorkflowID: workflowID, Type: event.ToolCalled, Turn: turn, DecisionCycle: turn, Step: step, ActionID: modelCall.ID, CallID: modelCall.ID, Payload: jsonPayload(toolCalledPayload)}
		if activeCallID != modelCall.ID {
			if err := r.append(ctx, toolPosition); err != nil {
				return toolFailed, err
			}
		}
		if err := r.saveCheckpoint(ctx, harness.Checkpoint{RunID: runID, Turn: turn, NextStep: step, Messages: *messages, Usage: usage, PendingToolCalls: append([]model.ToolCall(nil), calls[index:]...), ActiveToolCallID: modelCall.ID}); err != nil {
			return toolFailed, err
		}
		toolStarted := time.Now()
		var toolResult tool.Result
		executeErr := normalizeErr
		if executeErr == nil {
			executeErr = validateReviewerPlanCAS(*messages, modelCall)
		}
		if executeErr == nil {
			executeErr = validateDurableCommandRecovery(*messages, modelCall)
		}
		if executeErr == nil {
			executeErr = validateActiveFileChunkRecovery(*messages, modelCall)
		}
		if executeErr == nil {
			executeErr = validateFileFocusBudget(*messages, modelCall)
		}
		if executeErr == nil && duplicateFailedTool(*messages, modelCall.Name, modelCall.Arguments) {
			executeErr = tool.NewContractError("DUPLICATE_FAILED_CALL", modelCall.Name, "", "a changed call or a repaired workspace state", "unchanged", "the same tool call already failed immediately before; change the arguments or repair the reported cause before retrying", true)
		}
		var offeredSchema *model.ToolSchema
		if executeErr != nil && allowed != nil {
			if schema, ok := allowed[modelCall.Name]; ok {
				offeredSchema = &schema
			}
		}
		if allowed != nil && executeErr == nil {
			schema, ok := allowed[modelCall.Name]
			if !ok {
				executeErr = tool.NewContractErrorWithRepair("TOOL_NOT_OFFERED", modelCall.Name, "", strings.Join(offeredNames, ", "), modelCall.Name, fmt.Sprintf("tool %q was not offered for this step; choose one of: %s", modelCall.Name, strings.Join(offeredNames, ", ")), "Choose a tool from the offered schema. If a capability is waiting for approval or a plan transition, use the structured observation and wait for the next checkpoint.", nil, false)
			} else if compiled, err := contract.Compile(schema.Parameters); err != nil {
				offeredSchema = &schema
				executeErr = tool.NewContractErrorWithRepair("TOOL_SCHEMA_INVALID", modelCall.Name, "", "a valid offered JSON Schema", string(schema.Parameters), fmt.Sprintf("compile offered schema for %q: %v", modelCall.Name, err), "The runtime offered an invalid schema; do not retry unchanged. Continue only after the next corrected Tool Schema projection.", schema.Parameters, false)
			} else if err := compiled.ValidateJSON(normalizeArguments(modelCall.Arguments)); err != nil {
				offeredSchema = &schema
				executeErr = tool.NewContractErrorWithRepair("TOOL_SCHEMA_INVALID", modelCall.Name, "", string(schema.Parameters), string(modelCall.Arguments), fmt.Sprintf("tool %q arguments violate this step's offered schema: %v", modelCall.Name, err), "Correct only the indicated field types and required properties, then issue one changed structured call. Arrays and objects must remain JSON values, not encoded strings.", schema.Parameters, true)
			}
		}
		if executeErr == nil {
			toolCall := tool.Call{
				RunID: runID, WorkflowID: workflowID, WorkspaceID: workspaceID,
				Turn: turn, DecisionCycle: turn, Step: step,
				ID: modelCall.ID, ActionID: modelCall.ID, Name: modelCall.Name,
				Arguments: normalizeArguments(modelCall.Arguments),
			}
			if cached, ok := cachedReadObservation(*messages, modelCall); ok {
				toolResult = cached
			} else {
				toolResult, executeErr = r.tools.Execute(ctx, toolCall)
			}
			if executeErr == nil && len(toolResult.Content) > tool.InlineResultLimit && r.config.ToolResultOffloader != nil {
				if toolResult.Meta == nil {
					toolResult.Meta = make(map[string]string)
				}
				if strings.TrimSpace(toolResult.Meta["content_artifact_id"]) == "" {
					ref, offloadErr := r.config.ToolResultOffloader(ctx, toolCall, toolResult.Content)
					if offloadErr == nil && strings.TrimSpace(ref.ID) != "" {
						toolResult.Meta["content_artifact_id"] = ref.ID
						if ref.SHA256 != "" {
							toolResult.Meta["content_sha256"] = ref.SHA256
						}
						toolResult.Meta["content_bytes"] = fmt.Sprintf("%d", len(toolResult.Content))
						toolResult.ModelContent = tool.BuildResultReceipt(toolResult.Content, ref.ID)
					} else if offloadErr != nil {
						toolResult.Meta["content_offload_error"] = offloadErr.Error()
					}
				}
			}
		}
		if errors.Is(executeErr, approval.ErrRequired) || errors.Is(executeErr, delegation.ErrPending) || errors.Is(executeErr, interaction.ErrInputRequired) {
			return toolFailed, executeErr
		}
		if executeErr != nil {
			toolResult = structuredToolFailure(modelCall.Name, executeErr, offeredNames, offeredSchema, finishReason, normalizeErr != nil)
			toolPosition.Type = event.ToolFailed
			toolFailed = true
		} else if toolResult.IsError {
			// Handler-produced failures (for example a non-zero workspace
			// process or an MCP isError response) must expose the same stable
			// error contract as Go errors before the event and next model
			// message are persisted.
			toolResult = tool.NormalizeResultFailure(tool.Call{Name: modelCall.Name, ID: modelCall.ID, Arguments: normalizeArguments(modelCall.Arguments)}, toolResult)
			toolPosition.Type = event.ToolFailed
			toolFailed = true
		} else {
			toolPosition.Type = event.ToolCompleted
		}
		// Non-Postgres executors may not have a persistence wrapper, so apply the
		// same bounded fallback projection here. The complete result remains
		// available to the durable execution ledger and persistence wrapper.
		toolResult.ApplyStoredModelProjection()
		modelToolResult := toolResult.ModelVisible()
		toolPayload := map[string]any{"name": modelCall.Name, "arguments": normalizeArguments(modelCall.Arguments), "result": modelToolResult, "latency_ms": time.Since(toolStarted).Milliseconds(), "toolset_version_id": r.config.ToolSetVersionID, "workflow_id": workflowID, "action_id": modelCall.ID, "decision_cycle": turn}
		if toolResult.Meta != nil {
			if planNodeID := strings.TrimSpace(toolResult.Meta["plan_node_id"]); planNodeID != "" {
				toolPayload["plan_node_id"] = planNodeID
			}
		}
		toolPosition.Payload = jsonPayload(toolPayload)
		if err := r.append(ctx, toolPosition); err != nil {
			return toolFailed, err
		}
		if toolPosition.Type == event.ToolCompleted && r.config.PlanProgress != nil {
			planNodeID := ""
			if toolResult.Meta != nil {
				planNodeID = strings.TrimSpace(toolResult.Meta["plan_node_id"])
			}
			if planNodeID != "" {
				if err := r.config.PlanProgress(ctx, tool.Call{
					RunID: runID, WorkflowID: workflowID, WorkspaceID: workspaceID,
					Turn: turn, DecisionCycle: turn, Step: step,
					ID: modelCall.ID, ActionID: modelCall.ID, Name: modelCall.Name,
					Arguments:  normalizeArguments(modelCall.Arguments),
					PlanStepID: planNodeID, PlanNodeID: planNodeID,
				}); err != nil {
					return toolFailed, fmt.Errorf("reconcile durable plan after tool %s: %w", modelCall.Name, err)
				}
			}
		}
		toolMessage := model.Message{ID: fmt.Sprintf("%s:tool:%s", runID, modelCall.ID), Role: model.RoleTool, Name: modelCall.Name, ToolCallID: modelCall.ID, Content: string(modelToolResult.Content), Parts: []model.ContentPart{{Type: model.ContentJSON, JSON: modelToolResult.Content}}}
		if toolResult.IsError || executeErr != nil {
			toolMessage.Metadata = map[string]string{"tool_args_hash": toolArgumentsHash(modelCall.Arguments), "tool_failed": "true"}
		}
		*messages = append(*messages, toolMessage)
		*messages = updateExecutionLedger(*messages, modelCall, toolResult)
		if !toolFailed {
			// A successful call proves that the current call satisfied this
			// Tool's immediate recovery contract. Keep corrections for other
			// Tools, but do not carry a stale restriction indefinitely.
			*messages = clearToolFailureRecoveryMessagesForTool(*messages, modelCall.Name)
		}
		if toolFailed && r.config.MemoryFailureRecovery != nil {
			failure := parseToolFailure(modelCall.Name, modelCall.Arguments, toolResult)
			fingerprint := failureFingerprint(failure)
			if r.failureRecoverySeen[fingerprint] == 0 {
				r.failureRecoverySeen[fingerprint]++
				if recovered, recoveryErr := r.config.MemoryFailureRecovery(ctx, failure); recoveryErr == nil {
					*messages = replaceToolFailureRecoveryMessages(*messages, recovered)
					if r.config.MergeMemoryContextState != nil {
						if state := r.config.MergeMemoryContextState(r.config.ContextState); len(state) != 0 {
							r.config.ContextState = append(json.RawMessage(nil), state...)
						}
					}
				}
			}
		}
		if err := r.saveCheckpoint(ctx, harness.Checkpoint{RunID: runID, Turn: turn, NextStep: step, Messages: *messages, Usage: usage, PendingToolCalls: append([]model.ToolCall(nil), calls[index+1:]...)}); err != nil {
			return toolFailed, err
		}
		activeCallID = ""
	}
	return toolFailed, nil
}

func validateReviewerPlanCAS(messages []model.Message, call model.ToolCall) error {
	if call.Name != "update_plan" && call.Name != "revise_verification" {
		return nil
	}
	ledger := loadExecutionLedger(messages)
	if ledger.ProgressReview == nil || ledger.ProgressReview.Action != "replan" || ledger.ReviewerDecision == nil || ledger.ReviewerDecision.BasePlanRevision <= 0 {
		return nil
	}
	var mutation struct {
		BaseRevision *int `json:"base_revision"`
	}
	if json.Unmarshal(call.Arguments, &mutation) != nil {
		return nil // The exact Tool Schema reports malformed JSON/field types.
	}
	expected := ledger.ReviewerDecision.BasePlanRevision
	if mutation.BaseRevision != nil && *mutation.BaseRevision == expected {
		return nil
	}
	actual := "missing"
	if mutation.BaseRevision != nil {
		actual = fmt.Sprintf("%d", *mutation.BaseRevision)
	}
	return tool.NewContractErrorWithRepair(
		"REVIEW_PLAN_REVISION_MISMATCH", call.Name, "/base_revision", fmt.Sprintf("%d", expected), actual,
		"Reviewer recommendations are fenced to the Plan revision they inspected",
		fmt.Sprintf("Retry once with base_revision=%d. If PostgreSQL reports PLAN_REVISION_CONFLICT, the Runtime will discard the stale Reviewer recommendation and rebuild from the latest durable Plan.", expected),
		jsonPayload(map[string]any{"base_revision": expected}), true,
	)
}

func validateDurableCommandRecovery(messages []model.Message, call model.ToolCall) error {
	if call.Name != "run_command" {
		return nil
	}
	ledger := loadExecutionLedger(messages)
	state, blocked := ledger.CommandFailures[toolArgumentsHash(call.Arguments)]
	if !blocked || state.WorkspaceRevision != ledger.WorkspaceRevision {
		return nil
	}
	diagnostic := strings.TrimSpace(state.Diagnostic)
	correction := "Use the prior diagnostic, mutate the relevant workspace file or change command arguments, then retry once."
	if diagnostic != "" {
		correction += " Prior diagnostic: " + diagnostic
	}
	return tool.NewContractErrorWithRepair(
		"DETERMINISTIC_RETRY_BLOCKED", call.Name, "", "changed arguments or workspace revision", "unchanged",
		"the same deterministic run_command failure cannot be retried against an unchanged workspace",
		correction, nil, true,
	)
}

// replaceToolFailureRecoveryMessages keeps a small active set of recovery
// contracts. A failure of read_file must not erase a still-unresolved
// write_file limit; conversely, retaining every historical correction would
// turn recovery advice into protected-context growth. A successful call clears
// its own entry via clearToolFailureRecoveryMessagesForTool.
func replaceToolFailureRecoveryMessages(messages, replacements []model.Message) []model.Message {
	updated := make([]model.Message, 0, len(messages)+len(replacements))
	active := make([]model.Message, 0, maxActiveToolFailureRecoveryContracts+1)
	replacedTools := make(map[string]struct{})
	existingMemoryIDs := make(map[string]struct{})
	for _, message := range replacements {
		if message.Metadata == nil || message.Metadata[ToolFailureRecoveryMetadata] != "true" {
			continue
		}
		if toolName := strings.TrimSpace(message.Metadata[ToolFailureRecoveryToolMetadata]); toolName != "" {
			replacedTools[toolName] = struct{}{}
		}
	}
	for _, message := range messages {
		if message.Metadata != nil && message.Metadata[ToolFailureMemoryMetadata] == "true" {
			// The next failure lookup replaces the prior one. Complete retrieval
			// history remains in MEMORY_RETRIEVED/audit events.
			continue
		}
		if message.Metadata != nil && message.Metadata[ToolFailureRecoveryMetadata] == "true" {
			toolName := strings.TrimSpace(message.Metadata[ToolFailureRecoveryToolMetadata])
			// Legacy recovery reminders lacked a Tool identity. Drop them on
			// replacement because their lifetime cannot be safely determined.
			if toolName == "" {
				continue
			}
			if _, replaced := replacedTools[toolName]; replaced {
				continue
			}
			active = append(active, message)
			continue
		}
		for _, id := range metadataMemoryIDs(message) {
			existingMemoryIDs[id] = struct{}{}
		}
		updated = append(updated, message)
	}
	for _, message := range replacements {
		if strings.TrimSpace(message.TextContent()) != "" {
			if message.Metadata != nil && message.Metadata[ToolFailureRecoveryMetadata] == "true" {
				active = append(active, message)
			} else if message.Metadata != nil && message.Metadata[ToolFailureMemoryMetadata] == "true" {
				ids := metadataMemoryIDs(message)
				allSeen := len(ids) != 0
				for _, id := range ids {
					if _, seen := existingMemoryIDs[id]; !seen {
						allSeen = false
						break
					}
				}
				if !allSeen {
					updated = append(updated, message)
				}
			} else {
				updated = append(updated, message)
			}
		}
	}
	if len(active) > maxActiveToolFailureRecoveryContracts {
		active = active[len(active)-maxActiveToolFailureRecoveryContracts:]
	}
	return append(updated, active...)
}

func metadataMemoryIDs(message model.Message) []string {
	if message.Metadata == nil {
		return nil
	}
	raw := strings.TrimSpace(message.Metadata[contextpkg.MemoryIDsMetadataKey])
	if raw == "" {
		return nil
	}
	values := strings.Split(raw, ",")
	ids := make([]string, 0, len(values))
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			ids = append(ids, value)
		}
	}
	return ids
}

func clearToolFailureRecoveryMessagesForTool(messages []model.Message, toolName string) []model.Message {
	toolName = strings.TrimSpace(toolName)
	if toolName == "" {
		return messages
	}
	updated := make([]model.Message, 0, len(messages))
	for _, message := range messages {
		if message.Metadata != nil && strings.TrimSpace(message.Metadata[ToolFailureRecoveryToolMetadata]) == toolName {
			if message.Metadata[ToolFailureRecoveryMetadata] == "true" || message.Metadata[ToolFailureMemoryMetadata] == "true" {
				continue
			}
		}
		updated = append(updated, message)
	}
	return updated
}

// validateActiveFileChunkRecovery turns an oversized-file correction into a
// bounded, current runtime contract. This closes the gap where a model could
// acknowledge "split the file" and immediately emit another 8K-sized body.
// Normal calls still use the public 8192-character Schema ceiling; only the
// retry following an observed boundary failure is constrained to the safer
// chunk size.
func validateActiveFileChunkRecovery(messages []model.Message, call model.ToolCall) error {
	if call.Name != "write_file" && call.Name != "append_file" {
		return nil
	}
	var recovery model.Message
	for index := len(messages) - 1; index >= 0; index-- {
		candidate := messages[index]
		if candidate.Metadata == nil || candidate.Metadata[FileChunkRecoveryMetadata] != "true" {
			continue
		}
		if strings.TrimSpace(candidate.Metadata[ToolFailureRecoveryToolMetadata]) != call.Name {
			continue
		}
		recovery = candidate
		break
	}
	if recovery.Metadata == nil {
		return nil
	}
	var input struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(normalizeArguments(call.Arguments), &input); err != nil {
		// The normalizer/schema layer owns malformed JSON diagnostics.
		return nil
	}
	expectedPath := strings.TrimSpace(recovery.Metadata[FileChunkRecoveryPathMetadata])
	if expectedPath != "" && strings.TrimSpace(input.Path) != expectedPath {
		return tool.NewContractErrorWithRepair(
			"FILE_CHUNK_RECOVERY_REQUIRED", call.Name, "/path", expectedPath, input.Path,
			"file chunk recovery changed the rejected target path",
			fmt.Sprintf("Keep path=%q from the rejected call. Submit a first coherent chunk with path first and content at most %d characters; preserve the remaining implementation for append_file.", expectedPath, FileChunkRecoverySafeChars),
			nil, true,
		)
	}
	contentChars := utf8.RuneCountInString(input.Content)
	if contentChars > FileChunkRecoverySafeChars {
		return tool.NewContractErrorWithRepair(
			"FILE_CHUNK_RECOVERY_REQUIRED", call.Name, "/content", fmt.Sprintf("at most %d characters during recovery", FileChunkRecoverySafeChars), fmt.Sprintf("%d characters", contentChars),
			"file chunk recovery payload is still too close to the model and schema boundary",
			fmt.Sprintf("Keep path first and reduce content to at most %d characters. Write one coherent chunk without deleting requirements, then continue the same file with append_file instead of compacting the whole module.", FileChunkRecoverySafeChars),
			nil, true,
		)
	}
	return nil
}

func parseToolFailure(name string, arguments json.RawMessage, result tool.Result) ToolFailure {
	failure := ToolFailure{ToolName: name, Error: result.Error, Arguments: append(json.RawMessage(nil), arguments...)}
	if result.Meta != nil {
		failure.ErrorCode = strings.TrimSpace(result.Meta["error_code"])
	}
	var envelope struct {
		ErrorCode   string `json:"error_code"`
		FailureKind string `json:"failure_kind"`
		Correction  string `json:"correction"`
		Error       string `json:"error"`
		Diagnostic  string `json:"diagnostic"`
		Stdout      string `json:"stdout"`
		Stderr      string `json:"stderr"`
		StdoutTail  string `json:"stdout_tail"`
		StderrTail  string `json:"stderr_tail"`
		ExitCode    *int   `json:"exit_code"`
		TimedOut    bool   `json:"timed_out"`
	}
	if json.Unmarshal(result.Content, &envelope) == nil {
		if failure.ErrorCode == "" {
			failure.ErrorCode = strings.TrimSpace(envelope.ErrorCode)
		}
		failure.Correction = strings.TrimSpace(envelope.Correction)
		failure.FailureKind = strings.TrimSpace(envelope.FailureKind)
		failure.Diagnostic = strings.TrimSpace(envelope.Diagnostic)
		failure.StdoutTail = boundedTextTail(firstNonEmpty(envelope.StdoutTail, envelope.Stdout), 1200)
		failure.StderrTail = boundedTextTail(firstNonEmpty(envelope.StderrTail, envelope.Stderr), 1200)
		failure.ExitCode = envelope.ExitCode
		failure.TimedOut = envelope.TimedOut
		failure.HasStdout = strings.TrimSpace(envelope.Stdout) != "" || failure.StdoutTail != ""
		failure.HasStderr = strings.TrimSpace(envelope.Stderr) != "" || failure.StderrTail != ""
		if failure.Diagnostic == "" {
			failure.Diagnostic = lastNonEmptyDiagnostic(firstNonEmpty(failure.StderrTail, failure.StdoutTail), 500)
		}
		if failure.Error == "" {
			failure.Error = strings.TrimSpace(envelope.Error)
		}
	}
	if failure.ErrorCode == "" {
		failure.ErrorCode = "tool_execution_failed"
	}
	if failure.FailureKind == "" {
		failure.FailureKind = failureKindFromCode(failure.ErrorCode)
	}
	if failure.Correction == "" {
		failure.Correction = "Inspect the error and retry with corrected arguments."
	}
	return failure
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func boundedTextTail(value string, limit int) string {
	runes := []rune(strings.TrimSpace(value))
	if len(runes) <= limit {
		return string(runes)
	}
	return "…" + string(runes[len(runes)-limit:])
}

func lastNonEmptyDiagnostic(value string, limit int) string {
	lines := strings.Split(strings.TrimSpace(value), "\n")
	for index := len(lines) - 1; index >= 0; index-- {
		line := strings.TrimSpace(lines[index])
		if line == "" {
			continue
		}
		runes := []rune(line)
		if len(runes) > limit {
			return string(runes[:limit]) + "…"
		}
		return line
	}
	return ""
}

func failureKindFromCode(code string) string {
	return tool.FailureKind(code)
}

func failureFingerprint(failure ToolFailure) string {
	digest := sha256.Sum256([]byte(strings.Join([]string{failure.ToolName, failure.ErrorCode, failure.FailureKind, failure.Diagnostic, failure.Correction, toolArgumentsHash(failure.Arguments)}, "\n")))
	return hex.EncodeToString(digest[:])
}

func structuredToolFailure(name string, cause error, offered []string, schema *model.ToolSchema, finishReason string, normalization bool) tool.Result {
	message := cause.Error()
	code := "tool_execution_failed"
	correction := "Inspect the error and retry the same tool with corrected arguments."
	retryable := true
	contractDetails := map[string]any{}
	contractError := false
	if targetErr, ok := delegation.AsTargetNotAllowedError(cause); ok {
		code = "DELEGATION_TARGET_NOT_ALLOWED"
		correction = "Choose one allowed target_agent_version_id (or its unique target_agent alias) and an allowed mode from allowed_targets. Do not repeat the rejected target or mode."
		retryable = len(targetErr.AllowedTargets) > 0
		contractDetails["requested_target"] = targetErr.RequestedTarget
		contractDetails["requested_mode"] = targetErr.RequestedMode
		contractDetails["allowed_targets"] = targetErr.AllowedTargets
	} else if typed, ok := tool.AsContractError(cause); ok {
		contractError = true
		code = typed.Code
		correction = typed.Correction
		if correction == "" {
			correction = "Repair the indicated field or execution condition, then issue one changed call. Do not repeat an unchanged payload."
		}
		correction = toolSchemaFailureCorrection(name, message, correction)
		retryable = typed.Retryable
		contractDetails["path"] = typed.Path
		contractDetails["expected"] = typed.Expected
		contractDetails["actual"] = typed.Actual
		if len(typed.RetryTemplate) != 0 {
			contractDetails["retry_template"] = json.RawMessage(typed.RetryTemplate)
		}
	}
	if normalization {
		code = "tool_arguments_normalization_failed"
		correction = "Emit one complete JSON object matching the tool schema. Do not encode arrays or objects as strings; keep the payload compact."
		if name == "update_plan" {
			correction = "Re-emit update_plan as one complete compact JSON object. steps must be a JSON array, not a quoted JSON string; use 3-8 short steps and one minimal acceptance criterion per step. Do not repeat a truncated payload."
		}
	} else if strings.Contains(message, "was not offered") {
		code = "tool_not_offered"
		correction = "Use one of the offered tools. If the required capability was approved or the active Todo changed, wait for the next checkpoint and re-evaluate the offered tools."
	} else if !contractError && (strings.Contains(message, "schema") || strings.Contains(message, "expected array")) {
		code = "tool_schema_invalid"
		correction = "Correct the indicated field type and retry. Arrays must be JSON arrays, not JSON-encoded strings."
	} else if strings.Contains(message, "no such file or directory") || strings.Contains(message, "parent") {
		code = "PARENT_DIRECTORY_MISSING"
		if offeredTool(offered, "create_directory") {
			correction = "The target parent directory does not exist. Call the offered create_directory tool for the relative parent path, then retry the original write or edit with the same file path."
		} else {
			correction = "The target parent directory does not exist, and create_directory is not offered in this step. Do not call an unavailable tool; use an existing Run-workspace parent or choose one of the offered file tools that can create the target path."
		}
	} else if strings.Contains(message, "not allowed") || strings.Contains(message, "denied") || strings.Contains(message, "rejected") {
		code = "SANDBOX_POLICY_REJECTED"
		correction = "The Sandbox policy rejected this operation. Use the allowed command/file profile shown in the Tool Schema; do not retry the same prohibited operation."
		retryable = false
	}
	if strings.EqualFold(strings.TrimSpace(finishReason), "length") {
		code = "model_output_truncated"
		correction = "The model response reached max_tokens. Send a shorter complete tool call; reduce Plan steps/criteria and never repeat a truncated payload."
	}
	failureKind := tool.FailureKind(code)
	correction, deterministicTemplate := tool.RecoveryContract(name, code, correction)
	payload := map[string]any{"error": message, "error_code": code, "failure_kind": failureKind, "retryable": retryable, "correction": correction}
	for key, value := range contractDetails {
		if value != "" {
			payload[key] = value
		}
	}
	if len(offered) > 0 {
		payload["offered_tools"] = offered
	}
	if schema != nil && len(schema.Parameters) <= 4096 {
		payload["expected_schema"] = json.RawMessage(schema.Parameters)
	}
	if contract := toolFailureContract(name); len(contract) != 0 {
		payload["tool_contract"] = contract
	}
	if _, ok := payload["retry_template"]; !ok {
		// Keep the envelope shape stable even when no safe changed-call
		// template exists. An empty object is diagnostic, never an executable
		// retry request.
		payload["retry_template"] = deterministicTemplate
	}
	content := jsonPayload(payload)
	return tool.Result{Content: content, IsError: true, Error: message, Meta: map[string]string{"error_code": code, "failure_kind": failureKind, "retryable": fmt.Sprintf("%t", retryable), "correction": correction, "tool": name}}
}

func toolSchemaFailureCorrection(name, message, fallback string) string {
	lower := strings.ToLower(message)
	switch name {
	case "write_file", "append_file":
		if strings.Contains(lower, "maxlength") || strings.Contains(lower, "maximum") || strings.Contains(lower, "8192") {
			return fmt.Sprintf("The content field has a hard limit of 8192 characters. Keep path and content as native JSON fields; recover with a coherent chunk of at most %d characters, then use append_file for subsequent chunks. Preserve requirements; do not compact the implementation merely to fit or resend the oversized content.", FileChunkRecoverySafeChars)
		}
		if strings.Contains(lower, "missing properties") && strings.Contains(lower, "path") {
			return "Include the required path field first, followed by content. Both are required native JSON fields; do not retry with path omitted."
		}
		if strings.Contains(lower, "missing properties") && strings.Contains(lower, "content") {
			return "Include the required content field after path. Content must be a non-empty string of at most 8192 characters; do not retry without content."
		}
	case "edit_file":
		if strings.Contains(lower, "missing properties") {
			return "edit_file requires path, old_text, and new_text. Use a precise unique old_text replacement; do not send write_file-style content."
		}
	}
	if strings.Contains(lower, "duplicate read_file range") {
		return "Do not repeat the same read_file path and line range. Reuse the previous result or request a different start_line/line_count."
	}
	if strings.Contains(lower, "restricted python") || strings.Contains(lower, "python -c") {
		return "Do not use python3 -c. Write a short relative workspace script with write_file, then run it with run_command using command=python3 and args containing the script path."
	}
	return fallback
}

func toolFailureContract(name string) map[string]any {
	switch name {
	case "write_file":
		return map[string]any{"required": []string{"path", "content"}, "content_max_length": 8192, "recovery_chunk_max_length": FileChunkRecoverySafeChars, "large_file_strategy": "write_file first coherent chunk, append_file continuation chunks; preserve requirements rather than compacting to fit"}
	case "append_file":
		return map[string]any{"required": []string{"path", "content"}, "content_max_length": 8192, "recovery_chunk_max_length": FileChunkRecoverySafeChars, "purpose": "continue an existing file with a coherent chunk"}
	case "edit_file":
		return map[string]any{"required": []string{"path", "old_text", "new_text"}, "purpose": "unique precise replacement"}
	case "read_file":
		return map[string]any{"required": []string{"path"}, "no_duplicate_range": true}
	case "run_command":
		return map[string]any{"required": []string{"command", "args"}, "command": "python3", "timeout_seconds_max": 30, "python_c_inline": false}
	case "update_plan":
		return map[string]any{"required": []string{"goal", "steps"}, "steps_min": 1, "steps_max": 8, "verification": "must exactly match the offered verification schema; no platform-owned fields"}
	case "revise_verification":
		return map[string]any{"required": []string{"step_id", "criterion_id", "action", "reason"}, "action": []string{"replace", "skip_advisory"}, "verification_allowed_fields": []string{"kind", "target", "match", "tool", "arguments", "assertions"}, "forbidden": []string{"tool_hints", "state", "evidence", "receipt"}}
	default:
		return nil
	}
}

func offeredTool(offered []string, name string) bool {
	for _, candidate := range offered {
		if candidate == name {
			return true
		}
	}
	return false
}

func toolArgumentsHash(raw json.RawMessage) string {
	digest := sha256.Sum256(normalizeArguments(raw))
	return hex.EncodeToString(digest[:])
}

func duplicateFailedTool(messages []model.Message, name string, args json.RawMessage) bool {
	hash := toolArgumentsHash(args)
	for index := len(messages) - 1; index >= 0; index-- {
		message := messages[index]
		if message.Role == model.RoleUser {
			// A runtime correction, user answer, or approval creates a new
			// decision boundary; the same call is no longer an immediate retry.
			return false
		}
		if message.Role == model.RoleAssistant && len(message.ToolCalls) == 0 {
			return false
		}
		if message.Role != model.RoleTool {
			continue
		}
		if message.Name != name || message.Metadata == nil || message.Metadata["tool_failed"] != "true" {
			return false
		}
		return message.Metadata["tool_args_hash"] == hash
	}
	return false
}

func refreshContextManifest(base json.RawMessage, messages []model.Message, compactionGeneration int) json.RawMessage {
	if len(base) == 0 {
		return nil
	}
	var manifest contextpkg.Manifest
	if json.Unmarshal(base, &manifest) != nil {
		return base
	}
	encoded, err := json.Marshal(messages)
	if err != nil {
		return base
	}
	digest := sha256.Sum256(encoded)
	manifest.ContextHash = hex.EncodeToString(digest[:])
	manifest.MessageCount = len(messages)
	manifest.InputTokens = 0
	manifest.CompactionGeneration += compactionGeneration
	for _, message := range messages {
		manifest.InputTokens += contextpkg.EstimateTokens(message)
	}
	refreshed, err := json.Marshal(manifest)
	if err != nil {
		return base
	}
	return refreshed
}

func estimateMessages(messages []model.Message) int {
	total := 0
	for _, message := range messages {
		total += contextpkg.EstimateTokens(message)
	}
	return total
}

func toolSchemasDigest(schemas []model.ToolSchema) string {
	encoded, err := json.Marshal(schemas)
	if err != nil {
		return ""
	}
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:])
}

// projectRuntimeAwareMessages keeps the durable task and conversation history
// in Collapse, but removes runtime projections first. Those projections are
// rebuilt from the current Plan, execution ledger, and offered tools after the
// historical view has been compacted. The reserved runtime cost is deducted
// from the collapse budget so the final model-facing request still fits.
func projectRuntimeAwareMessages(
	ctx context.Context,
	modelMessages []model.Message,
	budget int,
	state contextpkg.CollapseState,
	options contextpkg.ProjectionOptions,
	fullLedger json.RawMessage,
	planContext string,
	runtimeToolPayload map[string]any,
	injectTools bool,
) ([]model.Message, contextpkg.CollapseState, contextpkg.ProjectionReport, error) {
	beforeTokens := estimateMessages(modelMessages)
	collapseMessages := contextpkg.StripRebuildableRuntimeBlocks(modelMessages)
	project := func(includeToolProjection bool) ([]model.Message, contextpkg.CollapseState, contextpkg.ProjectionReport, error) {
		// Derive the runtime reserve from a clean rebuild, rather than from the
		// current system message. This prevents stale/duplicated runtime blocks
		// in a recovered transcript from consuming the reserve twice.
		cleanCost := estimateMessages(collapseMessages)
		runtimeView := rebuildRuntimeProjection(collapseMessages, fullLedger, planContext, runtimeToolPayload, includeToolProjection)
		runtimeTokens := estimateMessages(runtimeView) - cleanCost
		if runtimeTokens < 0 {
			runtimeTokens = 0
		}
		effectiveBudget := budget - runtimeTokens
		if effectiveBudget <= 0 {
			return nil, state, contextpkg.ProjectionReport{BeforeTokens: beforeTokens}, contextpkg.ErrProjectionBudgetTooSmall
		}
		projected, nextState, report, err := contextpkg.ProjectMessagesWithOptions(ctx, collapseMessages, effectiveBudget, state, options)
		if err != nil {
			report.BeforeTokens = beforeTokens
			return nil, state, report, err
		}
		projected = rebuildRuntimeProjection(projected, fullLedger, planContext, runtimeToolPayload, includeToolProjection)
		report.BeforeTokens = beforeTokens
		report.AfterTokens = estimateMessages(projected)
		if report.AfterTokens > budget {
			return nil, state, report, contextpkg.ErrProjectionBudgetTooSmall
		}
		return projected, nextState, report, nil
	}

	projected, nextState, report, err := project(injectTools)
	if err == nil || !injectTools {
		return projected, nextState, report, err
	}
	// The actual function schemas are already sent in model.Request.Tools. The
	// verbose capability-state block is useful but reconstructable, so it is the
	// first runtime layer dropped when exact task/failure messages need space.
	return project(false)
}

func rebuildRuntimeProjection(messages []model.Message, fullLedger json.RawMessage, planContext string, runtimeToolPayload map[string]any, injectTools bool) []model.Message {
	projected := restoreExecutionLedger(messages, fullLedger)
	projected = projectExecutionLedgerForModel(projected)
	if strings.TrimSpace(planContext) != "" {
		projected = replaceRuntimePlanContext(projected, planContext)
	}
	if injectTools {
		projected = injectRuntimeToolProjection(projected, runtimeToolPayload)
	}
	return projected
}

func modelErrorKind(err error) string {
	message := strings.ToLower(err.Error())
	switch {
	case strings.Contains(message, "connection refused"):
		return "connection_refused"
	case strings.Contains(message, "deadline exceeded"), strings.Contains(message, "timeout"):
		return "timeout"
	case strings.Contains(message, "unauthorized"), strings.Contains(message, "status 401"):
		return "authentication"
	case strings.Contains(message, "not found"), strings.Contains(message, "status 404"):
		return "endpoint_or_model_not_found"
	case strings.Contains(message, "maximum context length"), strings.Contains(message, "context length"), strings.Contains(message, "context budget"), strings.Contains(message, "input_tokens"):
		return "context_window_exceeded"
	case strings.Contains(message, "context canceled"):
		return "cancelled"
	default:
		return "provider_error"
	}
}

// mergeSystemMessages normalizes provider-neutral Prompt and Skill instruction
// blocks for chat templates that accept exactly one leading system message.
// Source resources remain independently versioned; only the model wire request
// is merged, in deterministic input order.
func mergeSystemMessages(messages []model.Message) []model.Message {
	systems := make([]string, 0, 2)
	normalized := make([]model.Message, 0, len(messages))
	for _, message := range messages {
		if message.Role == model.RoleSystem {
			if content := strings.TrimSpace(message.TextContent()); content != "" {
				systems = append(systems, content)
			}
			continue
		}
		normalized = append(normalized, message)
	}
	if len(systems) == 0 {
		return normalized
	}
	return append([]model.Message{model.TextMessage(model.RoleSystem, strings.Join(systems, "\n\n"))}, normalized...)
}

const runtimeToolProjectionStart = "<RUNTIME_TOOL_PROJECTION>"
const runtimeToolProjectionEnd = "</RUNTIME_TOOL_PROJECTION>"

// injectRuntimeToolProjection adds the current capability/visibility layer to
// the model-facing system message. The marker is replaced on every decision
// cycle, so a resumed Run never accumulates stale projections in its durable
// history or in the provider request.
func injectRuntimeToolProjection(messages []model.Message, payload map[string]any) []model.Message {
	encoded, err := json.Marshal(payload)
	if err != nil {
		return messages
	}
	block := runtimeToolProjectionStart + "\n" + string(encoded) + "\n" + runtimeToolProjectionEnd
	projected := append([]model.Message(nil), messages...)
	if len(projected) > 0 && projected[0].Role == model.RoleSystem {
		content := projected[0].TextContent()
		if start := strings.Index(content, runtimeToolProjectionStart); start >= 0 {
			if end := strings.Index(content[start:], runtimeToolProjectionEnd); end >= 0 {
				end += start + len(runtimeToolProjectionEnd)
				content = strings.TrimSpace(content[:start] + content[end:])
			}
		}
		if content != "" {
			content += "\n\n"
		}
		content += block
		projected[0] = model.TextMessage(model.RoleSystem, content)
		return projected
	}
	return append([]model.Message{model.TextMessage(model.RoleSystem, block)}, projected...)
}

func (r *Runner) saveCheckpoint(ctx context.Context, checkpoint harness.Checkpoint) error {
	if r.config.Checkpoints == nil {
		return nil
	}
	if len(checkpoint.ContextState) == 0 && len(r.config.ContextState) != 0 {
		checkpoint.ContextState = append(json.RawMessage(nil), r.config.ContextState...)
	}
	if len(checkpoint.ExecutionLedger) == 0 {
		checkpoint.ExecutionLedger = executionLedgerJSON(checkpoint.Messages)
	}
	if err := r.config.Checkpoints.Save(ctx, checkpoint); err != nil {
		return fmt.Errorf("save checkpoint: %w", err)
	}
	return nil
}

func (r *Runner) append(ctx context.Context, input event.Input) error {
	if strings.TrimSpace(input.WorkflowID) == "" {
		input.WorkflowID = r.config.WorkflowID
	}
	if strings.TrimSpace(input.TurnID) == "" {
		input.TurnID = r.config.TurnID
	}
	if _, err := r.events.Append(ctx, input); err != nil {
		return fmt.Errorf("append %s event: %w", input.Type, err)
	}
	return nil
}

func (r *Runner) appendReviewerDecision(ctx context.Context, runID string, turn, step int, decision reviewerDecision) error {
	payload := map[string]any{}
	encoded := jsonPayload(decision)
	if err := json.Unmarshal(encoded, &payload); err != nil {
		return fmt.Errorf("encode Reviewer decision event: %w", err)
	}
	// A replay can observe the same idempotent delegation result more than
	// once. Consumers use this stable key to fold at-least-once ledger facts
	// without treating a retry as a second independent review.
	payload["decision_id"] = "review-decision:" + decision.CallID
	payload["decision_source"] = "runtime_scheduler"
	return r.append(ctx, event.Input{
		RunID: runID, Type: event.ReviewDecisionRecorded, Turn: turn, Step: step,
		CallID: decision.CallID, ActionID: decision.CallID, Payload: jsonPayload(payload),
	})
}

func (r *Runner) tryApplyReviewerPlanPatch(ctx context.Context, runID, workspaceID string, turn, step int, messages *[]model.Message, usage model.Usage, decision reviewerDecision, schemas []model.ToolSchema) (bool, bool, error) {
	if decision.ParseError != "" || decision.Verdict != reviewcontract.VerdictChangesRequired || len(decision.RecommendedPlanChanges) == 0 || r.config.PlanSnapshot == nil {
		return false, false, nil
	}
	plan, err := r.config.PlanSnapshot(ctx)
	if err != nil {
		return false, false, fmt.Errorf("load durable Plan for Reviewer patch: %w", err)
	}
	if plan.Revision != decision.BasePlanRevision {
		return false, false, nil
	}
	result := reviewcontract.Result{
		Verdict: reviewcontract.VerdictChangesRequired, Summary: decision.Summary,
		Findings: decision.Findings, RecommendedPlanChanges: decision.RecommendedPlanChanges,
	}
	toolName, schedulerAction, callSuffix := "update_plan", "review_plan_patch", "-plan-patch"
	var mutation any
	if len(decision.RecommendedPlanChanges) == 1 && decision.RecommendedPlanChanges[0].Operation == "revise_verification" {
		mutation, err = reviewcontract.CompileVerificationRevision(plan, result)
		toolName, schedulerAction, callSuffix = "revise_verification", "review_verification_patch", "-verification-patch"
	} else {
		mutation, err = reviewcontract.CompilePlanPatch(plan, result)
	}
	if err != nil {
		if errors.Is(err, reviewcontract.ErrPlanPatchRequiresPlanner) {
			return false, false, nil
		}
		return false, false, fmt.Errorf("compile Reviewer Plan patch: %w", err)
	}
	patchSchemas := onlyToolSchemas(ensureToolSchemas(nil, schemas, toolName), toolName)
	if len(patchSchemas) != 1 {
		return false, false, nil
	}
	arguments, err := json.Marshal(mutation)
	if err != nil {
		return false, false, fmt.Errorf("encode Reviewer Plan patch: %w", err)
	}
	call := model.ToolCall{ID: decision.CallID + callSuffix, Name: toolName, Arguments: arguments}
	*messages = append(*messages, runtimeSchedulerMessage(runID, turn, step, schedulerAction, call))
	projection := toolProjectionPayload(schemas, patchSchemas, effectivePlanningPolicy(r.config.PlanningPolicy), true, true, true)
	projection["decision_source"] = "runtime_scheduler"
	projection["scheduler_action"] = schedulerAction
	if err := r.append(ctx, event.Input{RunID: runID, Turn: turn, Step: step, Type: event.ToolSchemaProjected, Payload: jsonPayload(projection)}); err != nil {
		return true, false, err
	}
	failed, err := r.executeToolCalls(ctx, runID, workspaceID, turn, step, messages, usage, []model.ToolCall{call}, "", patchSchemas, "runtime_scheduler")
	return true, failed, err
}

func jsonPayload(value any) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err != nil {
		panic(fmt.Sprintf("marshal internal event payload: %v", err))
	}
	return encoded
}

func normalizeArguments(arguments json.RawMessage) json.RawMessage {
	if len(arguments) == 0 {
		return json.RawMessage(`{}`)
	}
	return arguments
}

func addUsage(total *model.Usage, current model.Usage) {
	total.InputTokens += current.InputTokens
	total.OutputTokens += current.OutputTokens
	total.TotalTokens += current.TotalTokens
}
