// Package tool defines versioned Tool contracts and execution interfaces.
package tool

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/contract"
)

var (
	// ErrExecutionInProgress reports a duplicate concurrent Tool Call.
	ErrExecutionInProgress = errors.New("tool execution is already in progress")
	// ErrCallIDConflict reports reuse of a call ID with different input/version.
	ErrCallIDConflict = errors.New("tool call id was reused with different input")
	// ErrDenied reports a policy-disabled Tool that must never reach its handler.
	ErrDenied = errors.New("tool execution is denied by policy")
)

// ContractError is a stable, model-visible tool contract failure. The
// provider-specific error remains in Message, while callers can use the
// machine fields to decide whether a retry is safe and what to repair.
type ContractError struct {
	Code      string `json:"code"`
	Tool      string `json:"tool,omitempty"`
	Path      string `json:"path,omitempty"`
	Expected  string `json:"expected,omitempty"`
	Actual    string `json:"actual,omitempty"`
	Retryable bool   `json:"retryable"`
	// Correction is a model-actionable repair instruction. It is deliberately
	// separate from Message so UI and retry policy do not have to parse prose.
	Correction string `json:"correction,omitempty"`
	// RetryTemplate is an optional JSON object showing the shape of one safe
	// changed retry. It is a template, never an executable call or a receipt.
	RetryTemplate json.RawMessage `json:"retry_template,omitempty"`
	Message       string          `json:"message"`
	Cause         error           `json:"-"`
}

func (e *ContractError) Error() string {
	if e == nil {
		return "tool contract error"
	}
	if e.Path != "" {
		return fmt.Sprintf("%s (%s): %s", e.Message, e.Path, e.Code)
	}
	return fmt.Sprintf("%s (%s)", e.Message, e.Code)
}

// Unwrap preserves scheduler sentinels such as approval.ErrRequired and
// delegation.ErrPending while exposing the stable JSON contract to the model.
func (e *ContractError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Cause
}

// NewContractError constructs a normalized contract error without exposing
// provider-specific internals as the only recovery signal.
func NewContractError(code, toolName, path, expected, actual, message string, retryable bool) error {
	return &ContractError{Code: code, Tool: toolName, Path: path, Expected: expected, Actual: actual, Message: message, Retryable: retryable}
}

// NewContractErrorWithRepair is the canonical constructor for failures that
// can be repaired by the model. Keeping the repair contract on the typed
// error prevents every provider from inventing a different error envelope.
func NewContractErrorWithRepair(code, toolName, path, expected, actual, message, correction string, retryTemplate json.RawMessage, retryable bool) error {
	return &ContractError{
		Code: code, Tool: toolName, Path: path, Expected: expected, Actual: actual,
		Message: message, Correction: correction, RetryTemplate: append(json.RawMessage(nil), retryTemplate...), Retryable: retryable,
	}
}

func AsContractError(err error) (*ContractError, bool) {
	var target *ContractError
	if errors.As(err, &target) {
		return target, true
	}
	return nil, false
}

// Risk is the immutable risk classification of a ToolVersion.
type Risk string

const (
	RiskRead     Risk = "READ"
	RiskInternal Risk = "INTERNAL"
	RiskLowWrite Risk = "LOW_WRITE"
	RiskHigh     Risk = "HIGH_RISK"
	RiskDenied   Risk = "DENIED"
)

// ExecutionMode controls whether the scheduler may overlap Tool bodies.
type ExecutionMode string

const (
	ExecutionSerial   ExecutionMode = "serial"
	ExecutionParallel ExecutionMode = "parallel"
)

// Definition is an immutable model-visible and policy-visible ToolVersion.
type Definition struct {
	Name          string          `json:"name"`
	Version       string          `json:"version"`
	Description   string          `json:"description"`
	InputSchema   json.RawMessage `json:"input_schema"`
	OutputSchema  json.RawMessage `json:"output_schema,omitempty"`
	Risk          Risk            `json:"risk"`
	ExecutionMode ExecutionMode   `json:"execution_mode"`
}

// Call is one model-originated Tool invocation.
type Call struct {
	RunID string `json:"run_id"`
	// WorkflowID is the stable parent workflow identity. It is normally the
	// RunID during the compatibility migration, but remains explicit so child
	// actions and resumed executions can be correlated without parsing payloads.
	WorkflowID string `json:"workflow_id,omitempty"`
	// WorkspaceID is the durable Session workspace scope. It may differ from
	// RunID when a continuation Run resumes the same user task.
	WorkspaceID string `json:"workspace_id,omitempty"`
	// PlanStepID is attached by the runtime after it validates the durable
	// active Todo. It is platform-owned metadata and is never model input.
	PlanStepID string `json:"plan_step_id,omitempty"`
	// PlanNodeID is the canonical name for PlanStepID. Both are accepted while
	// clients migrate; the runtime always emits plan_node_id in observations.
	PlanNodeID string `json:"plan_node_id,omitempty"`
	// The following generations are attached by the runtime. They are never
	// model-owned arguments; persistence uses them to reject a late receipt
	// from an older Workflow, Plan, node, or workspace snapshot.
	WorkflowGeneration int64           `json:"workflow_generation,omitempty"`
	PlanRevision       int             `json:"plan_revision,omitempty"`
	NodeRevision       int64           `json:"node_revision,omitempty"`
	WorkspaceRevision  int64           `json:"workspace_revision,omitempty"`
	DecisionCycle      int             `json:"decision_cycle,omitempty"`
	ActionID           string          `json:"action_id,omitempty"`
	Turn               int             `json:"turn"`
	Step               int             `json:"step"`
	ID                 string          `json:"id"`
	Name               string          `json:"name"`
	Arguments          json.RawMessage `json:"arguments"`
}

// Result is the single model-facing outcome of one Tool Call.
type Result struct {
	Content json.RawMessage `json:"content"`
	// ModelContent is a read-time projection. Content remains the complete
	// durable result, while callers building model messages/events should use
	// ModelVisible so a large result cannot consume the whole context window.
	// It is deliberately excluded from persistence JSON.
	ModelContent json.RawMessage   `json:"-"`
	IsError      bool              `json:"is_error,omitempty"`
	Error        string            `json:"error,omitempty"`
	Meta         map[string]string `json:"meta,omitempty"`
	// Artifacts carry bounded opaque outputs to persistence. They are never sent
	// directly to the model or duplicated in TOOL_COMPLETED payloads.
	Artifacts []Artifact `json:"-"`
}

// NormalizeResultFailure makes a handler-produced IsError result obey the
// same model-facing contract as an error returned from Execute. Providers
// such as MCP and workspace commands sometimes return a process/result
// payload with IsError=true rather than a Go error; without this normalization
// those paths would silently omit retryability and repair guidance.
func NormalizeResultFailure(call Call, result Result) Result {
	if !result.IsError {
		return result
	}
	code := "TOOL_EXECUTION_FAILED"
	retryable := true
	correction := "Inspect the structured error and issue one changed call; do not repeat an unchanged payload."
	failureKind := ""
	var existingPayload map[string]any
	if len(result.Content) > 0 && json.Unmarshal(result.Content, &existingPayload) == nil {
		if value, ok := existingPayload["error_code"].(string); ok && strings.TrimSpace(value) != "" {
			code = strings.TrimSpace(value)
		}
		if value, ok := existingPayload["failure_kind"].(string); ok {
			failureKind = strings.TrimSpace(value)
		}
		if value, ok := existingPayload["correction"].(string); ok && strings.TrimSpace(value) != "" {
			correction = strings.TrimSpace(value)
		}
	}
	if result.Meta != nil {
		if value := strings.TrimSpace(result.Meta["error_code"]); value != "" {
			code = value
		}
		if value := strings.TrimSpace(result.Meta["retryable"]); value != "" {
			retryable = strings.EqualFold(value, "true")
		}
		if value := strings.TrimSpace(result.Meta["correction"]); value != "" {
			correction = value
		}
		failureKind = strings.TrimSpace(result.Meta["failure_kind"])
	}
	message := strings.TrimSpace(result.Error)
	if message == "" {
		message = "tool returned an error result"
	}
	switch code {
	case "SANDBOX_COMMAND_REJECTED", "TOOL_POLICY_REJECTED", "TOOL_APPROVAL_REQUIRED", "DELEGATION_TARGET_NOT_ALLOWED":
		retryable = false
	}
	if failureKind == "" {
		failureKind = FailureKind(code)
	}
	correction, retryTemplate := RecoveryContract(call.Name, code, correction)
	if result.Meta == nil {
		result.Meta = make(map[string]string)
	}
	result.Meta["error_code"] = code
	result.Meta["retryable"] = fmt.Sprintf("%t", retryable)
	result.Meta["correction"] = correction
	result.Meta["failure_kind"] = failureKind
	payload := existingPayload
	if payload != nil {
		if _, ok := payload["error"]; !ok {
			payload["error"] = message
		}
	} else {
		payload = map[string]any{"error": message}
	}
	payload["error_code"] = code
	payload["retryable"] = retryable
	payload["correction"] = correction
	payload["failure_kind"] = failureKind
	// Always expose a template field. An empty object is deliberately a
	// non-executable placeholder for non-retryable failures; concrete providers
	// may replace it with a safe changed-call template.
	if _, ok := payload["retry_template"]; !ok {
		payload["retry_template"] = retryTemplate
	}
	if encoded, err := json.Marshal(payload); err == nil {
		result.Content = encoded
	}
	result.Error = message
	return result
}

// FailureKind is the stable taxonomy shared by Tool events, failure-memory
// matching, and deterministic recovery. It is deliberately independent of
// provider error prose.
func FailureKind(code string) string {
	switch strings.ToUpper(strings.TrimSpace(code)) {
	case "TOOL_SCHEMA_INVALID", "TOOL_ARGUMENTS_NORMALIZATION_FAILED", "INVALID_TOOL_RESULT":
		return "schema_invalid"
	case "SANDBOX_COMMAND_REJECTED", "SANDBOX_POLICY_REJECTED", "TOOL_POLICY_REJECTED", "TOOL_APPROVAL_REQUIRED":
		return "policy_rejected"
	case "PATH_NOT_FOUND", "PARENT_DIRECTORY_MISSING":
		return "path_not_found"
	case "EDIT_ANCHOR_MISMATCH":
		return "stale_edit_anchor"
	case "PROCESS_EXIT":
		return "process_exit"
	case "TOOL_INTERRUPTED":
		return "timeout"
	case "DUPLICATE_READ_RANGE":
		return "duplicate_observation"
	case "DETERMINISTIC_RETRY_BLOCKED", "DUPLICATE_FAILED_CALL":
		return "repeated_without_change"
	case "PLAN_REVISION_CONFLICT", "PLAN_CHANGE_MODE_REQUIRED", "PLAN_CHANGE_MODE_INVALID", "PLAN_REPLAN_REASON_REQUIRED", "PLAN_OPEN_NODE_OMITTED", "PLAN_EXTEND_NODE_CONFLICT", "PLAN_RETIREMENT_INVALID":
		return "plan_state_conflict"
	default:
		return "infrastructure_error"
	}
}

// RecoveryContract returns a Tool-specific, deterministic repair template.
// Historical Memory may add context only when it exactly matches this
// taxonomy; it never replaces this current contract.
func RecoveryContract(toolName, code, fallback string) (string, map[string]any) {
	upper := strings.ToUpper(strings.TrimSpace(code))
	switch toolName {
	case "read_file":
		if upper == "DUPLICATE_READ_RANGE" {
			return "Do not repeat the unchanged read. Runtime returns the cached observation or Event Ledger reference; request a different range only when new content is needed.", map[string]any{"name": "read_file", "arguments": map[string]any{"path": "<same-canonical-path>", "start_line": "<different-start>", "line_count": "<needed-count>"}}
		}
		if upper == "PATH_NOT_FOUND" || upper == "PARENT_DIRECTORY_MISSING" {
			return "Use the exact canonical workspace-relative path. List or search the nearest known parent once, then retry read_file with {path,start_line,line_count}; do not create a file to satisfy a read.", map[string]any{"name": "read_file", "arguments": map[string]any{"path": "<canonical-workspace-relative-path>", "start_line": 1, "line_count": 200}}
		}
		return "Correct read_file using its exact offered schema: path is required; start_line and line_count are optional integers. Reuse a cached observation for an unchanged duplicate range.", map[string]any{"name": "read_file", "arguments": map[string]any{"path": "<canonical-workspace-relative-path>", "start_line": 1, "line_count": 200}}
	case "edit_file":
		return "Re-read a narrow current range of the exact canonical path, then call edit_file with required path, one unique current old_text anchor, and new_text. Do not substitute write_file fields.", map[string]any{"name": "edit_file", "arguments": map[string]any{"path": "<canonical-workspace-relative-path>", "old_text": "<exact-current-unique-text>", "new_text": "<replacement>"}}
	case "run_command":
		if upper == "SANDBOX_COMMAND_REJECTED" || upper == "SANDBOX_POLICY_REJECTED" || upper == "TOOL_POLICY_REJECTED" {
			return "The command shape was rejected by Sandbox policy. Use a permitted executable with args as a JSON array, a canonical workspace-relative working_directory, and timeout_seconds within the offered maximum; write an actual script instead of python -c.", map[string]any{"name": "run_command", "arguments": map[string]any{"command": "python3", "args": []any{"<workspace-relative-script>"}, "timeout_seconds": 30}}
		}
		return "Use the process diagnostic to repair the workspace or change arguments. run_command requires command plus args as a JSON array; keep timeout_seconds within the offered maximum and do not repeat an unchanged failed command.", map[string]any{"name": "run_command", "arguments": map[string]any{"command": "<permitted-executable>", "args": []any{}, "timeout_seconds": 30}}
	case "write_file", "append_file":
		if upper == "PARENT_DIRECTORY_MISSING" || upper == "PATH_NOT_FOUND" {
			return "Create the missing workspace-relative parent directory, then retry the original operation with the exact canonical path.", map[string]any{"name": "create_directory", "arguments": map[string]any{"path": "<canonical-parent>"}}
		}
	}
	return fallback, map[string]any{}
}

// InlineResultLimit is intentionally below the 3k-token tool-result budget
// used by the 24k deployment. The remainder is reserved for surrounding
// messages, schemas, and the collapse safety margin.
const InlineResultLimit = 6 << 10

const resultPreviewLimit = 2 << 10

// ModelVisible returns a copy suitable for events and model messages. It
// never discards the complete Content held by the durable execution record.
func (r Result) ModelVisible() Result {
	if len(r.ModelContent) != 0 {
		r.Content = append(json.RawMessage(nil), r.ModelContent...)
	}
	r.ModelContent = nil
	r.Artifacts = nil
	return r
}

// ApplyStoredModelProjection reconstructs the non-persisted projection after
// an idempotent tool result is loaded from PostgreSQL.
func (r *Result) ApplyStoredModelProjection() {
	if r == nil || len(r.ModelContent) != 0 || len(r.Content) <= InlineResultLimit {
		return
	}
	artifactID := ""
	if r.Meta != nil {
		artifactID = strings.TrimSpace(r.Meta["content_artifact_id"])
	}
	r.ModelContent = BuildResultReceipt(r.Content, artifactID)
}

// BuildResultReceipt creates a bounded, valid JSON result for model-facing
// projections. The preview is advisory; the artifact reference is the
// authoritative path to the complete bytes.
func BuildResultReceipt(content []byte, artifactID string) json.RawMessage {
	digest := sha256.Sum256(content)
	preview := content
	if len(preview) > resultPreviewLimit {
		preview = preview[:resultPreviewLimit]
	}
	if !utf8.Valid(preview) {
		preview = []byte(strings.ToValidUTF8(string(preview), "�"))
	}
	status := "truncated"
	if strings.TrimSpace(artifactID) != "" {
		status = "offloaded"
	}
	receipt := map[string]any{
		"status":                status,
		"result_content_sha256": fmt.Sprintf("%x", digest[:]),
		"size_bytes":            len(content),
		"preview":               string(preview),
		"truncated":             true,
	}
	if artifactID = strings.TrimSpace(artifactID); artifactID != "" {
		receipt["artifact_id"] = artifactID
		receipt["artifact_uri"] = "/api/v1/artifacts/" + artifactID + "/content"
	}
	encoded, err := json.Marshal(receipt)
	if err != nil {
		return json.RawMessage(`{"status":"truncated"}`)
	}
	return encoded
}

// Artifact is an internal persistence handoff for diffs and large tool output.
type Artifact struct {
	Kind      string            `json:"kind"`
	Name      string            `json:"name"`
	MediaType string            `json:"media_type"`
	Content   []byte            `json:"-"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

// ArtifactRef is the minimal durable reference returned by a generic result
// offloader. It is intentionally separate from Artifact, whose Content is an
// in-process persistence handoff.
type ArtifactRef struct {
	ID        string `json:"id"`
	URI       string `json:"uri"`
	SHA256    string `json:"sha256"`
	SizeBytes int    `json:"size_bytes"`
}

// Executor resolves and executes version-pinned Tools.
type Executor interface {
	Definitions() []Definition
	Execute(ctx context.Context, call Call) (Result, error)
}

// Handler implements one in-process Tool body.
type Handler func(ctx context.Context, call Call) (Result, error)

// Registry is a concurrency-safe reference Executor for Native Tools.
type Registry struct {
	mu      sync.RWMutex
	entries map[string]entry
}

type entry struct {
	definition Definition
	handler    Handler
	input      *contract.Schema
	output     *contract.Schema
}

// NewRegistry creates an empty Tool registry.
func NewRegistry() *Registry {
	return &Registry{entries: make(map[string]entry)}
}

// Register adds one Tool. A Run registry contains exactly one resolved version
// for each name, so duplicate names are rejected.
func (r *Registry) Register(definition Definition, handler Handler) error {
	if err := ValidateDefinition(definition); err != nil {
		return err
	}
	if handler == nil {
		return errors.New("tool handler is required")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.entries[definition.Name]; exists {
		return fmt.Errorf("tool %q is already registered", definition.Name)
	}
	input, err := contract.Compile(definition.InputSchema)
	if err != nil {
		return fmt.Errorf("tool %q input schema: %w", definition.Name, err)
	}
	var output *contract.Schema
	if len(definition.OutputSchema) > 0 {
		output, err = contract.Compile(definition.OutputSchema)
		if err != nil {
			return fmt.Errorf("tool %q output schema: %w", definition.Name, err)
		}
	}
	r.entries[definition.Name] = entry{definition: definition, handler: handler, input: input, output: output}
	return nil
}

// Definitions returns a deterministic snapshot suitable for a model request.
func (r *Registry) Definitions() []Definition {
	r.mu.RLock()
	defer r.mu.RUnlock()
	definitions := make([]Definition, 0, len(r.entries))
	for _, item := range r.entries {
		if item.definition.Risk == RiskDenied {
			continue
		}
		definitions = append(definitions, item.definition)
	}
	sort.Slice(definitions, func(i, j int) bool { return definitions[i].Name < definitions[j].Name })
	return definitions
}

// Execute dispatches one call after resolving its registered definition.
func (r *Registry) Execute(ctx context.Context, call Call) (Result, error) {
	r.mu.RLock()
	item, exists := r.entries[call.Name]
	r.mu.RUnlock()
	if !exists {
		return Result{}, NewContractError("TOOL_NOT_REGISTERED", call.Name, "", "registered tool", "missing", fmt.Sprintf("tool %q is not registered", call.Name), false)
	}
	if item.definition.Risk == RiskDenied {
		return Result{}, fmt.Errorf("%w: %s", ErrDenied, call.Name)
	}
	if err := item.input.ValidateJSON(call.Arguments); err != nil {
		return Result{}, NewContractError("INVALID_ARGUMENTS", call.Name, "", "arguments matching the offered schema", "invalid", fmt.Sprintf("tool %q arguments: %s", call.Name, err), true)
	}
	result, err := item.handler(ctx, call)
	if err != nil {
		return result, NormalizeExecutionError(call, err)
	}
	if result.IsError {
		return NormalizeResultFailure(call, result), nil
	}
	if item.output == nil {
		return result, nil
	}
	if err := item.output.ValidateJSON(result.Content); err != nil {
		return Result{}, NewContractError("INVALID_TOOL_RESULT", call.Name, "", "result matching the output schema", "invalid", fmt.Sprintf("tool %q result: %s", call.Name, err), false)
	}
	return result, nil
}

// NormalizeExecutionError gives every provider path the same structured
// failure contract, including handlers executed behind the durable Postgres
// idempotency wrapper rather than through Registry.Execute.
func NormalizeExecutionError(call Call, err error) error {
	if err == nil {
		return nil
	}
	if _, ok := AsContractError(err); ok {
		return err
	}
	message := err.Error()
	lower := strings.ToLower(message)
	code := "TOOL_EXECUTION_FAILED"
	retryable := true
	correction := "Inspect the structured error and issue one changed call; do not repeat an unchanged payload."
	// Preserve the innermost structured taxonomy before considering outer
	// provider/Sandbox prose. In particular, JSON Schema says "not allowed",
	// which must never be relabelled as a Sandbox policy rejection.
	if strings.Contains(message, "(TOOL_SCHEMA_INVALID)") || strings.Contains(lower, "schema validation") || strings.Contains(lower, "additionalproperties") || strings.Contains(lower, "missing properties") {
		code = "TOOL_SCHEMA_INVALID"
		correction = "Correct only the indicated required field, type, enum, or additional property using the currently offered Tool Schema, then issue one changed structured call."
	} else if strings.Contains(message, "(EDIT_ANCHOR_MISMATCH)") || (call.Name == "edit_file" && (strings.Contains(lower, "old_text") || strings.Contains(lower, "match"))) {
		code = "EDIT_ANCHOR_MISMATCH"
	} else if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		code = "TOOL_INTERRUPTED"
		correction = "Wait for the runtime to become runnable, then retry once with the same call only if the checkpoint shows it was not committed."
	} else if strings.Contains(lower, "approval required") {
		code = "TOOL_APPROVAL_REQUIRED"
		retryable = false
		correction = "The platform has paused this call for user approval. Do not repeat it; wait for the approval decision and resume from the checkpoint."
	} else if strings.Contains(lower, "not allowed") || strings.Contains(lower, "denied") || strings.Contains(lower, "rejected") {
		code = "TOOL_POLICY_REJECTED"
		retryable = false
		correction = "Do not repeat this call. Use a visible allowed capability, request approval, or ask the user for a supported alternative."
	} else if strings.Contains(lower, "delegation") && strings.Contains(lower, "pending") {
		code = "DELEGATION_PENDING"
		retryable = false
		correction = "The child Agent is still running. Do not poll by repeating the delegation call; wait for the platform wake-up event."
	} else if strings.Contains(lower, "no such file") || strings.Contains(lower, "parent directory") {
		if call.Name == "read_file" {
			code = "PATH_NOT_FOUND"
		} else {
			code = "PARENT_DIRECTORY_MISSING"
		}
	}
	correction, template := RecoveryContract(call.Name, code, correction)
	retryTemplate, _ := json.Marshal(template)
	contractErr := NewContractErrorWithRepair(code, call.Name, "", "a valid provider result", "provider error", message, correction, retryTemplate, retryable)
	if normalized, ok := contractErr.(*ContractError); ok {
		normalized.Cause = err
	}
	return contractErr
}

// ValidateDefinition validates a versioned model-visible Tool contract.
func ValidateDefinition(definition Definition) error {
	if strings.TrimSpace(definition.Name) == "" {
		return errors.New("tool name is required")
	}
	if strings.TrimSpace(definition.Version) == "" {
		return errors.New("tool version is required")
	}
	if _, err := contract.Compile(definition.InputSchema); err != nil {
		return fmt.Errorf("tool %q input schema: %w", definition.Name, err)
	}
	if len(definition.OutputSchema) > 0 {
		if _, err := contract.Compile(definition.OutputSchema); err != nil {
			return fmt.Errorf("tool %q output schema: %w", definition.Name, err)
		}
	}
	if definition.Risk == "" {
		return fmt.Errorf("tool %q risk is required", definition.Name)
	}
	switch definition.Risk {
	case RiskRead, RiskInternal, RiskLowWrite, RiskHigh, RiskDenied:
	default:
		return fmt.Errorf("tool %q has unknown risk %q", definition.Name, definition.Risk)
	}
	if definition.ExecutionMode == "" {
		return fmt.Errorf("tool %q execution mode is required", definition.Name)
	}
	switch definition.ExecutionMode {
	case ExecutionSerial, ExecutionParallel:
	default:
		return fmt.Errorf("tool %q has unknown execution mode %q", definition.Name, definition.ExecutionMode)
	}
	return nil
}
