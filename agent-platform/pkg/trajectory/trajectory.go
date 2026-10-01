// Package trajectory builds a stable, read-only projection from append-only
// Run events. Events remain the audit source of truth; this package only joins
// request/completion pairs into records that are practical for UIs and APIs.
package trajectory

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
)

type Status string

const (
	StatusRunning   Status = "running"
	StatusCompleted Status = "completed"
	StatusFailed    Status = "failed"
	StatusCancelled Status = "cancelled"
	StatusWaiting   Status = "waiting"
)

// Record is a stable projection row. EventSequences permits lossless drill-down
// to the immutable ledger without copying unbounded payloads into every row.
type Record struct {
	ID                string         `json:"id"`
	TraceID           string         `json:"trace_id,omitempty"`
	WorkflowID        string         `json:"workflow_id,omitempty"`
	Sequence          int64          `json:"sequence"`
	LastSequence      int64          `json:"last_sequence"`
	Kind              string         `json:"kind"`
	ParentID          string         `json:"parent_id,omitempty"`
	RootID            string         `json:"root_id,omitempty"`
	Turn              int            `json:"turn,omitempty"`
	Step              int            `json:"step,omitempty"`
	PlanNodeID        string         `json:"plan_node_id,omitempty"`
	DecisionCycle     int            `json:"decision_cycle,omitempty"`
	ActionID          string         `json:"action_id,omitempty"`
	CallID            string         `json:"call_id,omitempty"`
	DelegationID      string         `json:"delegation_id,omitempty"`
	ChildRunID        string         `json:"child_run_id,omitempty"`
	AgentVersionID    string         `json:"agent_version_id,omitempty"`
	ModelResolutionID string         `json:"model_resolution_id,omitempty"`
	ModelID           string         `json:"model_id,omitempty"`
	ToolVersionID     string         `json:"tool_version_id,omitempty"`
	PromptVersionID   string         `json:"prompt_version_id,omitempty"`
	SkillSetVersionID string         `json:"skillset_version_id,omitempty"`
	ToolSetVersionID  string         `json:"toolset_version_id,omitempty"`
	Status            Status         `json:"status"`
	StartedAt         time.Time      `json:"started_at"`
	CompletedAt       *time.Time     `json:"completed_at,omitempty"`
	DurationMS        int64          `json:"duration_ms,omitempty"`
	InputTokens       int64          `json:"input_tokens,omitempty"`
	OutputTokens      int64          `json:"output_tokens,omitempty"`
	TotalCost         float64        `json:"total_cost,omitempty"`
	Summary           string         `json:"summary"`
	Metrics           map[string]any `json:"metrics,omitempty"`
	DetailRefs        map[string]any `json:"detail_refs,omitempty"`
	EventSequences    []int64        `json:"event_sequences"`
}

// Page is cursor-addressable by the first event sequence of each record.
type Page struct {
	Records    []Record `json:"records"`
	NextCursor int64    `json:"next_cursor,omitempty"`
	HasMore    bool     `json:"has_more"`
	Total      int      `json:"total"`
}

// Project deterministically merges model and tool lifecycle pairs. Unknown
// event types become generic lifecycle records so protocol evolution cannot
// make the trace disappear.
func Project(events []event.Event) []Record {
	ordered := append([]event.Event(nil), events...)
	sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].Sequence < ordered[j].Sequence })
	records := make([]Record, 0, len(ordered))
	indexes := make(map[string]int)
	for _, current := range ordered {
		key, kind := projectionKey(current)
		if index, exists := indexes[key]; exists {
			records[index] = merge(records[index], current)
			continue
		}
		record := beginRecord(current, key, kind)
		indexes[key] = len(records)
		records = append(records, record)
	}
	sort.SliceStable(records, func(i, j int) bool { return records[i].Sequence < records[j].Sequence })
	assignParents(records)
	return records
}

// assignParents turns the flat event projection into a stable observation
// tree. New records carry Workflow/DecisionCycle/Action semantics while the
// legacy run/turn/step identifiers remain the deterministic fallback for old
// events. Parent links never depend on arrival timing, so pagination and
// replay produce the same tree.
func assignParents(records []Record) {
	byID := make(map[string]*Record, len(records))
	turns := make(map[int]string)
	steps := make(map[string]string)
	for index := range records {
		record := &records[index]
		byID[record.ID] = record
		if record.Kind == "turn" && record.Turn > 0 {
			turns[record.Turn] = record.ID
		}
		if record.Kind == "step" && record.Turn > 0 && record.Step > 0 {
			steps[fmt.Sprintf("%d:%d", record.Turn, record.Step)] = record.ID
		}
	}
	for index := range records {
		record := &records[index]
		if record.Kind == "run" {
			record.RootID = record.ID
			continue
		}
		if record.ParentID == "" {
			stepKey := fmt.Sprintf("%d:%d", record.Turn, record.Step)
			switch record.Kind {
			case "tool":
				// Model and tool observations are siblings under the Step. The
				// model/tool/call IDs still correlate the causal exchange without
				// inventing a parent span when providers emit tool calls directly.
				record.ParentID = steps[stepKey]
			case "model", "context", "memory", "skill", "plan", "checkpoint", "verification":
				record.ParentID = steps[stepKey]
			case "step":
				record.ParentID = turns[record.Turn]
			case "turn":
				for _, candidate := range records {
					if candidate.Kind == "run" {
						record.ParentID = candidate.ID
						break
					}
				}
			case "agent":
				if parent := steps[stepKey]; parent != "" {
					record.ParentID = parent
				}
			default:
				if parent := steps[stepKey]; parent != "" {
					record.ParentID = parent
				} else {
					record.ParentID = turns[record.Turn]
				}
			}
		}
	}
	// Parent IDs are known now; propagate the root in a second pass so a child
	// encountered before its parent never receives a transient parent ID as its
	// root. The bounded loop also tolerates records arriving in arbitrary order.
	for pass := 0; pass < len(records); pass++ {
		changed := false
		for index := range records {
			record := &records[index]
			if record.Kind == "run" {
				if record.RootID != record.ID {
					record.RootID = record.ID
					changed = true
				}
				continue
			}
			if parent := byID[record.ParentID]; parent != nil {
				root := parent.RootID
				if root == "" {
					root = parent.ID
				}
				if record.RootID != root {
					record.RootID = root
					changed = true
				}
			}
		}
		if !changed {
			break
		}
	}
}

func Paginate(records []Record, after int64, limit int) Page {
	if limit <= 0 {
		limit = 100
	}
	if limit > 500 {
		limit = 500
	}
	page := Page{Records: make([]Record, 0, minInt(limit, len(records))), Total: len(records)}
	for _, record := range records {
		if record.Sequence <= after {
			continue
		}
		if len(page.Records) == limit {
			page.HasMore = true
			break
		}
		page.Records = append(page.Records, record)
		page.NextCursor = record.Sequence
	}
	return page
}

func minInt(left, right int) int {
	if left < right {
		return left
	}
	return right
}

func projectionKey(current event.Event) (string, string) {
	switch current.Type {
	case event.RunCreated, event.RunClaimed, event.RunResumed, event.RunSuspended, event.RunCancelRequested, event.RunCompleted, event.RunFailed, event.RunCancelled:
		return "run", "run"
	case event.ModelRequested, event.ModelCompleted, event.ModelFailed:
		return fmt.Sprintf("model:%d:%d", current.Turn, current.Step), "model"
	case event.ToolSchemaProjected:
		return fmt.Sprintf("tool-schema:%d:%d", current.Turn, current.Step), "tool_projection"
	case event.ToolCalled, event.ToolApprovalRequested, event.ToolApprovalResolved, event.ToolCompleted, event.ToolFailed:
		return "tool:" + current.CallID, "tool"
	case event.DelegationRequested, event.DelegationCompleted, event.DelegationFailed:
		payload := payloadMap(current.Payload)
		delegationID := payloadString(payload, "delegation_id", "child_run_id")
		if delegationID == "" {
			delegationID = current.CallID
		}
		if delegationID == "" {
			delegationID = fmt.Sprintf("%d:%d", current.Turn, current.Step)
		}
		return "agent:" + delegationID, "agent"
	case event.StepStarted, event.StepCompleted:
		return fmt.Sprintf("step:%d:%d", current.Turn, current.Step), "step"
	case event.TurnStarted, event.TurnCompleted:
		return fmt.Sprintf("turn:%d", current.Turn), "turn"
	default:
		return fmt.Sprintf("event:%d", current.Sequence), eventKind(current.Type)
	}
}

func beginRecord(current event.Event, key, kind string) Record {
	payload := payloadMap(current.Payload)
	workflowID, planNodeID, decisionCycle, actionID := event.SemanticFromPayload(current.Payload)
	if workflowID == "" {
		workflowID = current.RunID
	}
	if decisionCycle == 0 {
		decisionCycle = current.Turn
	}
	if actionID == "" {
		actionID = current.CallID
	}
	record := Record{
		ID: stableID(current.RunID, key), TraceID: current.RunID, WorkflowID: workflowID, Sequence: current.Sequence, LastSequence: current.Sequence,
		Kind: kind, Turn: current.Turn, Step: current.Step, PlanNodeID: planNodeID,
		DecisionCycle: decisionCycle, ActionID: actionID, CallID: current.CallID,
		Status: statusFor(current.Type), StartedAt: current.CreatedAt,
		Summary: summary(current), EventSequences: []int64{current.Sequence},
		DetailRefs: map[string]any{"events": []int64{current.Sequence}},
	}
	applyMetadata(&record, payload)
	if terminalEvent(current.Type) {
		completed := current.CreatedAt
		record.CompletedAt = &completed
	}
	if latency := payloadInt(current.Payload, "latency_ms"); latency > 0 {
		record.DurationMS = latency
		record.Metrics = map[string]any{"latency_ms": latency}
	}
	return record
}

func merge(record Record, current event.Event) Record {
	payload := payloadMap(current.Payload)
	workflowID, planNodeID, decisionCycle, actionID := event.SemanticFromPayload(current.Payload)
	if workflowID != "" {
		record.WorkflowID = workflowID
	}
	if planNodeID != "" {
		record.PlanNodeID = planNodeID
	}
	if decisionCycle > 0 {
		record.DecisionCycle = decisionCycle
	}
	if actionID != "" {
		record.ActionID = actionID
	}
	record.LastSequence = current.Sequence
	record.EventSequences = append(record.EventSequences, current.Sequence)
	record.DetailRefs["events"] = append([]int64(nil), record.EventSequences...)
	record.Status = statusFor(current.Type)
	record.Summary = summary(current)
	applyMetadata(&record, payload)
	if terminalEvent(current.Type) {
		completed := current.CreatedAt
		record.CompletedAt = &completed
		record.DurationMS = completed.Sub(record.StartedAt).Milliseconds()
	}
	if latency := payloadInt(current.Payload, "latency_ms"); latency > 0 {
		record.DurationMS = latency
		if record.Metrics == nil {
			record.Metrics = make(map[string]any)
		}
		record.Metrics["latency_ms"] = latency
	}
	return record
}

func applyMetadata(record *Record, payload map[string]any) {
	for _, item := range []struct {
		dst  *string
		keys []string
	}{
		{&record.DelegationID, []string{"delegation_id"}}, {&record.ChildRunID, []string{"child_run_id"}}, {&record.AgentVersionID, []string{"agent_version_id"}},
		{&record.ModelResolutionID, []string{"model_resolution_id"}}, {&record.ModelID, []string{"model_id", "model"}}, {&record.ToolVersionID, []string{"tool_version_id"}}, {&record.PromptVersionID, []string{"prompt_version_id", "prompt_version"}},
		{&record.SkillSetVersionID, []string{"skillset_version_id", "skill_set_version_id"}}, {&record.ToolSetVersionID, []string{"toolset_version_id", "tool_set_version_id"}},
	} {
		if value := payloadString(payload, item.keys...); value != "" {
			*item.dst = value
		}
	}
	usage, ok := payload["usage"].(map[string]any)
	if !ok {
		usage = payload
	}
	if value := payloadIntMap(usage, "input_tokens", "prompt_tokens"); value > 0 {
		record.InputTokens = value
	}
	if value := payloadIntMap(usage, "output_tokens", "completion_tokens"); value > 0 {
		record.OutputTokens = value
	}
	if value, ok := payload["cost"].(float64); ok {
		record.TotalCost = value
	}
	if value, ok := payload["total_cost"].(float64); ok {
		record.TotalCost = value
	}
	if record.Metrics == nil {
		record.Metrics = map[string]any{}
	}
	if record.InputTokens > 0 {
		record.Metrics["input_tokens"] = record.InputTokens
	}
	if record.OutputTokens > 0 {
		record.Metrics["output_tokens"] = record.OutputTokens
	}
	if record.TotalCost > 0 {
		record.Metrics["total_cost"] = record.TotalCost
	}
}

func payloadMap(raw json.RawMessage) map[string]any {
	var value map[string]any
	_ = json.Unmarshal(raw, &value)
	return value
}
func payloadString(payload map[string]any, keys ...string) string {
	for _, key := range keys {
		if value := strings.TrimSpace(fmt.Sprint(payload[key])); value != "" && value != "<nil>" {
			return value
		}
	}
	return ""
}
func payloadIntMap(payload map[string]any, keys ...string) int64 {
	for _, key := range keys {
		if value, ok := payload[key].(float64); ok {
			return int64(value)
		}
	}
	return 0
}

func statusFor(kind event.Type) Status {
	switch kind {
	case event.ModelFailed, event.ToolFailed, event.RunFailed:
		return StatusFailed
	case event.RunCancelled:
		return StatusCancelled
	case event.ToolApprovalRequested, event.UserInputRequested, event.RunSuspended:
		return StatusWaiting
	case event.ModelRequested, event.ToolSchemaProjected, event.ToolCalled, event.StepStarted, event.TurnStarted, event.RunCreated, event.RunClaimed, event.RunResumed:
		return StatusRunning
	default:
		return StatusCompleted
	}
}

func terminalEvent(kind event.Type) bool {
	switch kind {
	case event.ModelCompleted, event.ModelFailed, event.ToolCompleted, event.ToolFailed,
		event.StepCompleted, event.TurnCompleted, event.RunCompleted, event.RunFailed, event.RunCancelled,
		event.ToolApprovalRequested, event.ToolApprovalResolved, event.UserInputRequested, event.UserInputReceived:
		return true
	default:
		return false
	}
}

func eventKind(kind event.Type) string {
	value := string(kind)
	switch {
	case strings.Contains(value, "SKILL"):
		return "skill"
	case strings.Contains(value, "MEMORY"):
		return "memory"
	case strings.Contains(value, "CONTEXT"):
		return "context"
	case strings.Contains(value, "PLAN"), strings.Contains(value, "USER_INPUT"), kind == event.ExecutionModeSelected:
		return "plan"
	case strings.Contains(value, "CHECKPOINT"):
		return "checkpoint"
	case strings.Contains(value, "DELEGATION"), strings.Contains(value, "CHILD_RUN"):
		return "agent"
	case strings.Contains(value, "FILE_"), strings.Contains(value, "WORKSPACE"), strings.Contains(value, "TOOL_SCHEMA"):
		return "tool"
	default:
		return "lifecycle"
	}
}

func summary(current event.Event) string {
	var payload map[string]any
	_ = json.Unmarshal(current.Payload, &payload)
	if current.Type == event.ToolSchemaProjected {
		offered, _ := payload["offered_tools"].([]any)
		available, _ := payload["available_tools"].([]any)
		return fmt.Sprintf("offered %d/%d tools", len(offered), len(available))
	}
	if current.Type == event.ExecutionModeSelected {
		mode := strings.TrimSpace(fmt.Sprint(payload["mode"]))
		policy := strings.TrimSpace(fmt.Sprint(payload["policy"]))
		return strings.Trim(strings.Join([]string{mode, policy}, " · "), " ·")
	}
	for _, key := range []string{"name", "question", "goal", "answer", "error", "model_id", "status", "path", "reason"} {
		if value := strings.TrimSpace(fmt.Sprint(payload[key])); value != "" && value != "<nil>" {
			return value
		}
	}
	return strings.ReplaceAll(strings.ToLower(string(current.Type)), "_", " ")
}

func payloadInt(raw json.RawMessage, key string) int64 {
	var payload map[string]any
	if json.Unmarshal(raw, &payload) != nil {
		return 0
	}
	value, _ := payload[key].(float64)
	return int64(value)
}

func stableID(runID, key string) string {
	digest := sha256.Sum256([]byte(runID + "\x00" + key))
	return "tr_" + hex.EncodeToString(digest[:12])
}
