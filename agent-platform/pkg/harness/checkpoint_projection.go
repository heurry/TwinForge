package harness

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strings"

	contextpkg "github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/context"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
)

const (
	checkpointFailureReceiptMaxChars = 1200
	checkpointRecentToolHistoryLimit = 5
	checkpointArgumentPreviewChars   = 480
	checkpointResultPreviewChars     = 900
)

// ProjectCheckpointForStorage converts the live ReAct transcript into a
// resumable projection. Complete arguments/results remain authoritative in
// MODEL_COMPLETED, TOOL_CALLED and TOOL_{COMPLETED,FAILED} events; a Checkpoint
// keeps only what a takeover needs to continue safely.
func ProjectCheckpointForStorage(checkpoint Checkpoint) Checkpoint {
	projected := checkpoint
	// Plan/tool/ledger projections are rebuilt from their independently stored
	// durable fields. Keeping copies inside Messages caused each checkpoint to
	// multiply the same runtime state after every continuation.
	boundedMessages := contextpkg.StripRebuildableRuntimeBlocks(checkpoint.Messages)
	projected.Messages = projectCheckpointMessages(boundedMessages, checkpoint.PendingToolCalls, checkpoint.ActiveToolCallID, checkpoint.ContextState)
	projected.Answer = canonicalCheckpointMessage(checkpoint.Answer)
	return projected
}

// RestoreCheckpointFromStorage rehydrates compatibility Content fields after
// the storage representation removed content/parts duplication.
func RestoreCheckpointFromStorage(checkpoint Checkpoint) Checkpoint {
	restored := checkpoint
	// Re-run the bounded projection on read so Checkpoints written by an older
	// worker cannot reintroduce internal event_backed/arguments_sha256 fields as
	// model-callable Tool arguments during a rolling deployment.
	restored.Messages = projectCheckpointMessages(append([]model.Message(nil), checkpoint.Messages...), checkpoint.PendingToolCalls, checkpoint.ActiveToolCallID, checkpoint.ContextState)
	for index := range restored.Messages {
		restored.Messages[index] = restoreCheckpointMessage(restored.Messages[index])
	}
	restored.Answer = restoreCheckpointMessage(checkpoint.Answer)
	return restored
}

func projectCheckpointMessages(messages []model.Message, pending []model.ToolCall, activeCallID string, contextState json.RawMessage) []model.Message {
	active := make(map[string]struct{}, len(pending)+1)
	if strings.TrimSpace(activeCallID) != "" {
		active[activeCallID] = struct{}{}
	}
	for _, call := range pending {
		if strings.TrimSpace(call.ID) != "" {
			active[call.ID] = struct{}{}
		}
	}
	failed := make(map[string]struct{})
	for _, message := range messages {
		if message.Role == model.RoleTool && message.Metadata != nil && message.Metadata["tool_failed"] == "true" {
			failed[message.ToolCallID] = struct{}{}
		}
	}

	// Events are the immutable audit log; Checkpoint is a bounded takeover
	// projection. Legacy/live transcripts may still contain messages already
	// represented by CollapseState.Summary, so remove them before serializing.
	covered := make(map[string]struct{})
	var collapseState contextpkg.CollapseState
	if json.Unmarshal(contextState, &collapseState) == nil && strings.TrimSpace(collapseState.Summary) != "" {
		for _, id := range collapseState.CoveredMessageIDs {
			if strings.TrimSpace(id) != "" {
				covered[id] = struct{}{}
			}
		}
	}
	bounded := make([]model.Message, 0, len(messages))
	for _, message := range messages {
		_, isCovered := covered[message.ID]
		if isCovered && !checkpointMessageIsActive(message, active) {
			continue
		}
		bounded = append(bounded, message)
	}

	// A Checkpoint is a takeover view, not a second event ledger. Keep only the
	// newest completed Tool interactions. Full calls/results remain in events,
	// Artifacts and the workspace, while the execution ledger carries the latest
	// file hashes and verification facts.
	toolResults := make(map[string]model.Message)
	toolCalls := make(map[string]model.ToolCall)
	for _, message := range bounded {
		if message.Role == model.RoleAssistant {
			for _, call := range message.ToolCalls {
				toolCalls[call.ID] = call
			}
		}
		if message.Role == model.RoleTool && strings.TrimSpace(message.ToolCallID) != "" {
			toolResults[message.ToolCallID] = message
		}
	}
	type historyUnit struct {
		position int
		callID   string
		existing bool
	}
	units := make([]historyUnit, 0, len(toolResults))
	for index, message := range bounded {
		if isCheckpointToolHistory(message) {
			units = append(units, historyUnit{position: index, existing: true})
			continue
		}
		if message.Role != model.RoleTool || strings.TrimSpace(message.ToolCallID) == "" {
			continue
		}
		if _, isActive := active[message.ToolCallID]; isActive {
			continue
		}
		if _, hasCall := toolCalls[message.ToolCallID]; hasCall {
			units = append(units, historyUnit{position: index, callID: message.ToolCallID})
		}
	}
	if len(units) > checkpointRecentToolHistoryLimit {
		units = units[len(units)-checkpointRecentToolHistoryLimit:]
	}
	selectedPositions := make(map[int]struct{}, len(units))
	selectedCalls := make(map[string]struct{}, len(units))
	for _, unit := range units {
		if unit.existing {
			selectedPositions[unit.position] = struct{}{}
		} else {
			selectedCalls[unit.callID] = struct{}{}
		}
	}

	convertedCalls := make(map[string]struct{})
	droppedCalls := make(map[string]struct{})
	projected := make([]model.Message, 0, len(bounded))
	for index := range bounded {
		message := bounded[index]
		if isCheckpointToolHistory(message) {
			if _, keep := selectedPositions[index]; !keep {
				continue
			}
			projected = append(projected, canonicalCheckpointMessage(message))
			continue
		}
		message.Metadata = cloneStringMap(message.Metadata)
		message.Parts = cloneContentParts(message.Parts)
		message.ToolCalls = append([]model.ToolCall(nil), message.ToolCalls...)
		if message.Role == model.RoleAssistant && len(message.ToolCalls) != 0 {
			keptCalls := make([]model.ToolCall, 0, len(message.ToolCalls))
			historyMessages := make([]model.Message, 0, len(message.ToolCalls))
			for _, call := range message.ToolCalls {
				call.Arguments = append(json.RawMessage(nil), call.Arguments...)
				if _, keepExact := active[call.ID]; keepExact {
					keptCalls = append(keptCalls, call)
					continue
				}
				result, completed := toolResults[call.ID]
				if !completed {
					// An unmatched call may be resumable even if older Checkpoints did
					// not record it in PendingToolCalls. Never discard its arguments.
					keptCalls = append(keptCalls, call)
					continue
				}
				if _, keepRecent := selectedCalls[call.ID]; !keepRecent {
					droppedCalls[call.ID] = struct{}{}
					continue
				}
				_, callFailed := failed[call.ID]
				if callFailed || checkpointArgumentsAreVerbose(call.Name, call.Arguments) || checkpointArgumentsAreProjection(call.Arguments) {
					historyMessages = append(historyMessages, projectCheckpointToolHistory(message, call, result, callFailed))
					convertedCalls[call.ID] = struct{}{}
					continue
				}
				keptCalls = append(keptCalls, call)
			}
			projected = append(projected, historyMessages...)
			message.ToolCalls = keptCalls
			if len(keptCalls) == 0 {
				// The bounded Tool History already carries the assistant note and
				// outcome. Retaining a second prose-only copy adds attention noise.
				continue
			}
		}
		if message.Role == model.RoleTool {
			if _, converted := convertedCalls[message.ToolCallID]; converted {
				continue
			}
			if _, dropped := droppedCalls[message.ToolCallID]; dropped {
				continue
			}
		}
		if message.Role == model.RoleTool && message.Metadata != nil && message.Metadata["tool_failed"] == "true" {
			content := projectCheckpointFailureReceipt(message.Name, message.TextContent())
			message.Content = content
			message.Parts = []model.ContentPart{{Type: model.ContentJSON, JSON: json.RawMessage(content)}}
		} else if message.Role == model.RoleTool && isPlanControlTool(message.Name) {
			content := projectCheckpointPlanReceipt(message.Name, message.TextContent())
			message.Content = content
			message.Parts = []model.ContentPart{{Type: model.ContentJSON, JSON: json.RawMessage(content)}}
		}
		projected = append(projected, canonicalCheckpointMessage(message))
	}
	return projected
}

func isCheckpointToolHistory(message model.Message) bool {
	return message.Metadata != nil && message.Metadata[contextpkg.ToolHistoryMetadataKey] == "true"
}

func checkpointArgumentsAreProjection(raw json.RawMessage) bool {
	var input map[string]any
	if json.Unmarshal(raw, &input) != nil {
		return false
	}
	for _, key := range []string{"event_backed", "arguments_sha256", "plan_persisted_separately"} {
		if _, ok := input[key]; ok {
			return true
		}
	}
	return false
}

func projectCheckpointToolHistory(assistant model.Message, call model.ToolCall, result model.Message, failed bool) model.Message {
	argumentDigest := sha256.Sum256(call.Arguments)
	resultText := result.TextContent()
	resultDigest := sha256.Sum256([]byte(resultText))
	argumentFields, argumentPreview := checkpointArgumentPreview(call.Arguments)
	status := "succeeded"
	if failed {
		status = "failed"
	}
	resultPreview := truncateCheckpointText(resultText, checkpointResultPreviewChars)
	payload := map[string]any{
		"type":                     "tool_history",
		"executable":               false,
		"tool_name":                call.Name,
		"status":                   status,
		"argument_fields":          argumentFields,
		"arguments_preview":        argumentPreview,
		"arguments_sha256":         "sha256:" + hex.EncodeToString(argumentDigest[:]),
		"result_preview":           resultPreview,
		"result_chars":             len([]rune(resultText)),
		"result_sha256":            "sha256:" + hex.EncodeToString(resultDigest[:]),
		"authoritative_source":     "event_ledger_and_current_workspace",
		"recovery":                 "Read the current workspace range or durable Artifact/Event only when the full content is needed.",
		"do_not_copy_as_tool_call": true,
	}
	if failed {
		// Failure roots are commonly at the end of stderr (Python exceptions,
		// compiler summaries, test assertions). Preserve both ends and expose a
		// small structured diagnosis instead of treating a failure like ordinary
		// prefix-preview content.
		payload["result_preview"] = truncateCheckpointTextHeadTail(resultText, checkpointResultPreviewChars)
		if details := checkpointFailureDetails(resultText); len(details) != 0 {
			payload["failure_details"] = details
		}
	}
	if note := strings.TrimSpace(assistant.TextContent()); note != "" {
		payload["assistant_note"] = truncateCheckpointText(note, 240)
	}
	encoded, _ := json.Marshal(payload)
	content := "<tool-history executable=\"false\" trust=\"runtime-observation\">\n" + string(encoded) + "\n</tool-history>"
	message := model.TextMessage(model.RoleUser, content)
	message.ID = assistant.ID + ":tool-history:" + call.ID
	message.Metadata = map[string]string{
		contextpkg.ContextSectionMetadataKey: contextpkg.ContextSectionRecent,
		contextpkg.RuntimeControlMetadataKey: "true",
		contextpkg.ToolHistoryMetadataKey:    "true",
	}
	if failed {
		message.Metadata[contextpkg.ToolHistoryFailureMetadataKey] = "true"
	}
	return message
}

func checkpointArgumentPreview(raw json.RawMessage) ([]string, map[string]any) {
	var input map[string]any
	if json.Unmarshal(raw, &input) != nil {
		return nil, map[string]any{"invalid_json_preview": truncateCheckpointText(string(raw), checkpointArgumentPreviewChars)}
	}
	fields := make([]string, 0, len(input))
	preview := make(map[string]any)
	for key, value := range input {
		if key == "event_backed" || key == "arguments_sha256" || key == "plan_persisted_separately" {
			continue
		}
		fields = append(fields, key)
		switch typed := value.(type) {
		case string:
			if key == "content" || key == "text" || key == "old_text" || key == "new_text" {
				preview[key+"_preview"] = truncateCheckpointText(typed, checkpointArgumentPreviewChars)
				preview[key+"_chars"] = len([]rune(typed))
			} else {
				preview[key] = truncateCheckpointText(typed, checkpointArgumentPreviewChars)
			}
		default:
			encoded, _ := json.Marshal(value)
			if len(encoded) <= checkpointArgumentPreviewChars {
				preview[key] = value
			} else {
				preview[key+"_preview"] = truncateCheckpointText(string(encoded), checkpointArgumentPreviewChars)
			}
		}
	}
	sort.Strings(fields)
	return fields, preview
}

func checkpointMessageIsActive(message model.Message, active map[string]struct{}) bool {
	if message.ToolCallID != "" {
		if _, ok := active[message.ToolCallID]; ok {
			return true
		}
	}
	for _, call := range message.ToolCalls {
		if _, ok := active[call.ID]; ok {
			return true
		}
	}
	return false
}

func isPlanControlTool(name string) bool {
	switch name {
	case "update_plan", "update_plan_step", "revise_verification":
		return true
	default:
		return false
	}
}

func projectCheckpointPlanReceipt(name, content string) string {
	var input map[string]any
	_ = json.Unmarshal([]byte(content), &input)
	receipt := map[string]any{"tool_name": name, "plan_persisted_separately": true, "event_backed": true}
	for _, key := range []string{"plan_id", "workflow_id", "revision", "execution_outcome", "verification_outcome", "updated_step_id", "criterion_id"} {
		if value, ok := input[key]; ok {
			receipt[key] = value
		}
	}
	encoded, _ := json.Marshal(receipt)
	return string(encoded)
}

func checkpointArgumentsAreVerbose(name string, raw json.RawMessage) bool {
	if len(raw) > 2048 || name == "update_plan" {
		return true
	}
	var input map[string]any
	if json.Unmarshal(raw, &input) != nil {
		return false
	}
	for _, key := range []string{"content", "text", "old_text", "new_text"} {
		if value, ok := input[key].(string); ok && len(value) > 256 {
			return true
		}
	}
	return false
}

func projectCheckpointFailureReceipt(name, content string) string {
	var envelope map[string]any
	_ = json.Unmarshal([]byte(content), &envelope)
	receipt := map[string]any{"tool_name": name, "failure_projection": true, "event_backed": true}
	copyFailureFields := func(source map[string]any) {
		for _, key := range []string{"error_code", "failure_kind", "error", "correction", "retryable", "exit_code", "timed_out", "diagnostic", "guidance", "path", "expected", "actual"} {
			if _, exists := receipt[key]; exists {
				continue
			}
			if value, ok := source[key]; ok {
				if text, textOK := value.(string); textOK {
					receipt[key] = truncateCheckpointText(text, 320)
				} else {
					receipt[key] = value
				}
			}
		}
	}
	copyFailureFields(envelope)
	if nested, ok := envelope["content"].(map[string]any); ok {
		copyFailureFields(nested)
	}
	for _, key := range []string{"stderr", "stderr_tail", "stdout", "stdout_tail"} {
		if value, ok := envelope[key].(string); ok && strings.TrimSpace(value) != "" {
			receipt[key+"_preview"] = truncateCheckpointTextHeadTail(value, 420)
		}
	}
	encoded, _ := json.Marshal(receipt)
	if len(encoded) <= checkpointFailureReceiptMaxChars {
		return string(encoded)
	}
	minimal := map[string]any{"tool_name": name, "failure_projection": true, "event_backed": true}
	for _, key := range []string{"error_code", "correction", "path"} {
		if value, ok := receipt[key]; ok {
			minimal[key] = value
		}
	}
	encoded, _ = json.Marshal(minimal)
	return string(encoded)
}

func canonicalCheckpointMessage(message model.Message) model.Message {
	if len(message.Parts) != 0 && message.Content == message.TextContent() {
		message.Content = ""
	}
	return message
}

func restoreCheckpointMessage(message model.Message) model.Message {
	if message.Content == "" && len(message.Parts) != 0 {
		message.Content = message.TextContent()
	}
	return message
}

func cloneContentParts(parts []model.ContentPart) []model.ContentPart {
	cloned := append([]model.ContentPart(nil), parts...)
	for index := range cloned {
		cloned[index].JSON = append(json.RawMessage(nil), cloned[index].JSON...)
	}
	return cloned
}

func cloneStringMap(values map[string]string) map[string]string {
	if values == nil {
		return nil
	}
	cloned := make(map[string]string, len(values))
	for key, value := range values {
		cloned[key] = value
	}
	return cloned
}

func truncateCheckpointText(value string, limit int) string {
	runes := []rune(strings.TrimSpace(value))
	if len(runes) <= limit {
		return string(runes)
	}
	return string(runes[:limit]) + "…"
}

func truncateCheckpointTextHeadTail(value string, limit int) string {
	runes := []rune(strings.TrimSpace(value))
	if len(runes) <= limit {
		return string(runes)
	}
	if limit < 8 {
		return string(runes[:limit])
	}
	head := limit / 3
	tail := limit - head - 1
	return string(runes[:head]) + "…" + string(runes[len(runes)-tail:])
}

func checkpointFailureDetails(content string) map[string]any {
	var envelope map[string]any
	if json.Unmarshal([]byte(content), &envelope) != nil {
		return nil
	}
	details := make(map[string]any)
	for _, key := range []string{"error_code", "failure_kind", "exit_code", "timed_out", "diagnostic", "correction", "guidance"} {
		if value, ok := envelope[key]; ok {
			if text, textOK := value.(string); textOK {
				details[key] = truncateCheckpointTextHeadTail(text, 500)
			} else {
				details[key] = value
			}
		}
	}
	for _, key := range []string{"stderr_tail", "stderr", "stdout_tail", "stdout"} {
		if value, ok := envelope[key].(string); ok && strings.TrimSpace(value) != "" {
			previewKey := key
			if !strings.HasSuffix(previewKey, "_tail") {
				previewKey += "_tail"
			}
			if _, exists := details[previewKey]; !exists {
				details[previewKey] = truncateCheckpointTextHeadTail(value, 700)
			}
		}
	}
	return details
}
