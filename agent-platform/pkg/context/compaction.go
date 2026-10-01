package context

import (
	"errors"
	"fmt"
	"strings"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
)

type CompactionReport struct {
	Compacted       bool `json:"compacted"`
	BeforeTokens    int  `json:"before_tokens"`
	AfterTokens     int  `json:"after_tokens"`
	RemovedMessages int  `json:"removed_messages"`
}

// TaskAnchorMetadataKey marks the immutable user task that must survive every
// compaction generation. Role-based inference remains as a compatibility
// fallback for checkpoints created before this marker existed.
const TaskAnchorMetadataKey = "runtime.task_anchor"

// CompactMessages preserves the system contract, the original task, and the
// newest complete tool exchanges while replacing older model-visible history
// with a bounded digest. The original task is an immutable execution input: a
// lossy digest is not sufficient for paths, ranges, acceptance criteria, or
// other exact constraints needed by a long-running Agent.
func CompactMessages(messages []model.Message, budget int) ([]model.Message, CompactionReport, error) {
	report := CompactionReport{}
	for _, message := range messages {
		report.BeforeTokens += EstimateTokens(message)
	}
	if budget <= 0 {
		return nil, report, errors.New("context compaction budget must be positive")
	}
	if report.BeforeTokens <= budget {
		return append([]model.Message(nil), messages...), report, nil
	}
	prefix := 0
	for prefix < len(messages) && messages[prefix].Role == model.RoleSystem {
		prefix++
	}
	anchor := taskAnchorIndex(messages, prefix)
	if anchor < prefix || anchor >= len(messages) {
		return nil, report, ErrBudgetTooSmall
	}
	used := 0
	for index := 0; index < prefix; index++ {
		used += EstimateTokens(messages[index])
	}
	used += EstimateTokens(messages[anchor])
	if used >= budget {
		return nil, report, ErrBudgetTooSmall
	}
	summaryBudget := minInt(512, maxInt(64, budget/8))
	if remaining := budget - used; summaryBudget > remaining {
		summaryBudget = remaining
	}
	historyBudget := budget - used - summaryBudget
	if historyBudget < 0 {
		historyBudget = 0
	}
	start := len(messages)
	suffixTokens := 0
	for start > anchor+1 {
		groupStart := start - 1
		if messages[groupStart].Role == model.RoleTool {
			for groupStart > anchor+1 && messages[groupStart-1].Role == model.RoleTool {
				groupStart--
			}
			if groupStart > anchor+1 && messages[groupStart-1].Role == model.RoleAssistant {
				groupStart--
			}
		}
		cost := 0
		for index := groupStart; index < start; index++ {
			cost += EstimateTokens(messages[index])
		}
		if suffixTokens+cost > historyBudget {
			break
		}
		suffixTokens += cost
		start = groupStart
	}
	removed := make([]model.Message, 0, len(messages)-prefix)
	removed = append(removed, messages[prefix:anchor]...)
	removed = append(removed, messages[anchor+1:start]...)
	summary := buildContextDigest(removed, summaryBudget)
	result := append([]model.Message(nil), messages[:prefix]...)
	// With no execution after the anchor, keep the current user objective last.
	// Once execution exists, retain chronological task -> digest -> recent tail.
	if summary != "" && anchor == len(messages)-1 {
		result = append(result, model.TextMessage(model.RoleUser, summary))
	}
	result = append(result, messages[anchor])
	if summary != "" && anchor != len(messages)-1 {
		result = append(result, model.TextMessage(model.RoleUser, summary))
	}
	result = append(result, messages[start:]...)
	for _, message := range result {
		report.AfterTokens += EstimateTokens(message)
	}
	if report.AfterTokens > budget {
		return nil, report, ErrBudgetTooSmall
	}
	report.Compacted = true
	report.RemovedMessages = len(removed)
	return result, report, nil
}

// compactAroundTaskAnchor is the safe fallback when the newest assistant/tool
// exchange alone exceeds the message budget. It preserves the original task
// that led to the first real Tool Call and summarizes every execution message,
// including the oversized exchange. Durable plan/checkpoint state remains the
// source of truth for exact progress.
func compactAroundTaskAnchor(messages []model.Message, prefix, budget int, report CompactionReport) ([]model.Message, CompactionReport, error) {
	anchor := taskAnchorIndex(messages, prefix)
	if anchor < prefix || anchor >= len(messages) {
		return nil, report, ErrBudgetTooSmall
	}
	used := 0
	result := append([]model.Message(nil), messages[:prefix]...)
	for _, message := range result {
		used += EstimateTokens(message)
	}
	anchorCost := EstimateTokens(messages[anchor])
	if used+anchorCost >= budget {
		return nil, report, ErrBudgetTooSmall
	}
	removed := make([]model.Message, 0, len(messages)-prefix-1)
	removed = append(removed, messages[prefix:anchor]...)
	removed = append(removed, messages[anchor+1:]...)
	summaryBudget := budget - used - anchorCost
	if summaryBudget > 768 {
		summaryBudget = 768
	}
	if summaryBudget >= 64 {
		if summary := buildContextDigest(removed, summaryBudget); summary != "" {
			result = append(result, model.TextMessage(model.RoleUser, summary))
		}
	}
	report.AfterTokens = 0
	for _, message := range result {
		report.AfterTokens += EstimateTokens(message)
	}
	if report.AfterTokens > budget {
		return nil, report, ErrBudgetTooSmall
	}
	report.Compacted = true
	report.RemovedMessages = len(removed)
	return result, report, nil
}

func taskAnchorIndex(messages []model.Message, prefix int) int {
	for index := prefix; index < len(messages); index++ {
		if messages[index].Metadata[TaskAnchorMetadataKey] == "true" {
			return index
		}
	}
	firstToolCall := len(messages)
	for index := prefix; index < len(messages); index++ {
		if len(messages[index].ToolCalls) > 0 {
			firstToolCall = index
			break
		}
	}
	for index := firstToolCall - 1; index >= prefix; index-- {
		if messages[index].Role == model.RoleUser {
			return index
		}
	}
	for index := len(messages) - 1; index >= prefix; index-- {
		if messages[index].Role == model.RoleUser {
			return index
		}
	}
	return -1
}

func buildContextDigest(messages []model.Message, budget int) string {
	if len(messages) == 0 {
		return ""
	}
	const header = "<context_summary>Older execution history was compacted. Preserve these facts and continue the current task:\n"
	content := header
	for _, message := range messages {
		text := strings.TrimSpace(message.TextContent())
		if strings.HasPrefix(text, "<context_summary>") || strings.HasPrefix(text, "<context_collapse") {
			// A previous deterministic compaction is itself state, not disposable
			// noise. Keep a bounded representation so repeated fallback compaction
			// cannot erase every fact from earlier generations.
			text = "previous context summary: " + text
		}
		runes := []rune(text)
		if len(runes) > 180 {
			text = string(runes[:180]) + "…"
		}
		if len(message.ToolCalls) > 0 {
			names := make([]string, 0, len(message.ToolCalls))
			for _, call := range message.ToolCalls {
				names = append(names, call.Name)
			}
			text = "requested tools: " + strings.Join(names, ", ") + "; " + text
		}
		line := fmt.Sprintf("- %s%s: %s\n", message.Role, toolLabel(message), text)
		candidate := model.TextMessage(model.RoleUser, content+line+"</context_summary>")
		if EstimateTokens(candidate) > budget {
			break
		}
		content += line
	}
	if content == header {
		content += fmt.Sprintf("- %d older messages omitted; rely on the durable plan and recent tool results.\n", len(messages))
	}
	return strings.TrimSpace(content) + "\n</context_summary>"
}

func toolLabel(message model.Message) string {
	if message.Name == "" {
		return ""
	}
	return "[" + message.Name + "]"
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
