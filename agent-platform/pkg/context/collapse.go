package context

import (
	stdcontext "context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"sort"
	"strings"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
)

// CollapseState is the durable state of a read-time Context Collapse. The
// complete message history is intentionally not stored here: this state only
// tells the projector which bounded summary is already available.
type CollapseState struct {
	Generation                 int      `json:"generation,omitempty"`
	Summary                    string   `json:"summary,omitempty"`
	CoveredMessageIDs          []string `json:"covered_message_ids,omitempty"`
	CoveredMessageCount        int      `json:"covered_message_count,omitempty"`
	SourceHash                 string   `json:"source_hash,omitempty"`
	FailureStreak              int      `json:"failure_streak,omitempty"`
	LastCompactionMessageCount int      `json:"last_compaction_message_count,omitempty"`
	LastCompactionBeforeTokens int      `json:"last_compaction_before_tokens,omitempty"`
	// SurfacedMemories records body revisions already shown to the model. It
	// lives beside the projection state so a continuation Run does not inject
	// the same historical fact on every turn.
	SurfacedMemories   []SurfacedMemory `json:"surfaced_memories,omitempty"`
	RecentToolEvidence []string         `json:"recent_tool_evidence,omitempty"`
	// MemoryExtractionRunID owns LastMemoryExtractionSequence. Event sequence
	// numbers are monotonic only inside one Run, so comparing a continuation
	// Run's sequence with the prior attempt's cursor can produce an impossible
	// source range (for example 537..158) and silently skip memory extraction.
	MemoryExtractionRunID        string `json:"memory_extraction_run_id,omitempty"`
	LastMemoryExtractionSequence int64  `json:"last_memory_extraction_sequence,omitempty"`
}

// MemoryState is the memory subsystem's independently-owned slice of the
// durable context state.  Collapse and memory used to mutate separate copies
// of CollapseState and then replace the complete JSON blob.  A tool-failure
// lookup could therefore restore an older collapse generation and make the
// same message range eligible for compaction again.  Keeping an explicit
// owned slice lets callers merge memory changes into the latest projection
// state instead of performing a last-writer-wins replacement.
type MemoryState struct {
	SurfacedMemories             []SurfacedMemory `json:"surfaced_memories,omitempty"`
	RecentToolEvidence           []string         `json:"recent_tool_evidence,omitempty"`
	MemoryExtractionRunID        string           `json:"memory_extraction_run_id,omitempty"`
	LastMemoryExtractionSequence int64            `json:"last_memory_extraction_sequence,omitempty"`
}

// MemoryState returns a detached copy of the memory-owned fields.
func (s CollapseState) MemoryState() MemoryState {
	return MemoryState{
		SurfacedMemories:             append([]SurfacedMemory(nil), s.SurfacedMemories...),
		RecentToolEvidence:           append([]string(nil), s.RecentToolEvidence...),
		MemoryExtractionRunID:        s.MemoryExtractionRunID,
		LastMemoryExtractionSequence: s.LastMemoryExtractionSequence,
	}
}

// MergeMemoryState applies only memory-owned fields. Collapse generation,
// summary, coverage and cooldown fields always remain owned by the context
// projector and can no longer be rolled back by a memory callback.
func (s *CollapseState) MergeMemoryState(memory MemoryState) {
	if s == nil {
		return
	}
	s.SurfacedMemories = append([]SurfacedMemory(nil), memory.SurfacedMemories...)
	s.RecentToolEvidence = append([]string(nil), memory.RecentToolEvidence...)
	if memory.MemoryExtractionRunID != "" && memory.MemoryExtractionRunID != s.MemoryExtractionRunID {
		s.MemoryExtractionRunID = memory.MemoryExtractionRunID
		s.LastMemoryExtractionSequence = memory.LastMemoryExtractionSequence
		return
	}
	if s.MemoryExtractionRunID == "" && memory.MemoryExtractionRunID != "" {
		s.MemoryExtractionRunID = memory.MemoryExtractionRunID
	}
	if memory.LastMemoryExtractionSequence > s.LastMemoryExtractionSequence {
		s.LastMemoryExtractionSequence = memory.LastMemoryExtractionSequence
	}
}

// RebaseMemoryExtractionCursor binds the Run-local extraction cursor to the
// current attempt. preserveUnknown is used only for a same-attempt takeover of
// a checkpoint written before MemoryExtractionRunID existed.
func (s *MemoryState) RebaseMemoryExtractionCursor(runID string, preserveUnknown bool) {
	if s == nil || strings.TrimSpace(runID) == "" {
		return
	}
	if s.MemoryExtractionRunID == "" && preserveUnknown {
		s.MemoryExtractionRunID = runID
		return
	}
	if s.MemoryExtractionRunID != runID {
		s.MemoryExtractionRunID = runID
		s.LastMemoryExtractionSequence = 0
	}
}

type SurfacedMemory struct {
	ID          string `json:"id"`
	RevisionKey string `json:"revision_key,omitempty"`
	Turn        int    `json:"turn,omitempty"`
}

// RebaseForContinuation prepares collapse state for a new Run attempt in the
// same Workflow. The complete message history remains the durable source of
// truth, but the previous model projection is still reusable: Summary and
// CoveredMessageIDs describe an exact message-ID boundary and therefore do
// not depend on the physical Run ID. Keeping that boundary prevents a
// continuation from re-collapsing the entire historical ledger from scratch.
// Only attempt-local cooldown/failure counters are reset. Callers that detect
// a source-hash mismatch or a changed task must clear the projection before
// calling this method.
func (s *CollapseState) RebaseForContinuation() {
	if s == nil {
		return
	}
	s.FailureStreak = 0
	s.LastCompactionMessageCount = 0
	s.LastCompactionBeforeTokens = 0
}

type ProjectionReport struct {
	Projected    bool `json:"projected"`
	Compacted    bool `json:"compacted"`
	BeforeTokens int  `json:"before_tokens"`
	// WorkingBeforeTokens excludes messages already covered by Summary and
	// CoveredMessageIDs. BeforeTokens is retained for compatibility but counts
	// the complete durable replay assembled before read-time projection.
	WorkingBeforeTokens    int    `json:"working_before_tokens,omitempty"`
	AfterTokens            int    `json:"after_tokens"`
	RemovedMessages        int    `json:"removed_messages"`
	SummaryTokens          int    `json:"summary_tokens"`
	Generation             int    `json:"generation"`
	ProtectedTokens        int    `json:"protected_tokens,omitempty"`
	ProtectedFailureTokens int    `json:"protected_failure_tokens,omitempty"`
	SummaryMode            string `json:"summary_mode,omitempty"`
}

type SummaryInput struct {
	Generation int
	Previous   string
	Messages   []model.Message
	Budget     int
}

// CollapseBarrierInput identifies the exact message range that is about to
// leave the model-facing projection. A Memory Writer may persist an
// extraction job before the range is summarized, preventing Collapse from
// becoming an irreversible information-loss boundary.
type CollapseBarrierInput struct {
	Generation int
	SourceHash string
	Messages   []model.Message
}

type ProjectionOptions struct {
	TriggerRatio                float64
	RecentTurnTokens            int
	MemoryTokens                int
	KnowledgeTokens             int
	ToolResultTokens            int
	SummaryTokens               int
	TargetRatio                 float64
	MinMessagesBetweenCollapses int
	MinTokensBetweenCollapses   int
	Summarize                   func(stdcontext.Context, SummaryInput) (string, error)
	BeforeCollapse              func(stdcontext.Context, CollapseBarrierInput) error
}

var ErrProjectionBudgetTooSmall = errors.New("context collapse cannot fit protected messages")

// StripRebuildableRuntimeBlocks removes model-facing runtime projections while
// leaving the surrounding system message, IDs, metadata, and durable history
// intact. Plan, tool-capability, and execution-ledger blocks are regenerated
// from authoritative runtime state after Collapse; retaining old copies as
// protected system text would make the protected budget grow without bound.
func StripRebuildableRuntimeBlocks(messages []model.Message) []model.Message {
	result := make([]model.Message, 0, len(messages))
	for _, message := range messages {
		// Runtime blocks are injected only into the system projection. Never
		// interpret marker-looking text in a user's task or a Tool result as
		// runtime state; those messages are part of the durable task evidence.
		if message.Role != model.RoleSystem {
			result = append(result, message)
			continue
		}
		text := message.TextContent()
		for _, marker := range rebuildableRuntimeMarkers {
			text = removeDelimitedBlock(text, marker.start, marker.end)
		}
		if message.Role == model.RoleSystem && strings.TrimSpace(text) == "" {
			continue
		}
		if text == message.TextContent() {
			result = append(result, message)
			continue
		}
		updated := message
		updated.Content = text
		updated.Parts = []model.ContentPart{{Type: model.ContentText, Text: text}}
		result = append(result, updated)
	}
	return result
}

type runtimeBlockMarker struct {
	start string
	end   string
}

var rebuildableRuntimeMarkers = []runtimeBlockMarker{
	{start: "<RUNTIME_DURABLE_PLAN>", end: "</RUNTIME_DURABLE_PLAN>"},
	{start: "<RUNTIME_TOOL_PROJECTION>", end: "</RUNTIME_TOOL_PROJECTION>"},
	{start: "<RUNTIME_EXECUTION_LEDGER>", end: "</RUNTIME_EXECUTION_LEDGER>"},
}

func removeDelimitedBlock(text, start, end string) string {
	for {
		begin := strings.Index(text, start)
		if begin < 0 {
			return text
		}
		closeOffset := strings.Index(text[begin+len(start):], end)
		if closeOffset < 0 {
			return strings.TrimSpace(text[:begin])
		}
		closeOffset += begin + len(start) + len(end)
		text = strings.TrimSpace(text[:begin] + text[closeOffset:])
	}
}

// EnsureMessageIDs fills IDs only when a message does not already have one.
// IDs are stable across retries and checkpoints because callers pass a stable
// run/workflow prefix and existing IDs are never rewritten.
func EnsureMessageIDs(messages []model.Message, prefix string) []model.Message {
	result := append([]model.Message(nil), messages...)
	for index := range result {
		if strings.TrimSpace(result[index].ID) != "" {
			continue
		}
		result[index].ID = fmt.Sprintf("%s:message:%d", prefix, index)
	}
	return result
}

// ProjectMessages constructs a bounded model-facing view without mutating the
// durable message slice. It keeps all system and user messages exact, keeps the
// newest complete assistant/tool groups, and replaces only older execution
// groups with one cached summary. Semantic wording is optional; the bounded
// deterministic summary remains the fallback contract.
func ProjectMessages(messages []model.Message, budget int, state CollapseState) ([]model.Message, CollapseState, ProjectionReport, error) {
	return ProjectMessagesWithOptions(stdcontext.Background(), messages, budget, state, ProjectionOptions{})
}

// ProjectMessagesWithOptions is the policy-aware Context Collapse entrypoint.
// It can proactively collapse before the hard budget and optionally delegate
// summary wording to a model while retaining a deterministic fallback.
func ProjectMessagesWithOptions(ctx stdcontext.Context, messages []model.Message, budget int, state CollapseState, options ProjectionOptions) ([]model.Message, CollapseState, ProjectionReport, error) {
	report := ProjectionReport{}
	baseMessages := stripCollapseSummaries(messages)
	for _, message := range baseMessages {
		report.BeforeTokens += EstimateTokens(message)
	}
	if budget <= 0 {
		return nil, state, report, errors.New("context projection budget must be positive")
	}
	// CoveredMessageIDs are the durable boundary of the previous rolling
	// summary.  They must not be fed through the selector again: the complete
	// messages remain durable, while the model-facing projection is allowed to
	// represent that range by state.Summary.  Legacy states without a summary
	// or coverage continue through the full-history path below.
	workingMessages := baseMessages
	hasCoveredProjection := false
	if strings.TrimSpace(state.Summary) != "" && len(state.CoveredMessageIDs) != 0 {
		workingMessages = excludeCoveredMessages(baseMessages, state.CoveredMessageIDs)
		// A storage checkpoint may already have removed covered messages. The
		// durable state is still sufficient to restore the summary, so coverage
		// must not depend on finding those physical messages again.
		hasCoveredProjection = true
	}
	// Failure groups are protected for recovery, but their original assistant
	// tool arguments may contain an entire source file or Plan. Keep only a
	// bounded, structured receipt in the model projection; the complete call
	// and result remain in the durable event/checkpoint history.
	workingMessages = projectProtectedFailureMessages(workingMessages)
	workingBeforeTokens := 0
	for _, message := range workingMessages {
		workingBeforeTokens += EstimateTokens(message)
	}
	report.WorkingBeforeTokens = workingBeforeTokens

	triggerBudget := budget
	if options.TriggerRatio > 0 && options.TriggerRatio < 1 {
		triggerBudget = int(float64(budget) * options.TriggerRatio)
		if triggerBudget < 1 {
			triggerBudget = 1
		}
	}
	// If the uncovered tail still fits, reuse the existing summary and avoid a
	// second collapse over the same durable messages. This is the main
	// difference between an incremental collapse and re-compacting the complete
	// history on every model request.
	if hasCoveredProjection && workingBeforeTokens <= triggerBudget {
		projected := insertReusedSummary(workingMessages, state.Summary)
		for _, message := range projected {
			report.AfterTokens += EstimateTokens(message)
		}
		if report.AfterTokens <= budget {
			report.Projected = true
			report.Generation = state.Generation
			report.SummaryTokens = EstimateTokens(model.TextMessage(model.RoleUser, state.Summary))
			return projected, state, report, nil
		}
	}
	if workingBeforeTokens <= triggerBudget && !hasCoveredProjection {
		return append([]model.Message(nil), baseMessages...), state, report, nil
	}
	hardBudgetExceeded := workingBeforeTokens > budget
	if !hardBudgetExceeded && state.Generation > 0 &&
		((options.MinMessagesBetweenCollapses > 0 && len(workingMessages)-state.LastCompactionMessageCount < options.MinMessagesBetweenCollapses) ||
			(options.MinTokensBetweenCollapses > 0 && workingBeforeTokens-state.LastCompactionBeforeTokens < options.MinTokensBetweenCollapses)) {
		if !hasCoveredProjection {
			return append([]model.Message(nil), baseMessages...), state, report, nil
		}
		projected := insertReusedSummary(workingMessages, state.Summary)
		for _, message := range projected {
			report.AfterTokens += EstimateTokens(message)
		}
		if report.AfterTokens <= budget {
			report.Projected = true
			report.Generation = state.Generation
			return projected, state, report, nil
		}
	}

	groups := collapseGroups(workingMessages)
	summaryBudget := options.SummaryTokens
	if summaryBudget <= 0 {
		summaryBudget = minInt(1024, maxInt(256, budget/8))
	}
	if summaryBudget > budget {
		summaryBudget = budget
	}
	protectedCost := 0
	failureGroups := newestFailureGroups(groups, 2)
	for index := range groups {
		if failureGroups[index] {
			groups[index].protected = true
		}
	}
	for _, group := range groups {
		if group.protected {
			protectedCost += group.tokens
			report.ProtectedTokens += group.tokens
			if group.failure {
				report.ProtectedFailureTokens += group.tokens
			}
		}
	}
	// The configured summary reserve is an upper bound, not an additional
	// mandatory block. Small-context models and test/sandbox profiles can have
	// a large protected prompt relative to that reserve; shrink the summary to
	// the actual remainder so collapse can still preserve the protected prompt
	// and the newest execution evidence.
	if remaining := budget - protectedCost; remaining >= 0 && summaryBudget > remaining {
		summaryBudget = remaining
	}
	if protectedCost+summaryBudget > budget {
		return nil, state, report, ErrProjectionBudgetTooSmall
	}
	selectionBudget := budget
	if options.TargetRatio > 0 && options.TargetRatio < 1 {
		selectionBudget = int(float64(budget) * options.TargetRatio)
		minimum := protectedCost + summaryBudget
		if selectionBudget < minimum {
			selectionBudget = minimum
		}
		if selectionBudget > budget {
			selectionBudget = budget
		}
	}

	// Select newest execution groups first. Protected system/user messages are
	// always retained, so an exact user constraint can never be summarized away.
	keep := make(map[int]bool, len(groups))
	used := protectedCost
	recentBudget := options.RecentTurnTokens
	if recentBudget <= 0 {
		recentBudget = budget - protectedCost - summaryBudget
	}
	if recentBudget < 0 {
		recentBudget = 0
	}
	usedRecent := 0
	usedToolResults := 0
	usedMemory := 0
	usedKnowledge := 0
	for index := len(groups) - 1; index >= 0; index-- {
		group := groups[index]
		if group.protected {
			keep[index] = true
			continue
		}
		if group.section == ContextSectionMemory && options.MemoryTokens > 0 && usedMemory+group.tokens > options.MemoryTokens {
			continue
		}
		if group.section == ContextSectionKnowledge && options.KnowledgeTokens > 0 && usedKnowledge+group.tokens > options.KnowledgeTokens {
			continue
		}
		if recentBudget > 0 && usedRecent+group.tokens > recentBudget {
			continue
		}
		if options.ToolResultTokens > 0 && usedToolResults+group.toolTokens > options.ToolResultTokens {
			continue
		}
		if used+group.tokens+summaryBudget <= selectionBudget {
			keep[index] = true
			used += group.tokens
			usedRecent += group.tokens
			usedToolResults += group.toolTokens
			if group.section == ContextSectionMemory {
				usedMemory += group.tokens
			}
			if group.section == ContextSectionKnowledge {
				usedKnowledge += group.tokens
			}
		}
	}

	removed := make([]model.Message, 0)
	removedIDs := make([]string, 0)
	firstRemoved := -1
	for index, group := range groups {
		if keep[index] {
			continue
		}
		if firstRemoved < 0 {
			firstRemoved = group.start
		}
		removed = append(removed, workingMessages[group.start:group.end]...)
		for _, message := range workingMessages[group.start:group.end] {
			removedIDs = append(removedIDs, messageID(message))
		}
	}
	if len(removed) == 0 {
		if workingBeforeTokens > budget {
			return nil, state, report, ErrProjectionBudgetTooSmall
		}
		projected := insertReusedSummary(workingMessages, state.Summary)
		return projected, state, report, nil
	}

	// Only retain coverage IDs whose physical messages still exist in this
	// transcript. Once a compacted Live State/Checkpoint has discarded an old
	// range, its prior summary carries the information and those IDs no longer
	// need to occupy the boundary ledger. This also guarantees that a new
	// removed range changes SourceHash after the 256-entry bound is reached.
	coveredIDs := mergeIDs(presentMessageIDs(baseMessages, state.CoveredMessageIDs), removedIDs)
	sourceHash := hashIDs(coveredIDs)
	if state.SourceHash != sourceHash || strings.TrimSpace(state.Summary) == "" {
		if options.BeforeCollapse != nil {
			if err := options.BeforeCollapse(ctx, CollapseBarrierInput{Generation: state.Generation + 1, SourceHash: sourceHash, Messages: append([]model.Message(nil), removed...)}); err != nil {
				return nil, state, report, fmt.Errorf("collapse barrier: %w", err)
			}
		}
		state.Generation++
		state.SourceHash = sourceHash
		state.Summary, report.SummaryMode = buildCollapseSummaryWithOptions(ctx, removed, state.Summary, summaryBudget, state.Generation, options)
		state.CoveredMessageIDs = coveredIDs
		state.CoveredMessageCount = len(coveredIDs)
		state.LastCompactionMessageCount = len(workingMessages)
		state.LastCompactionBeforeTokens = workingBeforeTokens
		report.Compacted = true
		report.RemovedMessages = len(removed)
	}

	result := make([]model.Message, 0, len(workingMessages)-len(removed)+1)
	inserted := false
	for index, group := range groups {
		if !inserted && group.start == firstRemoved {
			result = append(result, collapseSummaryMessage(state.Summary))
			inserted = true
		}
		if keep[index] {
			result = append(result, workingMessages[group.start:group.end]...)
		}
	}
	if !inserted {
		result = append(result, collapseSummaryMessage(state.Summary))
	}
	for _, message := range result {
		report.AfterTokens += EstimateTokens(message)
	}
	if report.AfterTokens > budget {
		return nil, state, report, ErrProjectionBudgetTooSmall
	}
	report.Projected = true
	report.SummaryTokens = EstimateTokens(model.TextMessage(model.RoleUser, state.Summary))
	report.Generation = state.Generation
	return result, state, report, nil
}

type collapseGroup struct {
	start      int
	end        int
	tokens     int
	toolTokens int
	section    string
	protected  bool
	failure    bool
}

func collapseGroups(messages []model.Message) []collapseGroup {
	groups := make([]collapseGroup, 0, len(messages))
	for index := 0; index < len(messages); {
		start := index
		section := ""
		if messages[index].Metadata != nil {
			section = strings.TrimSpace(messages[index].Metadata[ContextSectionMetadataKey])
			// Static Auto/Team material is historical knowledge, not a
			// mandatory instruction section. Configured Managed/User/Project/
			// Local messages remain protected by their System role.
			if section == "static_memory" {
				section = ContextSectionKnowledge
			}
		}
		runtimeControl := messages[index].Metadata != nil && messages[index].Metadata[RuntimeControlMetadataKey] == "true"
		protected := messages[index].Role == model.RoleSystem || (messages[index].Role == model.RoleUser && !runtimeControl && section != ContextSectionMemory && section != ContextSectionKnowledge && section != ContextSectionSummary && section != ContextSectionRuntime)
		if messages[index].Role == model.RoleAssistant {
			index++
			for index < len(messages) && messages[index].Role == model.RoleTool {
				index++
			}
		} else {
			index++
		}
		cost := 0
		toolCost := 0
		for cursor := start; cursor < index; cursor++ {
			cost += EstimateTokens(messages[cursor])
			if messages[cursor].Role == model.RoleTool || (messages[cursor].Metadata != nil && messages[cursor].Metadata[ToolHistoryMetadataKey] == "true") {
				toolCost += EstimateTokens(messages[cursor])
			}
		}
		failure := groupContainsToolFailure(messages[start:index])
		groups = append(groups, collapseGroup{start: start, end: index, tokens: cost, toolTokens: toolCost, section: section, protected: protected, failure: failure})
	}
	return groups
}

const (
	maxFailureArgumentsChars = 320
	maxFailureReceiptChars   = 1000
)

func projectProtectedFailureMessages(messages []model.Message) []model.Message {
	groups := collapseGroups(messages)
	failureGroups := newestFailureGroups(groups, 2)
	if len(failureGroups) == 0 {
		return messages
	}
	projected := append([]model.Message(nil), messages...)
	for index, group := range groups {
		if !failureGroups[index] {
			continue
		}
		for position := group.start; position < group.end; position++ {
			message := projected[position]
			if message.Role == model.RoleAssistant && len(message.ToolCalls) != 0 {
				message.ToolCalls = append([]model.ToolCall(nil), message.ToolCalls...)
				for callIndex := range message.ToolCalls {
					call := &message.ToolCalls[callIndex]
					call.Arguments = compactFailureArguments(call.Name, call.Arguments)
				}
				message.Content = "Previous failed tool-call arguments were omitted; use the structured failure receipt below."
				message.Parts = []model.ContentPart{{Type: model.ContentText, Text: message.Content}}
			}
			if message.Role == model.RoleTool {
				message.Content = string(compactFailureReceipt(message.Name, message.Content, message.Parts))
				message.Parts = []model.ContentPart{{Type: model.ContentJSON, JSON: json.RawMessage(message.Content)}}
			}
			projected[position] = message
		}
	}
	return projected
}

func compactFailureArguments(name string, raw json.RawMessage) json.RawMessage {
	var input map[string]any
	compact := make(map[string]any)
	if json.Unmarshal(raw, &input) == nil {
		for _, key := range []string{"path", "source_path", "target_path", "start_line", "line_count", "step_id", "criterion_id", "command", "timeout_seconds", "overwrite"} {
			if value, ok := input[key]; ok {
				compact[key] = value
			}
		}
	}
	encoded, err := json.Marshal(compact)
	if err != nil {
		return json.RawMessage(`{}`)
	}
	if len(encoded) > maxFailureArgumentsChars {
		encoded = []byte(`{}`)
	}
	return encoded
}

func compactFailureReceipt(name, content string, parts []model.ContentPart) json.RawMessage {
	if content == "" && len(parts) != 0 {
		content = model.Message{Parts: parts}.TextContent()
	}
	var envelope map[string]any
	_ = json.Unmarshal([]byte(content), &envelope)
	receipt := map[string]any{"tool_name": name, "failure_projection": true}
	for _, key := range []string{"error_code", "error", "correction", "retryable", "path", "expected", "actual"} {
		if value, ok := envelope[key]; ok {
			receipt[key] = value
		}
	}
	if nested, ok := envelope["content"].(map[string]any); ok {
		for _, key := range []string{"error_code", "error", "correction", "retryable", "path", "expected", "actual"} {
			if _, exists := receipt[key]; exists {
				continue
			}
			if value, ok := nested[key]; ok {
				receipt[key] = value
			}
		}
	}
	encoded, err := json.Marshal(receipt)
	if err != nil {
		return json.RawMessage(`{"failure_projection":true}`)
	}
	if len(encoded) <= maxFailureReceiptChars {
		return encoded
	}
	// Error text is provider-controlled. Keep the stable repair fields and
	// bound prose instead of allowing a provider stack trace to become a
	// protected context block.
	bounded := map[string]any{"tool_name": name, "failure_projection": true}
	for _, key := range []string{"error_code", "correction", "retryable", "path"} {
		if value, ok := receipt[key]; ok {
			if text, ok := value.(string); ok {
				bounded[key] = compactFailureText(text, 240)
			} else {
				bounded[key] = value
			}
		}
	}
	encoded, _ = json.Marshal(bounded)
	return encoded
}

func compactFailureText(value string, limit int) string {
	runes := []rune(strings.TrimSpace(value))
	if len(runes) <= limit {
		return string(runes)
	}
	return string(runes[:limit]) + "…"
}

func groupContainsToolFailure(messages []model.Message) bool {
	for _, message := range messages {
		if message.Metadata == nil {
			continue
		}
		if (message.Role == model.RoleTool && message.Metadata["tool_failed"] == "true") || message.Metadata[ToolHistoryFailureMetadataKey] == "true" {
			return true
		}
	}
	return false
}

func newestFailureGroups(groups []collapseGroup, limit int) map[int]bool {
	selected := make(map[int]bool)
	if limit <= 0 {
		return selected
	}
	for index := len(groups) - 1; index >= 0 && limit > 0; index-- {
		if !groups[index].failure {
			continue
		}
		selected[index] = true
		limit--
	}
	return selected
}

func collapseSummaryMessage(summary string) model.Message {
	message := model.TextMessage(model.RoleUser, summary)
	message.Metadata = map[string]string{ContextSectionMetadataKey: ContextSectionSummary}
	return message
}

func isCollapseSummaryMessage(message model.Message) bool {
	section := ""
	if message.Metadata != nil {
		section = strings.TrimSpace(message.Metadata[ContextSectionMetadataKey])
	}
	text := strings.TrimSpace(message.TextContent())
	return section == ContextSectionSummary || strings.HasPrefix(text, "<context_collapse") || strings.HasPrefix(text, "<context_summary")
}

// stripCollapseSummaries ensures durable history remains authoritative and
// prevents nested summaries from becoming ordinary user instructions on a
// retry or continuation.
func stripCollapseSummaries(messages []model.Message) []model.Message {
	result := make([]model.Message, 0, len(messages))
	for _, message := range messages {
		if isCollapseSummaryMessage(message) {
			continue
		}
		result = append(result, message)
	}
	return result
}

func excludeCoveredMessages(messages []model.Message, covered []string) []model.Message {
	if len(covered) == 0 {
		return append([]model.Message(nil), messages...)
	}
	set := make(map[string]struct{}, len(covered))
	for _, id := range covered {
		if strings.TrimSpace(id) != "" {
			set[id] = struct{}{}
		}
	}
	result := make([]model.Message, 0, len(messages))
	for _, message := range messages {
		if _, ok := set[messageID(message)]; ok {
			continue
		}
		result = append(result, message)
	}
	return result
}

// insertReusedSummary reconstructs the model-facing view from the previous
// rolling summary and the uncovered tail. The complete durable history is not
// changed; only the old covered range is omitted from this read-time view.
func insertReusedSummary(messages []model.Message, summary string) []model.Message {
	if strings.TrimSpace(summary) == "" {
		return append([]model.Message(nil), messages...)
	}
	result := make([]model.Message, 0, len(messages)+1)
	inserted := false
	for _, message := range messages {
		if !inserted && (message.Role == model.RoleAssistant || message.Role == model.RoleTool) {
			result = append(result, collapseSummaryMessage(summary))
			inserted = true
		}
		result = append(result, message)
	}
	if !inserted {
		result = append(result, collapseSummaryMessage(summary))
	}
	return result
}

func buildCollapseSummary(messages []model.Message, previous string, budget, generation int) string {
	summary, _ := buildCollapseSummaryWithOptions(stdcontext.Background(), messages, previous, budget, generation, ProjectionOptions{})
	return summary
}

func buildCollapseSummaryWithOptions(ctx stdcontext.Context, messages []model.Message, previous string, budget, generation int, options ProjectionOptions) (string, string) {
	messages = filterCollapseSummaryInputs(messages)
	if options.Summarize != nil {
		semantic, err := options.Summarize(ctx, SummaryInput{Generation: generation, Previous: previous, Messages: append([]model.Message(nil), messages...), Budget: budget})
		semantic = strings.TrimSpace(semantic)
		if err == nil && semantic != "" {
			if !strings.Contains(semantic, "<context_collapse") {
				semantic = fmt.Sprintf("<context_collapse generation=\"%d\">\n%s\n</context_collapse>", generation, semantic)
			}
			semantic = preserveCanonicalToolPaths(semantic, previous, messages, budget, generation)
			if EstimateTokens(model.TextMessage(model.RoleUser, semantic)) <= budget {
				return semantic, "semantic"
			}
		}
	}
	return preserveCanonicalToolPaths(buildDeterministicCollapseSummary(messages, previous, budget, generation), previous, messages, budget, generation), "deterministic"
}

func preserveCanonicalToolPaths(summary, previous string, messages []model.Message, budget, generation int) string {
	seen := make(map[string]struct{})
	paths := make([]string, 0, 16)
	addPath := func(value string) {
		value = strings.TrimSpace(value)
		if value == "" || strings.HasPrefix(value, "/") || path.Clean(value) != value || value == "." || value == ".." || strings.HasPrefix(value, "../") {
			return
		}
		if _, ok := seen[value]; ok {
			return
		}
		seen[value] = struct{}{}
		paths = append(paths, value)
	}
	for _, source := range []string{previous, summary} {
		start := strings.Index(source, "<canonical_workspace_paths>")
		end := strings.Index(source, "</canonical_workspace_paths>")
		if start < 0 || end <= start {
			continue
		}
		start += len("<canonical_workspace_paths>")
		for _, value := range strings.Split(source[start:end], "\n") {
			addPath(value)
		}
	}
	for _, message := range messages {
		for _, call := range message.ToolCalls {
			var arguments map[string]any
			if json.Unmarshal(call.Arguments, &arguments) != nil {
				continue
			}
			for _, key := range []string{"path", "target_path", "working_directory"} {
				value, _ := arguments[key].(string)
				addPath(value)
			}
		}
	}
	if len(paths) == 0 {
		return summary
	}
	if len(paths) > 24 {
		paths = paths[len(paths)-24:]
	}
	sort.Strings(paths)
	block := "<canonical_workspace_paths>\n" + strings.Join(paths, "\n") + "\n</canonical_workspace_paths>"
	body := strings.TrimSpace(summary)
	body = strings.TrimPrefix(body, fmt.Sprintf("<context_collapse generation=\"%d\">", generation))
	body = strings.TrimSuffix(strings.TrimSpace(body), "</context_collapse>")
	if start := strings.Index(body, "<canonical_workspace_paths>"); start >= 0 {
		if end := strings.Index(body[start:], "</canonical_workspace_paths>"); end >= 0 {
			end += start + len("</canonical_workspace_paths>")
			body = strings.TrimSpace(body[:start] + body[end:])
		}
	}
	result := fmt.Sprintf("<context_collapse generation=\"%d\">\n%s\n%s\n</context_collapse>", generation, block, body)
	for EstimateTokens(model.TextMessage(model.RoleUser, result)) > budget && len([]rune(body)) > 64 {
		runes := []rune(body)
		body = string(runes[:len(runes)*3/4]) + "…"
		result = fmt.Sprintf("<context_collapse generation=\"%d\">\n%s\n%s\n</context_collapse>", generation, block, body)
	}
	return result
}

func filterCollapseSummaryInputs(messages []model.Message) []model.Message {
	filtered := make([]model.Message, 0, len(messages))
	for _, message := range messages {
		text := message.TextContent()
		// Durable Plan, execution-ledger, and tool-projection blocks are
		// refreshed independently on every request. Folding them into a rolling
		// historical summary is what allows an old Plan revision to masquerade as
		// current state after collapse.
		if strings.Contains(text, "<RUNTIME_DURABLE_PLAN>") ||
			strings.Contains(text, "<RUNTIME_EXECUTION_LEDGER>") ||
			strings.Contains(text, "<RUNTIME_TOOL_PROJECTION>") {
			continue
		}
		filtered = append(filtered, message)
	}
	return filtered
}

func buildDeterministicCollapseSummary(messages []model.Message, previous string, budget, generation int) string {
	const header = "<context_collapse generation=\"%d\">\nContinue the same task. Older execution observations are summarized below; preserve exact user constraints and use current tools to re-verify stale facts:\n"
	content := fmt.Sprintf(header, generation)
	if strings.TrimSpace(previous) != "" {
		previous = strings.TrimSpace(previous)
		if len([]rune(previous)) > 1200 {
			previous = string([]rune(previous)[:1200]) + "…"
		}
		content += "- prior collapse: " + previous + "\n"
	}
	for _, message := range messages {
		text := strings.TrimSpace(message.TextContent())
		if text == "" && len(message.ToolCalls) == 0 {
			continue
		}
		if len([]rune(text)) > 180 {
			text = string([]rune(text)[:180]) + "…"
		}
		if len(message.ToolCalls) > 0 {
			names := make([]string, 0, len(message.ToolCalls))
			for _, call := range message.ToolCalls {
				names = append(names, call.Name)
			}
			text = "requested tools: " + strings.Join(names, ", ") + "; " + text
		}
		line := fmt.Sprintf("- %s[%s]: %s\n", message.Role, messageID(message), text)
		candidate := model.TextMessage(model.RoleUser, content+line+"</context_collapse>")
		if EstimateTokens(candidate) > budget {
			break
		}
		content += line
	}
	if content == fmt.Sprintf(header, generation) {
		content += fmt.Sprintf("- older execution messages omitted: %d\n", len(messages))
	}
	return strings.TrimSpace(content) + "\n</context_collapse>"
}

func messageID(message model.Message) string {
	if strings.TrimSpace(message.ID) != "" {
		return message.ID
	}
	return "message-without-id"
}

func hashIDs(ids []string) string {
	digest := sha256.Sum256([]byte(strings.Join(ids, "\n")))
	return hex.EncodeToString(digest[:])
}

func mergeIDs(existing, added []string) []string {
	const maxRememberedIDs = 256
	seen := make(map[string]struct{}, len(existing)+len(added))
	all := make([]string, 0, len(existing)+len(added))
	for _, value := range append(append([]string(nil), existing...), added...) {
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		all = append(all, value)
	}
	if len(all) <= maxRememberedIDs {
		return all
	}
	return append([]string(nil), all[len(all)-maxRememberedIDs:]...)
}

func presentMessageIDs(messages []model.Message, candidates []string) []string {
	if len(candidates) == 0 {
		return nil
	}
	present := make(map[string]struct{}, len(messages))
	for _, message := range messages {
		if id := strings.TrimSpace(messageID(message)); id != "" {
			present[id] = struct{}{}
		}
	}
	result := make([]string, 0, len(candidates))
	for _, id := range candidates {
		if _, ok := present[id]; ok {
			result = append(result, id)
		}
	}
	return result
}
