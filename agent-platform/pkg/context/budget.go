package context

import (
	stdcontext "context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"unicode/utf8"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
)

// ModelMessageBudget returns the maximum message budget after accounting for
// output generation, model-visible tool schemas, and tokenizer/chat-template
// variance. OpenAI-compatible providers count tool definitions as prompt
// tokens, so budgeting messages alone is unsafe for tool-heavy Agents.
func ModelMessageBudget(contextWindow, outputTokens int, tools []model.ToolSchema) (int, int, error) {
	if contextWindow <= 0 || outputTokens < 0 || outputTokens >= contextWindow {
		return 0, 0, errors.New("invalid model context window")
	}
	toolTokens := EstimateToolSchemas(tools)
	// Reserve a provider/template envelope in addition to message and tool
	// estimates. OpenAI-compatible servers count role wrappers, system
	// prefixes, JSON tool envelopes, and tokenizer variance that are not
	// represented by our portable estimator. A small 192-token floor is not
	// sufficient for long Agent prompts; keep at least 4096 tokens available
	// for this envelope on large-context models.
	safetyTokens := contextWindow / 10
	if contextWindow >= 8192 && safetyTokens < 4096 {
		safetyTokens = 4096
	}
	if safetyTokens < 192 {
		safetyTokens = 192
	}
	budget := contextWindow - outputTokens - toolTokens - safetyTokens
	if budget <= 0 {
		return 0, toolTokens, ErrBudgetTooSmall
	}
	return budget, toolTokens, nil
}

// EstimateToolSchemas conservatively estimates the prompt cost of the
// provider-neutral function definitions. The extra per-tool allowance covers
// the OpenAI wire wrapper and chat-template delimiters.
func EstimateToolSchemas(tools []model.ToolSchema) int {
	total := 0
	for _, schema := range tools {
		encoded, _ := json.Marshal(schema)
		total += estimateTextTokens(string(encoded)) + 24
	}
	return total
}

// ErrBudgetTooSmall reports that mandatory system/current messages do not fit.
var ErrBudgetTooSmall = errors.New("context budget is too small for mandatory messages")

// BudgetConfig controls deterministic recent-history selection.
type BudgetConfig struct {
	MaxInputTokens      int
	ReserveOutputTokens int
}

// SectionAllocation is the effective per-request allocation after the model
// message budget is known. Requested section caps are proportionally reduced
// when their sum cannot fit, so no section can silently overrun the 24K
// envelope.
type SectionAllocation struct {
	RecentTurnTokens int `json:"recent_turn_tokens"`
	MemoryTokens     int `json:"memory_tokens"`
	KnowledgeTokens  int `json:"knowledge_tokens"`
	ToolResultTokens int `json:"tool_result_tokens"`
	SummaryTokens    int `json:"summary_tokens"`
	StaticInstructionTokens int `json:"static_instruction_tokens"`
	ReservedTokens   int `json:"reserved_section_tokens"`
}

func AllocateSectionBudgets(messageBudget, recent, memory, knowledge, toolResults int) SectionAllocation {
	return AllocateSectionBudgetsWithStatic(messageBudget, recent, memory, knowledge, toolResults, 0, 0)
}

// AllocateSectionBudgetsWithStatic applies the complete 24K allocation. The
// summary and configured-instruction reserves are removed before mutable
// sections are assigned, so those sections cannot collectively overrun the
// model message budget.
func AllocateSectionBudgetsWithStatic(messageBudget, recent, memory, knowledge, toolResults, summary, staticInstruction int) SectionAllocation {
	if messageBudget <= 0 {
		return SectionAllocation{}
	}
	if summary < 0 {
		summary = 0
	}
	if staticInstruction < 0 {
		staticInstruction = 0
	}
	reserved := summary + staticInstruction
	if reserved > messageBudget {
		reserved = messageBudget
		if summary > reserved {
			summary = reserved
			staticInstruction = 0
		} else {
			staticInstruction = reserved - summary
		}
	}
	values := []int{recent, memory, knowledge, toolResults}
	for index := range values {
		if values[index] < 0 {
			values[index] = 0
		}
	}
	total := values[0] + values[1] + values[2] + values[3]
	available := messageBudget - reserved
	if total > available {
		remaining := available
		for index := range values {
			allocated := values[index]
			if allocated > remaining {
				allocated = remaining
			}
			values[index] = allocated
			remaining -= allocated
		}
	}
	allocated := values[0] + values[1] + values[2] + values[3]
	return SectionAllocation{RecentTurnTokens: values[0], MemoryTokens: values[1], KnowledgeTokens: values[2], ToolResultTokens: values[3], SummaryTokens: summary, StaticInstructionTokens: staticInstruction, ReservedTokens: allocated + reserved}
}

// BudgetBuilder keeps the system and current message, then selects the newest
// history that fits. It uses a conservative tokenizer-independent estimate.
type BudgetBuilder struct {
	config BudgetConfig
}

// NewBudgetBuilder validates one immutable context budget.
func NewBudgetBuilder(config BudgetConfig) (*BudgetBuilder, error) {
	if config.MaxInputTokens <= 0 || config.ReserveOutputTokens < 0 ||
		config.ReserveOutputTokens >= config.MaxInputTokens {
		return nil, errors.New("invalid context token budget")
	}
	return &BudgetBuilder{config: config}, nil
}

// Build deterministically compacts old history without mutating source input.
func (b *BudgetBuilder) Build(_ stdcontext.Context, request Request) (Result, error) {
	if len(request.Messages) == 0 {
		return Result{}, errors.New("context messages are required")
	}
	budget := b.config.MaxInputTokens - b.config.ReserveOutputTokens
	firstHistory := request.MandatoryPrefix
	if firstHistory < 0 || firstHistory >= len(request.Messages) {
		return Result{}, errors.New("mandatory prefix must leave the current message selectable")
	}
	selected := make([]model.Message, 0, len(request.Messages))
	used := 0
	if firstHistory == 0 && request.Messages[0].Role == model.RoleSystem {
		firstHistory = 1
	}
	for index := 0; index < firstHistory; index++ {
		selected = append(selected, request.Messages[index])
		used += EstimateTokens(request.Messages[index])
	}
	currentIndex := len(request.Messages) - 1
	if currentIndex >= firstHistory {
		used += EstimateTokens(request.Messages[currentIndex])
	}
	if used > budget {
		return Result{}, ErrBudgetTooSmall
	}
	historyStart := currentIndex
	for end := currentIndex; end > firstHistory; {
		groupStart := end - 1
		// Session history is stored as User/Assistant pairs. Keep the pair
		// together so compaction never produces an orphan assistant message.
		if request.Messages[groupStart].Role == model.RoleAssistant &&
			groupStart-1 >= firstHistory && request.Messages[groupStart-1].Role == model.RoleUser {
			groupStart--
		}
		cost := 0
		for index := groupStart; index < end; index++ {
			cost += EstimateTokens(request.Messages[index])
		}
		if used+cost > budget {
			break
		}
		used += cost
		historyStart = groupStart
		end = groupStart
	}
	selected = append(selected, request.Messages[historyStart:currentIndex]...)
	if currentIndex >= firstHistory {
		selected = append(selected, request.Messages[currentIndex])
	}
	encoded, err := json.Marshal(selected)
	if err != nil {
		return Result{}, err
	}
	digest := sha256.Sum256(encoded)
	return Result{
		Messages: selected,
		Manifest: Manifest{InputTokens: used, MessageCount: len(selected), MandatoryMessageCount: firstHistory, ContextHash: hex.EncodeToString(digest[:])},
	}, nil
}

// EstimateTokens gives a conservative portable estimate without binding the
// framework to one provider tokenizer.
func EstimateTokens(message model.Message) int {
	content := message.TextContent()
	tokens := estimateTextTokens(content)
	structural := 4
	for _, call := range message.ToolCalls {
		// Tool Call names and JSON arguments are serialized outside message
		// content by OpenAI-compatible providers, but still consume prompt
		// tokens on every following step.
		structural += 12 + estimateTextTokens(call.Name) + estimateTextTokens(string(call.Arguments))
	}
	if message.ToolCallID != "" {
		structural += 4
	}
	if tokens == 0 && utf8.RuneCountInString(content) == 0 {
		tokens = 1
	}
	return tokens + structural
}

func estimateTextTokens(value string) int {
	asciiBytes, nonASCII := 0, 0
	for _, r := range value {
		if r <= 127 {
			asciiBytes++
		} else {
			nonASCII++
		}
	}
	// JSON, file paths, code, and tool arguments tokenize more densely than
	// natural-language prose. Three ASCII bytes per token is deliberately more
	// conservative than the common four-character heuristic and avoids relying
	// on one provider's tokenizer.
	return (asciiBytes+2)/3 + nonASCII
}
