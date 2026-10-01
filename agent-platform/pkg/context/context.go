// Package context defines model-context construction contracts.
package context

import (
	stdcontext "context"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
)

const (
	ContextSectionMetadataKey = "context_section"
	ContextSectionMemory      = "memory"
	ContextSectionKnowledge   = "knowledge"
	ContextSectionRecent      = "recent"
	ContextSectionSummary     = "summary"
	ContextSectionRuntime     = "runtime"
	// MemoryIDsMetadataKey makes a rendered memory block deduplicable without
	// parsing untrusted XML-like body text.
	MemoryIDsMetadataKey = "memory_ids"
	// RuntimeControlMetadataKey marks framework-generated user-role guidance.
	// It is useful current context, but it is not immutable human input.
	RuntimeControlMetadataKey = "runtime.control_message"
	// ToolHistoryMetadataKey marks a bounded, non-executable observation of a
	// completed Tool Call. It is deliberately not represented as
	// assistant.tool_calls: projected arguments are historical evidence, not a
	// valid invocation for the current Tool Schema.
	ToolHistoryMetadataKey = "runtime.tool_history"
	// ToolHistoryFailureMetadataKey lets Collapse protect a recent failed Tool
	// observation without requiring a synthetic RoleTool message.
	ToolHistoryFailureMetadataKey = "runtime.tool_history_failed"
)

// Manifest records the exact versioned inputs selected for one model request.
type Manifest struct {
	AgentVersion          string   `json:"agent_version"`
	IdentityHash          string   `json:"identity_hash,omitempty"`
	PromptVersion         string   `json:"prompt_version"`
	ToolSetVersion        string   `json:"toolset_version"`
	SkillSetVersion       string   `json:"skillset_version,omitempty"`
	SkillVersionIDs       []string `json:"skill_version_ids,omitempty"`
	SessionEventFrom      int64    `json:"session_event_from,omitempty"`
	SessionEventTo        int64    `json:"session_event_to,omitempty"`
	MemoryIDs             []string `json:"memory_ids,omitempty"`
	StaticMemorySourceIDs []string `json:"static_memory_source_ids,omitempty"`
	KnowledgeChunkIDs     []string `json:"knowledge_chunk_ids,omitempty"`
	ArtifactRefs          []string `json:"artifact_refs,omitempty"`
	InputTokens           int      `json:"input_tokens"`
	MessageCount          int      `json:"message_count"`
	MandatoryMessageCount int      `json:"mandatory_message_count"`
	CompactionGeneration  int      `json:"compaction_generation"`
	ContextHash           string   `json:"context_hash"`
}

// Request identifies the sources available to a Context Builder.
type Request struct {
	RunID           string
	Messages        []model.Message
	MandatoryPrefix int
}

// Result is a model-visible context and its reproducibility manifest.
type Result struct {
	Messages []model.Message
	Manifest Manifest
}

// Builder selects, budgets, and records model-visible context.
type Builder interface {
	Build(ctx stdcontext.Context, request Request) (Result, error)
}
