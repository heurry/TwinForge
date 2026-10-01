// Package protocol defines the model-to-runtime decision boundary. Model
// messages are transport data; the runtime converts them into a small,
// validated DecisionEnvelope before scheduling any side effect.
package protocol

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
)

var ErrLegacyToolMarkup = errors.New("legacy tool markup received; expected structured tool_calls")

type DecisionKind string

const (
	KindFinal        DecisionKind = "final"
	KindActionIntent DecisionKind = "action_intent"
	KindAskUser      DecisionKind = "ask_user"
	KindPlanPatch    DecisionKind = "plan_patch"
	KindDelegate     DecisionKind = "delegate"
	KindYield        DecisionKind = "yield"
)

// DecisionEnvelope is the only model-owned input accepted by the scheduler.
// Tool calls remain typed model.ToolCall values; receipt/evidence fields are
// deliberately absent because the platform creates them after execution.
type DecisionEnvelope struct {
	Kind          DecisionKind  `json:"kind"`
	PlanRevision  int           `json:"plan_revision,omitempty"`
	PlanNodeID    string        `json:"plan_node_id,omitempty"`
	ToolCalls     []model.ToolCall `json:"tool_calls,omitempty"`
	Text          string        `json:"text,omitempty"`
	RawAssistant  model.Message `json:"-"`
}

func Decode(message model.Message) (DecisionEnvelope, error) {
	text := strings.TrimSpace(message.TextContent())
	if len(message.ToolCalls) == 0 {
		if looksLikeLegacyToolMarkup(text) {
			return DecisionEnvelope{}, ErrLegacyToolMarkup
		}
		return DecisionEnvelope{Kind: KindFinal, Text: message.TextContent(), RawAssistant: message}, nil
	}
	for _, call := range message.ToolCalls {
		if strings.TrimSpace(call.ID) == "" || strings.TrimSpace(call.Name) == "" {
			return DecisionEnvelope{}, fmt.Errorf("structured tool call requires id and name")
		}
		arguments := call.Arguments
		if len(arguments) == 0 {
			arguments = json.RawMessage(`{}`)
		}
		var object map[string]any
		if err := json.Unmarshal(arguments, &object); err != nil {
			return DecisionEnvelope{}, fmt.Errorf("tool %q arguments are not valid JSON: %w", call.Name, err)
		}
		if object == nil {
			return DecisionEnvelope{}, fmt.Errorf("tool %q arguments must be a JSON object", call.Name)
		}
	}
	return DecisionEnvelope{Kind: KindActionIntent, ToolCalls: append([]model.ToolCall(nil), message.ToolCalls...), Text: message.TextContent(), RawAssistant: message}, nil
}

func looksLikeLegacyToolMarkup(value string) bool {
	lower := strings.ToLower(value)
	return strings.Contains(lower, "<toolcall") || strings.Contains(lower, "<tool_call") || strings.Contains(lower, "<function=")
}
