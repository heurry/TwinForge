package delegation

import (
	"encoding/json"
	"errors"
	"fmt"
)

var ErrPending = errors.New("agent delegation is pending")

// Target describes one immutable AgentVersion that the published parent
// version is allowed to call. It is intentionally model-safe metadata: it
// contains no prompt, credentials, or implementation details.
type Target struct {
	AgentID        string   `json:"agent_id"`
	AgentKey       string   `json:"agent_key,omitempty"`
	AgentName      string   `json:"agent_name,omitempty"`
	AgentVersionID string   `json:"agent_version_id"`
	Version        int      `json:"version,omitempty"`
	Modes          []string `json:"modes"`
	Available      bool     `json:"available"`
}

// ErrTargetNotAllowed is returned before a child Run is created when the
// requested target or mode is not part of the immutable parent allowlist.
var ErrTargetNotAllowed = errors.New("delegation target or mode is not allowed by the published AgentVersion")

// TargetNotAllowedError keeps the policy decision structured so the model and
// UI can repair a wrong alias/mode without guessing or repeating an unchanged
// call. The target list is derived from the parent AgentVersion allowlist.
type TargetNotAllowedError struct {
	RequestedTarget string   `json:"requested_target"`
	RequestedMode   string   `json:"requested_mode"`
	AllowedTargets  []Target `json:"allowed_targets"`
	Reason          string   `json:"reason,omitempty"`
}

func (e *TargetNotAllowedError) Error() string {
	if e == nil {
		return ErrTargetNotAllowed.Error()
	}
	if e.Reason != "" {
		return fmt.Sprintf("%s: %s", ErrTargetNotAllowed, e.Reason)
	}
	return ErrTargetNotAllowed.Error()
}

func (e *TargetNotAllowedError) Unwrap() error { return ErrTargetNotAllowed }

func AsTargetNotAllowedError(err error) (*TargetNotAllowedError, bool) {
	var target *TargetNotAllowedError
	if errors.As(err, &target) {
		return target, true
	}
	return nil, false
}

type Request struct {
	TargetAgentVersionID string `json:"target_agent_version_id"`
	// TargetAgent is an optional human-readable alias. The platform resolves it
	// only against the parent allowlist and always persists the canonical
	// AgentVersion ID. Keeping this field optional preserves the strict UUID
	// contract for existing callers.
	TargetAgent string          `json:"target_agent,omitempty"`
	Mode        string          `json:"mode"`
	Input       json.RawMessage `json:"input"`
}
type Outcome struct {
	DelegationID string          `json:"delegation_id"`
	ChildRunID   string          `json:"child_run_id"`
	Status       string          `json:"status"`
	Output       json.RawMessage `json:"output,omitempty"`
	Error        string          `json:"error,omitempty"`
}
