package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"time"
)

// MarshalJSON renders public runtime durations as readable Go duration strings.
func (p RuntimePolicy) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		RunTimeout    string `json:"run_timeout"`
		ModelTimeout  string `json:"model_timeout"`
		ToolTimeout   string `json:"tool_timeout"`
		MaxModelCalls int    `json:"max_model_calls"`
		MaxToolCalls  int    `json:"max_tool_calls"`
	}{
		RunTimeout: p.RunTimeout.String(), ModelTimeout: p.ModelTimeout.String(),
		ToolTimeout: p.ToolTimeout.String(), MaxModelCalls: p.MaxModelCalls, MaxToolCalls: p.MaxToolCalls,
	})
}

// UnmarshalJSON accepts readable strings and legacy nanosecond integers.
func (p *RuntimePolicy) UnmarshalJSON(data []byte) error {
	var wire struct {
		RunTimeout    json.RawMessage `json:"run_timeout"`
		ModelTimeout  json.RawMessage `json:"model_timeout"`
		ToolTimeout   json.RawMessage `json:"tool_timeout"`
		MaxModelCalls int             `json:"max_model_calls"`
		MaxToolCalls  int             `json:"max_tool_calls"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	var err error
	if p.RunTimeout, err = decodeDuration(wire.RunTimeout); err != nil {
		return fmt.Errorf("run_timeout: %w", err)
	}
	if p.ModelTimeout, err = decodeDuration(wire.ModelTimeout); err != nil {
		return fmt.Errorf("model_timeout: %w", err)
	}
	if p.ToolTimeout, err = decodeDuration(wire.ToolTimeout); err != nil {
		return fmt.Errorf("tool_timeout: %w", err)
	}
	p.MaxModelCalls, p.MaxToolCalls = wire.MaxModelCalls, wire.MaxToolCalls
	return nil
}

func (p MemoryPolicy) MarshalJSON() ([]byte, error) {
	type alias MemoryPolicy
	return json.Marshal(struct {
		alias
		DefaultTTL string `json:"default_ttl,omitempty"`
	}{alias: alias(p), DefaultTTL: durationText(p.DefaultTTL)})
}

func (p *MemoryPolicy) UnmarshalJSON(data []byte) error {
	type alias MemoryPolicy
	var wire struct {
		alias
		DefaultTTL json.RawMessage `json:"default_ttl"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*p = MemoryPolicy(wire.alias)
	duration, err := decodeDuration(wire.DefaultTTL)
	if err != nil {
		return fmt.Errorf("default_ttl: %w", err)
	}
	p.DefaultTTL = duration
	return nil
}

func (p ApprovalPolicy) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		RequireFor                []string `json:"require_for,omitempty"`
		AutoApproveSandboxCommand bool     `json:"auto_approve_sandbox_command,omitempty"`
		ExpiresIn                 string   `json:"expires_in,omitempty"`
	}{RequireFor: p.RequireFor, AutoApproveSandboxCommand: p.AutoApproveSandboxCommand, ExpiresIn: durationText(p.ExpiresIn)})
}

func (p *ApprovalPolicy) UnmarshalJSON(data []byte) error {
	var wire struct {
		RequireFor                []string        `json:"require_for"`
		AutoApproveSandboxCommand bool            `json:"auto_approve_sandbox_command"`
		ExpiresIn                 json.RawMessage `json:"expires_in"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	duration, err := decodeDuration(wire.ExpiresIn)
	if err != nil {
		return fmt.Errorf("expires_in: %w", err)
	}
	p.RequireFor, p.AutoApproveSandboxCommand, p.ExpiresIn = wire.RequireFor, wire.AutoApproveSandboxCommand, duration
	return nil
}

func decodeDuration(raw json.RawMessage) (time.Duration, error) {
	if len(bytes.TrimSpace(raw)) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return 0, nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		duration, err := time.ParseDuration(text)
		if err != nil {
			return 0, fmt.Errorf("invalid duration %q", text)
		}
		return duration, nil
	}
	integer, err := strconv.ParseInt(string(raw), 10, 64)
	if err != nil {
		return 0, fmt.Errorf("must be a duration string or nanosecond integer")
	}
	return time.Duration(integer), nil
}

func durationText(duration time.Duration) string {
	if duration == 0 {
		return ""
	}
	return duration.String()
}
