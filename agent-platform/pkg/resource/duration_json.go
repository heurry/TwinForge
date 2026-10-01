package resource

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"time"
)

// MarshalJSON renders the public timeout as a readable duration string.
func (p HTTPProvider) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Endpoint          string            `json:"endpoint"`
		Method            string            `json:"method"`
		HeaderEnvironment map[string]string `json:"header_environment,omitempty"`
		Timeout           string            `json:"timeout,omitempty"`
		MaxResponseBytes  int64             `json:"max_response_bytes,omitempty"`
	}{
		Endpoint: p.Endpoint, Method: p.Method, HeaderEnvironment: p.HeaderEnvironment,
		Timeout: durationText(p.Timeout), MaxResponseBytes: p.MaxResponseBytes,
	})
}

// UnmarshalJSON accepts strings such as "30s" and legacy nanosecond integers.
func (p *HTTPProvider) UnmarshalJSON(data []byte) error {
	var wire struct {
		Endpoint          string            `json:"endpoint"`
		Method            string            `json:"method"`
		HeaderEnvironment map[string]string `json:"header_environment"`
		Timeout           json.RawMessage   `json:"timeout"`
		MaxResponseBytes  int64             `json:"max_response_bytes"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	timeout, err := decodeDuration(wire.Timeout)
	if err != nil {
		return fmt.Errorf("timeout: %w", err)
	}
	p.Endpoint, p.Method, p.HeaderEnvironment = wire.Endpoint, wire.Method, wire.HeaderEnvironment
	p.Timeout, p.MaxResponseBytes = timeout, wire.MaxResponseBytes
	return nil
}

func decodeDuration(raw json.RawMessage) (time.Duration, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return 0, nil
	}
	var text string
	if json.Unmarshal(trimmed, &text) == nil {
		duration, err := time.ParseDuration(text)
		if err != nil {
			return 0, fmt.Errorf("invalid duration %q", text)
		}
		return duration, nil
	}
	integer, err := strconv.ParseInt(string(trimmed), 10, 64)
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
