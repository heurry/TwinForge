package agent

import (
	"crypto/sha256"
	"fmt"
	"strings"
)

// Identity is the governed role layer compiled ahead of the user Prompt.
type Identity struct {
	DisplayName        string   `json:"display_name,omitempty"`
	Role               string   `json:"role,omitempty"`
	Goal               string   `json:"goal,omitempty"`
	Responsibilities   []string `json:"responsibilities,omitempty"`
	Boundaries         []string `json:"boundaries,omitempty"`
	CommunicationStyle string   `json:"communication_style,omitempty"`
}

func (i Identity) empty() bool {
	return strings.TrimSpace(i.DisplayName) == "" && strings.TrimSpace(i.Role) == "" && strings.TrimSpace(i.Goal) == "" && len(i.Responsibilities) == 0 && len(i.Boundaries) == 0 && strings.TrimSpace(i.CommunicationStyle) == ""
}

func (i Identity) validate() error {
	if i.empty() {
		return nil
	}
	if strings.TrimSpace(i.Role) == "" || strings.TrimSpace(i.Goal) == "" {
		return fmt.Errorf("identity.role and identity.goal are required when identity is configured")
	}
	if len(i.Responsibilities) > 20 || len(i.Boundaries) > 20 {
		return fmt.Errorf("identity responsibilities and boundaries are limited to 20 items")
	}
	for _, value := range append(append([]string(nil), i.Responsibilities...), i.Boundaries...) {
		if strings.TrimSpace(value) == "" || len(value) > 500 {
			return fmt.Errorf("identity list items must be non-empty and no longer than 500 bytes")
		}
	}
	return nil
}

// CompileIdentity creates deterministic model-visible text. Configuration is
// structured in storage; markdown is only the provider-neutral projection.
func CompileIdentity(i Identity) string {
	if i.empty() {
		return ""
	}
	var sections []string
	sections = append(sections, "# Agent Identity")
	if value := strings.TrimSpace(i.DisplayName); value != "" {
		sections = append(sections, "Name: "+value)
	}
	sections = append(sections, "Role: "+strings.TrimSpace(i.Role), "Goal: "+strings.TrimSpace(i.Goal))
	if values := cleanIdentityValues(i.Responsibilities); len(values) > 0 {
		sections = append(sections, "Responsibilities:\n- "+strings.Join(values, "\n- "))
	}
	if values := cleanIdentityValues(i.Boundaries); len(values) > 0 {
		sections = append(sections, "Boundaries:\n- "+strings.Join(values, "\n- "))
	}
	if value := strings.TrimSpace(i.CommunicationStyle); value != "" {
		sections = append(sections, "Communication style: "+value)
	}
	return strings.Join(sections, "\n\n")
}

// IdentityDigest identifies the exact compiled identity used by a Run.
func IdentityDigest(i Identity) string {
	if compiled := CompileIdentity(i); compiled != "" {
		sum := sha256.Sum256([]byte(compiled))
		return fmt.Sprintf("sha256:%x", sum[:])
	}
	return ""
}

func cleanIdentityValues(values []string) []string {
	result := make([]string, 0, len(values))
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			result = append(result, value)
		}
	}
	return result
}
