// Package skill defines immutable, reusable Agent capability bundles.
package skill

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
)

type InstructionBlock struct {
	Name     string `json:"name"`
	Content  string `json:"content"`
	Priority int    `json:"priority,omitempty"`
}

type Example struct {
	Input  string `json:"input"`
	Output string `json:"output"`
}

// Spec is one immutable Skill revision. RequiredTools are merged into the
// Run-local registry when the Skill is activated.
type Spec struct {
	Description   string             `json:"description,omitempty"`
	Instructions  []InstructionBlock `json:"instructions"`
	Examples      []Example          `json:"examples,omitempty"`
	RequiredTools []agent.VersionRef `json:"required_tools,omitempty"`
}

func (s Spec) Validate() error {
	if len(s.Instructions) == 0 {
		return errors.New("skill instructions are required")
	}
	for i, block := range s.Instructions {
		if strings.TrimSpace(block.Name) == "" || strings.TrimSpace(block.Content) == "" {
			return fmt.Errorf("instructions[%d] name and content are required", i)
		}
	}
	for i, ref := range s.RequiredTools {
		if ref.ID == "" || ref.Version == "" {
			return fmt.Errorf("required_tools[%d] id and version are required", i)
		}
	}
	return nil
}

type SetSpec struct {
	Skills []agent.VersionRef `json:"skills"`
}

func (s SetSpec) Validate() error {
	seen := map[string]struct{}{}
	for i, ref := range s.Skills {
		if ref.ID == "" || ref.Version == "" {
			return fmt.Errorf("skills[%d] id and version are required", i)
		}
		if _, ok := seen[ref.ID]; ok {
			return fmt.Errorf("skill %q is duplicated", ref.ID)
		}
		seen[ref.ID] = struct{}{}
	}
	return nil
}

type Version struct {
	ID           string          `json:"id"`
	DefinitionID string          `json:"definition_id"`
	TenantID     string          `json:"tenant_id"`
	Key          string          `json:"key"`
	Name         string          `json:"name"`
	Version      int             `json:"version"`
	Spec         json.RawMessage `json:"spec"`
	SpecHash     string          `json:"spec_hash"`
	Status       string          `json:"status"`
	CreatedBy    *string         `json:"created_by,omitempty"`
	CreatedAt    time.Time       `json:"created_at"`
}

type CreateVersion struct {
	TenantID, Key, Name string
	Spec                Spec
	CreatedBy           *string
}
type CreateSetVersion struct {
	TenantID, Key, Name string
	Spec                SetSpec
	CreatedBy           *string
}

// Selection is the human-readable execution snapshot exposed by observability.
// Immutable version IDs remain in VersionIDs for replay and persistence, while
// operators can inspect the actual Skill name and injected instructions.
type Selection struct {
	Name         string   `json:"name"`
	Key          string   `json:"key"`
	Version      int      `json:"version"`
	Instructions []string `json:"instructions,omitempty"`
}

// Resolved is the deterministic model/tool contribution of one SkillSet.
type Resolved struct {
	InstructionMessages []model.Message
	ExampleMessages     []model.Message
	RequiredTools       []agent.VersionRef
	VersionIDs          []string
	Skills              []Selection
}
