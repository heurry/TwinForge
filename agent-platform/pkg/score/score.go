// Package score defines durable evaluation annotations for Agent traces.
package score

import (
	"errors"
	"strings"
	"time"
)

type Type string

const (
	Numeric     Type = "numeric"
	Categorical Type = "categorical"
	Boolean     Type = "boolean"
	Text        Type = "text"
)

type Score struct {
	ID                string         `json:"id"`
	TenantID          string         `json:"tenant_id"`
	RunID             string         `json:"run_id,omitempty"`
	ObservationID     string         `json:"observation_id,omitempty"`
	SessionID         string         `json:"session_id,omitempty"`
	DatasetRunID      string         `json:"dataset_run_id,omitempty"`
	Name              string         `json:"name"`
	ScoreType         Type           `json:"score_type"`
	Value             *float64       `json:"value,omitempty"`
	StringValue       *string        `json:"string_value,omitempty"`
	Source            string         `json:"source"`
	EvaluatorVersion  string         `json:"evaluator_version,omitempty"`
	AgentVersionID    string         `json:"agent_version_id,omitempty"`
	ModelResolutionID string         `json:"model_resolution_id,omitempty"`
	PromptVersionID   string         `json:"prompt_version_id,omitempty"`
	ToolSetVersionID  string         `json:"toolset_version_id,omitempty"`
	SkillSetVersionID string         `json:"skillset_version_id,omitempty"`
	Metadata          map[string]any `json:"metadata,omitempty"`
	CreatedBy         string         `json:"created_by,omitempty"`
	CreatedAt         time.Time      `json:"created_at"`
}

type Create struct {
	TenantID          string
	RunID             string
	ObservationID     string
	SessionID         string
	DatasetRunID      string
	Name              string
	ScoreType         Type
	Value             *float64
	StringValue       *string
	Source            string
	EvaluatorVersion  string
	AgentVersionID    string
	ModelResolutionID string
	PromptVersionID   string
	ToolSetVersionID  string
	SkillSetVersionID string
	Metadata          map[string]any
	CreatedBy         string
}

func (c Create) Validate() error {
	if strings.TrimSpace(c.TenantID) == "" || strings.TrimSpace(c.Name) == "" {
		return errors.New("tenant_id and name are required")
	}
	if strings.TrimSpace(c.RunID) == "" && strings.TrimSpace(c.SessionID) == "" && strings.TrimSpace(c.ObservationID) == "" && strings.TrimSpace(c.DatasetRunID) == "" {
		return errors.New("one score target is required")
	}
	switch c.ScoreType {
	case Numeric:
		if c.Value == nil {
			return errors.New("numeric score requires value")
		}
	case Boolean, Categorical, Text:
		if c.StringValue == nil && c.Value == nil {
			return errors.New("score value is required")
		}
	default:
		return errors.New("unsupported score_type")
	}
	if strings.TrimSpace(c.Source) == "" {
		return errors.New("source is required")
	}
	return nil
}
