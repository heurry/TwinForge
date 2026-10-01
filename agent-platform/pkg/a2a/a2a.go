// Package a2a contains the A2A 1.0 REST binding data model used by the gateway.
package a2a

import (
	"encoding/json"
	"time"
)

const ProtocolVersion = "1.0"

type Part struct {
	Text     string          `json:"text,omitempty"`
	Data     json.RawMessage `json:"data,omitempty"`
	Metadata map[string]any  `json:"metadata,omitempty"`
}
type Message struct {
	MessageID string         `json:"messageId"`
	ContextID string         `json:"contextId,omitempty"`
	TaskID    string         `json:"taskId,omitempty"`
	Role      string         `json:"role"`
	Parts     []Part         `json:"parts"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}
type SendMessageRequest struct {
	Message       Message        `json:"message"`
	Configuration map[string]any `json:"configuration,omitempty"`
	Metadata      map[string]any `json:"metadata,omitempty"`
}
type TaskStatus struct {
	State     string    `json:"state"`
	Message   *Message  `json:"message,omitempty"`
	Timestamp time.Time `json:"timestamp"`
}
type Artifact struct {
	ArtifactID  string         `json:"artifactId"`
	Name        string         `json:"name,omitempty"`
	Description string         `json:"description,omitempty"`
	Parts       []Part         `json:"parts"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}
type Task struct {
	ID        string         `json:"id"`
	ContextID string         `json:"contextId"`
	Status    TaskStatus     `json:"status"`
	History   []Message      `json:"history,omitempty"`
	Artifacts []Artifact     `json:"artifacts,omitempty"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}
type AgentInterface struct {
	URL             string `json:"url"`
	ProtocolBinding string `json:"protocolBinding"`
	ProtocolVersion string `json:"protocolVersion"`
}
type AgentCapabilities struct {
	Streaming         bool `json:"streaming"`
	PushNotifications bool `json:"pushNotifications"`
	ExtendedAgentCard bool `json:"extendedAgentCard"`
}
type AgentSkill struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Tags        []string `json:"tags"`
	Examples    []string `json:"examples,omitempty"`
	InputModes  []string `json:"inputModes,omitempty"`
	OutputModes []string `json:"outputModes,omitempty"`
}
type AgentProvider struct {
	Organization string `json:"organization"`
	URL          string `json:"url"`
}
type AgentCard struct {
	Name                 string                `json:"name"`
	Description          string                `json:"description"`
	SupportedInterfaces  []AgentInterface      `json:"supportedInterfaces"`
	Provider             AgentProvider         `json:"provider"`
	Version              string                `json:"version"`
	Capabilities         AgentCapabilities     `json:"capabilities"`
	DefaultInputModes    []string              `json:"defaultInputModes"`
	DefaultOutputModes   []string              `json:"defaultOutputModes"`
	Skills               []AgentSkill          `json:"skills"`
	SecuritySchemes      map[string]any        `json:"securitySchemes,omitempty"`
	SecurityRequirements []map[string][]string `json:"securityRequirements,omitempty"`
}
type StoredTask struct {
	Task            Task   `json:"task"`
	TenantID        string `json:"-"`
	AgentID         string `json:"-"`
	RunID           string `json:"-"`
	ClientMessageID string `json:"-"`
}
