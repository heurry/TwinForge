// Package openai implements the provider-neutral model.Provider contract for
// OpenAI-compatible chat completion endpoints, including vLLM and AIBrix.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
)

const maxResponseBytes = 8 << 20

// Config binds one immutable provider endpoint for a Run.
type Config struct {
	Endpoint        string
	APIKey          string
	Model           string
	DisableThinking bool
	Timeout         time.Duration
	HTTPClient      *http.Client
}

// Provider sends non-streaming OpenAI-compatible requests.
type Provider struct {
	endpoint        string
	apiKey          string
	model           string
	disableThinking bool
	client          *http.Client
}

// New validates and creates a provider. Endpoint may be a server root, a /v1
// base URL, or the complete /v1/chat/completions URL.
func New(config Config) (*Provider, error) {
	endpoint, err := completionURL(config.Endpoint)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(config.Model) == "" {
		return nil, errors.New("openai-compatible model is required")
	}
	client := config.HTTPClient
	if client == nil {
		timeout := config.Timeout
		if timeout <= 0 {
			timeout = 60 * time.Second
		}
		client = &http.Client{Timeout: timeout}
	}
	return &Provider{
		endpoint: endpoint, apiKey: config.APIKey, model: config.Model,
		disableThinking: config.DisableThinking, client: client,
	}, nil
}

// Complete performs one model call without hidden retries or model fallback.
func (p *Provider) Complete(ctx context.Context, request model.Request) (model.Response, error) {
	payload := chatRequest{
		Model: p.model, Messages: encodeMessages(request.Messages),
		MaxTokens: request.MaxTokens, Temperature: request.Temperature,
	}
	// Do not let an instruct checkpoint infer a textual tool protocol when
	// this request intentionally offers no executable tools.
	toolChoice := strings.TrimSpace(request.ToolChoice)
	if toolChoice == "" {
		if len(request.Tools) == 0 {
			toolChoice = "none"
		} else {
			toolChoice = "auto"
		}
	}
	if toolChoice != "auto" && toolChoice != "none" && toolChoice != "required" {
		return model.Response{}, fmt.Errorf("invalid tool_choice %q", toolChoice)
	}
	payload.ToolChoice = toolChoice
	if p.disableThinking {
		payload.ChatTemplateKwargs = &chatTemplateKwargs{EnableThinking: false}
	}
	for _, schema := range request.Tools {
		payload.Tools = append(payload.Tools, chatTool{
			Type:     "function",
			Function: chatFunction{Name: schema.Name, Description: schema.Description, Parameters: schema.Parameters},
		})
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return model.Response{}, fmt.Errorf("encode model request: %w", err)
	}
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint, bytes.NewReader(body))
	if err != nil {
		return model.Response{}, fmt.Errorf("create model request: %w", err)
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		httpRequest.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
	response, err := p.client.Do(httpRequest)
	if err != nil {
		return model.Response{}, fmt.Errorf("call model endpoint: %w", err)
	}
	defer response.Body.Close()
	limited := io.LimitReader(response.Body, maxResponseBytes+1)
	responseBody, err := io.ReadAll(limited)
	if err != nil {
		return model.Response{}, fmt.Errorf("read model response: %w", err)
	}
	if len(responseBody) > maxResponseBytes {
		return model.Response{}, errors.New("model response exceeds 8 MiB")
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		message := strings.TrimSpace(string(responseBody))
		if len(message) > 4096 {
			message = message[:4096]
		}
		return model.Response{}, fmt.Errorf("model endpoint returned HTTP %d: %s", response.StatusCode, message)
	}
	var decoded chatResponse
	if err := json.Unmarshal(responseBody, &decoded); err != nil {
		return model.Response{}, fmt.Errorf("decode model response: %w", err)
	}
	if len(decoded.Choices) == 0 {
		return model.Response{}, errors.New("model response has no choices")
	}
	choice := decoded.Choices[0]
	message, err := decodeMessage(choice.Message)
	if err != nil {
		return model.Response{}, err
	}
	return model.Response{
		Message: message, FinishReason: choice.FinishReason,
		Provider: "openai-compatible", ModelID: decoded.Model,
		Usage: model.Usage{
			InputTokens: decoded.Usage.PromptTokens, OutputTokens: decoded.Usage.CompletionTokens,
			TotalTokens: decoded.Usage.TotalTokens,
		},
	}, nil
}

func completionURL(endpoint string) (string, error) {
	value := strings.TrimRight(strings.TrimSpace(endpoint), "/")
	parsed, err := url.Parse(value)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return "", errors.New("openai-compatible endpoint must be an absolute HTTP URL")
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", errors.New("openai-compatible endpoint must use http or https")
	}
	if strings.HasSuffix(parsed.Path, "/chat/completions") {
		return parsed.String(), nil
	}
	if strings.HasSuffix(parsed.Path, "/v1") {
		parsed.Path += "/chat/completions"
	} else {
		parsed.Path = strings.TrimRight(parsed.Path, "/") + "/v1/chat/completions"
	}
	return parsed.String(), nil
}

type chatRequest struct {
	Model              string              `json:"model"`
	Messages           []chatMessage       `json:"messages"`
	Tools              []chatTool          `json:"tools,omitempty"`
	ToolChoice         string              `json:"tool_choice,omitempty"`
	MaxTokens          int                 `json:"max_tokens,omitempty"`
	Temperature        float64             `json:"temperature,omitempty"`
	ChatTemplateKwargs *chatTemplateKwargs `json:"chat_template_kwargs,omitempty"`
}

type chatTemplateKwargs struct {
	EnableThinking bool `json:"enable_thinking"`
}

type chatMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content,omitempty"`
	Name       string         `json:"name,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	ToolCalls  []chatToolCall `json:"tool_calls,omitempty"`
}

type chatTool struct {
	Type     string       `json:"type"`
	Function chatFunction `json:"function"`
}

type chatFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type chatToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function callFunctionBody `json:"function"`
}

type callFunctionBody struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type chatResponse struct {
	Model   string `json:"model"`
	Choices []struct {
		Message      chatMessage `json:"message"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

func encodeMessages(messages []model.Message) []chatMessage {
	encoded := make([]chatMessage, 0, len(messages))
	for _, message := range messages {
		current := chatMessage{
			Role: string(message.Role), Content: message.TextContent(),
			Name: message.Name, ToolCallID: message.ToolCallID,
		}
		for _, call := range message.ToolCalls {
			arguments := call.Arguments
			if len(arguments) == 0 {
				arguments = json.RawMessage(`{}`)
			}
			current.ToolCalls = append(current.ToolCalls, chatToolCall{
				ID: call.ID, Type: "function",
				Function: callFunctionBody{Name: call.Name, Arguments: string(arguments)},
			})
		}
		encoded = append(encoded, current)
	}
	return encoded
}

func decodeMessage(message chatMessage) (model.Message, error) {
	// Some instruct checkpoints emit a legacy XML-ish tool protocol in the
	// content field instead of the OpenAI tool_calls array. Never surface or
	// execute that text as an assistant answer; the runtime can safely retry
	// with a schema correction.
	if strings.Contains(strings.ToLower(message.Content), "<toolcall") ||
		strings.Contains(strings.ToLower(message.Content), "<function=") ||
		strings.Contains(strings.ToLower(message.Content), "<parameter=") {
		adapted, err := adaptLegacyToolMarkup(message.Content)
		if err != nil {
			return model.Message{}, fmt.Errorf("model_protocol_error: legacy tool markup received; expected structured tool_calls: %w", err)
		}
		return adapted, nil
	}
	// A model must not be able to manufacture a successful Tool receipt in
	// ordinary assistant text. Receipts are platform-authored from TOOL_CALLED
	// and TOOL_COMPLETED events; accepting this envelope would mark work done
	// without executing anything.
	var claimed struct {
		Status  string          `json:"status"`
		Receipt json.RawMessage `json:"receipt"`
	}
	if json.Unmarshal([]byte(strings.TrimSpace(message.Content)), &claimed) == nil &&
		strings.EqualFold(claimed.Status, "completed") && len(claimed.Receipt) != 0 && string(claimed.Receipt) != "null" {
		return model.Message{}, fmt.Errorf("model_protocol_error: fabricated tool receipt received; tool results must come from platform execution")
	}
	decoded := model.Message{
		Role: model.Role(message.Role), Content: message.Content,
		Name: message.Name, ToolCallID: message.ToolCallID,
	}
	if decoded.Role == "" {
		decoded.Role = model.RoleAssistant
	}
	if message.Content != "" {
		decoded.Parts = []model.ContentPart{{Type: model.ContentText, Text: message.Content}}
	}
	seenIDs := make(map[string]struct{}, len(message.ToolCalls))
	for _, call := range message.ToolCalls {
		if strings.TrimSpace(call.ID) == "" || strings.TrimSpace(call.Function.Name) == "" {
			return model.Message{}, errors.New("model_protocol_error: structured tool call requires non-empty id and function name")
		}
		if _, exists := seenIDs[call.ID]; exists {
			return model.Message{}, fmt.Errorf("model_protocol_error: duplicate tool call id %q", call.ID)
		}
		seenIDs[call.ID] = struct{}{}
		arguments := json.RawMessage(call.Function.Arguments)
		if len(arguments) == 0 {
			arguments = json.RawMessage(`{}`)
		}
		if !json.Valid(arguments) {
			return model.Message{}, fmt.Errorf("model_protocol_error: model tool call %q returned invalid JSON arguments", call.ID)
		}
		var object map[string]any
		if err := json.Unmarshal(arguments, &object); err != nil || object == nil {
			return model.Message{}, fmt.Errorf("model_protocol_error: model tool call %q arguments must be a JSON object", call.ID)
		}
		decoded.ToolCalls = append(decoded.ToolCalls, model.ToolCall{
			ID: call.ID, Name: call.Function.Name, Arguments: arguments,
		})
	}
	return decoded, nil
}

var (
	legacyFunctionPattern  = regexp.MustCompile(`(?is)<function\s*=\s*([A-Za-z0-9_.:-]+)\s*>(.*?)</function\s*>`)
	legacyParameterPattern = regexp.MustCompile(`(?is)<parameter\s*=\s*([A-Za-z0-9_.:-]+)\s*>(.*?)</parameter\s*>`)
)

// adaptLegacyToolMarkup is a narrow compatibility bridge for instruct models
// that ignore the provider's structured tool_calls field. It only accepts a
// closed function tag and closed parameter tags, converts values to a JSON
// object, and leaves execution/authorization to the normal Tool path. Any
// ambiguous or partial markup is rejected and retried as a protocol error.
func adaptLegacyToolMarkup(content string) (model.Message, error) {
	content = strings.TrimSpace(content)
	matches := legacyFunctionPattern.FindAllStringSubmatch(content, -1)
	if len(matches) == 0 {
		return model.Message{}, errors.New("legacy function block is incomplete or unsupported")
	}
	if len(matches) > 8 {
		return model.Message{}, errors.New("legacy response contains too many function blocks")
	}
	message := model.Message{Role: model.RoleAssistant, Metadata: map[string]string{"protocol_adapted": "legacy_tool_markup"}}
	for index, match := range matches {
		name := normalizeLegacyToolName(match[1])
		if name == "" {
			return model.Message{}, errors.New("legacy function name is empty")
		}
		parameters := legacyParameterPattern.FindAllStringSubmatch(match[2], -1)
		arguments := make(map[string]any, len(parameters))
		for _, parameter := range parameters {
			key := strings.TrimSpace(parameter[1])
			if _, exists := arguments[key]; exists {
				return model.Message{}, fmt.Errorf("legacy parameter %q is duplicated", key)
			}
			value := strings.TrimSpace(parameter[2])
			if value == "" {
				return model.Message{}, fmt.Errorf("legacy parameter %q is empty", key)
			}
			var decoded any
			if (strings.HasPrefix(value, "{") && strings.HasSuffix(value, "}")) || (strings.HasPrefix(value, "[") && strings.HasSuffix(value, "]")) {
				if err := json.Unmarshal([]byte(value), &decoded); err != nil {
					return model.Message{}, fmt.Errorf("legacy parameter %q is not valid JSON: %w", key, err)
				}
			} else {
				decoded = value
			}
			arguments[key] = decoded
		}
		encoded, err := json.Marshal(arguments)
		if err != nil {
			return model.Message{}, fmt.Errorf("encode legacy arguments: %w", err)
		}
		message.ToolCalls = append(message.ToolCalls, model.ToolCall{ID: fmt.Sprintf("legacy-call-%d", index+1), Name: name, Arguments: encoded})
	}
	return message, nil
}

func normalizeLegacyToolName(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	value = strings.ReplaceAll(value, "-", "_")
	value = strings.ReplaceAll(value, ".", "_")
	switch value {
	case "runcommand", "run_command":
		return "run_command"
	case "readfile", "read_file":
		return "read_file"
	case "writefile", "write_file":
		return "write_file"
	case "appendfile", "append_file":
		return "append_file"
	case "editfile", "edit_file":
		return "edit_file"
	case "promotefile", "promote_file":
		return "promote_file"
	case "listfiles", "list_files":
		return "list_files"
	case "searchfiles", "search_files":
		return "search_files"
	case "updateplan", "update_plan":
		return "update_plan"
	case "updateplanstep", "update_plan_step":
		return "update_plan_step"
	case "askuser", "ask_user":
		return "ask_user"
	default:
		return value
	}
}
