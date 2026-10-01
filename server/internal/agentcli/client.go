// Package agentcli 是 Go 控制面访问 Go Node Agent 的客户端（对应 Java AgentClient）。
// 不可达 / 出错时返回降级结构 {available:false, error, pods/deployments/events:[]}，不抛错。
package agentcli

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type Client struct {
	base string
	http *http.Client
}

func New(base string) *Client {
	return &Client{base: base, http: &http.Client{Timeout: 30 * time.Second}}
}

// FetchObject 拉取并解析 agent 某路径为对象；任何失败都返回降级对象（与 Java fetch 一致）。
func (c *Client) FetchObject(ctx context.Context, path string) map[string]any {
	m, _, err := c.RequestObject(ctx, http.MethodGet, path, nil)
	if err != nil {
		return degraded()
	}
	return m
}

// RequestObject 调用 Agent 的受约束写接口，并保留 HTTP 状态供控制面透传。
func (c *Client) RequestObject(ctx context.Context, method, path string, body any) (map[string]any, int, error) {
	var payload []byte
	var err error
	if body != nil {
		payload, err = json.Marshal(body)
		if err != nil {
			return nil, 0, err
		}
	}
	req, err := http.NewRequestWithContext(ctx, method, c.base+path, bytes.NewReader(payload))
	if err != nil {
		return nil, 0, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, resp.StatusCode, err
	}
	var result map[string]any
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &result); err != nil {
			return nil, resp.StatusCode, err
		}
	}
	if result == nil {
		result = map[string]any{}
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		message, _ := result["error"].(string)
		if message == "" {
			message = string(raw)
		}
		return result, resp.StatusCode, fmt.Errorf("agent request failed: %s", message)
	}
	return result, resp.StatusCode, nil
}

func degraded() map[string]any {
	return map[string]any{
		"available":   false,
		"error":       "agent unreachable",
		"pods":        []any{},
		"deployments": []any{},
		"events":      []any{},
	}
}
