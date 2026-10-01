// Package embedding connects Agent memory to the platform's embedding service.
package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"
)

var ErrDisabled = errors.New("Agent embedding provider is not configured")

type Result struct {
	Vectors [][]float32
	Model   string
	Dim     int
	Mode    string
}

type Status struct {
	Configured bool
	Ready      bool
	Model      string
	Mode       string
	Dim        int
	LastError  string
	CheckedAt  time.Time
}

type Client struct {
	base        string
	dimension   int
	requireLive bool
	http        *http.Client
	mu          sync.RWMutex
	status      Status
}

func New(base string, dimension int, requireLive bool, httpClient *http.Client) *Client {
	base = strings.TrimRight(strings.TrimSpace(base), "/")
	if dimension <= 0 {
		dimension = 1024
	}
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 15 * time.Second}
	}
	return &Client{base: base, dimension: dimension, requireLive: requireLive, http: httpClient, status: Status{Configured: base != "", Dim: dimension}}
}

func (c *Client) Enabled() bool { return c != nil && c.base != "" }

func (c *Client) Embed(ctx context.Context, texts []string, isQuery bool) (Result, error) {
	if !c.Enabled() {
		return Result{}, ErrDisabled
	}
	payload, err := json.Marshal(map[string]any{"texts": texts, "is_query": isQuery})
	if err != nil {
		return Result{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+"/internal/embed", bytes.NewReader(payload))
	if err != nil {
		return Result{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		c.record(Status{Configured: true, Dim: c.dimension, LastError: err.Error(), CheckedAt: time.Now().UTC()})
		return Result{}, fmt.Errorf("call embedding service: %w", err)
	}
	defer resp.Body.Close()
	data, readErr := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if readErr != nil {
		return Result{}, fmt.Errorf("read embedding response: %w", readErr)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		err = fmt.Errorf("embedding service returned HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
		c.record(Status{Configured: true, Dim: c.dimension, LastError: err.Error(), CheckedAt: time.Now().UTC()})
		return Result{}, err
	}
	var output struct {
		Embeddings [][]float32 `json:"embeddings"`
		Model      string      `json:"model"`
		Dim        int         `json:"dim"`
		Mode       string      `json:"mode"`
	}
	if err := json.Unmarshal(data, &output); err != nil {
		return Result{}, fmt.Errorf("decode embedding response: %w", err)
	}
	if len(output.Embeddings) != len(texts) || output.Dim != c.dimension {
		err = fmt.Errorf("embedding contract mismatch: received %d vector(s), dim=%d; expected %d vector(s), dim=%d", len(output.Embeddings), output.Dim, len(texts), c.dimension)
		c.record(Status{Configured: true, Model: output.Model, Mode: output.Mode, Dim: output.Dim, LastError: err.Error(), CheckedAt: time.Now().UTC()})
		return Result{}, err
	}
	if strings.TrimSpace(output.Model) == "" {
		return Result{}, errors.New("embedding response omitted model identity")
	}
	for _, vector := range output.Embeddings {
		if len(vector) != c.dimension {
			return Result{}, fmt.Errorf("embedding vector dimension is %d, expected %d", len(vector), c.dimension)
		}
		for _, value := range vector {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return Result{}, errors.New("embedding response contains a non-finite value")
			}
		}
	}
	if c.requireLive && output.Mode != "live" {
		err = fmt.Errorf("embedding provider returned %q vectors; live vectors are required", output.Mode)
		c.record(Status{Configured: true, Model: output.Model, Mode: output.Mode, Dim: output.Dim, LastError: err.Error(), CheckedAt: time.Now().UTC()})
		return Result{}, err
	}
	result := Result{Vectors: output.Embeddings, Model: output.Model, Dim: output.Dim, Mode: output.Mode}
	c.record(Status{Configured: true, Ready: true, Model: output.Model, Mode: output.Mode, Dim: output.Dim, CheckedAt: time.Now().UTC()})
	return result, nil
}

func (c *Client) Ping(ctx context.Context) error {
	_, err := c.Embed(ctx, []string{"Agent semantic memory readiness probe"}, false)
	return err
}

func (c *Client) Status() Status {
	if c == nil {
		return Status{}
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.status
}

func (c *Client) record(status Status) {
	c.mu.Lock()
	c.status = status
	c.mu.Unlock()
}
