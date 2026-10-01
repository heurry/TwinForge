// Package objectstore provides the narrow S3-compatible surface required by
// Agent artifacts. It deliberately uses path-style AWS Signature V4 requests
// so the Agent module does not pull a full cloud SDK into its runtime image.
package objectstore

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

var ErrDisabled = errors.New("agent object store is not configured")

type Config struct {
	Endpoint  string
	AccessKey string
	SecretKey string
	Bucket    string
	Region    string
	UseSSL    bool
	Client    *http.Client
}

type Client struct {
	endpoint  *url.URL
	accessKey string
	secretKey string
	bucket    string
	region    string
	http      *http.Client
	mu        sync.Mutex
	ready     bool
}

func New(cfg Config) (*Client, error) {
	endpoint := strings.TrimSpace(cfg.Endpoint)
	if endpoint == "" {
		return &Client{}, nil
	}
	if !strings.Contains(endpoint, "://") {
		scheme := "http"
		if cfg.UseSSL {
			scheme = "https"
		}
		endpoint = scheme + "://" + endpoint
	}
	parsed, err := url.Parse(endpoint)
	if err != nil || parsed.Host == "" {
		return nil, fmt.Errorf("parse Agent S3 endpoint: %w", err)
	}
	if strings.TrimSpace(cfg.AccessKey) == "" || strings.TrimSpace(cfg.SecretKey) == "" {
		return nil, errors.New("Agent S3 access key and secret key are required")
	}
	bucket := strings.TrimSpace(cfg.Bucket)
	if bucket == "" {
		bucket = "agent-artifacts"
	}
	region := strings.TrimSpace(cfg.Region)
	if region == "" {
		region = "us-east-1"
	}
	httpClient := cfg.Client
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 15 * time.Second}
	}
	return &Client{endpoint: parsed, accessKey: cfg.AccessKey, secretKey: cfg.SecretKey, bucket: bucket, region: region, http: httpClient}, nil
}

func (c *Client) Enabled() bool { return c != nil && c.endpoint != nil }
func (c *Client) Bucket() string {
	if c == nil {
		return ""
	}
	return c.bucket
}

func (c *Client) Ping(ctx context.Context) error {
	if !c.Enabled() {
		return ErrDisabled
	}
	resp, err := c.do(ctx, http.MethodHead, c.bucketPath(""), nil, "")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return c.ensureBucket(ctx)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return responseError("head bucket", resp)
	}
	c.mu.Lock()
	c.ready = true
	c.mu.Unlock()
	return nil
}

func (c *Client) Put(ctx context.Context, key string, content []byte, mediaType string) error {
	if !c.Enabled() {
		return ErrDisabled
	}
	if err := c.ensureBucket(ctx); err != nil {
		return err
	}
	resp, err := c.do(ctx, http.MethodPut, c.bucketPath(key), content, mediaType)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return responseError("put object", resp)
	}
	return nil
}

func (c *Client) Get(ctx context.Context, key string) ([]byte, error) {
	if !c.Enabled() {
		return nil, ErrDisabled
	}
	resp, err := c.do(ctx, http.MethodGet, c.bucketPath(key), nil, "")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, responseError("get object", resp)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, (10<<20)+1))
	if err != nil {
		return nil, fmt.Errorf("read Agent artifact object: %w", err)
	}
	if len(data) > 10<<20 {
		return nil, errors.New("Agent artifact object exceeds 10 MiB")
	}
	return data, nil
}

func (c *Client) ensureBucket(ctx context.Context) error {
	c.mu.Lock()
	if c.ready {
		c.mu.Unlock()
		return nil
	}
	defer c.mu.Unlock()
	resp, err := c.do(ctx, http.MethodHead, c.bucketPath(""), nil, "")
	if err != nil {
		return err
	}
	_ = resp.Body.Close()
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		c.ready = true
		return nil
	}
	if resp.StatusCode != http.StatusNotFound {
		return fmt.Errorf("Agent object store head bucket returned HTTP %d", resp.StatusCode)
	}
	resp, err = c.do(ctx, http.MethodPut, c.bucketPath(""), nil, "")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return responseError("create bucket", resp)
	}
	c.ready = true
	return nil
}

func (c *Client) bucketPath(key string) string {
	parts := []string{url.PathEscape(c.bucket)}
	for _, part := range strings.Split(strings.TrimPrefix(key, "/"), "/") {
		if part != "" {
			parts = append(parts, url.PathEscape(part))
		}
	}
	return "/" + strings.Join(parts, "/")
}

func (c *Client) do(ctx context.Context, method, canonicalPath string, body []byte, mediaType string) (*http.Response, error) {
	payloadHash := sha256.Sum256(body)
	payloadHex := hex.EncodeToString(payloadHash[:])
	now := time.Now().UTC()
	amzDate := now.Format("20060102T150405Z")
	date := now.Format("20060102")
	target := *c.endpoint
	target.Path = strings.TrimSuffix(c.endpoint.Path, "/") + canonicalPath
	req, err := http.NewRequestWithContext(ctx, method, target.String(), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("X-Amz-Date", amzDate)
	req.Header.Set("X-Amz-Content-Sha256", payloadHex)
	if mediaType != "" {
		req.Header.Set("Content-Type", mediaType)
	}
	canonicalHeaders := "host:" + req.URL.Host + "\n" + "x-amz-content-sha256:" + payloadHex + "\n" + "x-amz-date:" + amzDate + "\n"
	signedHeaders := "host;x-amz-content-sha256;x-amz-date"
	canonicalRequest := strings.Join([]string{method, canonicalPath, "", canonicalHeaders, signedHeaders, payloadHex}, "\n")
	requestHash := sha256.Sum256([]byte(canonicalRequest))
	scope := date + "/" + c.region + "/s3/aws4_request"
	stringToSign := "AWS4-HMAC-SHA256\n" + amzDate + "\n" + scope + "\n" + hex.EncodeToString(requestHash[:])
	dateKey := hmacSHA256([]byte("AWS4"+c.secretKey), date)
	regionKey := hmacSHA256(dateKey, c.region)
	serviceKey := hmacSHA256(regionKey, "s3")
	signingKey := hmacSHA256(serviceKey, "aws4_request")
	signature := hex.EncodeToString(hmacSHA256(signingKey, stringToSign))
	req.Header.Set("Authorization", "AWS4-HMAC-SHA256 Credential="+c.accessKey+"/"+scope+", SignedHeaders="+signedHeaders+", Signature="+signature)
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Agent object store request: %w", err)
	}
	return resp, nil
}

func hmacSHA256(key []byte, value string) []byte {
	h := hmac.New(sha256.New, key)
	_, _ = h.Write([]byte(value))
	return h.Sum(nil)
}

func responseError(action string, resp *http.Response) error {
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
	return fmt.Errorf("Agent object store %s returned HTTP %d: %s", action, resp.StatusCode, strings.TrimSpace(string(data)))
}
