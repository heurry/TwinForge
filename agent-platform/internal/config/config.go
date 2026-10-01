package config

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// API contains process-level settings for agent-api.
type API struct {
	Address               string
	DatabaseURL           string
	RedisURL              string
	AuthEnabled           bool
	AuthJWTSecret         string
	AuthDefaultTenant     string
	MCPAllowedHosts       []string
	A2ADefaultAgentID     string
	SandboxEndpoint       string
	SandboxToken          string
	ProjectAccessEndpoint string
	ProjectAccessToken    string
	S3Endpoint            string
	S3AccessKey           string
	S3SecretKey           string
	S3Bucket              string
	S3Region              string
	S3UseSSL              bool
	EmbeddingURL          string
	EmbeddingDim          int
	EmbeddingRequireLive  bool
	CollaborationBootstrapTenant string
}

// Worker contains process-level settings for agent-worker.
type Worker struct {
	ID                          string
	DatabaseURL                 string
	RedisURL                    string
	ModelServices               string
	ToolAllowedHosts            []string
	MCPAllowedHosts             []string
	WorkspaceRoot               string
	WorkspaceAllowWrite         bool
	WorkspaceMode               string
	SandboxEndpoint             string
	SandboxToken                string
	DependencyInstallerEndpoint string
	DependencyInstallerToken    string
	ProjectAccessEndpoint       string
	ProjectAccessToken          string
	DependencySources           string
	EnvironmentTemplate         string
	EnvironmentDependencies     []string
	PollInterval                time.Duration
	LeaseDuration               time.Duration
	HeartbeatInterval           time.Duration
	S3Endpoint                  string
	S3AccessKey                 string
	S3SecretKey                 string
	S3Bucket                    string
	S3Region                    string
	S3UseSSL                    bool
	EmbeddingURL                string
	EmbeddingDim                int
	EmbeddingRequireLive        bool
}

// LoadAPI reads the API configuration from the environment.
func LoadAPI() API {
	return API{
		Address:               envOrDefault("AGENT_API_ADDR", ":8180"),
		DatabaseURL:           strings.TrimSpace(os.Getenv("AGENT_DATABASE_URL")),
		RedisURL:              strings.TrimSpace(os.Getenv("AGENT_REDIS_URL")),
		AuthEnabled:           boolEnv("AGENT_AUTH_ENABLED"),
		AuthJWTSecret:         strings.TrimSpace(os.Getenv("AGENT_AUTH_JWT_SECRET")),
		AuthDefaultTenant:     strings.TrimSpace(os.Getenv("AGENT_AUTH_DEFAULT_TENANT")),
		MCPAllowedHosts:       splitCSV(os.Getenv("AGENT_MCP_ALLOWED_HOSTS")),
		A2ADefaultAgentID:     strings.TrimSpace(os.Getenv("AGENT_A2A_DEFAULT_AGENT_ID")),
		SandboxEndpoint:       strings.TrimSpace(os.Getenv("AGENT_SANDBOX_ENDPOINT")),
		SandboxToken:          strings.TrimSpace(os.Getenv("AGENT_SANDBOX_TOKEN")),
		ProjectAccessEndpoint: strings.TrimSpace(os.Getenv("AGENT_PROJECT_ACCESS_ENDPOINT")),
		ProjectAccessToken:    strings.TrimSpace(os.Getenv("AGENT_PROJECT_ACCESS_TOKEN")),
		S3Endpoint:            strings.TrimSpace(os.Getenv("AGENT_S3_ENDPOINT")),
		S3AccessKey:           strings.TrimSpace(os.Getenv("AGENT_S3_ACCESS_KEY")),
		S3SecretKey:           strings.TrimSpace(os.Getenv("AGENT_S3_SECRET_KEY")),
		S3Bucket:              envOrDefault("AGENT_S3_BUCKET", "agent-artifacts"),
		S3Region:              envOrDefault("AGENT_S3_REGION", "us-east-1"),
		S3UseSSL:              boolEnv("AGENT_S3_USE_SSL"),
		EmbeddingURL:          strings.TrimSpace(os.Getenv("AGENT_EMBEDDING_URL")),
		EmbeddingDim:          intEnvOrDefault("AGENT_EMBEDDING_DIM", 1024),
		EmbeddingRequireLive:  boolEnvOrDefault("AGENT_EMBEDDING_REQUIRE_LIVE", true),
		CollaborationBootstrapTenant: strings.TrimSpace(os.Getenv("AGENT_COLLABORATION_BOOTSTRAP_TENANT")),
	}
}

func boolEnv(key string) bool {
	value, err := strconv.ParseBool(strings.TrimSpace(os.Getenv(key)))
	return err == nil && value
}

// LoadWorker reads the worker configuration from the environment.
func LoadWorker() (Worker, error) {
	id := strings.TrimSpace(os.Getenv("AGENT_WORKER_ID"))
	if id == "" {
		hostname, err := os.Hostname()
		if err != nil {
			return Worker{}, fmt.Errorf("resolve worker id: %w", err)
		}
		id = hostname
	}
	databaseURL := strings.TrimSpace(os.Getenv("AGENT_DATABASE_URL"))
	if databaseURL == "" {
		return Worker{}, errors.New("AGENT_DATABASE_URL is required")
	}
	modelServices := strings.TrimSpace(os.Getenv("AGENT_MODEL_SERVICES"))
	if modelServices == "" {
		return Worker{}, errors.New("AGENT_MODEL_SERVICES is required")
	}
	poll, err := durationOrDefault("AGENT_WORKER_POLL_INTERVAL", time.Second)
	if err != nil {
		return Worker{}, err
	}
	lease, err := durationOrDefault("AGENT_WORKER_LEASE_DURATION", 30*time.Second)
	if err != nil {
		return Worker{}, err
	}
	heartbeat, err := durationOrDefault("AGENT_WORKER_HEARTBEAT_INTERVAL", 2*time.Second)
	if err != nil {
		return Worker{}, err
	}
	return Worker{
		ID: id, DatabaseURL: databaseURL, RedisURL: strings.TrimSpace(os.Getenv("AGENT_REDIS_URL")), ModelServices: modelServices,
		ToolAllowedHosts:            splitCSV(os.Getenv("AGENT_TOOL_ALLOWED_HOSTS")),
		MCPAllowedHosts:             splitCSV(os.Getenv("AGENT_MCP_ALLOWED_HOSTS")),
		WorkspaceRoot:               strings.TrimSpace(os.Getenv("AGENT_WORKSPACE_ROOT")),
		WorkspaceAllowWrite:         boolEnv("AGENT_WORKSPACE_ALLOW_WRITE"),
		WorkspaceMode:               envOrDefault("AGENT_WORKSPACE_MODE", "sandbox"),
		SandboxEndpoint:             strings.TrimSpace(os.Getenv("AGENT_SANDBOX_ENDPOINT")),
		SandboxToken:                strings.TrimSpace(os.Getenv("AGENT_SANDBOX_TOKEN")),
		DependencyInstallerEndpoint: strings.TrimSpace(os.Getenv("AGENT_DEPENDENCY_INSTALLER_ENDPOINT")),
		DependencyInstallerToken:    strings.TrimSpace(os.Getenv("AGENT_DEPENDENCY_INSTALLER_TOKEN")),
		ProjectAccessEndpoint:       strings.TrimSpace(os.Getenv("AGENT_PROJECT_ACCESS_ENDPOINT")),
		ProjectAccessToken:          strings.TrimSpace(os.Getenv("AGENT_PROJECT_ACCESS_TOKEN")),
		DependencySources:           envOrDefault("AGENT_DEPENDENCY_SOURCES", `{"pypi":"https://pypi.org/simple"}`),
		EnvironmentTemplate:         envOrDefault("AGENT_ENVIRONMENT_TEMPLATE", "python-game:1"),
		EnvironmentDependencies:     splitCSV(os.Getenv("AGENT_ENVIRONMENT_DEPENDENCIES")),
		S3Endpoint:                  strings.TrimSpace(os.Getenv("AGENT_S3_ENDPOINT")),
		S3AccessKey:                 strings.TrimSpace(os.Getenv("AGENT_S3_ACCESS_KEY")),
		S3SecretKey:                 strings.TrimSpace(os.Getenv("AGENT_S3_SECRET_KEY")),
		S3Bucket:                    envOrDefault("AGENT_S3_BUCKET", "agent-artifacts"),
		S3Region:                    envOrDefault("AGENT_S3_REGION", "us-east-1"),
		S3UseSSL:                    boolEnv("AGENT_S3_USE_SSL"),
		EmbeddingURL:                strings.TrimSpace(os.Getenv("AGENT_EMBEDDING_URL")),
		EmbeddingDim:                intEnvOrDefault("AGENT_EMBEDDING_DIM", 1024),
		EmbeddingRequireLive:        boolEnvOrDefault("AGENT_EMBEDDING_REQUIRE_LIVE", true),
		PollInterval:                poll, LeaseDuration: lease, HeartbeatInterval: heartbeat,
	}, nil
}

func envOrDefault(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func intEnvOrDefault(key string, fallback int) int {
	value, err := strconv.Atoi(strings.TrimSpace(os.Getenv(key)))
	if err != nil || value <= 0 {
		return fallback
	}
	return value
}

func boolEnvOrDefault(key string, fallback bool) bool {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseBool(value)
	if err != nil {
		return fallback
	}
	return parsed
}

func durationOrDefault(key string, fallback time.Duration) (time.Duration, error) {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback, nil
	}
	if integer, err := strconv.ParseInt(value, 10, 64); err == nil {
		if integer <= 0 {
			return 0, fmt.Errorf("%s must be positive", key)
		}
		return time.Duration(integer) * time.Millisecond, nil
	}
	duration, err := time.ParseDuration(value)
	if err != nil || duration <= 0 {
		return 0, fmt.Errorf("%s must be a positive duration: %q", key, value)
	}
	return duration, nil
}

func splitCSV(value string) []string {
	var result []string
	for _, item := range strings.Split(value, ",") {
		if item = strings.TrimSpace(item); item != "" {
			result = append(result, item)
		}
	}
	return result
}
