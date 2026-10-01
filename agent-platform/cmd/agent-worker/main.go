package main

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/config"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/embedding"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/execution"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/objectstore"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/observability"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/persistence/postgres"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/realtime"
	runtimepkg "github.com/heurry/cloudnative-infra-platform/agent-platform/internal/runtime"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

func main() {
	cfg, err := config.LoadWorker()
	if err != nil {
		slog.Error("load worker configuration", "err", err)
		os.Exit(2)
	}
	services, err := execution.ParseModelServices(cfg.ModelServices)
	if err != nil {
		slog.Error("load model services", "err", err)
		os.Exit(2)
	}
	connectCtx, connectCancel := context.WithTimeout(context.Background(), 10*time.Second)
	pool, err := postgres.Open(connectCtx, cfg.DatabaseURL)
	connectCancel()
	if err != nil {
		slog.Error("connect agent database", "err", err)
		os.Exit(1)
	}
	defer pool.Close()
	store := postgres.NewRunStore(pool)
	store.SetOutboxEnabled(cfg.RedisURL != "")
	artifactObjects, err := objectstore.New(objectstore.Config{Endpoint: cfg.S3Endpoint, AccessKey: cfg.S3AccessKey, SecretKey: cfg.S3SecretKey, Bucket: cfg.S3Bucket, Region: cfg.S3Region, UseSSL: cfg.S3UseSSL})
	if err != nil {
		slog.Error("configure Agent object store", "err", err)
		os.Exit(2)
	}
	store.SetArtifactObjectStore(artifactObjects)
	var redisBus *realtime.RedisBus
	if cfg.RedisURL != "" {
		redisBus, err = realtime.NewRedisBus(cfg.RedisURL, "agent:v1:")
		if err != nil {
			slog.Error("configure Agent Redis", "err", err)
			os.Exit(2)
		}
		defer redisBus.Close()
	}
	embeddingClient := embedding.New(cfg.EmbeddingURL, cfg.EmbeddingDim, cfg.EmbeddingRequireLive, nil)
	store.SetMemoryEmbeddingProvider(embeddingClient)
	telemetryShutdown, telemetryErr := observability.Init(context.Background(), observability.Config{ServiceName: "agent-worker", ServiceVersion: "0.2.0", OTLPEndpoint: os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")})
	if telemetryErr != nil {
		slog.Warn("agent worker tracing degraded", "err", telemetryErr)
	}
	defer func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = telemetryShutdown(shutdownCtx)
	}()
	metricsServer := &http.Server{Addr: ":8181", Handler: observability.MetricsHandler(), ReadHeaderTimeout: 5 * time.Second}
	go func() {
		if err := metricsServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Error("agent worker metrics server failed", "err", err)
		}
	}()
	defer func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = metricsServer.Shutdown(shutdownCtx)
	}()
	resolver, err := execution.NewResolver(store, execution.Config{
		ModelServices: services, ToolAllowedHosts: append(cfg.ToolAllowedHosts, cfg.MCPAllowedHosts...),
		WorkspaceRoot: cfg.WorkspaceRoot, WorkspaceAllowWrite: cfg.WorkspaceAllowWrite,
		WorkspaceMode: cfg.WorkspaceMode, SandboxEndpoint: cfg.SandboxEndpoint, SandboxToken: cfg.SandboxToken,
		DependencyInstallerEndpoint: cfg.DependencyInstallerEndpoint, DependencyInstallerToken: cfg.DependencyInstallerToken,
		ProjectAccessEndpoint: cfg.ProjectAccessEndpoint, ProjectAccessToken: cfg.ProjectAccessToken,
		DependencySources: parseDependencySources(cfg.DependencySources), EnvironmentTemplate: cfg.EnvironmentTemplate,
		EnvironmentDependencies: cfg.EnvironmentDependencies,
		HTTPClient:              &http.Client{Transport: otelhttp.NewTransport(http.DefaultTransport)},
	})
	if err != nil {
		slog.Error("create execution resolver", "err", err)
		os.Exit(1)
	}
	processor, err := runtimepkg.NewReActProcessor(resolver)
	if err != nil {
		slog.Error("create ReAct processor", "err", err)
		os.Exit(1)
	}
	worker, err := runtimepkg.NewWorker(store, processor, runtimepkg.Config{
		WorkerID: cfg.ID, RuntimeVersion: "agent-worker/0.3.0", ToolContractVersion: "tool-contract/v1", ProtocolVersion: "structured-tool-calls/v1", CapabilityHash: "agent-worker/0.3.0:tool-contract/v1:structured-tool-calls/v1", Capabilities: []string{"react-v1", "tool-contract-v1", "structured-tool-calls-v1", "sandbox-v1", "workspace-nested-write-v1", "delegate_agent-v1"}, PollInterval: cfg.PollInterval,
		LeaseDuration: cfg.LeaseDuration, HeartbeatInterval: cfg.HeartbeatInterval,
	})
	if err != nil {
		slog.Error("create worker", "err", err)
		os.Exit(1)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if os.Getenv("AGENT_MEMORY_EXTRACTOR_ENABLED") != "false" {
		memoryWorker, memoryWorkerErr := runtimepkg.NewMemoryWriteWorker(
			resolver, resolver, resolver, runtimepkg.MemoryWriteWorkerConfig{LeaseSeconds: 300, PollInterval: 2 * time.Second, MaxAttempts: 5, RetryBackoff: 5 * time.Second},
		)
		if memoryWorkerErr != nil {
			slog.Error("create memory write worker", "err", memoryWorkerErr)
		} else {
			go func() {
				if err := memoryWorker.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
					slog.Error("memory write worker stopped", "err", err)
				}
			}()
		}
	}
	if redisBus != nil && artifactObjects.Enabled() {
		archiver := &observability.ObservationArchiver{Queue: redisBus, Store: store, Objects: artifactObjects, Consumer: cfg.ID}
		go func() {
			if err := archiver.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
				slog.Error("Agent observation archiver stopped", "err", err)
			}
		}()
	}
	slog.Info("agent worker started", "worker_id", cfg.ID)
	if err := worker.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
		slog.Error("agent worker stopped unexpectedly", "worker_id", cfg.ID, "err", err)
		os.Exit(1)
	}
	slog.Info("agent worker stopped", "worker_id", cfg.ID)
}

func parseDependencySources(raw string) map[string]string {
	var result map[string]string
	if json.Unmarshal([]byte(raw), &result) != nil {
		return nil
	}
	return result
}
