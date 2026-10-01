package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	platformauth "github.com/heurry/cloudnative-infra-platform/agent-platform/internal/auth"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/bootstrap"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/config"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/embedding"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/httpapi"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/objectstore"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/observability"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/persistence/postgres"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/provider/sandbox"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/realtime"
)

func main() {
	cfg := config.LoadAPI()
	if cfg.DatabaseURL == "" {
		slog.Error("AGENT_DATABASE_URL is required")
		os.Exit(2)
	}
	if cfg.AuthEnabled && cfg.AuthJWTSecret == "" {
		slog.Error("AGENT_AUTH_JWT_SECRET is required when AGENT_AUTH_ENABLED=true")
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
	runs := postgres.NewRunStore(pool)
	runs.SetOutboxEnabled(cfg.RedisURL != "")
	if cfg.CollaborationBootstrapTenant != "" {
		bootstrapCtx, bootstrapCancel := context.WithTimeout(context.Background(), 30*time.Second)
		result, bootstrapErr := bootstrap.Ensure(bootstrapCtx, runs, runs, bootstrap.Config{TenantID: cfg.CollaborationBootstrapTenant})
		bootstrapCancel()
		if bootstrapErr != nil {
			slog.Error("install Agent collaboration defaults", "tenant", cfg.CollaborationBootstrapTenant, "err", bootstrapErr)
			os.Exit(1)
		}
		slog.Info("Agent collaboration defaults ready", "tenant", cfg.CollaborationBootstrapTenant, "reviewer_agent_id", result.ReviewerAgentID, "reviewer_version_id", result.ReviewerVersionID, "builder_version_id", result.BuilderVersionID, "builder_version_created", result.BuilderVersionCreated)
	}
	artifactObjects, err := objectstore.New(objectstore.Config{Endpoint: cfg.S3Endpoint, AccessKey: cfg.S3AccessKey, SecretKey: cfg.S3SecretKey, Bucket: cfg.S3Bucket, Region: cfg.S3Region, UseSSL: cfg.S3UseSSL})
	if err != nil {
		slog.Error("configure Agent object store", "err", err)
		os.Exit(2)
	}
	runs.SetArtifactObjectStore(artifactObjects)
	embeddingClient := embedding.New(cfg.EmbeddingURL, cfg.EmbeddingDim, cfg.EmbeddingRequireLive, nil)
	runs.SetMemoryEmbeddingProvider(embeddingClient)
	reconcileStorage := func(parent context.Context) {
		reconcileCtx, cancel := context.WithTimeout(parent, 30*time.Second)
		defer cancel()
		result, reconcileErr := runs.ReconcileStorageFoundation(reconcileCtx)
		if reconcileErr != nil {
			slog.Warn("Agent storage reconciliation degraded", "err", reconcileErr)
		} else if result.SessionMessagesInserted > 0 || result.RunStatesUpserted > 0 {
			slog.Info("Agent storage projections reconciled",
				"sessions_scanned", result.SessionsScanned,
				"session_messages_inserted", result.SessionMessagesInserted,
				"run_states_upserted", result.RunStatesUpserted)
		}
		if artifactObjects.Enabled() {
			if pingErr := artifactObjects.Ping(reconcileCtx); pingErr != nil {
				slog.Warn("Agent artifact object storage degraded", "err", pingErr)
			} else if artifactResult, artifactErr := runs.ReconcileArtifactObjects(reconcileCtx, 100); artifactErr != nil {
				slog.Warn("Agent artifact object reconciliation degraded", "err", artifactErr)
			} else if artifactResult.Migrated > 0 {
				slog.Info("Agent artifacts migrated to MinIO", "migrated", artifactResult.Migrated)
			}
		}
		if embeddingClient.Enabled() {
			if pingErr := embeddingClient.Ping(reconcileCtx); pingErr != nil {
				slog.Warn("Agent semantic memory degraded", "err", pingErr)
			} else if memoryResult, memoryErr := runs.ReconcileMemoryEmbeddings(reconcileCtx, 50); memoryErr != nil {
				slog.Warn("Agent memory embedding reconciliation degraded", "err", memoryErr)
			} else if memoryResult.Embedded > 0 {
				slog.Info("Agent memories embedded", "embedded", memoryResult.Embedded)
			}
		}
	}
	reconcileStorage(context.Background())
	telemetryShutdown, telemetryErr := observability.Init(context.Background(), observability.Config{ServiceName: "agent-api", ServiceVersion: "0.2.0", OTLPEndpoint: os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")})
	if telemetryErr != nil {
		slog.Warn("agent api tracing degraded", "err", telemetryErr)
	}
	defer func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = telemetryShutdown(shutdownCtx)
	}()
	router := http.NewServeMux()
	router.Handle("/metrics", observability.MetricsHandler())
	apiServer := httpapi.NewWithRuns(runs, pool)
	apiServer.SetSecurityEnforced(cfg.AuthEnabled)
	apiServer.SetMCPPolicy(cfg.MCPAllowedHosts, os.LookupEnv)
	apiServer.SetA2ADiscovery(cfg.AuthDefaultTenant, cfg.A2ADefaultAgentID)
	if artifactObjects.Enabled() {
		apiServer.SetColdStoreReadiness(artifactObjects)
	}
	if embeddingClient.Enabled() {
		apiServer.SetSemanticStoreReadiness(embeddingClient)
	}
	var redisBus *realtime.RedisBus
	if cfg.RedisURL != "" {
		redisBus, err = realtime.NewRedisBus(cfg.RedisURL, "agent:v1:")
		if err != nil {
			slog.Error("configure Agent Redis", "err", err)
			os.Exit(2)
		}
		pingCtx, pingCancel := context.WithTimeout(context.Background(), 2*time.Second)
		if pingErr := redisBus.Ping(pingCtx); pingErr != nil {
			slog.Warn("Agent Redis unavailable; PostgreSQL polling remains active", "err", pingErr)
		}
		pingCancel()
		defer func() { _ = redisBus.Close() }()
		apiServer.SetRunEventSubscriber(redisBus)
		apiServer.SetHotStoreReadiness(redisBus)
	}
	if cfg.ProjectAccessEndpoint != "" && cfg.ProjectAccessToken != "" {
		promoter, promoterErr := sandbox.NewPromoter(sandbox.Config{Endpoint: cfg.ProjectAccessEndpoint, Token: cfg.ProjectAccessToken, HTTPClient: &http.Client{Timeout: 30 * time.Second}})
		if promoterErr != nil {
			slog.Error("configure artifact promoter", "err", promoterErr)
			os.Exit(2)
		}
		apiServer.SetArtifactPromoter(promoter)
	}
	authenticatedAPI := (platformauth.Config{Enabled: cfg.AuthEnabled, JWTSecret: cfg.AuthJWTSecret, DefaultTenant: cfg.AuthDefaultTenant}).Middleware(apiServer.Handler())
	router.Handle("/", observability.HTTPMiddleware(authenticatedAPI))
	server := &http.Server{
		Addr:              cfg.Address,
		Handler:           router,
		ReadHeaderTimeout: 5 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		ticker := time.NewTicker(time.Minute)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				reconcileStorage(ctx)
			}
		}
	}()
	if redisBus != nil {
		relay := realtime.NewRelay(runs, redisBus)
		go func() {
			if relayErr := relay.Run(ctx); relayErr != nil && !errors.Is(relayErr, context.Canceled) {
				slog.Error("Agent realtime relay stopped", "err", relayErr)
			}
		}()
	}

	errCh := make(chan error, 1)
	go func() {
		slog.Info("agent api listening", "addr", cfg.Address)
		errCh <- server.ListenAndServe()
	}()

	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			slog.Error("agent api shutdown failed", "err", err)
			os.Exit(1)
		}
	case err := <-errCh:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Error("agent api failed", "err", err)
			os.Exit(1)
		}
	}
}
