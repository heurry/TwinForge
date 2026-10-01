// Command server 是云原生平台的 Go 控制面后端（迁移自 Java/Spring Boot）。
// Phase 0：连 PostgreSQL、应用迁移、暴露 /api/health。
// Phase 3：接入 Python AI 服务（诊断）+ SSE 反向代理（Copilot 单一入口）+ 优雅关机。
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

	"github.com/heurry/cloudnative-infra-platform/server/internal/agentcli"
	"github.com/heurry/cloudnative-infra-platform/server/internal/aiclient"
	"github.com/heurry/cloudnative-infra-platform/server/internal/auth"
	"github.com/heurry/cloudnative-infra-platform/server/internal/blob"
	"github.com/heurry/cloudnative-infra-platform/server/internal/cache"
	"github.com/heurry/cloudnative-infra-platform/server/internal/config"
	"github.com/heurry/cloudnative-infra-platform/server/internal/db"
	"github.com/heurry/cloudnative-infra-platform/server/internal/httpx"
	"github.com/heurry/cloudnative-infra-platform/server/internal/k8s"
	"github.com/heurry/cloudnative-infra-platform/server/internal/metrics"
	"github.com/heurry/cloudnative-infra-platform/server/internal/obs"
	"github.com/heurry/cloudnative-infra-platform/server/internal/serving"
	"github.com/heurry/cloudnative-infra-platform/server/internal/store"
	"github.com/heurry/cloudnative-infra-platform/server/internal/training"
)

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))
	cfg := config.Load()

	// 0) D1：全链路可观测。设全局 propagator + （配置了 OTLP 时）TracerProvider；
	// 未配置 endpoint 时为 no-op，绝不阻塞启动。/metrics 由路由始终暴露。
	otelShutdown, err := obs.Init(context.Background(), obs.Config{
		ServiceName:  cfg.OTelService,
		ServiceVer:   "dev",
		OTLPEndpoint: cfg.OTLPEndpoint,
		OTLPInsecure: cfg.OTLPInsecure,
	})
	if err != nil {
		slog.Warn("otel init failed (degraded, traces off)", "err", err)
		otelShutdown = func(context.Context) error { return nil }
	} else if cfg.OTLPEndpoint != "" {
		slog.Info("otel tracing enabled", "endpoint", cfg.OTLPEndpoint, "service", cfg.OTelService)
	}

	// 1) 迁移：空库可重建出与 Flyway 等价的 schema + 种子。
	if err := db.Migrate(cfg.MigrateURL()); err != nil {
		slog.Error("migrate failed", "err", err)
		os.Exit(1)
	}
	slog.Info("migrations applied")

	// 2) 连接池（注入各 handler）。
	ctx := context.Background()
	pool, err := db.Connect(ctx, cfg.DatabaseURL)
	if err != nil {
		slog.Error("db connect failed", "err", err)
		os.Exit(1)
	}
	defer pool.Close()
	slog.Info("db connected")

	// 3) AI 边界：诊断/嵌入客户端 + SSE 反向代理（不可用时降级为 nil，路由跳过对应接口）。
	aiProxy, err := httpx.NewProxy(cfg.AIServiceBaseURL)
	if err != nil {
		slog.Warn("ai proxy disabled", "err", err)
		aiProxy = nil
	}

	// 3.5) Phase 5/5B.1：控制面 client-go 直读集群（kubeconfig 加载失败时降级为 nil，
	// /api/kubernetes/* 返回 disabled，不阻塞启动）。
	k8sCollector, k8sErr := k8s.NewCollector(cfg.KubeconfigPath)
	k8sErrStr := ""
	if k8sErr != nil {
		slog.Warn("kubernetes collector unavailable (degraded)", "err", k8sErr)
		k8sErrStr = k8sErr.Error()
	} else {
		slog.Info("kubernetes collector ready")
		if cfg.AllowK8sWrites {
			if err := k8sCollector.EnsureAIBrixGatewayService(ctx); err != nil {
				slog.Warn("stable AIBrix NodePort unavailable (serving degraded)", "err", err)
			} else {
				slog.Info("stable AIBrix NodePort ready", "endpoint", cfg.AIBrixGatewayBaseURL)
			}
		}
	}

	// 3.6) Phase 5/Option A：vLLM Prometheus 指标抓取器（需 k8s collector；否则 nil→serving 指标走旧路径）。
	var servingScraper *serving.Scraper
	if k8sCollector != nil {
		servingScraper = serving.New(k8sCollector, cfg.VLLMMetricsPort)
	}

	// 3.7) 5B.4a：Redis 横切（缓存 / 限流 / 幂等）。不可达即降级，不阻塞启动。
	cacheClient := cache.New(context.Background(), cfg.RedisURL)
	defer cacheClient.Close()

	// 3.7) 5B.4b：对象存储（基准报告/评测产物/知识源文件）。endpoint 空或不可达即降级，不阻塞启动。
	blobStore := blob.New(context.Background(), blob.Config{
		Endpoint:   cfg.S3Endpoint,
		AccessKey:  cfg.S3AccessKey,
		SecretKey:  cfg.S3SecretKey,
		Bucket:     cfg.S3Bucket,
		UseSSL:     cfg.S3UseSSL,
		PresignTTL: cfg.S3PresignTTL,
	})

	// 3.8) Phase F/F1：Kubeflow 训练客户端（PyTorchJob via dynamic client）。
	// kubeconfig 加载失败时降级为 nil；CRD 未装时调用返回 error，均不阻塞启动。
	trainingClient, trainingErr := training.NewClient(cfg.KubeconfigPath)
	trainingErrStr := ""
	if trainingErr != nil {
		slog.Warn("training client unavailable (degraded)", "err", trainingErr)
		trainingErrStr = trainingErr.Error()
	} else {
		slog.Info("training client ready")
	}

	// 4) HTTP。
	st := store.New(pool)
	apiSvc := &httpx.API{
		Pool:                   pool,
		Agent:                  agentcli.New(cfg.AgentBaseURL),
		Metrics:                metrics.NewService(pool),
		Store:                  st,
		AI:                     aiclient.New(cfg.AIServiceBaseURL, cfg.AIRequestTimeout),
		AIProxy:                aiProxy,
		K8s:                    k8sCollector,
		K8sErr:                 k8sErrStr,
		AIBrixGatewayBaseURL:   cfg.AIBrixGatewayBaseURL,
		AllowK8sWrites:         cfg.AllowK8sWrites,
		K8sWriteNamespaces:     cfg.K8sWriteNamespaces,
		Training:               trainingClient,
		TrainingErr:            trainingErrStr,
		AllowTraining:          cfg.AllowTraining,
		TrainingNamespaces:     cfg.TrainingNamespaces,
		TrainingHostPath:       cfg.TrainingHostPath,
		TrainingMountPath:      cfg.TrainingMountPath,
		TrainingArtifactSecret: cfg.TrainingArtifactSecret,
		TrainingOutputRoot:     cfg.TrainingOutputRoot,
		Serving:                servingScraper,
		Cadvisor:               metrics.NewCadvisorCollector(cfg.CadvisorURL),
		CORSOrigins:            cfg.CORSOrigins,
		Cache:                  cacheClient,
		CacheTTL:               cfg.CacheTTL,
		IdempotencyTTL:         cfg.IdempotencyTTL,
		RateLimitRPS:           cfg.RateLimitRPS,
		RateLimitBurst:         cfg.RateLimitBurst,
		Blob:                   blobStore,
		StorageArchiveEnabled:  cfg.StorageArchiveEnabled,
		AuthEnabled:            cfg.AuthEnabled,
		Auth:                   auth.NewIssuer(cfg.AuthJWTSecret, cfg.AuthTokenTTL),
		Users:                  auth.ParseUsers(cfg.AuthUsers),
		RoutingShadowEnabled:   cfg.RoutingShadowEnabled,
		RAGRerankFeedback:      cfg.RAGRerankFeedback,
		CIProvider:             cfg.CIProvider,
		GitLabBaseURL:          cfg.GitLabBaseURL,
		GitLabProjectID:        cfg.GitLabProjectID,
		GitLabToken:            cfg.GitLabToken,
		GitLabRef:              cfg.GitLabRef,
		GitHubRepository:       cfg.GitHubRepository,
		GitHubToken:            cfg.GitHubToken,
		GitHubWorkflow:         cfg.GitHubWorkflow,
	}
	if cfg.AuthEnabled {
		slog.Info("auth enabled (RBAC)", "users", len(apiSvc.Users))
	}
	router := httpx.NewRouter(apiSvc)

	// 4.5) 后台任务（随关机一并停止）：服务注册表 reaper（5B.2）+ vLLM 指标抓取（Option A）。
	bgCtx, bgCancel := context.WithCancel(context.Background())
	defer bgCancel()
	go runRegistryReaper(bgCtx, st, cfg.RegistrySweep, cfg.RegistryTTL.Seconds())
	go apiSvc.RunInferenceServingReconciler(bgCtx, 30*time.Second)
	if servingScraper != nil {
		go runServingScraper(bgCtx, servingScraper, cfg.ServingScrape)
	}
	go runMetricsSampler(bgCtx, apiSvc.Metrics, servingScraper, 15*time.Second)
	// C2：周期自动归档（opt-in；手动 POST /api/storage/archive 始终可用）。
	if cfg.StorageArchiveEnabled {
		slog.Info("storage archiver enabled", "sweep", cfg.ArchiveSweep)
		go apiSvc.RunArchiveLoop(bgCtx, cfg.ArchiveSweep)
	}
	srv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           router,
		ReadHeaderTimeout: 10 * time.Second,
	}

	// 5) 后台监听 + 优雅关机：收到 SIGINT/SIGTERM 停止收新连接、放干在途请求。
	go func() {
		slog.Info("server listening", "addr", cfg.Addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Error("server failed", "err", err)
			os.Exit(1)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop
	slog.Info("shutdown signal received")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		slog.Error("graceful shutdown failed", "err", err)
	}
	if err := otelShutdown(shutdownCtx); err != nil {
		slog.Warn("otel shutdown (flush) failed", "err", err)
	}
	slog.Info("server stopped")
}

// runMetricsSampler 将指标采样与 GET /metrics/current 解耦，避免页面轮询制造历史数据。
// 空窗口不落库；没有请求时图表显示数据空档，而不是伪造 0 延迟。
func runMetricsSampler(ctx context.Context, svc *metrics.Service, scraper *serving.Scraper, interval time.Duration) {
	collect := func() {
		m, err := svc.Current(ctx)
		if err != nil {
			slog.Warn("metrics sample failed", "err", err)
			return
		}
		if scraper != nil {
			for key, value := range scraper.Snapshot() {
				m[key] = value
			}
		}
		requestCount, _ := m["request_count"].(int)
		if requestCount > 0 {
			svc.PersistSample(ctx, "sampler", m)
		}
	}
	collect()
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			collect()
		}
	}
}

// runRegistryReaper 周期清扫：把超 TTL 未心跳的服务实例置 unreachable，并 best-effort 落审计。
// 静态种子实例（last_heartbeat_at 为 NULL）不受影响。
func runRegistryReaper(ctx context.Context, st *store.Store, interval time.Duration, ttlSeconds float64) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			names, err := st.SweepStaleInstances(ctx, ttlSeconds)
			if err != nil {
				slog.Warn("registry reaper sweep failed", "err", err)
				continue
			}
			if len(names) > 0 {
				slog.Info("registry reaper: marked stale instances unreachable", "names", names)
				for _, n := range names {
					st.Audit(ctx, "system", "system", "service.stale", "service_instance", n,
						map[string]any{"reason": "heartbeat_ttl_expired"})
				}
			}
		}
	}
}

// runServingScraper 周期抓取 vLLM Prometheus 指标，聚合成真实 serving 指标（Option A）。
func runServingScraper(ctx context.Context, sc *serving.Scraper, interval time.Duration) {
	if err := sc.ScrapeOnce(ctx); err != nil {
		slog.Warn("serving scrape failed", "err", err)
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := sc.ScrapeOnce(ctx); err != nil {
				slog.Warn("serving scrape failed", "err", err)
			}
		}
	}
}
