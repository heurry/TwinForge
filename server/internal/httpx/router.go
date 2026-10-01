// Package httpx 组装 chi 路由、中间件、统一错误信封与各 handler。
package httpx

import (
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/cors"
	"github.com/heurry/cloudnative-infra-platform/server/internal/obs"
)

// NewRouter 构建带中间件链与 /api 路由组的 chi 路由器。
// 6B：Python 单体（legacy-python）已退役——所有 /api 端点均 Go 原生（或经 ai-service），
// 不再有 legacy 反向代理。
func NewRouter(a *API) *chi.Mux {
	r := chi.NewRouter()

	r.Use(RequestID)
	r.Use(Telemetry) // D1：server span（继承上游 traceparent）+ Prometheus 指标；须在 RequestID 之后。
	r.Use(Logger)
	r.Use(Recoverer)
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins:   a.CORSOrigins,
		AllowedMethods:   []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"*"},
		AllowCredentials: true,
		MaxAge:           300,
	}))

	// D1：Prometheus 抓取端点（根路径，按惯例供 Prometheus scrape；Telemetry 中间件对它免计数）。
	r.Handle("/metrics", obs.MetricsHandler())

	r.Route("/api", func(r chi.Router) {
		// 5B.4a：Redis 横切中间件（各自在 Redis 降级时透明直通）。
		// 限流按 IP 令牌桶；幂等仅对带 Idempotency-Key 的 POST 生效。
		r.Use(a.RateLimit())
		r.Use(a.Idempotency())
		// D2：认证（解析 JWT→context）+ 授权（AuthEnabled 时按方法+角色门禁；关闭时透传）。
		r.Use(a.Authn)
		r.Use(a.Authz)

		r.Get("/health", Health)

		// D2：认证端点（始终放行，便于前端引导登录）。
		r.Post("/auth/login", a.login)
		r.Get("/auth/me", a.me)

		// Phase 1：只读 + 可观测（接口/外形复刻现有 Java 控制面）
		r.Get("/service-instances", a.serviceInstances)
		// 5B.4a：overview 走 cache-aside（短 TTL；服务实例写操作显式失效）。
		r.Get("/platform/overview", a.cached(cacheKeyOverview, a.CacheTTL, a.overview))

		// 5B.1：控制面 client-go 直读集群级 k8s（取代经 Agent 透传）；新增 events/nodes。
		r.Route("/kubernetes", func(r chi.Router) {
			r.Get("/pods", a.k8sNative(true, false, false, false, false))
			r.Get("/deployments", a.k8sNative(false, true, false, false, false))
			r.Get("/events", a.k8sNative(false, false, true, false, false))
			r.Get("/nodes", a.k8sNative(false, false, false, true, false))
			r.Get("/hpa", a.k8sNative(false, false, false, false, true)) // 5B.3
			// 5B.4a：整集群快照走 cache-aside（命中真实 minikube，开销大；10s TTL）。
			r.Get("/snapshot", a.cached(cacheKeyK8sSnapshot, 10*time.Second, a.k8sNative(true, true, true, true, true)))
			// A2：弹性扩缩容真配（feature flag + 命名空间守卫；默认不碰 serving 命名空间）。
			r.Post("/deployments/{name}/scale", a.scaleDeployment)
			r.Put("/hpa", a.upsertHPA)
			r.Delete("/hpa/{name}", a.deleteHPA)
		})

		r.Route("/metrics", func(r chi.Router) {
			r.Get("/current", a.metricsCurrent)
			r.Get("/history", a.metricsHistory)
			r.Get("/requests", a.metricsRequests)
		})
		// Unified platform log index. GET supports DB search and direct Kubernetes
		// pod tailing; POST is the ingestion entry used by platform components.
		r.Get("/logs", a.platformLogs)
		r.Post("/logs", a.ingestPlatformLog)

		// 5A.3：基于当前指标快照的规则化告警评估（取代前端告警表 mock）。
		r.Get("/alerts", a.alerts)
		r.Get("/ai/status", a.aiStatus)

		// C3：真实服务拓扑——OTel span 进程内派生的调用图（连线粗细 = 真实 QPS）。
		r.Get("/topology/graph", a.topologyGraph)

		// Phase 2：写操作 + 平台治理（外形复刻现有 Java 控制面）
		r.Post("/service-instances/{name}/healthcheck", a.serviceInstanceHealthcheck)
		// 单机双卡推理实验 workload：服务启停与压测任务启停相互独立。
		r.Get("/inference/runtime", a.inferenceRuntimeStatus)
		r.Post("/inference/runtime", a.startInferenceRuntime)
		r.Delete("/inference/runtime", a.stopInferenceRuntime)
		r.Get("/inference/releases", a.inferenceReleaseCandidates)
		r.Post("/inference/releases", a.submitInferenceRelease)
		r.Delete("/inference/releases/{id}", a.stopInferenceRelease)
		r.Delete("/inference/serving-models/{model_id}", a.stopInferenceServingModel)
		// Phase 5 / 5B.2：服务注册表（注册 / 心跳 / 注销）；后台 reaper 据 TTL 清扫。
		r.Post("/service-instances/register", a.registerServiceInstance)
		r.Post("/service-instances/{name}/heartbeat", a.heartbeatServiceInstance)
		r.Delete("/service-instances/{name}", a.deregisterServiceInstance)

		r.Route("/config", func(r chi.Router) {
			r.Get("/items", a.configItems)
			r.Post("/items", a.createConfigItem)
			r.Get("/items/{id}/versions", a.configVersions)
			r.Post("/items/{id}/versions", a.publishConfigVersion)
			r.Post("/items/{id}/rollback", a.rollbackConfigVersion)
			r.Post("/items/{id}/launch", a.launchConfigVersion)
		})

		// deployments/incidents 用无尾斜杠形式（对齐 Java @RequestMapping("/api/deployments")）。
		r.Get("/deployments", a.deployments)
		r.Post("/deployments", a.triggerDeployment)
		r.Post("/deployments/{id}/finish", a.finishDeployment)
		r.Post("/deployments/{id}/rollback", a.rollbackDeployment)
		r.Get("/ci/runs", a.ciRuns)
		r.Post("/ci/runs", a.triggerCIRun)

		r.Get("/incidents", a.incidents)
		r.Post("/incidents", a.createIncident)
		r.Get("/incidents/{id}", a.incidentDetail)
		r.Post("/incidents/{id}/ack", a.ackIncident)
		r.Post("/incidents/{id}/resolve", a.resolveIncident)

		r.Get("/audit/events", a.auditEvents)

		// C2：分层存储生命周期（各层占用 + 归档清单 + 手动归档）。
		r.Get("/storage/tiers", a.storageTiers)
		r.Get("/storage/archives", a.listArchives)
		r.Get("/storage/archives/{id}", a.getArchive)
		r.Post("/storage/archive", a.runArchive)

		// 6A：models / proxy / benchmarks 组 Go 原生（取代 legacy 反代）。
		r.Get("/models", a.models)
		// C1：独立模型注册中心（版本化 + 血缘 + 产物 + 运行时绑定）。
		r.Get("/models/registry", a.listModelRegistry)
		r.Post("/models/registry", a.registerModelVersion)
		r.Get("/models/registry/{id}", a.getModelVersion)
		r.Patch("/models/registry/{id}/status", a.updateModelStatus)
		r.Delete("/models/registry/{id}", a.deleteModelVersion)
		r.Post("/models/registry/{id}/artifact", a.uploadModelArtifact)
		r.Get("/models/registry/{id}/artifact", a.modelArtifactURL)
		r.Post("/proxy/{endpoint_id}/v1/chat/completions", a.proxyChatCompletions)

		// E3：模型路由 / A-B / 影子流量（策略 CRUD + 灰度全量/回滚 + A/B 对比 + 路由数据面）。
		r.Route("/routing", func(r chi.Router) {
			r.Get("/policies", a.listRoutingPolicies)
			r.Post("/policies", a.createRoutingPolicy)
			r.Get("/policies/{name}", a.getRoutingPolicy)
			r.Patch("/policies/{name}", a.updateRoutingPolicy)
			r.Delete("/policies/{name}", a.deleteRoutingPolicy)
			r.Get("/policies/{name}/stats", a.routingPolicyStats)
			r.Post("/policies/{name}/promote", a.promoteRoutingVariant)
			r.Post("/policies/{name}/rollback", a.rollbackRoutingPolicy)
			// 数据面：按策略加权选主路 + 可选影子镜像。
			r.Post("/{policy}/v1/chat/completions", a.routedChatCompletions)
		})
		r.Post("/benchmarks/serving", a.createServingBenchmark)
		r.Get("/benchmarks/{run_id}", a.benchmarkRun)
		r.Delete("/benchmarks/{run_id}", a.cancelServingBenchmark)
		r.Get("/benchmarks/{run_id}/events", a.benchmarkEvents)

		// 6A：evals 组 Go 原生（检索召回评测；依赖 knowledge 检索）。
		r.Post("/evals/customer-support", a.createCustomerSupportEval)
		r.Get("/evals/{run_id}", a.evalRun)

		// E2：RAG 评测体系 + 在线反馈回流（反馈 → 评测数据集 / 重排信号 → 离线基线）。
		r.Route("/rag", func(r chi.Router) {
			r.Get("/dataset", a.ragDataset)
			r.Get("/signal", a.ragSignal)
			r.Post("/eval", a.runRagEval)
			r.Get("/eval/history", a.ragEvalHistory)
		})

		// 6A：chat 组 Go 原生（客服 RAG 对话；会话/消息→PG，messages:stream 流式 RAG）。
		r.Route("/chat", func(r chi.Router) {
			r.Post("/sessions", a.createChatSession)
			r.Get("/sessions", a.listChatSessions)
			r.Get("/sessions/{session_id}", a.getChatSession)
			r.Post("/sessions/{session_id}/messages:stream", a.streamChatMessage)
			r.Post("/messages/{message_id}/feedback", a.createMessageFeedback)
		})

		// 6A：knowledge 组 Go 原生（pgvector RAG；语料=基准测试日志，方案 A）。
		r.Route("/knowledge", func(r chi.Router) {
			r.Get("/versions", a.knowledgeVersions)
			r.Post("/versions", a.createKnowledgeVersion)
			r.Get("/documents", a.knowledgeDocuments)
			r.Post("/documents", a.createKnowledgeDocument)
			r.Post("/rebuild-index", a.rebuildKnowledgeIndex)
			r.Get("/search", a.knowledgeSearch)
		})

		// Phase 3：AI 边界（Go 取证 → Python 推理 → Go 落库+审计）。
		r.Route("/ai", func(r chi.Router) {
			r.Post("/diagnose", a.diagnose)
			r.Get("/inference/evidence", a.inferenceDiagnosisEvidence)
			r.Get("/training/evidence", a.trainingDiagnosisEvidence)
			// E1：agentic 诊断（多轮工具取证 + 推理轨迹）。
			r.Post("/diagnose:agent", a.diagnoseAgent)
			r.Get("/diagnoses", a.listDiagnoses)
			r.Get("/diagnoses/{id}", a.getDiagnosis)
			// SSE 流式问答经 Go 透传到 AI 服务，让 Copilot 走单一入口。
			if a.AIProxy != nil {
				r.Post("/chat:stream", a.AIProxy.To("/internal/chat:stream"))
			}
		})

		// Phase F：分布式训练（Kubeflow PyTorchJob 生命周期；写受 ALLOW_TRAINING + 命名空间守卫约束）。
		r.Route("/training", func(r chi.Router) {
			r.Get("/jobs", a.listTrainingJobs)
			r.Post("/jobs", a.submitTrainingJob)
			r.Get("/jobs/{id}", a.getTrainingJob)
			r.Delete("/jobs/{id}", a.cancelTrainingJob)
			r.Get("/jobs/{id}/kubernetes", a.trainingJobKubernetes)
			r.Get("/jobs/{id}/logs", a.trainingJobLogs)
		})

	})

	// 404 / 405 也走统一错误信封（对齐前端 ApiError 解析）。
	r.NotFound(func(w http.ResponseWriter, req *http.Request) {
		WriteError(w, req, http.StatusNotFound, "not_found", "resource not found")
	})
	r.MethodNotAllowed(func(w http.ResponseWriter, req *http.Request) {
		WriteError(w, req, http.StatusMethodNotAllowed, "method_not_allowed", "method not allowed")
	})

	return r
}
