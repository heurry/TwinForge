// Package config 从环境变量加载控制面运行配置（默认值对齐本地 dev 的 PostgreSQL）。
package config

import (
	"os"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	Addr                 string        // 监听地址，默认 :8081（迁移期与 Java :8080 并存）
	DatabaseURL          string        // pgxpool DSN：postgres://...
	CORSOrigins          []string      // 允许的前端来源
	AgentBaseURL         string        // Go Node Agent 地址（Phase 1 起用）
	AIServiceBaseURL     string        // Python AI 服务（Phase 3：diagnose + chat:stream + embed）
	AIRequestTimeout     time.Duration // 调 AI 诊断的超时（含 LLM 推理，默认 90s）
	AIBrixGatewayBaseURL string        // 应用侧稳定 AIBrix NodePort；不使用临时 port-forward
	KubeconfigPath       string        // Phase 5/5B.1：控制面 client-go 的 kubeconfig（空=in-cluster 或默认规则）
	// A2：K8s 写权限（弹性扩缩容真配）。默认关闭——读永远可用、写需显式开启。
	AllowK8sWrites     bool     // 总开关；false 时所有 K8s 写接口返回 403
	K8sWriteNamespaces []string // 可写命名空间允许名单（即便开启写也只放行名单内；默认 default）
	// Phase F：Kubeflow 分布式训练（PyTorchJob）。默认关——训练写需显式开启 + 命名空间允许名单；
	// serving 命名空间经 guardNamespace 硬禁，训练碰不到。
	AllowTraining          bool
	TrainingNamespaces     []string
	TrainingHostPath       string
	TrainingMountPath      string
	TrainingArtifactSecret string
	TrainingOutputRoot     string
	RegistryTTL            time.Duration // Phase 5/5B.2：心跳 TTL（超时未心跳→unreachable）
	RegistrySweep          time.Duration // Phase 5/5B.2：reaper 清扫周期
	ServingScrape          time.Duration // Phase 5/Option A：抓取 vLLM Prometheus 指标的周期
	VLLMMetricsPort        string        // vLLM /metrics 端口（默认 8000）
	CadvisorURL            string        // cAdvisor Prometheus metrics URL
	RedisURL               string        // 5B.4a：Redis DSN（空=禁用，全降级）
	CacheTTL               time.Duration // 5B.4a：cache-aside 默认 TTL
	IdempotencyTTL         time.Duration // 5B.4a：幂等键 TTL
	RateLimitRPS           float64       // 5B.4a：令牌桶速率（令牌/秒/IP；<=0 关闭）
	RateLimitBurst         int           // 5B.4a：令牌桶容量
	S3Endpoint             string        // 5B.4b：对象存储 endpoint（host:port，空=禁用）
	S3AccessKey            string        // 5B.4b：S3 access key
	S3SecretKey            string        // 5B.4b：S3 secret key
	S3Bucket               string        // 5B.4b：bucket（默认 infra-artifacts）
	S3UseSSL               bool          // 5B.4b：是否 TLS
	S3PresignTTL           time.Duration // 5B.4b：预签名 URL 有效期
	// D1：全链路可观测（OpenTelemetry traces + Prometheus metrics）。
	OTLPEndpoint string // OTLP/HTTP collector host:port（空=不导出 trace，优雅降级；/metrics 始终开）
	OTLPInsecure bool   // OTLP 走明文（本地 collector 通常 true）
	OTelService  string // trace 服务名（service.name 资源属性）
	// C2：分层存储生命周期（PG 过期数据 → MinIO 归档）。
	StorageArchiveEnabled bool          // 是否开启「周期自动归档」（默认关；手动触发端点始终可用）
	ArchiveSweep          time.Duration // 自动归档扫描周期
	// D2：认证授权 / RBAC（默认关——现有开放演示不受影响；开启后读需登录态、写需 operator+）。
	AuthEnabled   bool          // 总开关
	AuthJWTSecret string        // HS256 签名密钥
	AuthTokenTTL  time.Duration // 令牌有效期
	AuthUsers     string        // "user:pass:role,..."（空=默认 admin/operator/viewer 演示账户）
	// E3：模型路由 / A-B / 影子流量。影子镜像会对 serving 栈产生额外负载，默认关——
	// 策略 CRUD 与加权 A/B 路由始终可用，仅「镜像到影子目标」受此开关门禁。
	RoutingShadowEnabled bool
	// E2：在线反馈回流。默认关时检索行为不变（仅采集反馈、回流评测数据集）；
	// 开启后 chat 检索按反馈净分对候选做温和重排（命中过的好文档轻微上浮）。
	RAGRerankFeedback bool
	CIProvider        string // gitlab (default) | github_actions
	GitLabBaseURL     string // https://gitlab.com or self-managed instance root
	GitLabProjectID   string // numeric project id or URL-encoded path source (group/project)
	GitLabToken       string // personal/project access token with api permission
	GitLabRef         string // default branch used by console-triggered pipelines
	GitHubRepository  string // owner/repo; empty disables console-triggered CI
	GitHubToken       string // fine-grained token with Actions write permission
	GitHubWorkflow    string // workflow file name or workflow id
}

func Load() Config {
	return Config{
		Addr:                   env("SERVER_ADDR", ":8081"),
		DatabaseURL:            env("DATABASE_URL", "postgres://infra:infra@localhost:5432/infra_platform?sslmode=disable"),
		CORSOrigins:            splitCSV(env("CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")),
		AgentBaseURL:           env("AGENT_BASE_URL", "http://127.0.0.1:8090"),
		AIServiceBaseURL:       env("AI_SERVICE_BASE_URL", "http://127.0.0.1:8200"),
		AIRequestTimeout:       envSeconds("AI_REQUEST_TIMEOUT_SECONDS", 90*time.Second),
		AIBrixGatewayBaseURL:   env("AIBRIX_GATEWAY_BASE_URL", "http://minikube:30080/v1"),
		KubeconfigPath:         env("KUBECONFIG", ""),
		AllowK8sWrites:         envBool("ALLOW_K8S_WRITES", false),
		K8sWriteNamespaces:     splitCSV(env("K8S_WRITE_NAMESPACES", "default")),
		AllowTraining:          envBool("ALLOW_TRAINING", false),
		TrainingNamespaces:     splitCSV(env("TRAINING_NAMESPACES", "training")),
		TrainingHostPath:       env("TRAINING_HOST_PATH", ""),
		TrainingMountPath:      env("TRAINING_MOUNT_PATH", ""),
		TrainingArtifactSecret: env("TRAINING_ARTIFACT_SECRET", ""),
		TrainingOutputRoot:     env("TRAINING_OUTPUT_ROOT", "/tmp/twinforge-training"),
		RegistryTTL:            envSeconds("REGISTRY_TTL_SECONDS", 30*time.Second),
		RegistrySweep:          envSeconds("REGISTRY_SWEEP_SECONDS", 15*time.Second),
		ServingScrape:          envSeconds("SERVING_SCRAPE_SECONDS", 10*time.Second),
		VLLMMetricsPort:        env("VLLM_METRICS_PORT", "8000"),
		CadvisorURL:            env("CADVISOR_URL", "http://127.0.0.1:18080/metrics"),
		RedisURL:               env("REDIS_URL", ""),
		CacheTTL:               envSeconds("CACHE_TTL_SECONDS", 5*time.Second),
		IdempotencyTTL:         envSeconds("IDEMPOTENCY_TTL_SECONDS", 600*time.Second),
		RateLimitRPS:           envFloat("RATE_LIMIT_RPS", 50),
		RateLimitBurst:         envInt("RATE_LIMIT_BURST", 100),
		S3Endpoint:             env("S3_ENDPOINT", ""),
		S3AccessKey:            env("S3_ACCESS_KEY", "minioadmin"),
		S3SecretKey:            env("S3_SECRET_KEY", "minioadmin"),
		S3Bucket:               env("S3_BUCKET", "infra-artifacts"),
		S3UseSSL:               envBool("S3_USE_SSL", false),
		S3PresignTTL:           envSeconds("S3_PRESIGN_TTL_SECONDS", 15*time.Minute),
		OTLPEndpoint:           env("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
		OTLPInsecure:           envBool("OTEL_EXPORTER_OTLP_INSECURE", true),
		OTelService:            env("OTEL_SERVICE_NAME", "go-control-plane"),
		StorageArchiveEnabled:  envBool("STORAGE_ARCHIVE_ENABLED", false),
		ArchiveSweep:           envSeconds("ARCHIVE_SWEEP_SECONDS", time.Hour),
		AuthEnabled:            envBool("AUTH_ENABLED", false),
		AuthJWTSecret:          env("AUTH_JWT_SECRET", "dev-insecure-change-me"),
		AuthTokenTTL:           envSeconds("AUTH_TOKEN_TTL_SECONDS", 12*time.Hour),
		AuthUsers:              env("AUTH_USERS", ""),
		RoutingShadowEnabled:   envBool("ROUTING_SHADOW_ENABLED", false),
		RAGRerankFeedback:      envBool("RAG_RERANK_FEEDBACK", false),
		CIProvider:             env("CI_PROVIDER", "gitlab"),
		GitLabBaseURL:          env("GITLAB_BASE_URL", "https://gitlab.com"),
		GitLabProjectID:        env("GITLAB_PROJECT_ID", ""),
		GitLabToken:            env("GITLAB_TOKEN", ""),
		GitLabRef:              env("GITLAB_REF", "main"),
		GitHubRepository:       env("GITHUB_REPOSITORY", ""),
		GitHubToken:            env("GITHUB_TOKEN", ""),
		GitHubWorkflow:         env("GITHUB_WORKFLOW_FILE", "ci.yml"),
	}
}

// envInt 读取整数 env，缺省/非法时取 def。
func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

// envFloat 读取浮点 env，缺省/非法时取 def。
func envFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

// envBool 读取布尔 env，缺省/非法时取 def。
func envBool(key string, def bool) bool {
	if v := os.Getenv(key); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
	}
	return def
}

// envSeconds 读取以秒为单位的时长 env，缺省/非法时取 def。
func envSeconds(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return time.Duration(n) * time.Second
		}
	}
	return def
}

// MigrateURL 返回 golang-migrate 用的 URL（pgx5 scheme，由 database/pgx/v5 驱动注册）。
func (c Config) MigrateURL() string {
	if rest, ok := strings.CutPrefix(c.DatabaseURL, "postgres://"); ok {
		return "pgx5://" + rest
	}
	if rest, ok := strings.CutPrefix(c.DatabaseURL, "postgresql://"); ok {
		return "pgx5://" + rest
	}
	return c.DatabaseURL
}

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func splitCSV(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}
