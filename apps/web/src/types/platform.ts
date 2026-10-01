import type { SparkTone } from "./ui";

// 控制台行视图模型（由真实 API 派生；曾散落在 data/platformSnapshots）。
export type ServiceConsoleRow = {
  id: string;
  name: string;
  runtime: string;
  model: string;
  // 可用性和性能是两套状态：健康检查通过不代表延迟一定满足 SLO。
  status: "正常" | "异常" | "未知";
  performanceStatus: "正常" | "降级" | "异常" | "无流量" | "未采集";
  roleLabel: string;
  metricSource?: string;
  replicas: string;
  qps: number;
  p95: number;
  errorRate: number;
  cpu: number;
  gpu: number;
  tone: SparkTone;
  trend: number[];
};

export type KubernetesWorkloadRow = {
  id: string;
  name: string;
  kind: string;
  namespace: string;
  replicas: string;
  desired: number;
  availability: number;
  status: "正常" | "异常" | "警告";
};

export type BenchmarkTaskRow = {
  id: string;
  name: string;
  service: string;
  version: string;
  scenario: string;
  context: string;
  concurrency: string;
  p95: string;
  ttft: string;
  tpot: string;
  throughput: string;
  successRate: string;
  qualityRate: string;
  errorRate: string;
  status: string;
  startedAt: string;
  tone: SparkTone;
};

export type ModelRegistryRow = {
  id: string;
  name: string;
  description: string;
  type: string;
  version: string;
  status: string;
  env: string;
  quality: number | null;
  instances: number;
  updatedAt: string;
  source?: ServiceInstance;
};

export type ServiceInstance = {
  name: string;
  base_url: string;
  model_id: string;
  kind: string;
  gpu_id: string;
  routing_role: string;
  status: string;
  // 5B.2 服务注册表：最近心跳时间（动态注册实例有值；静态种子为 null）。
  last_heartbeat_at?: string | null;
  last_checked_at?: string | null;
  // 路由 / 网关类实例的策略元数据（后端 service_instances.metadata_json）。
  metadata?: {
    routing_strategies?: string[] | string;
    target_instances?: string[];
    candidate_endpoints?: string[];
    [key: string]: unknown;
  };
};

export type Metrics = {
	benchmark_run_id?: string;
	source?: string;
  window?: string;
  window_seconds?: number;
  qps: number;
  request_count: number;
  error_count: number;
  error_rate: number;
  mean_latency_ms: number | null;
  mean_ttft_ms: number | null;
  p50_latency_ms: number | null;
  p95_latency_ms: number | null;
  p99_latency_ms: number | null;
  p50_ttft_ms: number | null;
  p95_ttft_ms: number | null;
  p99_ttft_ms: number | null;
  p95_generation_ms: number | null;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  tokens_per_second: number;
  target_pod_counts: Record<string, number>;
  endpoint_counts: Record<string, number>;
  status_counts: Record<string, number>;
  fallback_counts: Record<string, number>;
  target_pod_stats: GroupStats[];
  endpoint_stats: GroupStats[];
  gpu: Array<{
    index: number;
    name: string;
    memory_used_mb: number;
    memory_total_mb: number;
    gpu_utilization_percent: number;
    memory_utilization_percent?: number;
    temperature_celsius?: number | null;
    power_watts?: number | null;
  }>;
  gpu_status?: {
    available: boolean;
    source?: string;
    gpu_count?: number;
    error?: string;
  };
  host: HostMetrics;
  cadvisor: CadvisorMetrics;
  service_instances: Array<{
    name: string;
    kind: string;
    base_url: string;
    model_id: string;
    routing_role: string;
    status: string;
    last_checked_at: string | null;
  }>;
  upstream_metrics: Array<{
    name: string;
    kind: string;
    model_id: string;
    metrics_url: string;
    reachable: boolean;
    selected: Record<string, number>;
    error: string;
  }>;
  kubernetes: {
    available: boolean;
    disabled?: boolean;
    error?: string;
    pods: Array<{
      namespace: string;
      name: string;
      phase: string;
      ready: string;
      restarts: number;
      pod_ip: string;
      node: string;
      component: string;
      containers?: string[];
    }>;
  };
};

export type HostMetrics = {
  cpu: {
    usage_percent: number | null;
    count: number;
    load_average: number[];
  };
  memory: {
    total_bytes: number;
    available_bytes: number;
    used_bytes: number;
    used_percent: number | null;
    swap_total_bytes: number;
    swap_free_bytes: number;
  };
  disk: Array<{
    path: string;
    total_bytes: number;
    used_bytes: number;
    free_bytes: number;
    used_percent: number | null;
  }>;
  network: {
    rx_bytes_per_second: number;
    tx_bytes_per_second: number;
    interfaces: Array<{
      name: string;
      rx_bytes_per_second: number;
      tx_bytes_per_second: number;
      rx_bytes: number;
      tx_bytes: number;
    }>;
  };
};

export type CadvisorMetrics = {
  configured_url: string;
  available: boolean;
  error: string;
  summary: {
    container_count: number;
    cpu_cores: number;
    memory_working_set_bytes: number;
    network_rx_bytes_per_second: number;
    network_tx_bytes_per_second: number;
  };
  containers: Array<{
    namespace: string;
    pod: string;
    container: string;
    name: string;
    image: string;
    cpu_cores?: number;
    memory_working_set_bytes?: number;
    fs_usage_bytes?: number;
    network_rx_bytes_per_second?: number;
    network_tx_bytes_per_second?: number;
  }>;
};

export type RequestTrace = {
  request_id: string;
  session_id: string;
  retrieval_ms: number | null;
  generation_ms: number | null;
  total_ms: number | null;
  ttft_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  target_pod: string;
  endpoint_id: string;
  status: string;
  fallback_reason: string | null;
  error: string | null;
  created_at: string;
};

export type MetricsHistorySample = {
  source: string;
  created_at: string;
  metrics: Partial<Metrics>;
};

// Go Agent 实集群快照（/api/kubernetes/snapshot）。
export type K8sPod = {
  namespace: string;
  name: string;
  phase: string;
  ready: string;
  restarts: number;
  pod_ip: string;
  node: string;
  component: string;
  containers?: string[];
};

export type K8sDeployment = {
  namespace: string;
  name: string;
  ready: string;
  available: number;
  desired: number;
  image: string;
};

export type K8sEvent = {
  namespace: string;
  resource_kind: string;
  resource_name: string;
  reason: string;
  message: string;
  type: string;
  event_time: string;
};

export type K8sNode = {
  name: string;
  status: string;
  roles: string;
  version: string;
  internal_ip: string;
  os_image: string;
  container_runtime: string;
  cpu_capacity: string;
  memory_bytes: number;
  gpu_capacity: string;
};

export type K8sHPA = {
  namespace: string;
  name: string;
  target_kind: string;
  target_name: string;
  min_replicas: number;
  max_replicas: number;
  current_replicas: number;
  desired_replicas: number;
  metric: string;
  scaling_active: boolean;
};

export type KubernetesSnapshot = {
  available: boolean;
  disabled?: boolean;
  error?: string;
  pods: K8sPod[];
  deployments: K8sDeployment[];
  events: K8sEvent[];
  nodes: K8sNode[];
  hpas: K8sHPA[];
  // A2：弹性扩缩容真配——后端据 feature flag 暴露写能力与可写命名空间名单。
  writes_enabled?: boolean;
  write_namespaces?: string[];
};

// A2：K8s 写操作入参（手动扩缩 Deployment / 配置 HPA）。
export type ScaleDeploymentInput = { namespace: string; name: string; replicas: number };
export type UpsertHpaInput = {
  namespace: string;
  name?: string;
  target_name: string;
  min_replicas: number;
  max_replicas: number;
  target_cpu_util: number;
};

export type BenchmarkEvent = {
  event_type: string;
  created_at: string;
  payload: Record<string, unknown>;
};

export type GroupStats = {
  name: string;
  request_count: number;
  error_count: number;
  error_rate: number;
  mean_latency_ms: number | null;
  p95_latency_ms: number | null;
  mean_ttft_ms: number | null;
  p95_ttft_ms: number | null;
  output_tokens: number;
};
