// E3：模型路由 / A-B / 影子流量类型。

// 一条加权候选：权重份额的流量路由到 endpoint（service_instances.name）；model 可覆盖（同模型多版本灰度）。
export type RoutingVariant = {
  label: string;
  endpoint: string;
  model?: string;
  weight: number;
};

// 影子目标：镜像同一请求、丢弃响应、只采指标。
export type RoutingTarget = {
  label: string;
  endpoint: string;
  model?: string;
};

export type RoutingResourceTransition = {
  phase?: string;
  cleanup_complete?: boolean;
  stable_deployment?: string;
  at?: string;
  actions?: Array<{ operation: string; deployment: string; model: string; status: string; detail?: string }>;
};

export type RoutingPolicy = {
  name: string;
  description: string;
  enabled: boolean;
  variants: RoutingVariant[];
  shadow?: RoutingTarget | null;
  metadata?: {
    source?: string;
    rollout_strategy?: string;
    rollout_phase?: string;
    deployment_id?: string;
    model_version_id?: string;
    canary_weight?: number;
    release_target?: string;
    resource_transition?: RoutingResourceTransition;
    [key: string]: unknown;
  };
  created_by: string;
  created_at: string;
  updated_at: string;
  // 列表附带：最近 1h 主路各候选实时指标（份额随真实样本回归到配置权重）。
  live?: VariantStat[];
  endpoint_status?: Record<string, string>;
};

export type VariantStat = {
  label: string;
  endpoint: string;
  count: number;
  share: number; // 0..1
  avg_ms: number;
  p95_ms: number;
  error_rate: number; // 0..1
};

export type RoutingPolicyList = {
  policies: RoutingPolicy[];
  shadow_enabled: boolean;
};

export type RoutingSample = {
  request_id: string;
  variant_label: string;
  variant_endpoint: string;
  primary_status: number;
  primary_latency_ms: number;
  shadow_label: string;
  shadow_status: number | null;
  shadow_latency_ms: number | null;
  created_at: string;
};

export type RoutingPolicyDetail = {
  policy: RoutingPolicy;
  recent: RoutingSample[];
};

export type RoutingStats = {
  window: number;
  variants: VariantStat[];
  shadow: VariantStat[];
};

export type SavePolicyInput = {
  name: string;
  description?: string;
  enabled?: boolean;
  variants: RoutingVariant[];
  shadow?: RoutingTarget | null;
};
