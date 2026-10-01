// Phase 2 运维域类型：配置中心 / 发布流水线 / 审计 / 故障事件。
// 对齐 Java 控制面契约（/api/config、/api/deployments、/api/audit、/api/incidents）。

import type { SparkTone } from "./ui";

// 控制台行视图模型（由真实 API 派生：配置项 → ConfigConsoleRow、部署 → PipelineConsoleRow）。
export type ConfigConsoleRow = {
  id: string;
  name: string;
  key: string;
  namespace: string;
  env: string;
  type: string;
  version: string;
  updatedAt: string;
  owner: string;
  status: string;
  versionCount: number;
  trend: number[];
  source: ConfigItem;
};

export type PipelineConsoleRow = {
  id: string;
  name: string;
  service: string;
  env: string;
  version: string;
  commit: string;
  status: string;
  stage: string;
  stageMeta: string;
  trigger: string;
  startedAt: string;
  duration: string;
  tone: SparkTone | "neutral";
  source: Deployment;
};

export type ConfigItem = {
  id: string;
  env: string;
  namespace: string | null;
  config_key: string;
  config_type: string;
  active_version: number;
  status: string;
  created_by: string | null;
  updated_at: string | null;
  version_count: number;
};

export type ConfigVersion = {
  version: number;
  content: string | null;
  change_reason: string | null;
  operator: string | null;
  status: string;
  created_at: string;
};

export type ConfigVersionsResponse = {
  active_version: number;
  versions: ConfigVersion[];
};

export type DeploymentMeta = {
  image?: string;
  branch?: string;
  gate?: string;
  owner?: string;
  rollback_of?: string;
  // A1：真实 K8s rollout 的实时状态（runner 写入 deployments.metadata）。
  mode?: string; // "k8s_rollout" | undefined（记录态）
  target_namespace?: string;
  target_name?: string;
  previous_image?: string;
  phase?: string; // queued|patching|progressing|succeeded|rolling_back|rolled_back|failed
  progress?: number; // ready/desired 百分比
  ready?: number;
  desired?: number;
  updated?: number;
  message?: string;
  events?: Array<{ at: string; phase: string; message: string }>;
	// Qwen3.6 inference release evidence and runtime binding.
	model_id?: string;
	model_version_id?: string;
	endpoint_id?: string;
	release_profile?: string;
	release_target?: "aibrix" | "direct";
	runtime_request?: Record<string, unknown>;
	release_spec?: string;
	benchmark_run_id?: string;
	benchmark_report?: string;
	previous_release_id?: string;
};

export type Deployment = {
  id: string;
  name: string;
  version: string | null;
  env: string | null;
  status: string;
  started_at: string | null;
  finished_at: string | null;
  metadata: DeploymentMeta;
};

export type AuditEvent = {
  id: string;
  actor_id: string | null;
  actor_role: string | null;
  action: string;
  resource_type: string | null;
  resource_id: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
};

export type Incident = {
  id: string;
  title: string;
  severity: string;
  status: string;
  summary: string | null;
  created_at: string;
  resolved_at: string | null;
};
