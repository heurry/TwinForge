import type { AgentStorageBackend, AgentStorageSummary } from "../types/agent";
import type { StorageTier, StorageTiers } from "../types/storage";

export type StorageScopeState = {
  status: string;
  detail: string;
};

export type StorageOverviewRow = {
  id: "postgresql" | "redis" | "minio" | "pgvector";
  name: string;
  platform: StorageScopeState;
  agent: StorageScopeState;
  metrics: string[];
  boundary: string;
};

const absent: StorageScopeState = { status: "not_connected", detail: "此作用域未接入" };

function platformState(tier: StorageTier | undefined): StorageScopeState {
  if (!tier) return { status: "unknown", detail: "控制面未返回该存储层" };
  return {
    status: tier.enabled ? "ready" : "not_connected",
    detail: tier.detail || (tier.enabled ? "已连接" : "未启用")
  };
}

function agentState(backend: AgentStorageBackend | undefined): StorageScopeState {
  if (!backend) return { status: "unknown", detail: "Agent API 未返回该存储层" };
  return { status: backend.status, detail: backend.detail || backend.role };
}

function metric(metrics: Record<string, number> | undefined, key: string): number {
  const value = metrics?.[key];
  return Number.isFinite(value) ? Number(value) : 0;
}

export function buildStorageOverview(
  platform: StorageTiers | undefined,
  agent: AgentStorageSummary | undefined,
  formatBytes: (value: number) => string
): StorageOverviewRow[] {
  const tiers = Array.isArray(platform?.tiers) ? platform.tiers : [];
  const pg = tiers.find((tier) => tier.kind === "relational" || tier.id === "relational");
  const redis = tiers.find((tier) => tier.kind === "cache" || tier.id === "hot");
  const minio = tiers.find((tier) => tier.kind === "object" || tier.id === "object");
  const pgMetrics = agent?.postgresql?.metrics;
	const artifactMetrics = agent?.minio?.metrics;
	const memoryMetrics = agent?.pgvector?.metrics;

  return [
    {
      id: "postgresql",
      name: "PostgreSQL",
      platform: platformState(pg),
      agent: agentState(agent?.postgresql),
      metrics: [
        `控制面 ${formatBytes(pg?.total_bytes ?? 0)}`,
        `Agent ${metric(pgMetrics, "sessions")} Session / ${metric(pgMetrics, "messages")} 消息`,
        `${metric(pgMetrics, "runs")} Run / ${metric(pgMetrics, "events")} Event`,
        `${metric(pgMetrics, "observations")} Observation / ${metric(pgMetrics, "scores")} Score`,
        `${metric(pgMetrics, "current_run_states")} 恢复状态`,
        `${metric(pgMetrics, "verification_intents")} Verification Intent / ${metric(pgMetrics, "verification_evidence")} Evidence`
      ],
      boundary: "唯一事实、事务、状态机与故障恢复"
    },
    {
      id: "redis",
      name: "Redis",
      platform: platformState(redis),
      agent: agentState(agent?.redis),
      metrics: [`${metric(pgMetrics, "pending_outbox")} 条 Agent Outbox 待发布`],
      boundary: "实时通知与异步摄取指针；事件正文不落 Redis"
    },
    {
      id: "minio",
      name: "MinIO",
      platform: platformState(minio),
      agent: agentState(agent?.minio),
      metrics: [
        `${minio?.manifests ?? 0} 个控制面归档批次`,
        `${formatBytes(minio?.archived_bytes ?? 0)} 已归档`,
		`${metric(artifactMetrics, "object_artifacts")} 个 Agent 对象产物`,
		`${formatBytes(metric(artifactMetrics, "artifact_bytes"))} 产物内容`,
		`${metric(artifactMetrics, "inline_artifacts")} 个遗留内联待迁移`,
		`${metric(artifactMetrics, "observation_archives")} 条原始 Observation 归档`
      ],
	  boundary: "控制面冷归档；Agent Artifact 内容寻址对象与 PG 元数据分离"
    },
    {
      id: "pgvector",
      name: "pgvector",
      platform: absent,
      agent: agentState(agent?.pgvector),
	  metrics: [
		`${metric(memoryMetrics, "embedded_memories")} 条已向量化 Memory`,
		`${metric(memoryMetrics, "pending_embeddings")} 条待补偿`
	  ],
	  boundary: "分层作用域过滤后进行向量、词法、重要度与时效性混合召回"
    }
  ];
}
