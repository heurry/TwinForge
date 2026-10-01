import { Database, HardDrive, Layers3, RefreshCw, ScanSearch } from "lucide-react";

import { ErrorState, Skeleton } from "../common/FeedbackStates";
import { bytes } from "../../lib/format";
import { buildStorageOverview } from "../../lib/storageOverview";
import type { Storage } from "../../lib/useStorage";

const icons = { postgresql: Database, redis: Layers3, minio: HardDrive, pgvector: ScanSearch };

function statusLabel(status: string) {
  if (status === "ready") return "已接通";
  if (status === "degraded") return "已降级";
  if (status === "not_configured") return "未配置";
  if (status === "not_connected") return "未接入";
  return "未知";
}

function ScopeStatus({ status, detail }: { status: string; detail: string }) {
  return <div className="storage-scope-state" title={detail}>
    <span className={`storage-status status-${status}`}><i />{statusLabel(status)}</span>
    <small>{detail}</small>
  </div>;
}

export function StorageRuntimeOverview({ storage }: { storage: Storage }) {
  const { tiers, agent } = storage;
  if (tiers.isLoading && agent.isLoading) return <Skeleton rows={5} />;
  if (tiers.isError && agent.isError) {
    return <ErrorState error={tiers.error || agent.error} onRetry={() => { void tiers.refetch(); void agent.refetch(); }} />;
  }

  const rows = buildStorageOverview(tiers.data, agent.data?.data, bytes);
  const metrics = agent.data?.data.postgresql.metrics ?? {};
  const updatedAt = Math.max(tiers.dataUpdatedAt || 0, agent.dataUpdatedAt || 0);
  const platformTables = Array.isArray(tiers.data?.tiers)
    ? tiers.data.tiers.find((tier) => tier.kind === "relational")?.tables ?? []
    : [];

  return <>
    <section className="infra-panel storage-runtime-panel">
      <header className="storage-section-head">
        <span><strong>实时连接与职责矩阵</strong><small>控制面与 Agent 使用同一基础设施，但数据归属和降级策略相互独立</small></span>
        <em><RefreshCw size={12}/>{updatedAt ? `${new Date(updatedAt).toLocaleTimeString()} 更新` : "等待刷新"}</em>
      </header>
      {(tiers.isError || agent.isError) && <p className="storage-partial-warning">部分数据源暂不可用，页面保留另一作用域的真实状态，不使用模拟数据。</p>}
      <div className="storage-runtime-table-wrap">
        <table className="storage-runtime-table">
          <thead><tr><th>存储</th><th>平台控制面</th><th>Agent 运行时</th><th>实时数据</th><th>职责边界</th></tr></thead>
          <tbody>{rows.map((row) => {
            const Icon = icons[row.id];
            return <tr key={row.id}>
              <td><span className={`storage-backend kind-${row.id}`}><Icon size={16}/><b>{row.name}</b></span></td>
              <td><ScopeStatus {...row.platform}/></td>
              <td><ScopeStatus {...row.agent}/></td>
              <td><div className="storage-live-metrics">{row.metrics.map(item => <span key={item}>{item}</span>)}</div></td>
              <td><p>{row.boundary}</p></td>
            </tr>;
          })}</tbody>
        </table>
      </div>
    </section>

    <section className="infra-panel storage-facts-panel">
      <header className="storage-section-head"><span><strong>当前事实数据</strong><small>来自 PostgreSQL 实时聚合，不由前端推断</small></span></header>
      <dl className="storage-facts-grid">
        <div><dt>Session</dt><dd>{(metrics.sessions ?? 0).toLocaleString()}</dd></div>
        <div><dt>持久化消息</dt><dd>{(metrics.messages ?? 0).toLocaleString()}</dd></div>
        <div><dt>Run / 活跃</dt><dd>{(metrics.runs ?? 0).toLocaleString()} <small>/ {(metrics.active_runs ?? 0).toLocaleString()}</small></dd></div>
		<div><dt>Event Ledger</dt><dd>{(metrics.events ?? 0).toLocaleString()}</dd></div>
		<div><dt>Observation</dt><dd>{(metrics.observations ?? 0).toLocaleString()}</dd></div>
		<div><dt>Trace Score</dt><dd>{(metrics.scores ?? 0).toLocaleString()}</dd></div>
		<div><dt>原始观测归档</dt><dd>{(metrics.observation_archives ?? 0).toLocaleString()} <small>/ MinIO</small></dd></div>
        <div><dt>恢复状态</dt><dd>{(metrics.current_run_states ?? 0).toLocaleString()}</dd></div>
        <div><dt>待发布 Outbox</dt><dd>{(metrics.pending_outbox ?? 0).toLocaleString()}</dd></div>
		<div><dt>验证 Intent / Spec</dt><dd>{(metrics.verification_intents ?? 0).toLocaleString()} <small>/ {(metrics.verification_specs ?? 0).toLocaleString()}</small></dd></div>
		<div><dt>验证尝试 / 证据</dt><dd>{(metrics.verification_attempts ?? 0).toLocaleString()} <small>/ {(metrics.verification_evidence ?? 0).toLocaleString()}</small></dd></div>
		<div><dt>Agent Artifact</dt><dd>{(metrics.object_artifacts ?? 0).toLocaleString()} <small>/ {(metrics.artifacts ?? 0).toLocaleString()} 对象化</small></dd></div>
		<div><dt>内联待迁移</dt><dd>{(metrics.inline_artifacts ?? 0).toLocaleString()} <small>/ {bytes(metrics.inline_artifact_bytes ?? 0)}</small></dd></div>
		<div><dt>语义记忆</dt><dd>{(metrics.embedded_memories ?? 0).toLocaleString()} <small>/ {(metrics.memories ?? 0).toLocaleString()} 已向量化</small></dd></div>
		<div><dt>Embedding 待补偿</dt><dd>{(metrics.pending_embeddings ?? 0).toLocaleString()}</dd></div>
      </dl>
      <div className="storage-control-tables">
        <strong>控制面 PostgreSQL 关键表</strong>
        <div>{platformTables.map(table => <span key={table.table}><b>{table.table}</b><em>{table.rows.toLocaleString()} 行</em><small>{bytes(table.bytes)}</small></span>)}</div>
      </div>
    </section>
  </>;
}
