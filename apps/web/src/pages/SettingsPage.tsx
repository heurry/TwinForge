import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { toast } from "sonner";
import { RefreshCw } from "lucide-react";

import { PageHeader, PanelHeader, StatusBadge } from "../components/common/PlatformPrimitives";
import { api } from "../lib/api";
import type { Metrics } from "../types/platform";

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) || "same-origin（dev 代理）";

function statusOf(query: Pick<UseQueryResult, "isSuccess" | "isError" | "isLoading">): string {
  if (query.isSuccess) return "healthy";
  if (query.isError) return "unreachable";
  if (query.isLoading) return "checking";
  return "degraded";
}

// 设置页只展示真实可探测的连接 / 环境信息：不再有无后端的可编辑配置表单（mock）。
export function SettingsPage() {
  const healthQuery = useQuery({
    queryKey: ["health", "settings"],
    queryFn: () => api<Record<string, unknown>>("/api/health"),
    refetchInterval: 15000,
    retry: false,
  });
  const metricsQuery = useQuery({
    queryKey: ["metrics", "settings"],
    queryFn: () => api<Metrics>("/api/metrics/current"),
    refetchInterval: 15000,
    retry: false,
  });
  const aiQuery = useQuery({
    queryKey: ["ai", "settings"],
    queryFn: () => api<{ diagnoses?: unknown[] }>("/api/ai/diagnoses?limit=1"),
    refetchInterval: 15000,
    retry: false,
  });
  const knowledgeQuery = useQuery({
    queryKey: ["knowledge", "settings"],
    queryFn: () => api<{ documents?: unknown[] }>("/api/knowledge/documents?limit=1"),
    refetchInterval: 15000,
    retry: false,
  });

  const tiers = [
    { label: "Go 控制面 API", endpoint: ":8081 · /api/health", status: statusOf(healthQuery) },
    { label: "Node Agent / 主机指标", endpoint: ":8090 · /api/metrics/current", status: metricsQuery.isSuccess && metricsQuery.data?.host ? "healthy" : statusOf(metricsQuery) },
    { label: "AI 服务", endpoint: ":8200 · /api/ai/*", status: statusOf(aiQuery) },
    { label: "知识库 / RAG（pgvector）", endpoint: "/api/knowledge", status: statusOf(knowledgeQuery) },
  ];

  const refetchAll = () => {
    void healthQuery.refetch();
    void metricsQuery.refetch();
    void aiQuery.refetch();
    void knowledgeQuery.refetch();
    toast.success("已重新探测服务连接");
  };

  const controlPlaneStatus = healthQuery.data
    ? String((healthQuery.data as { status?: unknown }).status ?? "ok")
    : healthQuery.isError ? "unreachable" : healthQuery.isLoading ? "checking" : "—";

  return (
    <section className="infra-page settings-replica">
      <PageHeader
        title="平台状态"
        subtitle="只读展示服务连接、健康探测与运行环境；可编辑策略统一在对应功能页面管理"
        actions={
          <button className="console-refresh" onClick={refetchAll} type="button">
            <RefreshCw className={healthQuery.isFetching ? "spinning" : undefined} size={14} /> 重新探测
          </button>
        }
      />

      <section className="infra-panel">
        <PanelHeader title="服务连接" action="实时探测 · 15s" />
        <table className="infra-table">
          <thead><tr>{["服务", "端点", "状态"].map((c) => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>
            {tiers.map((tier) => (
              <tr key={tier.label}>
                <td><strong>{tier.label}</strong></td>
                <td><span className="cell-subtle" style={{ marginTop: 0 }}>{tier.endpoint}</span></td>
                <td><StatusBadge status={tier.status} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="infra-panel">
        <PanelHeader title="运行环境" action="前端会话" />
        <div className="kv-card-grid kv-card-grid-4">
          <div className="kv-card"><div className="kv-label">API Base</div><div className="kv-value">{API_BASE}</div><div className="kv-detail">VITE_API_BASE</div></div>
          <div className="kv-card"><div className="kv-label">Web Origin</div><div className="kv-value">{typeof window !== "undefined" ? window.location.origin : "-"}</div><div className="kv-detail">控制台地址</div></div>
          <div className="kv-card"><div className="kv-label">控制面响应</div><div className="kv-value accent">{controlPlaneStatus}</div><div className="kv-detail">/api/health</div></div>
          <div className="kv-card"><div className="kv-label">探测间隔</div><div className="kv-value">15s</div><div className="kv-detail">自动刷新</div></div>
        </div>
      </section>
    </section>
  );
}
