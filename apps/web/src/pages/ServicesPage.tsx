import { useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { RefreshCw, Server } from "lucide-react";

import { useGoToPage } from "../lib/useGoToPage";

import { KpiGrid, PageHeader, PanelHeader, StatusBadge } from "../components/common/PlatformPrimitives";
import { EmptyState, ErrorState, Skeleton } from "../components/common/FeedbackStates";
import { CallGraph } from "../components/topology/CallGraph";
import { normalizeServiceInstance } from "../data/mockPlatformData";
import { api, topologyGraph } from "../lib/api";
import { compact, fmt, relativeTime } from "../lib/format";
import { cn } from "../lib/utils";
import type { Metrics, MetricsHistorySample, ServiceConsoleRow, ServiceInstance } from "../types/platform";
import type { KpiItem } from "../types/ui";

// "—" 占位：真实端点缺失/加载中时显示，不再用写死数字伪装成有数据（5A.1）。
function show(value: number | null | undefined, format: (n: number) => string): string {
  return value == null || Number.isNaN(value) ? "—" : format(value);
}

export function ServicesPage() {
  const goTo = useGoToPage();
  const queryClient = useQueryClient();
  const [checking, setChecking] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string>("");

  const instancesQuery = useQuery({
    queryKey: ["service-instances"],
    queryFn: () => api<{ instances: ServiceInstance[] }>("/api/service-instances"),
    select: (payload) => payload.instances.map(normalizeServiceInstance),
    refetchInterval: 10000
  });

  const metricsQuery = useQuery({
    queryKey: ["metrics", "current"],
    queryFn: () => api<Metrics>("/api/metrics/current"),
    refetchInterval: 5000
  });

  const historyQuery = useQuery({
    queryKey: ["metrics", "history", "services"],
    queryFn: () => api<{ samples: MetricsHistorySample[] }>("/api/metrics/history?limit=30"),
    refetchInterval: 15000
  });

  // C3：真实调用图（OTel span 进程内派生；连线粗细 = 近 60s 真实 QPS）。
  const callGraphQuery = useQuery({
    queryKey: ["topology", "graph"],
    queryFn: topologyGraph,
    refetchInterval: 5000
  });

  const metrics = metricsQuery.data ?? null;
  const rows = useMemo(() => deriveRows(instancesQuery.data, metrics), [instancesQuery.data, metrics]);
  const selected = rows.find((row) => row.id === selectedId) ?? rows[0];
  const selectedInstance = instancesQuery.data?.find((item) => item.name === selected?.id);
  const healthyCount = rows.filter((row) => row.status === "正常").length;
  const warningCount = rows.filter((row) => row.performanceStatus === "降级").length;
  const abnormalCount = rows.filter((row) => row.status === "异常").length;
  const unknownCount = rows.filter((row) => row.status === "未知").length;
  const qps = metrics?.qps ?? null;
  const p95 = metrics?.p95_latency_ms ?? null;
  const errorRate = metrics?.error_rate ?? null;
  const samples = useMemo(() => historyQuery.data?.samples ?? [], [historyQuery.data]);
  const series = (selector: (m: Partial<Metrics>) => number | null | undefined): number[] =>
    samples.map((s) => selector(s.metrics)).filter((v): v is number => typeof v === "number");
  const qpsSeries = series((m) => m.qps);
  const p95Series = series((m) => m.p95_latency_ms);
  const errSeries = series((m) => (typeof m.error_rate === "number" ? m.error_rate * 100 : null));

  async function healthcheck(name: string) {
    setChecking(name);
    try {
      const result = await api<{ status?: string }>(`/api/service-instances/${encodeURIComponent(name)}/healthcheck`, { method: "POST" });
      await queryClient.invalidateQueries({ queryKey: ["service-instances"] });
      toast.success(`${name} 健康检查已完成`, { description: result?.status ?? "unknown" });
    } catch (error) {
      toast.error(`${name} 健康检查失败`, { description: error instanceof Error ? error.message : "请求失败" });
    } finally {
      setChecking(null);
    }
  }

  const kpis: KpiItem[] = [
    { id: "services", label: "实例可用性", value: `${healthyCount}/${rows.length}`, detail: `${abnormalCount} 个不可达 · ${warningCount} 个性能降级${unknownCount ? ` · ${unknownCount} 个未知` : ""}`, trend: [], tone: abnormalCount ? "danger" : unknownCount || warningCount ? "warning" : "success" },
    { id: "qps", label: "请求吞吐", value: show(qps, (v) => fmt(v, 1)), detail: "实时 QPS", trend: qpsSeries, ...deltaOf(qpsSeries) },
    { id: "p95", label: "P95 延迟", value: show(p95, (v) => `${fmt(v, 0)}ms`), detail: "SLO < 200ms", trend: p95Series, ...deltaOf(p95Series), tone: p95 != null && p95 > 200 ? "warning" : "success" },
    { id: "error", label: "请求错误率", value: show(errorRate, (v) => `${fmt(v * 100, 2)}%`), detail: "SLO < 1%", trend: errSeries, ...deltaOf(errSeries), tone: errorRate != null && errorRate > 0.01 ? "danger" : errorRate != null && errorRate > 0.001 ? "warning" : "success" },
  ];

  return (
    <section className="infra-page services-page service-replica">
      <PageHeader
        title="服务目录"
        subtitle="统一查看微服务、模型服务和网关实例，聚合健康状态、流量指标、资源占用与真实调用关系"
        actions={
          <button
            className="console-refresh"
            type="button"
            onClick={() => {
              instancesQuery.refetch();
              metricsQuery.refetch();
            }}
          >
            <RefreshCw size={14} /> 刷新
          </button>
        }
      />

      <KpiGrid className="service-kpi-strip" items={kpis} />

      <div className="service-main-grid">
        <section className="infra-panel service-table-panel">
          <PanelHeader title="服务实例" action={`${rows.length} 个实例`} />
          <div className="service-console-table">
            <div className="service-console-row header">
              <span>服务 / 运行时</span>
              <span>模型版本</span>
              <span>可用性 / 性能</span>
              <span>实时流量</span>
              <span>副本 / GPU</span>
              <span>操作</span>
            </div>
            {instancesQuery.isLoading ? (
              <Skeleton rows={4} />
            ) : instancesQuery.isError ? (
              <ErrorState error={instancesQuery.error} onRetry={instancesQuery.refetch} />
            ) : rows.length === 0 ? (
              <EmptyState title="暂无服务实例" description="service_instances 目录为空" />
            ) : (
              rows.map((row) => (
                <div
                  className={cn("service-console-row", selected?.id === row.id && "active")}
                  key={row.id}
                  onClick={() => setSelectedId(row.id)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      setSelectedId(row.id);
                    }
                  }}
                  role="button"
                  tabIndex={0}
                >
                  <span className="service-name-cell">
                    <ServiceIcon tone={row.tone} />
                    <span>
                      <strong>{row.name}</strong>
                      <small>{row.roleLabel} · {row.runtime}</small>
                    </span>
                  </span>
                  <span className="service-model-cell" title={row.model}><strong>{row.model}</strong><small>{row.id}</small></span>
                  <span className="service-health-cell">
                    <span><small>可用</small><StatusBadge status={row.status} /></span>
                    <span><small>SLO</small><StatusBadge status={row.performanceStatus} /></span>
                  </span>
                  <span className="service-traffic-cell">
                    <span><small>QPS</small><strong>{compact(row.qps, 0)}</strong></span>
                    <span><small>P95</small><strong className={cn(row.p95 > 200 && "warning-text", row.p95 > 500 && "danger-text")}>{fmt(row.p95, 0)}ms</strong></span>
                    <span><small>错误</small><strong className={cn(row.errorRate > 0.01 && "danger-text", row.errorRate > 0.001 && row.errorRate <= 0.01 && "warning-text")}>{fmt(row.errorRate * 100, 2)}%</strong></span>
                  </span>
                  <span className="service-resource-cell">
                    <small>{row.replicas} replicas</small>
                    {row.gpu > 0 ? (
                      <>
                        <i style={{ width: `${row.gpu}%` }} />
                        <small>GPU {row.gpu}%</small>
                      </>
                    ) : null}
                  </span>
                  <span className="service-actions-cell">
                    <button
                      disabled={checking === row.name}
                      onClick={(event) => {
                        event.stopPropagation();
                        void healthcheck(row.name);
                      }}
                      type="button"
                    >
                      {checking === row.name ? "检查中" : "检查"}
                    </button>
                    <button
                      onClick={(event) => {
                        event.stopPropagation();
                        goTo("observability");
                      }}
                      type="button"
                    >
                      观测
                    </button>
                  </span>
                </div>
              ))
            )}
          </div>
        </section>

        <aside className="infra-panel service-detail-panel">
            <PanelHeader title="当前服务" action={selected ? `可用性 ${selected.status} · 性能 ${selected.performanceStatus}` : "未选择"} />
            {selected ? (
              <>
                <div className="service-detail-head">
                  <ServiceIcon tone={selected.tone} />
                  <div>
                    <strong>{selected.name}</strong>
                    <span>{selected.roleLabel} · {selected.runtime} · {selected.model}</span>
                    <small>{selectedInstance?.base_url || "未登记访问地址"}</small>
                  </div>
                </div>
                <div className="service-detail-metrics">
                  <MetricLine label="QPS" value={compact(selected.qps, 0)} tone="info" />
                  <MetricLine label="P95 · 目标 < 200ms" value={`${fmt(selected.p95, 0)}ms`} tone={selected.p95 > 200 ? "warning" : "success"} />
                  <MetricLine label="错误率 · 目标 < 1%" value={`${fmt(selected.errorRate * 100, 2)}%`} tone={selected.errorRate > 0.01 ? "danger" : "success"} />
                  <MetricLine label="GPU" value={selected.gpu ? `${selected.gpu}%` : "—"} tone={selected.gpu >= 85 ? "warning" : "info"} />
                  <MetricLine label="副本" value={selected.replicas} tone="info" />
                  <MetricLine label="指标来源" value={selected.metricSource || "当前实例没有请求样本"} tone="info" />
                  <MetricLine label="心跳" value={selectedInstance?.last_heartbeat_at ? relativeTime(selectedInstance.last_heartbeat_at) : "静态注册"} tone="info" />
                </div>
              </>
            ) : (
              <EmptyState title="未选择服务" />
            )}
        </aside>
      </div>
      <section className="infra-panel service-callgraph-panel">
        <PanelHeader
          title="真实服务调用关系"
          action={
            <span className="callgraph-action">
              OTel trace 派生 · 近 {callGraphQuery.data?.window_seconds ?? 60}s
              <button className="icon-button" onClick={() => callGraphQuery.refetch()} type="button" title="刷新">
                <RefreshCw className={callGraphQuery.isFetching ? "spinning" : undefined} size={13} />
              </button>
            </span>
          }
        />
        <p className="callgraph-note">由真实 OTel span 派生入口、控制面、AI 服务和推理网关之间的调用关系；连线粗细表示近 60 秒 QPS。</p>
        {callGraphQuery.isLoading ? (
          <Skeleton rows={3} />
        ) : callGraphQuery.isError ? (
          <ErrorState error={callGraphQuery.error} onRetry={callGraphQuery.refetch} />
        ) : (
          <CallGraph data={callGraphQuery.data} />
        )}
      </section>
    </section>
  );
}

function deriveRows(instances: ServiceInstance[] | undefined, metrics: Metrics | null): ServiceConsoleRow[] {
  if (!instances?.length) return [];
  const endpointStats = metrics?.endpoint_stats ?? [];
  const statsByName = new Map(endpointStats.map((item) => [item.name, item]));
  const podStats = [...(metrics?.target_pod_stats?.length ? metrics.target_pod_stats : endpointStats)]
    .filter((item) => /pod|qwen|vllm|replica|customer/i.test(item.name))
    .sort((a, b) => a.name.localeCompare(b.name));
  const aggregateStat = aggregateGroupStats(endpointStats.length ? endpointStats : metrics?.target_pod_stats ?? []);
  const gpuPeak = metrics?.gpu?.length ? Math.max(...metrics.gpu.map((g) => g.gpu_utilization_percent)) : 0;
  return instances.map((item) => {
    const normalized = normalizeServiceInstance(item);
    const availability = availabilityStatus(normalized.status);
    const stat = availability === "正常" ? pickServiceStat(normalized, statsByName, podStats, aggregateStat) : undefined;
    const p95 = stat?.p95_latency_ms ?? 0;
    const errorRate = stat?.error_rate ?? 0;
    const qps = stat?.request_count && metrics?.window_seconds ? stat.request_count / metrics.window_seconds : 0;
    const onGpu = !!normalized.gpu_id && !["auto", "gateway"].includes(normalized.gpu_id);
    const gpu = onGpu ? Math.round(gpuPeak) : 0;
    const performanceStatus: ServiceConsoleRow["performanceStatus"] = availability !== "正常"
      ? "未采集"
      : !stat || !stat.request_count ? "无流量"
      : errorRate > 0.01 ? "异常" : p95 > 200 ? "降级" : "正常";
    const tone: ServiceConsoleRow["tone"] = availability === "异常" || performanceStatus === "异常"
      ? "danger" : availability === "未知" || performanceStatus === "降级" ? "warning" : "success";
    return {
      id: normalized.name,
      name: normalized.name,
      runtime: normalized.kind === "vllm" ? "vLLM" : normalized.kind === "aibrix" ? "AIBrix Gateway" : normalized.kind || "Runtime",
      model: normalized.model_id || "-",
      status: availability,
      performanceStatus,
      roleLabel: serviceRoleLabel(normalized),
      metricSource: stat?.name,
      replicas: normalized.routing_role === "replica" ? "1/1" : normalized.kind === "aibrix" || normalized.kind.includes("router") ? "逻辑入口" : "1/1",
      qps,
      p95,
      errorRate,
      cpu: 0,
      gpu,
      tone,
      trend: [],
    };
  }).sort((a, b) => availabilityOrder(a.status) - availabilityOrder(b.status) || a.roleLabel.localeCompare(b.roleLabel) || a.name.localeCompare(b.name));
}

function pickServiceStat(
  instance: ServiceInstance,
  statsByName: Map<string, Metrics["endpoint_stats"][number]>,
  podStats: Metrics["endpoint_stats"],
  aggregateStat: Metrics["endpoint_stats"][number] | null,
): Metrics["endpoint_stats"][number] | undefined {
  const exact = statsByName.get(instance.name);
  if (exact) return exact;

  const role = String(instance.routing_role || "").toLowerCase();
  const kind = String(instance.kind || "").toLowerCase();
  if (kind === "vllm" || role === "replica") {
    const modelToken = String(instance.model_id || "").toLowerCase().replace(/[^a-z0-9]/g, "");
    return modelToken ? podStats.find((item) => item.name.toLowerCase().replace(/[^a-z0-9]/g, "").includes(modelToken)) : undefined;
  }

  if (role.includes("gateway") || kind.includes("router") || kind.includes("aibrix")) {
    return statsByName.get(instance.name) ?? statsByName.get("auto-router") ?? statsByName.get("aibrix-gateway") ?? aggregateStat ?? undefined;
  }

  return aggregateStat ?? undefined;
}

function availabilityStatus(status: string): ServiceConsoleRow["status"] {
  const value = status.toLowerCase();
  if (value === "healthy" || value === "ready" || value === "running") return "正常";
  if (["unreachable", "failed", "missing", "stopped", "error"].includes(value)) return "异常";
  return "未知";
}

function availabilityOrder(status: ServiceConsoleRow["status"]): number {
  return status === "正常" ? 0 : status === "未知" ? 1 : 2;
}

function serviceRoleLabel(instance: ServiceInstance): string {
  const kind = String(instance.kind || "").toLowerCase();
  const role = String(instance.routing_role || "").toLowerCase();
  if (kind === "aibrix" || role === "gateway") return "网关入口";
  if (kind === "auto_router") return "自动路由";
  if (kind === "client_round_robin") return "客户端轮询";
  if (kind === "vllm" && role === "replica") return "模型副本";
  if (kind === "vllm") return "模型运行时";
  return role || kind || "服务实例";
}

function aggregateGroupStats(items: Metrics["endpoint_stats"]): Metrics["endpoint_stats"][number] | null {
  if (!items.length) return null;
  const requestCount = items.reduce((sum, item) => sum + (item.request_count || 0), 0);
  const errorCount = items.reduce((sum, item) => sum + (item.error_count || 0), 0);
  const outputTokens = items.reduce((sum, item) => sum + (item.output_tokens || 0), 0);
  const weighted = (field: "mean_latency_ms" | "p95_latency_ms" | "mean_ttft_ms" | "p95_ttft_ms") => {
    if (!requestCount) return null;
    const total = items.reduce((sum, item) => sum + ((item[field] ?? 0) * (item.request_count || 0)), 0);
    return total / requestCount;
  };
  return {
    name: "aggregate",
    request_count: requestCount,
    error_count: errorCount,
    error_rate: requestCount ? errorCount / requestCount : 0,
    mean_latency_ms: weighted("mean_latency_ms"),
    p95_latency_ms: weighted("p95_latency_ms"),
    mean_ttft_ms: weighted("mean_ttft_ms"),
    p95_ttft_ms: weighted("p95_ttft_ms"),
    output_tokens: outputTokens,
  };
}

function deltaOf(series: number[]): { delta: string; deltaTone: "up" | "down" | "flat" } {
  if (series.length < 2) return { delta: "", deltaTone: "flat" };
  const change = ((series[series.length - 1] - series[0]) / (series[0] || 1)) * 100;
  return {
    delta: `${change >= 0 ? "+" : ""}${change.toFixed(1)}%`,
    deltaTone: change > 0 ? "up" : change < 0 ? "down" : "flat",
  };
}

function ServiceIcon({ tone }: { tone: ServiceConsoleRow["tone"] }) {
  return (
    <span className={cn("service-console-icon", tone)}>
      <Server size={15} />
    </span>
  );
}

function MetricLine({ label, tone, value }: { label: string; tone: "info" | "success" | "warning" | "danger"; value: string }) {
  return (
    <div className="service-detail-line">
      <span>{label}</span>
      <strong className={tone}>{value}</strong>
    </div>
  );
}
