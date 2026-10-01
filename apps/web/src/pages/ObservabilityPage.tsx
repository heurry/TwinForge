import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { MoreVertical, Pause, Play, RefreshCw, Search } from "lucide-react";

import { useGoToPage } from "../lib/useGoToPage";

import { Donut, KpiGrid, PageHeader, PanelHeader, Sparkline, StatusBadge } from "../components/common/PlatformPrimitives";
import { EmptyState, ErrorState, Skeleton } from "../components/common/FeedbackStates";
import { api } from "../lib/api";
import { byteRate, bytes, compact, fmt, relativeTime } from "../lib/format";
import { cn } from "../lib/utils";
import type { KubernetesSnapshot, Metrics, MetricsHistorySample, RequestTrace } from "../types/platform";
import type { KpiItem } from "../types/ui";

type ObsTone = "info" | "success" | "warning" | "danger" | "ai";

type AlertRow = {
  id: string;
  severity: string;
  name: string;
  service: string;
  instance: string;
  metric: string;
  value: string;
  threshold: string;
  status: string;
  triggered_at: string;
};
type PlatformLog = {
  id: number | string;
  timestamp: string | null;
  level: string;
  source: string;
  resource_type: string;
  resource_id: string;
  message: string;
  attributes: Record<string, unknown>;
};

type ServiceRow = {
  name: string;
  qps: number;
  p95: number;
  ttft: number;
  error: number;
  availability: number;
  outputTokens: number;
  tone: ObsTone;
  values: number[];
};
type SlowRow = { endpoint: string; service: string; p95: string; ratio: string };
type ContainerRow = { id: string; pod: string; container: string; cpu: string; memory: string; network: string; image: string };
type SloObjective = { id: string; label: string; target: string; current: string; remaining: string; status: string; tone: ObsTone };
type SloEndpointRow = {
  name: string;
  requests: number;
  p95: string;
  ttft: string;
  errorRate: string;
  availability: string;
  violations: string;
  status: string;
  tone: ObsTone;
};
const OBS_TABS = ["指标总览", "请求追踪", "日志检索", "SLO 监控", "资源监控", "告警事件"] as const;
type ObsTab = (typeof OBS_TABS)[number];
const ALL_SERVICES = "全部服务";
const ALL_INSTANCES = "全部实例";
const METRIC_FILTERS = ["延迟 (P95)", "TTFT", "错误率", "吞吐"] as const;
const GROUP_BY_FILTERS = ["服务", "实例"] as const;
type ObsMetricFilter = (typeof METRIC_FILTERS)[number];
type ObsGroupBy = (typeof GROUP_BY_FILTERS)[number];

// "—" 占位：真实端点缺失/加载中时显示，不再用写死数字伪装成有数据（5A.1）。
function show(value: number | null | undefined, format: (n: number) => string): string {
  return value == null || Number.isNaN(value) ? "—" : format(value);
}

export function ObservabilityPage() {
  const goTo = useGoToPage();
  const [activeTab, setActiveTab] = useState<ObsTab>("指标总览");
  const [query, setQuery] = useState("");
  const [logDraft, setLogDraft] = useState("");
  const [logSearch, setLogSearch] = useState("");
  const [logMode, setLogMode] = useState<"platform" | "kubernetes">("platform");
  const [logNamespace, setLogNamespace] = useState("default");
  const [logPod, setLogPod] = useState("");
  const [logContainer, setLogContainer] = useState("");
  const [logLevel, setLogLevel] = useState("");
  const [logSource, setLogSource] = useState("");
  const [logResourceType, setLogResourceType] = useState("");
  const [serviceFilter, setServiceFilter] = useState<string>(ALL_SERVICES);
  const [instanceFilter, setInstanceFilter] = useState<string>(ALL_INSTANCES);
  const [metricFilter, setMetricFilter] = useState<ObsMetricFilter>("延迟 (P95)");
  const [groupBy, setGroupBy] = useState<ObsGroupBy>("服务");
  const [refreshMs, setRefreshMs] = useState(5000);
  const [paused, setPaused] = useState(false);
  const [historyMinutes, setHistoryMinutes] = useState(30);
  const metricsQuery = useQuery({
    queryKey: ["metrics", "current"],
    queryFn: () => api<Metrics>("/api/metrics/current"),
    refetchInterval: paused ? false : refreshMs
  });
  const historyQuery = useQuery({
    queryKey: ["metrics", "history", historyMinutes],
    queryFn: () => api<{ samples: MetricsHistorySample[] }>(`/api/metrics/history?minutes=${historyMinutes}&limit=2000`),
    refetchInterval: paused ? false : refreshMs
  });
  const tracesQuery = useQuery({
    queryKey: ["metrics", "requests", 200],
    queryFn: () => api<{ requests: RequestTrace[] }>("/api/metrics/requests?limit=200"),
    refetchInterval: paused ? false : refreshMs
  });
  const alertsQuery = useQuery({
    queryKey: ["alerts"],
    queryFn: () => api<{ alerts: AlertRow[]; summary: Record<string, number> }>("/api/alerts"),
    refetchInterval: paused ? false : refreshMs
  });
  const logPodsQuery = useQuery({
    queryKey: ["kubernetes", "snapshot", "log-picker"],
    queryFn: () => api<KubernetesSnapshot>("/api/kubernetes/snapshot"),
    enabled: activeTab === "日志检索" && logMode === "kubernetes",
    staleTime: 15_000,
  });
  const logSince = () => new Date(Date.now() - historyMinutes * 60_000).toISOString();
  const logsQuery = useQuery({
    queryKey: ["platform-logs", logSearch, logMode, logNamespace, logPod, logContainer, logLevel, logSource, logResourceType, historyMinutes],
    queryFn: () => {
      const params = new URLSearchParams({ limit: "300", q: logSearch, since: logSince() });
      if (logMode === "kubernetes" && logPod.trim()) {
        params.set("namespace", logNamespace.trim() || "default");
        params.set("pod", logPod.trim());
        if (logContainer) params.set("container", logContainer);
      } else {
        if (logLevel) params.set("level", logLevel);
        if (logSource) params.set("source", logSource);
        if (logResourceType) params.set("resource_type", logResourceType);
      }
      return api<{ logs: PlatformLog[]; count: number; source?: string }>(`/api/logs?${params.toString()}`);
    },
    enabled: activeTab === "日志检索" && (logMode === "platform" || Boolean(logPod)),
    refetchInterval: paused ? false : refreshMs
  });

  const logPods = logPodsQuery.data?.pods ?? [];
  const logNamespaces = uniqueStrings(logPods.map((pod) => pod.namespace));
  const namespacePods = logPods.filter((pod) => pod.namespace === logNamespace);
  const selectedLogPod = namespacePods.find((pod) => pod.name === logPod);

  const metrics = metricsQuery.data ?? null;
  const reqCount = metrics?.request_count ?? null;
  const p95 = metrics?.p95_latency_ms ?? null;
  const ttft = metrics?.p95_ttft_ms ?? null;
  const tps = metrics?.tokens_per_second ?? null;
  const errorRate = metrics?.error_rate ?? null;
  const availability = errorRate == null ? null : Math.max(0, 100 - errorRate * 100);

  const samples = useMemo(() => historyQuery.data?.samples ?? [], [historyQuery.data]);
  const series = (selector: (m: Partial<Metrics>) => number | null | undefined): number[] =>
    samples.map((s) => selector(s.metrics)).filter((v): v is number => typeof v === "number");
  const reqSeries = series((m) => m.request_count);
  const p95Series = series((m) => m.p95_latency_ms);
  const ttftSeries = series((m) => m.p95_ttft_ms);
  const tpsSeries = series((m) => m.tokens_per_second);
  const errSeries = series((m) => (typeof m.error_rate === "number" ? m.error_rate * 100 : null));
  const availSeries = errSeries.map((e) => Math.max(0, 100 - e));

  const kpis: KpiItem[] = [
    { id: "qps", label: "请求总数", value: show(reqCount, (v) => compact(v, 0)), detail: "窗口内请求", trend: reqSeries, ...deltaOf(reqSeries), tone: "success" },
    { id: "p95", label: "P95 延迟", value: show(p95, (v) => `${fmt(v, 0)}ms`), detail: "SLO < 300ms", trend: p95Series, ...deltaOf(p95Series), tone: p95 != null && p95 > 300 ? "danger" : "warning" },
    { id: "ttft", label: "TTFT", value: show(ttft, (v) => `${fmt(v, 0)}ms`), detail: "首 token", trend: ttftSeries, ...deltaOf(ttftSeries) },
    { id: "tps", label: "Tokens/s", value: show(tps, (v) => compact(v, 0)), detail: "解码吞吐", trend: tpsSeries, ...deltaOf(tpsSeries) },
    { id: "error", label: "错误率", value: show(errorRate, (v) => `${fmt(v * 100, 2)}%`), detail: "请求失败", trend: errSeries, ...deltaOf(errSeries), tone: "danger" },
    { id: "availability", label: "可用性", value: show(availability, (v) => `${fmt(v, 2)}%`), detail: "1 - 错误率", trend: availSeries, ...deltaOf(availSeries), tone: "success" },
  ];

  const serviceOptions = useMemo(
    () => withCurrentOption([
      ALL_SERVICES,
      ...uniqueStrings(metrics?.endpoint_stats?.map((row) => row.name) ?? []),
      ...uniqueStrings((tracesQuery.data?.requests ?? []).map((row) => row.endpoint_id).filter(Boolean)),
    ], serviceFilter),
    [metrics?.endpoint_stats, serviceFilter, tracesQuery.data?.requests]
  );
  const instanceOptions = useMemo(
    () => withCurrentOption([
      ALL_INSTANCES,
      ...uniqueStrings(metrics?.target_pod_stats?.map((row) => row.name) ?? []),
      ...uniqueStrings(Object.keys(metrics?.target_pod_counts ?? {})),
      ...uniqueStrings((tracesQuery.data?.requests ?? []).map((row) => row.target_pod).filter(Boolean)),
    ], instanceFilter),
    [instanceFilter, metrics?.target_pod_counts, metrics?.target_pod_stats, tracesQuery.data?.requests]
  );
  const serviceRows = useMemo(() => deriveServiceRows(metrics, groupBy, metricFilter), [groupBy, metricFilter, metrics]);
  const slowRequests = useMemo(() => deriveSlowRequests(tracesQuery.data?.requests), [tracesQuery.data?.requests]);
  const filteredServiceRows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return serviceRows.filter((row) => {
      const matchesQuery = !q || row.name.toLowerCase().includes(q);
      const matchesService = groupBy !== "服务" || serviceFilter === ALL_SERVICES || row.name === serviceFilter;
      const matchesInstance = groupBy !== "实例" || instanceFilter === ALL_INSTANCES || row.name === instanceFilter;
      return matchesQuery && matchesService && matchesInstance;
    });
  }, [groupBy, instanceFilter, query, serviceFilter, serviceRows]);
  const filteredSlowRequests = useMemo(() => {
    const q = query.trim().toLowerCase();
    return slowRequests.filter((row) =>
      matchesSearch([row.endpoint, row.service], q)
      && (serviceFilter === ALL_SERVICES || row.endpoint === serviceFilter)
      && (instanceFilter === ALL_INSTANCES || row.service === instanceFilter)
    );
  }, [instanceFilter, query, serviceFilter, slowRequests]);
  const requestRows = useMemo(() => {
    const rows = tracesQuery.data?.requests ?? [];
    const q = query.trim().toLowerCase();
    return rows.filter((row) =>
      matchesSearch([row.request_id, row.endpoint_id, row.target_pod, row.status, row.error ?? ""], q)
      && (serviceFilter === ALL_SERVICES || row.endpoint_id === serviceFilter)
      && (instanceFilter === ALL_INSTANCES || row.target_pod === instanceFilter)
    );
  }, [instanceFilter, query, serviceFilter, tracesQuery.data?.requests]);
  const latencyChart = useMemo(() => buildMetricChart(samples, metricFilter), [metricFilter, samples]);
  const currentValues = useMemo(() => deriveCurrentValues(metrics).filter((row) => {
    const q = query.trim().toLowerCase();
    return matchesSearch([row.service], q) && (serviceFilter === ALL_SERVICES || row.service === serviceFilter);
  }), [metrics, query, serviceFilter]);
  const errorDist = useMemo(() => deriveErrorDist(metrics), [metrics]);
  const sloObjectives = useMemo(() => deriveSloObjectives(metrics), [metrics]);
  const sloEndpointRows = useMemo(() => deriveSloEndpointRows(metrics).filter((row) => {
    const q = query.trim().toLowerCase();
    return matchesSearch([row.name], q) && (serviceFilter === ALL_SERVICES || row.name === serviceFilter);
  }), [metrics, query, serviceFilter]);
  const gpuRows = metrics?.gpu ?? [];
  const gpuStatus = metrics?.gpu_status;
  const cadvisor = metrics?.cadvisor;
  const containerRows = useMemo(() => deriveContainerRows(metrics, query, instanceFilter), [instanceFilter, metrics, query]);
  const resetFilters = () => {
    setQuery("");
    setServiceFilter(ALL_SERVICES);
    setInstanceFilter(ALL_INSTANCES);
    setMetricFilter("延迟 (P95)");
    setGroupBy("服务");
    setLogNamespace("default");
    setLogPod("");
    setLogContainer("");
    setLogDraft("");
    setLogSearch("");
    setLogLevel("");
    setLogSource("");
    setLogResourceType("");
  };
  const refetchAll = () => {
    metricsQuery.refetch();
    historyQuery.refetch();
    tracesQuery.refetch();
    alertsQuery.refetch();
    if (activeTab === "日志检索") logsQuery.refetch();
  };
  const showOverview = activeTab === "指标总览";
  const showTracing = activeTab === "请求追踪";
  const showLogs = activeTab === "日志检索";
  const showSlo = activeTab === "SLO 监控";
  const showResources = activeTab === "资源监控";
  const showAlerts = activeTab === "告警事件";
  const tabClass = activeTab === "指标总览" ? "overview"
    : activeTab === "请求追踪" ? "tracing"
      : activeTab === "日志检索" ? "logs"
        : activeTab === "SLO 监控" ? "slo"
          : activeTab === "资源监控" ? "resources"
            : "alerts";

  return (
    <section className="infra-page observability-page obs-replica">
      <PageHeader
        title="可观测中心"
        subtitle="实时监控、指标分析、链路追踪与日志检索"
        actions={
          <button
            className="console-refresh"
            onClick={refetchAll}
            type="button"
          >
            <RefreshCw size={14} /> 刷新
          </button>
        }
      />

      <KpiGrid className="obs-kpi-strip" items={kpis} />

      <section className="infra-panel obs-filter-panel">
        <div className="obs-tabs" role="tablist">
          {OBS_TABS.map((tab) => (
            <button
              aria-selected={activeTab === tab}
              className={activeTab === tab ? "active" : undefined}
              key={tab}
              onClick={() => setActiveTab(tab)}
              role="tab"
              type="button"
            >
              {tab}
            </button>
          ))}
          <div className="obs-refresh-meta">
            <span>时间范围</span>
            <select onChange={(event) => setHistoryMinutes(Number(event.target.value))} value={historyMinutes}>
              <option value={30}>近 30 分钟</option>
              <option value={60}>近 1 小时</option>
              <option value={180}>近 3 小时</option>
              <option value={1440}>近 24 小时</option>
              <option value={10080}>近 7 天</option>
            </select>
            <span>刷新</span>
            <select disabled={paused} onChange={(event) => setRefreshMs(Number(event.target.value))} value={refreshMs}>
              <option value={5000}>5s</option>
              <option value={10000}>10s</option>
              <option value={30000}>30s</option>
            </select>
            <button onClick={() => setPaused((value) => !value)} title={paused ? "恢复刷新" : "暂停刷新"} type="button">
              {paused ? <Play size={14} /> : <Pause size={14} />}
            </button>
          </div>
        </div>
        {showLogs ? (
        <div className="obs-query-row obs-log-query-row">
          <label>日志来源
            <select onChange={(event) => { setLogMode(event.target.value as "platform" | "kubernetes"); setLogPod(""); setLogContainer(""); }} value={logMode}>
              <option value="platform">平台事件索引</option>
              <option value="kubernetes">Kubernetes Pod</option>
            </select>
          </label>
          {logMode === "platform" ? <>
            <label>级别<select onChange={(event) => setLogLevel(event.target.value)} value={logLevel}>
              <option value="">全部级别</option><option value="info">Info</option><option value="warning">Warning</option><option value="error">Error</option>
            </select></label>
            <label>事件来源<select onChange={(event) => setLogSource(event.target.value)} value={logSource}>
              <option value="">全部来源</option><option value="gitlab-ci">GitLab CI</option><option value="github-actions">GitHub Actions</option><option value="deployment-runner">部署执行器</option><option value="training-runner">训练任务</option><option value="routing-controller">流量治理</option><option value="healthcheck">健康检查</option>
            </select></label>
            <label>资源类型<select onChange={(event) => setLogResourceType(event.target.value)} value={logResourceType}>
              <option value="">全部资源</option><option value="deployment">Deployment</option><option value="service_instance">服务实例</option><option value="training_job">训练任务</option><option value="repository">代码仓库</option>
            </select></label>
          </> : <>
            <label>Namespace<select onChange={(event) => { setLogNamespace(event.target.value); setLogPod(""); setLogContainer(""); }} value={logNamespace}>
              {!logNamespaces.includes(logNamespace) ? <option value={logNamespace}>{logNamespace}</option> : null}
              {logNamespaces.map((namespace) => <option key={namespace} value={namespace}>{namespace}</option>)}
            </select></label>
            <label>Pod<select onChange={(event) => { setLogPod(event.target.value); setLogContainer(""); }} value={logPod}>
              <option value="">请选择 Pod</option>{namespacePods.map((pod) => <option key={pod.name} value={pod.name}>{pod.name}</option>)}
            </select></label>
            <label>容器<select disabled={!logPod} onChange={(event) => setLogContainer(event.target.value)} value={logContainer}>
              <option value="">{selectedLogPod?.containers?.length === 1 ? "默认容器" : "请选择容器"}</option>
              {(selectedLogPod?.containers ?? []).map((container) => <option key={container} value={container}>{container}</option>)}
            </select></label>
          </>}
          <label className="obs-query-search"><Search size={13} /><input onChange={(event) => setLogDraft(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") setLogSearch(logDraft.trim()); }} placeholder="搜索日志内容" value={logDraft} /></label>
          <button className="obs-reset" onClick={resetFilters} type="button">重置</button>
          <button className="obs-submit" disabled={logMode === "kubernetes" && !logPod} onClick={() => { const next = logDraft.trim(); if (next === logSearch) logsQuery.refetch(); else setLogSearch(next); }} type="button">查询</button>
        </div>
        ) : (
        <div className="obs-query-row">
          <label>服务
            <select aria-label="服务筛选" onChange={(event) => setServiceFilter(event.target.value)} value={serviceFilter}>
              {serviceOptions.map((option) => <option key={option} value={option}>{option}</option>)}
            </select>
          </label>
          <label>实例
            <select aria-label="实例筛选" onChange={(event) => setInstanceFilter(event.target.value)} value={instanceFilter}>
              {instanceOptions.map((option) => <option key={option} value={option}>{option}</option>)}
            </select>
          </label>
          <label>指标
            <select aria-label="指标筛选" onChange={(event) => setMetricFilter(event.target.value as ObsMetricFilter)} value={metricFilter}>
              {METRIC_FILTERS.map((option) => <option key={option} value={option}>{option}</option>)}
            </select>
          </label>
          <label>分组维度
            <select aria-label="分组维度" onChange={(event) => setGroupBy(event.target.value as ObsGroupBy)} value={groupBy}>
              {GROUP_BY_FILTERS.map((option) => <option key={option} value={option}>{option}</option>)}
            </select>
          </label>
          <label className="obs-query-search"><Search size={13} />
            <input onChange={(event) => setQuery(event.target.value)} placeholder="搜索 Trace / Label" value={query} />
          </label>
          <button className="obs-reset" onClick={resetFilters} type="button">重置</button>
          <button className="obs-submit" onClick={refetchAll} type="button">查询</button>
        </div>
        )}
      </section>

      <div className={cn("obs-tab-content", `tab-${tabClass}`)} role="tabpanel">
        {showOverview ? (
          <section className="infra-panel obs-trend-panel">
            <div className="obs-trend-main">
              <PanelHeader title={latencyChart.title} action={`近 ${historyMinutes} 分钟 · ${samples.length || 0} 个采样点`} />
              {historyQuery.isLoading ? (
                <Skeleton rows={4} />
              ) : historyQuery.isError ? (
                <ErrorState error={historyQuery.error} onRetry={historyQuery.refetch} />
              ) : (
                <LatencyChart chart={latencyChart} />
              )}
            </div>
            <aside className="obs-current-panel">
              <PanelHeader title="当前值" action="endpoint P95" />
              <div className="obs-current-table">
                <div className="obs-current-row header">
                  <span>服务</span>
                  <span>P95</span>
                  <span>SLO 对比</span>
                  <span>状态</span>
                </div>
                {currentValues.length === 0 ? (
                  <EmptyState title="暂无 endpoint 指标" />
                ) : (
                  currentValues.map((row) => (
                    <div className="obs-current-row" key={row.service}>
                      <strong>{row.service}</strong>
                      <span>{row.p95}</span>
                      <span>{row.slo}</span>
                      <StatusBadge status={row.status} />
                    </div>
                  ))
                )}
              </div>
            </aside>
          </section>
        ) : null}

        {showResources ? (
          <div className="obs-resource-grid">
            <section className="infra-panel obs-gpu-panel">
              <PanelHeader
                title="GPU 资源"
                action={gpuRows.length ? `${gpuRows.length} 张 GPU` : gpuStatus?.available === false ? "采集异常" : "Agent /api/gpu"}
              />
              {metricsQuery.isLoading ? (
                <Skeleton rows={3} />
              ) : metricsQuery.isError ? (
                <ErrorState error={metricsQuery.error} onRetry={metricsQuery.refetch} />
              ) : gpuRows.length === 0 ? (
                <>
                  <EmptyState title="暂无 GPU 指标" description={gpuStatus?.error || "等待 Agent 返回 nvidia-smi 采样"} />
                  {gpuStatus?.error && <p className="gpu-diagnostic">{gpuStatus.error}</p>}
                </>
              ) : (
                <div className="gpu-card-grid compact">
                  {gpuRows.map((gpu) => {
                    const memPct = gpu.memory_utilization_percent ?? (gpu.memory_total_mb ? (gpu.memory_used_mb / gpu.memory_total_mb) * 100 : 0);
                    const tone = gpu.gpu_utilization_percent >= 90 ? "danger" : gpu.gpu_utilization_percent >= 75 ? "warn" : "ok";
                    return (
                      <article className="gpu-mini-card" key={gpu.index}>
                        <div className="gpu-mini-head">
                          <strong>GPU {gpu.index}</strong>
                          <span title={gpu.name}>{gpu.name}</span>
                          <b className={tone}>{fmt(gpu.gpu_utilization_percent, 0)}%</b>
                        </div>
                        <GpuMeter label="Util" value={gpu.gpu_utilization_percent} />
                        <GpuMeter label="Mem" value={memPct} detail={`${fmt(gpu.memory_used_mb / 1024, 1)} / ${fmt(gpu.memory_total_mb / 1024, 1)} GiB`} />
                        <div className="gpu-metric-line">
                          <span>{gpu.temperature_celsius != null ? `${fmt(gpu.temperature_celsius, 0)}°C` : "温度 —"}</span>
                          <strong>{gpu.power_watts != null ? `${fmt(gpu.power_watts, 0)} W` : "功耗 —"}</strong>
                        </div>
                      </article>
                    );
                  })}
                </div>
              )}
            </section>

            <section className="infra-panel obs-cadvisor-panel">
              <PanelHeader
                title="容器资源 Top"
                action={cadvisor?.available ? `${cadvisor.summary.container_count} containers` : "cAdvisor"}
              />
              <div className="obs-resource-summary">
                <span><strong>{cadvisor?.available ? fmt(cadvisor.summary.cpu_cores, 2) : "—"}</strong><small>CPU cores</small></span>
                <span><strong>{cadvisor?.available ? bytes(cadvisor.summary.memory_working_set_bytes) : "—"}</strong><small>Working set</small></span>
                <span><strong>{cadvisor?.available ? `↓${byteRate(cadvisor.summary.network_rx_bytes_per_second)} ↑${byteRate(cadvisor.summary.network_tx_bytes_per_second)}` : "—"}</strong><small>Network</small></span>
              </div>
              {metricsQuery.isLoading ? (
                <Skeleton rows={4} />
              ) : !cadvisor?.available ? (
                <EmptyState title="cAdvisor 未接入" description={cadvisor?.error || "请启动 cAdvisor port-forward 并配置 CADVISOR_URL"} />
              ) : containerRows.length === 0 ? (
                <EmptyState title="暂无容器资源" description={query ? "无匹配容器" : "等待 cAdvisor 样本"} />
              ) : (
                <div className="cadvisor-table-wrap">
                  <div className="cadvisor-table header">
                    <span>Pod</span>
                    <span>Container</span>
                    <span>CPU</span>
                    <span>内存</span>
                    <span>网络</span>
                    <span>镜像</span>
                  </div>
                  {containerRows.map((row) => (
                    <div className="cadvisor-table" key={row.id}>
                      <strong title={row.pod}>{row.pod}</strong>
                      <span title={row.container}>{row.container}</span>
                      <span>{row.cpu}</span>
                      <span>{row.memory}</span>
                      <span title={row.network}>{row.network}</span>
                      <span title={row.image}>{row.image}</span>
                    </div>
                  ))}
                </div>
              )}
            </section>
          </div>
        ) : null}

        {(showOverview || showTracing) ? (
          <div className={cn("obs-mid-grid", showTracing && "trace-only")}>
            {showOverview ? (
              <section className="infra-panel obs-service-panel">
                <PanelHeader title={groupBy === "服务" ? "服务概览" : "实例概览"} action={`${filteredServiceRows.length} / ${serviceRows.length}`} />
                <div className="obs-service-table">
                  <div className="obs-service-row header">
                    <span>{groupBy === "服务" ? "服务名称" : "实例名称"}</span>
                    <span>请求 (QPS)</span>
                    <span>P95 延迟</span>
                    <span>错误率</span>
                    <span>可用性</span>
                    <span>操作</span>
                  </div>
                  {metricsQuery.isLoading ? (
                    <Skeleton rows={4} />
                  ) : metricsQuery.isError ? (
                    <ErrorState error={metricsQuery.error} onRetry={metricsQuery.refetch} />
                  ) : filteredServiceRows.length === 0 ? (
                    <EmptyState title="暂无服务指标" description={query ? "无匹配结果" : "等待 endpoint 指标采集"} />
                  ) : (
                    filteredServiceRows.map((row) => (
                      <div className="obs-service-row" key={row.name}>
                        <span className="obs-service-name">
                          <strong>{row.name}</strong>
                          <Sparkline values={row.values} tone={row.tone} width={70} height={24} />
                        </span>
                        <span>{compact(row.qps, 0)}</span>
                        <strong className={row.p95 > 200 ? "warning-text" : "success-text"}>{fmt(row.p95, 0)}ms</strong>
                        <strong className={row.error > 0.01 ? "danger-text" : undefined}>{fmt(row.error * 100, 2)}%</strong>
                        <span>{fmt(row.availability, 2)}%</span>
                        <button onClick={() => goTo("aiOps")} type="button"><MoreVertical size={13} /></button>
                      </div>
                    ))
                  )}
                </div>
              </section>
            ) : null}

            {(showOverview || showTracing) ? (
              <section className="infra-panel obs-slow-panel">
                <PanelHeader title="最慢请求 TOP 5" />
                <div className="obs-slow-table">
                  <div className="obs-slow-row header">
                    <span>接口</span>
                    <span>服务</span>
                    <span>P95 延迟</span>
                    <span>占比</span>
                  </div>
                  {tracesQuery.isLoading ? (
                    <Skeleton rows={3} />
                  ) : tracesQuery.isError ? (
                    <ErrorState error={tracesQuery.error} onRetry={tracesQuery.refetch} />
                  ) : filteredSlowRequests.length === 0 ? (
                    <EmptyState title="暂无请求链路" description={query ? "无匹配结果" : "等待 request traces"} />
                  ) : (
                    filteredSlowRequests.map((row) => (
                      <div className="obs-slow-row" key={row.endpoint}>
                        <strong>{row.endpoint}</strong>
                        <span>{row.service}</span>
                        <strong className="danger-text">{row.p95}</strong>
                        <span>{row.ratio}</span>
                      </div>
                    ))
                  )}
                </div>
              </section>
            ) : null}

            {showTracing ? (
              <section className="infra-panel obs-request-panel">
                <PanelHeader title="请求明细" action={`${requestRows.length} 条 trace`} />
                <div className="obs-request-table">
                  <div className="obs-request-row header">
                    <span>请求 ID</span>
                    <span>Endpoint</span>
                    <span>目标实例</span>
                    <span>状态</span>
                    <span>Total</span>
                    <span>TTFT</span>
                    <span>时间</span>
                  </div>
                  {tracesQuery.isLoading ? (
                    <Skeleton rows={5} />
                  ) : tracesQuery.isError ? (
                    <ErrorState error={tracesQuery.error} onRetry={tracesQuery.refetch} />
                  ) : requestRows.length === 0 ? (
                    <EmptyState title="暂无请求明细" description={query ? "无匹配结果" : "等待 request traces"} />
                  ) : (
                    requestRows.slice(0, 12).map((trace) => (
                      <div className="obs-request-row" key={trace.request_id}>
                        <strong title={trace.request_id}>{trace.request_id}</strong>
                        <span title={trace.endpoint_id}>{trace.endpoint_id || "—"}</span>
                        <span title={trace.target_pod}>{trace.target_pod || "—"}</span>
                        <StatusBadge status={trace.status || "unknown"} />
                        <span>{trace.total_ms == null ? "—" : `${fmt(trace.total_ms, 0)}ms`}</span>
                        <span>{trace.ttft_ms == null ? "—" : `${fmt(trace.ttft_ms, 0)}ms`}</span>
                        <span>{relativeTime(trace.created_at)}</span>
                      </div>
                    ))
                  )}
                </div>
              </section>
            ) : null}

            {showOverview ? (
              <section className="infra-panel obs-error-panel">
                <PanelHeader title="错误分布" action="按 endpoint" />
                <div className="obs-error-content">
                  <Donut value={errorDist.errorRatePct} max={100} size={138} thickness={20} tone="danger" label={`错误 ${errorDist.errorCount} (${fmt(errorDist.errorRatePct, 2)}%)`} />
                  <div className="obs-error-list">
                    {errorDist.rows.length === 0 ? (
                      <EmptyState title={errorDist.totalCount === 0 ? "暂无请求" : "无错误"} />
                    ) : (
                      errorDist.rows.map((item) => (
                        <div className="obs-error-row" key={item.label}>
                          <span className={item.tone}><i />{item.label}</span>
                          <strong>{item.value}</strong>
                          <small>{item.detail}</small>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </section>
            ) : null}
          </div>
        ) : null}

        {showSlo ? (
          <section className="infra-panel obs-slo-panel">
            <PanelHeader title="SLO 监控" action="目标 / 违约 / 预算" />
            <div className="obs-slo-objectives">
              {sloObjectives.map((item) => (
                <article className={cn("obs-slo-objective", item.tone)} key={item.id}>
                  <span>{item.label}</span>
                  <strong>{item.current}</strong>
                  <small>{item.target}</small>
                  <em>{item.remaining}</em>
                  <StatusBadge status={item.status} />
                </article>
              ))}
            </div>
            <div className="obs-slo-table">
              <div className="obs-slo-row header">
                <span>Endpoint</span>
                <span>请求</span>
                <span>P95</span>
                <span>TTFT</span>
                <span>错误率</span>
                <span>可用性</span>
                <span>违约项</span>
                <span>状态</span>
              </div>
              {metricsQuery.isLoading ? (
                <Skeleton rows={5} />
              ) : metricsQuery.isError ? (
                <ErrorState error={metricsQuery.error} onRetry={metricsQuery.refetch} />
              ) : sloEndpointRows.length === 0 ? (
                <EmptyState title="暂无 SLO 数据" description="等待 endpoint 指标采集" />
              ) : (
                sloEndpointRows.map((row) => (
                  <div className="obs-slo-row" key={row.name}>
                    <strong title={row.name}>{row.name}</strong>
                    <span>{compact(row.requests, 0)}</span>
                    <span className={row.tone === "danger" ? "danger-text" : undefined}>{row.p95}</span>
                    <span>{row.ttft}</span>
                    <span>{row.errorRate}</span>
                    <span>{row.availability}</span>
                    <span title={row.violations}>{row.violations}</span>
                    <StatusBadge status={row.status} />
                  </div>
                ))
              )}
            </div>
          </section>
        ) : null}

        {showLogs ? (
          <section className="infra-panel obs-log-panel">
            <PanelHeader title="日志检索" action={logsQuery.data ? `${logsQuery.data.count} 条 · ${logsQuery.data.source === "kubernetes" ? `${logNamespace}/${logPod}${logContainer ? `/${logContainer}` : ""}` : "平台事件索引"}` : `近 ${historyMinutes} 分钟`} />
            {logsQuery.isLoading ? <Skeleton rows={6} /> : logsQuery.isError ? (
              <ErrorState error={logsQuery.error} onRetry={logsQuery.refetch} />
            ) : !logsQuery.data?.logs.length ? (
              <EmptyState title={logMode === "kubernetes" && !logPod ? "请选择 Pod" : "暂无匹配日志"} description={logMode === "kubernetes" ? (logPodsQuery.isError ? "集群 Pod 列表读取失败，请检查 Kubernetes 连接。" : `未读取到 ${logNamespace}/${logPod || "所选 Pod"} 的匹配日志。`) : "当前时间范围内没有符合条件的平台操作或运行事件。Pod 标准输出请切换到 Kubernetes Pod。"} />
            ) : (
              <div className="obs-log-table">
                <div className="obs-log-row header"><span>时间</span><span>级别</span><span>来源</span><span>资源</span><span>消息</span></div>
                {logsQuery.data.logs.map((log) => <div className="obs-log-row" key={log.id}>
                  <time>{log.timestamp ? relativeTime(log.timestamp) : "实时"}</time>
                  <StatusBadge status={log.level === "error" ? "严重" : log.level === "warning" ? "警告" : "正常"} />
                  <strong>{log.source}</strong>
                  <span title={`${log.resource_type}/${log.resource_id}`}>{log.resource_id || log.resource_type || "-"}</span>
                  <code title={log.message}>{log.message}</code>
                </div>)}
              </div>
            )}
          </section>
        ) : null}

        {showAlerts ? (
          <section className="infra-panel obs-alert-panel">
            <PanelHeader
              title={`实时告警${alertsQuery.data ? ` (${alertsQuery.data.alerts.length})` : ""}`}
              action={alertsQuery.data ? `${alertsQuery.data.summary.critical ?? 0} 严重 · ${alertsQuery.data.summary.warning ?? 0} 警告` : "规则评估"}
            />
            <div className="obs-alert-table">
              <div className="obs-alert-row header">
                <span>级别</span>
                <span>告警名称</span>
                <span>服务</span>
                <span>实例</span>
                <span>指标</span>
                <span>当前值</span>
                <span>阈值</span>
                <span>持续时间</span>
                <span>发生时间</span>
                <span>操作</span>
              </div>
              {alertsQuery.isLoading ? (
                <Skeleton rows={3} />
              ) : alertsQuery.isError ? (
                <ErrorState error={alertsQuery.error} onRetry={alertsQuery.refetch} />
              ) : (alertsQuery.data?.alerts.length ?? 0) === 0 ? (
                <EmptyState title="暂无触发的告警" description="当前指标均在阈值内" />
              ) : (
                alertsQuery.data!.alerts.map((alert) => (
                  <div className="obs-alert-row" key={alert.id}>
                    <StatusBadge status={alert.severity} />
                    <strong>{alert.name}</strong>
                    <span>{alert.service}</span>
                    <span>{alert.instance}</span>
                    <span>{alert.metric}</span>
                    <strong className={alert.severity === "critical" ? "danger-text" : "warning-text"}>{alert.value}</strong>
                    <span>{alert.threshold}</span>
                    <span>—</span>
                    <span>{relativeTime(alert.triggered_at)}</span>
                    <button onClick={() => goTo("aiOps")} type="button">处理</button>
                  </div>
                ))
              )}
            </div>
          </section>
        ) : null}
      </div>
    </section>
  );
}

function deriveServiceRows(metrics: Metrics | null, groupBy: ObsGroupBy, metric: ObsMetricFilter): ServiceRow[] {
  const source = groupBy === "实例" ? metrics?.target_pod_stats : metrics?.endpoint_stats;
  if (!source?.length) return [];
  return source
    .map((row): ServiceRow => ({
      name: row.name,
      qps: row.request_count,
      p95: row.p95_latency_ms ?? row.mean_latency_ms ?? 0,
      ttft: row.p95_ttft_ms ?? row.mean_ttft_ms ?? 0,
      error: row.error_rate,
      availability: Math.max(0, 100 - row.error_rate * 100),
      outputTokens: row.output_tokens,
      tone: row.error_rate > 0.02 ? "danger" : (row.p95_latency_ms ?? 0) > 250 ? "warning" : "info",
      values: [],
    }))
    .sort((a, b) => metricSortValue(b, metric) - metricSortValue(a, metric))
    .slice(0, 6);
}

function metricSortValue(row: ServiceRow, metric: ObsMetricFilter): number {
  if (metric === "TTFT") return row.ttft;
  if (metric === "错误率") return row.error;
  if (metric === "吞吐") return row.outputTokens;
  return row.p95;
}

function uniqueStrings(values: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    const clean = value.trim();
    if (!clean || seen.has(clean)) continue;
    seen.add(clean);
    out.push(clean);
  }
  return out;
}

function withCurrentOption(options: string[], current: string): string[] {
  const unique = uniqueStrings(options);
  return current && !unique.includes(current) ? [...unique, current] : unique;
}

function matchesSearch(values: Array<string | number | null | undefined>, keyword: string): boolean {
  if (!keyword) return true;
  return values.some((value) => String(value ?? "").toLowerCase().includes(keyword));
}

function deriveContainerRows(metrics: Metrics | null, query: string, instanceFilter: string): ContainerRow[] {
  const keyword = query.trim().toLowerCase();
  const rows = metrics?.cadvisor?.containers ?? [];
  return rows
    .filter((item) => {
      const matchesInstance = instanceFilter === ALL_INSTANCES || [item.pod, item.container, item.name].some((value) => String(value || "") === instanceFilter);
      return matchesInstance && matchesSearch([item.namespace, item.pod, item.container, item.name, item.image], keyword);
    })
    .slice(0, 8)
    .map((item, index) => {
      const rx = item.network_rx_bytes_per_second;
      const tx = item.network_tx_bytes_per_second;
      const hasNetwork = typeof rx === "number" || typeof tx === "number";
      return {
        id: `${item.namespace}/${item.pod}/${item.container}/${index}`,
        pod: item.pod ? `${item.namespace || "default"}/${item.pod}` : item.name || item.container || "container",
        container: item.container || "container",
        cpu: typeof item.cpu_cores === "number" ? `${fmt(item.cpu_cores, 2)}` : "采样中",
        memory: bytes(item.memory_working_set_bytes),
        network: hasNetwork ? `↓${byteRate(rx ?? 0)} ↑${byteRate(tx ?? 0)}` : "采样中",
        image: item.image || "—",
      };
    });
}

function GpuMeter({ detail, label, value }: { detail?: string; label: string; value: number }) {
  const safe = Math.max(0, Math.min(100, value || 0));
  return (
    <div className="gpu-mini-meter">
      <span>{label}</span>
      <div className="route-track"><i style={{ width: `${safe}%` }} /></div>
      <strong>{detail ?? `${fmt(safe, 0)}%`}</strong>
    </div>
  );
}

function deriveSlowRequests(traces: RequestTrace[] | undefined): SlowRow[] {
  if (!traces?.length) return [];
  return traces
    .slice()
    .sort((a, b) => (b.total_ms ?? b.generation_ms ?? 0) - (a.total_ms ?? a.generation_ms ?? 0))
    .slice(0, 5)
    .map((trace) => ({
      endpoint: trace.endpoint_id || "/v1/chat/completions",
      service: trace.target_pod || "未绑定实例",
      p95: `${fmt(trace.total_ms ?? trace.generation_ms ?? 0, 0)}ms`,
      ratio: "实时",
    }));
}

function deltaOf(series: number[]): { delta: string; deltaTone: "up" | "down" | "flat" } {
  if (series.length < 2) return { delta: "", deltaTone: "flat" };
  const change = ((series[series.length - 1] - series[0]) / (series[0] || 1)) * 100;
  return {
    delta: `${change >= 0 ? "+" : ""}${change.toFixed(1)}%`,
    deltaTone: change > 0 ? "up" : change < 0 ? "down" : "flat",
  };
}

type LatLine = { id: string; label: string; tone: ObsTone; values: Array<number | null> };
type LatencyLabel = { label: string; index: number };
type TrendSlo = { value: number; label: string };
type LatencyChartData = {
  title: string;
  unit: string;
  lines: LatLine[];
  labels: LatencyLabel[];
  slo: TrendSlo | null;
  sampleCount: number;
  percentilesOverlap: boolean;
};

// 由 /api/metrics/history 派生当前筛选指标的趋势（旧→新）。
function buildMetricChart(samples: MetricsHistorySample[], metric: ObsMetricFilter): LatencyChartData {
  const metricConfig = trendConfig(metric);
  const valid = [...samples].reverse().filter((s) => metricConfig.defs.some((d) => typeof d.sel(s.metrics) === "number"));
  const lines = metricConfig.defs
    .map((d) => ({ id: d.id, label: d.label, tone: d.tone, values: valid.map((s) => {
      const value = d.sel(s.metrics);
      return typeof value === "number" ? value : null;
    }) }))
    .filter((line) => line.values.filter((value) => value != null).length >= 2);
  const percentilesOverlap = lines.length === 3 && valid.length > 0 && valid.every((_, index) => {
    const values = lines.map((line) => line.values[index]).filter((value): value is number => value != null);
    return values.length === 3 && values.every((value) => value === values[0]);
  });
  const sampleCount = Math.max(0, ...valid.map((sample) => sample.metrics.request_count ?? 0));
  return { title: metricConfig.title, unit: metricConfig.unit, lines, labels: pickTimeLabels(valid), slo: metricConfig.slo, sampleCount, percentilesOverlap };
}

function trendConfig(metric: ObsMetricFilter): {
  title: string;
  unit: string;
  slo: TrendSlo | null;
  defs: Array<{ id: string; label: string; tone: ObsTone; sel: (m: Partial<Metrics>) => number | null | undefined }>;
} {
  if (metric === "TTFT") {
    return {
      title: "TTFT 趋势 (P50 / P95 / P99)",
      unit: "ms",
      slo: { value: 200, label: "SLO (200ms)" },
      defs: [
        { id: "p50-ttft", label: "P50", tone: "success", sel: (m) => m.p50_ttft_ms },
        { id: "p95-ttft", label: "P95", tone: "warning", sel: (m) => m.p95_ttft_ms },
        { id: "p99-ttft", label: "P99", tone: "danger", sel: (m) => m.p99_ttft_ms },
      ],
    };
  }
  if (metric === "错误率") {
    return {
      title: "错误率趋势",
      unit: "%",
      slo: { value: 1, label: "SLO (1%)" },
      defs: [
        { id: "error-rate", label: "错误率", tone: "danger", sel: (m) => (typeof m.error_rate === "number" ? m.error_rate * 100 : null) },
      ],
    };
  }
  if (metric === "吞吐") {
    return {
      title: "吞吐趋势",
      unit: "tok/s",
      slo: null,
      defs: [
        { id: "tokens", label: "Tokens/s", tone: "info", sel: (m) => m.tokens_per_second },
      ],
    };
  }
  return {
    title: "延迟趋势 (P50 / P95 / P99)",
    unit: "ms",
    slo: { value: 300, label: "SLO (300ms)" },
    defs: [
      { id: "p50-latency", label: "P50", tone: "success", sel: (m) => m.p50_latency_ms },
      { id: "p95-latency", label: "P95", tone: "warning", sel: (m) => m.p95_latency_ms },
      { id: "p99-latency", label: "P99", tone: "danger", sel: (m) => m.p99_latency_ms },
    ],
  };
}

function pickTimeLabels(samples: MetricsHistorySample[]): LatencyLabel[] {
  if (!samples.length) return [];
  const fmtTime = (iso: string) => {
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? "" : d.toLocaleTimeString("zh-CN", { hour12: false, hour: "2-digit", minute: "2-digit" });
  };
  const n = samples.length;
  const count = Math.min(7, n);
  const out: LatencyLabel[] = [];
  for (let i = 0; i < count; i++) {
    const idx = Math.round((i * (n - 1)) / (count - 1 || 1));
    out.push({ label: fmtTime(samples[idx].created_at), index: idx });
  }
  return out;
}

function deriveCurrentValues(metrics: Metrics | null): Array<{ service: string; p95: string; slo: string; status: string }> {
  if (!metrics?.endpoint_stats?.length) return [];
  return metrics.endpoint_stats.slice(0, 6).map((e) => {
    const p95 = e.p95_latency_ms ?? e.mean_latency_ms ?? null;
    const breached = p95 != null && p95 > 300;
    return {
      service: e.name,
      p95: p95 == null ? "—" : `${fmt(p95, 0)}ms`,
      slo: "< 300ms",
      status: e.error_rate > 0.02 ? "异常" : breached ? "降级" : "正常",
    };
  });
}

function deriveSloObjectives(metrics: Metrics | null): SloObjective[] {
  const p95 = metrics?.p95_latency_ms ?? null;
  const ttft = metrics?.p95_ttft_ms ?? null;
  const errPct = metrics?.error_rate == null ? null : metrics.error_rate * 100;
  const availPct = errPct == null ? null : Math.max(0, 100 - errPct);
  return [
    sloObjective("latency", "P95 延迟", "< 300ms", p95, 300, "ms", false),
    sloObjective("ttft", "P95 TTFT", "< 200ms", ttft, 200, "ms", false),
    sloObjective("availability", "可用性", ">= 99.9%", availPct, 99.9, "%", true),
    sloObjective("error", "错误率", "<= 1.00%", errPct, 1, "%", false),
  ];
}

function sloObjective(
  id: string,
  label: string,
  target: string,
  current: number | null,
  threshold: number,
  unit: "ms" | "%",
  higherIsBetter: boolean
): SloObjective {
  if (current == null || Number.isNaN(current)) {
    return { id, label, target, current: "—", remaining: "等待数据", status: "未知", tone: "info" };
  }
  const ok = higherIsBetter ? current >= threshold : current <= threshold;
  const diff = higherIsBetter ? current - threshold : threshold - current;
  const currentText = unit === "ms" ? `${fmt(current, 0)}ms` : `${fmt(current, 2)}%`;
  const remaining = unit === "ms"
    ? `${diff >= 0 ? "余量" : "超出"} ${fmt(Math.abs(diff), 0)}ms`
    : `${diff >= 0 ? "余量" : "超出"} ${fmt(Math.abs(diff), 2)}%`;
  return {
    id,
    label,
    target,
    current: currentText,
    remaining,
    status: ok ? "达标" : "违约",
    tone: ok ? "success" : "danger",
  };
}

function deriveSloEndpointRows(metrics: Metrics | null): SloEndpointRow[] {
  if (!metrics?.endpoint_stats?.length) return [];
  return metrics.endpoint_stats.slice(0, 12).map((row) => {
    const p95 = row.p95_latency_ms ?? row.mean_latency_ms;
    const ttft = row.p95_ttft_ms ?? row.mean_ttft_ms;
    const errPct = row.error_rate * 100;
    const availability = Math.max(0, 100 - errPct);
    const violations: string[] = [];
    if (p95 != null && p95 > 300) violations.push("P95");
    if (ttft != null && ttft > 200) violations.push("TTFT");
    if (row.error_rate > 0.01) violations.push("错误率");
    if (availability < 99.9) violations.push("可用性");
    return {
      name: row.name,
      requests: row.request_count,
      p95: p95 == null ? "—" : `${fmt(p95, 0)}ms`,
      ttft: ttft == null ? "—" : `${fmt(ttft, 0)}ms`,
      errorRate: `${fmt(errPct, 2)}%`,
      availability: `${fmt(availability, 2)}%`,
      violations: violations.length ? violations.join(" / ") : "—",
      status: violations.length ? "违约" : "达标",
      tone: violations.length ? "danger" : "success",
    };
  });
}

function deriveErrorDist(metrics: Metrics | null): {
  totalCount: number;
  errorCount: number;
  errorRatePct: number;
  rows: Array<{ label: string; value: number; detail: string; tone: ObsTone }>;
} {
  const totalCount = metrics?.request_count ?? 0;
  const errorCount = metrics?.error_count ?? 0;
  const errorRatePct = metrics?.error_rate != null ? metrics.error_rate * 100 : 0;
  const rows = (metrics?.endpoint_stats ?? [])
    .filter((e) => e.error_count > 0)
    .sort((a, b) => b.error_count - a.error_count)
    .slice(0, 6)
    .map((e) => ({
      label: e.name,
      value: e.error_count,
      detail: `${fmt(e.error_rate * 100, 2)}%`,
      tone: (e.error_rate > 0.05 ? "danger" : "warning") as ObsTone,
    }));
  return { totalCount, errorCount, errorRatePct, rows };
}

function LatencyChart({ chart }: { chart: LatencyChartData }) {
  if (!chart.lines.length) {
    return <EmptyState title="暂无趋势历史" description="等待 metrics 样本累积（≥2 个采样点）" />;
  }
  const W = 760;
  const H = 188;
  const TOP = 18;
  const LEFT = 54;
  const RIGHT = 10;
  const plotW = W - LEFT - RIGHT;
  const pointCount = chart.lines[0]?.values.length ?? 1;
  const numericValues = chart.lines.flatMap((line) => line.values).filter((value): value is number => value != null);
  const maxRaw = Math.max(...numericValues, chart.slo?.value ?? 0, 1);
  const max = niceCeil(maxRaw);
  const ticks = yTicks(max, 5);
  const yOf = (v: number) => TOP + H - (v / max) * H;
  const xOf = (i: number, n: number) => LEFT + (n > 1 ? (i / (n - 1)) * plotW : 0);
  const sloY = chart.slo ? yOf(chart.slo.value) : null;
  return (
    <div className="obs-latency-chart">
      <div className="obs-chart-legend">
        {chart.lines.map((line) => (
          <span className={line.tone} key={line.id}><i />{line.label}</span>
        ))}
        {chart.slo ? <span className="neutral"><i />{chart.slo.label}</span> : null}
      </div>
      {chart.percentilesOverlap ? (
        <p className="obs-chart-notice">单个采样窗口最多 {chart.sampleCount} 个请求，P50、P95、P99 数值完全重合；图中使用不同线型区分。</p>
      ) : chart.sampleCount > 0 && chart.sampleCount < 20 ? (
        <p className="obs-chart-notice">单个采样窗口最多 {chart.sampleCount} 个请求，分位数仅供参考。</p>
      ) : null}
      <svg aria-label={chart.title} role="img" viewBox="0 0 760 230">
        <text className="obs-axis-title" x="0" y="13">{chart.unit}</text>
        <g className="obs-chart-grid">
          {ticks.map((tick) => {
            const y = yOf(tick);
            return (
              <g key={tick}>
                <line x1={LEFT} x2={W - RIGHT} y1={y} y2={y} />
                <text className="obs-y-label" x={LEFT - 10} y={y + 4}>{formatAxisTick(tick, max, chart.unit)}</text>
              </g>
            );
          })}
        </g>
        <line className="obs-axis-line" x1={LEFT} x2={W - RIGHT} y1={TOP + H} y2={TOP + H} />
        {sloY != null ? <line className="obs-slo-line" x1={LEFT} x2={W - RIGHT} y1={sloY} y2={sloY} /> : null}
        {[...chart.lines].reverse().map((line) => (
          <g className={`series-${line.id}`} key={line.id}>
            {lineSegments(line.values).map((segment, segmentIndex) => (
              <polyline
                className={`line-${line.tone}`}
                fill="none"
                key={`${line.id}-${segmentIndex}`}
                points={segment.map(({ value, index }) => `${xOf(index, line.values.length).toFixed(1)},${yOf(value).toFixed(1)}`).join(" ")}
              />
            ))}
            {line.values.map((value, index) => value == null ? null : (
              <circle className={`point-${line.tone}`} cx={xOf(index, line.values.length)} cy={yOf(value)} key={`${line.id}-point-${index}`} r="2.5">
                <title>{`${line.label}: ${fmt(value, chart.unit === "%" ? 2 : 0)}${chart.unit}`}</title>
              </circle>
            ))}
          </g>
        ))}
        <g className="obs-x-axis">
          {chart.labels.map((item) => (
            <text key={`${item.label}-${item.index}`} x={xOf(item.index, pointCount)} y="224">{item.label}</text>
          ))}
        </g>
      </svg>
    </div>
  );
}

function lineSegments(values: Array<number | null>): Array<Array<{ value: number; index: number }>> {
  const segments: Array<Array<{ value: number; index: number }>> = [];
  let current: Array<{ value: number; index: number }> = [];
  values.forEach((value, index) => {
    if (value == null) {
      if (current.length >= 2) segments.push(current);
      current = [];
      return;
    }
    current.push({ value, index });
  });
  if (current.length >= 2) segments.push(current);
  return segments;
}

function formatAxisTick(value: number, max: number, unit: string): string {
  if (unit === "%" && max <= 5) return fmt(value, 2);
  if (unit === "tok/s" && max < 10) return fmt(value, 1);
  return fmt(value, 0);
}

function niceCeil(value: number): number {
  if (value <= 0) return 1;
  const exp = Math.floor(Math.log10(value));
  const base = 10 ** exp;
  const n = value / base;
  const nice = n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10;
  return nice * base;
}

function yTicks(max: number, count: number): number[] {
  const steps = Math.max(2, count - 1);
  const out: number[] = [];
  for (let i = steps; i >= 0; i--) out.push((max / steps) * i);
  return out;
}
