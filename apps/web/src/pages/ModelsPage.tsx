import { useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Activity, BarChart3, Box, Eye, MoreVertical, Plus, RefreshCw, Rocket, Route, Search } from "lucide-react";

import { useGoToPage } from "../lib/useGoToPage";
import { useDeliveryContext } from "../lib/useDeliveryContext";
import { useModelRegistry } from "../lib/useModelRegistry";

import { KpiGrid, PageHeader, PanelHeader, StatusBadge } from "../components/common/PlatformPrimitives";
import { describeError, EmptyState, ErrorState, Skeleton } from "../components/common/FeedbackStates";
import { Drawer, DrawerField } from "../components/common/Drawer";
import { ModelRegistryPanel } from "../components/registry/ModelRegistryPanel";
import { RegisterVersionDrawer } from "../components/registry/RegisterVersionDrawer";
import { normalizeServiceInstance } from "../data/mockPlatformData";
import { api } from "../lib/api";
import type { Metrics, ModelRegistryRow, ServiceInstance } from "../types/platform";
import type { RegisteredModelVersion } from "../types/registry";
import type { KpiItem } from "../types/ui";

function isRouter(item: ServiceInstance): boolean {
  return item.kind === "auto_router" || item.kind === "client_round_robin" || item.kind === "aibrix" || item.routing_role === "gateway" || item.routing_role === "auto_router";
}

export function ModelsPage() {
  const goTo = useGoToPage();
  const { context, update } = useDeliveryContext();
  const queryClient = useQueryClient();
  const [checking, setChecking] = useState<string | null>(null);
  const [routeTarget, setRouteTarget] = useState<ServiceInstance | null>(null);
  const [creating, setCreating] = useState(false);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [modelTab, setModelTab] = useState("模型版本");
  const registry = useModelRegistry();

  const instancesQuery = useQuery({
    queryKey: ["service-instances"],
    queryFn: () => api<{ instances: ServiceInstance[] }>("/api/service-instances"),
    select: (payload) => payload.instances.map(normalizeServiceInstance)
  });

  const metricsQuery = useQuery({
    queryKey: ["metrics", "current"],
    queryFn: () => api<Metrics>("/api/metrics/current"),
    refetchInterval: 5000
  });

  async function healthcheck(instance: ServiceInstance) {
    setChecking(instance.name);
    try {
      const result = await api<{ status?: string }>(`/api/service-instances/${encodeURIComponent(instance.name)}/healthcheck`, { method: "POST" });
      await queryClient.invalidateQueries({ queryKey: ["service-instances"] });
      const status = result?.status ?? "已完成";
      if (status === "healthy") toast.success(`${instance.name} 健康检查通过`);
      else toast.warning(`${instance.name} 健康检查：${status}`);
    } catch (error) {
      toast.error(`${instance.name} 健康检查失败`, { description: describeError(error) });
    } finally {
      setChecking(null);
    }
  }

  const rows = useMemo(() => deriveModelRows(instancesQuery.data), [instancesQuery.data]);
  const typeOptions = useMemo(() => Array.from(new Set(rows.map((row) => row.type))).sort(), [rows]);
  const statusOptions = useMemo(() => Array.from(new Set(rows.map((row) => row.status))).sort(), [rows]);
  const filteredRows = useMemo(() => {
    const keyword = search.trim().toLowerCase();
    return rows.filter((row) =>
      (typeFilter === "all" || row.type === typeFilter) &&
      (statusFilter === "all" || row.status === statusFilter) &&
      (!keyword || `${row.name} ${row.description} ${row.type} ${row.version}`.toLowerCase().includes(keyword))
    );
  }, [rows, search, typeFilter, statusFilter]);
  const totalPages = Math.max(1, Math.ceil(filteredRows.length / pageSize));
  const currentPage = Math.min(page, totalPages);
  const pageRows = filteredRows.slice((currentPage - 1) * pageSize, currentPage * pageSize);
  const kpis = useMemo(
    () => buildModelKpis(instancesQuery.data, metricsQuery.data, registry.list.data?.versions),
    [instancesQuery.data, metricsQuery.data, registry.list.data]
  );

  return (
    <section className="infra-page models-page models-replica">
      <PageHeader
        title="模型与版本"
        subtitle="管理原始模型、LoRA 微调产物、版本血缘、产物归档与生产运行时绑定"
        actions={
          <div className="models-header-actions">
            <button className="console-refresh" disabled={!context.modelVersionId} title="发布当前推理模型版本" onClick={() => goTo("release", { deliveryKind: "inference" })} type="button"><Rocket size={14} /> 发布当前版本</button>
            <button className="console-refresh primary" onClick={() => setCreating(true)} type="button"><Plus size={14} /> 注册版本</button>
          </div>
        }
      />

      <KpiGrid className="models-kpi-strip" items={kpis} />

      <section className="infra-panel models-main-panel">
        <div className="models-tabs">
          {["模型版本", "运行时绑定"].map((t) => (
            <button className={modelTab === t ? "active" : undefined} key={t} onClick={() => setModelTab(t)} type="button">{t}</button>
          ))}
        </div>
        {modelTab === "模型列表" && (
        <>
        <div className="models-toolbar">
          <select className="models-filter" onChange={(event) => { setTypeFilter(event.target.value); setPage(1); }} value={typeFilter}>
            <option value="all">全部类型</option>
            {typeOptions.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
          </select>
          <select className="models-filter" onChange={(event) => { setStatusFilter(event.target.value); setPage(1); }} value={statusFilter}>
            <option value="all">全部状态</option>
            {statusOptions.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
          </select>
          <label className="models-search-box"><Search size={14} /> <input onChange={(event) => {
            setSearch(event.target.value);
            setPage(1);
          }} placeholder="搜索模型名称、描述..." value={search} /></label>
          <button type="button" onClick={() => instancesQuery.refetch()}><RefreshCw size={13} /></button>
          <button className="primary" onClick={() => setCreating(true)} type="button"><Plus size={13} /> 注册模型</button>
        </div>

        <div className="models-table">
          <div className="models-table-row header">
            <span>模型名称</span>
            <span>类型</span>
            <span>最新版本</span>
            <span>状态</span>
            <span>环境</span>
            <span>质量评分</span>
            <span>运行实例</span>
            <span>更新时间</span>
            <span>操作</span>
          </div>
          {instancesQuery.isLoading && rows.length === 0 ? (
            <Skeleton rows={5} />
          ) : instancesQuery.isError && rows.length === 0 ? (
            <ErrorState error={instancesQuery.error} onRetry={instancesQuery.refetch} />
          ) : filteredRows.length === 0 ? (
            <EmptyState title="暂无模型" description={search ? "无匹配结果" : "service_instances 目录为空"} />
          ) : (
          pageRows.map((row) => (
            <div className="models-table-row" key={row.id}>
              <span className="models-name-cell">
                <Box size={16} />
                <span>
                  <strong>{row.name}</strong>
                  <small>{row.description}</small>
                </span>
              </span>
              <span><em className="models-type-pill">{row.type}</em></span>
              <span className="models-version-cell">
                <strong>{row.version}</strong>
                <small>最新</small>
              </span>
              <StatusBadge status={row.status} />
              <span>{row.env}</span>
              <span className="models-quality-cell">
                {row.quality === null ? <strong>-</strong> : <><strong>{row.quality.toFixed(1)}</strong><i style={{ width: `${row.quality}%` }} /></>}
              </span>
              <span>{row.instances}</span>
              <span>{row.updatedAt}</span>
              <span className="models-row-actions">
                <button onClick={() => goTo("observability")} type="button" title="查看"><Eye color="#2563eb" size={13} strokeWidth={2.5} /></button>
                <button disabled={!row.source || checking === row.source.name} onClick={() => row.source && void healthcheck(row.source)} type="button" title="健康检查">
                  <Activity color="#2563eb" size={13} strokeWidth={2.5} />
                </button>
                <button disabled={!row.source || !isRouter(row.source)} onClick={() => row.source && setRouteTarget(row.source)} type="button" title="路由策略">
                  <Route color="#2563eb" size={13} strokeWidth={2.5} />
                </button>
                <button onClick={() => goTo("observability")} type="button" title="趋势"><BarChart3 color="#2563eb" size={13} strokeWidth={2.5} /></button>
                <button onClick={() => goTo("observability")} type="button" title="更多"><MoreVertical color="#2563eb" size={13} strokeWidth={2.5} /></button>
              </span>
            </div>
          )))}
        </div>

        <div className="models-pagination">
          <span>共 {filteredRows.length} 项</span>
          <button disabled={currentPage === 1} onClick={() => setPage((value) => Math.max(1, value - 1))} type="button">‹</button>
          {Array.from({ length: totalPages }, (_, index) => index + 1).map((item) => (
            <button className={item === currentPage ? "active" : undefined} key={item} onClick={() => setPage(item)} type="button">{item}</button>
          ))}
          <button disabled={currentPage === totalPages} onClick={() => setPage((value) => Math.min(totalPages, value + 1))} type="button">›</button>
          <button onClick={() => { setPageSize((value) => (value === 10 ? 20 : value === 20 ? 50 : 10)); setPage(1); }} type="button">{pageSize} 条/页</button>
        </div>
        </>
        )}

        {modelTab === "运行时绑定" && (
          <table className="infra-table models-tab-table">
            <thead><tr>{["模型 / 实例", "类型", "路由角色", "Endpoint", "状态"].map((c) => <th key={c}>{c}</th>)}</tr></thead>
            <tbody>
              {(instancesQuery.data ?? []).length === 0 ? (
                <tr><td colSpan={5}><EmptyState title="暂无运行时实例" description="service_instances 目录为空" /></td></tr>
              ) : (instancesQuery.data ?? []).map((inst) => (
                <tr key={inst.name}>
                  <td><strong>{inst.name}</strong></td>
                  <td>{inst.kind}</td>
                  <td>{inst.routing_role || "-"}</td>
                  <td><span className="cell-subtle" style={{ marginTop: 0 }}>{inst.base_url}</span></td>
                  <td><StatusBadge status={inst.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {modelTab === "模型版本" && <ModelRegistryPanel
          registry={registry}
          selectedVersionID={context.modelVersionId}
          onSelect={(version) => update({ modelId: version.model_id, modelVersionId: version.id })}
          onBenchmark={(version) => goTo("benchmarks", { deliveryKind: "inference", modelId: version.model_id, modelVersionId: version.id, trainingJobId: null })}
          onRelease={(version) => goTo("release", { deliveryKind: "inference", modelId: version.model_id, modelVersionId: version.id, trainingJobId: null })}
        />}
      </section>

      <RoutingPolicyDrawer instance={routeTarget} onClose={() => setRouteTarget(null)} />
      {creating ? <RegisterVersionDrawer registry={registry} onClose={() => setCreating(false)} /> : null}
    </section>
  );
}

function deriveModelRows(instances: ServiceInstance[] | undefined): ModelRegistryRow[] {
  if (!instances?.length) return [];
  return instances.slice(0, 8).map((item) => ({
    id: item.name,
    name: item.name,
    description: item.kind === "vllm" ? "vLLM 推理服务模型" : item.kind === "aibrix" ? "AIBrix 网关模型绑定" : "运行时模型实例",
    type: item.model_id?.includes("embed") ? "Embedding" : "LLM",
    version: item.model_id || "—",
    status: item.status === "healthy" ? "运行中" : "告警",
    env: "prod",
    quality: null, // 质量评分暂无后端来源（待评测端点）
    instances: item.kind === "vllm" ? 1 : 0,
    updatedAt: "实时",
    source: item,
  }));
}

function buildModelKpis(instances: ServiceInstance[] | undefined, metrics: Metrics | undefined, registryVersions: RegisteredModelVersion[] | undefined): KpiItem[] {
  const list = instances ?? [];
  const versions = registryVersions ?? [];
  // 真实注册中心数据：模型数 = 不同 model_id；活跃版本 = status=active。
  const registeredModels = new Set(versions.map((v) => v.model_id)).size;
  const activeVersions = versions.filter((v) => v.status === "active" || v.status === "serving").length;
  const healthy = list.filter((item) => item.status === "healthy").length;
  const totalInstances = metrics?.service_instances?.length ?? list.length;
  const types = new Set(list.map((item) => item.kind)).size;
  const compliance = list.length ? (healthy / list.length) * 100 : null;
  return [
    { id: "total", label: "注册模型", value: String(registeredModels), detail: `${versions.length} 个版本`, trend: [] },
    { id: "versions", label: "活跃版本", value: String(activeVersions), detail: "active / serving", trend: [], tone: activeVersions ? "success" : undefined },
    { id: "instances", label: "运行实例", value: String(totalInstances), detail: "绑定实例总数", trend: [] },
    { id: "types", label: "模型类型", value: String(types), detail: "Runtime 类型", trend: [] },
    { id: "compliance", label: "合规检查", value: compliance == null ? "—" : `${compliance.toFixed(1)}%`, detail: "健康占比", trend: [], tone: compliance != null && compliance < 90 ? "warning" : "success" },
  ];
}

function RoutingPolicyDrawer({ instance, onClose }: { instance: ServiceInstance | null; onClose: () => void }) {
  if (!instance) return null;
  const meta = instance.metadata ?? {};
  const strategies = Array.isArray(meta.routing_strategies)
    ? meta.routing_strategies
    : meta.routing_strategies
      ? [String(meta.routing_strategies)]
      : [];
  const kindLabel =
    instance.kind === "auto_router" ? "自动路由器"
      : instance.kind === "client_round_robin" ? "客户端轮询"
        : instance.kind === "aibrix" ? "AIBrix 网关"
          : instance.kind;
  return (
    <Drawer open={!!instance} onClose={onClose} title={instance.name} subtitle={`路由策略 · ${kindLabel}`}>
      <div className="infra-drawer-section-title">类型</div>
      <DrawerField label="kind" value={instance.kind} />
      <DrawerField label="路由角色" value={instance.routing_role || "-"} />

      <div className="infra-drawer-section-title">路由策略</div>
      {strategies.length > 0 ? (
        strategies.map((s) => <DrawerField key={s} label="strategy" value={s} />)
      ) : (
        <DrawerField label="strategy" value="未配置（使用默认 least-request）" />
      )}

      {Array.isArray(meta.candidate_endpoints) && meta.candidate_endpoints.length > 0 && (
        <>
          <div className="infra-drawer-section-title">候选 Endpoint</div>
          {meta.candidate_endpoints.map((ep) => <DrawerField key={ep} label="candidate" value={ep} />)}
        </>
      )}

      {Array.isArray(meta.target_instances) && meta.target_instances.length > 0 && (
        <>
          <div className="infra-drawer-section-title">轮询目标</div>
          {meta.target_instances.map((t) => <DrawerField key={t} label="target" value={t} />)}
        </>
      )}
    </Drawer>
  );
}
