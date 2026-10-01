import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Boxes, MoreVertical, RefreshCw, Scaling } from "lucide-react";

import { useGoToPage } from "../lib/useGoToPage";
import { useK8sScaling } from "../lib/useK8sScaling";

import { KpiGrid, PageHeader, PanelHeader, StatusBadge } from "../components/common/PlatformPrimitives";
import { EmptyState, ErrorState, Skeleton } from "../components/common/FeedbackStates";
import { WorkloadScaleDrawer, type ScaleTarget } from "../components/kubernetes/WorkloadScaleDrawer";
import { api } from "../lib/api";
import { bytes, fmt, shortTime } from "../lib/format";
import type { K8sDeployment, K8sEvent, K8sNode, K8sPod, KubernetesSnapshot, KubernetesWorkloadRow, Metrics } from "../types/platform";
import type { KpiItem } from "../types/ui";

type ResourceTone = "info" | "success" | "warning" | "danger";

export function KubernetesPage() {
  const goTo = useGoToPage();
  const [workloadFilter, setWorkloadFilter] = useState<"all" | "abnormal" | "warning" | "normal">("all");
  const [eventFilter, setEventFilter] = useState<"all" | "warning" | "error">("all");
  const [workloadSearch, setWorkloadSearch] = useState("");
  const [podFilter, setPodFilter] = useState<"all" | "abnormal" | "running">("all");
  const [podSearch, setPodSearch] = useState("");
  const [scaleTarget, setScaleTarget] = useState<ScaleTarget | null>(null);
  const scaling = useK8sScaling();
  const snapshotQuery = useQuery({
    queryKey: ["kubernetes", "snapshot"],
    queryFn: () => api<KubernetesSnapshot>("/api/kubernetes/snapshot"),
    refetchInterval: 15000
  });
  const metricsQuery = useQuery({
    queryKey: ["metrics", "current"],
    queryFn: () => api<Metrics>("/api/metrics/current"),
    refetchInterval: 5000
  });

  const snapshot = snapshotQuery.data;
  const pods = snapshot?.pods ?? [];
  const deployments = snapshot?.deployments ?? [];
  const events = snapshot?.events ?? [];
  const nodes = snapshot?.nodes ?? [];
  const hpas = snapshot?.hpas ?? [];
  const available = snapshot?.available ?? false;
  const disabled = snapshot?.disabled ?? false;
  // A2：写能力由后端 feature flag 决定；前端据此灰显/启用扩缩容控件（后端仍强约束）。
  const writesEnabled = snapshot?.writes_enabled ?? false;
  const writeNamespaces = snapshot?.write_namespaces ?? [];
  const canWrite = (namespace: string, name: string) => writesEnabled && isWritable(namespace, name, writeNamespaces);
  const hpaFor = (namespace: string, name: string) =>
    hpas.find((h) => h.namespace === namespace && h.target_name === name) ?? null;
  const openScaleFor = (namespace: string, name: string) => {
    const dep = deployments.find((d) => d.namespace === namespace && d.name === name);
    const desired = dep?.desired ?? hpaFor(namespace, name)?.desired_replicas ?? 1;
    setScaleTarget({ namespace, name, desired, ready: dep?.ready ?? "—" });
  };

  const workloadRows = useMemo(() => deriveWorkloads(deployments, pods), [deployments, pods]);
  const eventRows = useMemo(() => deriveEvents(events), [events]);
  const nodeRows = useMemo(() => deriveNodeRows(nodes, pods), [nodes, pods]);
  const resourceMeters = useMemo(() => deriveResourceMeters(metricsQuery.data), [metricsQuery.data]);
  const filteredWorkloadRows = useMemo(
    () => workloadRows.filter((row) => {
      const matchesFilter = workloadFilter === "all"
        || (workloadFilter === "abnormal" && row.status === "异常")
        || (workloadFilter === "warning" && row.status === "警告")
        || (workloadFilter === "normal" && row.status === "正常");
      const keyword = workloadSearch.trim().toLowerCase();
      const matchesSearch = !keyword || `${row.name} ${row.kind} ${row.namespace}`.toLowerCase().includes(keyword);
      return matchesFilter && matchesSearch;
    }),
    [workloadRows, workloadFilter, workloadSearch]
  );
  const filteredEventRows = useMemo(
    () => eventRows.filter((event) => {
      if (eventFilter === "all") return true;
      if (eventFilter === "warning") return event.type === "Warning" || event.type === "警告";
      return event.type === "Error" || event.type === "错误";
    }),
    [eventRows, eventFilter]
  );
  const filteredPods = useMemo(() => {
    const keyword = podSearch.trim().toLowerCase();
    return [...pods]
      .sort((a, b) => podRiskRank(a) - podRiskRank(b) || b.restarts - a.restarts || a.namespace.localeCompare(b.namespace) || a.name.localeCompare(b.name))
      .filter((pod) => {
        const healthy = isPodHealthy(pod);
        const matchesFilter = podFilter === "all" || (podFilter === "running" && healthy) || (podFilter === "abnormal" && !healthy);
        const matchesSearch = !keyword || `${pod.name} ${pod.namespace} ${pod.node} ${pod.component}`.toLowerCase().includes(keyword);
        return matchesFilter && matchesSearch;
      });
  }, [podFilter, podSearch, pods]);
  const podCount = pods.length;
  const healthyPods = pods.filter(isPodHealthy).length;
  const abnormalPods = podCount - healthyPods;
  const completedPods = pods.filter((pod) => pod.phase === "Succeeded").length;
  const restartedPods = pods.filter((pod) => pod.restarts > 0).length;
  const deploymentCount = deployments.length;
  const namespaceCount = new Set(pods.map((pod) => pod.namespace).filter(Boolean)).size;
  const nodeCount = nodes.length;
  const readyNodes = nodes.filter((node) => node.status === "Ready").length;
  const warningEvents = eventRows.filter((event) => event.type === "Warning").length;
  const healthyWorkloads = workloadRows.filter((row) => row.status === "正常").length;

  const kpis: KpiItem[] = [
    { id: "health", label: "集群连接", value: available ? "正常" : disabled ? "未配置" : "不可用", detail: available ? `${namespaceCount} Namespace · ${deploymentCount} Deployment` : disabled ? "未提供 kubeconfig" : "集群 API 不可达", trend: [], tone: available ? "success" : "warning" },
    { id: "nodes", label: "Node Ready", value: nodeCount ? `${readyNodes} / ${nodeCount}` : "—", detail: readyNodes === nodeCount && nodeCount ? "全部可调度" : `${nodeCount - readyNodes} 个节点异常`, trend: [], tone: nodeCount && readyNodes === nodeCount ? "success" : nodeCount ? "danger" : undefined },
    { id: "pods", label: "Pod 健康", value: podCount ? `${healthyPods} / ${podCount}` : "—", detail: abnormalPods ? `${abnormalPods} 个异常 · ${restartedPods} 个有重启` : `${podCount - completedPods} 个运行 · ${completedPods} 个完成 · ${restartedPods} 个有重启`, trend: [], tone: abnormalPods ? "danger" : restartedPods ? "warning" : "success" },
    { id: "workloads", label: "工作负载", value: deploymentCount ? `${healthyWorkloads} / ${deploymentCount}` : "—", detail: warningEvents ? `${warningEvents} 个 Warning 事件` : "无 Warning 事件", trend: [], tone: healthyWorkloads < deploymentCount || warningEvents ? "warning" : "success" },
  ];

  return (
    <section className="infra-page kubernetes-page k8s-replica">
      <PageHeader
        title="集群与资源"
        subtitle="优先呈现集群风险与资源压力，并统一查看 Node、Deployment、Pod、HPA 和事件"
        actions={
          <button className="console-refresh" onClick={() => snapshotQuery.refetch()} type="button">
            <RefreshCw className={snapshotQuery.isFetching ? "spinning" : undefined} size={14} /> 刷新
          </button>
        }
      />

      <KpiGrid className="k8s-kpi-strip" items={kpis} />

      {!available && !snapshotQuery.isLoading ? (
        <section className="infra-panel">
          {snapshotQuery.isError ? (
            <ErrorState error={snapshotQuery.error} onRetry={() => snapshotQuery.refetch()} />
          ) : (
            <EmptyState
              title={disabled ? "Kubernetes 集成未配置" : "集群不可达"}
              description={snapshot?.error || (disabled ? "控制面未提供 kubeconfig（KUBECONFIG）" : "client-go 无法连接集群 API")}
            />
          )}
        </section>
      ) : null}

      <section className="infra-panel k8s-node-resource-panel">
          <PanelHeader title="Node 与资源压力" action={nodeCount ? `${readyNodes}/${nodeCount} Ready · Agent 指标` : "等待 Node 数据"} />
          <div className="k8s-resource-grid">
            {metricsQuery.isLoading ? (
              <Skeleton rows={2} />
            ) : (
              resourceMeters.map((resource) => (
                <ResourceMeter
                  detail={resource.detail}
                  key={resource.id}
                  label={resource.label}
                  tone={resource.tone}
                  value={resource.value}
                />
              ))
            )}
          </div>
          <div className="k8s-node-table">
            <div className="k8s-node-row header">
              <span>Node</span>
              <span>Ready</span>
              <span>角色 / 版本</span>
              <span>CPU 容量</span>
              <span>内存 GiB</span>
              <span>Pod</span>
              <span>GPU</span>
            </div>
            {snapshotQuery.isLoading ? (
              <Skeleton rows={2} />
            ) : nodeRows.length === 0 ? (
              <EmptyState title="暂无节点数据" />
            ) : (
              nodeRows.map((node) => (
                <div className="k8s-node-row" key={node.name}>
                  <span className="k8s-node-identity"><strong>{node.name}</strong><small>{node.ip}</small></span>
                  <StatusBadge status={node.status} />
                  <span className="k8s-node-platform"><strong>{node.roles || "worker"}</strong><small>{node.version}</small></span>
                  <span>{node.cpu}</span>
                  <span>{node.memGiB}</span>
                  <span>{node.podCount}</span>
                  <span>{node.gpu}</span>
                </div>
              ))
            )}
          </div>
      </section>

      <div className="k8s-mid-grid">
        <section className="infra-panel k8s-workload-panel">
          <PanelHeader title="Deployment 与弹性策略" action={`显示 ${filteredWorkloadRows.length} / ${workloadRows.length}`} />
          {available && !writesEnabled ? (
            <p className="k8s-write-hint">只读模式 · 设置 ALLOW_K8S_WRITES=true 并配置 K8S_WRITE_NAMESPACES 后可在行内扩缩容</p>
          ) : null}
          <div className="k8s-filter-row">
            <button className={workloadFilter === "all" ? "active" : undefined} onClick={() => setWorkloadFilter("all")} type="button">全部 ({workloadRows.length})</button>
            <button className={workloadFilter === "abnormal" ? "active" : undefined} onClick={() => setWorkloadFilter("abnormal")} type="button">异常 ({workloadRows.filter((row) => row.status === "异常").length})</button>
            <button className={workloadFilter === "warning" ? "active" : undefined} onClick={() => setWorkloadFilter("warning")} type="button">警告 ({workloadRows.filter((row) => row.status === "警告").length})</button>
            <button className={workloadFilter === "normal" ? "active" : undefined} onClick={() => setWorkloadFilter("normal")} type="button">正常 ({workloadRows.filter((row) => row.status === "正常").length})</button>
            <label>
              <input onChange={(event) => setWorkloadSearch(event.target.value)} placeholder="搜索 Deployment / StatefulSet..." value={workloadSearch} />
            </label>
          </div>
          <div className="k8s-workload-table">
            <div className="k8s-workload-row header">
              <span>工作负载</span>
              <span>Namespace</span>
              <span>副本</span>
              <span>可用性</span>
              <span>HPA</span>
              <span>状态</span>
              <span>操作</span>
            </div>
            {snapshotQuery.isLoading ? (
              <Skeleton rows={4} />
            ) : filteredWorkloadRows.length === 0 ? (
              <EmptyState title="暂无工作负载" description={workloadSearch ? "无匹配结果" : undefined} />
            ) : (
              filteredWorkloadRows.map((row) => (
                <div className="k8s-workload-row" key={row.id}>
                  <span className="k8s-workload-name">
                    <Boxes size={14} />
                    <span><strong>{row.name}</strong><small>{row.kind}</small></span>
                  </span>
                  <span>{row.namespace}</span>
                  <span>{row.replicas}</span>
                  <span>{row.availability}%</span>
                  <span className="k8s-hpa-summary">{hpaFor(row.namespace, row.name) ? `${hpaFor(row.namespace, row.name)!.min_replicas}–${hpaFor(row.namespace, row.name)!.max_replicas} · ${hpaFor(row.namespace, row.name)!.metric}` : "未配置"}</span>
                  <StatusBadge status={row.status} />
                  {canWrite(row.namespace, row.name) ? (
                    <button aria-label={`${row.name} 弹性伸缩`} onClick={() => openScaleFor(row.namespace, row.name)} title="弹性伸缩（扩缩副本 / 配 HPA）" type="button"><Scaling size={14} /></button>
                  ) : (
                    <button aria-label={`${row.name} 操作`} onClick={() => goTo("observability")} title="更多操作" type="button"><MoreVertical size={14} /></button>
                  )}
                </div>
              ))
            )}
          </div>
          {workloadFilter !== "all" || workloadSearch ? <button className="dashboard-link-button" onClick={() => {
            setWorkloadFilter("all");
            setWorkloadSearch("");
          }} type="button">清除筛选</button> : null}
        </section>

        <section className="infra-panel k8s-pod-panel">
          <PanelHeader title="Pod 运行清单" action={`显示 ${filteredPods.length} / ${pods.length}`} />
          <div className="k8s-filter-row">
            <button className={podFilter === "all" ? "active" : undefined} onClick={() => setPodFilter("all")} type="button">全部 ({pods.length})</button>
            <button className={podFilter === "abnormal" ? "active" : undefined} onClick={() => setPodFilter("abnormal")} type="button">异常 ({abnormalPods})</button>
            <button className={podFilter === "running" ? "active" : undefined} onClick={() => setPodFilter("running")} type="button">健康 ({healthyPods})</button>
            <label><input onChange={(event) => setPodSearch(event.target.value)} placeholder="搜索 Pod / Namespace / Node..." value={podSearch} /></label>
          </div>
          <div className="k8s-pod-table">
            <div className="k8s-pod-row header"><span>Pod</span><span>Namespace</span><span>Node</span><span>Ready</span><span>Phase</span><span>重启</span><span>Pod IP</span></div>
            {snapshotQuery.isLoading ? <Skeleton rows={5} /> : filteredPods.length === 0 ? <EmptyState title="暂无匹配 Pod" /> : filteredPods.map((pod) => (
              <div className="k8s-pod-row" key={`${pod.namespace}/${pod.name}`}>
                <span className="k8s-pod-name"><strong>{pod.name}</strong><small>{pod.component || "未标记组件"}</small></span>
                <span>{pod.namespace}</span><span>{pod.node || "未调度"}</span><span>{pod.ready || "—"}</span>
                <StatusBadge status={pod.phase} /><strong className={pod.restarts > 0 ? "warning-text" : undefined}>{pod.restarts}</strong><span>{pod.pod_ip || "—"}</span>
              </div>
            ))}
          </div>
        </section>
      </div>


      <section className="infra-panel k8s-event-panel">
        <PanelHeader title="集群事件与排障证据" action={`显示 ${filteredEventRows.length} / ${eventRows.length}`} />
        <div className="k8s-event-tabs">
          <button className={eventFilter === "all" ? "active" : undefined} onClick={() => setEventFilter("all")} type="button">全部</button>
          <button className={eventFilter === "warning" ? "active" : undefined} onClick={() => setEventFilter("warning")} type="button">Warning</button>
          <button className={eventFilter === "error" ? "active" : undefined} onClick={() => setEventFilter("error")} type="button">Error</button>
        </div>
        <div className="k8s-event-table">
          <div className="k8s-event-row header"><span>时间</span><span>类型</span><span>对象</span><span>消息</span></div>
          {snapshotQuery.isLoading ? <Skeleton rows={4} /> : filteredEventRows.length === 0 ? <EmptyState title="暂无匹配事件" /> : filteredEventRows.map((event) => (
            <div className="k8s-event-row" key={event.id}><span>{event.time}</span><StatusBadge status={event.type} /><span title={event.object}>{event.object}</span><strong title={event.message}>{event.message}</strong></div>
          ))}
        </div>
      </section>

      {scaleTarget ? (
        <WorkloadScaleDrawer
          target={scaleTarget}
          hpa={hpaFor(scaleTarget.namespace, scaleTarget.name)}
          scaling={scaling}
          onClose={() => setScaleTarget(null)}
        />
      ) : null}
    </section>
  );
}

// isWritable：前端镜像后端命名空间守卫（仅用于灰显控件；后端始终强约束）。
const PROTECTED_NS = ["kube-system", "kube-public", "kube-node-lease"];
const PROTECTED_SUBSTR = ["aibrix", "envoy", "vllm", "qwen"];
function isWritable(namespace: string, name: string, allowlist: string[]): boolean {
  const ns = namespace.toLowerCase();
  const target = name.toLowerCase();
  if (PROTECTED_NS.includes(ns)) return false;
  if (PROTECTED_SUBSTR.some((p) => ns.includes(p) || target.includes(p))) return false;
  return allowlist.some((w) => w.toLowerCase() === ns);
}

function deriveWorkloads(deployments: K8sDeployment[], pods: K8sPod[]): KubernetesWorkloadRow[] {
  return deployments.map((deployment) => {
    const matched = pods.filter((pod) => pod.namespace === deployment.namespace && pod.name.startsWith(deployment.name));
    const unavailable = matched.some((pod) => pod.phase !== "Running");
    const desired = deployment.desired || matched.length || 1;
    const available = deployment.available || matched.filter((pod) => pod.phase === "Running").length;
    return {
      id: `${deployment.namespace}/${deployment.name}`,
      name: deployment.name,
      kind: "Deployment",
      namespace: deployment.namespace,
      replicas: deployment.ready || `${available}/${desired}`,
      desired,
      availability: Math.round((available / Math.max(desired, 1)) * 100),
      status: unavailable || available < desired ? "异常" : "正常",
    };
  });
}

function deriveEvents(events: K8sEvent[]) {
  return events.slice(0, 20).map((event, index) => ({
    id: `${event.namespace}/${event.resource_name}/${index}`,
    time: event.event_time ? shortTime(event.event_time) : "-",
    type: event.type,
    object: `${event.resource_kind}/${event.resource_name}`,
    message: event.message || event.reason,
  }));
}

type NodeRow = { name: string; ip: string; status: string; roles: string; version: string; cpu: string; memGiB: string; podCount: number; gpu: string };

function deriveNodeRows(nodes: K8sNode[], pods: K8sPod[]): NodeRow[] {
  return nodes.map((node) => ({
    name: node.name,
    ip: node.internal_ip,
    status: node.status,
    roles: node.roles,
    version: node.version,
    cpu: node.cpu_capacity || "—",
    memGiB: node.memory_bytes ? fmt(node.memory_bytes / 1024 ** 3, 1) : "—",
    podCount: pods.filter((pod) => pod.node === node.name).length,
    gpu: node.gpu_capacity || "0",
  }));
}

function isPodHealthy(pod: K8sPod): boolean {
  if (pod.phase === "Succeeded") return true;
  if (pod.phase !== "Running") return false;
  const match = pod.ready?.match(/^(\d+)\/(\d+)$/);
  return !match || (Number(match[1]) > 0 && Number(match[1]) === Number(match[2]));
}

function podRiskRank(pod: K8sPod): number {
  if (!isPodHealthy(pod)) return 0;
  if (pod.restarts > 0) return 1;
  return pod.phase === "Running" ? 2 : 3;
}

type ResourceMeterRow = { id: string; label: string; value: number | null; detail: string; tone: ResourceTone };

// 节点资源用量由 metrics.host（Agent 采集）+ metrics.gpu 派生；无 Agent 时显示"等待采集"。
function deriveResourceMeters(metrics: Metrics | undefined): ResourceMeterRow[] {
  const host = metrics?.host;
  const cpuCount = host?.cpu?.count || null;
  const cpuUtil = host?.cpu?.usage_percent ?? null;
  const memUtil = host?.memory?.used_percent ?? null;
  const memTotal = host?.memory?.total_bytes || null;
  const disks = host?.disk ?? [];
  const root = disks.find((d) => d.path === "/") ?? [...disks].sort((a, b) => b.total_bytes - a.total_bytes)[0];
  const gpuCount = metrics?.gpu?.length ?? 0;
  const gpuUtil = gpuCount ? Math.max(...(metrics?.gpu ?? []).map((g) => g.gpu_utilization_percent)) : null;
  const gpuStatus = metrics?.gpu_status;
  const gpuDetail = gpuCount ? `${gpuCount}×GPU` : gpuStatus?.error ? "Agent GPU 采集异常" : "等待 GPU 指标";
  const tone = (v: number | null): ResourceTone => (v == null ? "info" : v >= 90 ? "danger" : v >= 75 ? "warning" : "success");
  return [
    { id: "cpu", label: "CPU", value: cpuUtil, tone: tone(cpuUtil), detail: cpuCount && cpuUtil != null ? `${fmt((cpuCount * cpuUtil) / 100, 1)} / ${cpuCount} Core` : cpuCount ? `— / ${cpuCount} Core` : "等待 Agent 采集" },
    { id: "memory", label: "内存", value: memUtil, tone: tone(memUtil), detail: memTotal ? `${bytes(host?.memory?.used_bytes)} / ${bytes(memTotal)}` : "等待 Agent 采集" },
    { id: "storage", label: "存储", value: root?.used_percent ?? null, tone: tone(root?.used_percent ?? null), detail: root ? `${bytes(root.used_bytes)} / ${bytes(root.total_bytes)}` : "等待 Agent 采集" },
    { id: "gpu", label: "GPU", value: gpuUtil, tone: gpuStatus?.error && !gpuCount ? "warning" : tone(gpuUtil), detail: gpuDetail },
  ];
}

function ResourceMeter({ detail, label, tone, value }: { detail: string; label: string; tone: ResourceTone; value: number | null }) {
  const known = value != null && !Number.isNaN(value);
  return (
    <div className="k8s-resource-meter">
      <span>{label}</span>
      <div><i className={tone} style={{ width: known ? `${value}%` : "0%" }} /></div>
      <strong>{known ? `${fmt(value as number, 0)}%` : "—"}</strong>
      <small>{detail}</small>
    </div>
  );
}
