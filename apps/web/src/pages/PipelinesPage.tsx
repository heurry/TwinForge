import { useMemo, useState, type ChangeEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  CheckCircle,
  Eye,
  MoreVertical,
  Plus,
  RefreshCw,
  RotateCcw,
  Search,
  XCircle,
} from "lucide-react";

import { Donut, KpiGrid, PageHeader, PanelHeader, StatusBadge } from "../components/common/PlatformPrimitives";
import { describeError, EmptyState, ErrorState, Skeleton } from "../components/common/FeedbackStates";
import { Drawer } from "../components/common/Drawer";
import { api } from "../lib/api";
import { compactMeta, fmt, relativeTime, shortTime } from "../lib/format";
import { cn } from "../lib/utils";
import { useGoToPage } from "../lib/useGoToPage";
import type { Deployment, DeploymentMeta, PipelineConsoleRow } from "../types/ops";
import type { KpiItem } from "../types/ui";

const RUNNING = "running";
type CIRun = { id: number; name: string; display_title: string; status: string; conclusion: string | null; html_url: string; head_branch: string; head_sha?: string; created_at: string };
type CIRunsResponse = { configured: boolean; provider: string; repository?: string; workflow?: string; default_ref?: string; message?: string; runs: CIRun[] };

export function PipelinesPage({ embedded = false }: { embedded?: boolean }) {
  const [creating, setCreating] = useState(false);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [envFilter, setEnvFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [timeFilter, setTimeFilter] = useState("all");
  const qc = useQueryClient();

  const deploymentsQuery = useQuery({
    queryKey: ["deployments"],
    queryFn: () => api<{ deployments: Deployment[] }>("/api/deployments"),
    refetchInterval: 8000
  });
  const ciQuery = useQuery({
    queryKey: ["ci", "runs"],
    queryFn: () => api<CIRunsResponse>("/api/ci/runs?limit=8"),
    refetchInterval: 15000
  });
  const ciMutation = useMutation({
    mutationFn: () => api("/api/ci/runs", { method: "POST", body: JSON.stringify({ ref: ciQuery.data?.default_ref || "main", operator: "frontend" }) }),
    onSuccess: () => {
      toast.success(`${ciProviderLabel(ciQuery.data?.provider)} 流水线已触发`);
      setTimeout(() => ciQuery.refetch(), 2000);
    },
    onError: (e) => toast.error(`CI 触发失败：${describeError(e)}`)
  });

  const invalidate = () => qc.invalidateQueries({ queryKey: ["deployments"] });

  const finishMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: "success" | "failed" }) =>
      api(`/api/deployments/${id}/finish`, { method: "POST", body: JSON.stringify({ status }) }),
    onSuccess: (_d, vars) => {
      toast.success(vars.status === "success" ? "已标记部署成功" : "已标记部署失败");
      invalidate();
    },
    onError: (e) => toast.error(`操作失败：${describeError(e)}`)
  });

  const rollbackMutation = useMutation({
    mutationFn: (id: string) => api(`/api/deployments/${id}/rollback`, { method: "POST" }),
    onSuccess: () => {
      toast.success("已触发回滚");
      invalidate();
    },
    onError: (e) => toast.error(`回滚失败：${describeError(e)}`)
  });

  const rows = useMemo(() => derivePipelineRows(deploymentsQuery.data?.deployments), [deploymentsQuery.data?.deployments]);
  const kpis = useMemo(() => buildPipelineKpis(deploymentsQuery.data?.deployments), [deploymentsQuery.data?.deployments]);
  const envDist = useMemo(() => deriveEnvDistribution(deploymentsQuery.data?.deployments), [deploymentsQuery.data?.deployments]);
  const busy = finishMutation.isPending || rollbackMutation.isPending;
  const envOptions = useMemo(() => Array.from(new Set(rows.map((row) => row.env))).sort(), [rows]);
  const statusOptions = useMemo(() => Array.from(new Set(rows.map((row) => row.status))).sort(), [rows]);
  const filteredRows = useMemo(() => {
    const q = search.trim().toLowerCase();
    const now = Date.now();
    const windowMs = timeFilter === "7d" ? 7 * 86_400_000 : timeFilter === "30d" ? 30 * 86_400_000 : 0;
    return rows.filter((row) => {
      if (envFilter !== "all" && row.env !== envFilter) return false;
      if (statusFilter !== "all" && row.status !== statusFilter) return false;
      if (windowMs > 0) {
        const ts = row.source.started_at ? Date.parse(row.source.started_at) : NaN;
        if (Number.isNaN(ts) || now - ts > windowMs) return false;
      }
      if (q && ![row.name, row.service, row.env, row.version, row.commit, row.status, row.stage, row.trigger]
        .some((value) => String(value).toLowerCase().includes(q))) return false;
      return true;
    });
  }, [rows, search, envFilter, statusFilter, timeFilter]);
  const totalPages = Math.max(1, Math.ceil(filteredRows.length / pageSize));
  const currentPage = Math.min(page, totalPages);
  const pageRows = filteredRows.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  return (
    <section className={embedded ? "pipelines-page pipeline-replica release-center-embedded" : "infra-page pipelines-page pipeline-replica"}>
      {!embedded ? <PageHeader
        title="CI/CD 流水线"
        subtitle="通用微服务的源码构建、测试、镜像制品和 Helm 校验；主入口已合并到发布中心"
        actions={
          <button className="console-refresh primary" onClick={() => setCreating(true)} type="button"><Plus size={14} /> 新建服务流水线</button>
        }
      /> : null}

      <KpiGrid className="pipeline-kpi-strip" items={kpis} />

      <div className="pipeline-main-grid">
        <section className="infra-panel pipeline-list-panel">
          <div className="pipeline-panel-title">
            <PanelHeader title="流水线列表" action={`${filteredRows.length} 条`} />
            <div className="pipeline-toolbar">
              <select className="pipeline-filter" onChange={(event) => { setEnvFilter(event.target.value); setPage(1); }} value={envFilter}>
                <option value="all">全部环境</option>
                {envOptions.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
              </select>
              <select className="pipeline-filter" onChange={(event) => { setStatusFilter(event.target.value); setPage(1); }} value={statusFilter}>
                <option value="all">全部状态</option>
                {statusOptions.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
              </select>
              <label className="pipeline-search"><Search size={14} />
                <input
                  onChange={(event) => {
                    setSearch(event.target.value);
                    setPage(1);
                  }}
                  placeholder="搜索流水线名称、服务..."
                  value={search}
                />
              </label>
              <select className="pipeline-filter" onChange={(event) => { setTimeFilter(event.target.value); setPage(1); }} value={timeFilter}>
                <option value="all">全部时间</option>
                <option value="7d">近 7 天</option>
                <option value="30d">近 30 天</option>
              </select>
              <button onClick={() => deploymentsQuery.refetch()} type="button"><RefreshCw size={13} /> 刷新</button>
              <button className="primary" onClick={() => setCreating(true)} type="button"><Plus size={13} /> 新建流水线</button>
            </div>
          </div>

          <div className="pipeline-table">
            <div className="pipeline-table-row header">
              <span>流水线名称</span>
              <span>环境</span>
              <span>版本 / 提交</span>
              <span>状态</span>
              <span>当前阶段</span>
              <span>耗时</span>
              <span>操作</span>
            </div>
            {deploymentsQuery.isLoading ? (
              <Skeleton rows={5} />
            ) : deploymentsQuery.isError ? (
              <ErrorState error={deploymentsQuery.error} onRetry={deploymentsQuery.refetch} />
            ) : filteredRows.length === 0 ? (
              <EmptyState title="暂无流水线" description={search ? "无匹配结果" : "点击「新建流水线」触发一次部署"} />
            ) : (
              pageRows.map((row) => (
                <PipelineTableRow
                  busy={busy}
                  key={row.id}
                  onFinish={(status) => finishMutation.mutate({ id: row.source.id, status })}
                  onRollback={() => rollbackMutation.mutate(row.source.id)}
                  row={row}
                />
              ))
            )}
          </div>

          <div className="pipeline-pagination">
            <span>共 {filteredRows.length} 条</span>
            <button disabled={currentPage === 1} onClick={() => setPage((value) => Math.max(1, value - 1))} type="button">‹</button>
            {Array.from({ length: totalPages }, (_, index) => index + 1).map((item) => (
              <button className={item === currentPage ? "active" : undefined} key={item} onClick={() => setPage(item)} type="button">{item}</button>
            ))}
            <button disabled={currentPage === totalPages} onClick={() => setPage((value) => Math.min(totalPages, value + 1))} type="button">›</button>
            <button onClick={() => { setPageSize((value) => (value === 10 ? 20 : value === 20 ? 50 : 10)); setPage(1); }} type="button">{pageSize} 条/页</button>
          </div>
        </section>

        <aside className="pipeline-side-stack">
          <section className="infra-panel">
            <PanelHeader title={`CI / ${ciProviderLabel(ciQuery.data?.provider)}`} action={ciQuery.data?.repository || "待配置"} />
            {ciQuery.isLoading ? <Skeleton rows={3} /> : ciQuery.isError ? <ErrorState error={ciQuery.error} onRetry={ciQuery.refetch} /> : !ciQuery.data?.configured ? (
              <>
                <EmptyState title="CI 连接未配置" description={ciQuery.data?.message || "配置 GitLab CI 连接后可从控制台触发并查看构建、测试结果。"} />
                <p className="pipeline-ci-scope">当前仓库：{ciQuery.data?.repository || "未设置"} · 分支：{ciQuery.data?.default_ref || "main"}<br />流水线定义在仓库的 <code>.gitlab-ci.yml</code>：执行 Go/Python/前端测试、sqlc 校验、Helm lint，并在主分支构建并推送 4 个容器镜像。它不负责本机 vLLM 模型发布；模型发布仍由“模型服务发布”完成。</p>
              </>
            ) : <>
              <button className="infra-action-btn" disabled={ciMutation.isPending} onClick={() => ciMutation.mutate()} type="button">
                <CheckCircle size={14} /> {ciMutation.isPending ? "触发中..." : `运行 ${ciQuery.data.default_ref || "main"} CI`}
              </button>
              <div className="pipeline-ci-list">
                {ciQuery.data.runs.length ? ciQuery.data.runs.map((run) => <a href={run.html_url} key={run.id} rel="noreferrer" target="_blank">
                  <i className={cn(run.conclusion === "success" ? "success" : run.status === "in_progress" || run.status === "queued" ? "warning" : "danger")} />
                  <span><strong>{run.display_title || run.name}</strong><small>{run.head_branch}{run.head_sha ? ` · ${run.head_sha.slice(0, 8)}` : ""} · {relativeTime(run.created_at)}</small></span>
                  <StatusBadge status={run.conclusion || run.status} />
                </a>) : <EmptyState title="暂无 CI 运行" />}
              </div>
              <p className="pipeline-ci-scope">GitLab CI 只负责执行仓库 <code>.gitlab-ci.yml</code>。实际部署内容由该文件的 job 决定；本平台的模型验收、vLLM 启停和发布门禁不由 CI 代替。</p>
            </>}
          </section>
          <section className="infra-panel">
            <PanelHeader title="环境分布" action={`${envDist.total} 条`} />
            {envDist.items.length === 0 ? (
              <EmptyState title="暂无部署" />
            ) : (
              <div className="pipeline-donut-block">
                <Donut value={envDist.total} max={envDist.total || 1} size={124} thickness={20} tone="info" label={`${envDist.total} 总流水线`} />
                <div className="pipeline-dist-list">
                  {envDist.items.map((item) => (
                    <div className={item.tone} key={item.label}>
                      <span><i />{item.label}</span>
                      <strong>{item.value}</strong>
                      <small>{item.percent}</small>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        </aside>
      </div>

      {creating ? <TriggerDeployDrawer onClose={() => setCreating(false)} /> : null}
    </section>
  );
}

function ciProviderLabel(provider?: string) {
  return provider === "github_actions" || provider === "github" ? "GitHub Actions" : "GitLab CI";
}

function PipelineTableRow({
  row,
  onFinish,
  onRollback,
  busy,
}: {
  row: PipelineConsoleRow;
  onFinish: (status: "success" | "failed") => void;
  onRollback: () => void;
  busy: boolean;
}) {
  const isRunning = normalizeStatus(row.source.status) === RUNNING || row.status === "运行中";
  const meta = row.source.metadata;
  const isRollout = meta.mode === "k8s_rollout";
  const goTo = useGoToPage();
  return (
    <div className="pipeline-table-row">
      <span className="pipeline-name-cell">
        <i className={cn("pipeline-status-dot", row.tone)} />
        <span>
          <strong>{row.name}</strong>
          <small>{row.service}</small>
        </span>
      </span>
      <span><em className={cn("pipeline-env-pill", row.env.toLowerCase())}>{row.env}</em></span>
      <span className="pipeline-version-cell">
        <strong>{row.version}</strong>
        <small>{row.commit}</small>
      </span>
      <span className="pipeline-status-cell">
        <StatusBadge status={row.status} />
        <small>{isRollout && meta.phase ? phaseLabel(meta.phase) : row.stageMeta}</small>
      </span>
      <span className="pipeline-stage-cell">
        {isRollout ? (
          <RolloutProgress meta={meta} running={isRunning} />
        ) : (
          <>
            <strong>{row.stage}</strong>
            {isRunning ? <em /> : null}
          </>
        )}
      </span>
      <span>{row.duration}</span>
      <span className="pipeline-row-actions">
        <button onClick={() => goTo("observability")} type="button" title="查看详情"><Eye color="#2563eb" size={14} strokeWidth={2.5} /></button>
        {isRollout ? (
          isRunning ? (
            <em className="pipeline-rollout-managed" title={meta.message}>自动发布中</em>
          ) : (
            <button onClick={() => goTo("observability")} type="button" title="更多"><MoreVertical color="#2563eb" size={14} strokeWidth={2.5} /></button>
          )
        ) : isRunning ? (
          <>
            <button disabled={busy} onClick={() => onFinish("success")} type="button" title="标记成功"><CheckCircle color="#2563eb" size={14} strokeWidth={2.5} /></button>
            <button disabled={busy} onClick={() => onFinish("failed")} type="button" title="标记失败"><XCircle color="#2563eb" size={14} strokeWidth={2.5} /></button>
          </>
        ) : (
          <>
            <button disabled={busy} onClick={onRollback} type="button" title="回滚"><RotateCcw color="#2563eb" size={14} strokeWidth={2.5} /></button>
            <button onClick={() => goTo("observability")} type="button" title="更多"><MoreVertical color="#2563eb" size={14} strokeWidth={2.5} /></button>
          </>
        )}
      </span>
    </div>
  );
}

// RolloutProgress：真实 K8s rollout 的实时进度（相位 + ready/desired + 进度条）。
function RolloutProgress({ meta, running }: { meta: DeploymentMeta; running: boolean }) {
  const pct = typeof meta.progress === "number" ? meta.progress : running ? 0 : 100;
  const tone = meta.phase === "rolled_back" || meta.phase === "failed" ? "danger" : meta.phase === "succeeded" ? "success" : "warning";
  return (
    <div className="pipeline-rollout">
      <div className="pipeline-rollout-head">
        <strong>{phaseLabel(meta.phase)}</strong>
        <small>{meta.ready ?? 0}/{meta.desired ?? "?"} ready</small>
      </div>
      <div className="pipeline-rollout-bar"><i className={tone} style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} /></div>
    </div>
  );
}

function phaseLabel(phase?: string): string {
  switch (phase) {
    case "queued": return "排队中";
    case "patching": return "更新镜像";
    case "progressing": return "滚动发布中";
    case "succeeded": return "发布成功";
    case "rolling_back": return "回滚中";
    case "rolled_back": return "已回滚";
    case "failed": return "失败";
    default: return phase || "—";
  }
}

function TriggerDeployDrawer({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [form, setForm] = useState({ name: "", version: "latest", env: "prod", namespace: "default", image: "", owner: "system" });
  const set = (k: keyof typeof form) => (e: ChangeEvent<HTMLInputElement>) =>
    setForm((f) => ({ ...f, [k]: e.target.value }));

  const realRollout = form.image.trim() !== "";

  const triggerMutation = useMutation({
    mutationFn: () =>
      api("/api/deployments", {
        method: "POST",
        body: JSON.stringify({
          name: form.name.trim(),
          version: form.version,
          env: form.env,
          namespace: form.namespace.trim(),
          image: form.image.trim(),
          operator: form.owner
        })
      }),
    onSuccess: () => {
      toast.success(realRollout ? "已触发真实 rollout" : "部署记录已创建");
      qc.invalidateQueries({ queryKey: ["deployments"] });
      onClose();
    },
    onError: (e) => toast.error(`触发失败：${describeError(e)}`)
  });

  return (
    <Drawer open title="触发部署" subtitle="创建一次新的发布流水线" onClose={onClose}>
      <div className="drawer-section">
        <input className="drawer-input" placeholder="服务名 / Deployment 名" value={form.name} onChange={set("name")} />
        <input className="drawer-input" placeholder="版本" value={form.version} onChange={set("version")} />
        <input className="drawer-input" placeholder="环境（dev/staging/prod）" value={form.env} onChange={set("env")} />
        <input className="drawer-input" placeholder="目标命名空间（真实 rollout 用）" value={form.namespace} onChange={set("namespace")} />
        <input className="drawer-input" placeholder="镜像（留空=仅登记记录；填写=真实 rollout）" value={form.image} onChange={set("image")} />
        <input className="drawer-input" placeholder="Owner" value={form.owner} onChange={set("owner")} />
        <p className="pipeline-rollout-hint">
          {realRollout
            ? `将 patch ${form.namespace || "default"}/${form.name || "<deployment>"} 镜像为 ${form.image.trim()} 并跟踪滚动发布，失败自动回滚（需 ALLOW_K8S_WRITES，且非 serving 命名空间）。`
            : "留空镜像：仅登记部署记录（不触达集群）。填写镜像 + 命名空间即触发真实 K8s rollout。"}
        </p>
        <button
          className="infra-action-btn"
          type="button"
          disabled={!form.name.trim() || triggerMutation.isPending}
          onClick={() => triggerMutation.mutate()}
        >
          {triggerMutation.isPending ? "触发中..." : realRollout ? "触发真实 rollout" : "登记部署记录"}
        </button>
      </div>
    </Drawer>
  );
}

function buildPipelineKpis(deployments: Deployment[] | undefined): KpiItem[] {
  const list = deployments ?? [];
  const running = list.filter((item) => normalizeStatus(item.status) === RUNNING).length;
  const success = list.filter((item) => normalizeStatus(item.status) === "success").length;
  const failed = list.filter((item) => normalizeStatus(item.status) === "failed").length;
  const finished = success + failed;
  const successRate = finished ? (success / finished) * 100 : null;
  const durations = list
    .filter((item) => item.started_at && item.finished_at)
    .map((item) => (new Date(item.finished_at as string).getTime() - new Date(item.started_at as string).getTime()) / 1000)
    .filter((seconds) => seconds >= 0);
  const avgDuration = durations.length ? durations.reduce((a, b) => a + b, 0) / durations.length : null;

  return [
    { id: "pipelines", label: "流水线总数", value: String(list.length), detail: `活跃 ${running}`, trend: [] },
    { id: "deploys", label: "部署次数", value: String(list.length), detail: successRate == null ? "暂无完成" : `${fmt(successRate, 0)}% 成功`, trend: [], tone: "success" },
    { id: "success", label: "变更成功率", value: successRate == null ? "—" : `${fmt(successRate, 1)}%`, detail: finished ? `已完成 ${finished} 次` : "暂无完成记录", trend: [], tone: successRate != null && successRate < 95 ? "warning" : "success" },
    { id: "duration", label: "平均部署时长", value: avgDuration == null ? "—" : formatDuration(avgDuration), detail: durations.length ? `${durations.length} 次完成` : "暂无耗时数据", trend: [], tone: "success" },
    { id: "rollback", label: "失败次数", value: String(failed), detail: failed ? "需要关注" : "无失败", trend: [], tone: failed ? "warning" : "success" },
  ];
}

function deriveEnvDistribution(deployments: Deployment[] | undefined): { total: number; items: Array<{ label: string; value: number; percent: string; tone: string }> } {
  const list = deployments ?? [];
  const counts = new Map<string, number>();
  for (const item of list) counts.set(item.env || "unknown", (counts.get(item.env || "unknown") ?? 0) + 1);
  const total = list.length;
  const tones = ["info", "success", "warning", "ai", "neutral"];
  return {
    total,
    items: [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([label, value], index) => ({ label, value, percent: total ? `${Math.round((value / total) * 100)}%` : "0%", tone: tones[index % tones.length] })),
  };
}

function formatDuration(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return minutes > 0 ? `${minutes}m ${secs}s` : `${secs}s`;
}

function derivePipelineRows(deployments: Deployment[] | undefined): PipelineConsoleRow[] {
  if (!deployments?.length) return [];
  return deployments.map((item) => {
    const status = statusLabel(item.status);
    const normalized = normalizeStatus(item.status);
    const version = item.version ?? "latest";
    const service = item.name || "deployment";
    return {
      id: item.id,
      name: `${service} 发布`,
      service,
      env: item.env ?? "prod",
      version,
      commit: compactMeta([item.metadata.branch, item.metadata.gate]) || "-",
      status,
      stage: normalized === RUNNING ? "部署中" : normalized === "failed" ? "部署失败" : "已完成",
      stageMeta: normalized === RUNNING ? "Rolling Update" : normalized === "failed" ? "Deploy Failed" : "Deployed",
      trigger: item.metadata.owner ?? "system",
      startedAt: item.started_at ? shortTime(item.started_at) : "-",
      duration: item.finished_at && item.started_at ? relativeTime(item.started_at) : "进行中",
      tone: normalized === "failed" ? "danger" : normalized === RUNNING ? "warning" : normalized === "success" ? "success" : "neutral",
      source: item,
    };
  });
}

function normalizeStatus(value: string) {
  const lower = String(value || "").toLowerCase();
  if (lower.includes("run") || lower.includes("progress") || lower.includes("deploy")) return RUNNING;
  if (lower.includes("fail") || lower.includes("error")) return "failed";
  if (lower.includes("cancel")) return "cancelled";
  return "success";
}

function statusLabel(value: string) {
  const normalized = normalizeStatus(value);
  if (normalized === RUNNING) return "运行中";
  if (normalized === "failed") return "失败";
  if (normalized === "cancelled") return "已取消";
  return "成功";
}
