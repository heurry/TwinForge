// Phase F：训练任务列表（实时相位/副本进度 + 事件 + Pod 日志 + 取消 + 注册版本跳转）。
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Bot, Boxes, ChevronDown, ChevronRight, Cpu, Database, FileText, RefreshCw, Server, XCircle } from "lucide-react";

import { EmptyState, ErrorState, Skeleton } from "../common/FeedbackStates";
import { trainingJobKubernetes, trainingJobLogs } from "../../lib/api";
import { describeError } from "../common/FeedbackStates";
import { relativeTime } from "../../lib/format";
import type { Training } from "../../lib/useTraining";
import type { TrainingJob, TrainingKubernetesDetail, TrainingLogs } from "../../types/training";

const statusText: Record<string, string> = {
  pending: "等待中",
  running: "运行中",
  succeeded: "成功",
  failed: "失败",
  cancelled: "已取消"
};

export function TrainingJobsPanel({ training, onOpenModel, onDiagnose }: { training: Training; onOpenModel: (job: TrainingJob) => void; onDiagnose: (job: TrainingJob) => void }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const [logs, setLogs] = useState<Record<string, TrainingLogs>>({});
  const [loadingLogs, setLoadingLogs] = useState<string | null>(null);

  const q = training.jobs;
  const expandedJob = q.data?.jobs.find((job) => job.id === expanded);
  const kubernetesQuery = useQuery({
    queryKey: ["training", "jobs", expanded, "kubernetes"],
    queryFn: () => trainingJobKubernetes(expanded as string),
    enabled: Boolean(expanded),
    retry: false,
    refetchInterval: expandedJob?.status === "pending" || expandedJob?.status === "running" ? 5000 : false,
  });
  if (q.isLoading) return <Skeleton rows={4} />;
  if (q.isError) return <ErrorState error={q.error} onRetry={q.refetch} />;
  const jobs = q.data?.jobs ?? [];
  if (jobs.length === 0) {
    return <EmptyState title="暂无训练任务" description="点击「提交训练」启动一次分布式 LoRA 微调" />;
  }

  async function fetchLogs(id: string) {
    setLoadingLogs(id);
    try {
      const res = await trainingJobLogs(id);
      setLogs((m) => ({ ...m, [id]: res }));
    } finally {
      setLoadingLogs(null);
    }
  }

  return (
    <div className="training-jobs-table">
      <div className="training-jobs-row header">
        <span />
        <span>名称 / 基座</span>
        <span>副本</span>
        <span>相位</span>
        <span>状态</span>
        <span>更新</span>
        <span>操作</span>
      </div>
      {jobs.map((job) => {
        const isOpen = expanded === job.id;
        const phase = job.metadata?.phase ?? "—";
        const cancellable = job.status === "pending" || job.status === "running";
        return (
          <div className="training-job" key={job.id}>
            <div className="training-jobs-row" onClick={() => setExpanded(isOpen ? null : job.id)}>
              <span className="training-toggle">{isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}</span>
              <span className="training-name">
                <strong>{job.name}</strong>
                <small>{job.base_model}</small>
              </span>
              <span>
                {job.workers > 0 ? `1+${job.workers}` : "1"}
                {job.gpus_per_worker > 0 ? ` · ${job.gpus_per_worker} GPU` : ""}
              </span>
              <span className="training-phase">{phase}</span>
              <span>
                <em className={`training-status-pill status-${job.status}`}>{statusText[job.status] ?? job.status}</em>
              </span>
              <span className="training-time">{relativeTime(job.updated_at)}</span>
              <span className="training-row-actions" onClick={(e) => e.stopPropagation()}>
                {cancellable ? (
                  <button title="取消" type="button" disabled={training.cancel.isPending} onClick={() => training.cancel.mutate(job.id)}>
                    <XCircle size={14} />
                  </button>
                ) : null}
              </span>
            </div>

            {isOpen ? (
              <TrainingJobDetail
                job={job}
                kubernetes={kubernetesQuery.data}
                kubernetesError={kubernetesQuery.error}
                kubernetesLoading={kubernetesQuery.isLoading}
                logs={logs[job.id]}
                logsLoading={loadingLogs === job.id}
                onFetchLogs={() => fetchLogs(job.id)}
                onOpenModel={() => onOpenModel(job)}
                onDiagnose={() => onDiagnose(job)}
                onRefreshKubernetes={() => kubernetesQuery.refetch()}
              />
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

function TrainingJobDetail({
  job,
  kubernetes,
  kubernetesError,
  kubernetesLoading,
  logs,
  logsLoading,
  onFetchLogs,
  onOpenModel,
  onDiagnose,
  onRefreshKubernetes,
}: {
  job: TrainingJob;
  kubernetes?: TrainingKubernetesDetail;
  kubernetesError: Error | null;
  kubernetesLoading: boolean;
  logs?: TrainingLogs;
  logsLoading: boolean;
  onFetchLogs: () => void;
  onOpenModel: () => void;
  onDiagnose: () => void;
  onRefreshKubernetes: () => void;
}) {
  const resource = kubernetes?.resource;
  const replicas = resource?.replica_statuses ?? job.metadata?.replica_statuses ?? {};
  const controlEvents = job.metadata?.events ?? [];
  const clusterError = kubernetes?.cluster.error || resource?.error;
  return (
    <div className="training-job-detail">
      <div className="training-detail-head">
        <div><Server size={17} /><span><strong>{resource?.kind || "PyTorchJob"}</strong><small>{resource?.api_version || "kubeflow.org/v1"} · {job.k8s_job_ref || `${job.namespace}/${job.name}`}</small></span></div>
        <div className="training-detail-head-actions">
          <em className={`training-status-pill status-${job.status}`}>平台：{statusText[job.status] ?? job.status}</em>
          <em className={`training-status-pill status-${resource?.phase?.toLowerCase() || "unknown"}`}>K8s：{resource?.phase || (kubernetesLoading ? "读取中" : "不可用")}</em>
          <button title="刷新 Kubernetes 状态" type="button" onClick={onRefreshKubernetes}><RefreshCw size={13} /></button>
        </div>
      </div>

      {clusterError || kubernetesError ? (
        <div className="training-k8s-warning"><AlertTriangle size={15} /><span><strong>当前未读取到实时 Kubernetes 资源</strong><small>{clusterError || describeError(kubernetesError)}</small></span></div>
      ) : null}

      <section className="training-detail-section">
        <h3>任务配置</h3>
        <div className="training-config-grid">
          <DetailValue icon={Database} label="基座模型" value={job.base_model} />
          <DetailValue icon={Database} label="数据集" value={job.dataset_uri || "未配置"} />
          <DetailValue icon={Boxes} label="训练镜像" value={job.image} />
          <DetailValue icon={Cpu} label="分布式资源" value={`Master 1 + Worker ${job.workers} · ${job.gpus_per_worker} GPU/副本`} />
          <DetailValue label="Namespace" value={job.namespace} />
          <DetailValue label="超参数" value={formatHyperparams(job.hyperparams)} />
        </div>
      </section>

      <section className="training-detail-section">
        <h3>Kubernetes 副本与 Pods</h3>
        {Object.keys(replicas).length > 0 ? (
          <div className="training-replicas">
            {Object.entries(replicas).map(([role, rs]) => <span className="training-replica-pill" key={role}><Boxes size={12} />{role}: active {rs.active ?? 0} / ok {rs.succeeded ?? 0} / fail {rs.failed ?? 0}</span>)}
          </div>
        ) : null}
        {kubernetesLoading ? <p className="training-muted">正在读取 PyTorchJob 与 Pod...</p> : kubernetes?.pods.length ? (
          <div className="training-pod-table">
            <div className="training-pod-row header"><span>Pod</span><span>角色</span><span>阶段</span><span>Ready</span><span>重启</span><span>节点</span><span>Pod IP</span></div>
            {kubernetes.pods.map((pod) => <div className="training-pod-row" key={pod.name}><strong>{pod.name}</strong><span>{podRole(pod.name)}</span><span>{pod.phase}</span><span>{pod.ready}</span><span>{pod.restarts}</span><span>{pod.node || "—"}</span><span>{pod.pod_ip || "—"}</span></div>)}
          </div>
        ) : <p className="training-muted">当前没有对应的 Master/Worker Pod。</p>}
      </section>

      <div className="training-detail-columns">
        <section className="training-detail-section">
          <h3>控制面提交事件</h3>
          <EventList events={controlEvents.map((event) => ({ title: event.phase, message: event.message, time: event.at }))} />
        </section>
        <section className="training-detail-section">
          <h3>Kubernetes Events</h3>
          <EventList events={(kubernetes?.events ?? []).map((event) => ({ title: `${event.type} · ${event.reason}`, message: `${event.resource_name}: ${event.message}`, time: event.event_time }))} />
        </section>
      </div>

      <div className="training-detail-links">
        {job.model_version_id ? <button className="training-link" type="button" onClick={onOpenModel}>已注册模型版本 → 模型与版本{job.output_artifact_uri ? `（产物 ${job.output_artifact_uri}）` : ""}</button> : null}
        <button className="training-link" type="button" onClick={onDiagnose}><Bot size={13} /> 诊断此训练任务</button>
      </div>

      <section className="training-detail-section training-logs-block">
        <div className="training-detail-section-title"><h3>Pod 日志</h3><button className="training-logs-btn" type="button" disabled={logsLoading} onClick={onFetchLogs}><FileText size={13} />{logsLoading ? "拉取中..." : "拉取 Master Pod 日志"}</button></div>
        {logs ? logs.logs ? <pre className="training-logs">{logs.logs}</pre> : <p className="training-muted">{logs.note ?? "暂无日志"}</p> : <p className="training-muted">按需拉取最近 200 行日志。</p>}
      </section>
    </div>
  );
}

function DetailValue({ icon: Icon, label, value }: { icon?: typeof Database; label: string; value: string }) {
  return <div>{Icon ? <Icon size={14} /> : <span className="training-detail-dot" />}<span><small>{label}</small><strong title={value}>{value}</strong></span></div>;
}

function EventList({ events }: { events: Array<{ title: string; message: string; time: string }> }) {
  if (!events.length) return <p className="training-muted">暂无事件。</p>;
  return <div className="training-events">{events.slice(-12).reverse().map((event, index) => <div className="training-event" key={`${event.time}-${index}`}><em>{event.title}</em><span title={event.message}>{event.message}</span><small>{relativeTime(event.time)}</small></div>)}</div>;
}

function formatHyperparams(value: Record<string, unknown>): string {
  const entries = Object.entries(value ?? {});
  return entries.length ? entries.map(([key, item]) => `${key}=${String(item)}`).join(" · ") : "默认参数";
}

function podRole(name: string): string {
  if (name.includes("master")) return "Master";
  if (name.includes("worker")) return "Worker";
  return "训练副本";
}
