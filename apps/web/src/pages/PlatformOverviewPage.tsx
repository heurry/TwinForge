import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Activity, AlertTriangle, ArrowRight, Bot, Boxes, CheckCircle2, Cpu,
  Database, Gauge, GraduationCap, RefreshCw, Rocket, Server, Square,
} from "lucide-react";

import { EmptyState, ErrorState, Skeleton } from "../components/common/FeedbackStates";
import { PageHeader, PanelHeader, StatusBadge } from "../components/common/PlatformPrimitives";
import { api, listModelRegistry } from "../lib/api";
import { fmt, relativeTime } from "../lib/format";
import { useGoToPage } from "../lib/useGoToPage";
import type { Page } from "../types/navigation";
import type { Deployment, Incident } from "../types/ops";
import type { Metrics } from "../types/platform";
import type { TrainingJobsList } from "../types/training";

type Runtime = {
  available?: boolean;
  status?: string;
  model?: string;
  profile?: string;
  prefix_caching?: boolean;
  config?: { max_num_seqs?: number; max_num_batched_tokens?: number; chunked_prefill?: boolean };
};
type Scenario = {
  context_length?: number;
  concurrency?: number;
  success_rate?: number;
  quality_gate_pass_rate?: number;
  p95_ttft_ms?: number;
  p95_tpot_ms?: number;
  p95_ms?: number;
  output_throughput_tokens_per_second?: number;
  output_tokens_per_second?: number;
};
type Evidence = {
  inference?: {
    benchmark?: { run_id?: string; endpoint_id?: string; updated_at?: string; prefix_caching?: boolean; summary?: { scenarios?: Scenario[] } };
  };
};

const trainingLifecycle: Array<{ label: string; page: Page; icon: typeof Database }> = [
  { label: "模型与数据", page: "datasets", icon: Database },
  { label: "训练任务", page: "training", icon: GraduationCap },
  { label: "版本归档", page: "models", icon: Boxes },
  { label: "训练诊断", page: "aiOps", icon: Bot },
];
const inferenceLifecycle: Array<{ label: string; page: Page; icon: typeof Database }> = [
  { label: "推理模型", page: "models", icon: Boxes },
  { label: "运行配置", page: "config", icon: Database },
  { label: "压测验收", page: "benchmarks", icon: Gauge },
  { label: "发布上线", page: "release", icon: Rocket },
  { label: "生产诊断", page: "aiOps", icon: Bot },
];

export function PlatformOverviewPage({ setPage }: { setPage: (page: Page) => void }) {
  const goTo = useGoToPage();
  const runtime = useQuery({ queryKey: ["inference", "runtime"], queryFn: () => api<Runtime>("/api/inference/runtime"), retry: false, refetchInterval: 5000 });
  const training = useQuery({ queryKey: ["training", "jobs"], queryFn: () => api<TrainingJobsList>("/api/training/jobs"), retry: false, refetchInterval: 5000 });
  const evidence = useQuery({ queryKey: ["ai", "inference", "evidence", "latest"], queryFn: () => api<Evidence>("/api/ai/inference/evidence"), retry: false, refetchInterval: 15000 });
  const metrics = useQuery({ queryKey: ["metrics", "current"], queryFn: () => api<Metrics>("/api/metrics/current"), retry: false, refetchInterval: 5000 });
  const registry = useQuery({ queryKey: ["model-registry"], queryFn: listModelRegistry, retry: false });
  const deployments = useQuery({ queryKey: ["deployments"], queryFn: () => api<{ deployments: Deployment[] }>("/api/deployments"), retry: false, refetchInterval: 10000 });
  const incidents = useQuery({ queryKey: ["incidents", "dashboard"], queryFn: () => api<{ incidents: Incident[] }>("/api/incidents?status=open"), retry: false, refetchInterval: 15000 });

  const jobs = training.data?.jobs ?? [];
  const activeTraining = jobs.find((job) => job.status === "running" || job.status === "pending");
  const runtimeActive = runtime.data?.status === "ready" || runtime.data?.status === "starting";
  const channel = runtimeActive && activeTraining ? "conflict" : runtimeActive ? "inference" : activeTraining ? "training" : "idle";
  const latestTraining = jobs[0];
  const scenarios = evidence.data?.inference?.benchmark?.summary?.scenarios ?? [];
  const latestDeployment = deployments.data?.deployments?.find((item) => item.metadata.mode === "inference_runtime");
  const openIncidents = incidents.data?.incidents ?? [];
  const gpu = metrics.data?.gpu ?? [];
  const gpuMemory = gpu.length ? Math.max(...gpu.map((item) => item.memory_utilization_percent ?? (item.memory_used_mb / Math.max(item.memory_total_mb, 1)) * 100)) : null;
  const gpuUtil = gpu.length ? Math.max(...gpu.map((item) => item.gpu_utilization_percent)) : null;
  const benchmark = useMemo(() => summarizeScenarios(scenarios), [scenarios]);
  const evidenceModelID = modelIDFromEndpoint(evidence.data?.inference?.benchmark?.endpoint_id) || runtime.data?.model;
  const versions = registry.data?.versions ?? [];
  const activeVersions = versions.filter((version) => version.status === "active" || version.status === "serving").length;

  function refreshAll() {
    runtime.refetch(); training.refetch(); evidence.refetch(); metrics.refetch(); registry.refetch(); deployments.refetch(); incidents.refetch();
  }

  return (
    <section className="infra-page twinforge-overview">
      <PageHeader
        title="TwinForge 模型交付总览"
        subtitle="单机双卡 3090 · 训练微调与生产推理双轨交付，配置、任务、压测、发布和诊断证据全程关联"
        actions={<button className="console-refresh" type="button" onClick={refreshAll}><RefreshCw size={14} /> 刷新</button>}
      />

      <div className="overview-delivery-tracks">
        <DeliveryTrack label="训练交付" stages={trainingLifecycle} onOpen={(stage) => goTo(stage.page, {
          deliveryKind: "training",
          trainingJobId: latestTraining?.id,
          modelVersionId: latestTraining?.model_version_id,
          benchmarkRunId: null,
          deploymentId: null,
        })} />
        <DeliveryTrack label="推理交付" stages={inferenceLifecycle} onOpen={(stage) => {
          const releasing = stage.page === "release";
          goTo(stage.page, {
            deliveryKind: "inference",
            modelId: releasing ? latestDeployment?.metadata.model_id : evidenceModelID,
            benchmarkRunId: releasing ? latestDeployment?.metadata.benchmark_run_id : evidence.data?.inference?.benchmark?.run_id,
            modelVersionId: releasing ? latestDeployment?.metadata.model_version_id : null,
            deploymentId: releasing ? latestDeployment?.id : null,
            trainingJobId: null,
          });
        }} />
      </div>

      <section className={`gpu-channel ${channel}`}>
        <div className="gpu-channel-title"><Cpu size={19} /><span><strong>GPU 执行通道</strong><small>训练微调与 27B 推理服务分时使用双卡，不要求同时运行</small></span></div>
        <div className="gpu-channel-owner">
          {channel === "training" ? <GraduationCap size={17} /> : channel === "inference" ? <Gauge size={17} /> : channel === "conflict" ? <AlertTriangle size={17} /> : <Square size={15} />}
          <span><strong>{channelLabel(channel)}</strong><small>{channelDetail(channel, activeTraining?.name, runtime.data)}</small></span>
        </div>
        <div className="gpu-channel-metrics"><span>GPU 利用率 <strong>{metric(gpuUtil, "%")}</strong></span><span>显存占用 <strong>{metric(gpuMemory, "%")}</strong></span></div>
      </section>

      <div className="overview-kpi-strip">
        <Kpi label="训练任务" value={String(jobs.length)} detail={latestTraining ? `${latestTraining.status} · ${relativeTime(latestTraining.updated_at)}` : "暂无任务"} />
        <Kpi label="模型版本" value={String(versions.length)} detail={`${activeVersions} 个 active`} />
        <Kpi label="P95 TTFT" value={benchmark.ttft == null ? "—" : `${fmt(benchmark.ttft, 0)} ms`} detail={benchmark.scenarioCount ? `${benchmark.scenarioCount} 个压测场景最大值` : "等待压测"} tone="blue" />
        <Kpi label="P95 TPOT" value={benchmark.tpot == null ? "—" : `${fmt(benchmark.tpot, 1)} ms`} detail="已完成场景最大值" tone="blue" />
        <Kpi label="请求成功率" value={benchmark.success == null ? "—" : `${fmt(benchmark.success * 100, 1)}%`} detail={`输出正确性 ${benchmark.quality == null ? "—" : `${fmt(benchmark.quality * 100, 1)}%`}`} tone={benchmark.success != null && benchmark.success < 0.99 ? "orange" : "green"} />
        <Kpi label="待处理事件" value={String(openIncidents.length)} detail={openIncidents[0]?.title || "无未解决事件"} tone={openIncidents.length ? "orange" : "green"} />
      </div>

      <div className="overview-workbench-grid">
        <Workflow
          icon={GraduationCap}
          title="训练任务控制面"
          model="Qwen3.5-4B"
          state={activeTraining ? `执行中 · ${activeTraining.name}` : latestTraining ? `最近 ${latestTraining.status}` : "待启动"}
          detail="DianJin-CSC 客服数据 · LoRA / PEFT · Kubeflow PyTorchJob"
          action="进入训练"
          onClick={() => goTo("training", { deliveryKind: "training", trainingJobId: latestTraining?.id, modelVersionId: latestTraining?.model_version_id, benchmarkRunId: null, deploymentId: null })}
        />
        <Workflow
          icon={Gauge}
          title="推理服务控制面"
          model={runtime.data?.model || evidenceModelID || "Qwen3.6-27B"}
          state={runtime.data?.status || "未启动"}
          detail={`${runtime.data?.profile || "baseline"} · Prefix Cache ${runtime.data?.prefix_caching ? "ON" : "OFF"} · 1K/2K × C1-C16`}
          action="进入实验"
          onClick={() => goTo("benchmarks", { deliveryKind: "inference", modelId: evidenceModelID, benchmarkRunId: evidence.data?.inference?.benchmark?.run_id, trainingJobId: null })}
        />
        <Workflow
          icon={Rocket}
          title="模型发布上线"
          model={latestDeployment?.name || "等待可发布版本"}
          state={latestDeployment?.metadata.phase || latestDeployment?.status || "待发布"}
          detail={latestDeployment ? `${latestDeployment.env || "-"} · ${latestDeployment.version || "latest"} · ${relativeTime(latestDeployment.started_at)}` : "版本门禁 · K8s rollout · 失败自动回滚"}
          action="进入发布"
          onClick={() => goTo("release", { deliveryKind: "inference", modelVersionId: latestDeployment?.metadata.model_version_id, benchmarkRunId: latestDeployment?.metadata.benchmark_run_id, deploymentId: latestDeployment?.id, trainingJobId: null })}
        />
        <Workflow
          icon={Bot}
          title="智能诊断"
          model={openIncidents.length ? `${openIncidents.length} 个待处理事件` : "当前无未解决事件"}
          state={openIncidents[0]?.severity || "healthy"}
          detail={openIncidents[0]?.summary || "训练证据、推理压测、运行日志、K8s 与 GPU 指标统一诊断"}
          action="进入诊断"
          onClick={() => goTo("aiOps", channel === "training" ? { deliveryKind: "training", trainingJobId: latestTraining?.id, benchmarkRunId: null, deploymentId: null, diagnosisId: null } : { deliveryKind: "inference", benchmarkRunId: evidence.data?.inference?.benchmark?.run_id, trainingJobId: null, diagnosisId: null })}
        />
      </div>

      <div className="overview-bottom-grid">
        <section className="infra-panel overview-scenario-panel">
          <PanelHeader title="最近推理验收矩阵" action={evidence.data?.inference?.benchmark?.run_id ? `run ${evidence.data.inference.benchmark.run_id.slice(0, 8)}` : "暂无正式结果"} />
          {evidence.isLoading ? <Skeleton rows={3} /> : evidence.isError || !scenarios.length ? <EmptyState title="暂无已完成压测" description="启动 vLLM 后在推理服务控制面运行 1K/2K × 1-16 并发矩阵" /> : (
            <div className="overview-scenario-table">
              <div className="overview-scenario-row header"><span>上下文</span><span>并发</span><span>TTFT P95</span><span>TPOT P95</span><span>端到端 P95</span><span>吞吐</span><span>成功 / 正确</span></div>
              {scenarios.slice(0, 10).map((row, index) => <div className="overview-scenario-row" key={`${row.context_length}-${row.concurrency}-${index}`}><strong>{row.context_length ?? "-"}</strong><span>{row.concurrency ?? "-"}</span><span>{ms(row.p95_ttft_ms)}</span><span>{ms(row.p95_tpot_ms, 1)}</span><span>{ms(row.p95_ms)}</span><span>{throughput(row)} tok/s</span><span><StatusBadge status={Number(row.success_rate ?? 0) >= 0.99 && Number(row.quality_gate_pass_rate ?? 0) >= 0.99 ? "通过" : "未通过"} /></span></div>)}
            </div>
          )}
        </section>
        <section className="infra-panel overview-infra-panel">
          <PanelHeader title="平台底座" action="次级能力" />
          <InfraLink icon={Server} title="服务目录" detail={`${metrics.data?.service_instances?.length ?? 0} 个运行实例`} onClick={() => setPage("services")} />
          <InfraLink icon={Activity} title="可观测中心" detail={metrics.data ? `QPS ${fmt(metrics.data.qps ?? 0, 1)} · P95 ${metrics.data.p95_latency_ms == null ? "—" : `${fmt(metrics.data.p95_latency_ms, 0)} ms`}` : "等待指标"} onClick={() => setPage("observability")} />
          <InfraLink icon={Boxes} title="集群与资源" detail={metrics.data?.kubernetes?.available ? `${metrics.data.kubernetes.pods.length} Pods` : "集群状态不可用"} onClick={() => setPage("kubernetes")} />
        </section>
      </div>
    </section>
  );
}

function DeliveryTrack({ label, stages, onOpen }: { label: string; stages: Array<{ label: string; page: Page; icon: typeof Database }>; onOpen: (stage: { label: string; page: Page; icon: typeof Database }) => void }) {
  return <section><header>{label}</header><div className="overview-lifecycle">
    {stages.map((stage, index) => {
      const Icon = stage.icon;
      return <button key={`${label}-${stage.page}`} type="button" onClick={() => onOpen(stage)}><span>{String(index + 1).padStart(2, "0")}</span><Icon size={17} /><strong>{stage.label}</strong>{index < stages.length - 1 ? <ArrowRight className="overview-stage-arrow" size={14} /> : null}</button>;
    })}
  </div></section>;
}

function Workflow({ icon: Icon, title, model, state, detail, action, onClick }: { icon: typeof Gauge; title: string; model: string; state: string; detail: string; action: string; onClick: () => void }) {
  return <section className="workflow-item"><div className="workflow-item-head"><Icon size={18} /><span><strong>{title}</strong><small>{model}</small></span><StatusBadge status={state} /></div><p>{detail}</p><button type="button" onClick={onClick}>{action}<ArrowRight size={13} /></button></section>;
}

function Kpi({ label, value, detail, tone = "neutral" }: { label: string; value: string; detail: string; tone?: string }) {
  return <div className={tone}><span>{label}</span><strong>{value}</strong><small title={detail}>{detail}</small></div>;
}

function InfraLink({ icon: Icon, title, detail, onClick }: { icon: typeof Server; title: string; detail: string; onClick: () => void }) {
  return <button type="button" onClick={onClick}><Icon size={17} /><span><strong>{title}</strong><small>{detail}</small></span><ArrowRight size={14} /></button>;
}

function summarizeScenarios(scenarios: Scenario[]) {
  const max = (key: keyof Scenario) => scenarios.length ? Math.max(...scenarios.map((item) => Number(item[key] ?? 0))) : null;
  const min = (key: keyof Scenario) => scenarios.length ? Math.min(...scenarios.map((item) => Number(item[key] ?? 0))) : null;
  return { scenarioCount: scenarios.length, ttft: max("p95_ttft_ms"), tpot: max("p95_tpot_ms"), success: min("success_rate"), quality: min("quality_gate_pass_rate") };
}

function channelLabel(channel: string) {
  if (channel === "training") return "训练微调占用";
  if (channel === "inference") return "推理服务占用";
  if (channel === "conflict") return "检测到资源冲突";
  return "当前空闲";
}

function channelDetail(channel: string, job?: string, runtime?: Runtime) {
  if (channel === "training") return job || "PyTorchJob";
  if (channel === "inference") return `${runtime?.model || "qwen36-27b-fp8"} · ${runtime?.profile || "baseline"}`;
  if (channel === "conflict") return "请停止其中一侧，避免双卡显存争用";
  return "可启动训练任务或 vLLM 推理服务";
}

function metric(value: number | null, suffix: string) { return value == null || !Number.isFinite(value) ? "—" : `${fmt(value, 0)}${suffix}`; }
function ms(value?: number, digits = 0) { return value == null ? "—" : `${fmt(value, digits)} ms`; }
function throughput(row: Scenario) { const value = row.output_throughput_tokens_per_second ?? row.output_tokens_per_second; return value == null ? "—" : fmt(value, 1); }
function modelIDFromEndpoint(endpoint?: string) { return endpoint?.replace(/-vllm$/, ""); }
