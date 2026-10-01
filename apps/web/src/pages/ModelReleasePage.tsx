import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { parse, stringify } from "yaml";
import { AlertTriangle, CheckCircle2, Circle, Code2, ExternalLink, RefreshCw, Rocket, Server, SlidersHorizontal, Square } from "lucide-react";

import { describeError, EmptyState, ErrorState, Skeleton } from "../components/common/FeedbackStates";
import { PageHeader, PanelHeader, StatusBadge } from "../components/common/PlatformPrimitives";
import { api, listModelRegistry } from "../lib/api";
import { relativeTime } from "../lib/format";
import { useGoToPage } from "../lib/useGoToPage";
import { useDeliveryContext } from "../lib/useDeliveryContext";
import type { Deployment } from "../types/ops";
import type { ServiceInstance } from "../types/platform";
import { PipelinesPage } from "./PipelinesPage";

type ReleaseCandidate = {
  profile: string;
  label: string;
  description: string;
  available: boolean;
  gate_passed: boolean;
  run_id?: string;
  scenarios: number;
  min_success_rate: number;
  min_quality_rate: number;
  average_p95_ttft_ms: number;
  average_p95_tpot_ms: number;
  average_output_tokens_per_second: number;
  max_p95_ttft_ms: number;
  max_p95_tpot_ms: number;
  slo_ttft_limit_ms: number;
  slo_tpot_limit_ms: number;
  max_num_seqs: number;
  max_num_batched_tokens: number;
  runtime_request?: Record<string, unknown>;
  error?: string;
};

type ReleaseState = {
  model_id: string;
  endpoint_id: string;
  release_endpoint_id?: string;
  candidates: ReleaseCandidate[];
  requested_candidate?: ReleaseCandidate;
  supported_models?: Array<{ model_id: string; endpoint_id: string; gpu_count: number; tensor_parallel_size: number; pipeline_parallel_size: number; prefix_caching: boolean; aibrix_supported?: boolean }>;
  runtime: { status?: string; profile?: string; endpoint?: string; config?: Record<string, unknown>; error?: string };
  serving_status?: {
    overall: string;
    model_id: string;
    gateway_endpoint: string;
    namespace: string;
    deployment: string;
    checked_at: string;
    release: { id?: string; status: string; phase?: string };
    workload: ServingLayerState;
    gateway: ServingLayerState;
    model_route: ServingLayerState;
	tool_calling: ServingLayerState;
	  target: "aibrix" | "direct";
	  managed: boolean;
	  can_stop: boolean;
  };
	serving_models?: ServingModelState[];
  progress: {
    active_stage: string;
    weight_percent?: number;
    stages: Array<{ key: string; label: string; state: "pending" | "active" | "complete"; detail?: string }>;
  };
};

type ServingModelState = NonNullable<ReleaseState["serving_status"]>;

type ServingLayerState = { status: string; detail?: string; desired?: number; ready?: number; available?: number; http_status?: number };

type RuntimeForm = {
  parallelism: "tp1" | "tp2" | "pp2";
  maxNumSeqs: number;
  maxNumBatchedTokens: number;
  schedulingPolicy: "fcfs" | "priority";
  prefixCaching: boolean;
  asyncScheduling: boolean;
  gpuMemoryUtilization: number;
  maxModelLen: number;
  kvCacheDType: "auto" | "fp8";
};

type BenchmarkRun = {
  run_id: string;
  status: string;
  endpoint_id?: string;
  config?: { vllm?: Record<string, unknown> };
};

const DEFAULT_RUNTIME: RuntimeForm = {
  parallelism: "tp2",
  maxNumSeqs: 8,
  maxNumBatchedTokens: 4096,
  schedulingPolicy: "fcfs",
  prefixCaching: true,
  asyncScheduling: true,
  gpuMemoryUtilization: 0.9,
  maxModelLen: 4096,
  kvCacheDType: "auto",
};

function runtimeFormForModel(modelID: string): RuntimeForm {
  const is4B = modelID.includes("4b");
  const isQwen38 = modelID === "qwen38-27b-fp8";
  return {
    ...DEFAULT_RUNTIME,
    parallelism: is4B ? "tp1" : "tp2",
    prefixCaching: !is4B,
    kvCacheDType: "auto",
    maxNumSeqs: isQwen38 ? 4 : 8,
    maxNumBatchedTokens: isQwen38 ? 8192 : 4096,
    gpuMemoryUtilization: isQwen38 ? 0.88 : 0.9,
    maxModelLen: isQwen38 ? 24576 : 4096,
  };
}

export function ModelReleasePage({ initialTab = "model" }: { initialTab?: "model" | "pipeline" }) {
  const qc = useQueryClient();
  const goTo = useGoToPage();
  const { context, update } = useDeliveryContext();
  const [versionID, setVersionID] = useState("");
  const [env, setEnv] = useState("prod");
  const [approved, setApproved] = useState(false);
  const [centerTab, setCenterTab] = useState<"model" | "pipeline">(initialTab);
  const [modelID, setModelID] = useState(context.modelId || "qwen36-27b-fp8");
  const [configMode, setConfigMode] = useState<"form" | "yaml">("form");
  const [runtimeForm, setRuntimeForm] = useState<RuntimeForm>(() => runtimeFormForModel(context.modelId || "qwen36-27b-fp8"));
  const [yamlDraft, setYamlDraft] = useState("");
  const [yamlError, setYamlError] = useState("");
  const [yamlDirty, setYamlDirty] = useState(false);
  const [rolloutStrategy, setRolloutStrategy] = useState<"canary" | "full">("canary");
  const [releaseTarget, setReleaseTarget] = useState<"aibrix" | "direct">("aibrix");
  const [stableEndpoint, setStableEndpoint] = useState("");
  const [canaryWeight, setCanaryWeight] = useState(10);
  const hydratedRun = useRef("");

  const runtimeRequest = useMemo(() => runtimeRequestFromForm(runtimeForm), [runtimeForm]);
  const releaseQuery = useMemo(() => {
    const params = new URLSearchParams({
      max_num_seqs: String(runtimeForm.maxNumSeqs),
      max_num_batched_tokens: String(runtimeForm.maxNumBatchedTokens),
      prefix_caching: String(runtimeForm.prefixCaching),
    });
    params.set("model_id", modelID);
    if (context.benchmarkRunId) params.set("benchmark_run_id", context.benchmarkRunId);
    return params.toString();
  }, [context.benchmarkRunId, modelID, runtimeForm.maxNumBatchedTokens, runtimeForm.maxNumSeqs, runtimeForm.prefixCaching]);

  const registry = useQuery({ queryKey: ["model-registry"], queryFn: listModelRegistry, refetchInterval: 10000 });
  const releases = useQuery({ queryKey: ["inference", "releases", releaseQuery], queryFn: () => api<ReleaseState>(`/api/inference/releases?${releaseQuery}`), refetchInterval: 5000 });
  const benchmarkRun = useQuery({
    queryKey: ["benchmark", "release", context.benchmarkRunId],
    queryFn: () => api<BenchmarkRun>(`/api/benchmarks/${encodeURIComponent(context.benchmarkRunId!)}`),
    enabled: Boolean(context.benchmarkRunId),
  });
  const deployments = useQuery({ queryKey: ["deployments"], queryFn: () => api<{ deployments: Deployment[] }>("/api/deployments"), refetchInterval: 5000 });
  const instances = useQuery({ queryKey: ["service-instances"], queryFn: () => api<{ instances: ServiceInstance[] }>("/api/service-instances"), refetchInterval: 10000 });

  const releaseModelID = releases.data?.model_id || modelID;
  const selectedModelSpec = releases.data?.supported_models?.find((item) => item.model_id === releaseModelID);
  const singleGPUModel = selectedModelSpec ? selectedModelSpec.gpu_count === 1 : releaseModelID.includes("4b");
  const aibrixSupported = selectedModelSpec?.aibrix_supported ?? singleGPUModel;
  const productionPolicyName = `${releaseModelID}${releaseTarget === "aibrix" ? "" : "-direct"}-production`;
  const benchmarkModelID = modelIDFromEndpoint(benchmarkRun.data?.endpoint_id);
  const benchmarkModelMatches = !benchmarkModelID || benchmarkModelID === releaseModelID;
  const modelOptions = useMemo(() => {
    const supported = new Set((releases.data?.supported_models ?? []).map((item) => item.model_id));
    const fallback = new Set(["qwen38-27b-fp8", "qwen36-27b-fp8", "qwen36-27b-awq", "qwen35-4b-customer", "qwen3-4b-customer"]);
    return Array.from(new Set((registry.data?.versions ?? []).map((item) => item.model_id).filter((id) => supported.size > 0 ? supported.has(id) : fallback.has(id))));
  }, [registry.data, releases.data]);
  const versions = useMemo(() => (registry.data?.versions ?? []).filter((item) => item.model_id === releaseModelID), [registry.data, releaseModelID]);
  useEffect(() => {
    if (context.modelVersionId && versions.some((item) => item.id === context.modelVersionId)) {
      if (versionID !== context.modelVersionId) setVersionID(context.modelVersionId);
      return;
    }
    if (!versionID && versions[0]) setVersionID(versions.find((item) => item.status === "serving")?.id ?? versions[0].id);
  }, [context.modelVersionId, versionID, versions]);
  const selected = versions.find((item) => item.id === versionID);
  const candidate = releases.data?.requested_candidate ?? releases.data?.candidates.find((item) =>
    item.max_num_seqs === runtimeForm.maxNumSeqs && item.max_num_batched_tokens === runtimeForm.maxNumBatchedTokens && Boolean(item.runtime_request?.prefix_caching) === runtimeForm.prefixCaching
  );
  const recent = useMemo(() => (deployments.data?.deployments ?? []).filter((item) => item.metadata.mode === "inference_runtime" || item.metadata.mode === "inference_runtime_manual").slice(0, 8), [deployments.data]);
  const latest = recent[0];
  const recordedReleaseActive = latest?.metadata.mode === "inference_runtime" && (latest.status === "running" || latest.status === "success");
  const activeModelID = typeof latest?.metadata.model_id === "string" ? latest.metadata.model_id : releaseModelID;
  const activeReleaseTarget = latest?.metadata.release_target === "aibrix" ? "aibrix" : "direct";
  const activeProductionPolicyName = `${activeModelID}${activeReleaseTarget === "aibrix" ? "" : "-direct"}-production`;
  const productionEndpoint = `http://127.0.0.1:8081/api/routing/${activeProductionPolicyName}/v1/chat/completions`;
  const runtimeStatus = releases.data?.runtime.status ?? "unknown";
  const validatedReleaseActive = Boolean(recordedReleaseActive && (latest?.metadata.release_target === "aibrix" ? releases.data?.serving_status?.overall === "ready" : runtimeStatus === "ready"));
  const runtimeActive = runtimeStatus === "ready" || runtimeStatus === "starting";
  const activeBindings = selected ? (registry.data?.bindings[selected.model_id] ?? []) : [];
  const stableOptions = useMemo(() => (instances.data?.instances ?? []).filter((item) =>
    item.status === "healthy" && item.name !== releases.data?.release_endpoint_id &&
    (releaseTarget === "aibrix" ? item.name === "aibrix-gateway" : (item.name === "aibrix-gateway" || item.kind === "vllm"))
  ), [instances.data, releaseTarget, releases.data?.release_endpoint_id]);
  const stableInstance = stableOptions.find((item) => item.name === stableEndpoint);
  const generatedYAML = useMemo(() => stringify(releaseSpec(versionID, env, runtimeRequest, releaseModelID, releaseTarget, {
    strategy: rolloutStrategy, stableEndpoint, stableModel: stableInstance?.model_id || "", canaryWeight,
  })), [canaryWeight, env, releaseModelID, releaseTarget, rolloutStrategy, runtimeRequest, stableEndpoint, stableInstance?.model_id, versionID]);

  useEffect(() => {
    if (!aibrixSupported && releaseTarget === "aibrix") setReleaseTarget("direct");
  }, [aibrixSupported, releaseTarget]);

  useEffect(() => {
    if (stableEndpoint && stableOptions.some((item) => item.name === stableEndpoint)) return;
    setStableEndpoint(stableOptions[0]?.name ?? "");
    if (!stableOptions.length) setRolloutStrategy("full");
  }, [stableEndpoint, stableOptions]);

  useEffect(() => {
    if (!context.modelId || context.modelId === modelID) return;
    if (context.modelId !== "qwen38-27b-fp8" && context.modelId !== "qwen36-27b-fp8" && context.modelId !== "qwen36-27b-awq" && context.modelId !== "qwen35-4b-customer" && context.modelId !== "qwen3-4b-customer") return;
    setModelID(context.modelId);
    setVersionID("");
    applyTemplate(runtimeFormForModel(context.modelId));
  // 外部从模型注册中心进入发布中心时同步模型；表单变更不应反向触发此 effect。
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [context.modelId]);

  const updateRuntime = <K extends keyof RuntimeForm,>(key: K, value: RuntimeForm[K]) => {
    setRuntimeForm((current) => ({ ...current, [key]: value }));
    setApproved(false);
    setYamlDraft("");
    setYamlError("");
    setYamlDirty(false);
  };

  const applyTemplate = (next: RuntimeForm) => {
    setRuntimeForm(next);
    setApproved(false);
    setYamlDraft("");
    setYamlError("");
    setYamlDirty(false);
  };

  const applyBenchmarkConfig = () => {
    const vllm = benchmarkRun.data?.config?.vllm;
    if (!vllm) return;
    setRuntimeForm((current) => runtimeFormFromBenchmark(vllm, current));
    setApproved(false);
    setYamlDraft("");
    setYamlError("");
    setYamlDirty(false);
  };

  useEffect(() => {
    const runID = benchmarkRun.data?.run_id;
    if (!runID || hydratedRun.current === runID || !benchmarkRun.data?.config?.vllm) return;
    hydratedRun.current = runID;
    applyBenchmarkConfig();
  // 只在交付上下文切换到新的 Run 时自动回填；随后允许人工调整并触发“不匹配”门禁。
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [benchmarkRun.data?.run_id]);

  const applyYAML = () => {
    try {
      const parsed = parse(yamlDraft || generatedYAML) as Record<string, unknown>;
      const next = runtimeFormFromSpec(parsed);
      setRuntimeForm(next.runtime);
      if (next.modelVersionID) setVersionID(next.modelVersionID);
      if (next.env) setEnv(next.env);
      if (next.rolloutStrategy) setRolloutStrategy(next.rolloutStrategy);
      if (next.releaseTarget) setReleaseTarget(next.releaseTarget);
      if (next.stableEndpoint !== undefined) setStableEndpoint(next.stableEndpoint);
      if (next.canaryWeight) setCanaryWeight(next.canaryWeight);
      setYamlError("");
      setYamlDirty(false);
      setApproved(false);
      toast.success("YAML 已校验并应用到发布参数");
    } catch (error) {
      const message = error instanceof Error ? error.message : "YAML 格式无效";
      setYamlError(message);
      toast.error("YAML 校验失败", { description: message });
    }
  };

  const release = useMutation({
    mutationFn: () => api<{ id: string; status: string }>("/api/inference/releases", {
      method: "POST",
      body: JSON.stringify({
        model_version_id: versionID,
        runtime_request: runtimeRequest,
        release_spec: yamlDraft || generatedYAML,
        benchmark_run_id: candidate?.run_id || context.benchmarkRunId || "",
        env,
        operator: "frontend",
        rollout_strategy: rolloutStrategy,
        stable_endpoint: rolloutStrategy === "canary" ? stableEndpoint : "",
        stable_model: rolloutStrategy === "canary" ? stableInstance?.model_id || "" : "",
        canary_weight: rolloutStrategy === "canary" ? canaryWeight : 100,
        release_target: releaseTarget,
      }),
    }),
    onSuccess: (payload) => {
      toast.success("发布任务已提交", { description: "正在启动并检查 vLLM OpenAI-Compatible endpoint" });
      setApproved(false);
      update({
        deliveryKind: "inference",
        modelId: selected?.model_id || releaseModelID,
        modelVersionId: versionID,
        benchmarkRunId: candidate?.run_id || context.benchmarkRunId,
        deploymentId: payload.id,
        trainingJobId: null,
      });
      qc.invalidateQueries({ queryKey: ["deployments"] });
      qc.invalidateQueries({ queryKey: ["inference", "releases"] });
      qc.invalidateQueries({ queryKey: ["routing", "policies"] });
    },
    onError: (error) => toast.error("发布失败", { description: describeError(error) }),
  });

  const stop = useMutation({
    mutationFn: ({ modelID, target }: { modelID: string; target: "aibrix" | "direct" }) =>
      api<{ status: string }>(`/api/inference/serving-models/${encodeURIComponent(modelID)}?target=${target}`, { method: "DELETE" }),
    onSuccess: () => {
      toast.success("生产推理服务已下线");
      qc.invalidateQueries({ queryKey: ["deployments"] });
      qc.invalidateQueries({ queryKey: ["inference", "releases"] });
      qc.invalidateQueries({ queryKey: ["model-registry"] });
    },
    onError: (error) => toast.error("下线失败", { description: describeError(error) }),
  });

  const requestStop = (serving: Pick<ServingModelState, "model_id" | "target" | "managed">) => {
	const ownership = serving.managed ? "平台发布记录会保留用于审计。" : "该 workload 不存在平台发布记录，属于外部启动；平台将接管并缩容到 0。";
	if (!window.confirm(`确认下线 ${serving.model_id}（${serving.target === "aibrix" ? "AIBrix" : "Direct vLLM"}）？\n${ownership}`)) return;
	stop.mutate({ modelID: serving.model_id, target: serving.target });
  };

  const evidenceMatches = !context.benchmarkRunId || (benchmarkModelMatches && candidate?.run_id === context.benchmarkRunId);
  const rolloutReady = rolloutStrategy === "full" || Boolean(stableEndpoint && stableInstance?.status === "healthy");
  const targetReady = releaseTarget === "direct" || aibrixSupported;
  const hardReady = Boolean(selected && selected.status !== "deprecated" && candidate?.gate_passed && evidenceMatches && rolloutReady && targetReady && approved && !yamlDirty && !yamlError && !release.isPending);

  return (
    <section className="infra-page release-page">
      <PageHeader
        title="发布中心"
        subtitle="统一管理通用服务 CI/CD、模型推理发布、发布门禁、上线进度、发布记录和回滚"
        actions={centerTab === "model" ? <button className="console-refresh" type="button" onClick={() => { registry.refetch(); releases.refetch(); deployments.refetch(); instances.refetch(); }}><RefreshCw size={14} /> 刷新</button> : null}
      />

      <div className="release-center-tabs" role="tablist" aria-label="发布中心工作区">
        <button aria-selected={centerTab === "model"} className={centerTab === "model" ? "active" : ""} onClick={() => setCenterTab("model")} role="tab" type="button">模型服务发布</button>
        <button aria-selected={centerTab === "pipeline"} className={centerTab === "pipeline" ? "active" : ""} onClick={() => setCenterTab("pipeline")} role="tab" type="button">通用服务 CI/CD</button>
      </div>

      <div className="release-center-explainer">
        {centerTab === "model"
          ? "模型服务发布：选择已注册模型和 GPU 运行参数，必须绑定同参数压测证据，通过质量/SLO 门禁后启动 vLLM，并提供稳定的 OpenAI-Compatible API。"
          : "通用服务 CI/CD：从 GitLab 源码触发构建、测试、镜像推送和 Helm 清单校验，适用于控制面、网关及普通微服务；当前仓库没有自动执行 Kubernetes 发布，也不替代模型性能验收。"}
      </div>

      {centerTab === "model" && releases.data?.serving_models ? <ServingFleetPanel values={releases.data.serving_models} stopping={stop.isPending} onStop={requestStop} /> : null}
	  {centerTab === "model" && releases.data?.serving_status ? <ServingTruthPanel value={releases.data.serving_status} /> : null}

      {centerTab === "pipeline" ? <PipelinesPage embedded /> : <>

      <div className="delivery-flow" aria-label="模型交付链路">
        {["推理模型版本", "运行时配置", "压测验收", "生产发布", "服务观测", "故障诊断"].map((label, index) => (
          <button type="button" key={label} onClick={() => goTo((["models", "config", "benchmarks", "release", "observability", "aiOps"] as const)[index])}>
            <span>{index + 1}</span><strong>{label}</strong>
          </button>
        ))}
      </div>

      <section className="infra-panel release-progress-panel" aria-live="polite">
        <PanelHeader title="发布进度" action={latest ? `${latest.metadata.phase ?? latest.status} · ${relativeTime(latest.started_at)}` : "等待首次发布"} />
        <div className="release-progress-track">
          {(releases.data?.progress?.stages ?? []).map((stage, index) => {
            const Icon = stage.state === "complete" ? CheckCircle2 : Circle;
            const weightProgress = stage.key === "weights" && stage.state === "active" ? releases.data?.progress?.weight_percent : undefined;
            return <div className={`release-progress-step ${stage.state}`} key={stage.key}>
              <div className="release-progress-marker"><Icon size={16} /><span>{index + 1}</span></div>
              <div><strong>{stage.label}</strong><small>{stage.detail || (stage.state === "pending" ? "等待前序阶段" : "处理中")}</small></div>
              {weightProgress !== undefined ? <div className="release-weight-progress" role="progressbar" aria-label="模型权重加载进度" aria-valuenow={weightProgress} aria-valuemin={0} aria-valuemax={100}><span style={{ width: `${weightProgress}%` }} /></div> : null}
            </div>;
          })}
        </div>
        {latest?.metadata.events?.length ? <div className="release-progress-events">
          {latest.metadata.events.slice(-4).map((event) => <div key={`${event.at}-${event.phase}`}><time>{event.at.slice(11, 19)}</time><StatusBadge status={event.phase} /><span>{releaseEventMessage(event.phase, event.message)}</span></div>)}
        </div> : <p className="release-progress-empty">提交发布后，这里会显示容器创建、权重加载、编译和健康检查状态。</p>}
      </section>

      <div className="release-main-grid">
        <section className="infra-panel release-control-panel">
          <PanelHeader title="发布配置" action="受控 vLLM 生产运行时" />
          {registry.isLoading || releases.isLoading ? <Skeleton rows={5} /> : registry.isError ? <ErrorState error={registry.error} onRetry={registry.refetch} /> : releases.isError ? <ErrorState error={releases.error} onRetry={releases.refetch} /> : versions.length === 0 ? (
            <EmptyState title={`暂无 ${releaseModelID} 版本`} description="请先在模型与版本中心注册该模型的版本" />
          ) : (
            <div className="release-form">
              <label>推理模型<select value={releaseModelID} onChange={(event) => {
                const nextModelID = event.target.value;
                setModelID(nextModelID);
                setVersionID("");
                applyTemplate(runtimeFormForModel(nextModelID));
                update({ deliveryKind: "inference", modelId: nextModelID, modelVersionId: null, benchmarkRunId: null });
              }}>{(modelOptions.length ? modelOptions : [releaseModelID]).map((id) => <option key={id} value={id}>{id === "qwen38-27b-fp8" ? "Qwen3.8-27B-FP8 · 双卡" : id === "qwen36-27b-fp8" ? "Qwen3.6-27B-FP8 · 双卡" : id === "qwen36-27b-awq" ? "Qwen3.6-27B-AWQ · 双卡" : id === "qwen35-4b-customer" ? "Qwen3.5-4B Customer · 单卡" : id}</option>)}</select><small className="field-hint">注册模型决定版本列表、并行方式和 GPU 资源。</small></label>
              <label>模型版本<select value={versionID} onChange={(event) => {
                const nextID = event.target.value;
                const next = versions.find((item) => item.id === nextID);
                setVersionID(nextID);
                update({ deliveryKind: "inference", modelId: next?.model_id || releaseModelID, modelVersionId: nextID });
              }}>{versions.map((item) => <option key={item.id} value={item.id}>{item.model_id} · {item.version} · {item.status}</option>)}</select></label>
              <div className="release-version-info"><strong>{selected?.model_id} · {selected?.version}</strong><span>{selected?.base_model || "未登记基座模型"}</span><small>版本 ID：{selected?.id}</small></div>

              <div className="release-config-heading">
                <div><strong>运行时配置</strong><small>模板仅用于快速填充，最终发布值由参数表单或 YAML 决定</small></div>
                <div className="release-template-actions">
                  <button type="button" onClick={() => applyTemplate(runtimeFormForModel(releaseModelID))}>稳定起点</button>
                  <button type="button" onClick={() => applyTemplate({ ...runtimeFormForModel(releaseModelID), maxNumSeqs: 16, maxNumBatchedTokens: 8192 })}>高并发示例</button>
                </div>
              </div>
              <div className="release-config-mode" role="tablist" aria-label="发布配置编辑方式">
                <button aria-selected={configMode === "form"} className={configMode === "form" ? "active" : ""} onClick={() => setConfigMode("form")} role="tab" type="button"><SlidersHorizontal size={13} /> 参数配置</button>
                <button aria-selected={configMode === "yaml"} className={configMode === "yaml" ? "active" : ""} onClick={() => { setConfigMode("yaml"); if (!yamlDraft) setYamlDraft(generatedYAML); }} role="tab" type="button"><Code2 size={13} /> YAML 配置</button>
              </div>

              {configMode === "form" ? <div className="release-runtime-grid">
                <label>并行方式<select value={runtimeForm.parallelism} onChange={(event) => updateRuntime("parallelism", event.target.value as RuntimeForm["parallelism"])}>{singleGPUModel ? <option value="tp1">TP=1 / PP=1 · 单卡</option> : <><option value="tp2">TP=2 / PP=1 · 双卡</option><option value="pp2">TP=1 / PP=2 · 双卡</option></>}</select></label>
                <label>最大并发序列<select value={runtimeForm.maxNumSeqs} onChange={(event) => updateRuntime("maxNumSeqs", Number(event.target.value) as RuntimeForm["maxNumSeqs"])}>{[4, 8, 12, 16, 24, 32].map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
                <label>批处理 Token 预算<select value={runtimeForm.maxNumBatchedTokens} onChange={(event) => updateRuntime("maxNumBatchedTokens", Number(event.target.value) as RuntimeForm["maxNumBatchedTokens"])}>{[2048, 4096, 8192].map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
                <label>调度策略<select value={runtimeForm.schedulingPolicy} onChange={(event) => updateRuntime("schedulingPolicy", event.target.value as RuntimeForm["schedulingPolicy"])}><option value="fcfs">FCFS</option><option value="priority">Priority</option></select></label>
                <label>GPU 显存比例<select value={runtimeForm.gpuMemoryUtilization} onChange={(event) => updateRuntime("gpuMemoryUtilization", Number(event.target.value) as RuntimeForm["gpuMemoryUtilization"])}>{[0.85, 0.88, 0.9, 0.92].map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
                <label>最大模型长度<select value={runtimeForm.maxModelLen} onChange={(event) => updateRuntime("maxModelLen", Number(event.target.value) as RuntimeForm["maxModelLen"])}>{[3072, 4096, 8192, 16384, 24576].map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
                <label>KV Cache 类型<select value={runtimeForm.kvCacheDType} onChange={(event) => updateRuntime("kvCacheDType", event.target.value as RuntimeForm["kvCacheDType"])}><option value="auto">auto</option><option value="fp8">fp8</option></select></label>
                <label className="release-runtime-toggle"><input checked={runtimeForm.prefixCaching} onChange={(event) => updateRuntime("prefixCaching", event.target.checked)} type="checkbox" />启用 Prefix Cache</label>
                <label className="release-runtime-toggle"><input checked={runtimeForm.asyncScheduling} onChange={(event) => updateRuntime("asyncScheduling", event.target.checked)} type="checkbox" />启用异步调度</label>
              </div> : <div className="release-yaml-editor">
                <textarea aria-label="推理发布 YAML" spellCheck={false} value={yamlDraft || generatedYAML} onChange={(event) => { setYamlDraft(event.target.value); setYamlError(""); setYamlDirty(true); setApproved(false); }} />
                <div><button type="button" onClick={() => { setYamlDraft(generatedYAML); setYamlDirty(false); setYamlError(""); }}>从当前参数重新生成</button><button className="primary" type="button" onClick={applyYAML}>校验并应用 YAML</button></div>
                {yamlError ? <p className="release-yaml-error">{yamlError}</p> : null}
                {yamlDirty && !yamlError ? <p className="release-yaml-pending">YAML 已修改，请先“校验并应用”再提交发布。</p> : null}
              </div>}

              <p className="release-config-note">压测证据按 Prefix Cache、max_num_seqs 和 max_num_batched_tokens 匹配；完整运行参数和 YAML 会写入发布记录与审计事件。</p>

              {candidate ? <div className="release-evidence-strip">
                <div><small>验证场景</small><strong>{candidate.scenarios}</strong></div>
                <div><small>最低成功率</small><strong>{percent(candidate.min_success_rate)}</strong></div>
                <div><small>平均 P95 TTFT</small><strong>{milliseconds(candidate.average_p95_ttft_ms)}</strong></div>
                <div><small>平均 P95 TPOT</small><strong>{milliseconds(candidate.average_p95_tpot_ms)}</strong></div>
                <div><small>平均吞吐</small><strong>{candidate.average_output_tokens_per_second.toFixed(1)} tok/s</strong></div>
              </div> : null}

              {!benchmarkModelMatches ? <div className="release-context-warning"><AlertTriangle size={15} /><span>Run {context.benchmarkRunId?.slice(0, 12)} 验收的是 {benchmarkModelID}，当前发布通道是 {releaseModelID}；不同权重的证据不能混用。</span><button type="button" onClick={() => goTo("benchmarks", { deliveryKind: "inference", modelId: releaseModelID, benchmarkRunId: null })}>验收当前模型</button></div> : !evidenceMatches ? <div className="release-context-warning"><AlertTriangle size={15} /><span>Run {context.benchmarkRunId?.slice(0, 12)} 的验收参数与当前表单不一致，不能用另一组参数的证据发布。</span>{benchmarkRun.data?.config?.vllm ? <button type="button" onClick={applyBenchmarkConfig}>应用该 Run 参数</button> : <button type="button" onClick={() => goTo("benchmarks")}>重新验收</button>}</div> : null}

              <label>发布环境<select value={env} onChange={(event) => setEnv(event.target.value)}><option value="staging">staging</option><option value="prod">prod</option></select></label>
              <label>发布目标<select value={releaseTarget} onChange={(event) => { setReleaseTarget(event.target.value as "aibrix" | "direct"); setApproved(false); }}>
                <option value="aibrix" disabled={!aibrixSupported}>AIBrix / Kubernetes 稳定池（正式）</option>
                <option value="direct">本机 vLLM :8020（调试）</option>
              </select><small className="field-hint">AIBrix 会创建受 Kubernetes 管理的候选 Pod 并接入网关；本机模式只用于兼容和调试。</small></label>
              {!aibrixSupported ? <p className="release-context-warning"><AlertTriangle size={15} /><span>当前模型没有可用的 AIBrix 模型卷，暂时只能使用本机运行时。</span></p> : null}
              <div className="release-config-heading">
                <div><strong>上线流量策略</strong><small>发布成功后自动同步到“模型与服务路由策略”，并保留全量与回滚入口</small></div>
              </div>
              <div className="release-runtime-grid">
                <label>发布方式<select value={rolloutStrategy} onChange={(event) => { setRolloutStrategy(event.target.value as "canary" | "full"); setApproved(false); }}>
                  <option value="canary" disabled={!stableOptions.length}>灰度发布</option>
                  <option value="full">直接全量</option>
                </select><small className="field-hint">灰度会同时保留稳定服务和本次候选运行时。</small></label>
                {rolloutStrategy === "canary" ? <>
                  <label>稳定基线服务<select value={stableEndpoint} onChange={(event) => { setStableEndpoint(event.target.value); setApproved(false); }}>
                    {stableOptions.map((item) => <option key={item.name} value={item.name}>{item.name} · {item.model_id}</option>)}
                  </select><small className="field-hint">必须是当前健康且与候选 endpoint 不同的服务。</small></label>
                  <label>候选流量<select value={canaryWeight} onChange={(event) => { setCanaryWeight(Number(event.target.value)); setApproved(false); }}>
                    {[5, 10, 20, 30, 50].map((value) => <option key={value} value={value}>{value}%</option>)}
                  </select><small className="field-hint">稳定版获得 {100 - canaryWeight}%，候选版获得 {canaryWeight}%。</small></label>
                </> : null}
              </div>
              {rolloutStrategy === "canary" ? <p className="release-config-note">发布成功后，策略 <code>{productionPolicyName}</code> 会显示 stable {100 - canaryWeight}% / canary {canaryWeight}%；真实占比只统计经过该策略入口的请求。</p> : <p className="release-config-note">发布成功后生产策略会直接切到当前版本 100%，并保存切换前权重供回滚。</p>}
              <label className="release-approval"><input type="checkbox" checked={approved} onChange={(event) => setApproved(event.target.checked)} /><span>我已核对模型版本、压测 run、参数和 YAML，确认按 {singleGPUModel ? "单卡" : "双卡"} 资源启动生产服务</span></label>
              <button className="release-submit" disabled={!hardReady} type="button" onClick={() => release.mutate()}><Rocket size={15} />{release.isPending ? "正在提交..." : releaseTarget === "aibrix" ? "发布到 AIBrix 稳定池" : runtimeActive ? "替换本机调试运行时" : "启动本机调试运行时"}</button>
              <p className="release-note">{releaseTarget === "aibrix" ? "候选版本会作为 Kubernetes Deployment 接入 AIBrix Gateway；灰度完成后可在路由策略中全量或回滚。" : `本机 ${singleGPUModel ? "单卡" : "双卡"} vLLM 只用于调试与兼容，不属于 AIBrix 稳定池。`}</p>
            </div>
          )}
        </section>

        <aside className="release-gates">
          <section className="infra-panel">
            <PanelHeader title="发布门禁" action={candidate?.gate_passed ? "允许人工审批" : "存在未通过项"} />
            <Gate passed={Boolean(selected && selected.model_id === releaseModelID && selected.status !== "deprecated")} title="模型与通道匹配" detail={selected ? `${selected.model_id} · ${selected.status}` : "未选择版本"} />
            <Gate passed={Boolean(candidate?.available)} title="参数证据匹配" detail={candidate?.run_id ? `${candidate.run_id.slice(0, 8)} · seqs ${candidate.max_num_seqs} · tokens ${candidate.max_num_batched_tokens}` : candidate?.error || "无匹配压测"} />
            <Gate passed={Boolean(candidate?.gate_passed)} title="成功率与输出质量" detail={candidate ? `${percent(candidate.min_success_rate)} / ${percent(candidate.min_quality_rate)}，共 ${candidate.scenarios} 个场景` : "等待证据"} />
            <Gate passed={Boolean(candidate && candidate.max_p95_ttft_ms <= candidate.slo_ttft_limit_ms && candidate.max_p95_tpot_ms <= candidate.slo_tpot_limit_ms)} title="推理 SLO 门禁" detail={candidate ? `最差 TTFT ${milliseconds(candidate.max_p95_ttft_ms)} / TPOT ${milliseconds(candidate.max_p95_tpot_ms)}` : "等待证据"} />
            <Gate passed={activeBindings.length > 0} soft title="OpenAI-Compatible 绑定" detail={activeBindings.length ? `${activeBindings.length} 个固定 endpoint` : "没有服务绑定"} />
            {!candidate?.gate_passed ? <button className="release-link" type="button" onClick={() => goTo("benchmarks")}>前往推理服务控制面 <ExternalLink size={13} /></button> : null}
          </section>

          <section className="infra-panel release-production-state">
            <PanelHeader title="运行时状态" action={runtimeStatus === "ready" ? (validatedReleaseActive ? "已验收发布" : "运行中·未验收") : runtimeStatus} />
            <div><Server size={18} /><span><strong>{runtimeStatus === "ready" && !validatedReleaseActive ? "vLLM 运行中（手工启动）" : runtimeStatus}</strong><small>{releases.data?.runtime.endpoint || "http://127.0.0.1:8020/v1"}</small></span></div>
            <div><Rocket size={18} /><span><strong>{latest?.metadata.mode === "inference_runtime_manual" ? "手工启动记录" : latest?.metadata.phase ?? latest?.status ?? "暂无"}</strong><small>{latest ? `${latest.name} · ${relativeTime(latest.started_at)}` : "正式发布记录"}</small></span></div>
            {runtimeStatus === "ready" && !validatedReleaseActive ? <p className="release-context-warning"><AlertTriangle size={15} /><span>当前 8020 服务可调用，但它只是推理服务控制面启动的运行时，没有绑定通过发布门禁的压测证据，因此不能当作“已验收生产发布”。</span></p> : null}
            {releases.data?.serving_status?.can_stop ? <div className="release-runtime-actions">
              <button type="button" disabled={stop.isPending} onClick={() => requestStop(releases.data!.serving_status!)}><Square size={13} />{stop.isPending ? "下线中..." : "下线当前模型"}</button>
            </div> : null}
          </section>

          <section className="infra-panel release-service-access">
            <PanelHeader title="服务调用" action={validatedReleaseActive ? "生产可调用" : "仅内部调试"} />
            <p>{validatedReleaseActive ? "正式发布后通过生产路由入口调用，不需要进入容器。" : "当前地址仅供控制面、压测和故障排查使用，未通过门禁时不会作为生产入口。"}</p>
            <code>{validatedReleaseActive ? productionEndpoint : releases.data?.runtime.endpoint || "http://127.0.0.1:8020/v1"}</code>
            <div className="release-runtime-actions">
              <button type="button" onClick={() => copyText(validatedReleaseActive ? productionEndpoint : `${releases.data?.runtime.endpoint || "http://127.0.0.1:8020/v1"}/models`)}>{validatedReleaseActive ? "复制生产 Endpoint" : "复制 Models 地址"}</button>
              <button type="button" onClick={() => copyText(validatedReleaseActive ? productionModelCurl(productionEndpoint, activeModelID) : modelCurl(releases.data?.runtime.endpoint, releases.data?.runtime.config?.model as string | undefined))}>复制 curl 示例</button>
              <button type="button" onClick={() => goTo("routing")}>查看发布流量策略</button>
            </div>
            <small>生产业务应调用 <code>POST /api/routing/{activeProductionPolicyName}/v1/chat/completions</code>；该入口会检查正式发布状态并执行权重分流。AIBrix 网关承载模型实例，平台入口负责发布隔离、审计和流量统计。</small>
          </section>
        </aside>
      </div>

      <section className="infra-panel release-history-panel">
        <PanelHeader title="推理发布记录" action={`${recent.length} 条`} />
        {deployments.isLoading ? <Skeleton rows={3} /> : recent.length === 0 ? <EmptyState title="暂无推理发布记录" /> : (
          <div className="release-history-table">
            <div className="release-history-row header"><span>服务</span><span>版本</span><span>运行参数</span><span>压测证据</span><span>阶段</span><span>时间</span></div>
            {recent.map((item) => <div className="release-history-row" key={item.id}><strong>{item.name}</strong><span>{item.version || "-"}</span><span>{runtimeSummary(item.metadata.runtime_request)}</span><span>{item.metadata.benchmark_run_id?.slice(0, 8) || "-"}</span><StatusBadge status={item.metadata.phase || item.status} /><span>{relativeTime(item.started_at)}</span></div>)}
          </div>
        )}
      </section>
      </>}
    </section>
  );
}

function ServingFleetPanel({ values, stopping, onStop }: { values: ServingModelState[]; stopping: boolean; onStop: (value: ServingModelState) => void }) {
  const running = values.filter(value => value.overall === "ready" || value.overall === "starting");
  return <section className="infra-panel serving-fleet-panel">
    <header><span><Server size={17}/><strong>当前实际运行模型</strong></span><StatusBadge status={running.length ? `${running.length} 个运行中` : "无运行模型"}/></header>
    <p>来自 Kubernetes、AIBrix 模型探针和直连 vLLM 的实时结果，不以历史“发布成功”代替当前运行状态。</p>
    <div className="serving-fleet-table">
      <div className="serving-fleet-row header"><span>模型</span><span>运行目标</span><span>数据面状态</span><span>Agent Tool</span><span>纳管状态</span><span>工作负载</span><span>操作</span></div>
      {values.map(value => <div className={`serving-fleet-row ${value.can_stop ? "active" : ""}`} key={`${value.model_id}-${value.target}`}>
        <strong>{value.model_id}</strong>
        <span>{value.target === "aibrix" ? "AIBrix / Kubernetes" : "Direct vLLM"}</span>
        <StatusBadge status={servingStatusLabel(value.overall)}/>
        <StatusBadge status={value.tool_calling?.status === "ready" ? "可用" : value.tool_calling?.status === "not_checked" ? "未检查" : "不可用"}/>
        <span>{value.managed ? "平台纳管" : value.can_stop ? "外部启动" : "无发布记录"}</span>
        <code title={`${value.namespace}/${value.deployment}`}>{value.target === "aibrix" ? `${value.namespace}/${value.deployment}` : value.gateway_endpoint}</code>
        <span>{value.can_stop ? <button className="serving-stop-button" type="button" disabled={stopping} onClick={() => onStop(value)}><Square size={12}/>{stopping ? "处理中" : "下线"}</button> : "—"}</span>
      </div>)}
    </div>
  </section>;
}

function Gate({ passed, soft = false, title, detail }: { passed: boolean; soft?: boolean; title: string; detail: string }) {
  const Icon = passed ? CheckCircle2 : soft ? Circle : AlertTriangle;
  return <div className={`release-gate ${passed ? "passed" : soft ? "soft" : "blocked"}`}><Icon size={17} /><span><strong>{title}</strong><small title={detail}>{detail}</small></span></div>;
}

function ServingTruthPanel({ value }: { value: NonNullable<ReleaseState["serving_status"]> }) {
  const workloadDetail = `${value.namespace}/${value.deployment}${value.workload.desired !== undefined ? ` · ${value.workload.ready ?? 0}/${value.workload.desired} Ready` : ""}`;
  return <section className={`infra-panel serving-truth-panel status-${value.overall}`}>
    <header><span><Server size={17} /><strong>Serving 运行真值</strong><StatusBadge status={servingStatusLabel(value.overall)} /></span><small>{value.model_id} · {new Date(value.checked_at).toLocaleTimeString()}</small></header>
    <div className="serving-truth-grid">
      <ServingLayer label="发布记录（期望）" state={value.release.status} detail={value.release.id ? `${value.release.phase || value.release.status} · ${value.release.id.slice(0, 8)}` : "当前模型没有正式发布记录"} />
      <ServingLayer label="Kubernetes Workload" state={value.workload.status} detail={value.workload.detail || workloadDetail} meta={workloadDetail} />
      <ServingLayer label="AIBrix Gateway" state={value.gateway.status} detail={value.gateway.detail || value.gateway_endpoint} meta={value.gateway_endpoint} />
      <ServingLayer label="模型路由" state={value.model_route.status} detail={value.model_route.detail || "尚未执行端到端模型探针"} meta={value.model_route.http_status ? `HTTP ${value.model_route.http_status}` : undefined} />
	  <ServingLayer label="Agent Tool Calling" state={value.tool_calling?.status || "not_checked"} detail={value.tool_calling?.detail || "尚未执行 Tool Schema 探针"} meta={value.tool_calling?.http_status ? `HTTP ${value.tool_calling.http_status}` : undefined} />
    </div>
    <p>“发布成功”只代表历史控制面结果；只有 Workload、Gateway 和模型路由当前全部就绪，平台才标记为生产可调用。</p>
  </section>;
}

function ServingLayer({ label, state, detail, meta }: { label: string; state: string; detail: string; meta?: string }) {
  const ready = state === "ready" || state === "success" || state === "running";
  const Icon = ready ? CheckCircle2 : state === "starting" || state === "unknown" || state === "not_checked" || state === "not_released" ? Circle : AlertTriangle;
  return <article className={ready ? "ready" : state}><Icon size={17} /><span><small>{label}</small><strong>{servingStatusLabel(state)}</strong><em title={detail}>{detail}</em></span>{meta ? <code>{meta}</code> : null}</article>;
}

function servingStatusLabel(status: string) {
  return ({ ready: "已就绪", success: "历史成功", running: "执行中", starting: "启动中", stopped: "已停止", failed: "失败", degraded: "降级", unreachable: "不可达", unavailable: "不可用", missing: "不存在", unknown: "未知", not_checked: "未检查", not_released: "未发布", rolled_back: "已回滚" } as Record<string, string>)[status] || status;
}

function percent(value: number): string {
  return `${(value * 100).toFixed(value === 1 ? 0 : 1)}%`;
}

function milliseconds(value: number): string {
  return value >= 1000 ? `${(value / 1000).toFixed(2)}s` : `${value.toFixed(0)}ms`;
}

function runtimeRequestFromForm(form: RuntimeForm): Record<string, unknown> {
  return {
    profile: "scheduler",
    tensor_parallel_size: form.parallelism === "tp2" ? 2 : 1,
    pipeline_parallel_size: form.parallelism === "pp2" ? 2 : 1,
    max_num_seqs: form.maxNumSeqs,
    max_num_batched_tokens: form.maxNumBatchedTokens,
    scheduling_policy: form.schedulingPolicy,
    max_num_partial_prefills: 1,
    max_long_partial_prefills: 1,
    long_prefill_token_threshold: 0,
    stream_interval: 1,
    prefix_caching: form.prefixCaching,
    async_scheduling: form.asyncScheduling,
    scheduler_reserve_full_isl: true,
    disable_custom_all_reduce: true,
    gpu_memory_utilization: form.gpuMemoryUtilization,
    max_model_len: form.maxModelLen,
    kv_cache_dtype: form.kvCacheDType,
    speculative_decoding: "none",
  };
}

function runtimeFormFromBenchmark(vllm: Record<string, unknown>, current: RuntimeForm): RuntimeForm {
  const tp = Number(vllm.tensor_parallel_size ?? (current.parallelism === "tp2" ? 2 : 1));
  const pp = Number(vllm.pipeline_parallel_size ?? (current.parallelism === "pp2" ? 2 : 1));
  const scheduling = String(vllm.scheduling_policy ?? vllm.scheduler ?? current.schedulingPolicy);
  const kv = String(vllm.kv_cache_dtype ?? current.kvCacheDType);
  return {
    parallelism: tp === 1 && pp === 1 ? "tp1" : pp === 2 && tp !== 2 ? "pp2" : "tp2",
    maxNumSeqs: Number(vllm.max_num_seqs ?? current.maxNumSeqs),
    maxNumBatchedTokens: Number(vllm.max_num_batched_tokens ?? current.maxNumBatchedTokens),
    schedulingPolicy: scheduling === "priority" ? "priority" : "fcfs",
    prefixCaching: vllm.prefix_caching === true,
    asyncScheduling: vllm.async_scheduling !== false,
    gpuMemoryUtilization: Number(vllm.gpu_memory_utilization ?? current.gpuMemoryUtilization),
    maxModelLen: Number(vllm.max_model_len ?? current.maxModelLen),
    kvCacheDType: kv === "fp8" ? "fp8" : "auto",
  };
}

function copyText(value: string) {
  navigator.clipboard.writeText(value).then(() => toast.success("已复制调用信息")).catch(() => toast.error("复制失败"));
}

function modelCurl(endpoint?: string, model?: string): string {
  const base = endpoint || "http://127.0.0.1:8020/v1";
  return `curl -X POST ${base}/chat/completions -H 'Content-Type: application/json' -d '{"model":"${model || "qwen36-27b-fp8"}","messages":[{"role":"user","content":"你好"}]}'`;
}

function productionModelCurl(endpoint: string, model: string): string {
  return `curl -X POST ${endpoint} -H 'Content-Type: application/json' -d '{"model":"${model}","messages":[{"role":"user","content":"你好"}]}'`;
}

function modelIDFromEndpoint(endpoint?: string): string {
  return endpoint?.replace(/-vllm$/, "") || "";
}

function releaseSpec(
  modelVersionID: string,
  env: string,
  runtime: Record<string, unknown>,
  modelID: string,
  releaseTarget: "aibrix" | "direct",
  rollout: { strategy: "canary" | "full"; stableEndpoint: string; stableModel: string; canaryWeight: number },
): Record<string, unknown> {
  const is4B = modelID.includes("4b");
  return {
    apiVersion: "platform.twinforge.io/v1alpha1",
    kind: "InferenceRelease",
    metadata: { name: `${modelID}-production`, environment: env },
    spec: {
      modelVersionId: modelVersionID || "<select-model-version>",
      target: releaseTarget === "aibrix" ? "AIBrix" : "DirectVLLM",
      runtime,
      resources: { replicas: 1, gpu: is4B ? 1 : 2 },
      rollout: rollout.strategy === "canary"
        ? { strategy: "Canary", stableEndpoint: rollout.stableEndpoint, stableModel: rollout.stableModel, candidateWeight: rollout.canaryWeight, automaticRollback: true }
        : { strategy: "Full", automaticRollback: true },
      healthCheck: { path: "/v1/models", timeoutSeconds: 600 },
    },
  };
}

function runtimeFormFromSpec(document: Record<string, unknown>): {
  runtime: RuntimeForm;
  modelVersionID?: string;
  env?: string;
  rolloutStrategy?: "canary" | "full";
  releaseTarget?: "aibrix" | "direct";
  stableEndpoint?: string;
  canaryWeight?: number;
} {
  const spec = asObject(document.spec, "spec");
  const runtime = asObject(spec.runtime, "spec.runtime");
  const tp = numberField(runtime, "tensor_parallel_size", 2);
  const pp = numberField(runtime, "pipeline_parallel_size", 1);
  if (tp * pp !== 1 && tp * pp !== 2) throw new Error("tensor_parallel_size × pipeline_parallel_size 必须等于 1 或 2");
  const maxNumSeqs = numberField(runtime, "max_num_seqs", 8);
  const maxNumBatchedTokens = numberField(runtime, "max_num_batched_tokens", 4096);
  const gpuMemoryUtilization = numberField(runtime, "gpu_memory_utilization", 0.9);
  const maxModelLen = numberField(runtime, "max_model_len", 4096);
  if (![4, 8, 12, 16, 24, 32].includes(maxNumSeqs)) throw new Error("max_num_seqs 必须是 4/8/12/16/24/32");
  if (![2048, 4096, 8192].includes(maxNumBatchedTokens)) throw new Error("max_num_batched_tokens 必须是 2048/4096/8192");
  if (![0.85, 0.88, 0.9, 0.92].includes(gpuMemoryUtilization)) throw new Error("gpu_memory_utilization 必须是 0.85/0.88/0.9/0.92");
  if (![3072, 4096, 8192, 16384, 24576].includes(maxModelLen)) throw new Error("max_model_len 必须是 3072、4096、8192、16384 或 24576");
  const schedulingPolicy = String(runtime.scheduling_policy ?? "fcfs");
  if (schedulingPolicy !== "fcfs" && schedulingPolicy !== "priority") throw new Error("scheduling_policy 必须是 fcfs 或 priority");
  const kvCacheDType = String(runtime.kv_cache_dtype ?? "auto");
  if (kvCacheDType !== "auto" && kvCacheDType !== "fp8") throw new Error("kv_cache_dtype 必须是 auto 或 fp8");
  const metadata = document.metadata && typeof document.metadata === "object" ? document.metadata as Record<string, unknown> : {};
  const rollout = spec.rollout && typeof spec.rollout === "object" && !Array.isArray(spec.rollout) ? spec.rollout as Record<string, unknown> : {};
  const rolloutStrategy = String(rollout.strategy || "Full").toLowerCase() === "canary" ? "canary" : "full";
  const releaseTarget = String(spec.target || "AIBrix").toLowerCase() === "directvllm" ? "direct" : "aibrix";
  const canaryWeight = numberField(rollout, "candidateWeight", 10);
  if (rolloutStrategy === "canary" && (canaryWeight < 1 || canaryWeight > 50)) throw new Error("灰度 candidateWeight 必须在 1-50 之间");
  return {
    runtime: {
      parallelism: tp === 1 && pp === 1 ? "tp1" : pp === 2 ? "pp2" : "tp2",
      maxNumSeqs,
      maxNumBatchedTokens,
      schedulingPolicy,
      prefixCaching: runtime.prefix_caching !== false,
      asyncScheduling: runtime.async_scheduling !== false,
      gpuMemoryUtilization: gpuMemoryUtilization as RuntimeForm["gpuMemoryUtilization"],
      maxModelLen: maxModelLen as RuntimeForm["maxModelLen"],
      kvCacheDType,
    },
    modelVersionID: typeof spec.modelVersionId === "string" && !spec.modelVersionId.startsWith("<") ? spec.modelVersionId : undefined,
    env: typeof metadata.environment === "string" ? metadata.environment : undefined,
    rolloutStrategy,
    releaseTarget,
    stableEndpoint: typeof rollout.stableEndpoint === "string" ? rollout.stableEndpoint : undefined,
    canaryWeight,
  };
}

function asObject(value: unknown, name: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${name} 必须是对象`);
  return value as Record<string, unknown>;
}

function numberField(object: Record<string, unknown>, key: string, fallback: number): number {
  const value = object[key] ?? fallback;
  const parsed = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`${key} 必须是数字`);
  return parsed;
}

function runtimeSummary(request?: Record<string, unknown>): string {
  if (!request) return "历史模板";
  return `seqs ${request.max_num_seqs ?? "-"} · tokens ${request.max_num_batched_tokens ?? "-"}`;
}

function releaseEventMessage(phase: string, fallback: string): string {
  return ({
    starting: "启动已通过门禁的 vLLM 生产 workload",
    replacing: "停止上一版本的推理 workload",
    warming: "容器已创建，等待 OpenAI-Compatible 健康检查",
    succeeded: "vLLM 生产 endpoint 已就绪",
    rolling_back: "新配置启动异常，正在恢复原运行时",
  } as Record<string, string>)[phase] ?? fallback;
}
