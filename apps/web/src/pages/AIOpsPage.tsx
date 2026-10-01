import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  Activity, AlertTriangle, CheckCircle, Cpu, Database, FileText, Gauge,
  GraduationCap, Play, RefreshCw, Search, Server, SquareTerminal, Zap
} from "lucide-react";

import { EmptyState, ErrorState, Skeleton } from "../components/common/FeedbackStates";
import { KpiGrid, PageHeader, PanelHeader, StatusBadge } from "../components/common/PlatformPrimitives";
import { api } from "../lib/api";
import { fmt } from "../lib/format";
import { useDeliveryContext } from "../lib/useDeliveryContext";
import type { Incident } from "../types/ops";
import type { KpiItem } from "../types/ui";

type Scope = "inference" | "training";
type EvidenceItem = { label: string; detail: string; source: string };
type RecommendedAction = { action: string; risk: string; impact: string };
type RelatedResource = { type: string; id?: string | null; name?: string | null };
type Diagnosis = {
  id: string;
  question: string;
  status: string;
  root_cause?: string | null;
  confidence?: number | null;
  impact?: string | null;
  evidence?: EvidenceItem[];
  recommended_actions?: RecommendedAction[];
  related_resources?: RelatedResource[];
  model_id?: string | null;
  error?: string | null;
  created_at?: string;
  category?: string;
  severity?: string;
};
type DiagnosisResponse = {
  diagnosis: Diagnosis;
  mode?: string;
  incident_id?: string | null;
  incident_created?: boolean;
};
type AIStatus = {
  status?: string;
  llm_connected?: boolean;
  effective_mode?: string;
  model?: string;
  configured_model?: string;
  model_selection?: string;
  available_models?: string[];
  llm_base_url?: string;
  llm_error?: string;
  error?: string;
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
  gpu_after?: { max_memory_utilization_percent?: number };
};
type InferenceEvidence = {
  inference: {
    benchmark: {
      run_id: string;
      endpoint_id?: string;
      workload?: string;
      prefix_caching?: boolean;
      chunked_prefill?: boolean;
      max_num_seqs?: number;
      max_num_batched_tokens?: number;
      summary?: { scenarios?: Scenario[] };
      updated_at?: string;
      config?: { vllm?: Record<string, unknown> };
    };
    baseline?: { run_id?: string; summary?: { scenarios?: Scenario[] } };
    runtime?: Record<string, unknown>;
    gpu?: Record<string, unknown>;
  };
};
type TrainingJob = {
  id: string;
  name: string;
  status: string;
  namespace: string;
  base_model: string;
  dataset_uri?: string | null;
  workers: number;
  gpus_per_worker: number;
  k8s_job_ref?: string | null;
  output_artifact_uri?: string | null;
  updated_at?: string;
};
type TrainingEvidence = {
  training: {
    job: TrainingJob;
    pytorch_job?: { available?: boolean; phase?: string; reason?: string; message?: string; replica_statuses?: Record<string, unknown> };
    pod?: { pod?: string | null; available?: boolean; logs?: string; error?: string };
    gpu?: Record<string, unknown>;
  };
};

export function AIOpsPage() {
  const queryClient = useQueryClient();
  const { context, update } = useDeliveryContext();
  const [scope, setScope] = useState<Scope>(context.deliveryKind ?? "inference");
  const [selectedDiagnosisID, setSelectedDiagnosisID] = useState(context.diagnosisId ?? "");
  const [question, setQuestion] = useState("");
  const [incidentSearch, setIncidentSearch] = useState("");
  const [severityFilter, setSeverityFilter] = useState("all");
  const previousTarget = useRef("");
  const targetKey = `${scope}:${context.benchmarkRunId || context.trainingJobId || "history"}`;

  useEffect(() => {
    if (previousTarget.current && previousTarget.current !== targetKey) setSelectedDiagnosisID("");
    previousTarget.current = targetKey;
  }, [targetKey]);

  const inferenceQuery = useQuery({
    queryKey: ["ai", "inference", "evidence", context.benchmarkRunId ?? "latest"],
    queryFn: () => api<InferenceEvidence>(`/api/ai/inference/evidence${context.benchmarkRunId ? `?run_id=${encodeURIComponent(context.benchmarkRunId)}` : ""}`),
    retry: false,
    refetchInterval: 15000,
  });
  const aiStatusQuery = useQuery({
    queryKey: ["ai", "status"],
    queryFn: () => api<AIStatus>("/api/ai/status"),
    refetchInterval: 10_000,
    retry: false,
  });
  const trainingQuery = useQuery({
    queryKey: ["ai", "training", "evidence", context.trainingJobId ?? "latest"],
    queryFn: () => api<TrainingEvidence>(`/api/ai/training/evidence${context.trainingJobId ? `?job_id=${encodeURIComponent(context.trainingJobId)}` : ""}`),
    retry: false,
    refetchInterval: 15000,
  });
  const incidentsQuery = useQuery({
    queryKey: ["incidents", "aiops"],
    queryFn: () => api<{ incidents: Incident[] }>("/api/incidents"),
    refetchInterval: 15000,
  });
  const diagnosesQuery = useQuery({
    queryKey: ["ai", "diagnoses"],
    queryFn: () => api<{ diagnoses: Diagnosis[] }>("/api/ai/diagnoses?limit=20"),
    refetchInterval: 15000,
  });
  const diagnoses = useMemo(() => diagnosesQuery.data?.diagnoses ?? [], [diagnosesQuery.data]);
  useEffect(() => {
    if (context.deliveryKind && context.deliveryKind !== scope) setScope(context.deliveryKind);
  }, [context.deliveryKind, scope]);
  useEffect(() => {
    // 有明确 Run/Job 时不能沿用全局最新诊断（可能属于另一模型）；
    // 当前专项诊断由成功回调显式选中。无交付上下文时才展示历史最新记录。
    if (context.diagnosisId && !context.benchmarkRunId && !context.trainingJobId && context.diagnosisId !== selectedDiagnosisID) {
      setSelectedDiagnosisID(context.diagnosisId);
    }
  }, [context.diagnosisId, context.benchmarkRunId, context.trainingJobId, selectedDiagnosisID]);
  useEffect(() => {
    if (!selectedDiagnosisID && !context.benchmarkRunId && !context.trainingJobId && diagnoses[0]?.id) setSelectedDiagnosisID(diagnoses[0].id);
  }, [diagnoses, selectedDiagnosisID, context.benchmarkRunId, context.trainingJobId]);
  const diagnosisQuery = useQuery({
    queryKey: ["ai", "diagnosis", selectedDiagnosisID],
    queryFn: () => api<{ diagnosis: Diagnosis }>(`/api/ai/diagnoses/${selectedDiagnosisID}`),
    enabled: Boolean(selectedDiagnosisID),
  });

  const diagnoseMutation = useMutation({
    mutationFn: (target: Scope) => api<DiagnosisResponse>("/api/ai/diagnose", {
      method: "POST",
      body: JSON.stringify({
        scope: target,
        question: question.trim() || (target === "inference"
          ? "请基于指定 vLLM 压测、运行时和 GPU 证据，检查请求成功率与输出正确性门禁，并对 TTFT、TPOT 瓶颈进行归因和给出下一轮控制变量实验。"
          : "请基于指定训练台账、PyTorchJob 状态、Pod 日志、GPU 和产物证据，诊断 LoRA 微调任务并给出修复建议。"),
        benchmark_run_id: target === "inference" ? context.benchmarkRunId : undefined,
        training_job_id: target === "training" ? context.trainingJobId : undefined,
        create_incident: true,
        // 诊断包含证据、动作和关联资源；768 容易截断 JSON，至少保留 1536 tokens。
        max_tokens: 1536,
        temperature: 0.2,
        operator: "frontend",
      }),
      timeoutMs: 60_000,
    }),
    onSuccess: (payload, target) => {
      setSelectedDiagnosisID(payload.diagnosis.id);
      setQuestion("");
      update({ deliveryKind: target, diagnosisId: payload.diagnosis.id });
      toast.success("专项诊断已完成", {
        description: payload.incident_created
          ? `已关联新 Incident ${payload.incident_id?.slice(0, 8)}`
          : `${payload.diagnosis.id.slice(0, 8)} · ${payload.mode ?? "diagnosis"}`,
      });
      queryClient.invalidateQueries({ queryKey: ["ai", "diagnoses"] });
      queryClient.invalidateQueries({ queryKey: ["incidents", "aiops"] });
    },
    onError: (error) => toast.error("专项诊断失败", { description: describeError(error) }),
  });
  const incidentTransitionMutation = useMutation({
    mutationFn: ({ id, action }: { id: string; action: "ack" | "resolve" }) =>
      api(`/api/incidents/${id}/${action}`, { method: "POST", body: JSON.stringify({ operator: "frontend" }) }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["incidents", "aiops"] }),
    onError: (error) => toast.error("Incident 操作失败", { description: describeError(error) }),
  });

  const activeEvidenceQuery = scope === "inference" ? inferenceQuery : trainingQuery;
  const evidenceReady = activeEvidenceQuery.isSuccess;
  const storedDiagnosis = diagnosisQuery.data?.diagnosis;
  // 兼容历史版本把未解析 JSON 直接写进 root_cause 的记录，避免前端展示一大段原文。
  const embeddedRca = extractEmbeddedRca(storedDiagnosis?.root_cause);
  const diagnosis: Diagnosis | undefined = embeddedRca && storedDiagnosis ? { ...storedDiagnosis, ...embeddedRca } : storedDiagnosis;
  const classification = embeddedRca?.classification ?? parseClassification(diagnosis?.evidence);
  const inference = inferenceQuery.data?.inference;
  const training = trainingQuery.data?.training;
  const scenarios = inference?.benchmark.summary?.scenarios ?? [];
  const worst = [...scenarios].sort((a, b) => (b.p95_ttft_ms ?? 0) - (a.p95_ttft_ms ?? 0))[0];
  const minSuccess = scenarios.length ? Math.min(...scenarios.map((item) => item.success_rate ?? 0)) : null;
  const gpuPeak = scenarios.length ? Math.max(...scenarios.map((item) => item.gpu_after?.max_memory_utilization_percent ?? 0)) : null;
  const incidents = incidentsQuery.data?.incidents ?? [];
  const filteredIncidents = incidents.filter((incident) => {
    const query = incidentSearch.trim().toLowerCase();
    return (severityFilter === "all" || incident.severity === severityFilter)
      && (!query || `${incident.id} ${incident.title} ${incident.summary ?? ""}`.toLowerCase().includes(query));
  });
  const busy = diagnoseMutation.isPending || incidentTransitionMutation.isPending;

  const kpis: KpiItem[] = scope === "inference" ? [
    { id: "run", label: "压测场景", value: scenarios.length ? String(scenarios.length) : "—", detail: inference?.benchmark.run_id?.slice(0, 12) || "等待压测", trend: [] },
    { id: "ttft", label: "最差 P95 TTFT", value: metric(worst?.p95_ttft_ms, "ms"), detail: worst ? `${contextLabel(worst.context_length)} / C${worst.concurrency}` : "等待证据", trend: [], tone: (worst?.p95_ttft_ms ?? 0) > 3000 ? "warning" : "success" },
    { id: "tpot", label: "对应 P95 TPOT", value: metric(worst?.p95_tpot_ms, "ms"), detail: "decode 时延", trend: [] },
    { id: "success", label: "最低成功率", value: minSuccess == null ? "—" : `${fmt(minSuccess * 100, 1)}%`, detail: "正确性优先门禁", trend: [], tone: minSuccess != null && minSuccess < 0.99 ? "danger" : "success" },
    { id: "gpu", label: "显存峰值", value: gpuPeak == null ? "—" : `${fmt(gpuPeak, 1)}%`, detail: "场景采集值", trend: [], tone: (gpuPeak ?? 0) >= 95 ? "warning" : "success" },
  ] : [
    { id: "job", label: "训练任务", value: training?.job.status ?? "—", detail: training?.job.name ?? "暂无任务", trend: [], tone: training?.job.status === "failed" ? "danger" : "success" },
    { id: "phase", label: "PyTorchJob", value: training?.pytorch_job?.phase ?? "—", detail: training?.pytorch_job?.available === false ? "集群暂不可达" : "实时状态", trend: [] },
    { id: "replicas", label: "训练副本", value: training ? `${training.job.workers} × ${training.job.gpus_per_worker}` : "—", detail: "workers × GPU", trend: [] },
    { id: "logs", label: "Pod 日志", value: training?.pod?.available ? "已采集" : "—", detail: training?.pod?.pod ?? "等待运行 Pod", trend: [], tone: training?.pod?.available ? "success" : "warning" },
    { id: "artifact", label: "LoRA 产物", value: training?.job.output_artifact_uri ? "已归档" : "—", detail: training?.job.output_artifact_uri ?? "尚未产出", trend: [], tone: training?.job.status === "succeeded" && !training.job.output_artifact_uri ? "warning" : "success" },
  ];

  const refresh = () => {
    aiStatusQuery.refetch();
    inferenceQuery.refetch();
    trainingQuery.refetch();
    incidentsQuery.refetch();
    diagnosesQuery.refetch();
    if (selectedDiagnosisID) diagnosisQuery.refetch();
  };

  return (
    <section className="infra-page aiops-page aiops-replica">
      <PageHeader
        title="AIOps / 智能诊断"
        subtitle="训练任务与推理服务的多源取证、瓶颈归因和 Incident 闭环"
        actions={<>
          <button className="console-refresh primary" disabled={busy || !evidenceReady} onClick={() => diagnoseMutation.mutate(scope)} type="button">
            <Play size={14} /> {diagnoseMutation.isPending ? "诊断中..." : `运行${scope === "inference" ? "推理" : "训练"}诊断`}
          </button>
          <button className="console-refresh" disabled={busy} onClick={refresh} type="button"><RefreshCw size={14} /> 刷新证据</button>
        </>}
      />

      <div className="aiops-scope-switch" role="tablist" aria-label="诊断范围">
        <button aria-selected={scope === "inference"} className={scope === "inference" ? "active" : ""} onClick={() => { setScope("inference"); setSelectedDiagnosisID(""); update({ deliveryKind: "inference", trainingJobId: null, diagnosisId: null }); }} role="tab" type="button"><Zap size={15} /> 推理服务诊断</button>
        <button aria-selected={scope === "training"} className={scope === "training" ? "active" : ""} onClick={() => { setScope("training"); setSelectedDiagnosisID(""); update({ deliveryKind: "training", benchmarkRunId: null, deploymentId: null, diagnosisId: null }); }} role="tab" type="button"><GraduationCap size={15} /> 训练任务诊断</button>
      </div>

      <div className={`aiops-llm-status ${aiStatusQuery.data?.llm_connected ? "live" : "degraded"}`}>
        <Cpu size={15} />
        <strong>{aiStatusQuery.data?.llm_connected ? "LLM 实时推理已连接" : "LLM 未连接，诊断将使用规则降级"}</strong>
        <span title={aiStatusQuery.data?.llm_base_url || ""}>
          {aiStatusQuery.data?.model || "未配置模型"} · {aiStatusQuery.data?.model_selection === "auto_probe" ? "自动探活" : aiStatusQuery.data?.effective_mode || "unknown"}
          {aiStatusQuery.data?.available_models?.length ? ` · 可用 ${aiStatusQuery.data.available_models.join(", ")}` : ""}
        </span>
        {!aiStatusQuery.data?.llm_connected && (aiStatusQuery.data?.llm_error || aiStatusQuery.data?.error) ? <small>{aiStatusQuery.data.llm_error || aiStatusQuery.data.error}</small> : null}
      </div>

      <div className="aiops-question-bar">
        <label><Search size={14} /><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder={`可选：补充本次${scope === "inference" ? "推理压测" : "训练任务"}的诊断问题；留空使用标准模板`} /></label>
        <span>{scope === "inference" ? `Run ${context.benchmarkRunId?.slice(0, 12) || "latest"}` : `Job ${context.trainingJobId?.slice(0, 12) || "latest"}`}</span>
      </div>

      <KpiGrid className="aiops-kpi-strip" items={kpis} />

      <section className="infra-panel aiops-context-panel">
        <PanelHeader title={`${scope === "inference" ? "推理" : "训练"}证据快照`} action={evidenceReady ? "真实采集 · 每 15 秒刷新" : "等待可诊断任务"} />
        {activeEvidenceQuery.isLoading ? <Skeleton rows={2} /> : activeEvidenceQuery.isError ? (
          <EmptyState title={scope === "training" ? "暂无训练任务证据" : "暂无已完成压测"} description={scope === "training" ? "创建并启动训练任务后，将自动采集台账、PyTorchJob、Pod 日志与 GPU 快照。" : "完成一次推理压测后即可进行专项归因。"} />
        ) : (
          <div className="aiops-context-grid">
            {(scope === "inference" ? inferenceContext(inference) : trainingContext(training)).map((item) => (
              <div className="aiops-context-card" key={item.label} title={item.value}>
                <ContextIcon id={item.id} />
                <div><span>{item.label}</span><strong>{item.value}</strong></div>
              </div>
            ))}
          </div>
        )}
      </section>

      <div className="aiops-diagnosis-grid">
        <section className="infra-panel aiops-rca-panel">
          <PanelHeader title="根因与证据链" action={diagnosis ? `诊断 ${diagnosis.id.slice(0, 8)}` : "尚未运行"} />
          {diagnosisQuery.isLoading ? <Skeleton rows={3} /> : diagnosisQuery.isError ? <ErrorState error={diagnosisQuery.error} onRetry={diagnosisQuery.refetch} /> : diagnosis ? <>
            <div className={`aiops-root-cause ${classification.severity === "info" ? "healthy" : ""}`}>
              <div className="aiops-diagnosis-meta">
                <StatusBadge status={severityLabel(classification.severity)} />
                <span>{categoryLabel(classification.category)}</span>
                <span>{diagnosis.confidence == null ? "置信度 —" : `置信度 ${fmt(diagnosis.confidence * 100, 0)}%`}</span>
              </div>
              <strong>根因结论</strong>
              <p>{diagnosis.root_cause}</p>
              {diagnosis.impact ? <small>影响：{diagnosis.impact}</small> : null}
              {diagnosis.error ? <small>诊断校验：{diagnosis.error}（已使用当前证据规则结论）</small> : null}
            </div>
            <div className="aiops-evidence-list">
              <strong>关键证据</strong>
              {(diagnosis.evidence ?? []).filter((item) => item.source !== "classifier").map((item, index) => (
                <span key={`${item.label}-${index}`}><b>{item.label}</b> · {item.detail} <em>{item.source}</em></span>
              ))}
            </div>
          </> : <EmptyState title="暂无诊断结论" description="选择有证据的作用域并运行专项诊断。" />}
        </section>

        <section className="infra-panel aiops-actions-panel">
          <PanelHeader title="优化与修复建议" action="人工确认后执行" />
          {diagnosis?.recommended_actions?.length ? <div className="aiops-action-list">
            {diagnosis.recommended_actions.map((action, index) => <div className="aiops-action-row" key={`${action.action}-${index}`}>
              <span>{index + 1}</span>
              <div><strong>{action.action}</strong><small>{action.impact}</small></div>
              <StatusBadge status={riskLabel(action.risk)} />
            </div>)}
          </div> : <EmptyState title="暂无建议" description="完成诊断后生成可验证的修复或控制变量实验。" />}
          {diagnosis?.related_resources?.length ? <div className="aiops-resource-chips">
            {diagnosis.related_resources.map((resource, index) => <span key={`${resource.type}-${resource.id}-${index}`}>{resource.type} · {resource.name || resource.id || "-"}</span>)}
          </div> : null}
        </section>
      </div>

      <section className="infra-panel aiops-incident-panel">
        <PanelHeader title="关联 Incident" action="warning / critical 诊断自动创建或复用" />
        <div className="aiops-toolbar">
          <select onChange={(event) => setSeverityFilter(event.target.value)} value={severityFilter}>
            <option value="all">所有严重性</option>
            <option value="critical">严重</option><option value="warning">警告</option><option value="info">提示</option>
          </select>
          <label className="aiops-search"><Search size={13} /><input onChange={(event) => setIncidentSearch(event.target.value)} placeholder="搜索 ID、标题或证据摘要" value={incidentSearch} /></label>
        </div>
        {incidentsQuery.isLoading ? <Skeleton rows={3} /> : incidentsQuery.isError ? <ErrorState error={incidentsQuery.error} onRetry={incidentsQuery.refetch} /> : filteredIncidents.length ? (
          <div className="aiops-incident-table">
            <div className="aiops-incident-row header"><span>ID</span><span>级别</span><span>标题</span><span>证据摘要</span><span>状态</span><span>操作</span></div>
            {filteredIncidents.map((incident) => <div className="aiops-incident-row" key={incident.id}>
              <span>{incident.id.slice(0, 8)}</span><StatusBadge status={severityLabel(incident.severity)} /><strong>{incident.title}</strong>
              <span title={incident.summary ?? ""}>{incident.summary || "诊断事件"}</span><StatusBadge status={statusLabel(incident.status)} />
              <span className="aiops-row-actions">
                <button disabled={busy || incident.status !== "open"} onClick={() => incidentTransitionMutation.mutate({ id: incident.id, action: "ack" })} title="确认 Incident" type="button"><CheckCircle size={13} /></button>
                <button disabled={busy || incident.status === "resolved"} onClick={() => incidentTransitionMutation.mutate({ id: incident.id, action: "resolve" })} title="解决 Incident" type="button"><Activity size={13} /></button>
              </span>
            </div>)}
          </div>
        ) : <EmptyState title="暂无关联 Incident" description="专项诊断仅在发现 warning 或 critical 异常时自动创建。" />}
      </section>
    </section>
  );
}

function inferenceContext(evidence?: InferenceEvidence["inference"]) {
  const run = evidence?.benchmark;
  const runtimeStatus = String(evidence?.runtime?.status ?? "stopped");
  const vllm = run?.config?.vllm ?? {};
  const model = String(vllm.model_id ?? vllm.model ?? evidence?.runtime?.model ?? modelIDFromEndpoint(run?.endpoint_id) ?? "未探测");
  return [
    { id: "model", label: "推理模型", value: model },
    { id: "run", label: "Benchmark Run", value: run?.run_id ?? "—" },
    { id: "data", label: "客服工作负载", value: run?.workload ?? "—" },
    { id: "runtime", label: "vLLM 运行时", value: runtimeStatus },
    { id: "cache", label: "Prefix Cache", value: run?.prefix_caching ? "开启" : "关闭" },
    { id: "prefill", label: "Chunked Prefill", value: run?.chunked_prefill ? "开启" : "关闭" },
    { id: "seqs", label: "max_num_seqs", value: run?.max_num_seqs == null ? "—" : String(run.max_num_seqs) },
    { id: "tokens", label: "Batch Token 预算", value: run?.max_num_batched_tokens == null ? "—" : String(run.max_num_batched_tokens) },
    { id: "baseline", label: "可比 Baseline", value: evidence?.baseline?.run_id ?? "—" },
    { id: "updated", label: "证据时间", value: formatTime(run?.updated_at) },
  ];
}

function modelIDFromEndpoint(endpoint?: string) {
  return endpoint?.replace(/-vllm$/, "") || "";
}

function trainingContext(evidence?: TrainingEvidence["training"]) {
  const job = evidence?.job;
  return [
    { id: "model", label: "基础模型", value: job?.base_model ?? "Qwen3.5-4B" },
    { id: "job", label: "训练任务", value: job?.name ?? "—" },
    { id: "data", label: "客服数据集", value: job?.dataset_uri ?? "DianJin-CSC-Data" },
    { id: "namespace", label: "K8s Namespace", value: job?.namespace ?? "—" },
    { id: "phase", label: "实时 Phase", value: evidence?.pytorch_job?.phase ?? job?.status ?? "—" },
    { id: "pod", label: "训练 Pod", value: evidence?.pod?.pod ?? "—" },
    { id: "logs", label: "日志采集", value: evidence?.pod?.available ? "可用" : evidence?.pod?.error ?? "等待 Pod" },
    { id: "replicas", label: "分布式配置", value: job ? `${job.workers} workers / ${job.gpus_per_worker} GPU` : "—" },
    { id: "artifact", label: "Adapter 产物", value: job?.output_artifact_uri ?? "尚未产出" },
    { id: "updated", label: "证据时间", value: formatTime(job?.updated_at) },
  ];
}

function parseClassification(evidence?: EvidenceItem[]) {
  const detail = evidence?.find((item) => item.source === "classifier")?.detail ?? "";
  const category = /category=([^,]+)/.exec(detail)?.[1] ?? "general";
  const severity = /severity=([^,]+)/.exec(detail)?.[1] ?? "info";
  return { category, severity };
}

function extractEmbeddedRca(raw?: string | null): Partial<Diagnosis> & { classification: { category: string; severity: string } } | null {
  const text = raw?.trim() ?? "";
  if (!text.startsWith("{") || !text.includes('"root_cause"')) return null;
  try {
    const parsed = JSON.parse(text) as Record<string, unknown>;
    if (typeof parsed.root_cause !== "string") return null;
    return {
      root_cause: parsed.root_cause,
      confidence: typeof parsed.confidence === "number" ? parsed.confidence : undefined,
      impact: typeof parsed.impact === "string" ? parsed.impact : undefined,
      evidence: Array.isArray(parsed.evidence) ? parsed.evidence as EvidenceItem[] : undefined,
      recommended_actions: Array.isArray(parsed.recommended_actions) ? parsed.recommended_actions as RecommendedAction[] : undefined,
      related_resources: Array.isArray(parsed.related_resources) ? parsed.related_resources as RelatedResource[] : undefined,
      classification: { category: stringField(parsed.category) || "general", severity: stringField(parsed.severity) || "info" },
    };
  } catch {
    const rootCause = jsonStringField(text, "root_cause");
    if (!rootCause) return null;
    return {
      root_cause: rootCause,
      confidence: numberField(text, "confidence"),
      impact: jsonStringField(text, "impact") || "LLM 返回的结构化内容不完整；请重新运行诊断以补齐证据与建议。",
      classification: { category: jsonStringField(text, "category") || "general", severity: jsonStringField(text, "severity") || "info" },
    };
  }
}

function stringField(value: unknown): string { return typeof value === "string" ? value : ""; }
function numberField(text: string, name: string): number | undefined {
  const value = new RegExp(`"${name}"\\s*:\\s*([0-9]+(?:\\.[0-9]+)?)`).exec(text)?.[1];
  return value == null ? undefined : Number(value);
}
function jsonStringField(text: string, name: string): string {
  const match = new RegExp(`"${name}"\\s*:\\s*("(?:\\\\.|[^"\\\\])*")`).exec(text)?.[1];
  if (!match) return "";
  try { return JSON.parse(match) as string; } catch { return ""; }
}

function metric(value?: number, unit = "") { return value == null ? "—" : `${fmt(value, 1)}${unit}`; }
function contextLabel(value?: number) { return value ? `${fmt(value / 1024, 0)}K` : "—"; }
function formatTime(value?: string) { return value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "—"; }
function describeError(error: unknown) { return error instanceof Error ? error.message : "unknown error"; }
function severityLabel(value: string) { return value === "critical" ? "严重" : value === "warning" ? "警告" : "提示"; }
function statusLabel(value: string) { return value === "resolved" ? "已解决" : value === "acknowledged" ? "处理中" : "待确认"; }
function riskLabel(value: string) { return value === "high" ? "高风险" : value === "medium" ? "中风险" : "低风险"; }
function categoryLabel(value: string) {
  const labels: Record<string, string> = {
    request_failure: "请求失败", quality_regression: "输出质量回归", scheduler_saturation: "调度排队",
    memory_pressure: "显存压力", decode_bottleneck: "Decode 瓶颈", prefill_bottleneck: "Prefill 瓶颈",
    training_oom: "训练 OOM", distributed_failure: "分布式通信", data_failure: "训练数据",
    artifact_failure: "产物归档", training_in_progress: "训练进行中", inference_healthy: "推理正常",
    training_healthy: "训练正常", general: "通用诊断",
  };
  return labels[value] ?? value;
}
function ContextIcon({ id }: { id: string }) {
  const Icon = id === "model" || id === "data" ? Database
    : id === "runtime" || id === "pod" ? Server
      : id === "logs" ? SquareTerminal
        : id === "run" || id === "job" ? FileText
          : id === "phase" ? Activity
            : id === "cache" || id === "prefill" ? Zap
              : id === "seqs" || id === "tokens" || id === "replicas" ? Cpu
                : id === "artifact" ? CheckCircle
                  : id === "baseline" ? Gauge
                    : AlertTriangle;
  return <Icon size={16} />;
}
