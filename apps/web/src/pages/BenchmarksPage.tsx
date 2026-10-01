import { useMemo, useState, type ChangeEvent, type ReactNode } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import { AlertTriangle, Bot, CheckCircle2, Eye, Flame, Play, Plus, RefreshCw, Rocket, Search, Server, SlidersHorizontal, Square } from "lucide-react";

import { KpiGrid, PageHeader, PanelHeader, StatusBadge } from "../components/common/PlatformPrimitives";
import { describeError, EmptyState, ErrorState } from "../components/common/FeedbackStates";
import { Drawer } from "../components/common/Drawer";
import { api } from "../lib/api";
import { fmt } from "../lib/format";
import { useDeliveryContext } from "../lib/useDeliveryContext";
import { useGoToPage } from "../lib/useGoToPage";
import type { BenchmarkEvent, BenchmarkTaskRow } from "../types/platform";
import type { ConfigItem } from "../types/ops";
import type { KpiItem } from "../types/ui";

const TERMINAL = new Set(["completed", "failed", "cancelled"]);

type InferenceRuntime = {
  available: boolean;
  status: "disabled" | "stopped" | "starting" | "ready" | "error";
  container_status?: string;
  container_name: string;
  model: string;
  endpoint: string;
  profile: string;
  prefix_caching: boolean;
  config: InferenceLaunchConfig;
  message?: string;
};

type ServiceProfile = "baseline" | "prefix_cache" | "scheduler";
const INFERENCE_MODELS = [
  { id: "qwen38-27b-fp8", label: "Qwen3.8-27B-FP8 · 双卡" },
  { id: "qwen36-27b-fp8", label: "Qwen3.6-27B-FP8 · 双卡" },
  { id: "qwen36-27b-awq", label: "Qwen3.6-27B-AWQ · 双卡" },
  { id: "qwen35-4b-customer", label: "Qwen3.5-4B Customer · 本地" },
];

type InferenceLaunchConfig = {
  max_num_seqs: number;
  max_num_batched_tokens: number;
  scheduling_policy: "fcfs" | "priority";
  max_num_partial_prefills: number;
  max_long_partial_prefills: number;
  long_prefill_token_threshold: number;
  stream_interval: number;
  prefix_caching: boolean;
  chunked_prefill: boolean;
  async_scheduling: boolean;
  scheduler_reserve_full_isl: boolean;
  disable_custom_all_reduce: boolean;
  gpu_memory_utilization: number;
  max_model_len: number;
  kv_cache_dtype: "auto" | "fp8";
  profiling: boolean;
  speculative_decoding: "none" | "ngram";
};

type BenchmarkForm = {
  endpoint_id: string;
  dataset: string;
  workload: string;
  routing_strategy: string;
  context_lengths: string;
  concurrency_levels: string;
  requests_per_level: string;
  max_tokens: string;
  max_num_seqs: string;
  max_num_batched_tokens: string;
  gpu_memory_utilization: string;
  max_model_len: string;
  prefix_caching: boolean;
  chunked_prefill: boolean;
};

type EvidenceScenario = {
  context_length?: number;
  concurrency?: number;
  success_rate?: number;
  quality_gate_pass_rate?: number;
  p95_ttft_ms?: number;
  p95_tpot_ms?: number;
  p95_ms?: number;
  output_tokens_per_second?: number;
  output_throughput_tokens_per_second?: number;
};
type BenchmarkEvidenceItem = {
  run_id?: string;
  updated_at?: string;
  prefix_caching?: boolean;
  chunked_prefill?: boolean;
  max_num_seqs?: number;
  max_num_batched_tokens?: number;
  summary?: { scenarios?: EvidenceScenario[] };
};
type InferenceEvidence = { inference?: { benchmark?: BenchmarkEvidenceItem; baseline?: BenchmarkEvidenceItem } };
type TextBenchmarkKey = Exclude<keyof BenchmarkForm, "prefix_caching" | "chunked_prefill">;

const defaultBenchmarkForm: BenchmarkForm = {
  endpoint_id: "qwen36-27b-fp8-vllm",
  dataset: "DianJin/DianJin-CSC-Data",
  workload: "customer_support_shared_prefix",
  routing_strategy: "least-request",
  context_lengths: "1024,2048",
  concurrency_levels: "1,2,4,8,16",
  requests_per_level: "16",
  max_tokens: "256",
  max_num_seqs: "8",
  max_num_batched_tokens: "4096",
  gpu_memory_utilization: "0.90",
  max_model_len: "4096",
  prefix_caching: false,
  chunked_prefill: true
};

export function BenchmarksPage() {
  const goTo = useGoToPage();
  const { context, update } = useDeliveryContext();
  const [runId, setRunId] = useState(context.benchmarkRunId ?? "");
  const [creating, setCreating] = useState(false);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [statusFilter, setStatusFilter] = useState("all");
  const [scenarioFilter, setScenarioFilter] = useState("all");
  const [localRows, setLocalRows] = useState<BenchmarkTaskRow[]>([]);
  const [serviceProfile, setServiceProfile] = useState<ServiceProfile>("baseline");
  const [modelID, setModelID] = useState("qwen36-27b-fp8");
  const [serviceConfigID, setServiceConfigID] = useState(context.deliveryKind === "inference" ? context.configItemId ?? "" : "");
  const [selectedRow, setSelectedRow] = useState<BenchmarkTaskRow | null>(null);

  const runtimeQuery = useQuery({
    queryKey: ["inference", "runtime"],
    queryFn: () => api<InferenceRuntime>("/api/inference/runtime"),
    refetchInterval: 2000,
    retry: false
  });
  const runtime = runtimeQuery.data;
  const runtimeActive = runtime?.status === "ready" || runtime?.status === "starting";
  const evidenceQuery = useQuery({
    queryKey: ["ai", "inference", "evidence", context.benchmarkRunId ?? "latest"],
    queryFn: () => api<InferenceEvidence>(`/api/ai/inference/evidence${context.benchmarkRunId ? `?run_id=${encodeURIComponent(context.benchmarkRunId)}` : ""}`),
    retry: false,
    refetchInterval: 15000,
  });
  const inferenceConfigs = useQuery({
    queryKey: ["config", "items", "inference"],
    queryFn: () => api<{ items: ConfigItem[] }>("/api/config/items"),
    select: (payload) => payload.items.filter((item) => /infer|serve|vllm/i.test(item.config_key))
  });

  const startRuntimeMutation = useMutation({
    mutationFn: () => api<InferenceRuntime>(serviceConfigID ? `/api/config/items/${serviceConfigID}/launch` : "/api/inference/runtime", {
      method: "POST",
      body: serviceConfigID ? JSON.stringify({ kind: "inference", model_id: modelID, version: context.configItemId === serviceConfigID && context.configVersion ? Number(context.configVersion) : undefined, operator: "frontend" }) : JSON.stringify(inferenceRuntimeRequestFor(modelID, serviceProfile))
    }),
    onSuccess: (payload) => {
      toast.success("vLLM 服务正在启动", { description: `${payload.model} · ${serviceProfile}` });
      setRunId("");
      update({ deliveryKind: "inference", modelId: payload.model || context.modelId, configItemId: serviceConfigID || context.configItemId, trainingJobId: null, benchmarkRunId: null, deploymentId: null });
      void runtimeQuery.refetch();
    },
    onError: (error) => toast.error("启动推理服务失败", { description: describeError(error) })
  });

  const stopRuntimeMutation = useMutation({
    mutationFn: () => api<InferenceRuntime>("/api/inference/runtime", { method: "DELETE", timeoutMs: 30_000 }),
    onSuccess: () => {
      toast.success("vLLM 服务已停止，两张 GPU 已释放");
      void runtimeQuery.refetch();
    },
    onError: (error) => toast.error("停止推理服务失败", { description: describeError(error) })
  });

  const startMutation = useMutation({
    mutationFn: (form: BenchmarkForm) =>
      api<{ run_id: string }>("/api/benchmarks/serving", {
        method: "POST",
        body: JSON.stringify({
          endpoint_id: form.endpoint_id,
          dataset: form.dataset,
          workload: form.workload,
          routing_strategy: form.routing_strategy,
          context_lengths: parsePositiveList(form.context_lengths, [1024, 2048]),
          concurrency_levels: parsePositiveList(form.concurrency_levels, [1, 2, 4, 8, 16]),
          requests_per_level: Number(form.requests_per_level) || 16,
          max_tokens: Number(form.max_tokens) || 256,
          vllm: {
            max_num_seqs: runtime?.config.max_num_seqs ?? (Number(form.max_num_seqs) || 8),
            max_num_batched_tokens: runtime?.config.max_num_batched_tokens ?? (Number(form.max_num_batched_tokens) || 4096),
            gpu_memory_utilization: runtime?.config.gpu_memory_utilization ?? (Number(form.gpu_memory_utilization) || 0.9),
            max_model_len: runtime?.config.max_model_len ?? (Number(form.max_model_len) || 4096),
            prefix_caching: runtime?.config.prefix_caching ?? form.prefix_caching,
            chunked_prefill: runtime?.config.chunked_prefill ?? form.chunked_prefill,
            enable_thinking: false,
            // 4B 模型使用 BF16/默认精度；只有 Qwen3.6 27B 运行时才带 FP8。
            quantization: runtime?.model === "qwen36-27b-fp8" ? "fp8" : runtime?.model === "qwen36-27b-awq" ? "awq" : undefined,
            scheduler: runtime?.config.scheduling_policy ?? "fcfs",
            async_scheduling: runtime?.config.async_scheduling,
            max_num_partial_prefills: runtime?.config.max_num_partial_prefills,
            max_long_partial_prefills: runtime?.config.max_long_partial_prefills,
            long_prefill_token_threshold: runtime?.config.long_prefill_token_threshold,
            scheduler_reserve_full_isl: runtime?.config.scheduler_reserve_full_isl,
            stream_interval: runtime?.config.stream_interval,
            disable_custom_all_reduce: runtime?.config.disable_custom_all_reduce,
            kv_cache_dtype: runtime?.config.kv_cache_dtype,
            profiling: runtime?.config.profiling,
            speculative_decoding: runtime?.config.speculative_decoding
          }
        })
      }),
    onSuccess: (payload, form) => {
      setRunId(payload.run_id);
      update({ deliveryKind: "inference", benchmarkRunId: payload.run_id, modelId: runtime?.model || context.modelId, trainingJobId: null });
      setCreating(false);
      setPage(1);
      setLocalRows((items) => [{
        id: payload.run_id,
        name: `实时压测 ${payload.run_id.slice(0, 8)}`,
        service: form.endpoint_id,
        version: "baseline",
        scenario: "DianJin CSC",
        context: form.context_lengths,
        concurrency: "运行中",
        p95: "-",
        ttft: "-",
        tpot: "-",
        throughput: "-",
        successRate: "-",
        qualityRate: "-",
        errorRate: "-",
        status: "运行中",
        startedAt: "刚刚",
        tone: "info"
      }, ...items]);
      toast.success("压测已启动", { description: `Run ${payload.run_id.slice(0, 8)}` });
    },
    onError: (error) => toast.error("启动压测失败", { description: describeError(error) })
  });

  const runQuery = useQuery({
    queryKey: ["benchmark", runId],
    queryFn: () => api<Record<string, unknown> & { status?: string }>(`/api/benchmarks/${runId}`),
    enabled: !!runId,
    refetchInterval: (q) => (TERMINAL.has(String(q.state.data?.status ?? "")) ? false : 2000)
  });
  const run = runQuery.data ?? null;
  const benchmarkActive = !!runId && !TERMINAL.has(String(run?.status ?? ""));

  const stopMutation = useMutation({
	mutationFn: () => api<{ status: string }>(`/api/benchmarks/${runId}`, { method: "DELETE" }),
	onSuccess: () => {
		toast.success("压测已停止");
		void runQuery.refetch();
	},
	onError: (error) => toast.error("停止压测失败", { description: describeError(error) })
  });

  const eventsQuery = useQuery({
    queryKey: ["benchmark", runId, "events"],
    queryFn: () => api<{ events: BenchmarkEvent[] }>(`/api/benchmarks/${runId}/events`),
    enabled: !!runId,
    select: (payload) => payload.events ?? [],
    refetchInterval: () => (TERMINAL.has(String(run?.status ?? "")) ? false : 2000)
  });

  const rows = useMemo(() => {
    const liveRows = deriveBenchmarkRows(eventsQuery.data);
    const persistedRows = deriveEvidenceRows(evidenceQuery.data?.inference?.benchmark);
    return liveRows.length ? liveRows : persistedRows.length ? persistedRows : localRows;
  }, [eventsQuery.data, evidenceQuery.data, localRows]);
  const statusOptions = useMemo(() => Array.from(new Set(rows.map((row) => row.status))).sort(), [rows]);
  const scenarioOptions = useMemo(() => Array.from(new Set(rows.map((row) => row.scenario))).sort(), [rows]);
  const filteredRows = useMemo(() => {
    const keyword = search.trim().toLowerCase();
    return rows.filter((row) =>
      (statusFilter === "all" || row.status === statusFilter) &&
      (scenarioFilter === "all" || row.scenario === scenarioFilter) &&
      (!keyword || `${row.name} ${row.service} ${row.version} ${row.scenario} ${row.status}`.toLowerCase().includes(keyword))
    );
  }, [rows, search, statusFilter, scenarioFilter]);
  const totalPages = Math.max(1, Math.ceil(filteredRows.length / pageSize));
  const currentPage = Math.min(page, totalPages);
  const pageRows = filteredRows.slice((currentPage - 1) * pageSize, currentPage * pageSize);
  const persistedScenarios = evidenceQuery.data?.inference?.benchmark?.summary?.scenarios ?? [];
  const kpis = useMemo(() => buildBenchmarkKpis(eventsQuery.data, persistedScenarios), [eventsQuery.data, persistedScenarios]);
  const comparison = useMemo(() => compareEvidence(evidenceQuery.data?.inference?.baseline, evidenceQuery.data?.inference?.benchmark), [evidenceQuery.data]);

  return (
    <section className="infra-page benchmarks-page benchmark-replica">
      <PageHeader
        title="推理服务控制面"
        subtitle={`Qwen3.6-27B 推理系列（当前运行时 ${runtime?.model || context.modelId || "FP8"}）：服务启停、压测、Profiling、瓶颈归因与控制变量优化`}
        actions={
          <>
			{context.benchmarkRunId || (runId && TERMINAL.has(String(run?.status ?? ""))) ? <>
			  <button className="console-refresh" onClick={() => goTo("aiOps", { deliveryKind: "inference", benchmarkRunId: context.benchmarkRunId || runId, diagnosisId: null })} type="button"><Bot size={13} /> 诊断本次压测</button>
			  <button className="console-refresh" onClick={() => goTo("release", { deliveryKind: "inference", benchmarkRunId: context.benchmarkRunId || runId })} type="button"><Rocket size={13} /> 进入发布验收</button>
			</> : null}
			{benchmarkActive ? (
			  <button className="console-refresh" disabled={stopMutation.isPending} onClick={() => stopMutation.mutate()} type="button">
				<Square size={13} /> {stopMutation.isPending ? "停止中..." : "停止压测"}
			  </button>
			) : null}
			<button className="console-refresh primary" disabled={startMutation.isPending || runtime?.status !== "ready"} onClick={() => setCreating(true)} title={runtime?.status === "ready" ? "新建推理压测" : "请先启动并等待 vLLM 服务就绪"} type="button">
			  <Plus size={14} /> {startMutation.isPending ? "启动中..." : "新建压测"}
			</button>
		  </>
        }
      />

      <div className="benchmark-runtime-bar">
        <div className="benchmark-runtime-copy">
          <span className={`benchmark-runtime-state ${runtime?.status ?? "unknown"}`} title={runtime?.message || runtime?.container_status || "Node Agent workload 状态"}>
            <Server size={13} /> vLLM {runtimeLabel(runtime?.status)}
          </span>
          <span>{runtime?.model || modelID} · {runtime?.model?.includes("4b") || modelID.includes("4b") ? "单卡 TP=1" : "双卡 TP=2"} · OpenAI-Compatible API :8020</span>
        </div>
        <div className="benchmark-runtime-actions">
          <select aria-label="推理模型" className="benchmark-runtime-profile" disabled={runtimeActive} value={runtimeActive ? (runtime?.model || modelID) : modelID} onChange={(event) => { setModelID(event.target.value); setServiceConfigID(""); }}>
            {INFERENCE_MODELS.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
          </select>
          {(inferenceConfigs.data?.length ?? 0) > 0 ? <select
            aria-label="配置中心推理模板"
            className="benchmark-runtime-profile"
            disabled={runtimeActive}
            onChange={(event) => {
              const itemID = event.target.value;
              const item = inferenceConfigs.data?.find((candidate) => candidate.id === itemID);
              setServiceConfigID(itemID);
              update({
                deliveryKind: "inference",
                configItemId: itemID || null,
                configVersion: itemID ? String(item?.active_version ?? "") : null,
                trainingJobId: null,
              });
            }}
            value={serviceConfigID}
          >
            <option value="">内置运行时配置</option>
            {inferenceConfigs.data!.map((item) => <option key={item.id} value={item.id}>{item.config_key} · v{item.active_version}</option>)}
          </select> : null}
          <select
            aria-label="推理服务配置"
            className="benchmark-runtime-profile"
            disabled={runtimeActive}
            onChange={(event) => {
              setServiceProfile(event.target.value as ServiceProfile);
              setServiceConfigID("");
              update({ deliveryKind: "inference", configItemId: null, configVersion: null, trainingJobId: null });
            }}
            value={runtimeActive ? runtimeServiceProfile(runtime?.profile) : serviceProfile}
          >
            <option value="baseline">Baseline</option>
            <option value="prefix_cache">Prefix caching（均衡）</option>
            <option value="scheduler">高并发 TTFT/吞吐</option>
          </select>
          {runtimeActive ? (
            <button
              className="console-refresh"
              disabled={benchmarkActive || stopRuntimeMutation.isPending}
              onClick={() => stopRuntimeMutation.mutate()}
              title={benchmarkActive ? "请先停止正在运行的压测" : "停止 vLLM 并释放 GPU"}
              type="button"
            >
              <Square size={13} /> {stopRuntimeMutation.isPending ? "停止中..." : "停止服务"}
            </button>
          ) : (
            <button
              className="console-refresh"
              disabled={startRuntimeMutation.isPending || runtimeQuery.isError}
              onClick={() => startRuntimeMutation.mutate()}
              title={serviceConfigID ? "按配置中心不可变版本启动，并关联部署/审计/AIOps" : "在本机双卡启动固定 Qwen3.6 vLLM workload"}
              type="button"
            >
              <Play size={13} /> {startRuntimeMutation.isPending ? "启动中..." : serviceConfigID ? "按配置启动" : "启动服务"}
            </button>
          )}
        </div>
      </div>
      {run?.status === "failed" ? <div className="benchmark-failure-banner"><AlertTriangle size={16} /><span><strong>压测失败</strong><small>{String(run.error || "未记录具体原因，请查看事件明细和日志检索")}</small></span></div> : null}

      <div className="inference-profile-strip">
        <Profile active={(runtimeActive ? runtimeServiceProfile(runtime?.profile) : serviceProfile) === "baseline"} icon={Server} title="Baseline" params="seqs 8 · tokens 4096" detail="Prefix Cache OFF · Chunked Prefill ON" />
        <Profile active={(runtimeActive ? runtimeServiceProfile(runtime?.profile) : serviceProfile) === "prefix_cache"} icon={CheckCircle2} title="Prefix Cache" params="seqs 8 · tokens 4096" detail="仅开启共享前缀复用，适合公平消融" />
        <Profile active={(runtimeActive ? runtimeServiceProfile(runtime?.profile) : serviceProfile) === "scheduler"} icon={SlidersHorizontal} title="Scheduler" params="seqs 16 · tokens 8192" detail="Prefix Cache + 异步调度 + 批处理参数" />
        <Profile active={Boolean(runtime?.config.profiling)} icon={Flame} title="Profiling" params={runtime?.config.profiling ? "ON" : "按需开启"} detail="用于定位 prefill / decode / kernel / 通信瓶颈" />
      </div>

      <KpiGrid className="benchmark-kpi-strip" items={kpis} />

      <section className="infra-panel benchmark-comparison-panel">
        <PanelHeader title="Baseline 与当前正式结果" action={comparison.available ? "同 endpoint / workload 可比" : "等待可比 baseline"} />
        {evidenceQuery.isLoading ? <div className="benchmark-compare-empty">加载正式压测证据...</div> : !comparison.available ? (
          <div className="benchmark-compare-empty">完成一轮 Baseline 和一轮 Prefix Cache / Scheduler 压测后，此处自动计算 TTFT、TPOT、P95 与吞吐变化。</div>
        ) : (
          <div className="benchmark-compare-grid">
            {comparison.metrics.map((item) => <div className={item.improved ? "improved" : item.delta === null ? "neutral" : "regressed"} key={item.label}><span>{item.label}</span><strong>{item.current}</strong><small>Baseline {item.baseline}</small><em>{item.delta == null ? "不可计算" : `${item.delta > 0 ? "+" : ""}${fmt(item.delta, 1)}%`}</em></div>)}
            <div className="benchmark-compare-gate"><StatusBadge status={comparison.gatePassed ? "质量门禁通过" : "质量门禁未通过"} /><small>只有请求成功率与输出正确性均 ≥ 99% 时，性能提升才可作为发布依据。</small></div>
          </div>
        )}
      </section>

      <section className="infra-panel benchmark-main-panel">
        <div className="benchmark-tabs">
          {["压测任务"].map((t) => (
            <button className="active" key={t} type="button">{t}</button>
          ))}
        </div>
        <div className="benchmark-toolbar">
          <select className="benchmark-filter" onChange={(event) => { setStatusFilter(event.target.value); setPage(1); }} value={statusFilter}>
            <option value="all">全部状态</option>
            {statusOptions.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
          </select>
          <select className="benchmark-filter" onChange={(event) => { setScenarioFilter(event.target.value); setPage(1); }} value={scenarioFilter}>
            <option value="all">全部场景</option>
            {scenarioOptions.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
          </select>
          <label className="benchmark-search-box"><Search size={14} /> <input onChange={(event) => {
            setSearch(event.target.value);
            setPage(1);
          }} placeholder="搜索任务名称、服务、版本..." value={search} /></label>
          <button type="button" onClick={() => runQuery.refetch()}><RefreshCw size={13} /></button>
          <button className="primary" disabled={startMutation.isPending || runtime?.status !== "ready"} onClick={() => setCreating(true)} title={runtime?.status === "ready" ? "新建推理压测" : "请先启动并等待 vLLM 服务就绪"} type="button">
            <Plus size={13} /> 新建压测
          </button>
        </div>
        {runId ? <div className="benchmark-run-meta">Run ID: {runId} · 状态: {String(run?.status ?? "running")}</div> : null}
        {runId && eventsQuery.isError ? (
          <ErrorState error={eventsQuery.error} onRetry={() => eventsQuery.refetch()} />
        ) : filteredRows.length === 0 ? (
          <EmptyState title="尚未发起压测" description={search ? "无匹配结果" : "点击「新建压测」对某个 endpoint 发起 serving 基准"} />
        ) : (
          <BenchmarkTable onSelect={setSelectedRow} rows={pageRows} />
        )}
        <div className="benchmark-pagination">
          <span>共 {filteredRows.length} 条</span>
          <button disabled={currentPage === 1} onClick={() => setPage((value) => Math.max(1, value - 1))} type="button">‹</button>
          {Array.from({ length: totalPages }, (_, index) => index + 1).map((item) => (
            <button className={item === currentPage ? "active" : undefined} key={item} onClick={() => setPage(item)} type="button">{item}</button>
          ))}
          <button disabled={currentPage === totalPages} onClick={() => setPage((value) => Math.min(totalPages, value + 1))} type="button">›</button>
          <button onClick={() => { setPageSize((value) => (value === 10 ? 20 : value === 20 ? 50 : 10)); setPage(1); }} type="button">{pageSize} 条/页</button>
        </div>
      </section>

      <NewBenchmarkDrawer
        busy={startMutation.isPending}
        key={`${runtime?.profile ?? serviceProfile}-${runtime?.status ?? "unknown"}`}
        onClose={() => setCreating(false)}
        onSubmit={(form) => startMutation.mutate(form)}
        open={creating}
        runtimeConfig={runtime?.config}
        runtimeModel={runtime?.model}
      />
      {selectedRow ? <BenchmarkDetailDrawer row={selectedRow} onClose={() => setSelectedRow(null)} /> : null}
    </section>
  );
}

function BenchmarkTable({ rows, onSelect }: { rows: BenchmarkTaskRow[]; onSelect: (row: BenchmarkTaskRow) => void }) {
  return (
    <div className="benchmark-table">
      <div className="benchmark-table-row header">
        <span>任务名称</span>
        <span>服务 / 应用</span>
        <span>数据集</span>
        <span>上下文</span>
        <span>并发</span>
        <span>P95 延迟</span>
        <span>TTFT</span>
        <span>TPOT</span>
        <span>吞吐量</span>
        <span>成功率</span>
        <span>状态</span>
        <span>质量门禁</span>
        <span>操作</span>
      </div>
      {rows.map((row) => (
        <div className="benchmark-table-row" key={row.id}>
          <span className="benchmark-name-cell">
            <strong>{row.name}</strong>
            <small>基准对比</small>
          </span>
          <span>{row.service}</span>
          <span>{row.scenario}</span>
          <span>{row.context}</span>
          <span>{row.concurrency}</span>
          <strong className={row.p95.startsWith("8") ? "warning-text" : "success-text"}>{row.p95}</strong>
          <strong className={row.ttft.startsWith("5") ? "warning-text" : "success-text"}>{row.ttft}</strong>
          <strong>{row.tpot}</strong>
          <span>{row.throughput}</span>
          <strong className={row.successRate.startsWith("100") ? "success-text" : "danger-text"}>{row.successRate}</strong>
          <StatusBadge status={row.status} />
          <span>{row.qualityRate}</span>
          <span className="benchmark-row-actions">
            <button aria-label={`${row.name} 详情`} onClick={() => onSelect(row)} title="查看场景指标" type="button"><Eye color="#2563eb" size={13} strokeWidth={2.5} /></button>
          </span>
        </div>
      ))}
    </div>
  );
}

function buildBenchmarkKpis(events: BenchmarkEvent[] | undefined, persisted: EvidenceScenario[] = []): KpiItem[] {
  const list = events ?? [];
  const requestEvents = list.filter((event) => event.event_type === "request");
  const requestErrors = requestEvents.filter((event) => event.payload?.error).length;
  const latestSummary = ([...list].reverse().find((item) => item.event_type === "scenario_summary")?.payload?.summary ?? persisted[persisted.length - 1]) as Record<string, unknown> | undefined;
  const p95 = latestSummary?.p95_latency_seconds != null
    ? Number(latestSummary.p95_latency_seconds) * 1000
    : latestSummary?.p95_ms != null ? Number(latestSummary.p95_ms) : null;
  const ttft = latestSummary?.p95_ttft_ms != null ? Number(latestSummary.p95_ttft_ms) : null;
  const tpot = latestSummary?.p95_tpot_ms != null ? Number(latestSummary.p95_tpot_ms) : null;
  const throughput = latestSummary?.output_tokens_per_second != null ? Number(latestSummary.output_tokens_per_second) : null;
  const qualityRate = latestSummary?.quality_gate_pass_rate != null ? Number(latestSummary.quality_gate_pass_rate) * 100 : null;
  const persistedSuccess = latestSummary?.success_rate == null ? null : Number(latestSummary.success_rate) * 100;
  const errorRate = requestEvents.length ? (requestErrors / requestEvents.length) * 100 : persistedSuccess == null ? null : 100 - persistedSuccess;
  const countLabel = requestEvents.length ? "请求数" : "场景数";
  const countValue = requestEvents.length || persisted.length;
  return [
    { id: "tasks", label: countLabel, value: String(countValue), detail: requestEvents.length ? "当前 Run" : "最近正式结果", trend: [] },
    { id: "passRate", label: "通过率", value: errorRate == null ? "—" : errorRate ? `${fmt(100 - errorRate, 1)}%` : "100%", detail: "当前任务", trend: [], tone: errorRate ? "warning" : "success" },
    { id: "qualityRate", label: "质量门禁", value: qualityRate == null ? "—" : `${fmt(qualityRate, 1)}%`, detail: "完整输出与安全检查", trend: [], tone: qualityRate != null && qualityRate < 99 ? "warning" : "success" },
    { id: "p95", label: "P95 延迟", value: p95 == null ? "—" : `${fmt(p95, 0)}ms`, detail: "SLO ≤ 300ms", trend: [], tone: p95 != null && p95 > 300 ? "warning" : "success" },
    { id: "ttft", label: "TTFT", value: ttft == null ? "—" : `${fmt(ttft, 0)}ms`, detail: "SLO ≤ 200ms", trend: [], tone: ttft != null && ttft > 200 ? "warning" : "success" },
    { id: "tpot", label: "TPOT", value: tpot == null ? "—" : `${fmt(tpot, 1)}ms`, detail: "生成阶段", trend: [], tone: tpot != null && tpot > 80 ? "warning" : "success" },
    { id: "throughput", label: "吞吐量 (tokens/s)", value: throughput == null ? "—" : fmt(throughput, 0), detail: "解码吞吐", trend: [], tone: "success" },
  ];
}

function deriveEvidenceRows(item?: BenchmarkEvidenceItem): BenchmarkTaskRow[] {
  return (item?.summary?.scenarios ?? []).map((summary, index) => {
    const success = summary.success_rate == null ? null : Number(summary.success_rate) * 100;
    const quality = summary.quality_gate_pass_rate == null ? null : Number(summary.quality_gate_pass_rate) * 100;
    const passed = success != null && success >= 99 && quality != null && quality >= 99;
    return {
      id: `persisted-${item?.run_id ?? "latest"}-${index}`,
      name: `场景 ${index + 1}`,
      service: "qwen36-27b-fp8-vllm",
      version: item?.prefix_caching ? "optimized" : "baseline",
      scenario: "DianJin CSC",
      context: summary.context_length == null ? "-" : `${summary.context_length} tokens`,
      concurrency: String(summary.concurrency ?? "-"),
      p95: metricMs(summary.p95_ms),
      ttft: metricMs(summary.p95_ttft_ms),
      tpot: metricMs(summary.p95_tpot_ms, 1),
      throughput: throughputValue(summary),
      successRate: success == null ? "-" : `${fmt(success, 1)}%`,
      qualityRate: quality == null ? "-" : `${fmt(quality, 1)}%`,
      errorRate: success == null ? "-" : `${fmt(100 - success, 1)}%`,
      status: passed ? "通过" : "失败",
      startedAt: item?.updated_at || "-",
      tone: passed ? "success" : "danger",
    };
  });
}

function throughputValue(summary: EvidenceScenario): string {
  const value = summary.output_throughput_tokens_per_second ?? summary.output_tokens_per_second;
  return value == null ? "-" : `${fmt(value, 1)} tok/s`;
}

function Profile({ active, icon: Icon, title, params, detail }: { active: boolean; icon: typeof Server; title: string; params: string; detail: string }) {
  return <div className={active ? "active" : undefined}><Icon size={16} /><span><strong>{title}</strong><small>{params}</small><em>{detail}</em></span></div>;
}

function compareEvidence(baseline?: BenchmarkEvidenceItem, current?: BenchmarkEvidenceItem) {
  const base = baseline?.summary?.scenarios ?? [];
  const next = current?.summary?.scenarios ?? [];
  if (!base.length || !next.length) return { available: false, gatePassed: false, metrics: [] as CompareMetric[] };
  const worst = (rows: EvidenceScenario[], key: keyof EvidenceScenario) => Math.max(...rows.map((row) => Number(row[key] ?? 0)));
  const meanThroughput = (rows: EvidenceScenario[]) => rows.reduce((sum, row) => sum + Number(row.output_throughput_tokens_per_second ?? row.output_tokens_per_second ?? 0), 0) / rows.length;
  const metrics: CompareMetric[] = [
    compareMetric("TTFT P95 最大值", worst(base, "p95_ttft_ms"), worst(next, "p95_ttft_ms"), "ms", false, 0),
    compareMetric("TPOT P95 最大值", worst(base, "p95_tpot_ms"), worst(next, "p95_tpot_ms"), "ms", false, 1),
    compareMetric("端到端 P95 最大值", worst(base, "p95_ms"), worst(next, "p95_ms"), "ms", false, 0),
    compareMetric("平均生成吞吐", meanThroughput(base), meanThroughput(next), "tok/s", true, 1),
  ];
  const gatePassed = next.every((row) => Number(row.success_rate ?? 0) >= 0.99 && Number(row.quality_gate_pass_rate ?? 0) >= 0.99);
  return { available: true, gatePassed, metrics };
}

type CompareMetric = { label: string; baseline: string; current: string; delta: number | null; improved: boolean };

function compareMetric(label: string, baseline: number, current: number, unit: string, higherIsBetter: boolean, digits: number): CompareMetric {
  const delta = baseline > 0 ? (higherIsBetter ? ((current - baseline) / baseline) * 100 : ((baseline - current) / baseline) * 100) : null;
  return { label, baseline: `${fmt(baseline, digits)} ${unit}`, current: `${fmt(current, digits)} ${unit}`, delta, improved: delta != null && delta >= 0 };
}

function BenchmarkDetailDrawer({ row, onClose }: { row: BenchmarkTaskRow; onClose: () => void }) {
  const fields = [
    ["上下文长度", row.context], ["并发", row.concurrency], ["P95 端到端延迟", row.p95],
    ["P95 TTFT", row.ttft], ["P95 TPOT", row.tpot], ["生成吞吐", row.throughput],
    ["请求成功率", row.successRate], ["输出正确性门禁", row.qualityRate], ["错误率", row.errorRate],
  ];
  return <Drawer open title={row.name} subtitle={`${row.service} · ${row.scenario}`} onClose={onClose}><div className="benchmark-detail-list">{fields.map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}<div><span>验收结论</span><StatusBadge status={row.status} /></div></div></Drawer>;
}

function deriveBenchmarkRows(events: BenchmarkEvent[] | undefined): BenchmarkTaskRow[] {
  if (!events?.length) return [];
  const started = events.find((event) => event.event_type === "started");
  const endpoint = String(started?.payload?.endpoint ?? "-");
  return events.filter((event) => event.event_type === "scenario_summary").map((event, index) => {
    const summary = event.payload?.summary as Record<string, unknown> | undefined;
    const success = summary?.success_rate == null ? null : Number(summary.success_rate) * 100;
    return {
      id: `scenario-${index}`,
      name: `场景 ${index + 1}`,
      service: endpoint,
      version: "baseline",
      scenario: "DianJin CSC",
      context: summary?.context_length == null ? "-" : `${summary.context_length} tokens`,
      concurrency: String(summary?.concurrency ?? "-"),
      p95: metricMs(summary?.p95_ms),
      ttft: metricMs(summary?.p95_ttft_ms),
      tpot: metricMs(summary?.p95_tpot_ms, 1),
      throughput: summary?.output_tokens_per_second == null ? "-" : `${fmt(Number(summary.output_tokens_per_second), 1)} tok/s`,
      successRate: success == null ? "-" : `${fmt(success, 1)}%`,
      qualityRate: summary?.quality_gate_pass_rate == null ? "-" : `${fmt(Number(summary.quality_gate_pass_rate) * 100, 1)}%`,
      errorRate: success == null ? "-" : `${fmt(100 - success, 1)}%`,
      status: success != null && success >= 99 && Number(summary?.quality_gate_pass_rate ?? 0) >= 0.99 ? "通过" : "失败",
      startedAt: new Date(event.created_at).toLocaleTimeString(),
      tone: success != null && success >= 99 ? "success" : "danger",
    };
  });
}

function NewBenchmarkDrawer({
  busy,
  onClose,
  onSubmit,
  open,
  runtimeConfig,
  runtimeModel
}: {
  busy: boolean;
  onClose: () => void;
  onSubmit: (form: BenchmarkForm) => void;
  open: boolean;
  runtimeConfig?: InferenceLaunchConfig;
  runtimeModel?: string;
}) {
  const [form, setForm] = useState<BenchmarkForm>(() => ({ ...benchmarkFormForRuntime(runtimeConfig), endpoint_id: runtimeModel ? `${runtimeModel}-vllm` : defaultBenchmarkForm.endpoint_id }));
  if (!open) return null;
  const set = (key: TextBenchmarkKey) => (event: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => setForm({ ...form, [key]: event.target.value });
  const toggle = (key: "prefix_caching" | "chunked_prefill") => (event: ChangeEvent<HTMLInputElement>) => setForm({ ...form, [key]: event.target.checked });
  return (
    <Drawer
      open={open}
      onClose={onClose}
      title="新建压测"
      subtitle="提交到 /api/benchmarks/serving"
      footer={
        <>
          <button className="ghost-button" onClick={onClose} type="button">取消</button>
          <button className="primary-button" disabled={busy} onClick={() => onSubmit(form)} type="button">
            {busy ? "启动中..." : "启动压测"}
          </button>
        </>
      }
    >
      <div className="infra-drawer-section-title">目标与场景</div>
      <BenchmarkField label="Endpoint"><input className="drawer-input" value={form.endpoint_id} onChange={set("endpoint_id")} /></BenchmarkField>
      <BenchmarkField label="数据集"><select className="drawer-input" value={form.dataset} onChange={set("dataset")}>
          <option value="DianJin/DianJin-CSC-Data">DianJin-CSC-Data（中文客服）</option>
        </select></BenchmarkField>
      <BenchmarkField label="负载场景"><select className="drawer-input" value={form.workload} onChange={set("workload")}>
          <option value="faq_short">faq_short</option>
          <option value="mixed_peak">mixed_peak</option>
          <option value="customer_support_shared_prefix">customer_support_shared_prefix</option>
          <option value="ticket_long_context">ticket_long_context</option>
        </select></BenchmarkField>
      <BenchmarkField label="路由策略"><select className="drawer-input" value={form.routing_strategy} onChange={set("routing_strategy")}>
          <option value="least-request">least-request</option>
          <option value="round-robin">round-robin</option>
          <option value="least_queue">least_queue</option>
        </select></BenchmarkField>

      <div className="infra-drawer-section-title">压测参数</div>
      <BenchmarkField label="上下文长度"><input className="drawer-input" value={form.context_lengths} onChange={set("context_lengths")} /></BenchmarkField>
      <BenchmarkField label="并发级别"><input className="drawer-input" value={form.concurrency_levels} onChange={set("concurrency_levels")} /></BenchmarkField>
      <BenchmarkField label="每级请求数"><input className="drawer-input" value={form.requests_per_level} onChange={set("requests_per_level")} /></BenchmarkField>
      <BenchmarkField label="最大输出 token"><input className="drawer-input" value={form.max_tokens} onChange={set("max_tokens")} /></BenchmarkField>

      <div className="infra-drawer-section-title">vLLM 参数快照</div>
      <BenchmarkField label="max_num_seqs"><input className="drawer-input" disabled value={form.max_num_seqs} onChange={set("max_num_seqs")} /></BenchmarkField>
      <BenchmarkField label="max_num_batched_tokens"><input className="drawer-input" disabled value={form.max_num_batched_tokens} onChange={set("max_num_batched_tokens")} /></BenchmarkField>
      <BenchmarkField label="gpu_memory_utilization"><input className="drawer-input" disabled value={form.gpu_memory_utilization} onChange={set("gpu_memory_utilization")} /></BenchmarkField>
      <BenchmarkField label="max_model_len"><input className="drawer-input" disabled value={form.max_model_len} onChange={set("max_model_len")} /></BenchmarkField>
      <div className="benchmark-toggle-row">
        <label><input checked={form.prefix_caching} disabled onChange={toggle("prefix_caching")} type="checkbox" /> 前缀缓存</label>
        <label title="Qwen3.6 在当前 vLLM 版本下要求开启"><input checked={form.chunked_prefill} disabled onChange={toggle("chunked_prefill")} type="checkbox" /> Chunked prefill</label>
      </div>
    </Drawer>
  );
}

function BenchmarkField({ children, label }: { children: ReactNode; label: string }) {
  return <label className="benchmark-form-field"><span>{label}</span>{children}</label>;
}

function parsePositiveList(value: string, fallback: number[]) {
  const parsed = value.split(",").map((item) => Number(item.trim())).filter((item) => Number.isFinite(item) && item > 0);
  return parsed.length ? parsed : fallback;
}

function benchmarkFormForRuntime(config?: InferenceLaunchConfig): BenchmarkForm {
  if (!config) return defaultBenchmarkForm;
  return {
    ...defaultBenchmarkForm,
    max_num_seqs: String(config.max_num_seqs),
    max_num_batched_tokens: String(config.max_num_batched_tokens),
    gpu_memory_utilization: String(config.gpu_memory_utilization),
    max_model_len: String(config.max_model_len),
    prefix_caching: config.prefix_caching,
    chunked_prefill: config.chunked_prefill
  };
}

function inferenceRuntimeRequestFor(modelID: string, requestedProfile: ServiceProfile): Record<string, unknown> {
  const is4B = modelID.includes("4b");
  // 4B 模型不能沿用 baseline/prefix_cache 的双卡默认值；统一转成显式单卡 scheduler。
  if (is4B || requestedProfile === "scheduler") {
    return {
      model_id: modelID,
      profile: "scheduler",
      tensor_parallel_size: is4B ? 1 : 2,
      pipeline_parallel_size: 1,
      max_num_seqs: requestedProfile === "scheduler" ? 16 : 8,
      max_num_batched_tokens: requestedProfile === "scheduler" ? 8192 : 4096,
      scheduling_policy: "fcfs",
      max_num_partial_prefills: 1,
      max_long_partial_prefills: 1,
      long_prefill_token_threshold: 0,
      stream_interval: 1,
      prefix_caching: is4B ? false : true,
      async_scheduling: true,
      scheduler_reserve_full_isl: true,
      disable_custom_all_reduce: true,
      gpu_memory_utilization: 0.9,
      max_model_len: 4096,
      kv_cache_dtype: "auto",
      speculative_decoding: "none"
    };
  }
  return { profile: requestedProfile, model_id: modelID };
}

function runtimeServiceProfile(profile?: string): ServiceProfile {
  return profile?.startsWith("scheduler") ? "scheduler" : profile === "prefix_cache" ? "prefix_cache" : "baseline";
}

function metricMs(value: unknown, digits = 0) {
  return value == null ? "-" : `${fmt(Number(value), digits)}ms`;
}

function runtimeLabel(status?: InferenceRuntime["status"]) {
  switch (status) {
    case "ready": return "就绪";
    case "starting": return "启动中";
    case "stopped": return "已停止";
    case "disabled": return "未启用";
    case "error": return "异常";
    default: return "不可用";
  }
}
