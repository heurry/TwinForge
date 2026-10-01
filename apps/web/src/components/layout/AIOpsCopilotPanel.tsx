import { useMemo, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { AlertTriangle, CheckCircle, RefreshCw, Search, Send, Sparkles, X } from "lucide-react";

import { api } from "../../lib/api";
import { fmt } from "../../lib/format";
import { cn } from "../../lib/utils";
import type { Page } from "../../types/navigation";
import type { Incident } from "../../types/ops";
import type { Metrics } from "../../types/platform";

type DiagnosisListItem = {
  id: string;
  question: string;
  status: string;
  root_cause?: string | null;
  confidence?: number | null;
  impact?: string | null;
  model_id?: string | null;
  endpoint_id?: string | null;
  latency_ms?: number | null;
  error?: string | null;
  created_at?: string;
};

type DiagnosisDetail = DiagnosisListItem & {
  evidence?: unknown[];
  recommended_actions?: unknown[];
  related_resources?: unknown[];
};

const DEFAULT_QUESTION = "请基于当前 metrics、Kubernetes、配置变更和部署记录诊断生产服务异常的根因，并给出修复建议。";

export function AIOpsCopilotPanel({
  hidden = false,
  setPage
}: {
  hidden?: boolean;
  setPage: (page: Page) => void;
}) {
  const queryClient = useQueryClient();
  const [question, setQuestion] = useState("");
  const [selectedDiagnosisId, setSelectedDiagnosisId] = useState<string | null>(null);

  const incidentsQuery = useQuery({
    queryKey: ["incidents", "aiops"],
    queryFn: () => api<{ incidents: Incident[] }>("/api/incidents"),
    enabled: !hidden,
    refetchInterval: hidden ? false : 15000
  });
  const diagnosesQuery = useQuery({
    queryKey: ["ai", "diagnoses"],
    queryFn: () => api<{ diagnoses: DiagnosisListItem[] }>("/api/ai/diagnoses?limit=20"),
    enabled: !hidden,
    refetchInterval: hidden ? false : 15000
  });
  const metricsQuery = useQuery({
    queryKey: ["metrics", "current"],
    queryFn: () => api<Metrics>("/api/metrics/current"),
    enabled: !hidden,
    refetchInterval: hidden ? false : 5000
  });
  const detailQuery = useQuery({
    queryKey: ["ai", "diagnosis", selectedDiagnosisId],
    queryFn: () => api<{ diagnosis: DiagnosisDetail }>(`/api/ai/diagnoses/${selectedDiagnosisId}`),
    enabled: !hidden && !!selectedDiagnosisId
  });

  const incidents = incidentsQuery.data?.incidents ?? [];
  const openIncidents = incidents.filter((item) => item.status !== "resolved");
  const criticalCount = openIncidents.filter((item) => severityLevel(item.severity) === "critical").length;
  const latestIncident = openIncidents[0] ?? incidents[0] ?? null;
  const latestDiagnosis = diagnosesQuery.data?.diagnoses?.[0] ?? null;
  const activeDiagnosis = detailQuery.data?.diagnosis ?? latestDiagnosis;
  const confidence = activeDiagnosis?.confidence ?? latestDiagnosis?.confidence ?? null;
  const metrics = metricsQuery.data;
  const p95 = metrics?.p95_latency_ms ?? null;
  const ttft = metrics?.p95_ttft_ms ?? null;
  const errorRate = metrics?.error_rate ?? null;
  const targetPod = firstKey(metrics?.target_pod_counts) ?? "当前生产服务";

  const createIncidentMutation = useMutation({
    mutationFn: () => api<{ id: string; status: string }>("/api/incidents", {
      method: "POST",
      body: JSON.stringify({
        title: diagnosisTitle(latestDiagnosis) || "AI Ops 检测到 Serving 风险",
        severity: criticalCount ? "critical" : p95 && p95 > 200 ? "warning" : "info",
        summary: latestDiagnosis?.root_cause || metricsSummary(metrics) || "由 AI Copilot 基于当前平台状态创建",
        operator: "frontend"
      })
    }),
    onSuccess: (payload) => {
      toast.success("Incident 已创建", { description: payload.id.slice(0, 8) });
      queryClient.invalidateQueries({ queryKey: ["incidents", "aiops"] });
    },
    onError: (error) => toast.error("创建 Incident 失败", { description: describeError(error) })
  });

  const diagnoseMutation = useMutation({
    mutationFn: (nextQuestion: string) => api<{ diagnosis?: DiagnosisDetail; mode?: string }>("/api/ai/diagnose:agent", {
      method: "POST",
      body: JSON.stringify({
        question: nextQuestion,
        max_tokens: 768,
        temperature: 0.2,
        operator: "frontend"
      }),
      timeoutMs: 60_000
    }),
    onSuccess: (payload) => {
      const id = payload.diagnosis?.id;
      if (id) setSelectedDiagnosisId(id);
      setQuestion("");
      toast.success("AI 诊断已完成", { description: id?.slice(0, 8) ?? payload.mode ?? "diagnosis" });
      queryClient.invalidateQueries({ queryKey: ["ai", "diagnoses"] });
    },
    onError: (error) => toast.error("AI 诊断失败", { description: describeError(error) })
  });

  const incidentTransitionMutation = useMutation({
    mutationFn: ({ id, action }: { id: string; action: "ack" | "resolve" }) => api<{ id: string; status: string }>(`/api/incidents/${id}/${action}`, {
      method: "POST",
      body: JSON.stringify({ operator: "frontend" })
    }),
    onSuccess: (payload) => {
      toast.success("Incident 状态已更新", { description: `${payload.id.slice(0, 8)} · ${payload.status}` });
      queryClient.invalidateQueries({ queryKey: ["incidents", "aiops"] });
    },
    onError: (error) => toast.error("Incident 操作失败", { description: describeError(error) })
  });

  const isBusy = createIncidentMutation.isPending || diagnoseMutation.isPending || incidentTransitionMutation.isPending;
  const hasBackendData = Boolean(incidentsQuery.data || diagnosesQuery.data || metricsQuery.data);
  const alertTitle = openIncidents.length
    ? `检测到 ${openIncidents.length} 个待处理 Incident`
    : latestDiagnosis
      ? "已加载最新 AI 诊断"
      : "等待后端诊断数据";
  const alertText = latestIncident?.summary || activeDiagnosis?.root_cause || metricsSummary(metrics) || "启动 Go 后端后将显示实时 Incident、指标和诊断结果。";
  const evidenceItems = useMemo(
    () => deriveEvidence(activeDiagnosis, metrics, latestIncident),
    [activeDiagnosis, metrics, latestIncident]
  );
  const recommendedActions = useMemo(
    () => deriveActions(activeDiagnosis),
    [activeDiagnosis]
  );

  function submit(event?: FormEvent) {
    event?.preventDefault();
    diagnoseMutation.mutate(question.trim() || DEFAULT_QUESTION);
  }

  return (
    <aside className={cn("infra-copilot aiops-copilot aiops-replica-copilot", hidden && "infra-copilot-hidden")} aria-hidden={hidden}>
      <div className="infra-copilot-header aiops-copilot-titlebar">
        <strong>AI Copilot</strong>
        <div className="infra-copilot-header-actions">
          <span className={cn("copilot-online-dot", !hasBackendData && "degraded")} />
          <Search size={16} />
          <button className="aiops-icon-button" onClick={() => setPage("dashboard")} type="button" title="返回总览">
            <X size={14} />
          </button>
        </div>
      </div>

      <div className="aiops-copilot-scroll">
        <div className={cn("aiops-copilot-alert-card", criticalCount > 0 && "danger")}>
          <AlertTriangle size={18} />
          <div>
            <strong>{alertTitle}</strong>
            <span>{alertText}</span>
            <button disabled={isBusy} onClick={() => submit()} type="button">
              {diagnoseMutation.isPending ? "诊断中..." : "运行诊断"}
            </button>
          </div>
        </div>

        <section className="aiops-copilot-card">
          <header>当前问题</header>
          <strong>{latestIncident?.title || diagnosisTitle(activeDiagnosis) || "生产服务待诊断"}</strong>
          <div className="aiops-copilot-pills">
            <span>{p95 ? `P95 ${fmt(p95, 0)}ms` : latestIncident ? severityLabel(latestIncident.severity) : "实时指标"}</span>
            <span>{ttft ? `TTFT ${fmt(ttft, 0)}ms` : activeDiagnosis?.status ?? "待诊断"}</span>
            <span>{confidence !== null ? `置信度 ${fmt(confidence * 100, 0)}%` : "Go 取证"}</span>
          </div>
          <dl>
            <div><dt>影响对象</dt><dd>{activeDiagnosis?.endpoint_id || latestIncident?.summary?.split(" ")[0] || targetPod}</dd></div>
            <div><dt>严重性</dt><dd>{latestIncident ? severityLabel(latestIncident.severity) : p95 && p95 > 300 ? "严重" : "警告"}</dd></div>
            <div><dt>错误率</dt><dd>{errorRate !== null ? `${fmt(errorRate * 100, 2)}%` : "-"}</dd></div>
            <div><dt>状态</dt><dd>{latestIncident ? statusLabel(latestIncident.status) : activeDiagnosis?.status ?? "待确认"}</dd></div>
          </dl>
        </section>

        <section className="aiops-copilot-card">
          <header>AI 诊断摘要</header>
          <p>{activeDiagnosis?.root_cause || "暂无后端诊断结论。点击运行诊断会通过 Go 后端聚合 metrics、Incident、部署、配置和 Kubernetes 证据，再调用 AI 服务生成 RCA。"}</p>
          <div className="aiops-copilot-evidence">
            {evidenceItems.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
          {latestDiagnosis?.id ? (
            <button className="aiops-copilot-link" onClick={() => setSelectedDiagnosisId(latestDiagnosis.id)} type="button">
              <Sparkles size={14} /> 查看完整分析报告
            </button>
          ) : null}
        </section>

        <section className="aiops-copilot-card">
          <header>推荐动作</header>
          <div className="aiops-copilot-action-list">
            {recommendedActions.map((action) => (
              <button key={action.label} onClick={() => setPage(action.page)} type="button">
                <CheckCircle size={15} />
                <span>
                  <strong>{action.label}</strong>
                  <small>{action.detail}</small>
                </span>
              </button>
            ))}
          </div>
        </section>

        <section className="aiops-copilot-card">
          <header>后端操作</header>
          <div className="aiops-copilot-actions-grid">
            <button disabled={isBusy} onClick={() => createIncidentMutation.mutate()} type="button">创建 Incident</button>
            <button disabled={!latestIncident || isBusy} onClick={() => latestIncident && incidentTransitionMutation.mutate({ id: latestIncident.id, action: "ack" })} type="button">确认 Incident</button>
            <button disabled={!latestIncident || isBusy} onClick={() => latestIncident && incidentTransitionMutation.mutate({ id: latestIncident.id, action: "resolve" })} type="button">解决 Incident</button>
            <button onClick={() => {
              incidentsQuery.refetch();
              diagnosesQuery.refetch();
              metricsQuery.refetch();
            }} type="button">刷新证据</button>
          </div>
        </section>
      </div>

      <form className="copilot-input aiops-copilot-input" onSubmit={submit}>
        <input
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="向 Copilot 提问..."
          value={question}
        />
        <button disabled={diagnoseMutation.isPending} type="submit" title="发送">
          <Send size={15} />
        </button>
      </form>

      <div className="dashboard-copilot-foot">
        <span>{hasBackendData ? "后端实时数据" : "等待 Go 后端"} · {new Date().toLocaleTimeString("zh-CN", { hour12: false })}</span>
        <RefreshCw size={13} />
      </div>
    </aside>
  );

}

function deriveEvidence(diagnosis: DiagnosisDetail | DiagnosisListItem | null, metrics: Metrics | undefined, incident: Incident | null): string[] {
  const items: string[] = [];
  if (diagnosis?.confidence !== undefined && diagnosis.confidence !== null) items.push(`RCA 置信度 ${fmt(diagnosis.confidence * 100, 0)}%`);
  if (metrics?.p95_latency_ms !== null && metrics?.p95_latency_ms !== undefined) items.push(`P95 ${fmt(metrics.p95_latency_ms, 0)}ms`);
  if (metrics?.p95_ttft_ms !== null && metrics?.p95_ttft_ms !== undefined) items.push(`TTFT ${fmt(metrics.p95_ttft_ms, 0)}ms`);
  if (metrics?.error_rate !== null && metrics?.error_rate !== undefined) items.push(`错误率 ${fmt(metrics.error_rate * 100, 2)}%`);
  if (incident) items.push(`Incident ${statusLabel(incident.status)}`);
  return items.length ? items.slice(0, 5) : ["metrics / incidents / deployments / config / k8s 证据聚合"];
}

function deriveActions(diagnosis: DiagnosisDetail | DiagnosisListItem | null): Array<{ label: string; detail: string; page: Page }> {
  const root = diagnosis?.root_cause ?? "";
  if (root.includes("配置") || root.toLowerCase().includes("config")) {
    return [
      { label: "复核配置 Diff", detail: "进入配置中心查看版本差异与回滚点", page: "config" },
      { label: "查看 Trace", detail: "确认变更后延迟与错误率是否收敛", page: "observability" },
      { label: "发起验证压测", detail: "用 SLO 门禁验证修复效果", page: "benchmarks" }
    ];
  }
  return [
    { label: "查看指标证据", detail: "打开可观测性查看慢请求、P95 和 TTFT", page: "observability" },
    { label: "检查 Kubernetes", detail: "定位 Pod、Deployment 与事件异常", page: "kubernetes" },
    { label: "运行压测验证", detail: "对修复动作做 SLO 回归", page: "benchmarks" }
  ];
}

function diagnosisTitle(diagnosis: DiagnosisListItem | DiagnosisDetail | null) {
  if (!diagnosis) return "";
  return diagnosis.question.length > 26 ? `${diagnosis.question.slice(0, 26)}...` : diagnosis.question;
}

function metricsSummary(metrics: Metrics | undefined) {
  if (!metrics) return "";
  const parts = [];
  if (metrics.p95_latency_ms !== null) parts.push(`P95 ${fmt(metrics.p95_latency_ms, 0)}ms`);
  if (metrics.p95_ttft_ms !== null) parts.push(`TTFT ${fmt(metrics.p95_ttft_ms, 0)}ms`);
  parts.push(`错误率 ${fmt((metrics.error_rate ?? 0) * 100, 2)}%`);
  return parts.join(" / ");
}

function firstKey(record: Record<string, number> | undefined) {
  if (!record) return null;
  return Object.entries(record).sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;
}

function describeError(error: unknown) {
  if (error instanceof Error) return error.message;
  return "请求失败";
}

function severityLevel(value: string) {
  const lower = value.toLowerCase();
  if (lower.includes("critical") || lower.includes("severe") || lower.includes("严重")) return "critical";
  if (lower.includes("warn") || lower.includes("警告")) return "warning";
  return "info";
}

function severityLabel(value: string) {
  const level = severityLevel(value);
  return level === "critical" ? "严重" : level === "warning" ? "警告" : "提示";
}

function statusLabel(value: string) {
  const lower = value.toLowerCase();
  if (lower.includes("resolved")) return "已解决";
  if (lower.includes("ack")) return "处理中";
  if (lower.includes("failed")) return "失败";
  if (lower.includes("completed")) return "诊断完成";
  return "等待确认";
}
