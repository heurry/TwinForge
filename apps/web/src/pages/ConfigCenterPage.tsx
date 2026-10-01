import { useEffect, useMemo, useState, type ChangeEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { parse, stringify } from "yaml";
import {
  CheckCircle2,
  Copy,
  FileCode2,
  History,
  Info,
  Plus,
  RefreshCw,
  Rocket,
  RotateCcw,
  Search,
  WandSparkles,
} from "lucide-react";

import { KpiGrid, PageHeader, PanelHeader } from "../components/common/PlatformPrimitives";
import { EmptyState, ErrorState, QueryBoundary, Skeleton, describeError } from "../components/common/FeedbackStates";
import { Drawer, DrawerField } from "../components/common/Drawer";
import { api } from "../lib/api";
import { fmt, relativeTime } from "../lib/format";
import { useGoToPage } from "../lib/useGoToPage";
import { cn } from "../lib/utils";
import type { ConfigConsoleRow, ConfigItem, ConfigVersion, ConfigVersionsResponse } from "../types/ops";
import type { KpiItem } from "../types/ui";

type AuditEventLite = { id: string; action: string; resource_id: string | null; created_at: string };

export function ConfigCenterPage() {
  const [activeItem, setActiveItem] = useState<ConfigItem | null>(null);
  const [creating, setCreating] = useState(false);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [namespaceFilter, setNamespaceFilter] = useState("all");
  const [envFilter, setEnvFilter] = useState("all");
  const [typeFilter, setTypeFilter] = useState("all");
  const [configTab, setConfigTab] = useState("配置项");

  const configQuery = useQuery({
    queryKey: ["config", "items"],
    queryFn: () => api<{ items: ConfigItem[] }>("/api/config/items"),
    refetchInterval: 15000
  });

  const auditQuery = useQuery({
    queryKey: ["audit", "config_item"],
    queryFn: () => api<{ events: AuditEventLite[] }>("/api/audit/events?resourceType=config_item&limit=20"),
    refetchInterval: 15000
  });

  const rows = useMemo(() => deriveConfigRows(configQuery.data?.items), [configQuery.data?.items]);
  const kpis = useMemo(() => buildConfigKpis(configQuery.data?.items), [configQuery.data?.items]);
  const namespaceOptions = useMemo(() => Array.from(new Set(rows.map((row) => row.namespace))).sort(), [rows]);
  const envOptions = useMemo(() => Array.from(new Set(rows.map((row) => row.env))).sort(), [rows]);
  const typeOptions = useMemo(() => Array.from(new Set(rows.map((row) => row.type))).sort(), [rows]);
  const filteredRows = useMemo(() => {
    const q = search.trim().toLowerCase();
    return rows.filter((row) =>
      (namespaceFilter === "all" || row.namespace === namespaceFilter) &&
      (envFilter === "all" || row.env === envFilter) &&
      (typeFilter === "all" || row.type === typeFilter) &&
      (!q || [row.name, row.key, row.namespace, row.env, row.type, row.status, row.owner]
        .some((value) => String(value).toLowerCase().includes(q)))
    );
  }, [rows, search, namespaceFilter, envFilter, typeFilter]);
  const totalPages = Math.max(1, Math.ceil(filteredRows.length / pageSize));
  const currentPage = Math.min(page, totalPages);
  const pageRows = filteredRows.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  return (
    <section className="infra-page config-page config-replica">
      <PageHeader
        title="配置中心"
        subtitle="管理训练、推理、微服务与网关的运行参数；每个配置项都有明确作用域、不可变版本和审计记录"
        actions={
          <button className="console-refresh primary" onClick={() => setCreating(true)} type="button">
            <Plus size={14} /> 新建配置项
          </button>
        }
      />

      <KpiGrid className="config-kpi-strip" items={kpis} />

      <section className="infra-panel config-main-panel">
        <div className="config-tabs">
          {["配置项", "变更记录", "版本历史"].map((t) => (
            <button className={configTab === t ? "active" : undefined} key={t} onClick={() => setConfigTab(t)} type="button">{t}</button>
          ))}
        </div>

        {configTab === "配置项" && (
        <>
        <div className="config-toolbar">
          <select className="config-filter" onChange={(event) => { setNamespaceFilter(event.target.value); setPage(1); }} value={namespaceFilter}>
            <option value="all">全部命名空间</option>
            {namespaceOptions.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
          </select>
          <select className="config-filter" onChange={(event) => { setEnvFilter(event.target.value); setPage(1); }} value={envFilter}>
            <option value="all">全部环境</option>
            {envOptions.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
          </select>
          <select className="config-filter" onChange={(event) => { setTypeFilter(event.target.value); setPage(1); }} value={typeFilter}>
            <option value="all">全部类型</option>
            {typeOptions.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
          </select>
          <label className="config-search"><Search size={14} />
            <input
              onChange={(event) => {
                setSearch(event.target.value);
                setPage(1);
              }}
              placeholder="搜索配置 Key、命名空间或类型..."
              value={search}
            />
          </label>
          <button className="icon-button" onClick={() => configQuery.refetch()} type="button" title="刷新">
            <RefreshCw size={14} />
          </button>
          <button className="config-create-button" onClick={() => setCreating(true)} type="button">
            <Plus size={14} /> 新建配置项
          </button>
        </div>

        <div className="config-table">
          <div className="config-table-row header">
            <span>名称 / Key</span>
            <span>命名空间</span>
            <span>环境</span>
            <span>类型</span>
            <span>当前版本</span>
            <span>更新时间</span>
            <span>状态</span>
            <span>操作</span>
          </div>
          {configQuery.isLoading ? (
            <Skeleton rows={5} />
          ) : configQuery.isError ? (
            <ErrorState error={configQuery.error} onRetry={configQuery.refetch} />
          ) : filteredRows.length === 0 ? (
            <EmptyState title="暂无配置项" description={search ? "无匹配结果" : "点击「新建配置项」创建"} />
          ) : (
            pageRows.map((row) => (
              <ConfigTableRow item={row} key={row.id} onOpen={() => setActiveItem(row.source)} />
            ))
          )}
        </div>

        <div className="config-pagination">
          <span>共 {filteredRows.length} 项</span>
          <button disabled={currentPage === 1} onClick={() => setPage((value) => Math.max(1, value - 1))} type="button">‹</button>
          {Array.from({ length: totalPages }, (_, index) => index + 1).map((item) => (
            <button className={item === currentPage ? "active" : undefined} key={item} onClick={() => setPage(item)} type="button">{item}</button>
          ))}
          <button disabled={currentPage === totalPages} onClick={() => setPage((value) => Math.min(totalPages, value + 1))} type="button">›</button>
          <button onClick={() => { setPageSize((value) => (value === 10 ? 20 : value === 20 ? 50 : 10)); setPage(1); }} type="button">{pageSize} 条/页</button>
          <span className="jump">第 <strong>{currentPage}</strong> / {totalPages} 页</span>
        </div>
        </>
        )}

        {configTab !== "配置项" && (
          <div className="config-change-list config-tab-feed">
            {auditQuery.isLoading ? (
              <Skeleton rows={4} />
            ) : auditQuery.isError ? (
              <ErrorState error={auditQuery.error} onRetry={auditQuery.refetch} />
            ) : (() => {
              const all = auditQuery.data?.events ?? [];
              const events = configTab === "版本历史" ? all.filter((event) => /publish|rollback|version/i.test(event.action)) : all;
              return events.length === 0 ? (
                <EmptyState title={configTab === "版本历史" ? "暂无版本生效 / 回滚记录" : "暂无变更记录"} description="操作配置项后这里会出现审计记录" />
              ) : (
                events.map((event) => (
                  <div className="config-change-row" key={event.id}>
                    <History size={15} />
                    <div>
                      <strong>{event.action}</strong>
                      <span>{event.resource_id ?? "config_item"}</span>
                    </div>
                    <small>{relativeTime(event.created_at)}</small>
                  </div>
                ))
              );
            })()}
          </div>
        )}
      </section>

      <div className="config-bottom-grid">
        <section className="infra-panel">
          <PanelHeader title="最近变更" action="审计链路" />
          <div className="config-change-list">
            {auditQuery.isLoading ? (
              <Skeleton rows={3} />
            ) : auditQuery.isError ? (
              <ErrorState error={auditQuery.error} onRetry={auditQuery.refetch} />
            ) : !auditQuery.data?.events?.length ? (
              <EmptyState title="暂无变更记录" />
            ) : (
              auditQuery.data.events.slice(0, 5).map((event) => (
                <div className="config-change-row" key={event.id}>
                  <History size={15} />
                  <div>
                    <strong>{event.action}</strong>
                    <span>{event.resource_id ?? "config_item"}</span>
                  </div>
                  <small>{relativeTime(event.created_at)}</small>
                </div>
              ))
            )}
          </div>
        </section>
      </div>

      {activeItem ? <ConfigItemDrawer item={activeItem} onClose={() => setActiveItem(null)} /> : null}
      {creating ? <CreateConfigDrawer onClose={() => setCreating(false)} /> : null}
    </section>
  );
}

function ConfigTableRow({ item, onOpen }: { item: ConfigConsoleRow; onOpen: () => void }) {
  const copyKey = async () => {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(item.source.config_key);
      } else {
        const input = document.createElement("textarea");
        input.value = item.source.config_key;
        input.style.position = "fixed";
        input.style.opacity = "0";
        document.body.appendChild(input);
        input.select();
        document.execCommand("copy");
        input.remove();
      }
      toast.success("配置 Key 已复制");
    } catch {
      toast.error("复制失败，请手动复制配置 Key");
    }
  };

  return (
    <div className="config-table-row">
      <span className="config-name-cell">
        <span className="config-name-line">
          <strong>{item.name}</strong>
          <button className="config-inline-copy" onClick={copyKey} type="button" title="复制配置 Key" aria-label={`复制 ${item.source.config_key}`}>
            <Copy size={12} />
          </button>
        </span>
        <small>{item.key}</small>
      </span>
      <span>{item.namespace}</span>
      <span><em className={cn("config-env-pill", item.env.toLowerCase())}>{item.env}</em></span>
      <span>{item.type}</span>
      <span className="config-version-cell">
        <strong>{item.version}</strong>
        <small>当前</small>
      </span>
      <span className="config-time-cell">
        <strong>{item.updatedAt}</strong>
        <small>{item.owner}</small>
      </span>
      <span><em className="config-state-pill">{item.status}</em></span>
      <span className="config-row-actions">
        <button className="config-row-action primary" onClick={onOpen} type="button" title="查看版本、发布新版本或回滚">
          <History size={13} /> 版本与发布
        </button>
      </span>
    </div>
  );
}

function ConfigItemDrawer({ item, onClose }: { item: ConfigItem; onClose: () => void }) {
  const qc = useQueryClient();
  const goTo = useGoToPage();
  const [content, setContent] = useState("");
  const [reason, setReason] = useState("");
  const [editorInitialized, setEditorInitialized] = useState(false);

  const versionsQuery = useQuery({
    queryKey: ["config", "versions", item.id],
    queryFn: () => api<ConfigVersionsResponse>(`/api/config/items/${item.id}/versions`)
  });

  const activeVersion = versionsQuery.data?.active_version ?? item.active_version;

  useEffect(() => {
    if (!versionsQuery.data || editorInitialized) return;
    const active = versionsQuery.data.versions.find((version) => version.version === versionsQuery.data.active_version);
    setContent(active?.content ?? "");
    setEditorInitialized(true);
  }, [editorInitialized, versionsQuery.data]);

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["config", "items"] });
    qc.invalidateQueries({ queryKey: ["config", "versions", item.id] });
    qc.invalidateQueries({ queryKey: ["audit", "config_item"] });
  };

  const publishMutation = useMutation({
    mutationFn: () =>
      api(`/api/config/items/${item.id}/versions`, {
        method: "POST",
        body: JSON.stringify({ content, change_reason: reason || "更新配置" })
      }),
    onSuccess: () => {
      toast.success("已创建并启用新版本");
      setReason("");
      invalidate();
    },
    onError: (e) => toast.error(`发布失败：${describeError(e)}`)
  });

  const rollbackMutation = useMutation({
    mutationFn: (version: number) =>
      api(`/api/config/items/${item.id}/rollback`, { method: "POST", body: JSON.stringify({ version }) }),
    onSuccess: (_d, version) => {
      toast.success(`已回滚到 v${version}`);
      invalidate();
    },
    onError: (e) => toast.error(`回滚失败：${describeError(e)}`)
  });

  const launchKind = configLaunchKind(item.config_key, item.config_type);
  const useInWorkbench = (version: number) => {
    if (!launchKind) return;
    onClose();
    goTo(launchKind === "training" ? "training" : "benchmarks", {
      deliveryKind: launchKind,
      configItemId: item.id,
      configVersion: String(version),
      trainingJobId: null,
      benchmarkRunId: null,
      deploymentId: null,
    });
  };

  return (
    <Drawer
      className="config-version-drawer"
      open
      title={item.config_key}
      subtitle={`${item.env} · ${item.namespace ?? "default"} · ${normalizeConfigType(item.config_type)}`}
      onClose={onClose}
    >
      <DrawerField label="当前生效版本" value={`v${activeVersion}`} />

      <div className="drawer-section">
        <h4>配置内容与发布</h4>
        <p className="config-version-publish-note">
          {versionsQuery.isLoading ? "正在加载当前配置内容…" : `以下内容来自当前 v${activeVersion}。直接修改后发布，将生成并启用 v${activeVersion + 1}。`}
        </p>
        <textarea
          className="drawer-textarea config-version-editor"
          rows={13}
          disabled={versionsQuery.isLoading}
          placeholder={versionsQuery.isLoading ? "正在读取当前生效版本..." : "输入配置内容..."}
          value={content}
          onChange={(e) => setContent(e.target.value)}
          spellCheck={false}
        />
        <input
          className="drawer-input"
          placeholder="变更原因"
          value={reason}
          onChange={(e) => setReason(e.target.value)}
        />
        <button
          className="infra-action-btn"
          type="button"
          disabled={!content.trim() || publishMutation.isPending}
          onClick={() => publishMutation.mutate()}
        >
          {publishMutation.isPending ? "发布中..." : "发布并设为生效版本"}
        </button>
      </div>

      {launchKind ? <div className="drawer-section">
        <h4>用于工作台</h4>
        <p>配置中心只维护不可变版本。进入{launchKind === "training" ? "训练任务" : "推理服务"}控制面后再确认资源状态并启动，任务会自动记录配置引用。</p>
        <button className="infra-action-btn" type="button" onClick={() => useInWorkbench(activeVersion)}>
          <Rocket size={14} /> 使用当前 v{activeVersion}
        </button>
      </div> : null}

      <div className="drawer-section">
        <h4>历史版本</h4>
        <QueryBoundary query={versionsQuery}>
          {(data) => (
            <ul className="version-list">
              {data.versions.map((v) => (
                <VersionRow
                  key={v.version}
                  v={v}
                  active={v.version === data.active_version}
                  onRollback={() => rollbackMutation.mutate(v.version)}
                  rollingBack={rollbackMutation.isPending}
                  onUse={launchKind ? () => useInWorkbench(v.version) : undefined}
                />
              ))}
            </ul>
          )}
        </QueryBoundary>
      </div>
    </Drawer>
  );
}

function VersionRow({
  v,
  active,
  onRollback,
  rollingBack,
  onUse,
}: {
  v: ConfigVersion;
  active: boolean;
  onRollback: () => void;
  rollingBack: boolean;
  onUse?: () => void;
}) {
  return (
    <li className="version-item">
      <div className="version-head">
        <strong>v{v.version}</strong>
        {active ? <em className="config-state-pill">生效</em> : null}
        <small>{v.operator ?? "-"} · {relativeTime(v.created_at)}</small>
        {!active ? (
          <button className="link-btn" type="button" disabled={rollingBack} onClick={onRollback}>
            <RotateCcw size={13} /> 回滚到此版本
          </button>
        ) : null}
        {onUse ? (
          <button className="link-btn" type="button" onClick={onUse}>
            <Rocket size={13} /> 用于工作台
          </button>
        ) : null}
      </div>
      {v.change_reason ? <p className="version-reason">{v.change_reason}</p> : null}
      {v.content ? <pre className="version-content">{v.content}</pre> : null}
    </li>
  );
}

function configLaunchKind(key: string, configType = ""): "training" | "inference" | null {
  const lower = `${key} ${configType}`.toLowerCase();
  if (lower.includes("train")) return "training";
  if (lower.includes("infer") || lower.includes("serve") || lower.includes("vllm")) return "inference";
  return null;
}

type ConfigPurpose = "training" | "inference" | "service" | "gateway" | "observability" | "platform";
type ConfigFormat = "yaml" | "json" | "properties" | "env" | "text";

type ConfigPurposeDefinition = {
  value: ConfigPurpose;
  label: string;
  short: string;
  description: string;
  prefix: string;
  targetLabel: string;
  targetPlaceholder: string;
  exampleTarget: string;
};

type CreateConfigForm = {
  purpose: ConfigPurpose;
  target: string;
  config_key: string;
  env: string;
  namespace: string;
  format: ConfigFormat;
  content: string;
  change_reason: string;
};

const CONFIG_PURPOSES: ConfigPurposeDefinition[] = [
  { value: "training", label: "训练任务", short: "训", description: "模型、数据集、超参数与 GPU 资源", prefix: "training", targetLabel: "任务模板", targetPlaceholder: "例如 qwen3-lora", exampleTarget: "qwen3-lora" },
  { value: "inference", label: "推理服务", short: "推", description: "模型路径、运行时、并发与弹性参数", prefix: "inference", targetLabel: "推理服务", targetPlaceholder: "例如 qwen3-8b", exampleTarget: "qwen3-8b" },
  { value: "service", label: "微服务", short: "服", description: "端口、依赖、功能开关与业务参数", prefix: "services", targetLabel: "服务名称", targetPlaceholder: "例如 order-api", exampleTarget: "order-api" },
  { value: "gateway", label: "网关路由", short: "网", description: "路由匹配、上游服务与限流策略", prefix: "gateway", targetLabel: "网关名称", targetPlaceholder: "例如 public-gateway", exampleTarget: "public-gateway" },
  { value: "observability", label: "监控告警", short: "监", description: "采集、告警阈值与通知规则", prefix: "observability", targetLabel: "规则组", targetPlaceholder: "例如 inference-alerts", exampleTarget: "inference-alerts" },
  { value: "platform", label: "平台策略", short: "平", description: "集群级默认值、配额与治理策略", prefix: "platform", targetLabel: "策略名称", targetPlaceholder: "例如 resource-defaults", exampleTarget: "resource-defaults" },
];

const CONFIG_FORMATS: Array<{ value: ConfigFormat; label: string; extension: string }> = [
  { value: "yaml", label: "YAML", extension: "yaml" },
  { value: "json", label: "JSON", extension: "json" },
  { value: "properties", label: "Properties", extension: "properties" },
  { value: "env", label: "ENV", extension: "env" },
  { value: "text", label: "纯文本", extension: "txt" },
];

function purposeDefinition(value: ConfigPurpose) {
  return CONFIG_PURPOSES.find((item) => item.value === value) ?? CONFIG_PURPOSES[2];
}

function slugifyConfigTarget(value: string) {
  return value.trim().toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
}

function suggestedConfigKey(purpose: ConfigPurpose, target: string, format: ConfigFormat) {
  const targetSlug = slugifyConfigTarget(target);
  if (!targetSlug) return "";
  const definition = purposeDefinition(purpose);
  const extension = CONFIG_FORMATS.find((item) => item.value === format)?.extension ?? format;
  return `${definition.prefix}/${targetSlug}/application.${extension}`;
}

function configTemplateData(purpose: ConfigPurpose, target: string, env: string): Record<string, unknown> {
  if (purpose === "training") {
    return {
      kind: "TrainingJobConfig",
      metadata: { name: target, environment: env },
      spec: {
        model: "Qwen/Qwen3-8B",
        dataset: "/data/train.jsonl",
        resources: { gpu: 1, cpu: "8", memory: "32Gi" },
        training: { epochs: 3, learningRate: 0.0002, batchSize: 4 },
      },
    };
  }
  if (purpose === "inference") {
    return {
      kind: "InferenceServiceConfig",
      metadata: { name: target, environment: env },
      spec: {
        model: "Qwen/Qwen3-8B",
        runtime: { engine: "vllm", tensorParallelSize: 1, maxModelLen: 8192 },
        resources: { gpu: 1, cpu: "4", memory: "24Gi" },
        autoscaling: { minReplicas: 1, maxReplicas: 4, targetConcurrency: 16 },
      },
    };
  }
  if (purpose === "gateway") {
    return {
      gateway: { name: target, environment: env },
      routes: [{ id: "api-route", path: "/api/*", upstream: "order-api:8080", timeout: "30s" }],
      rateLimit: { requestsPerMinute: 1200 },
    };
  }
  if (purpose === "observability") {
    return {
      ruleGroup: target,
      environment: env,
      scrapeInterval: "30s",
      alerts: [{ name: "HighErrorRate", expression: "error_rate > 0.05", for: "5m", severity: "warning" }],
    };
  }
  if (purpose === "platform") {
    return {
      policy: target,
      environment: env,
      defaults: { cpuRequest: "500m", memoryRequest: "1Gi" },
      quota: { maxGpu: 4, maxReplicas: 20 },
    };
  }
  return {
    application: { name: target, environment: env },
    server: { port: 8080, gracefulShutdown: "30s" },
    features: { enableCache: true, enableTracing: true },
    dependencies: { requestTimeout: "5s", retryCount: 2 },
  };
}

function buildConfigTemplate(purpose: ConfigPurpose, target: string, env: string, format: ConfigFormat) {
  const data = configTemplateData(purpose, target, env);
  if (format === "json") return JSON.stringify(data, null, 2);
  if (format === "properties") {
    return `app.name=${target}\napp.environment=${env}\nserver.port=8080\nfeature.tracing.enabled=true\n`;
  }
  if (format === "env") {
    return `APP_NAME=${target}\nAPP_ENV=${env}\nSERVER_PORT=8080\nTRACING_ENABLED=true\n`;
  }
  if (format === "text") {
    return `# ${purposeDefinition(purpose).label}配置\n# 作用对象：${target}\n# 环境：${env}\n`;
  }
  return stringify(data, { indent: 2 });
}

function validateConfigContent(format: ConfigFormat, content: string) {
  if (!content.trim()) return "请填写首个版本的配置内容";
  try {
    if (format === "yaml") parse(content);
    if (format === "json") JSON.parse(content);
  } catch (error) {
    return `${format.toUpperCase()} 格式有误：${error instanceof Error ? error.message.split("\n")[0] : "无法解析"}`;
  }
  if (format === "env") {
    const invalid = content.split("\n").find((line) => line.trim() && !line.trim().startsWith("#") && !line.includes("="));
    if (invalid) return `ENV 格式有误：${invalid}`;
  }
  return null;
}

function CreateConfigDrawer({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const initialPurpose: ConfigPurpose = "service";
  const initialTarget = "order-api";
  const [form, setForm] = useState<CreateConfigForm>({
    purpose: initialPurpose,
    target: initialTarget,
    config_key: suggestedConfigKey(initialPurpose, initialTarget, "yaml"),
    env: "dev",
    namespace: "default",
    format: "yaml" as ConfigFormat,
    content: buildConfigTemplate(initialPurpose, initialTarget, "dev", "yaml"),
    change_reason: "创建服务运行配置",
  });
  const [customKey, setCustomKey] = useState(false);
  const set = (k: keyof typeof form) => (e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) =>
    setForm((f) => ({ ...f, [k]: e.target.value }));
  const definition = purposeDefinition(form.purpose);
  const effectiveKey = customKey ? form.config_key : suggestedConfigKey(form.purpose, form.target, form.format);
  const contentError = validateConfigContent(form.format, form.content);
  const keyError = !effectiveKey.trim()
    ? "请填写作用对象，系统会生成配置 Key"
    : !/^[a-zA-Z0-9][a-zA-Z0-9._/-]*$/.test(effectiveKey)
      ? "配置 Key 只能包含字母、数字、点、下划线、短横线和斜杠"
      : null;
  const formError = keyError || (!form.target.trim() ? `请填写${definition.targetLabel}` : null) || (!form.change_reason.trim() ? "请填写创建说明，便于后续审计" : null) || contentError;

  const choosePurpose = (purpose: ConfigPurpose) => {
    const nextDefinition = purposeDefinition(purpose);
    const nextTarget = nextDefinition.exampleTarget;
    setCustomKey(false);
    setForm((current) => ({
      ...current,
      purpose,
      target: nextTarget,
      config_key: suggestedConfigKey(purpose, nextTarget, current.format),
      content: buildConfigTemplate(purpose, nextTarget, current.env, current.format),
      change_reason: `创建${nextDefinition.label}配置`,
    }));
  };

  const chooseFormat = (format: ConfigFormat) => {
    setForm((current) => ({
      ...current,
      format,
      config_key: customKey ? current.config_key : suggestedConfigKey(current.purpose, current.target, format),
    }));
  };

  const applyTemplate = () => {
    const target = form.target.trim() || definition.exampleTarget;
    setForm((current) => ({ ...current, content: buildConfigTemplate(current.purpose, target, current.env, current.format) }));
    toast.success(`已填入${definition.label} ${form.format.toUpperCase()} 模板`);
  };

  const createMutation = useMutation({
    mutationFn: () => api("/api/config/items", {
      method: "POST",
      body: JSON.stringify({
        config_key: effectiveKey.trim(),
        env: form.env,
        namespace: form.namespace.trim() || "default",
        config_type: `${form.purpose}-${form.format}`,
        content: form.content,
        change_reason: form.change_reason.trim(),
      }),
    }),
    onSuccess: () => {
      toast.success("配置项已创建");
      qc.invalidateQueries({ queryKey: ["config", "items"] });
      qc.invalidateQueries({ queryKey: ["audit", "config_item"] });
      onClose();
    },
    onError: (e) => toast.error(`创建失败：${describeError(e)}`)
  });

  return (
    <Drawer
      className="config-create-drawer"
      open
      title="新建配置项"
      subtitle="定义谁使用、在哪个环境生效，以及首个不可变版本的内容"
      onClose={onClose}
      footer={
        <>
          <button className="config-drawer-cancel" type="button" onClick={onClose}>取消</button>
          <button
            className="infra-action-btn"
            type="button"
            disabled={Boolean(formError) || createMutation.isPending}
            onClick={() => createMutation.mutate()}
          >
            {createMutation.isPending ? "创建中..." : "创建并启用 v1"}
          </button>
        </>
      }
    >
      <div className="config-create-explainer">
        <Info size={16} />
        <div>
          <strong>这里配置的是组件运行时读取的参数</strong>
          <p>例如训练超参数、推理运行时、服务端口或网关规则。创建后生成并启用 v1；后续修改会产生新版本，可审计、发布和回滚。</p>
        </div>
      </div>

      <div className="drawer-section config-create-section">
        <div className="config-create-section-title"><span>1</span><div><h4>配置用途</h4><p>先选择这份配置由哪类工作负载使用</p></div></div>
        <div className="config-purpose-grid">
          {CONFIG_PURPOSES.map((purpose) => (
            <button
              className={cn("config-purpose-card", form.purpose === purpose.value && "active")}
              key={purpose.value}
              onClick={() => choosePurpose(purpose.value)}
              type="button"
            >
              <span>{purpose.short}</span>
              <div><strong>{purpose.label}</strong><small>{purpose.description}</small></div>
              {form.purpose === purpose.value ? <CheckCircle2 size={15} /> : null}
            </button>
          ))}
        </div>
      </div>

      <div className="drawer-section config-create-section">
        <div className="config-create-section-title"><span>2</span><div><h4>作用范围</h4><p>决定配置属于哪个对象、环境和 Kubernetes 命名空间</p></div></div>
        <label className="config-form-field config-form-field-wide">
          <span>{definition.targetLabel} <em>必填</em></span>
          <input className="drawer-input" placeholder={definition.targetPlaceholder} value={form.target} onChange={(event) => {
            const target = event.target.value;
            setForm((current) => ({ ...current, target, config_key: customKey ? current.config_key : suggestedConfigKey(current.purpose, target, current.format) }));
          }} />
          <small>用于标识谁会读取这份配置，不会在这里直接启动或发布工作负载。</small>
        </label>
        <div className="config-form-grid">
          <label className="config-form-field">
            <span>环境 <em>必填</em></span>
            <select className="drawer-input" value={form.env} onChange={set("env")}>
              <option value="dev">dev · 开发</option>
              <option value="staging">staging · 预发布</option>
              <option value="prod">prod · 生产</option>
            </select>
          </label>
          <label className="config-form-field">
            <span>Namespace <em>必填</em></span>
            <input className="drawer-input" placeholder="default" value={form.namespace} onChange={set("namespace")} />
          </label>
        </div>
      </div>

      <div className="drawer-section config-create-section">
        <div className="config-create-section-title"><span>3</span><div><h4>配置标识与格式</h4><p>配置 Key 是版本与审计记录中的稳定标识</p></div></div>
        <div className="config-form-grid config-key-format-grid">
          <label className="config-form-field">
            <span>内容格式 <em>必填</em></span>
            <select className="drawer-input" value={form.format} onChange={(event) => chooseFormat(event.target.value as ConfigFormat)}>
              {CONFIG_FORMATS.map((format) => <option key={format.value} value={format.value}>{format.label}</option>)}
            </select>
          </label>
          <div className="config-type-preview">
            <span>保存类型</span>
            <strong>{definition.label} · {form.format.toUpperCase()}</strong>
          </div>
        </div>
        <label className="config-form-field config-form-field-wide">
          <span>配置 Key <em>必填</em></span>
          <input className="drawer-input config-key-input" value={effectiveKey} onChange={(event) => {
            setCustomKey(true);
            setForm((current) => ({ ...current, config_key: event.target.value }));
          }} />
          <small>默认按“用途 / 作用对象 / 文件名”生成；可以修改，但创建后应保持稳定。</small>
        </label>
        {customKey ? <button className="config-reset-key" type="button" onClick={() => setCustomKey(false)}>恢复自动生成</button> : null}
      </div>

      <div className="drawer-section config-create-section">
        <div className="config-create-section-title config-content-title">
          <span>4</span>
          <div><h4>首个版本内容</h4><p>这里的内容会保存为 v1 并立即标记为当前生效版本</p></div>
          <button className="config-template-button" type="button" onClick={applyTemplate}><WandSparkles size={13} /> 应用模板</button>
        </div>
        <div className="config-editor-heading"><FileCode2 size={14} /><strong>{effectiveKey || `application.${form.format}`}</strong><span>{form.format.toUpperCase()}</span></div>
        <textarea className="drawer-textarea config-content-editor" rows={14} placeholder="输入配置内容..." value={form.content} onChange={set("content")} spellCheck={false} />
        <div className={cn("config-validation", contentError ? "error" : "success")}>
          {contentError ? <Info size={13} /> : <CheckCircle2 size={13} />}
          <span>{contentError ?? `${form.format.toUpperCase()} 格式检查通过`}</span>
        </div>
        <label className="config-form-field config-form-field-wide">
          <span>创建说明 <em>必填</em></span>
          <input className="drawer-input" placeholder="例如：初始化开发环境推理参数" value={form.change_reason} onChange={set("change_reason")} />
          <small>会写入 v1 版本记录和审计链路，说明为什么创建这份配置。</small>
        </label>
      </div>

      <div className="config-create-summary">
        <strong>创建结果</strong>
        <span>{definition.label} / {form.env} / {form.namespace || "default"}</span>
        <span>{effectiveKey || "等待生成配置 Key"}</span>
        <p>生成配置项和活跃版本 v1。配置中心仅管理版本；训练任务、推理服务或发布中心会引用具体版本执行。</p>
      </div>
      {formError ? <div className="config-submit-hint"><Info size={13} /> {formError}</div> : null}
    </Drawer>
  );
}

function buildConfigKpis(items: ConfigItem[] | undefined): KpiItem[] {
  const list = items ?? [];
  const namespaces = new Set(list.map((item) => item.namespace ?? "default")).size;
  const envs = new Set(list.map((item) => item.env || "dev")).size;
  const totalVersions = list.reduce((acc, item) => acc + (item.version_count ?? 0), 0);
  const pending = list.filter((item) => String(item.status).toLowerCase().includes("pending")).length;
  const compliance = list.length ? ((list.length - pending) / list.length) * 100 : null;

  return [
    { id: "configs", label: "配置项总数", value: String(list.length), detail: "已托管", trend: [] },
    { id: "namespaces", label: "命名空间", value: String(namespaces), detail: "命名空间数", trend: [] },
    { id: "envs", label: "环境", value: String(envs), detail: "环境数", trend: [] },
    { id: "versions", label: "活跃版本", value: String(totalVersions), detail: "当前生效", trend: [], tone: "success" },
    { id: "changes", label: "待审批", value: String(pending), detail: "pending 配置", trend: [], tone: pending ? "warning" : "success" },
    { id: "compliance", label: "合规检查", value: compliance == null ? "—" : `${fmt(compliance, 1)}%`, detail: "非 pending 占比", trend: [], tone: "success" },
  ];
}

function deriveConfigRows(items: ConfigItem[] | undefined): ConfigConsoleRow[] {
  if (!items?.length) return [];
  return items.map((item) => ({
    id: item.id,
    name: item.config_key,
    key: item.config_key.includes(".") ? item.config_key : `${item.namespace ?? "default"}.${item.config_key}`,
    namespace: item.namespace ?? "default",
    env: item.env || "dev",
    type: normalizeConfigType(item.config_type),
    version: `v${item.active_version}`,
    updatedAt: formatDateTime(item.updated_at),
    owner: item.created_by ?? "system",
    status: normalizeConfigStatus(item.status),
    versionCount: item.version_count ?? 0,
    trend: [],
    source: item,
  }));
}

function normalizeConfigType(value: string) {
  const lower = String(value || "").toLowerCase();
  const purpose = CONFIG_PURPOSES.find((item) => lower.startsWith(`${item.value}-`));
  const format = lower.includes("yaml") || lower.includes("yml") ? "YAML"
    : lower.includes("properties") ? "Properties"
      : lower.includes("json") ? "JSON"
        : lower.includes("configmap") ? "ConfigMap"
          : lower.includes("env") ? "ENV"
            : lower.includes("text") ? "Text"
              : value || "YAML";
  return purpose ? `${purpose.label} · ${format}` : format;
}

function normalizeConfigStatus(value: string) {
  const lower = String(value || "").toLowerCase();
  if (lower.includes("pending") || lower.includes("draft")) return "待审批";
  if (lower.includes("disabled") || lower.includes("failed")) return "异常";
  return "生效";
}

function formatDateTime(value: string | null | undefined) {
  if (!value) return "刚刚";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const d = date.toLocaleDateString("zh-CN", { year: "numeric", month: "2-digit", day: "2-digit" }).replace(/\//g, "-");
  const t = date.toLocaleTimeString("zh-CN", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
  return `${d} ${t}`;
}
