import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useMutation, useQuery, useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Bot, Boxes, Braces, CheckCircle2, ChevronRight, CircleAlert, Copy, FileCode2, FlaskConical, GitCompareArrows, Network, Plus, RefreshCw, Rocket, ShieldCheck, Sparkles, Upload, Wrench, X } from "lucide-react";
import { toast } from "sonner";

import { EmptyState, ErrorState, Skeleton, describeError } from "../components/common/FeedbackStates";
import { StatusBadge } from "../components/common/PlatformPrimitives";
import {
  createAgentDefinition,
  createAgentRun,
  createAgentSession,
  createAgentPromptVersion,
  createAgentSkillSetVersion,
  createAgentToolSetVersion,
  createAgentToolVersion,
  createAgentVersion,
  createMCPServerVersion,
  cancelA2ATask,
  getA2ATask,
  getA2AAgentCard,
  listAgentDefinitions,
  listAgentEnvironmentTemplates,
  listAgentDependencyInstalls,
  listAgentPromptVersions,
  listAgentSkillVersions,
  listAgentSkillSetVersions,
  listAgentToolSetVersions,
  listAgentToolVersions,
  listAgentVersions,
  listMCPServerVersions,
  releaseAgentVersion,
  sendA2AMessage,
  subscribeA2ATask,
  syncMCPServerTools,
  testMCPServerVersion,
  uploadAgentSkillVersion
} from "../lib/api";
import type { A2AStreamEvent, A2ATask, AgentDefinition, AgentDependencyInstall, AgentEnvironmentTemplate, AgentPromptVersion, AgentSkillSetVersion, AgentSkillVersion, AgentToolSetVersion, AgentToolVersion, AgentVersion, AgentVersionSpecInput, MCPServerVersion, MCPToolSnapshot } from "../types/agent";

type StudioTab = "agents" | "prompts" | "skills" | "tools" | "environment" | "collaboration";

// Defaults for newly composed AgentVersions. Published versions retain their
// immutable snapshot; changing these values only affects a new draft.
const DEFAULT_HARNESS_MAX_TURNS = 12;
const DEFAULT_HARNESS_MAX_STEPS = 8;
const DEFAULT_RUNTIME_MAX_MODEL_CALLS = 96;

const tabs: Array<{ id: StudioTab; label: string; detail: string; icon: typeof Bot }> = [
  { id: "agents", label: "Agents", detail: "身份与版本", icon: Bot },
  { id: "prompts", label: "Prompts", detail: "系统提示词", icon: FileCode2 },
  { id: "skills", label: "Skills", detail: "能力包与上传", icon: Sparkles },
  { id: "tools", label: "Tools", detail: "工具与 ToolSet", icon: Wrench },
  { id: "environment", label: "Environments", detail: "模板与依赖审批", icon: Boxes },
  { id: "collaboration", label: "Collaboration", detail: "MCP 与 Agent 协作", icon: Network }
];

export function AgentStudioPage() {
  const navigate = useNavigate();
  const [tab, setTab] = useState<StudioTab>("agents");
  const agents = useQuery({ queryKey: ["agent-studio", "agents"], queryFn: listAgentDefinitions, retry: false });
  const prompts = useQuery({ queryKey: ["agent-studio", "prompts"], queryFn: listAgentPromptVersions, retry: false });
  const skills = useQuery({ queryKey: ["agent-studio", "skills"], queryFn: listAgentSkillVersions, retry: false });
  const skillsets = useQuery({ queryKey: ["agent-studio", "skillsets"], queryFn: listAgentSkillSetVersions, retry: false });
  const tools = useQuery({ queryKey: ["agent-studio", "tools"], queryFn: listAgentToolVersions, retry: false });
  const toolsets = useQuery({ queryKey: ["agent-studio", "toolsets"], queryFn: listAgentToolSetVersions, retry: false });
  const environments = useQuery({ queryKey: ["agent-studio", "environments"], queryFn: listAgentEnvironmentTemplates, retry: false });
  const installs = useQuery({ queryKey: ["agent-studio", "dependency-installs"], queryFn: () => listAgentDependencyInstalls(), retry: false });
  const refreshing = agents.isFetching || prompts.isFetching || skills.isFetching || skillsets.isFetching || tools.isFetching || toolsets.isFetching || environments.isFetching || installs.isFetching;
  const counts: Record<StudioTab, number | undefined> = {
    agents: agents.data?.data?.length,
    prompts: distinctDefinitions(prompts.data?.data).length,
    skills: distinctDefinitions(skills.data?.data).length,
    tools: distinctDefinitions(tools.data?.data).length,
    environment: environments.data?.data.length,
    collaboration: undefined
  };
  const refresh = () => {
    void agents.refetch(); void prompts.refetch(); void skills.refetch(); void skillsets.refetch(); void tools.refetch(); void toolsets.refetch(); void environments.refetch(); void installs.refetch();
  };

  return <div className="agent-studio-page">
    <header className="agent-studio-header">
      <span className="agent-studio-brand"><i><Boxes size={18}/></i><span><strong>Agent Studio</strong><small>配置、验证和发布可复用 Agent</small></span></span>
      <span className="agent-studio-stage"><b>Agent Platform · v0.8</b><small>版本化配置、治理执行与互操作</small></span>
      <div className="agent-studio-header-actions"><button type="button" onClick={refresh}><RefreshCw className={refreshing ? "agent-spin" : ""} size={14}/>刷新</button><button className="primary" type="button" onClick={() => navigate("/agents")}>进入运行台<ChevronRight size={14}/></button></div>
    </header>
    <div className="agent-studio-layout">
      <aside className="agent-studio-nav">
        <header><strong>配置资源</strong><small>发布时精确固定版本</small></header>
        {tabs.map(item => { const Icon = item.icon; return <button type="button" className={tab === item.id ? "active" : ""} key={item.id} onClick={() => setTab(item.id)}><Icon size={16}/><span><b>{item.label}</b><small>{item.detail}</small></span>{counts[item.id] !== undefined ? <em>{counts[item.id]}</em> : null}</button>; })}
        <footer><ShieldCheck size={14}/><span><b>能力状态真实呈现</b><small>MCP、委派与 A2A 已接入真实执行链路。</small></span></footer>
      </aside>
      <main className="agent-studio-main">
        {tab === "agents" ? <AgentsPanel query={agents} prompts={prompts} skillsets={skillsets} toolsets={toolsets}/> : null}
        {tab === "prompts" ? <PromptsPanel query={prompts}/> : null}
        {tab === "skills" ? <SkillsPanel query={skills} skillsets={skillsets}/> : null}
        {tab === "tools" ? <ToolsPanel tools={tools} toolsets={toolsets}/> : null}
        {tab === "environment" ? <EnvironmentPanel environments={environments} installs={installs}/> : null}
        {tab === "collaboration" ? <CollaborationPanel agents={agents.data?.data || []}/> : null}
      </main>
    </div>
  </div>;
}

function AgentsPanel({ query, prompts, skillsets, toolsets }: {
  query: UseQueryResult<{ data: AgentDefinition[] }>;
  prompts: UseQueryResult<{ data: AgentPromptVersion[] }>;
  skillsets: UseQueryResult<{ data: AgentSkillSetVersion[] }>;
  toolsets: UseQueryResult<{ data: AgentToolSetVersion[] }>;
}) {
  const client = useQueryClient();
  const navigate = useNavigate();
  const [selectedID, setSelectedID] = useState<string>();
  const [creating, setCreating] = useState(false);
  const [composing, setComposing] = useState(false);
  const [review, setReview] = useState<{ mode: "diff" | "test"; versionID: string }>();
  const [form, setForm] = useState({ key: "", name: "", description: "", owner: "web-console" });
  const selected = query.data?.data?.find(agent => agent.id === selectedID) || query.data?.data?.[0];
  const versions = useQuery({ queryKey: ["agent-studio", "versions", selected?.id], queryFn: () => listAgentVersions(selected!.id), enabled: Boolean(selected), retry: false });
  const create = useMutation({
    mutationFn: createAgentDefinition,
    onSuccess: async result => {
      toast.success("Agent 已创建", { description: "下一步为它装配 Prompt、Skill、Tool 和模型版本。" });
      setCreating(false); setForm({ key: "", name: "", description: "", owner: "web-console" }); setSelectedID(result.data.id);
      await client.invalidateQueries({ queryKey: ["agent-studio", "agents"] });
    },
    onError: error => toast.error("创建 Agent 失败", { description: describeError(error) })
  });
  const release = useMutation({
    mutationFn: releaseAgentVersion,
    onSuccess: async result => {
      toast.success(`Agent v${result.data.version} 已发布`, { description: "该版本现在可以在 Agent 运行台创建 Session 和 Run。" });
      await Promise.all([
        client.invalidateQueries({ queryKey: ["agent-studio", "versions", selected?.id] }),
        client.invalidateQueries({ queryKey: ["agent-studio", "agents"] }),
        client.invalidateQueries({ queryKey: ["agent", "versions"] })
      ]);
    },
    onError: error => toast.error("发布 AgentVersion 失败", { description: describeError(error) })
  });
  if (query.isLoading) return <Skeleton rows={8}/>;
  if (query.isError) return <ErrorState title="无法读取 Agent Catalog" error={query.error} onRetry={() => query.refetch()}/>;
  return <StudioSection title="Agent Catalog" detail="稳定身份、资源装配、草稿版本与发布版本分离" action={<button className="studio-primary" type="button" onClick={() => setCreating(value => !value)}><Plus size={14}/>{creating ? "取消" : "新建 Agent"}</button>}>
    {creating ? <form className="agent-studio-form" onSubmit={event => { event.preventDefault(); create.mutate({ key: form.key.trim(), name: form.name.trim(), description: form.description.trim() || undefined, owner: form.owner.trim() || undefined }); }}><label>唯一 Key<input required pattern="[a-z][a-z0-9._-]*" value={form.key} onChange={event => setForm({ ...form, key: event.target.value })} placeholder="release-reviewer"/></label><label>显示名称<input required value={form.name} onChange={event => setForm({ ...form, name: event.target.value })} placeholder="发布验证 Agent"/></label><label>Owner<input value={form.owner} onChange={event => setForm({ ...form, owner: event.target.value })}/></label><label className="wide">描述<textarea value={form.description} onChange={event => setForm({ ...form, description: event.target.value })} placeholder="说明职责，不在这里填写系统提示词"/></label><footer><span>创建的是稳定身份；运行前仍需创建并发布版本。</span><button type="submit" disabled={create.isPending || !form.key.trim() || !form.name.trim()}>{create.isPending ? "创建中…" : "确认创建"}</button></footer></form> : null}
    {!query.data?.data?.length ? <EmptyState title="暂无 Agent" description="创建第一个业务 Agent；Smoke Agent 应仅保留在测试环境。" icon={Bot}/> : <div className="agent-catalog-layout"><div className="agent-catalog-list">{query.data.data.map(agent => <button type="button" className={selected?.id === agent.id ? "active" : ""} key={agent.id} onClick={() => { setSelectedID(agent.id); setComposing(false); setReview(undefined); }}><i><Bot size={15}/></i><span><b>{agent.name}</b><small>{agent.key} · {agent.owner || "未设置 Owner"}</small></span><StatusBadge status={agent.status}/></button>)}</div>{selected ? <section className="agent-definition-detail"><header><span><b>{selected.name}</b><small>{selected.description || "未填写职责描述"}</small></span><span className="agent-definition-actions"><StatusBadge status={selected.status}/><button type="button" onClick={() => setComposing(value => !value)}><Plus size={13}/>{composing ? "关闭装配" : "创建版本"}</button></span></header>{composing ? <AgentVersionComposer key={selected.id} agent={selected} agents={query.data.data} prompts={prompts.data?.data || []} skillsets={skillsets.data?.data || []} toolsets={toolsets.data?.data || []} onCreated={async () => { setComposing(false); await client.invalidateQueries({ queryKey: ["agent-studio", "versions", selected.id] }); }}/>: <><dl><div><dt>Agent Key</dt><dd>{selected.key}</dd></div><div><dt>Owner</dt><dd>{selected.owner || "—"}</dd></div><div><dt>当前发布版本</dt><dd>{selected.active_version_id ? shortID(selected.active_version_id) : "尚未发布"}</dd></div><div><dt>更新时间</dt><dd>{formatDate(selected.updated_at)}</dd></div></dl><div className="agent-version-stack"><header><b>版本历史</b><small>{versions.data?.data?.length || 0} 个版本</small></header>{versions.isLoading ? <Skeleton rows={3}/> : versions.isError ? <ErrorState title="无法读取版本" error={versions.error}/> : versions.data?.data?.length ? versions.data.data.map((version, index) => <article key={version.id}><span><b>v{version.version}</b><small>{shortID(version.spec_hash)}</small></span><StatusBadge status={version.status}/><time>{formatDate(version.published_at || version.created_at)}</time><span className="agent-version-actions"><button type="button" onClick={() => setReview({ mode: "test", versionID: version.id })}><FlaskConical size={12}/>试运行</button><button type="button" disabled={!versions.data?.data?.[index + 1]} onClick={() => setReview({ mode: "diff", versionID: version.id })}><GitCompareArrows size={12}/>Diff</button>{version.status !== "published" ? <button type="button" disabled={release.isPending} onClick={() => release.mutate(version.id)}><Rocket size={12}/>发布</button> : null}</span></article>) : <p>尚无 AgentVersion。点击“创建版本”完成资源装配。</p>}</div>{review && versions.data?.data ? <AgentVersionReview agent={selected} versions={versions.data.data} review={review} onClose={() => setReview(undefined)} onOpenRun={(runID, sessionID) => navigate(`/agents?run=${encodeURIComponent(runID)}&session=${encodeURIComponent(sessionID)}`)}/> : null}</>}</section> : null}</div>}
  </StudioSection>;
}

function AgentVersionReview({ agent, versions, review, onClose, onOpenRun }: {
  agent: AgentDefinition;
  versions: AgentVersion[];
  review: { mode: "diff" | "test"; versionID: string };
  onClose: () => void;
  onOpenRun: (runID: string, sessionID: string) => void;
}) {
  const version = versions.find(item => item.id === review.versionID);
  const index = versions.findIndex(item => item.id === review.versionID);
  const previous = index >= 0 ? versions[index + 1] : undefined;
  const [question, setQuestion] = useState("请说明你的身份、目标、职责边界，以及当前可以使用的 Skill 和工具。");
  const [copied, setCopied] = useState(false);
  const test = useMutation({
    mutationFn: async () => {
      if (!version) throw new Error("AgentVersion 不存在");
      const session = await createAgentSession(agent.id, `[Test Run] ${agent.name} v${version.version}`);
      const run = await createAgentRun(version.id, { question: question.trim() }, session.data.id, "studio_test");
      return { runID: run.data.id, sessionID: session.data.id };
    },
    onSuccess: result => { toast.success(`v${version?.version} Test Run 已创建`, { description: "草稿版本已按不可变快照进入真实 Worker 执行链路。" }); onOpenRun(result.runID, result.sessionID); },
    onError: error => toast.error("Test Run 创建失败", { description: describeError(error) })
  });
  if (!version) return null;
  const changes = previous ? diffValues(previous.spec, version.spec) : [];
  return <section className="agent-version-review">
    <header><span>{review.mode === "diff" ? <GitCompareArrows size={15}/> : <FlaskConical size={15}/>}<span><b>{review.mode === "diff" ? `版本差异 · v${previous?.version ?? "—"} → v${version.version}` : `Test Run · ${agent.name} v${version.version}`}</b><small>{review.mode === "diff" ? `${changes.length} 个字段变化 · 比较不可变 Spec` : `${version.status} 版本 · 使用真实模型、Skill、Memory 和 Tool 链路`}</small></span></span><button type="button" onClick={onClose} title="关闭"><X size={14}/></button></header>
    {review.mode === "diff" ? <>{previous ? <div className="agent-version-diff"><div className="agent-version-diff-head"><span>字段</span><span>v{previous.version}</span><span>v{version.version}</span></div>{changes.length ? changes.map(change => <div key={change.path}><code>{change.path}</code><pre>{formatDiffValue(change.before)}</pre><pre>{formatDiffValue(change.after)}</pre></div>) : <p>两个版本的 Spec 内容完全一致；Hash 变化应视为异常。</p>}</div> : <p className="agent-studio-notice"><CircleAlert size={14}/>这是首个版本，没有前序版本可比较。</p>}<footer><span>Spec Hash：{shortID(previous?.spec_hash || "—")} → {shortID(version.spec_hash)}</span><button type="button" onClick={async () => { await navigator.clipboard.writeText(JSON.stringify({ from: previous?.spec, to: version.spec, changes }, null, 2)); setCopied(true); window.setTimeout(() => setCopied(false), 1400); }}><Copy size={12}/>{copied ? "已复制" : "复制完整 Diff"}</button></footer></> : <form onSubmit={event => { event.preventDefault(); test.mutate(); }}><label>测试输入<textarea required value={question} onChange={event => setQuestion(event.target.value)} /></label><p><ShieldCheck size={13}/>Test Run 可以执行草稿，但不会改变 Active Version；Run 会冻结当前版本 Spec，并进入正常审计轨迹。</p><footer><span>触发类型 studio_test · 创建独立 Session</span><button type="submit" disabled={test.isPending || !question.trim()}>{test.isPending ? "创建中…" : "运行并打开轨迹"}</button></footer></form>}
  </section>;
}

function AgentVersionComposer({ agent, agents, prompts, skillsets, toolsets, onCreated }: {
  agent: AgentDefinition;
  agents: AgentDefinition[];
  prompts: AgentPromptVersion[];
  skillsets: AgentSkillSetVersion[];
  toolsets: AgentToolSetVersion[];
  onCreated: () => Promise<void>;
}) {
  const [form, setForm] = useState({
    description: agent.description || "",
    role: "",
    goal: "",
    responsibilities: "",
    boundaries: "",
    communicationStyle: "简洁、结论优先，并明确区分事实与推断",
    promptID: prompts[0]?.id || "",
    toolsetID: toolsets[0]?.id || "",
    skillsetID: "",
    planningPolicy: "auto" as "auto" | "required" | "disabled",
    selectionPolicy: "auto" as "auto" | "pinned",
    provider: "vllm",
    serviceRef: "qwen-customer",
    modelID: "qwen35-4b-customer",
    modelCandidates: "qwen35-4b-customer",
    maxTurns: DEFAULT_HARNESS_MAX_TURNS,
    maxSteps: DEFAULT_HARNESS_MAX_STEPS,
    memoryEnabled: true,
    sandboxAutoExecute: true,
    delegatedAgentIDs: [] as string[]
  });
  const create = useMutation({
    mutationFn: () => {
      const prompt = prompts.find(item => item.id === form.promptID);
      const toolset = toolsets.find(item => item.id === form.toolsetID);
      const skillset = skillsets.find(item => item.id === form.skillsetID);
      if (!prompt || !toolset) throw new Error("必须选择 PromptVersion 和 ToolSetVersion");
      const candidates = form.modelCandidates.split(/[,\n]/).map(item => item.trim()).filter(Boolean);
      const spec: AgentVersionSpecInput = {
        name: agent.name,
        description: form.description.trim() || undefined,
        identity: {
          display_name: agent.name,
          role: form.role.trim(), goal: form.goal.trim(),
          responsibilities: lines(form.responsibilities), boundaries: lines(form.boundaries),
          communication_style: form.communicationStyle.trim()
        },
        harness: { name: "react-v1", max_turns: form.maxTurns, max_steps: form.maxSteps },
        planning: { policy: form.planningPolicy },
        model: {
          provider: form.provider.trim(), service_ref: form.serviceRef.trim(), capability: "chat",
          selection_policy: form.selectionPolicy,
          ...(form.selectionPolicy === "pinned" ? { model_id: form.modelID.trim() } : { model_candidates: candidates })
        },
        prompt_ref: { id: prompt.id, version: String(prompt.version) },
        toolset_ref: { id: toolset.id, version: String(toolset.version) },
        ...(skillset ? { skillset_ref: { id: skillset.id, version: String(skillset.version) } } : {}),
        input_schema: { type: "object", additionalProperties: true, properties: { question: { type: "string" } }, required: ["question"] },
        output_schema: { type: "object", additionalProperties: true },
        // 24k deployment profile: keep enough output headroom while leaving
        // room for dynamic Tool Schemas and the Context Collapse view.
        context: { reserve_output_tokens: 2048, recent_turn_tokens: 4096, memory_tokens: 1280, knowledge_tokens: 1024, tool_result_tokens: 3072, compaction: "context-collapse", collapse_trigger_ratio: 0.82 },
        memory: form.memoryEnabled ? { enabled: true, read_scopes: ["tenant", "agent", "user", "session"], read_layers: ["user", "project", "local", "auto"], read_types: ["user", "feedback", "project", "reference"], write_scope: "session", write_layer: "auto", max_recall: 8, candidate_limit: 20, router_enabled: false, router_top_k: 5, manifest_tokens: 384, repeat_suppression_turns: 3, minimum_score: 0 } : { enabled: false },
        runtime: { run_timeout: "30m", model_timeout: "2m", tool_timeout: "2m", max_model_calls: Math.max(DEFAULT_RUNTIME_MAX_MODEL_CALLS, form.maxTurns * form.maxSteps), max_tool_calls: Math.max(64, form.maxTurns * form.maxSteps * 2) },
        approval: { require_for: ["HIGH_RISK"], auto_approve_sandbox_command: form.sandboxAutoExecute, expires_in: "30m" },
        ...(form.delegatedAgentIDs.length ? { collaboration: {
          allowed_targets: agents.filter(item => form.delegatedAgentIDs.includes(item.id) && item.active_version_id).map(item => ({ agent_id: item.id, agent_version_id: item.active_version_id!, modes: ["sync" as const, "async" as const] })),
          max_depth: 3, max_fan_out: 2, max_child_runs: 8, share_session_memory: false, propagate_user_identity: true,
          child_timeout: 120_000_000_000, budget: { max_model_calls: 96, max_tool_calls: 192, max_tokens: 786432 }
        }} : {}),
        metadata: { created_from: "agent-studio" }
      };
      return createAgentVersion(agent.id, spec);
    },
    onSuccess: async result => { toast.success(`Agent v${result.data.version} 草稿已创建`, { description: "确认版本配置后，可在版本历史中发布。" }); await onCreated(); },
    onError: error => toast.error("创建 AgentVersion 失败", { description: describeError(error) })
  });
  const blocked = !prompts.length || !toolsets.length;
  return <form className="agent-version-composer" onSubmit={event => { event.preventDefault(); create.mutate(); }}>
    <div className="agent-composer-heading"><span><b>装配新版本</b><small>资源引用将在草稿中精确固定；发布后不可修改。</small></span><StatusBadge status="draft"/></div>
    {blocked ? <p className="agent-studio-notice"><CircleAlert size={14}/>缺少 {prompts.length ? "" : "PromptVersion"}{!prompts.length && !toolsets.length ? " 和 " : ""}{toolsets.length ? "" : "ToolSetVersion"}，请先创建资源。</p> : null}
    <label className="wide">版本说明<textarea value={form.description} onChange={event => setForm({ ...form, description: event.target.value })}/></label>
    <div className="agent-identity-heading"><span><b>结构化角色身份</b><small>发布时确定性编译到 Prompt 之前，并在轨迹中记录 Identity Digest。</small></span><StatusBadge status={form.role.trim()&&form.goal.trim()?"ready":"required"}/></div>
    <label>角色职责<input required value={form.role} onChange={event => setForm({ ...form, role: event.target.value })} placeholder="例如：发布质量审查员"/></label>
    <label className="wide">目标<input required value={form.goal} onChange={event => setForm({ ...form, goal: event.target.value })} placeholder="例如：基于健康、性能和回归证据给出发布结论"/></label>
    <label className="wide">责任清单（每行一项）<textarea value={form.responsibilities} onChange={event => setForm({ ...form, responsibilities: event.target.value })} placeholder={'检查服务健康\n运行基准测试'}/></label>
    <label className="wide">行为边界（每行一项）<textarea value={form.boundaries} onChange={event => setForm({ ...form, boundaries: event.target.value })} placeholder={'不得直接修改生产流量\n证据不足时不得判定通过'}/></label>
    <label className="wide">沟通风格<input value={form.communicationStyle} onChange={event => setForm({ ...form, communicationStyle: event.target.value })}/></label>
    <label>Prompt Version<select required value={form.promptID} onChange={event => setForm({ ...form, promptID: event.target.value })}><option value="">请选择</option>{prompts.map(item => <option value={item.id} key={item.id}>{item.name} · v{item.version}</option>)}</select></label>
    <label>SkillSet Version<select value={form.skillsetID} onChange={event => setForm({ ...form, skillsetID: event.target.value })}><option value="">不注入 Skill</option>{skillsets.map(item => <option value={item.id} key={item.id}>{item.name} · v{item.version} · {item.spec.skills.length} Skills</option>)}</select></label>
    <label>ToolSet Version<select required value={form.toolsetID} onChange={event => setForm({ ...form, toolsetID: event.target.value })}><option value="">请选择</option>{toolsets.map(item => <option value={item.id} key={item.id}>{item.name} · v{item.version} · {item.spec.tools.length} Tools</option>)}</select></label>
    <label>规划策略<select value={form.planningPolicy} onChange={event => setForm({ ...form, planningPolicy: event.target.value as "auto" | "required" | "disabled" })}><option value="auto">自动判断（推荐）</option><option value="required">始终制定计划</option><option value="disabled">仅对话，不执行工具</option></select><small>自动模式允许简单问题直接回答；文件、命令、MCP 和 Agent 委派会先建立持久化计划。</small></label>
    <label>模型策略<select value={form.selectionPolicy} onChange={event => setForm({ ...form, selectionPolicy: event.target.value as "auto" | "pinned" })}><option value="auto">自动探测并冻结</option><option value="pinned">固定模型</option></select><small>自动模式会探测可调用模型，并把模型、服务及实际上下文窗口冻结到本次 Run。</small></label>
    <label>Provider<input required value={form.provider} onChange={event => setForm({ ...form, provider: event.target.value })}/></label>
    <label>Service Ref<input required value={form.serviceRef} onChange={event => setForm({ ...form, serviceRef: event.target.value })}/></label>
    {form.selectionPolicy === "auto" ? <label className="wide">候选模型（逗号或换行分隔）<textarea value={form.modelCandidates} onChange={event => setForm({ ...form, modelCandidates: event.target.value })}/></label> : <label className="wide">固定 Model ID<input required value={form.modelID} onChange={event => setForm({ ...form, modelID: event.target.value })}/></label>}
    <label>最大 Turn<input type="number" min="1" max="50" value={form.maxTurns} onChange={event => setForm({ ...form, maxTurns: Number(event.target.value) })}/></label>
    <label>最大 Step<input type="number" min="1" max="100" value={form.maxSteps} onChange={event => setForm({ ...form, maxSteps: Number(event.target.value) })}/></label>
    <label className="agent-checkbox"><input type="checkbox" checked={form.memoryEnabled} onChange={event => setForm({ ...form, memoryEnabled: event.target.checked })}/><span><b>启用分层记忆</b><small>按租户、Agent、用户和 Session 召回</small></span></label>
    <label className="agent-checkbox"><input type="checkbox" checked={form.sandboxAutoExecute} onChange={event => setForm({ ...form, sandboxAutoExecute: event.target.checked })}/><span><b>允许自主执行 Sandbox 代码</b><small>无需逐次审批；仅限当前 Run 隔离工作区、Python 白名单、30 秒超时，无 Shell 与公网权限</small></span></label>
    <fieldset className="agent-collaboration-targets"><legend>允许委派的 Agent（仅已发布版本）</legend>{agents.filter(item=>item.id!==agent.id&&item.active_version_id).length?agents.filter(item=>item.id!==agent.id&&item.active_version_id).map(item=><label key={item.id}><input type="checkbox" checked={form.delegatedAgentIDs.includes(item.id)} onChange={()=>setForm({...form,delegatedAgentIDs:toggleList(form.delegatedAgentIDs,item.id)})}/><span><b>{item.name}</b><small>{item.key} · 固定 {shortID(item.active_version_id!)}</small></span></label>):<p>暂无其他已发布 Agent。先发布目标 Agent，才能加入调用白名单。</p>}</fieldset>
    <footer><span>默认执行预算：{DEFAULT_HARNESS_MAX_TURNS} Turn × {DEFAULT_HARNESS_MAX_STEPS} Step，最多 {DEFAULT_RUNTIME_MAX_MODEL_CALLS} 次模型调用；上下文窗口从模型 /v1/models 的 max_model_len 自动读取并冻结到 Run。</span><button type="submit" disabled={blocked || create.isPending || !form.role.trim() || !form.goal.trim()}>{create.isPending ? "校验并保存中…" : "创建草稿版本"}</button></footer>
  </form>;
}

function PromptsPanel({ query }: { query: UseQueryResult<{ data: AgentPromptVersion[] }> }) {
  const client = useQueryClient();
  const [creating, setCreating] = useState(false);
  const [selectedID, setSelectedID] = useState<string>();
  const [form, setForm] = useState({ key: "", name: "", content: "" });
  const create = useMutation({ mutationFn: createAgentPromptVersion, onSuccess: async result => { toast.success(`Prompt ${result.data.name} v${result.data.version} 已创建`); setSelectedID(result.data.id); setCreating(false); setForm({ key: "", name: "", content: "" }); await client.invalidateQueries({ queryKey: ["agent-studio", "prompts"] }); }, onError: error => toast.error("创建 Prompt 失败", { description: describeError(error) }) });
  if (query.isLoading) return <Skeleton rows={8}/>;
  if (query.isError) return <ErrorState title="无法读取 Prompt" error={query.error} onRetry={() => query.refetch()}/>;
  const selected = query.data?.data?.find(item => item.id === selectedID) || query.data?.data?.[0];
  return <StudioSection title="Prompt Registry" detail="每次修改创建新版本，历史版本保持可回放" action={<button className="studio-primary" type="button" onClick={() => setCreating(value => !value)}><Plus size={14}/>{creating ? "取消" : "新建版本"}</button>}>
    {creating ? <form className="agent-prompt-form" onSubmit={event => { event.preventDefault(); create.mutate(form); }}><div><label>Prompt Key<input required value={form.key} onChange={event => setForm({ ...form, key: event.target.value })} placeholder="release-system"/></label><label>名称<input required value={form.name} onChange={event => setForm({ ...form, name: event.target.value })} placeholder="发布验证系统提示词"/></label></div><label>Prompt 内容<textarea required value={form.content} onChange={event => setForm({ ...form, content: event.target.value })} placeholder="定义身份以外的任务规则、输入约束和输出要求…"/></label><footer><span>{form.content.length} 字符 · 发布后不可修改</span><button type="submit" disabled={create.isPending || !form.key.trim() || !form.name.trim() || !form.content.trim()}>{create.isPending ? "保存中…" : "创建不可变版本"}</button></footer></form> : null}
    {!query.data?.data?.length ? <EmptyState title="暂无 Prompt" description="创建 Prompt 后才能装配 AgentVersion。" icon={FileCode2}/> : <ResourceSplit items={query.data.data} selectedID={selected?.id} onSelect={setSelectedID} renderMeta={item => `v${item.version} · ${formatDate(item.created_at)}`} renderDetail={item => <><ResourceIdentity item={item}/><pre className="agent-resource-content">{item.content}</pre></>}/>}
  </StudioSection>;
}

function SkillsPanel({ query, skillsets }: { query: UseQueryResult<{ data: AgentSkillVersion[] }>; skillsets: UseQueryResult<{ data: AgentSkillSetVersion[] }> }) {
  const client = useQueryClient();
  const input = useRef<HTMLInputElement>(null);
  const [selectedID, setSelectedID] = useState<string>();
  const [assembling, setAssembling] = useState(false);
  const [setForm, setSetForm] = useState({ key: "", name: "", selected: [] as string[] });
  const upload = useMutation({ mutationFn: uploadAgentSkillVersion, onSuccess: async result => { toast.success(`${result.data.name} v${result.data.version} 上传成功`, { description: "Skill 已解析、校验并保存为不可变版本。" }); setSelectedID(result.data.id); await client.invalidateQueries({ queryKey: ["agent-studio", "skills"] }); }, onError: error => toast.error("Skill 上传失败", { description: describeError(error) }) });
  const createSet = useMutation({ mutationFn: () => createAgentSkillSetVersion({ key: setForm.key.trim(), name: setForm.name.trim(), skills: latestSkills.filter(item => setForm.selected.includes(item.id)).map(item => ({ id: item.id, version: String(item.version) })) }), onSuccess: async result => { toast.success(`${result.data.name} v${result.data.version} 已创建`, { description: "现在可在 Agent 版本装配器中选择该 SkillSet。" }); setAssembling(false); setSetForm({ key: "", name: "", selected: [] }); await client.invalidateQueries({ queryKey: ["agent-studio", "skillsets"] }); }, onError: error => toast.error("创建 SkillSet 失败", { description: describeError(error) }) });
  if (query.isLoading) return <Skeleton rows={8}/>;
  if (query.isError) return <ErrorState title="无法读取 Skill Catalog" error={query.error} onRetry={() => query.refetch()}/>;
  const selected = query.data?.data?.find(item => item.id === selectedID) || query.data?.data?.[0];
  const latestSkills = latestDefinitionVersions(query.data?.data || []);
  return <StudioSection title="Skill Catalog" detail={`上传 SkillVersion，并组合为 Agent 可引用的 SkillSet · ${skillsets.data?.data?.length || 0} 个 SkillSet`} action={<span className="agent-section-actions"><button type="button" onClick={() => setAssembling(value => !value)}><Boxes size={14}/>{assembling ? "取消组合" : "创建 SkillSet"}</button><input ref={input} hidden type="file" accept=".md,.markdown,.json,text/markdown,application/json" onChange={event => { const file = event.target.files?.[0]; if (file) upload.mutate(file); event.target.value = ""; }}/><button className="studio-primary" type="button" disabled={upload.isPending} onClick={() => input.current?.click()}><Upload size={14}/>{upload.isPending ? "上传解析中…" : "上传 Skill"}</button></span>}>
    <div className="agent-upload-guide"><Upload size={20}/><span><b>支持标准 SKILL.md</b><small>最大 1 MiB · UTF-8 · YAML Front Matter 至少包含 name 与 key；Markdown 正文作为指令注入。</small></span><code>---{`\n`}name: Release Reviewer{`\n`}key: release-reviewer{`\n`}description: Reviews releases{`\n`}---{`\n`}# Workflow…</code></div>
    {assembling ? <form className="agent-skillset-form" onSubmit={event => { event.preventDefault(); createSet.mutate(); }}><div><label>SkillSet Key<input required pattern="[a-z][a-z0-9._-]*" value={setForm.key} onChange={event => setSetForm({ ...setForm, key: event.target.value })} placeholder="release-agent-skills"/></label><label>显示名称<input required value={setForm.name} onChange={event => setSetForm({ ...setForm, name: event.target.value })} placeholder="Release Agent Skills"/></label></div><fieldset><legend>选择 Skill 最新版本</legend>{latestSkills.map(item => <label key={item.id}><input type="checkbox" checked={setForm.selected.includes(item.id)} onChange={() => setSetForm({ ...setForm, selected: toggleList(setForm.selected, item.id) })}/><span><b>{item.name} · v{item.version}</b><small>{item.spec.description || item.key}</small></span></label>)}</fieldset><footer><span>SkillSet 同样不可变；Skill 更新后需创建新的 SkillSet 版本。</span><button type="submit" disabled={createSet.isPending || !setForm.key.trim() || !setForm.name.trim()}>{createSet.isPending ? "创建中…" : `创建 SkillSet（${setForm.selected.length}）`}</button></footer></form> : null}
    {!query.data?.data?.length ? <EmptyState title="暂无 Skill" description="上传第一个 SKILL.md，上传结果会立即出现在此处。" icon={Sparkles}/> : <ResourceSplit items={query.data.data} selectedID={selected?.id} onSelect={setSelectedID} renderMeta={item => `v${item.version} · ${item.spec.instructions.length} 个指令块`} renderDetail={item => <><ResourceIdentity item={item}/><div className="agent-skill-summary"><span><small>描述</small><b>{item.spec.description || "未填写"}</b></span><span><small>所需工具</small><b>{item.spec.required_tools?.length || 0}</b></span><span><small>示例</small><b>{item.spec.examples?.length || 0}</b></span></div>{item.spec.instructions.map((instruction, index) => <section className="agent-instruction-block" key={`${instruction.name}-${index}`}><header><b>{instruction.name}</b><span>优先级 {instruction.priority || 0}</span></header><pre>{instruction.content}</pre></section>)}</>}/>}
  </StudioSection>;
}

function ToolsPanel({ tools, toolsets }: { tools: UseQueryResult<{ data: AgentToolVersion[] }>; toolsets: UseQueryResult<{ data: AgentToolSetVersion[] }> }) {
  const client = useQueryClient();
  const items = tools.data?.data || [];
  const providerCounts = useMemo(() => items.reduce((result: Record<string, number>, item) => { const key = item.spec?.provider_type || "unknown"; result[key] = (result[key] || 0) + 1; return result; }, {}), [items]);
  const latestTools = latestDefinitionVersions(items);
  const [creating, setCreating] = useState(false);
  const [assembling, setAssembling] = useState(false);
  const [form, setForm] = useState({ operation: "read_file", key: "workspace-read-file", name: "Workspace Read File", maxBytes: 262144 });
  const [toolSetForm, setToolSetForm] = useState({ key: "workspace-tools", name: "Workspace Tools", selected: [] as string[] });
  const create = useMutation({ mutationFn: () => createAgentToolVersion(workspaceToolInput(form.operation, form.key.trim(), form.name.trim(), form.maxBytes)), onSuccess: async result => { toast.success(`${result.data.name} v${result.data.version} 已创建`, { description: "内置 Workspace Provider 已绑定路径边界和结果限额。" }); setCreating(false); await client.invalidateQueries({ queryKey: ["agent-studio", "tools"] }); }, onError: error => toast.error("创建文件 Tool 失败", { description: describeError(error) }) });
  const createSet = useMutation({ mutationFn: () => createAgentToolSetVersion({ key: toolSetForm.key.trim(), name: toolSetForm.name.trim(), tools: latestTools.filter(item => toolSetForm.selected.includes(item.id)).map(item => ({ id: item.id, version: String(item.version) })) }), onSuccess: async result => { toast.success(`${result.data.name} v${result.data.version} 已创建`, { description: "现在可装配到 AgentVersion。" }); setAssembling(false); setToolSetForm(current => ({ ...current, selected: [] })); await client.invalidateQueries({ queryKey: ["agent-studio", "toolsets"] }); }, onError: error => toast.error("创建 ToolSet 失败", { description: describeError(error) }) });
  if (tools.isLoading || toolsets.isLoading) return <Skeleton rows={8}/>;
  if (tools.isError) return <ErrorState title="无法读取 Tool Registry" error={tools.error} onRetry={() => tools.refetch()}/>;
  return <StudioSection title="Tools & ToolSets" detail="HTTP 与受边界约束的 Workspace Provider" action={<span className="agent-section-actions"><button type="button" onClick={() => setAssembling(value => !value)}><Boxes size={14}/>{assembling ? "取消组合" : "创建 ToolSet"}</button><button className="studio-primary" type="button" onClick={() => setCreating(value => !value)}><Plus size={14}/>{creating ? "取消创建" : "文件 Tool"}</button></span>}>
    <div className="agent-resource-kpis"><span><Wrench size={17}/><b>{items.length}</b><small>Tool Version</small></span><span><Braces size={17}/><b>{toolsets.data?.data?.length || 0}</b><small>ToolSet Version</small></span><span><CheckCircle2 size={17}/><b>{providerCounts.http || 0}</b><small>HTTP Provider</small></span><span><FileCode2 size={17}/><b>{providerCounts.workspace || 0}</b><small>Workspace Provider</small></span><span><Network size={17}/><b>{providerCounts.mcp || 0}</b><small>MCP Provider</small></span></div>
    {creating ? <form className="agent-workspace-tool-form" onSubmit={event => { event.preventDefault(); create.mutate(); }}><header><span><b>创建内置工作区工具</b><small>每个 Run 使用隔离目录，Agent 和 ToolVersion 都不能选择或越过根目录。</small></span><StatusBadge status={workspaceRisk(form.operation)}/></header><div><label>操作<select value={form.operation} onChange={event => { const operation=event.target.value; setForm(current => ({ ...current, operation, key: `workspace-${operation.replace(/_/g,"-")}`, name: `Workspace ${titleWords(operation)}` })); }}><option value="read_file">读取文件</option><option value="list_files">列出文件</option><option value="search_files">搜索文件</option><option value="write_file">创建/覆盖文件</option><option value="append_file">分块追加文件</option><option value="edit_file">精确编辑</option><option value="promote_file">原子发布临时文件</option><option value="run_command">受控 Python 命令</option></select></label><label>Tool Key<input required value={form.key} onChange={event => setForm({ ...form, key: event.target.value })}/></label><label>显示名称<input required value={form.name} onChange={event => setForm({ ...form, name: event.target.value })}/></label><label>单文件上限<input type="number" min="1024" max="1048576" value={form.maxBytes} onChange={event => setForm({ ...form, maxBytes: Number(event.target.value)||262144 })}/></label></div><p><ShieldCheck size={13}/>绝对路径、<code>..</code> 与逃逸根目录的符号链接会被拒绝；读写返回 SHA-256，写入和命令执行进入审批及审计轨迹。</p><footer><span>{workspaceMutating(form.operation) ? "写操作还受部署策略和人工审批控制。" : "只读操作可并行执行。"}</span><button type="submit" disabled={create.isPending || !form.key.trim() || !form.name.trim()}>{create.isPending ? "创建中…" : "创建不可变 ToolVersion"}</button></footer></form> : null}
    {assembling ? <form className="agent-skillset-form" onSubmit={event => { event.preventDefault(); createSet.mutate(); }}><div><label>ToolSet Key<input required pattern="[a-z][a-z0-9._-]*" value={toolSetForm.key} onChange={event => setToolSetForm({ ...toolSetForm, key: event.target.value })}/></label><label>显示名称<input required value={toolSetForm.name} onChange={event => setToolSetForm({ ...toolSetForm, name: event.target.value })}/></label></div><fieldset><legend>选择 Tool 最新版本</legend>{latestTools.map(item => <label key={item.id}><input type="checkbox" checked={toolSetForm.selected.includes(item.id)} onChange={() => setToolSetForm({ ...toolSetForm, selected: toggleList(toolSetForm.selected, item.id) })}/><span><b>{item.name} · v{item.version}</b><small>{item.spec?.provider_type} · {item.spec?.definition?.risk}</small></span></label>)}</fieldset><footer><span>ToolSet 固定精确版本；工具升级后需创建新的 ToolSet 版本。</span><button type="submit" disabled={createSet.isPending || !toolSetForm.key.trim() || !toolSetForm.name.trim()}>{createSet.isPending ? "创建中…" : `创建 ToolSet（${toolSetForm.selected.length}）`}</button></footer></form> : null}
    {!items.length ? <EmptyState title="暂无 Tool" description="创建 read_file、list_files、search_files、write_file、edit_file 或 promote_file。" icon={Wrench}/> : <div className="agent-tool-grid">{items.map(item => <article key={item.id}><header><span><b>{item.name}</b><small>{item.key} · v{item.version}</small></span><StatusBadge status={item.spec?.provider_type || "unknown"}/></header><p>{item.spec?.definition?.description || "未填写工具说明"}</p><dl><div><dt>风险</dt><dd>{item.spec?.definition?.risk || "未声明"}</dd></div><div><dt>Provider</dt><dd>{item.spec?.workspace?.operation || item.spec?.http?.endpoint || "—"}</dd></div></dl></article>)}</div>}
  </StudioSection>;
}

function EnvironmentPanel({ environments, installs }: { environments: UseQueryResult<{ data: AgentEnvironmentTemplate[] }>; installs: UseQueryResult<{ data: AgentDependencyInstall[] }> }) {
  if (environments.isLoading || installs.isLoading) return <Skeleton rows={8}/>;
  if (environments.isError) return <ErrorState title="无法读取运行环境模板" error={environments.error} onRetry={() => environments.refetch()}/>;
  return <StudioSection title="Execution Environments" detail="稳定依赖固化在不可变模板；未知依赖按包、版本、来源和 Run 作用域逐次审批">
    <section className="agent-collaboration-block"><header><span><b>预构建模板</b><small>Agent 不能在线修改模板，也不能向安装器提交命令。</small></span></header><div className="agent-mcp-list">{environments.data?.data.length ? environments.data.data.map(item => <article key={item.id}><header><span><b>{item.name} · v{item.version}</b><small>{item.key} · {item.runtime} · {item.image_ref}</small></span><StatusBadge status={item.status}/></header><div><span><b>稳定依赖</b><small>{item.dependencies.length ? item.dependencies.map(dep => `${dep.name}==${dep.version}`).join(" · ") : "仅基础运行时"}</small></span><span><b>边界</b><small>无 Shell · 无运行时公网 · Run 工作区隔离</small></span></div></article>) : <EmptyState title="暂无环境模板" description="迁移完成后会显示部署方维护的不可变模板。"/>}</div></section>
    <section className="agent-collaboration-block"><header><span><b>Run 级依赖安装记录</b><small>临时依赖不会污染宿主机、模板或其他 Run。</small></span></header>{installs.isError ? <ErrorState title="无法读取依赖安装审计" error={installs.error} onRetry={() => installs.refetch()}/> : <div className="agent-mcp-list">{installs.data?.data.length ? installs.data.data.map(item => <article key={item.id}><header><span><b>{item.packages.map(dep => `${dep.name}==${dep.version}`).join(", ")}</b><small>Run {shortID(item.run_id)} · {item.source} · scope={item.scope}</small></span><StatusBadge status={item.status}/></header>{item.error ? <pre>{item.error}</pre> : null}</article>) : <EmptyState title="暂无临时依赖" description="Agent 请求模板外依赖并获批准后，记录会出现在这里。"/>}</div>}</section>
  </StudioSection>;
}

function CollaborationPanel({agents}:{agents:AgentDefinition[]}) {
  const client=useQueryClient();
  const servers=useQuery({queryKey:["agent-studio","mcp"],queryFn:listMCPServerVersions,retry:false});
  const [creating,setCreating]=useState(false);
  const [form,setForm]=useState({key:"",name:"",endpoint:"",authHeader:"Authorization",secretEnv:""});
  const [card,setCard]=useState<{agent:string;value:Record<string,unknown>}>();
  const [mcpResult,setMCPResult]=useState<{server:string;mode:"test"|"sync";value?:MCPServerVersion;error?:string;at:string}>();
  const publishedAgents=agents.filter(agent=>Boolean(agent.active_version_id));
  const [a2aAgentID,setA2AAgentID]=useState("");
  const selectedA2AAgent=publishedAgents.find(agent=>agent.id===a2aAgentID)||publishedAgents[0];
  const [a2aInput,setA2AInput]=useState("请检查当前运行环境并简要返回结果。");
  const [a2aTask,setA2ATask]=useState<A2ATask>();
  const [a2aEvents,setA2AEvents]=useState<Array<{at:string;event:string;data:unknown}>>([]);
  const streamController=useRef<AbortController>();
  useEffect(()=>()=>streamController.current?.abort(),[]);
  const create=useMutation({mutationFn:()=>createMCPServerVersion({key:form.key.trim(),name:form.name.trim(),endpoint:form.endpoint.trim(),header_environment:form.secretEnv.trim()?{[form.authHeader.trim()||"Authorization"]:form.secretEnv.trim()}:undefined}),onSuccess:async()=>{setCreating(false);setForm({key:"",name:"",endpoint:"",authHeader:"Authorization",secretEnv:""});await client.invalidateQueries({queryKey:["agent-studio","mcp"]});toast.success("MCP ServerVersion 已注册");},onError:error=>toast.error("MCP 注册失败",{description:describeError(error)})});
  const action=useMutation({mutationFn:({id,mode}:{id:string;mode:"test"|"sync"})=>mode==="test"?testMCPServerVersion(id):syncMCPServerTools(id),onSuccess:async(result,variables)=>{const server=(servers.data?.data||[]).find(item=>item.id===variables.id);setMCPResult({server:server?.name||variables.id,mode:variables.mode,value:result.data,at:new Date().toISOString()});await client.invalidateQueries({queryKey:["agent-studio","mcp"]});toast.success(result.data.health?.status==="healthy"?"MCP 连接正常":"MCP 操作完成");},onError:(error,variables)=>{const server=(servers.data?.data||[]).find(item=>item.id===variables.id);setMCPResult({server:server?.name||variables.id,mode:variables.mode,error:describeError(error),at:new Date().toISOString()});toast.error("MCP 连接失败",{description:describeError(error)})}});
  const materialize=useMutation({mutationFn:(tool:{serverID:string;snapshot:MCPToolSnapshot})=>createAgentToolVersion({key:`mcp-${tool.snapshot.name.toLowerCase().replace(/[^a-z0-9._-]+/g,"-")}`,name:`MCP · ${tool.snapshot.name}`,spec:{definition:{name:tool.snapshot.name,version:"1",description:tool.snapshot.description||`MCP tool ${tool.snapshot.name}`,input_schema:tool.snapshot.input_schema,risk:tool.snapshot.risk||"READ",execution_mode:"serial"},provider_type:"mcp",mcp:{server_version_id:tool.serverID,tool_name:tool.snapshot.name,schema_hash:tool.snapshot.schema_hash}}}),onSuccess:async()=>{await client.invalidateQueries({queryKey:["agent-studio","tools"]});toast.success("MCP ToolVersion 已创建",{description:"Schema Hash 和 ServerVersion 已精确固定，可加入 ToolSet。"});},onError:error=>toast.error("创建 MCP ToolVersion 失败",{description:describeError(error)})});
  const inspectCard=async(agent:AgentDefinition)=>{try{const value=await getA2AAgentCard(agent.id);setCard({agent:agent.name,value});}catch(error){toast.error("读取 Agent Card 失败",{description:describeError(error)})}};
  const appendA2AEvent=(event:string,data:unknown)=>setA2AEvents(current=>[...current,{at:new Date().toISOString(),event,data}]);
  const send=useMutation({mutationFn:()=>sendA2AMessage(selectedA2AAgent!.id,a2aInput.trim()),onSuccess:result=>{setA2ATask(result.task);setA2AEvents([{at:new Date().toISOString(),event:"message:send",data:result.task}]);toast.success("A2A Task 已创建",{description:result.task.id});},onError:error=>{appendA2AEvent("message:send/error",describeError(error));toast.error("A2A 消息发送失败",{description:describeError(error)})}});
  const refreshTask=async()=>{if(!a2aTask)return;try{const value=await getA2ATask(a2aTask.id);setA2ATask(value);appendA2AEvent("tasks/get",value);}catch(error){appendA2AEvent("tasks/get/error",describeError(error));toast.error("查询 A2A Task 失败",{description:describeError(error)})}};
  const cancelTask=async()=>{if(!a2aTask)return;try{const value=await cancelA2ATask(a2aTask.id);setA2ATask(value);appendA2AEvent("tasks/cancel",value);}catch(error){appendA2AEvent("tasks/cancel/error",describeError(error));toast.error("取消 A2A Task 失败",{description:describeError(error)})}};
  const subscribeTask=async()=>{if(!a2aTask)return;streamController.current?.abort();const controller=new AbortController();streamController.current=controller;appendA2AEvent("tasks/subscribe",{taskId:a2aTask.id,state:"connected"});try{await subscribeA2ATask(a2aTask.id,(event:A2AStreamEvent)=>{appendA2AEvent(event.event,event.data);const statusUpdate=event.data.statusUpdate as {status?:A2ATask["status"]}|undefined;if(statusUpdate?.status)setA2ATask(current=>current?{...current,status:statusUpdate.status!}:current);},controller.signal);}catch(error){if(!controller.signal.aborted){appendA2AEvent("tasks/subscribe/error",describeError(error));toast.error("A2A SSE 订阅失败",{description:describeError(error)})}}};
  return <StudioSection title="Collaboration" detail="MCP 2025-06-18、平台内委派和 A2A 1.0 的实际控制面">
    <div className="agent-coming-grid"><article><i><Network size={19}/></i><span><b>MCP Server Registry</b><small>连接、发现、Schema 快照并固定到 ToolVersion。</small></span><StatusBadge status="available"/></article><article><i><Bot size={19}/></i><span><b>内部 Agent 委派</b><small>目标白名单、父子 Run、深度/数量预算、循环检测与级联取消。</small></span><StatusBadge status="available"/></article><article><i><Boxes size={19}/></i><span><b>A2A 1.0 HTTP+JSON</b><small>Agent Card、Message/Task、SSE 状态和 Run 输出 Artifact。</small></span><StatusBadge status="available"/></article></div>
    <section className="agent-collaboration-block"><header><span><b>MCP Server Versions</b><small>Secret 仅填写环境变量名；Endpoint 必须位于运维允许的 Host 白名单。</small></span><button type="button" onClick={()=>setCreating(value=>!value)}><Plus size={13}/>{creating?"取消":"注册 Server"}</button></header>{creating?<form onSubmit={event=>{event.preventDefault();create.mutate()}}><label>Key<input required pattern="[a-z][a-z0-9._-]*" value={form.key} onChange={event=>setForm({...form,key:event.target.value})}/></label><label>名称<input required value={form.name} onChange={event=>setForm({...form,name:event.target.value})}/></label><label className="wide">Streamable HTTP Endpoint<input required type="url" value={form.endpoint} onChange={event=>setForm({...form,endpoint:event.target.value})} placeholder="https://mcp.example.com/mcp"/></label><label>认证 Header<input value={form.authHeader} onChange={event=>setForm({...form,authHeader:event.target.value})}/></label><label>Secret 环境变量<input value={form.secretEnv} onChange={event=>setForm({...form,secretEnv:event.target.value})} placeholder="MCP_TOKEN"/></label><footer><span>协议版本固定为 2025-06-18。</span><button type="submit" disabled={create.isPending}>创建不可变版本</button></footer></form>:null}<div className="agent-mcp-list">{servers.isLoading?<Skeleton rows={3}/>:servers.isError?<ErrorState title="无法读取 MCP Registry" error={servers.error}/>:servers.data?.data?.length?servers.data.data.map(server=><article key={server.id}><header><span><b>{server.name} · v{server.version}</b><small>{server.spec.endpoint}</small></span><StatusBadge status={server.health?.status||"unchecked"}/><button type="button" disabled={action.isPending} onClick={()=>action.mutate({id:server.id,mode:"test"})}>连接测试</button><button type="button" disabled={action.isPending} onClick={()=>action.mutate({id:server.id,mode:"sync"})}>发现 Tools</button></header>{server.health?.error?<pre>{server.health.error}</pre>:null}{(server.tools||[]).map(tool=><div key={tool.id||tool.name}><span><b>{tool.name}</b><small>{tool.description||"无描述"} · {shortID(tool.schema_hash)}</small></span><StatusBadge status={tool.risk}/><button type="button" disabled={materialize.isPending} onClick={()=>materialize.mutate({serverID:server.id,snapshot:tool})}>创建 ToolVersion</button></div>)}</article>):<EmptyState title="暂无 MCP Server" description="注册 ServerVersion 后执行连接测试与 Tool 发现。"/>}</div>{mcpResult?<ProtocolResult title={`${mcpResult.server} · ${mcpResult.mode==="test"?"连接测试":"Tool 发现"}`} status={mcpResult.error?"failed":"completed"} steps={mcpResult.mode==="test"?["initialize","notifications/initialized"]:["initialize","notifications/initialized","tools/list","保存 Schema 快照"]} value={mcpResult.error?{error:mcpResult.error}:mcpResult.value} at={mcpResult.at}/>:null}</section>
    <section className="agent-collaboration-block"><header><span><b>A2A 协议测试台</b><small>逐项验证 Agent Card、Message、Task 查询、SSE 状态流与取消。</small></span></header><div className="agent-a2a-workbench"><div className="agent-a2a-form"><label>目标 Agent<select value={selectedA2AAgent?.id||""} onChange={event=>setA2AAgentID(event.target.value)} disabled={!publishedAgents.length}>{publishedAgents.length?publishedAgents.map(agent=><option value={agent.id} key={agent.id}>{agent.name}</option>):<option value="">没有已发布 Agent</option>}</select></label><label>测试消息<textarea value={a2aInput} onChange={event=>setA2AInput(event.target.value)}/></label><div className="agent-a2a-actions"><button type="button" disabled={!selectedA2AAgent} onClick={()=>selectedA2AAgent&&void inspectCard(selectedA2AAgent)}>1. Agent Card</button><button className="primary" type="button" disabled={!selectedA2AAgent||!a2aInput.trim()||send.isPending} onClick={()=>send.mutate()}>2. 发送 Message</button><button type="button" disabled={!a2aTask} onClick={()=>void refreshTask()}>3. 查询 Task</button><button type="button" disabled={!a2aTask} onClick={()=>void subscribeTask()}>4. 订阅 SSE</button><button className="danger" type="button" disabled={!a2aTask||a2aTask.status.state==="completed"||a2aTask.status.state==="failed"||a2aTask.status.state==="canceled"} onClick={()=>void cancelTask()}>取消 Task</button></div>{a2aTask?<dl className="agent-a2a-summary"><div><dt>Task ID</dt><dd>{a2aTask.id}</dd></div><div><dt>Context ID</dt><dd>{a2aTask.contextId}</dd></div><div><dt>状态</dt><dd><StatusBadge status={a2aTask.status.state}/></dd></div><div><dt>Run ID</dt><dd>{String(a2aTask.metadata?.runId||"—")}</dd></div></dl>:null}</div><div className="agent-protocol-output"><header><b>协议事件</b><button type="button" disabled={!a2aEvents.length} onClick={()=>setA2AEvents([])}>清空</button></header>{a2aEvents.length?<ol>{a2aEvents.map((entry,index)=><li key={`${entry.at}-${index}`}><span><b>{entry.event}</b><time>{formatProtocolTime(entry.at)}</time></span><pre>{JSON.stringify(entry.data,null,2)}</pre></li>)}</ol>:<p>发送测试消息后，这里会按真实发生顺序记录协议事件。</p>}</div></div>{card?<div className="agent-a2a-card"><header><strong>{card.agent} · Agent Card</strong><button type="button" onClick={()=>void navigator.clipboard.writeText(JSON.stringify(card.value,null,2))}><Copy size={12}/>复制</button></header><pre>{JSON.stringify(card.value,null,2)}</pre></div>:null}</section>
  </StudioSection>;
}

function ProtocolResult({title,status,steps,value,at}:{title:string;status:"completed"|"failed";steps:string[];value:unknown;at:string}){
  return <div className="agent-protocol-result"><header><span><b>{title}</b><small>{formatProtocolTime(at)}</small></span><StatusBadge status={status}/></header><ol>{steps.map((step,index)=><li key={step}><i>{index+1}</i><span>{step}</span></li>)}</ol><pre>{JSON.stringify(value,null,2)}</pre></div>;
}

function formatProtocolTime(value:string){return new Date(value).toLocaleTimeString("zh-CN",{hour12:false})}

function workspaceToolInput(operation:string,key:string,name:string,maxBytes:number){
  const schemas:Record<string,Record<string,unknown>>={
    read_file:{type:"object",additionalProperties:false,required:["path"],properties:{path:{type:"string",description:"工作区相对路径"},start_line:{type:"integer",minimum:1},line_count:{type:"integer",minimum:1}}},
    list_files:{type:"object",additionalProperties:false,properties:{path:{type:"string",description:"工作区相对目录，默认为根目录"},recursive:{type:"boolean"},max_entries:{type:"integer",minimum:1,maximum:500}}},
    search_files:{type:"object",additionalProperties:false,required:["query"],properties:{path:{type:"string"},query:{type:"string",minLength:1},max_results:{type:"integer",minimum:1,maximum:200}}},
    write_file:{type:"object",additionalProperties:false,required:["path","content"],properties:{path:{type:"string",description:"Run 工作区相对路径；必须先生成此字段。"},content:{type:"string",maxLength:4096,description:"完整文件内容；源码优先一次性写入，不要把模块拆成多个 append_file。"},overwrite:{type:"boolean"},expected_sha256:{type:"string"}}},
    append_file:{type:"object",additionalProperties:false,required:["path","content"],properties:{path:{type:"string",description:"已有文件或明确的临时分片文件的 Run 工作区相对路径；必须先生成此字段。"},content:{type:"string",minLength:1,maxLength:4096,description:"追加内容；仅用于有意追加文本或临时分片，不用于拼接最终源码模块。"},expected_sha256:{type:"string"}}},
    edit_file:{type:"object",additionalProperties:false,required:["path","old_text","new_text"],properties:{path:{type:"string"},old_text:{type:"string",minLength:1,maxLength:2000},new_text:{type:"string",maxLength:2000},expected_sha256:{type:"string"}}},
    promote_file:{type:"object",additionalProperties:false,required:["source_path","target_path","expected_source_sha256"],properties:{source_path:{type:"string",description:"已完成的临时/分片文件路径"},target_path:{type:"string",description:"最终发布路径"},expected_source_sha256:{type:"string",description:"发布前读取到的临时文件 hash"},expected_target_sha256:{type:"string",description:"目标已存在时必须提供的当前 hash"}}},
    run_command:{type:"object",additionalProperties:false,required:["command","args"],properties:{command:{type:"string",enum:["python3"]},args:{type:"array",minItems:1,maxItems:32,items:{type:"string"},description:"直接执行工作区 Python 脚本，或使用 -m py_compile|compileall|unittest；禁止 -c 和标准输入。"},working_directory:{type:"string"},timeout_seconds:{type:"integer",minimum:1,maximum:30}}}
  };
  const descriptions:Record<string,string>={read_file:"读取当前 Run 工作区内的 UTF-8 文本文件，可按行截取。",list_files:"列出当前 Run 隔离工作区内容。",search_files:"在当前 Run 工作区 UTF-8 文本文件中进行字面量搜索。",write_file:"创建文件或在明确授权后原子覆盖；源码文件优先一次性完整写入，并返回文件 hash、行数和验证状态。",append_file:"原子追加文本或临时分片；不要用它拼接最终源码模块，并返回文件 hash、行数和验证状态。",edit_file:"对工作区文本文件进行唯一匹配替换；验证失败时优先精确修复。",promote_file:"校验临时文件 hash 后，将完整文件原子发布到最终路径；目标已存在时必须提供目标 hash。",run_command:"在当前 Run 的隔离 Sandbox 中直接执行 Python 脚本或受控模块，返回 stdout、stderr、退出码和超时；禁止 shell、-c、标准输入和公网访问。"};
  return {key,name,spec:{definition:{name:operation,version:"1",description:descriptions[operation],input_schema:schemas[operation],risk:workspaceRisk(operation),execution_mode:workspaceMutating(operation)?"serial":"parallel"},provider_type:"workspace",workspace:{operation,max_bytes:Math.min(1048576,Math.max(1024,maxBytes))}}};
}
function workspaceMutating(operation:string){return operation==="write_file"||operation==="append_file"||operation==="edit_file"||operation==="promote_file"||operation==="run_command"}
function workspaceRisk(operation:string){return operation==="run_command"?"HIGH_RISK":workspaceMutating(operation)?"LOW_WRITE":"READ"}
function titleWords(value:string){return value.split("_").map(word=>word.charAt(0).toUpperCase()+word.slice(1)).join(" ")}
function lines(value:string){return value.split("\n").map(item=>item.trim()).filter(Boolean)}

function StudioSection({ title, detail, action, children }: { title: string; detail: string; action?: ReactNode; children: ReactNode }) { return <section className="agent-studio-section"><header className="agent-studio-section-head"><span><h1>{title}</h1><p>{detail}</p></span>{action}</header>{children}</section>; }
function ResourceSplit<T extends { id: string; key: string; name: string; version: number; status: string }>({ items, selectedID, onSelect, renderMeta, renderDetail }: { items: T[]; selectedID?: string; onSelect: (id: string) => void; renderMeta: (item: T) => string; renderDetail: (item: T) => ReactNode }) { const selected = items.find(item => item.id === selectedID) || items[0]; return <div className="agent-resource-split"><div className="agent-resource-list">{items.map(item => <button type="button" className={selected?.id === item.id ? "active" : ""} key={item.id} onClick={() => onSelect(item.id)}><span><b>{item.name}</b><small>{item.key} · {renderMeta(item)}</small></span><StatusBadge status={item.status}/></button>)}</div><article className="agent-resource-detail">{selected ? renderDetail(selected) : null}</article></div>; }
function ResourceIdentity({ item }: { item: { name: string; key: string; version: number; status: string; id: string; spec_hash?: string; content_hash?: string; created_at: string } }) { return <header className="agent-resource-identity"><span><h2>{item.name}</h2><p>{item.key} · v{item.version}</p></span><StatusBadge status={item.status}/><dl><div><dt>Version ID</dt><dd>{shortID(item.id)}</dd></div><div><dt>Digest</dt><dd>{shortID(item.spec_hash || item.content_hash || "")}</dd></div><div><dt>创建时间</dt><dd>{formatDate(item.created_at)}</dd></div></dl></header>; }
function distinctDefinitions<T extends { definition_id: string }>(items?: T[]) { return [...new Set((items || []).map(item => item.definition_id))]; }
function latestDefinitionVersions<T extends { definition_id: string; version: number }>(items: T[]): T[] { const latest = new Map<string, T>(); for (const item of items) { const current = latest.get(item.definition_id); if (!current || item.version > current.version) latest.set(item.definition_id, item); } return [...latest.values()]; }
function toggleList(items: string[], id: string) { return items.includes(id) ? items.filter(item => item !== id) : [...items, id]; }
function diffValues(before: unknown, after: unknown) {
  const left = flattenValue(before); const right = flattenValue(after);
  return [...new Set([...left.keys(), ...right.keys()])].sort().flatMap(path => {
    const previous = left.get(path); const current = right.get(path);
    return JSON.stringify(previous) === JSON.stringify(current) ? [] : [{ path, before: previous, after: current }];
  });
}
function flattenValue(value: unknown, path = "$", result = new Map<string, unknown>()) {
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b));
    if (!entries.length) result.set(path, value);
    else entries.forEach(([key, item]) => flattenValue(item, `${path}.${key}`, result));
  } else result.set(path, value);
  return result;
}
function formatDiffValue(value: unknown) {
  if (value === undefined) return "—";
  const formatted = typeof value === "string" ? value : JSON.stringify(value);
  return formatted.length > 320 ? `${formatted.slice(0, 317)}…` : formatted;
}
function shortID(value: string) { return value.length > 18 ? `${value.slice(0, 10)}…${value.slice(-6)}` : value || "—"; }
function formatDate(value?: string) { return value ? new Date(value).toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }) : "—"; }
