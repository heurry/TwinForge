import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { Activity, Bot, Brain, Check, ChevronDown, ChevronRight, CircleStop, Clock3, Coins, Copy, Download, FileDiff, GitBranch, History, Layers3, ListChecks, MessageSquarePlus, Plus, RefreshCw, Search, Send, ShieldCheck, Square, Trash2, Wrench } from "lucide-react";
import { toast } from "sonner";

import { EmptyState, ErrorState, Skeleton } from "../components/common/FeedbackStates";
import { StatusBadge } from "../components/common/PlatformPrimitives";
import { ApiError, answerAgentRunQuestion, cancelAgentRun, createAgentMemory, createAgentRun, createAgentSession, decideAgentApproval, deleteAgentMemory, getAgentArtifactContent, getAgentObservabilitySummary, getAgentPlatformCapabilities, getAgentRunEvents, getAgentRunManifest, getAgentRunPlan, getAgentRunQuestion, getAgentRunTrajectory, getAgentSessionAudit, getAgentStorageSummary, listAgentApprovals, listAgentMemories, listAgentMemoryLifecycleEvents, listAgentMemoryTimeline, listAgentRunArtifacts, listAgentRunChildren, listAgentRunScores, listAgentRunVerifications, listAgentRuns, listAgentSessionRuns, listAgentWorkflowRuns, listAgentSessions, listAgentWorkflows, listExecutableAgentVersions, promoteAgentArtifact } from "../lib/api";
import { executionModeLabel, planningPolicyLabel } from "../lib/agentExecutionMode";
import { relativeTime } from "../lib/format";
import { MarkdownText } from "../lib/markdown";
import { buildAgentActivity, visibleModelText } from "../lib/agentActivity";
import type { AgentActivityItem } from "../lib/agentActivity";
import type { AgentArtifact, AgentEvent, AgentExecutableVersion, AgentMemory, AgentMemoryTimelineEntry, AgentMessage, AgentPlatformCapabilities, AgentRun, AgentScore, AgentSession, AgentSessionAudit, AgentStorageSummary, AgentTaskPlan, AgentTrajectoryRecord, AgentUserQuestion, AgentVerificationRecord, AgentWorkflow } from "../types/agent";

export function AgentOperationsPage() {
  const client = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const [selected, setSelected] = useState<string>();
  const [selectedRunSnapshot, setSelectedRunSnapshot] = useState<AgentRun>();
  const [versionID, setVersionID] = useState("");
  const [sessionID, setSessionID] = useState<string>();
  const [workflowIntent, setWorkflowIntent] = useState<"auto" | "new_workflow" | "resume" | "new_turn">("auto");
  const [routingCandidates, setRoutingCandidates] = useState<AgentWorkflow[]>([]);
  const [question, setQuestion] = useState("");
  const [questionAnswer, setQuestionAnswer] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [navigationView, setNavigationView] = useState<"sessions" | "runs">("sessions");
  const [workspaceView, setWorkspaceView] = useState<"chat" | "trace">("chat");
  const [inspectorView, setInspectorView] = useState<"trace" | "memory" | "audit" | "runtime">("trace");
  const [selectedEventSequence, setSelectedEventSequence] = useState<number>();
  const [navigationFilter, setNavigationFilter] = useState("");
  const [copiedRunID, setCopiedRunID] = useState<string>();
  const [stopRequestedRunIDs, setStopRequestedRunIDs] = useState<Set<string>>(new Set());
  const conversationRef = useRef<HTMLDivElement>(null);
  const versionSelectionPinned = useRef(false);
  const summary = useQuery({ queryKey: ["agent", "summary"], queryFn: getAgentObservabilitySummary, retry: false, refetchInterval: 5000 });
  const capabilities = useQuery({ queryKey: ["agent", "capabilities"], queryFn: getAgentPlatformCapabilities, retry: false });
  const storage = useQuery({ queryKey: ["agent", "storage"], queryFn: getAgentStorageSummary, retry: false, refetchInterval: 5000 });
  const versions = useQuery({ queryKey: ["agent", "versions"], queryFn: listExecutableAgentVersions, retry: false, refetchInterval: 5000 });
  const runs = useQuery({ queryKey: ["agent", "runs"], queryFn: () => listAgentRuns(50), retry: false, refetchInterval: 3000 });
  useEffect(() => {
    const latest = versions.data?.data?.[0];
    if (latest && (!versionID || (!versionSelectionPinned.current && versionID !== latest.id))) setVersionID(latest.id);
  }, [versions.data, versionID]);
  const selectedVersion = versions.data?.data?.find((version) => version.id === versionID);
  const sessions = useQuery({ queryKey: ["agent", "sessions", selectedVersion?.agent_id], queryFn: () => listAgentSessions(selectedVersion!.agent_id), enabled: Boolean(selectedVersion?.agent_id), retry: false, refetchInterval: 5000 });
  useEffect(() => {
    const list = sessions.data?.data;
    if (!list) return;
    if (sessionID) return;
    setSessionID(list[0]?.id);
  }, [sessions.data, sessionID]);
  const sessionRuns = useQuery({ queryKey: ["agent", "session-runs", sessionID], queryFn: () => listAgentSessionRuns(sessionID!), enabled: Boolean(sessionID), retry: false, refetchInterval: 2500 });
  const sessionAudit = useQuery({ queryKey: ["agent", "session-audit", sessionID], queryFn: () => getAgentSessionAudit(sessionID!), enabled: Boolean(sessionID), retry: false, refetchInterval: 5000 });
  const workflows = useQuery({ queryKey: ["agent", "workflows", sessionID], queryFn: () => listAgentWorkflows(sessionID), enabled: Boolean(sessionID), retry: false, refetchInterval: 3000 });
  const selectedSession = sessions.data?.data?.find((session) => session.id === sessionID);
  useEffect(() => {
    const list = sessionRuns.data?.data;
    if (list?.length && !list.some((run) => run.id === selected)) setSelected(list[list.length - 1].id);
  }, [sessionRuns.data, selected]);
  const events = useQuery({ queryKey: ["agent", "events", selected], queryFn: () => getAgentRunEvents(selected!), enabled: Boolean(selected), retry: false, refetchInterval: 2000 });
  const plan = useQuery({ queryKey: ["agent", "plan", selected], queryFn: () => getAgentRunPlan(selected!), enabled: Boolean(selected), retry: false, refetchInterval: 2000 });
  const verifications = useQuery({ queryKey: ["agent", "verifications", selected], queryFn: () => listAgentRunVerifications(selected!), enabled: Boolean(selected), retry: false, refetchInterval: 2000 });
  const pendingQuestion = useQuery({ queryKey: ["agent", "question", selected], queryFn: () => getAgentRunQuestion(selected!), enabled: Boolean(selected), retry: false, refetchInterval: 2000 });
  const pendingQuestionID = pendingQuestion.data?.data?.id;
  useEffect(() => { setQuestionAnswer(""); }, [pendingQuestionID]);
  const selectedRun = sessionRuns.data?.data?.find((run) => run.id === selected) || runs.data?.data?.find((run) => run.id === selected) || (selectedRunSnapshot?.id === selected ? selectedRunSnapshot : undefined);
  const selectedRunVersion = versions.data?.data?.find((version) => version.id === selectedRun?.agent_version_id);
  const delegationRecoveryVersion = selectedRunVersion ? (versions.data?.data || []).find((version) => version.agent_id === selectedRunVersion.agent_id && version.id !== selectedRunVersion.id && Boolean(version.spec.collaboration?.allowed_targets?.length)) : undefined;
  const traceGroups = buildTraceGroups(events.data?.data || []);
  const selectedEvents = events.data?.data || [];
  const delegationBlocked = Boolean(selectedRun?.status === "failed" && (selectedRun.error_message?.includes("delegation target or mode is not allowed") || selectedEvents.some(event => event.type === "TOOL_FAILED" && String(event.payload?.error_code || "") === "DELEGATION_TARGET_NOT_ALLOWED")));
  const latestEventSequence = selectedEvents.length ? selectedEvents[selectedEvents.length - 1].sequence : 0;
  useEffect(() => {
    const element = conversationRef.current;
    if (!element || workspaceView !== "chat") return;
    const distance = element.scrollHeight - element.scrollTop - element.clientHeight;
    if (distance > 180) return;
    window.requestAnimationFrame(() => { element.scrollTop = element.scrollHeight; });
  }, [latestEventSequence, workspaceView]);
  useEffect(() => {
    const element = conversationRef.current;
    if (!element || workspaceView !== "chat") return;
    window.requestAnimationFrame(() => { element.scrollTop = element.scrollHeight; });
  }, [selected, sessionID, workspaceView]);
  useEffect(() => {
    const requestedRunID = searchParams.get("run");
    if (!requestedRunID || !runs.data?.data?.length) return;
    const requestedRun = runs.data.data.find(run => run.id === requestedRunID);
    if (!requestedRun) return;
    const publishedVersion = versions.data?.data?.find(version => version.id === requestedRun.agent_version_id);
    if (publishedVersion) { versionSelectionPinned.current = true; setVersionID(publishedVersion.id); }
    if (requestedRun.session_id) setSessionID(requestedRun.session_id);
    setSelected(requestedRun.id);
    setSelectedRunSnapshot(requestedRun);
    setNavigationView("runs");
    setWorkspaceView("trace");
    setInspectorView("trace");
    setSelectedEventSequence(undefined);
    setSearchParams({}, { replace: true });
  }, [runs.data, searchParams, setSearchParams, versions.data]);
  const send = useMutation({
    mutationFn: async () => {
      let activeSessionID = sessionID;
      if (!activeSessionID) {
        const created = await createAgentSession(selectedVersion!.agent_id, sessionTitle(question));
        activeSessionID = created.data.id;
      }
      const workflowID = workflowIntent === "new_workflow" ? undefined : selectedRun?.workflow_id;
      const createdRun = await createAgentRun(versionID, { question: question.trim() }, activeSessionID, "web", workflowID, workflowIntent === "new_workflow", workflowIntent);
      return { run: createdRun.data, sessionID: activeSessionID };
    },
    onSuccess: async ({ run, sessionID: activeSessionID }) => { setSessionID(activeSessionID); setSelected(run.id); setSelectedRunSnapshot(run); setQuestion(""); setWorkflowIntent("auto"); setRoutingCandidates([]); setExpanded(new Set()); await client.invalidateQueries({ queryKey: ["agent"] }); },
    onError: (error) => {
      if (error instanceof ApiError) {
        try {
          const body = JSON.parse(error.body) as { error?: { code?: string }; data?: { candidates?: AgentWorkflow[] } };
          if (body.error?.code === "workflow_ambiguous" && Array.isArray(body.data?.candidates)) {
            setRoutingCandidates(body.data.candidates);
            toast.info("当前 Session 有多个未完成任务，请选择要继续的任务");
            return;
          }
        } catch { /* fall through to the generic error surface */ }
      }
      toast.error("创建 Agent Run 失败", { description: error instanceof Error ? error.message : "请求未被接受" });
    }
  });
  const recoverDelegation = useMutation({
    mutationFn: async ({ run, version }: { run: AgentRun; version: AgentExecutableVersion }) => {
      if (!run.session_id || !run.workflow_id) throw new Error("当前 Run 缺少 Session 或 Workflow，无法续跑");
      return createAgentRun(version.id, {
        question: "恢复当前任务：原 Run 因委派策略版本不兼容而停止。请沿用已有 Plan、Checkpoint 和 Workspace，使用可委派目标的精确 AgentVersion ID，并先采用 sync 模式继续执行。不要重新创建任务。"
      }, run.session_id, "web", run.workflow_id, false, "resume");
    },
    onSuccess: async ({ data: run }) => {
      versionSelectionPinned.current = true;
      setVersionID(run.agent_version_id);
      setSessionID(run.session_id);
      setSelected(run.id);
      setSelectedRunSnapshot(run);
      setWorkspaceView("chat");
      toast.success("已切换到兼容 AgentVersion 并继续", { description: "原 Workflow、Plan、Checkpoint 和 Workspace 已保留。" });
      await client.invalidateQueries({ queryKey: ["agent"] });
    },
    onError: error => toast.error("恢复委派任务失败", { description: error instanceof Error ? error.message : "续跑请求未被接受" })
  });
  const newSession = useMutation({
    mutationFn: () => createAgentSession(selectedVersion!.agent_id, `新对话 · ${new Date().toLocaleString()}`),
    onSuccess: async ({ data }) => {
      setNavigationView("sessions");
      setWorkspaceView("chat");
      setNavigationFilter("");
      setSelected(undefined);
      setSelectedRunSnapshot(undefined);
      setWorkflowIntent("auto");
      setSessionID(data.id);
      setExpanded(new Set());
      setInspectorView("trace");
      await client.invalidateQueries({ queryKey: ["agent", "sessions", data.agent_id] });
    }
  });
  const answerQuestion = useMutation({
    mutationFn: (answer: string) => answerAgentRunQuestion(pendingQuestion.data!.data!.id, answer),
    onSuccess: async () => {
      setQuestionAnswer("");
      await Promise.all([
        client.invalidateQueries({ queryKey: ["agent", "question", selected] }),
        client.invalidateQueries({ queryKey: ["agent", "events", selected] }),
        client.invalidateQueries({ queryKey: ["agent", "runs"] }),
        client.invalidateQueries({ queryKey: ["agent", "session-runs", sessionID] })
      ]);
    }
  });
  const cancelRun = useMutation({
    mutationFn: (runID: string) => cancelAgentRun(runID),
    onMutate: (runID) => setStopRequestedRunIDs((current) => new Set(current).add(runID)),
    onSuccess: async (_, runID) => {
      toast.success("已请求停止 Agent", { description: `Run ${shortID(runID)} 正在安全结束当前模型或工具调用。` });
      await client.invalidateQueries({ queryKey: ["agent"] });
    },
    onError: (error, runID) => {
      setStopRequestedRunIDs((current) => { const next = new Set(current); next.delete(runID); return next; });
      toast.error("停止 Agent 失败", { description: error instanceof Error ? error.message : "取消请求未被接受" });
    }
  });
  useEffect(() => {
    const visibleRuns = [...(sessionRuns.data?.data || []), ...(runs.data?.data || [])];
    setStopRequestedRunIDs((current) => {
      const next = new Set(current);
      for (const run of visibleRuns) if (isTerminalRun(run)) next.delete(run.id);
      return next.size === current.size ? current : next;
    });
  }, [runs.data, sessionRuns.data]);
  const data = summary.data?.data;
  const refresh = () => { summary.refetch(); capabilities.refetch(); storage.refetch(); versions.refetch(); sessions.refetch(); sessionRuns.refetch(); runs.refetch(); events.refetch(); plan.refetch(); pendingQuestion.refetch(); };
  const submit = () => { if (versionID && question.trim() && !send.isPending) send.mutate(); };
  const sessionRunList = sessionRuns.data?.data || [];
  const activeSessionRun = [...sessionRunList].reverse().find((run) => !isTerminalRun(run));
  const stoppingRun = (run: AgentRun) => Boolean(run.cancel_requested_at || stopRequestedRunIDs.has(run.id));
  const stopRun = (run: AgentRun) => { if (!stoppingRun(run) && !cancelRun.isPending) cancelRun.mutate(run.id); };
  const sessionItems = (sessions.data?.data || []).filter((session) => sessionLabel(session).toLowerCase().includes(navigationFilter.trim().toLowerCase()));
  const runItems = (runs.data?.data || []).filter((run) => {
    const version = versions.data?.data?.find((item) => item.id === run.agent_version_id);
    return runTitle(run, version).toLowerCase().includes(navigationFilter.trim().toLowerCase());
  });
  const selectRun = (run: AgentRun) => {
    if (run.agent_version_id !== versionID && versions.data?.data?.some(version => version.id === run.agent_version_id)) {
      versionSelectionPinned.current = true;
      setVersionID(run.agent_version_id);
    }
    if (run.session_id) setSessionID(run.session_id);
    setSelected(run.id);
    setSelectedRunSnapshot(run);
    setExpanded(new Set());
    setWorkspaceView("trace");
    setInspectorView("trace");
    setSelectedEventSequence(undefined);
    setWorkflowIntent("auto");
  };
  const selectWorkflow = (workflowID: string) => {
    const run = sessionRuns.data?.data?.find(item => item.workflow_id === workflowID) || runs.data?.data?.find(item => item.workflow_id === workflowID);
    if (run) { selectRun(run); setWorkflowIntent("resume"); return; }
    setWorkflowIntent("resume");
  };
  const startNewWorkflow = () => { setWorkflowIntent("new_workflow"); setWorkspaceView("chat"); setQuestion(""); };
  const resumeSelectedWorkflow = () => { if (selectedRun?.workflow_id) { setWorkflowIntent("resume"); setWorkspaceView("chat"); } };
  const appendToSelectedWorkflow = () => { if (selectedRun?.workflow_id) { setWorkflowIntent("new_turn"); setWorkspaceView("chat"); } };
  const copyRunOutput = async (run: AgentRun) => {
    const value = run.output ? assistantDisplayText(outputText(run.output)) : inputText(run);
    await copyToClipboard(value);
    setCopiedRunID(run.id);
    window.setTimeout(() => setCopiedRunID((current) => current === run.id ? undefined : current), 1600);
  };

  return <section className="infra-page agent-ops-page">
    <header className="agent-workbench-header">
      <span className="agent-workbench-title"><i><Bot size={18}/></i><span><strong>Agent Workspace</strong><small>对话、运行轨迹与上下文治理</small></span></span>
      <label className="agent-version-picker"><small>Agent / Version</small><select value={versionID} onChange={(event)=>{versionSelectionPinned.current=true;setVersionID(event.target.value);setSessionID(undefined);setSelected(undefined);setSelectedRunSnapshot(undefined)}} disabled={versions.isLoading}>{versions.data?.data?.map(version=><option value={version.id} key={version.id}>{version.agent_name} · v{version.version}</option>)}</select></label>
      {selectedVersion?<div className="agent-model-binding"><i/><span><small>{modelPolicyLabel(selectedVersion)}</small><strong>{configuredModelLabel(selectedVersion)}</strong></span></div>:<span className="agent-version-empty">暂无已发布 AgentVersion</span>}
      <button className="agent-refresh-button" type="button" onClick={refresh}><RefreshCw size={15}/><span>刷新</span></button>
    </header>

    <div className="agent-workbench-layout">
      <aside className="agent-navigation-panel">
        <div className="agent-navigation-head"><span><Layers3 size={16}/><strong>工作记录</strong></span><div className="agent-navigation-actions"><button type="button" title="创建新 Workflow（保留当前 Session）" disabled={!selectedVersion} onClick={startNewWorkflow}><Plus size={15}/><span>新任务</span></button><button type="button" title="创建新 Session" disabled={!selectedVersion||newSession.isPending} onClick={()=>newSession.mutate()}><MessageSquarePlus size={16}/><span>新对话</span></button></div></div>
        <div className="agent-navigation-tabs"><button type="button" className={navigationView==="sessions"?"active":""} onClick={()=>setNavigationView("sessions")}>Sessions <em>{sessions.data?.data?.length ?? 0}</em></button><button type="button" className={navigationView==="runs"?"active":""} onClick={()=>setNavigationView("runs")}>Runs <em>{runs.data?.data?.length ?? 0}</em></button></div>
        <label className="agent-navigation-search"><Search size={14}/><input value={navigationFilter} onChange={(event)=>setNavigationFilter(event.target.value)} placeholder={navigationView==="sessions"?"搜索对话":"搜索运行记录"}/></label>
        <div className="agent-navigation-scroll">
          {navigationView==="sessions"?(sessions.isLoading?<Skeleton rows={5}/>:sessions.isError?<ErrorState title="无法读取 Session" error={sessions.error} onRetry={()=>sessions.refetch()}/>:!sessionItems.length?<EmptyState title={navigationFilter?"没有匹配的 Session":"暂无 Session"} description={navigationFilter?"尝试其他关键词":"发送第一条消息时会自动创建。"}/>:sessionItems.map(session=><SessionButton key={session.id} session={session} active={session.id===sessionID} onClick={()=>{setSessionID(session.id);setSelected(undefined);setSelectedRunSnapshot(undefined);setExpanded(new Set());setWorkspaceView("chat");setInspectorView("trace");setSelectedEventSequence(undefined)}}/>)):(runs.isLoading?<Skeleton rows={6}/>:runs.isError?<ErrorState title="无法读取 Run" error={runs.error} onRetry={()=>runs.refetch()}/>:!runItems.length?<EmptyState title={navigationFilter?"没有匹配的 Run":"暂无 Agent Run"}/>:runItems.map(run=>{const version=versions.data?.data?.find(item=>item.id===run.agent_version_id);return <RunButton key={run.id} run={run} version={version} active={selected===run.id} onClick={()=>selectRun(run)}/> }))}
        </div>
        <footer className="agent-navigation-foot"><span className={summary.isError?"offline":"online"}/><span><strong>{summary.isError?"Runtime 未连接":"Runtime 已连接"}</strong><small>{data?.active_runs ?? 0} 个 Run 执行中</small></span></footer>
      </aside>

      <section className={`agent-conversation-panel ${workspaceView === "trace" ? "trace-mode" : ""}`}>
        <header className="agent-conversation-head"><span><strong>{selectedSession?sessionLabel(selectedSession):"新对话"}</strong><small>{selectedVersion?`${selectedVersion.agent_name} · v${selectedVersion.version} · ${selectedVersion.spec.description||"已发布版本"}`:"请选择可执行版本"}</small></span><nav className="agent-workspace-tabs"><button type="button" className={workspaceView==="chat"?"active":""} onClick={()=>setWorkspaceView("chat")}>对话</button><button type="button" className={workspaceView==="trace"?"active":""} disabled={!selectedRun} onClick={()=>setWorkspaceView("trace")}>轨迹</button></nav><div className="agent-workflow-actions"><select value={selectedRun?.workflow_id||""} onChange={event=>selectWorkflow(event.target.value)} disabled={!workflows.data?.data?.length} aria-label="选择 Workflow"><option value="">选择任务</option>{workflows.data?.data?.map(item=><option value={item.workflow_id} key={item.workflow_id}>{workflowLabel(item)}</option>)}</select><button type="button" disabled={!selectedRun?.workflow_id} onClick={resumeSelectedWorkflow}>继续任务</button><button type="button" disabled={!selectedRun?.workflow_id} onClick={appendToSelectedWorkflow}>追加要求</button>{selectedSession?<><span className="agent-session-state"><i/>上下文持续</span><code>{shortID(selectedSession.id)}</code></>:null}</div></header>
        {workspaceView === "chat" ? <><RunActivityRail runID={selectedRun?.id} plan={plan.data?.data} verifications={verifications.data?.data||[]} question={pendingQuestion.data?.data} answer={questionAnswer} answering={answerQuestion.isPending} error={answerQuestion.error} onAnswerChange={setQuestionAnswer} onAnswer={(value)=>answerQuestion.mutate(value)} onEvidenceSelect={(callID)=>{const evidenceEvent=(events.data?.data||[]).find((event)=>event.call_id===callID&&event.type==="TOOL_COMPLETED");if(!evidenceEvent)return;setWorkspaceView("trace");setInspectorView("trace");setSelectedEventSequence(evidenceEvent.sequence);}} /><div className="agent-conversation" ref={conversationRef}>
          {sessionRuns.isLoading ? <Skeleton rows={4}/> : sessionRuns.isError ? <ErrorState title="无法读取对话" error={sessionRuns.error} onRetry={()=>sessionRuns.refetch()}/> : sessionRuns.data?.data?.length ? <div className="agent-message-column">{sessionRuns.data.data.map(run=>{const runEvents:AgentEvent[]=selected===run.id?(events.data?.data||[]):[];const showActivity=selected===run.id&&runEvents.length>0;const canRecover=selected===run.id&&delegationBlocked&&delegationRecoveryVersion;return <article className={`agent-session-turn ${selected===run.id?"active":""}`} key={run.id}><ChatBubble role="user" content={inputText(run)}/>{showActivity?<AgentExecutionFeed events={runEvents} terminal={isTerminalRun(run)} stopping={stoppingRun(run)} onOpenTrace={()=>selectRun(run)}/>:null}{run.output ? <AssistantRunReply run={run}/> : run.status === "failed" ? <><ChatBubble role="error" content={run.error_message || "Agent 执行失败"}/>{canRecover?<div className="agent-routing-choice"><strong>当前 AgentVersion 不允许委派子 Agent</strong><small>保留原 Workflow、Plan、Checkpoint 和 Workspace，切换到包含委派白名单的 v{delegationRecoveryVersion.version} 后继续。</small><button type="button" disabled={recoverDelegation.isPending} onClick={()=>recoverDelegation.mutate({run,version:delegationRecoveryVersion})}>{recoverDelegation.isPending?"切换并恢复中…":"切换兼容版本并继续"}</button></div>:null}</> : run.status === "cancelled" ? <ChatBubble role="pending" content="Agent 任务已停止。"/> : !showActivity?<ChatBubble role="pending" content={stoppingRun(run)?"正在停止 Agent，当前模型或工具调用即将结束…":`Agent ${statusLabel(run.status)}，正在等待首个执行事件…`}/>:null}<footer><span className="agent-turn-meta"><StatusBadge status={stoppingRun(run)&&!isTerminalRun(run)?"停止中":statusLabel(run.status)}/><span>{executionModeLabel(run,runEvents)}</span><span>{run.model_resolution?.model_id||"模型待解析"}</span><span>{durationText(run)}</span><span>{relativeTime(run.created_at)}</span></span><span className="agent-turn-actions">{!isTerminalRun(run)?<button className="agent-turn-stop" type="button" disabled={stoppingRun(run)||cancelRun.isPending} onClick={()=>stopRun(run)}><Square size={11} fill="currentColor"/>{stoppingRun(run)?"停止中":"停止"}</button>:null}<button type="button" onClick={()=>selectRun(run)}><Activity size={12}/>查看轨迹</button><button type="button" onClick={()=>void copyRunOutput(run)}>{copiedRunID===run.id?<><Check size={12}/>已复制</>:<><Copy size={12}/>{run.output?"复制回复":"复制问题"}</>}</button></span></footer></article>})}</div> : <EmptyState title="开始一次 Agent 对话" description="Session 会保留完整上下文；每次提问仍生成独立、可审计的 Run。" icon={Bot}/>}
        </div>
        <div className="agent-composer-shell">
          {routingCandidates.length ? <div className="agent-routing-choice"><strong>请选择要继续的任务</strong><small>平台不会替你猜测 Workflow；选择后保留当前输入，再点击发送。</small><div>{routingCandidates.map(candidate=><button type="button" key={candidate.workflow_id} onClick={()=>selectWorkflow(candidate.workflow_id)}><b>{candidate.goal||"未命名任务"}</b><span>{workflowStatusLabel(candidate.status)} · {shortID(candidate.workflow_id)}</span></button>)}</div></div>:null}
          {send.isError ? <div className="agent-send-error">{send.error instanceof Error ? send.error.message : "创建 Run 失败"}</div>:null}
          <div className={`agent-composer ${activeSessionRun?"running":""}`}><textarea value={question} onChange={(event)=>setQuestion(event.target.value)} onKeyDown={(event)=>{if(event.key==="Enter"&&!event.shiftKey&&!activeSessionRun){event.preventDefault();submit();}}} placeholder={activeSessionRun?"Agent 正在执行；停止后可继续发送消息…":"向 Agent 提问，或描述需要完成的任务…"}/><div className="agent-composer-actions"><span>{activeSessionRun?(stoppingRun(activeSessionRun)?"正在停止，等待当前调用安全退出…":"Agent 正在自主执行，点击方形按钮可停止"):<><kbd>Enter</kbd> 发送 · <kbd>Shift Enter</kbd> 换行</>}</span>{activeSessionRun?<button className={`agent-stop-button ${stoppingRun(activeSessionRun)?"pending":""}`} type="button" title={stoppingRun(activeSessionRun)?"正在停止 Agent":"停止 Agent 任务"} disabled={stoppingRun(activeSessionRun)||cancelRun.isPending} onClick={()=>stopRun(activeSessionRun)}>{stoppingRun(activeSessionRun)?<RefreshCw className="agent-spin" size={15}/>:<Square size={13} fill="currentColor"/>}</button>:<button type="button" title="发送消息" disabled={!versionID||!question.trim()||send.isPending} onClick={submit}>{send.isPending?<RefreshCw className="agent-spin" size={16}/>:<Send size={16}/>}</button>}</div></div>
          <small className="agent-composer-note">{workflowIntent==="new_workflow"?"下一条消息将创建当前 Session 下的新任务。":workflowIntent==="resume"?"下一条消息将复用当前任务的 Workflow 和 Checkpoint。":workflowIntent==="new_turn"?"下一条消息将追加到当前 Workflow，保留已有计划与记忆。":activeSessionRun?"停止会取消当前模型请求、工具调用及其尚未完成的子 Agent。":"Agent 可能调用工具并写入运行记录，请核验重要结果。"}</small>
        </div></> : <TraceWorkspace run={selectedRun} version={selectedRunVersion} events={events.data?.data || []} groups={traceGroups} expanded={expanded} loading={events.isLoading} error={events.error} selectedSequence={selectedEventSequence} onSelect={setSelectedEventSequence} onToggle={(key)=>setExpanded(toggle(expanded,key))} onRetry={()=>events.refetch()} onOpenRun={selectRun} onCancel={(runID)=>{const run=selectedRun;if(run&&run.id===runID)stopRun(run)}} cancelling={Boolean(selectedRun&&(cancelRun.isPending||stoppingRun(selectedRun)))}/>}
      </section>

      <aside className="agent-inspector-panel">
        <header className="agent-inspector-tabs"><button type="button" className={inspectorView==="trace"?"active":""} onClick={()=>setInspectorView("trace")}><Activity size={14}/>运行轨迹</button><button type="button" className={inspectorView==="memory"?"active":""} onClick={()=>setInspectorView("memory")}><Brain size={14}/>记忆</button><button type="button" className={inspectorView==="audit"?"active":""} onClick={()=>setInspectorView("audit")}><Coins size={14}/>审计</button><button type="button" className={inspectorView==="runtime"?"active":""} onClick={()=>setInspectorView("runtime")}><ShieldCheck size={14}/>Runtime</button></header>
        <div className="agent-inspector-scroll">
          {inspectorView==="trace"?<TraceInspector run={selectedRun} version={selectedRunVersion} events={events.data?.data || []} selectedSequence={selectedEventSequence}/>:null}
          {inspectorView==="memory"?(selectedVersion?<MemoryPanel version={selectedVersion} session={selectedSession}/>:<EmptyState title="请选择 Agent 版本"/>):null}
          {inspectorView==="audit"?<SessionAuditPanel value={sessionAudit.data?.data} loading={sessionAudit.isLoading} error={sessionAudit.error} onRetry={()=>sessionAudit.refetch()}/>:null}
          {inspectorView==="runtime"?<div className="agent-runtime-view">{summary.isLoading?<Skeleton rows={3}/>:summary.isError?<ErrorState title="Agent Runtime 未连接" error={summary.error} onRetry={refresh}/>:<><div className="agent-kpis"><Kpi icon={Bot} label="Run" value={data?.total_runs ?? 0} detail={`${data?.active_runs ?? 0} 执行中 · ${data?.failed_runs ?? 0} 失败`}/><Kpi icon={Clock3} label="模型平均延迟" value={`${Math.round(data?.average_model_latency_ms ?? 0)} ms`} detail={`${data?.model_calls ?? 0} 次调用`}/><Kpi icon={Coins} label="Token" value={(data?.input_tokens ?? 0)+(data?.output_tokens ?? 0)} detail={`输入 ${data?.input_tokens ?? 0} · 输出 ${data?.output_tokens ?? 0}`}/><Kpi icon={Wrench} label="Tool" value={data?.tool_calls ?? 0} detail={`${data?.failed_tool_calls ?? 0} 次失败`}/></div><div className="agent-runtime-diagnostics"><span>协议错误 <b>{data?.model_protocol_failures ?? 0}</b></span><span>验收失败 <b>{data?.verification_failures ?? 0}</b></span><span>Plan 错误 <b>{data?.plan_failures ?? 0}</b></span><span>上下文压缩 <b>{data?.compaction_count ?? 0}</b></span><span>压缩 Token <b>{data?.compaction_before_tokens ?? 0} → {data?.compaction_after_tokens ?? 0}</b></span></div>{Object.entries(data?.tool_failures_by_code||{}).length?<div className="agent-runtime-error-clusters"><strong>Tool 错误聚类</strong>{Object.entries(data?.tool_failures_by_code||{}).slice(0,6).map(([code,count])=><span key={code}><code>{code}</code><b>{count}</b></span>)}</div>:null}</>}<StoragePanel value={storage.data?.data} loading={storage.isLoading} error={storage.error}/><CapabilityPanel value={capabilities.data?.data} loading={capabilities.isLoading}/></div>:null}
        </div>
      </aside>
    </div>
  </section>;
}

function RunActivityRail({runID,plan,verifications,question,answer,answering,error,onAnswerChange,onAnswer,onEvidenceSelect}:{runID?:string;plan?:AgentTaskPlan|null;verifications:AgentVerificationRecord[];question?:AgentUserQuestion|null;answer:string;answering:boolean;error:unknown;onAnswerChange:(value:string)=>void;onAnswer:(value:string)=>void;onEvidenceSelect:(callID:string)=>void}) {
  const client=useQueryClient();
  const [diffPreview,setDiffPreview]=useState<{approvalID:string;name:string;content?:string;error?:string}>();
  const approvals=useQuery({queryKey:["agent","approvals","pending"],queryFn:()=>listAgentApprovals("pending"),enabled:Boolean(runID),retry:false,refetchInterval:2000});
  const runApprovals=(approvals.data?.data||[]).filter(item=>item.run_id===runID);
  const decision=useMutation({
    mutationFn:({id,approved,reason}:{id:string;approved:boolean;reason:string})=>decideAgentApproval(id,approved,reason),
    onSuccess:async(_,variables)=>{
      setDiffPreview(current=>current?.approvalID===variables.id?undefined:current);
      toast.success(variables.approved?"已批准，Agent 将继续执行":"已拒绝本次工具调用");
      await client.invalidateQueries({queryKey:["agent"]});
    },
    onError:error=>toast.error("审批操作失败",{description:error instanceof Error?error.message:"审批请求未被接受"})
  });
  const previewDiff=async(approvalID:string,artifactID:string,name:string)=>{
    setDiffPreview({approvalID,name});
    try{const content=await getAgentArtifactContent(artifactID);setDiffPreview({approvalID,name,content:await content.text()});}
    catch(previewError){setDiffPreview({approvalID,name,error:previewError instanceof Error?previewError.message:"无法读取拟议 Diff"});}
  };
  if (!plan && !question && !runApprovals.length) return null;
  const completed = plan?.steps.filter((step) => step.status === "completed").length || 0;
  const submit = () => { if (answer.trim() && !answering) onAnswer(answer.trim()); };
  return <section className="agent-run-activity">
    {runApprovals.map(item=>{const dependency=item.tool_name==="install_dependency"?dependencyApproval(item.request):undefined;const projectRead=item.tool_name==="read_project_file"?projectFileApproval(item.request):undefined;return <section className="agent-approval-bar pending" key={item.id}>
      <ShieldCheck size={18}/><span><strong>{dependency?"Agent 请求安装 Run 级依赖":projectRead?"Agent 请求读取项目文件":`Agent 请求工具审批 · ${item.tool_name}`}</strong><small>T{item.turn} · S{item.step} · {item.risk} · 仅批准当前这一次调用</small>{dependency?<code>{dependency.packages}<br/>来源：{dependency.source} · 作用域：{dependency.scope}<br/>用途：{dependency.reason}</code>:projectRead?<code>文件：{projectRead.path}<br/>范围：{projectRead.range}<br/>用途：{projectRead.reason}<br/>权限：只读 · 单次调用 · 结果固化为输入快照</code>:<code>{pretty(item.request)}</code>}{diffPreview?.approvalID===item.id?<div className="agent-approval-diff"><header><strong>{diffPreview.name}</strong><button type="button" onClick={()=>setDiffPreview(undefined)}>关闭</button></header>{diffPreview.error?<p>{diffPreview.error}</p>:diffPreview.content!==undefined?<pre>{diffPreview.content}</pre>:<small>正在读取 Diff…</small>}</div>:null}</span><div>{item.diff_artifact_id?<button type="button" disabled={decision.isPending} onClick={()=>void previewDiff(item.id,item.diff_artifact_id!,`${item.tool_name} · proposed.diff`)}>查看拟议 Diff</button>:null}<button type="button" disabled={decision.isPending} onClick={()=>decision.mutate({id:item.id,approved:false,reason:"Rejected from Agent conversation"})}>拒绝</button><button className="approve" type="button" disabled={decision.isPending} onClick={()=>decision.mutate({id:item.id,approved:true,reason:"Approved from Agent conversation"})}>{decision.isPending?"处理中…":"批准并继续"}</button></div>
    </section>})}
    {plan ? <details className="agent-live-plan" open>
      <summary><span><ListChecks size={15}/><b>执行计划</b><em>r{plan.revision}</em></span><small>{completed}/{plan.steps.length} 已完成 · {planVerificationOutcomeLabel(plan.verification_outcome)}</small></summary>
      <div className="agent-live-plan-body"><header><strong>{plan.goal}</strong>{plan.explanation?<p>{plan.explanation}</p>:null}{plan.graph_state?<p className="agent-graph-state">图状态：{planGraphStateLabel(plan.graph_state.status)} · 下一动作：{planActionLabel(plan.graph_state.next_action)}{plan.graph_state.next_node_id?` · 节点 ${plan.graph_state.next_node_id}`:""}{plan.graph_state.retry_node_id?` · 返工 ${plan.graph_state.retry_node_id}`:""}</p>:null}</header><ol>{plan.steps.map((step)=><li className={step.status} key={step.id}><i>{step.status==="completed"?<Check size={11}/>:step.status==="in_progress"?<RefreshCw size={11}/>:step.id}</i><span><b>{step.description}</b><small>执行：{planStepStatusLabel(step.status)} · 验证：{stepVerificationSummary(step)}{step.assignee?` · ${step.assignee}`:""}{step.depends_on?.length?` · 依赖 ${step.depends_on.join(", ")}`:""}</small>{step.state?<small className="agent-node-state">状态对象：{step.state.attempts||0} 次尝试{step.state.usage?.total_tokens?` · ${step.state.usage.total_tokens} tokens`:""}{step.state.usage?.duration_ms?` · ${step.state.usage.duration_ms} ms`:""}{step.state.tests?.length?` · ${step.state.tests.filter(test=>test.status==="passed").length}/${step.state.tests.length} tests 通过`:""}{step.state.artifact_ids?.length?` · ${step.state.artifact_ids.length} 个 Artifact`:""}</small>:null}{step.acceptance_criteria?.length?<ul className="agent-plan-criteria">{step.acceptance_criteria.map((criterion)=>{const fact=latestVerificationFact(verifications,plan.revision,step.id,criterion.id);return <li className={criterion.status} key={criterion.id}><i>{criterion.status==="passed"?<Check size={10}/>:criterion.status==="failed"||criterion.status==="invalid"?"!":criterion.status==="unsupported"?"?":"·"}</i><span>{criterion.description}<small>{criterionStatusLabel(criterion.status)} · {enforcementLabel(criterion.enforcement)}</small>{criterion.verification?.kind?<small>校验：{verificationLabel(criterion.verification.kind)}{criterion.verification.target?` · ${criterion.verification.target}`:""}</small>:null}{fact?.spec?<small className="agent-verification-fact">已编译：{fact.spec.provider_key}@{fact.spec.provider_version} · {fact.attempts.length} 次尝试 · {fact.evidence.length} 条证据 · {shortDigest(fact.spec.spec_digest)}</small>:fact?<small className="agent-verification-fact">Intent 已记录 · {fact.intent.status}{fact.intent.diagnostic_code?` · ${fact.intent.diagnostic_code}`:""}</small>:null}{criterion.verification_message?<small className="agent-verification-diagnostic">{criterion.verification_message}</small>:null}{criterion.evidence?<small>平台证据：{criterion.evidence}</small>:null}{criterion.evidence_call_ids?.length?<span className="agent-plan-evidence-links">{criterion.evidence_call_ids.map((callID)=><button type="button" key={callID} title={`查看平台绑定的成功工具事件 ${callID}`} onClick={()=>onEvidenceSelect(callID)}><Activity size={10}/>平台证据 {shortID(callID)}</button>)}</span>:null}</span></li>})}</ul>:null}{step.result?<em>{step.result}</em>:null}</span></li>)}</ol></div>
    </details> : null}
    {question ? <form className="agent-user-question" onSubmit={(event)=>{event.preventDefault();submit();}}>
      <MessageSquarePlus size={18}/><span><strong>Agent 需要你的决定</strong><p>{question.question}</p>{question.context?<small>{question.context}</small>:null}{question.options?.length?<><div>{question.options.map((option)=><button type="button" disabled={answering} className={answer===option?"active":""} key={option} onClick={()=>{onAnswerChange(option);onAnswer(option)}}>{option}</button>)}</div><small>点击选项将直接提交并从检查点继续。</small></>:null}<textarea value={answer} disabled={answering} onChange={(event)=>onAnswerChange(event.target.value)} placeholder="也可以输入自定义回答"/>{error?<em className="agent-question-error">{error instanceof Error?error.message:"提交回答失败"}</em>:null}</span><button className="agent-question-submit" type="submit" disabled={!answer.trim()||answering}>{answering?<RefreshCw className="agent-spin" size={14}/>:<Send size={14}/>}提交自定义回答</button>
    </form> : null}
  </section>;
}

function dependencyApproval(request:Record<string,unknown>){
  const root=request as {arguments?:unknown};
  const args=(root.arguments&&typeof root.arguments==="object"?root.arguments:{}) as {packages?:Array<{name?:unknown;version?:unknown}>;source?:unknown;scope?:unknown;reason?:unknown};
  if(!Array.isArray(args.packages))return undefined;
  const packages=args.packages.map(item=>`${String(item?.name||"?")}==${String(item?.version||"?")}`).join(", ");
  return {packages,source:String(args.source||"未声明"),scope:String(args.scope||"未声明"),reason:String(args.reason||"未说明")};
}

function projectFileApproval(request:Record<string,unknown>){
  const root=request as {arguments?:unknown};
  const args=(root.arguments&&typeof root.arguments==="object"?root.arguments:{}) as {path?:unknown;reason?:unknown;start_line?:unknown;line_count?:unknown};
  if(typeof args.path!=="string"||!args.path.trim())return undefined;
  const start=typeof args.start_line==="number"&&args.start_line>0?args.start_line:1;
  const count=typeof args.line_count==="number"&&args.line_count>0?args.line_count:undefined;
  return {path:args.path,reason:String(args.reason||"未说明"),range:count?`第 ${start}-${start+count-1} 行`:`未声明（请求将被拒绝）`};
}

function TraceWorkspace({run,version,events,groups,expanded,loading,error,selectedSequence,onSelect,onToggle,onRetry,onOpenRun,onCancel,cancelling}:{run?:AgentRun;version?:AgentExecutableVersion;events:AgentEvent[];groups:TraceGroup[];expanded:Set<string>;loading:boolean;error:unknown;selectedSequence?:number;onSelect:(sequence:number)=>void;onToggle:(key:string)=>void;onRetry:()=>void;onOpenRun:(run:AgentRun)=>void;onCancel:(runID:string)=>void;cancelling:boolean}){
  const [filter,setFilter]=useState("");
  const [category,setCategory]=useState("all");
  const [artifactPreview,setArtifactPreview]=useState<{id:string;name:string;content:string}>();
  const [promotion,setPromotion]=useState<{artifact:AgentArtifact;targetPath:string;expectedHash:string;currentHash?:string;success?:string}>();
  const trajectory=useQuery({queryKey:["agent","trajectory",run?.id],queryFn:()=>getAgentRunTrajectory(run!.id),enabled:Boolean(run),retry:false,refetchInterval:2000});
  const memoryTimeline=useQuery({queryKey:["agent","memory-timeline",run?.id],queryFn:()=>listAgentMemoryTimeline(run!.id,100),enabled:Boolean(run),retry:false,refetchInterval:3000});
  const scores=useQuery({queryKey:["agent","scores",run?.id],queryFn:()=>listAgentRunScores(run!.id),enabled:Boolean(run),retry:false,refetchInterval:5000});
  const artifacts=useQuery({queryKey:["agent","artifacts",run?.id],queryFn:()=>listAgentRunArtifacts(run!.id),enabled:Boolean(run),retry:false,refetchInterval:3000});
  const manifest=useQuery({queryKey:["agent","manifest",run?.id],queryFn:()=>getAgentRunManifest(run!.id),enabled:Boolean(run),retry:false,refetchInterval:3000});
  const children=useQuery({queryKey:["agent","children",run?.id],queryFn:()=>listAgentRunChildren(run!.id),enabled:Boolean(run),retry:false,refetchInterval:3000});
  const workflowRuns=useQuery({queryKey:["agent","workflow-runs",run?.workflow_id],queryFn:()=>listAgentWorkflowRuns(run!.workflow_id!),enabled:Boolean(run?.workflow_id),retry:false,refetchInterval:3000});
  const promote=useMutation({
    mutationFn:({expectedHash}:{expectedHash:string})=>{if(!promotion)throw new Error("未选择 Artifact");return promoteAgentArtifact(promotion.artifact.id,promotion.targetPath,expectedHash);},
    onSuccess:result=>setPromotion(current=>current?{...current,success:`已合并到 ${result.data.target_path} · ${shortDigest(result.data.result_target_sha256||"")}`,currentHash:undefined}:current),
    onError:error=>{
      let currentHash="";
      if(error instanceof ApiError){try{const payload=JSON.parse(error.body) as {error?:{current_target_sha256?:string}};currentHash=payload.error?.current_target_sha256||"";}catch{/* keep the typed API error */}}
      setPromotion(current=>current?{...current,currentHash:currentHash||undefined}:current);
    }
  });
  if(loading)return <div className="agent-trace-workspace-state"><Skeleton rows={9}/></div>;
  if(error)return <div className="agent-trace-workspace-state"><ErrorState title="无法读取 Trace" error={error} onRetry={onRetry}/></div>;
  if(!run||!events.length)return <div className="agent-trace-workspace-state"><EmptyState title="选择 Run 查看完整轨迹" description="模型解析、Skill、Memory、Context 与 Tool 调用都会在这里呈现。" icon={Activity}/></div>;
  const query=filter.trim().toLowerCase();
  const visible=events.filter(event=>(category==="all"||eventCategory(event)===category)&&(!query||`${eventLabel(event.type)} ${event.type} ${eventSummary(event)}`.toLowerCase().includes(query)));
  const workspaceID=runWorkspaceID(events);
  const latestWorkspaceArtifactIDs=new Set<string>();
  const latestByName=new Map<string,AgentArtifact>();
  for(const item of artifacts.data?.data||[]){if(item.kind==="workspace_file")latestByName.set(item.name,item);}
  for(const item of latestByName.values())latestWorkspaceArtifactIDs.add(item.id);
  const previewArtifact=async(id:string,name:string)=>{const content=await getAgentArtifactContent(id);setArtifactPreview({id,name,content:await content.text()});};
  return <div className="agent-trace-workspace">
    <header className="agent-trace-summary"><span><small>RUN TRACE</small><strong>{version?.agent_name||"Agent 执行"}</strong><em>{shortID(run.id)} · {durationText(run)}</em></span><span className="agent-trace-run-actions"><StatusBadge status={cancelling&&!isTerminalRun(run)?"停止中":statusLabel(run.status)}/>{!isTerminalRun(run)?<button type="button" disabled={cancelling} onClick={()=>onCancel(run.id)}>{cancelling?<RefreshCw className="agent-spin" size={13}/>:<CircleStop size={13}/>} {cancelling?"停止中…":"停止任务"}</button>:null}</span><div><Info label="执行方式" value={executionModeLabel(run,events)}/><Info label="事件" value={String(events.length)}/><Info label="步骤" value={String(groups.filter(group=>group.kind==="step").length)}/><Info label="模型调用" value={String(events.filter(event=>event.type==="MODEL_COMPLETED"||event.type==="MODEL_FAILED").length)}/><Info label="工具调用" value={String(events.filter(event=>event.type==="TOOL_CALLED").length)}/><Info label="隔离工作区" value={workspaceID?shortID(workspaceID):"待创建"}/></div></header>
    <TraceLanes events={events} selectedSequence={selectedSequence} onSelect={onSelect}/>
    <ProjectedTrajectory records={trajectory.data?.data?.records||[]} loading={trajectory.isLoading} expanded={expanded} onToggle={onToggle} onSelect={onSelect}/>
    {manifest.data?.data?<section className="agent-run-manifest"><header><span><strong>Run Manifest</strong><small>平台维护的最终交付投影，不接受模型自行填写的 receipt</small></span><em>{manifest.data.data.final_output_present||manifest.data.data.final_artifact_ids?.length?"最终交付已固化":"等待最终输出"}</em></header><div><span><b>状态</b>{manifest.data.data.status}</span><span><b>Canonical Artifact</b>{manifest.data.data.canonical_artifacts?.length||0} 个</span><span><b>最终 Artifact</b>{manifest.data.data.final_artifact_ids?.length||0} 个</span><span><b>Child Run</b>{manifest.data.data.child_runs?.length||0} 个</span><span><b>更新时间</b>{new Date(manifest.data.data.updated_at).toLocaleTimeString()}</span></div></section>:null}
    <MemoryTimeline records={memoryTimeline.data?.data||[]} loading={memoryTimeline.isLoading} error={memoryTimeline.error}/>
    {workflowRuns.data?.data && workflowRuns.data.data.length>1?<section className="agent-workflow-history"><header><span><strong>Workflow Attempts</strong><small>同一长期任务的 Run Attempt，按时间顺序保留断点与失败历史</small></span><span>{workflowRuns.data.data.length} 次尝试</span></header><div>{workflowRuns.data.data.map((attempt,index)=><button type="button" className={attempt.id===run.id?"active":""} key={attempt.id} onClick={()=>onOpenRun(attempt)}><i>↳</i><span><b>Attempt {index+1}</b><small>{attempt.status} · {attempt.created_at?new Date(attempt.created_at).toLocaleString():""}</small></span><em>{attempt.id===run.id?"当前":"查看"}</em></button>)}</div></section>:null}
    <ScoreStrip scores={scores.data?.data||[]} loading={scores.isLoading}/>
    <section className="agent-event-ledger">
      <header><span><strong>Event Ledger</strong><small>按真实发生顺序记录，可筛选并点击检查原始输入/输出</small></span><div><label><Search size={13}/><input value={filter} onChange={event=>setFilter(event.target.value)} placeholder="搜索事件"/></label><select value={category} onChange={event=>setCategory(event.target.value)}><option value="all">全部类型</option><option value="runtime">Runtime</option><option value="plan">Plan / Input</option><option value="identity">Identity</option><option value="model">Model</option><option value="context">Context</option><option value="memory">Memory</option><option value="skill">Skill</option><option value="tool">Tool</option><option value="agent">Agent Delegation</option></select></div></header>
      <div className="agent-event-ledger-head"><span>#</span><span>时间</span><span>类型</span><span>阶段</span><span>摘要</span><span>耗时</span></div>
      <VirtualEventLedger events={visible} allEvents={events} selectedSequence={selectedSequence} onSelect={onSelect}/>
    </section>
    {(artifacts.data?.data?.length||children.data?.data?.length)?<section className="agent-run-assets"><header><span><strong>Artifacts & Child Runs</strong><small>工具产物和内部 Agent 委派关系均来自当前 Run；已完成 Run 的最新文件快照可受控合并回项目</small></span></header><div>{artifacts.data?.data?.map(item=><div className="agent-artifact-row" key={item.id}><button type="button" onClick={()=>void previewArtifact(item.id,item.name)}><FileDiff size={15}/><span><b>{item.name}</b><small>{item.kind} · {Math.max(1,Math.ceil(item.size_bytes/1024))} KiB · {item.storage_backend === "minio" ? "MinIO" : "PG 内联"} · {shortDigest(item.content_hash)}</small></span><Download size={13}/></button>{run.status==="completed"&&latestWorkspaceArtifactIDs.has(item.id)?<button className="promote" type="button" onClick={()=>setPromotion({artifact:item,targetPath:item.name,expectedHash:""})}>合并到项目</button>:null}</div>)}{children.data?.data?.map(item=><button type="button" key={item.id} onClick={()=>onOpenRun(item)}><GitBranch size={15}/><span><b>Child Run · {shortID(item.id)}</b><small>depth {item.delegation_depth||1} · {statusLabel(item.status)}</small></span><ChevronRight size={13}/></button>)}</div>{artifactPreview?<div className="agent-artifact-preview"><header><strong>{artifactPreview.name}</strong><button type="button" onClick={()=>setArtifactPreview(undefined)}>关闭</button></header><pre>{artifactPreview.content}</pre></div>:null}{promotion?<form className="agent-artifact-promotion" onSubmit={event=>{event.preventDefault();promote.mutate({expectedHash:promotion.currentHash||promotion.expectedHash});}}><header><span><strong>合并 Artifact 到项目工作区</strong><small>源文件 {promotion.artifact.name} · {shortDigest(promotion.artifact.content_hash)}</small></span><button type="button" onClick={()=>setPromotion(undefined)}>关闭</button></header><label>目标相对路径<input value={promotion.targetPath} disabled={promote.isPending||Boolean(promotion.success)} onChange={event=>setPromotion({...promotion,targetPath:event.target.value,currentHash:undefined,expectedHash:""})}/></label>{promotion.currentHash?<p className="conflict">目标已存在，当前摘要为 <code>{promotion.currentHash}</code>。确认 Diff 后再次提交才会覆盖。</p>:null}{promotion.success?<p className="success">{promotion.success}</p>:null}{promote.isError&&!promotion.currentHash?<p className="conflict">{promote.error instanceof Error?promote.error.message:"合并失败"}</p>:null}<footer><small>后端仅接受已完成 Run 的最新文件 Artifact；目标已存在时使用 SHA-256 乐观锁。</small>{!promotion.success?<button type="submit" disabled={promote.isPending||!promotion.targetPath.trim()}>{promote.isPending?"合并中…":promotion.currentHash?"确认覆盖当前版本":"执行合并"}</button>:null}</footer></form>:null}</section>:null}
    <section className="agent-stage-section"><header><span><strong>Stage Trace</strong><small>Workflow → Plan Node → Decision Cycle → Action 的聚合视图</small></span></header>{groups.map(group=><StepTrace key={group.key} group={group} open={expanded.has(group.key)} onToggle={()=>onToggle(group.key)}/>)}</section>
  </div>;
}

function MemoryTimeline({records,loading,error}:{records:AgentMemoryTimelineEntry[];loading:boolean;error:unknown}){
  if(loading)return <section className="agent-memory-timeline"><header><strong>Memory Timeline</strong><small>正在合并 Run Event 与 Memory Lifecycle…</small></header><Skeleton rows={3}/></section>;
  if(error)return <section className="agent-memory-timeline"><header><strong>Memory Timeline</strong><small>读取失败：{error instanceof Error?error.message:"未知错误"}</small></header></section>;
  return <section className="agent-memory-timeline"><header><span><strong>Memory Timeline</strong><small>Run Event 与 Memory Lifecycle 的统一时间线</small></span><em>{records.length} 条</em></header>{records.length?<div>{records.slice(-80).map(item=><article key={item.id}><time>{new Date(item.created_at).toLocaleTimeString()}</time><b className={item.source}>{item.event_type}</b><span>{item.memory_id?`memory:${shortID(item.memory_id)}`:item.source}</span></article>)}</div>:<p>当前 Run 尚无 Memory 生命周期事件</p>}</section>;
}

function ScoreStrip({scores,loading}:{scores:AgentScore[];loading:boolean}){
  if(loading||!scores.length)return null;
  return <section className="agent-score-strip"><header><strong>Evaluation Scores</strong><small>可关联 Run、Observation、Prompt/Tool/Model 版本</small></header><div>{scores.map(item=><span key={item.id}><b>{item.name}</b><strong>{item.value!==undefined?String(item.value):item.string_value||"—"}</strong><small>{item.source}{item.evaluator_version?` · ${item.evaluator_version}`:""}</small></span>)}</div></section>;
}

function ProjectedTrajectory({records,loading,expanded,onToggle,onSelect}:{records:AgentTrajectoryRecord[];loading:boolean;expanded:Set<string>;onToggle:(key:string)=>void;onSelect:(sequence:number)=>void}){
  if(loading)return <section className="agent-projected-trajectory"><header><strong>Trajectory Projection</strong><small>后端正在合并生命周期事件…</small></header></section>;
  const children = new Map<string,AgentTrajectoryRecord[]>();
  const roots: AgentTrajectoryRecord[] = [];
  for (const record of records) {
    if (record.parent_id) {
      const list = children.get(record.parent_id) || [];
      list.push(record); children.set(record.parent_id, list);
    } else roots.push(record);
  }
  const renderNode = (record:AgentTrajectoryRecord, depth=0):ReactNode => {
    const nested = children.get(record.id) || [];
    const open = expanded.has(`trajectory:${record.id}`);
    return <div className={`agent-trajectory-node ${record.kind} ${record.status}`} key={record.id}>
      <button type="button" className="agent-trajectory-row" style={{paddingLeft: `${8 + depth * 18}px`}} title={`底层事件 ${record.event_sequences.join(", ")}`} onClick={()=>onSelect(record.sequence)}>
        <span className="agent-trajectory-toggle" onClick={(event)=>{event.stopPropagation();if(nested.length)onToggle(`trajectory:${record.id}`)}}>{nested.length?(open?<ChevronDown size={12}/>:<ChevronRight size={12}/>):<i/>}</span>
        <span className="agent-trajectory-kind"><b>{trajectoryKindLabel(record.kind)}</b><small>{record.plan_node_id?`Plan ${shortID(record.plan_node_id)}`:record.decision_cycle?`Decision ${record.decision_cycle}`:record.turn?`兼容 Turn ${record.turn}`:"任务级"}{record.action_id?` · Action ${shortID(record.action_id)}`:""}</small></span>
        <span className="agent-trajectory-summary"><strong>{record.summary}</strong>{record.agent_version_id||record.model_resolution_id||record.model_id||record.tool_version_id||record.prompt_version_id||record.toolset_version_id||record.skillset_version_id?<small>{[record.agent_version_id&&`agent:${shortID(record.agent_version_id)}`,record.model_id&&`model:${shortID(record.model_id)}`,record.model_resolution_id&&`resolution:${shortID(record.model_resolution_id)}`,record.tool_version_id&&`tool:${shortID(record.tool_version_id)}`,record.prompt_version_id&&`prompt:${shortID(record.prompt_version_id)}`,record.skillset_version_id&&`skill:${shortID(record.skillset_version_id)}`,record.toolset_version_id&&`tools:${shortID(record.toolset_version_id)}`].filter(Boolean).join(" · ")}</small>:null}</span>
        <span className="agent-trajectory-metrics">{record.input_tokens||record.output_tokens?<small>{record.input_tokens||0}+{record.output_tokens||0} tok</small>:null}{record.total_cost?<small>${record.total_cost.toFixed(4)}</small>:null}<em>{record.duration_ms?formatMS(record.duration_ms):record.status}</em></span>
      </button>
      {open&&nested.length?<div className="agent-trajectory-children">{nested.map(child=>renderNode(child,depth+1))}</div>:null}
    </div>;
  };
  return <section className="agent-projected-trajectory"><header><strong>Trajectory Projection</strong><small>按父子 Observation 展开调用关系；Event Ledger 仍是审计真相源</small><em>{records.length} 个节点</em></header><div className="agent-trajectory-tree">{roots.length?roots.map(record=>renderNode(record)): <p>暂无可投影轨迹</p>}</div></section>;
}

function VirtualEventLedger({events,allEvents,selectedSequence,onSelect}:{events:AgentEvent[];allEvents:AgentEvent[];selectedSequence?:number;onSelect:(sequence:number)=>void}){
  const rowHeight=40;
  const height=Math.min(520,Math.max(80,events.length*rowHeight));
  const [scrollTop,setScrollTop]=useState(0);
  const overscan=8;
  const start=Math.max(0,Math.floor(scrollTop/rowHeight)-overscan);
  const end=Math.min(events.length,Math.ceil((scrollTop+height)/rowHeight)+overscan);
  return <div className="agent-event-ledger-body virtual" style={{height}} onScroll={event=>setScrollTop(event.currentTarget.scrollTop)}><div style={{height:events.length*rowHeight,position:"relative"}}>{events.slice(start,end).map((event,offset)=>{const index=start+offset;return <button type="button" key={event.sequence} style={{position:"absolute",top:index*rowHeight,height:rowHeight,left:0,right:0}} className={selectedSequence===event.sequence?"active":""} onClick={()=>onSelect(event.sequence)}><code>{String(event.sequence).padStart(2,"0")}</code><time>{new Date(event.created_at).toLocaleTimeString([], {hour12:false})}</time><span className={`agent-event-kind ${eventCategory(event)}`}>{eventLabel(event.type)}</span><span>{eventPositionLabel(event)}</span><strong>{eventSummary(event)}</strong><em>{eventDuration(event,allEvents,allEvents.findIndex(item=>item.sequence===event.sequence))||"—"}</em></button>})}</div></div>;
}

function trajectoryKindLabel(value:string){return ({run:"Workflow",model:"Model Call",tool:"Action / Tool",tool_projection:"Tool Projection",step:"Action",turn:"Decision Cycle",skill:"Skill",memory:"Memory",context:"Context",plan:"Plan / User Input",agent:"Child Workflow",checkpoint:"Checkpoint",verification:"Verification",lifecycle:"Lifecycle"} as Record<string,string>)[value]||value}

function TraceLanes({events,selectedSequence,onSelect}:{events:AgentEvent[];selectedSequence?:number;onSelect:(sequence:number)=>void}){
  const lanes=["runtime","plan","identity","skill","memory","context","model","tool","agent"];
  return <section className="agent-trace-lanes"><header><strong>Execution Timeline</strong><small>Workflow → Plan Node → Decision Cycle → Action</small></header><div>{lanes.map(lane=>{const items=events.filter(event=>eventCategory(event)===lane);return <section key={lane}><b>{categoryLabel(lane)}</b><div>{items.length?items.map(event=><button type="button" key={event.sequence} className={`${eventTone(event.type)} ${selectedSequence===event.sequence?"active":""}`} title={`${eventLabel(event.type)} · ${eventSummary(event)}`} onClick={()=>onSelect(event.sequence)}><i/><span>{eventLabel(event.type)}</span><small>{eventPositionLabel(event)}</small></button>):<em>无事件</em>}</div></section>})}</div></section>;
}

function TraceInspector({run,version,events,selectedSequence}:{run?:AgentRun;version?:AgentExecutableVersion;events:AgentEvent[];selectedSequence?:number}){
  const [copied,setCopied]=useState(false);
  const event=events.find(item=>item.sequence===selectedSequence);
  if(!run)return <EmptyState title="选择 Run 查看 Inspector" description="点击事件账本中的任意节点，可检查该事件的业务字段。" icon={Activity}/>;
  if(!event)return <section className="agent-trace-inspector"><header><span><small>EVENT INSPECTOR</small><strong>选择一个执行事件</strong><em>{version?.agent_name||"Agent 执行"} · {shortID(run.id)}</em></span><StatusBadge status={statusLabel(run.status)}/></header><div className="agent-inspector-empty"><Activity size={21}/><strong>检查模型与工具的真实输入输出</strong><p>点击中间的泳道节点或 Event Ledger 行。Inspector 会按事件类型展示业务字段；内部 ID、租约令牌等噪声默认隐藏，原始内容仍可复制。</p></div></section>;
  const visible=filteredPayload(event.payload||{});
  const copy=async()=>{await copyToClipboard(pretty({type:event.type,turn:event.turn,step:event.step,sequence:event.sequence,created_at:event.created_at,payload:visible}));setCopied(true);window.setTimeout(()=>setCopied(false),1500)};
  return <section className="agent-trace-inspector"><header><span><small>{eventCategory(event).toUpperCase()} EVENT · #{event.sequence}</small><strong>{eventLabel(event.type)}</strong><em>{new Date(event.created_at).toLocaleString()}</em></span><button type="button" onClick={()=>void copy()}>{copied?<Check size={13}/>:<Copy size={13}/>} {copied?"已复制":"复制事件"}</button></header><div className="agent-trace-inspector-meta"><Info label="事件类型" value={event.type}/><Info label="执行位置" value={eventPositionLabel(event)}/></div><EventDetail event={event} modelRequest={previousModelRequest(events,event)}/><details className="agent-json-block"><summary>清洗后的事件 JSON</summary><pre>{pretty(visible)}</pre></details></section>;
}

function eventCategory(event:AgentEvent){
  if(event.type.includes("PLAN_")||event.type.includes("VERIFICATION_")||event.type.includes("USER_INPUT_")||event.type==="EXECUTION_MODE_SELECTED")return "plan";
  if(event.type.includes("IDENTITY_"))return "identity";
  if(event.type.includes("MODEL_"))return "model";
  if(event.type.includes("TOOL_"))return "tool";
  if(event.type.includes("DELEGATION_")||event.type.includes("CHILD_RUN_"))return "agent";
  if(event.type.includes("MEMORY_"))return "memory";
  if(event.type.includes("SKILL_"))return "skill";
  if(event.type.includes("CONTEXT_"))return "context";
  return "runtime";
}
function categoryLabel(value:string){return ({runtime:"Runtime",plan:"Plan / Input",identity:"Identity",skill:"Skill",memory:"Memory",context:"Context",model:"Model",tool:"Tool",agent:"Agent"} as Record<string,string>)[value]||value}
function eventSummary(event:AgentEvent){
  const payload=event.payload||{};
  if(event.type==="EXECUTION_MODE_SELECTED")return `${payload.mode==="planned"?"规划执行":"直接回答"} · ${planningPolicyLabel(String(payload.policy||"auto"))}`;
  if(event.type==="MODEL_RESOLVED")return `${payload.model_id||"待解析"} · ${selectionLabel(String(payload.selection_policy||"pinned"))}`;
  if(event.type==="MODEL_REQUESTED")return `${payload.model_id||"模型"} · ${payload.messages?.length||0} 条消息`;
  if(event.type==="MODEL_COMPLETED")return `${payload.usage?.input_tokens||0}+${payload.usage?.output_tokens||0} Token`;
  if(event.type==="MODEL_FAILED")return String(payload.error||"模型调用失败");
  if(event.type==="TOOL_CALLED"||event.type==="TOOL_COMPLETED"||event.type==="TOOL_FAILED")return String(payload.name||"未命名工具");
  if(event.type==="MEMORY_RETRIEVED")return `${payload.count||0} 条记忆`;
  if(event.type==="SKILL_ACTIVATED")return `${Array.isArray(payload.skills)?payload.skills.length:0} 个 Skill`;
  if(event.type==="IDENTITY_COMPILED")return String(payload.identity?.role||"结构化身份已编译");
  if(event.type==="CONTEXT_BUILT")return `${payload.message_count||0} 条消息 · ${payload.input_tokens||payload.estimated_tokens||0} Token`;
  if(event.type==="CONTEXT_COMPACTED")return Number(payload.removed_messages||0)===0?`执行状态投影 · ${payload.before_tokens||0} → ${payload.after_tokens||0} Token · 未压缩历史`:`${payload.before_tokens||0} → ${payload.after_tokens||0} Token · 移除 ${payload.removed_messages||0} 条`;
  if(event.type==="PLAN_CREATED"||event.type==="PLAN_UPDATED")return `${payload.goal||"执行计划"} · r${payload.revision||1}`;
  if(event.type==="VERIFICATION_INTENT_CREATED")return `${payload.criterion_key||"验收意图"} · ${payload.status||"pending"}`;
  if(event.type==="VERIFICATION_SPEC_COMPILED")return `${payload.provider_key||"Provider"}@${payload.provider_version||"v1"}`;
  if(event.type==="VERIFICATION_SPEC_REVISED")return `${payload.criterion_key||"验收项"} · ${payload.action||"replace"}`;
  if(event.type==="VERIFICATION_COMPLETED"||event.type==="VERIFICATION_FAILED")return `${payload.criterion_key||"验收项"} · ${payload.verdict||"unknown"}`;
  if(event.type==="VERIFICATION_LOOP_DETECTED")return `${payload.criterion_id||payload.step_id||"验收恢复"} · 已熔断重复循环`;
  if(event.type==="PLAN_COMPLETION_BLOCKED")return payload.reason==="artifact_task_requires_plan"?"产物任务需要 Plan 与工具验证，正在继续执行":"Plan 仍有未完成 Todo，继续执行";
  if(event.type==="FINAL_OUTPUT_REJECTED")return "候选最终结果未通过完整性校验，继续执行";
  if(event.type==="USER_INPUT_REQUESTED")return String(payload.question||"等待用户输入");
  if(event.type==="USER_INPUT_RECEIVED")return "已收到回答并恢复执行";
  return event.step?`Turn ${event.turn||1} · Step ${event.step}`:"任务生命周期";
}
function semanticEvent(event:AgentEvent){
  const value=event.payload?._workflow;
  const envelope=value&&typeof value==="object"?value as Record<string,unknown>:{};
  return {
    workflowID:event.workflow_id||String(envelope.workflow_id||event.run_id),
    planNodeID:event.plan_node_id||String(envelope.plan_node_id||""),
    decisionCycle:event.decision_cycle||Number(envelope.decision_cycle||event.turn||0),
    actionID:event.action_id||String(envelope.action_id||event.call_id||"")
  };
}
function eventPositionLabel(event:AgentEvent){
  const semantic=semanticEvent(event);
  if(semantic.planNodeID)return `Plan ${shortID(semantic.planNodeID)}${semantic.decisionCycle?` · D${semantic.decisionCycle}`:""}${semantic.actionID?` · A${shortID(semantic.actionID)}`:""}`;
  if(semantic.decisionCycle)return `Decision ${semantic.decisionCycle}${semantic.actionID?` · Action ${shortID(semantic.actionID)}`:""}`;
  return "Workflow";
}
function previousModelRequest(events:AgentEvent[],event:AgentEvent){return [...events].reverse().find(item=>item.sequence<event.sequence&&item.type==="MODEL_REQUESTED")}
function runWorkspaceID(events:AgentEvent[]){for(let index=events.length-1;index>=0;index--){const value=events[index].payload?.result?.meta?.workspace_id;if(typeof value==="string"&&value)return value}return ""}

function CapabilityPanel({value,loading}:{value?:AgentPlatformCapabilities;loading:boolean}){
  if(loading)return <Skeleton rows={1}/>;
  if(!value)return null;
  return <section className="infra-panel agent-capability-panel"><header><span><ShieldCheck size={16}/><strong>生产能力契约</strong><small>runtime {value.framework_version}</small></span><em>{value.harnesses.map(item=>item.display_name).join(" · ")}</em></header><div>{value.features.map(feature=><article key={feature.key} className={`status-${feature.status}`} title={feature.description}><i/><span><b>{feature.name}</b><small>{feature.description}</small></span><em>{capabilityStatusLabel(feature.status)}{feature.frontend?" · UI":""}</em></article>)}</div></section>
}

function StoragePanel({value,loading,error}:{value?:AgentStorageSummary;loading:boolean;error:unknown}){
  if(loading)return <Skeleton rows={2}/>;
  if(error||!value)return <div className="agent-storage-error">存储拓扑暂不可用</div>;
  const backends=[value.postgresql,value.redis,value.minio,value.pgvector];
  return <section className="agent-storage-topology"><header><strong>Agent Storage</strong><small>运行时真实连接状态</small></header><div>{backends.map(backend=><article key={backend.name} title={backend.detail}><i className={`status-${backend.status}`}/><span><b>{backend.name}</b><small>{backend.role}</small></span><em>{storageStatusLabel(backend.status)}</em></article>)}</div><footer><span>{value.postgresql.metrics?.messages||0} 条消息</span><span>{value.postgresql.metrics?.current_run_states||0} 个恢复状态</span><span>{value.postgresql.metrics?.verification_evidence||0} 条验证证据</span><span>{value.postgresql.metrics?.pending_outbox||0} 条待发布</span><span>{value.minio.metrics?.object_artifacts||0} 个对象产物</span><span>{value.pgvector.metrics?.embedded_memories||0} 条语义记忆</span></footer></section>;
}

function storageStatusLabel(status:string){if(status==="ready")return "已接通";if(status==="degraded")return "已降级";if(status==="not_configured")return "未配置";if(status==="not_connected")return "未接入";return status}
function formatBytes(value:number){if(value<1024)return `${value} B`;if(value<1024*1024)return `${(value/1024).toFixed(1)} KB`;return `${(value/1024/1024).toFixed(1)} MB`}

function SessionButton({session,active,onClick}:{session:AgentSession;active:boolean;onClick:()=>void}){
  const activity=session.last_activity_at||session.updated_at||session.created_at;
  return <button type="button" className={`agent-session-button ${active?"active":""}`} onClick={onClick}><i className="agent-session-icon"><Layers3 size={14}/></i><span><b>{sessionLabel(session)}</b><small>{session.run_count||0} 个 Workflow · {session.message_count||0} 条持久化消息 · {relativeTime(activity)}</small></span>{session.active_run_count?<em>执行中</em>:null}</button>
}

function RunButton({run,version,active,onClick}:{run:AgentRun;version?:AgentExecutableVersion;active:boolean;onClick:()=>void}){
  return <button type="button" className={`agent-run-button ${active?"active":""}`} onClick={onClick}><i className={`agent-run-state ${run.status}`}/><span><b>{runTitle(run,version)}</b><small>{run.model_resolution?.model_id||"模型待解析"} · {durationText(run)}</small></span><History size={13}/></button>
}

function MemoryPanel({version,session}:{version:AgentExecutableVersion;session?:AgentSession}){
  const client=useQueryClient();
  const [scope,setScope]=useState<AgentMemory["scope"]>("session");
  const [kind,setKind]=useState<AgentMemory["kind"]>("preference");
  const [content,setContent]=useState("");
  const [ttlDays,setTTLDays]=useState(30);
  const [inspectedMemoryID,setInspectedMemoryID]=useState<string>();
  const query=useQuery({queryKey:["agent","memories",version.agent_id,session?.id,session?.user_id],queryFn:()=>listAgentMemories({agentID:version.agent_id,sessionID:session!.id,userID:session?.user_id}),enabled:Boolean(session),retry:false,refetchInterval:10000});
  const lifecycle=useQuery({queryKey:["agent","memory-lifecycle",inspectedMemoryID],queryFn:()=>listAgentMemoryLifecycleEvents(inspectedMemoryID!,50),enabled:Boolean(inspectedMemoryID),retry:false});
  const create=useMutation({mutationFn:()=>createAgentMemory({scope,kind,content:content.trim(),importance:kind==="preference"?.8:.5,agent_id:scope==="agent"||scope==="user"||scope==="session"?version.agent_id:undefined,user_id:scope==="user"?session?.user_id:undefined,session_id:scope==="session"?session?.id:undefined,ttl_seconds:ttlDays>0?ttlDays*86400:undefined}),onSuccess:async()=>{setContent("");await client.invalidateQueries({queryKey:["agent","memories"]})}});
  const remove=useMutation({mutationFn:deleteAgentMemory,onSuccess:async()=>client.invalidateQueries({queryKey:["agent","memories"]})});
  const memoryEnabled=Boolean(version.spec.memory?.enabled);
  const canCreate=Boolean(session&&content.trim()&&(scope!=="user"||session.user_id)&&!create.isPending);
  return <details className="agent-memory-panel" open>
    <summary><span><Brain size={14}/><b>分层记忆</b><em>{query.data?.data?.length||0}</em></span><small>{memoryEnabled?`本版本召回 ${version.spec.memory?.read_scopes?.join(" / ")||"—"} · ${version.spec.context?.memory_tokens||0} Token` : "当前版本不读取记忆；可先维护资源"}</small></summary>
    {!session?<p>选择或创建 Session 后管理记忆。</p>:<div className="agent-memory-body">
      <div className="agent-memory-editor"><select value={scope} onChange={event=>setScope(event.target.value as AgentMemory["scope"])}><option value="session">当前 Session</option><option value="user" disabled={!session.user_id}>当前用户</option><option value="agent">当前 Agent</option><option value="tenant">当前租户</option></select><select value={kind} onChange={event=>setKind(event.target.value as AgentMemory["kind"])}><option value="preference">偏好</option><option value="semantic">事实</option><option value="episodic">经历</option></select><label>TTL 天<input type="number" min="0" max="1825" value={ttlDays} onChange={event=>setTTLDays(Math.max(0,Number(event.target.value)||0))}/></label><textarea value={content} onChange={event=>setContent(event.target.value)} placeholder="写入明确事实或偏好；0 天表示永久保留"/><button type="button" disabled={!canCreate} onClick={()=>create.mutate()}><Plus size={13}/>{create.isPending?"写入中":"写入"}</button></div>
      {create.isError?<p className="agent-memory-error">{create.error instanceof Error?create.error.message:"记忆写入失败"}</p>:null}
      {query.isLoading?<Skeleton rows={2}/>:query.isError?<ErrorState title="无法读取记忆" error={query.error} onRetry={()=>query.refetch()}/>:query.data?.data?.length?<div className="agent-memory-list">{query.data.data.map(memory=><article key={memory.id}><span><b>{memoryScopeLabel(memory.scope)} · {memoryKindLabel(memory.kind)}</b><small>权重 {memory.importance.toFixed(1)} · {memory.expires_at?`${relativeTime(memory.expires_at)}过期`:"永久"} · {memory.embedding_status === "ready" ? `语义索引 ${memory.embedding_model || "ready"}` : memory.embedding_status === "failed" ? "语义索引待重试" : "等待语义索引"}</small></span><p>{memory.content}</p><div><button type="button" title="查看生命周期审计" onClick={()=>setInspectedMemoryID(inspectedMemoryID===memory.id?undefined:memory.id)}><History size={13}/></button><button type="button" title="软删除记忆" disabled={remove.isPending} onClick={()=>remove.mutate(memory.id)}><Trash2 size={13}/></button></div></article>)}</div>:<p>当前作用域没有有效记忆。</p>}
      {inspectedMemoryID?<div className="agent-memory-audit"><strong>Memory Lifecycle · {shortID(inspectedMemoryID)}</strong>{lifecycle.isLoading?<Skeleton rows={2}/>:lifecycle.isError?<ErrorState title="无法读取生命周期" error={lifecycle.error} onRetry={()=>lifecycle.refetch()}/>:lifecycle.data?.data?.length?<div>{lifecycle.data.data.map(item=><p key={item.id}><b>{item.event_type}</b><small>{new Date(item.created_at).toLocaleString()} · {item.actor||"system"}</small></p>)}</div>:<span>暂无生命周期事件</span>}</div>:null}
    </div>}
  </details>
}

function memoryScopeLabel(scope:string){return ({tenant:"租户",agent:"Agent",user:"用户",session:"Session"} as Record<string,string>)[scope]||scope}
function memoryKindLabel(kind:string){return ({semantic:"事实",episodic:"经历",preference:"偏好"} as Record<string,string>)[kind]||kind}

function capabilityStatusLabel(status:string){return ({enforced:"已强制",available:"可用",declared:"仅声明",missing:"未实现"} as Record<string,string>)[status]||status}
function sessionTitle(question:string){const text=question.trim().replace(/\s+/g," ");return text.length>28?`${text.slice(0,28)}…`:text||"新对话"}
function sessionLabel(session:AgentSession){return typeof session.metadata?.title==="string"?session.metadata.title:`Session · ${new Date(session.created_at).toLocaleString()}`}
function workflowLabel(item:{workflow_id:string;status:string;goal?:string;updated_at:string}){const goal=(item.goal||"").replace(/\s+/g," ").trim();return `${goal?goal.slice(0,28)+(goal.length>28?"…":""):"未命名任务"} · ${workflowStatusLabel(item.status)}`}
function workflowStatusLabel(value:string){return ({active:"执行中",waiting:"等待中",completed:"已完成",failed:"失败可恢复",cancelled:"已取消",archived:"已归档"} as Record<string,string>)[value]||value}
function shortID(value:string){return value.length>12?`${value.slice(0,8)}…`:value}

type TraceGroup={key:string;kind:"setup"|"step"|"result";turn:number;step:number;planNodeID?:string;events:AgentEvent[]};

function buildTraceGroups(events:AgentEvent[]):TraceGroup[]{
  const groups=new Map<string,TraceGroup>();
  const setupTypes=new Set(["RUN_CREATED","RUN_CLAIMED","RUN_RESUMED","MODEL_RESOLVED","IDENTITY_COMPILED","SKILL_ACTIVATED","TURN_STARTED"]);
  const stepTrailingTypes=new Set(["CHECKPOINT_CREATED","TURN_COMPLETED"]);
  let lastStepKey="";
  for(const event of events){
    const semantic=semanticEvent(event);
    let kind:TraceGroup["kind"];
    let key:string;
    if(event.step||semantic.actionID){
      kind="step";
      const cycle=semantic.decisionCycle||event.turn||1;
      const action=semantic.actionID||String(event.step||1);
      key=`action-${semantic.planNodeID||"unassigned"}-${cycle}-${action}`;lastStepKey=key;
    }else if(stepTrailingTypes.has(event.type)&&lastStepKey){
      kind="step";key=lastStepKey;
    }else if(setupTypes.has(event.type)&&!lastStepKey){
      kind="setup";key="setup";
    }else{
      kind="result";key="result";
    }
    const current=groups.get(key)||{key,kind,turn:semantic.decisionCycle||event.turn||0,step:event.step||0,planNodeID:semantic.planNodeID||undefined,events:[]};
    current.events.push(event);groups.set(key,current);
  }
  return [...groups.values()].sort((a,b)=>(a.events[0]?.sequence||0)-(b.events[0]?.sequence||0));
}

function TaskTrace({run,version,events}:{run:AgentRun;version?:AgentExecutableVersion;events:AgentEvent[]}){
  const steps=new Set(events.filter(event=>event.step).map(event=>`${event.turn}-${event.step}`)).size;
  const modelCalls=events.filter(event=>event.type==="MODEL_COMPLETED"||event.type==="MODEL_FAILED").length;
  const toolCalls=events.filter(event=>event.type==="TOOL_CALLED").length;
  const usage=events.filter(event=>event.type==="MODEL_COMPLETED").reduce((total,event)=>total+Number(event.payload?.usage?.total_tokens||0),0);
  const recalled=Number(events.find(event=>event.type==="MEMORY_RETRIEVED")?.payload?.count||0);
  const resolved=run.model_resolution;
  const identity=resolved?.model_version||shortDigest(resolved?.artifact_digest)||"上游未报告版本";
  return <section className="agent-task-trace"><header><span><b>Workflow Trace</b><small>一个长时任务的完整生命周期</small></span><StatusBadge status={statusLabel(run.status)}/></header><p>{inputText(run)}</p><div><Info label="Agent" value={version?.agent_name||"Agent"}/><Info label="选择策略" value={resolved?selectionLabel(resolved.selection_policy):modelPolicyLabel(version)}/><Info label="本次实际模型" value={resolved?.model_id||"等待 Worker 解析"}/><Info label="模型服务" value={resolved?.service_ref||version?.spec.model?.service_ref||"—"}/><Info label="版本 / Digest" value={identity}/><Info label="召回记忆" value={`${recalled} 条`}/><Info label="总耗时" value={durationText(run)}/><Info label="执行规模" value={`${steps} Action Group · ${events.length} Event`}/><Info label="模型 / Tool 调用" value={`${modelCalls} / ${toolCalls} 次 · ${usage} Token`}/></div>{run.error_message?<pre>{run.error_message}</pre>:null}</section>
}

function StepTrace({group,open,onToggle}:{group:TraceGroup;open:boolean;onToggle:()=>void}){
  const failed=group.events.some(event=>event.type.includes("FAILED")||event.payload?.status==="failed");
  const completed=group.events.some(event=>event.type==="STEP_COMPLETED"||event.type==="RUN_COMPLETED"||event.type==="RUN_FAILED");
  const title=group.kind==="setup"?"Workflow 准备与调度":group.kind==="result"?"Workflow 收尾与结果":`Decision Cycle ${group.turn} · Action Group ${group.step}`;
  return <article className={`agent-step-trace ${failed?"failed":completed?"completed":"running"}`}><button type="button" onClick={onToggle}><i><Activity size={13}/></i><span><b>{title}</b><small>{stepSummary(group)}</small></span><em>{groupDuration(group)}</em>{open?<ChevronDown size={15}/>:<ChevronRight size={15}/>}</button>{open?<div className="agent-step-detail"><TraceGroupDetail group={group}/><RawEventTrace events={group.events}/></div>:null}</article>
}

function stepSummary(group:TraceGroup){
  if(group.kind==="setup"){const skills=group.events.find(event=>event.type==="SKILL_ACTIVATED")?.payload?.skills?.length||0;return `创建任务、Worker 调度${skills?`、注入 ${skills} 个 Skill`:""}`}
  if(group.kind==="result"){const failed=group.events.find(event=>event.type==="RUN_FAILED");return failed?`失败 · ${String(failed.payload?.error||"执行异常")}`:"Run 完成并提交最终结果"}
  const model=group.events.find(event=>event.type==="MODEL_REQUESTED")?.payload?.model_id||"模型";
  const tools=group.events.filter(event=>event.type==="TOOL_CALLED").length;
  const memories=Number(group.events.find(event=>event.type==="MEMORY_RETRIEVED")?.payload?.count||0);
  const failed=group.events.find(event=>event.type==="MODEL_FAILED"||event.type==="TOOL_FAILED");
  return `${model} · ${tools} 次工具调用${memories?` · 召回 ${memories} 条记忆`:""}${failed?` · ${eventLabel(failed.type)}`:""}`;
}
function groupDuration(group:TraceGroup){const reported=group.events.find(event=>event.type==="STEP_COMPLETED")?.payload?.latency_ms;if(Number.isFinite(Number(reported)))return formatMS(Number(reported));const first=group.events[0],last=group.events[group.events.length-1];return first&&last?formatMS(Math.max(0,new Date(last.created_at).getTime()-new Date(first.created_at).getTime())):""}

function TraceGroupDetail({group}:{group:TraceGroup}){
  if(group.kind==="setup"){
    const created=group.events.find(event=>event.type==="RUN_CREATED");const resolved=group.events.find(event=>event.type==="MODEL_RESOLVED");const skills=group.events.find(event=>event.type==="SKILL_ACTIVATED");
    return <div className="agent-group-content"><DetailTitle label="任务准备结果" meta={`${group.events.length} 个底层事件`}/><p className="agent-detail-note">任务已创建，Worker 已接管执行并冻结本次模型解析结果。</p>{created?<JsonBlock label="原始任务输入" value={created.payload?.input??{}}/>:null}{resolved?<EventDetail event={resolved}/>:null}{skills?<EventDetail event={skills}/>:null}</div>;
  }
  if(group.kind==="result"){
    const terminal=[...group.events].reverse().find(event=>["RUN_COMPLETED","RUN_FAILED","RUN_CANCELLED"].includes(event.type));
    return <div className="agent-group-content">{terminal?<EventDetail event={terminal}/>:<p className="agent-detail-note">轮次已经结束，执行状态已写入持久化存储。</p>}</div>;
  }
  const context=group.events.find(event=>event.type==="CONTEXT_BUILT");
  const memory=group.events.find(event=>event.type==="MEMORY_RETRIEVED");
  const request=group.events.find(event=>event.type==="MODEL_REQUESTED");
  const response=group.events.find(event=>event.type==="MODEL_COMPLETED");
  const failure=group.events.find(event=>event.type==="MODEL_FAILED");
  const tools=group.events.filter(event=>event.type==="TOOL_COMPLETED"||event.type==="TOOL_FAILED");
  return <div className="agent-group-content">{memory?<EventDetail event={memory}/>:null}{context?<ContextSummary value={context.payload}/>:null}{failure?<EventDetail event={failure} modelRequest={request}/>:<>{request?<EventDetail event={request}/>:null}{response?<EventDetail event={response}/>:null}</>}{tools.map(event=><EventDetail event={event} key={event.sequence}/>)}</div>;
}

function RawEventTrace({events}:{events:AgentEvent[]}){
  return <details className="agent-raw-events"><summary>Event Trace · {events.length} 条原始事件</summary><div>{events.map((event,index)=>{const visible=filteredPayload(event.payload);return <details key={event.sequence}><summary><span>{eventLabel(event.type)}</span><em>{eventDuration(event,events,index)||"状态事件"}</em><time>{new Date(event.created_at).toLocaleTimeString()}</time></summary>{hasData(visible)?<pre>{pretty(visible)}</pre>:<p>该事件只记录生命周期状态，没有额外业务数据。</p>}</details>})}</div></details>
}

function AgentExecutionFeed({events,terminal,stopping,onOpenTrace}:{events:AgentEvent[];terminal:boolean;stopping:boolean;onOpenTrace:()=>void}){
  const allItems=buildAgentActivity(events);
  const hidden=Math.max(0,allItems.length-100);
  const items=hidden?allItems.slice(-100):allItems;
  const latest=items[items.length-1];
  if(!items.length)return null;
  return <details className={`agent-execution-feed ${terminal?"terminal":"live"}`} open={!terminal}>
    <summary><span><i className={terminal?"completed":"running"}>{terminal?<Check size={12}/>:<RefreshCw className="agent-spin" size={12}/>}</i><span><strong>{terminal?"执行过程":"Agent 执行动态"}</strong><small>{terminal?`${items.length} 个可见节点`:stopping?"正在安全停止":latestActivityLabel(latest)}</small></span></span><em>{terminal?"展开查看":"实时更新"}<ChevronDown size={13}/></em></summary>
    <div className="agent-execution-stream">
      {hidden?<p className="agent-execution-folded">较早的 {hidden} 个节点已折叠，可在完整轨迹中查看。</p>:null}
      {items.map(item=><AgentExecutionItem item={item} key={item.key}/>)}
    </div>
    <footer><span>内容来自持久化 Event Ledger；仅展示模型可见输出，不展示内部推理。</span><button type="button" onClick={onOpenTrace}><Activity size={11}/>完整轨迹</button></footer>
  </details>;
}

function AgentExecutionItem({item}:{item:AgentActivityItem}){
  const Icon=item.kind==="model"?Bot:item.kind==="tool"?Wrench:item.kind==="plan"?ListChecks:item.kind==="context"?Brain:item.kind==="agent"?GitBranch:item.kind==="input"?MessageSquarePlus:Activity;
  return <article className={`agent-execution-item ${item.kind} ${item.status}`} title={`Event ${item.eventSequences.join(", ")}`}>
    <i><Icon className={item.status==="running"?"agent-spin":""} size={13}/></i>
    <span><header><strong>{item.title}</strong><time>{new Date(item.createdAt).toLocaleTimeString([], {hour12:false})}</time></header>{item.detail?<small>{item.detail}</small>:null}{item.content?<div className="agent-execution-model-output"><MarkdownText text={item.content}/></div>:null}</span>
  </article>;
}

function latestActivityLabel(item:AgentActivityItem){
  if(item.status==="waiting")return item.title;
  if(item.status==="failed")return item.title;
  return item.status==="running"?item.title:`${item.title}，继续执行中`;
}

function ChatBubble({role,content}:{role:"user"|"assistant"|"error"|"pending";content:string}){return <div className={`agent-chat-bubble ${role}`}><small>{role==="user"?"你":role==="assistant"?"Agent":role==="error"?"执行失败":"Agent"}</small>{role==="assistant"?<MarkdownText text={assistantDisplayText(content)}/>:<pre>{content}</pre>}</div>}
function AssistantRunReply({run}:{run:AgentRun}){
  const artifacts=useQuery({queryKey:["agent","artifacts",run.id],queryFn:()=>listAgentRunArtifacts(run.id),retry:false,staleTime:30_000});
  const latestByName=new Map<string,AgentArtifact>();
  for(const artifact of artifacts.data?.data||[])if(artifact.kind==="workspace_file")latestByName.set(artifact.name,artifact);
  const files=[...latestByName.values()];
  const replyText=assistantDisplayText(outputText(run.output));
  const referencedFiles=files.filter(artifact=>replyText.includes(artifact.name));
  const deliveryFiles=referencedFiles.length?referencedFiles:files.filter(artifact=>!artifact.name.startsWith("_"));
  const download=async(artifact:AgentArtifact)=>{
    try{
      const blob=await getAgentArtifactContent(artifact.id);
      const url=URL.createObjectURL(blob);
      const anchor=document.createElement("a");
      anchor.href=url;anchor.download=artifact.name;document.body.appendChild(anchor);anchor.click();anchor.remove();
      window.setTimeout(()=>URL.revokeObjectURL(url),1000);
    }catch(error){toast.error("下载交付文件失败",{description:error instanceof Error?error.message:artifact.name});}
  };
  const artifactForLink=(href:string)=>{
    if(/^(?:https?:|mailto:|#)/i.test(href))return undefined;
    const clean=decodeURIComponent(href.split(/[?#]/,1)[0]).replace(/\\/g,"/").replace(/^\.\//,"");
    return latestByName.get(clean)||latestByName.get(clean.slice(clean.lastIndexOf("/")+1));
  };
  return <div className="agent-chat-bubble assistant"><small>Agent</small><MarkdownText text={replyText} onLink={(href)=>{const artifact=artifactForLink(href);if(!artifact)return false;void download(artifact);return true;}}/>{deliveryFiles.length?<div className="agent-delivery-files"><small>交付文件 · 点击下载已归档版本</small><div>{deliveryFiles.map(artifact=><button type="button" key={artifact.id} onClick={()=>void download(artifact)} title={`${artifact.name} · ${artifact.storage_backend==="minio"?"MinIO":"PostgreSQL"}`}><Download size={12}/><span>{artifact.name}</span><em>{formatBytes(artifact.size_bytes)}</em></button>)}</div></div>:null}</div>;
}
function Kpi({icon:Icon,label,value,detail}:{icon:typeof Bot;label:string;value:string|number;detail:string}){return <section><Icon size={17}/><span><small>{label}</small><strong>{value}</strong><em>{detail}</em></span></section>}
function SessionAuditPanel({value,loading,error,onRetry}:{value?:AgentSessionAudit;loading:boolean;error:unknown;onRetry:()=>void}){
  if(loading)return <div className="agent-runtime-view"><Skeleton rows={5}/></div>;
  if(error)return <div className="agent-runtime-view"><ErrorState title="无法读取 Session 审计" error={error} onRetry={onRetry}/></div>;
  if(!value)return <EmptyState title="请选择 Session" description="选择一个对话后查看 Token、耗时和工具成功率。"/>;
  return <div className="agent-runtime-view agent-session-audit">
    <header><span><strong>Session 执行审计</strong><small>{shortID(value.session_id)} · 基于持久化 Run / Model Call / Tool Execution</small></span><em>{value.last_activity_at?relativeTime(value.last_activity_at):"暂无执行"}</em></header>
    <div className="agent-kpis"><Kpi icon={Coins} label="Token" value={formatCount(value.total_tokens)} detail={`输入 ${formatCount(value.input_tokens)} · 输出 ${formatCount(value.output_tokens)}`}/><Kpi icon={Clock3} label="累计执行时间" value={formatAuditDuration(value.execution_duration_ms)} detail={`会话跨度 ${formatAuditDuration(value.wall_duration_ms)}`}/><Kpi icon={Wrench} label="工具成功率" value={`${value.tool_success_rate_percent.toFixed(1)}%`} detail={`${value.successful_tool_calls}/${value.terminal_tool_calls} 成功 · ${value.tool_calls} 次调用`}/><Kpi icon={Bot} label="模型调用" value={value.model_calls} detail={`${value.successful_model_calls} 成功 · ${value.failed_model_calls} 失败`}/></div>
    <div className="agent-runtime-diagnostics"><span>Run <b>{value.run_count}</b></span><span>完成 <b>{value.completed_runs}</b></span><span>失败 <b>{value.failed_runs}</b></span><span>执行中 <b>{value.active_runs}</b></span></div>
    <section className="agent-audit-runs"><header><strong>逐 Run 明细</strong><small>成功率仅统计已有终态的工具调用</small></header>{value.runs.length?<div>{[...value.runs].reverse().map(run=><article key={run.run_id}><i className={`agent-run-state ${run.status}`}/><span><b>Run {shortID(run.run_id)}</b><small>{formatAuditDuration(run.duration_ms)} · {formatCount(run.total_tokens)} tokens</small></span><em>{run.tool_calls} tools · {run.tool_success_rate_percent.toFixed(1)}%</em></article>)}</div>:<p>该 Session 尚无 Run。</p>}</section>
  </div>;
}
function formatAuditDuration(value:number){if(value<1000)return `${Math.max(0,Math.round(value))} ms`;const seconds=value/1000;if(seconds<60)return `${seconds.toFixed(seconds<10?1:0)} s`;const minutes=Math.floor(seconds/60);const rest=Math.round(seconds%60);if(minutes<60)return `${minutes}m ${rest}s`;const hours=Math.floor(minutes/60);return `${hours}h ${minutes%60}m`;}
function formatCount(value:number){return new Intl.NumberFormat("zh-CN",{notation:value>=10000?"compact":"standard",maximumFractionDigits:1}).format(value);}
function statusLabel(value:string){return value==="completed"?"正常":value==="failed"?"严重":value==="running"?"运行中":value==="queued"?"排队中":value==="waiting_approval"?"等待审批":value==="waiting_input"?"等待你的回答":value==="waiting_external"?"等待子 Agent":value==="cancelled"?"已取消":value==="suspended"?"已暂停":value}
function eventTone(value:string){return value.includes("FAILED")?"bad":value.includes("COMPLETED")?"good":"active"}
function eventLabel(value:string){const labels:Record<string,string>={WORKFLOW_CREATED:"创建 Workflow",RUN_CREATED:"创建 Run",RUN_ATTEMPT_CREATED:"创建执行尝试",WORKFLOW_ROUTING_DECIDED:"确定任务路由",WORKFLOW_RESUMED:"恢复 Workflow",WORKFLOW_SUSPENDED:"挂起 Workflow",WORKFLOW_COMPLETED:"Workflow 完成",RUN_CLAIMED:"Worker 接管",RUN_RESUMED:"恢复执行",RUN_SUSPENDED:"挂起执行",RUN_CANCEL_REQUESTED:"请求终止",MODEL_RESOLVED:"模型发现并冻结",IDENTITY_COMPILED:"编译角色身份",SKILL_ACTIVATED:"注入 Skill",EXECUTION_MODE_SELECTED:"确定执行方式",PLAN_CREATED:"创建执行计划",PLAN_UPDATED:"更新执行计划",PLAN_COMPLETION_BLOCKED:"Plan 未闭环，继续执行",VERIFICATION_INTENT_CREATED:"记录验收意图",VERIFICATION_SPEC_COMPILED:"编译验收规范",VERIFICATION_SPEC_REVISED:"修订验收规范",VERIFICATION_COMPLETED:"验证通过",VERIFICATION_FAILED:"验证未通过",VERIFICATION_LOOP_DETECTED:"验收循环已熔断",FINAL_OUTPUT_REJECTED:"最终结果校验未通过",USER_INPUT_REQUESTED:"询问用户",USER_INPUT_RECEIVED:"收到用户回答",TURN_CREATED:"创建轮次",TURN_STARTED:"开始轮次",STEP_STARTED:"开始步骤",MEMORY_RETRIEVED:"召回分层记忆",CONTEXT_BUILT:"构建上下文",CONTEXT_COMPACTED:"上下文处理",MODEL_REQUESTED:"请求模型",MODEL_COMPLETED:"模型回复",MODEL_FAILED:"模型失败",TOOL_CALLED:"调用工具",TOOL_APPROVAL_REQUESTED:"请求工具审批",TOOL_APPROVAL_RESOLVED:"工具审批完成",TOOL_COMPLETED:"工具返回",TOOL_FAILED:"工具失败",DELEGATION_REQUESTED:"委派子 Agent",DELEGATION_COMPLETED:"子 Agent 完成",DELEGATION_FAILED:"子 Agent 失败",STEP_COMPLETED:"步骤结束",STEP_FAILED:"步骤失败",TURN_COMPLETED:"轮次结束",CHECKPOINT_CREATED:"保存检查点",RUN_COMPLETED:"Run 完成",RUN_FAILED:"Run 失败",RUN_CANCELLED:"Run 取消"};return labels[value]||value}
function planStepStatusLabel(value:string){return ({pending:"待执行",in_progress:"执行中",completed:"已完成",blocked:"受阻",skipped:"已跳过"} as Record<string,string>)[value]||value}
function planVerificationOutcomeLabel(value?:string){return ({verified:"全部验证",partially_verified:"部分验证",unverified:"尚未验证",rejected:"强制验收未通过"} as Record<string,string>)[value||""]||"验证状态计算中"}
function planGraphStateLabel(value?:string){return ({idle:"空闲",ready:"可执行",running:"执行中",waiting:"等待条件",retry_required:"需要返工",completed:"已完成"} as Record<string,string>)[value||""]||value||"未计算"}
function planActionLabel(value?:string){return ({run:"执行",retry:"返工",wait:"等待",complete:"完成",none:"无"} as Record<string,string>)[value||""]||value||"未计算"}
function criterionStatusLabel(value:string){return ({pending:"未验证",passed:"验证通过",failed:"验证未通过",skipped:"已跳过",invalid:"验收规范无效",unsupported:"当前环境不支持",stale:"证据已过期"} as Record<string,string>)[value]||value}
function enforcementLabel(value?:string){return ({informational:"仅供参考",advisory:"建议验证",required:"必须验证",release_gate:"发布门禁"} as Record<string,string>)[value||"advisory"]||value||"建议验证"}
function stepVerificationSummary(step:AgentTaskPlan["steps"][number]){const criteria=step.acceptance_criteria||[];if(!criteria.length)return "无验收项";const passed=criteria.filter(item=>item.status==="passed").length;const requiredOpen=criteria.filter(item=>(item.enforcement==="required"||item.enforcement==="release_gate")&&item.status!=="passed").length;if(requiredOpen)return `${passed}/${criteria.length} 通过，${requiredOpen} 项强制待处理`;if(passed===criteria.length)return "全部通过";return `${passed}/${criteria.length} 通过（不阻塞完成）`}
function latestVerificationFact(records:AgentVerificationRecord[],revision:number,stepID:string,criterionID:string){
  const history=records.filter(item=>item.intent.plan_step_key===stepID&&item.intent.criterion_key===criterionID);
  const latest=history.find(item=>item.intent.plan_revision===revision)||history.sort((left,right)=>right.intent.plan_revision-left.intent.plan_revision)[0];
  if(!latest)return undefined;
  return {...latest,attempts:history.flatMap(item=>item.attempts),evidence:history.flatMap(item=>item.evidence)};
}
function verificationLabel(value:string){return ({tool_success:"工具成功",file_exists:"文件存在",file_contains:"内容匹配",python_syntax:"Python 语法",command_exit_zero:"实际运行",test_pass:"测试通过",list_nonempty:"目录非空",search_nonempty:"搜索命中"} as Record<string,string>)[value]||value}
function isTerminalRun(run:AgentRun){return run.status==="completed"||run.status==="failed"||run.status==="cancelled"}
function toggle(source:Set<string>,id:string){const next=new Set(source);if(next.has(id))next.delete(id);else next.add(id);return next}
function durationText(run:AgentRun){if(!run.started_at)return "未开始";const end=run.finished_at?new Date(run.finished_at).getTime():Date.now();return formatMS(Math.max(0,end-new Date(run.started_at).getTime()))}
function eventDuration(event:AgentEvent,all:AgentEvent[],index:number){const latency=Number(event.payload?.latency_ms);if(Number.isFinite(latency))return formatMS(latency);if(index===0)return "";return `+${formatMS(new Date(event.created_at).getTime()-new Date(all[index-1].created_at).getTime())}`}
function formatMS(ms:number){return ms>=1000?`${(ms/1000).toFixed(ms>=10000?1:2)} s`:`${Math.round(ms)} ms`}
function inputText(run:AgentRun){const value=run.input as Record<string,unknown>;return typeof value?.question==="string"?value.question:pretty(run.input)}
function runTitle(run:AgentRun,version?:AgentExecutableVersion){const question=inputText(run).replace(/\s+/g," ").trim();const short=question.length>32?`${question.slice(0,32)}…`:question;return `${version?.agent_name || "Agent"}${short?` · ${short}`:""}`}
function modelPolicyLabel(version?:AgentExecutableVersion){return selectionLabel(version?.spec.model?.selection_policy||"pinned")}
function selectionLabel(policy?:string){return policy==="auto"?"自动探测并冻结":"固定模型"}
function configuredModelLabel(version:AgentExecutableVersion){const model=version.spec.model;return model?.selection_policy==="auto"?`自动探测 · ${(model.model_candidates||[]).join(" / ")||model.model_id||"任意可用模型"}`:`固定 · ${model?.model_id||model?.service_ref||"未配置"}`}
function shortDigest(value?:string){if(!value)return "";return value.length>20?`${value.slice(0,12)}…${value.slice(-6)}`:value}
function outputText(output:unknown){const value=output as Record<string,unknown>;return typeof value?.content==="string"?value.content:pretty(output)}
function assistantDisplayText(value:string){
  const visible=visibleModelText(value);
  if(!visible)return "Agent 已完成执行，但模型没有返回可显示的文本内容。";
  const label=/^(?:#{1,6}\s*)?(?:\*\*)?(?:response|final answer|最终答案|最终回答|回答|回复)\s*[:：](?:\*\*)?\s*(.*)$/gim;
  let current:RegExpExecArray|null,last:RegExpExecArray|null=null;
  while((current=label.exec(visible))!==null)last=current;
  if(!last)return visible;
  const inline=last[1]?.trim();
  const tail=visible.slice((last.index||0)+last[0].length).trim();
  return [inline,tail].filter(Boolean).join("\n\n")||visible;
}
async function copyToClipboard(value:string){
  if(navigator.clipboard?.writeText){try{await navigator.clipboard.writeText(value);return}catch{/* fall back for restricted browser contexts */}}
  const area=document.createElement("textarea");
  area.value=value;area.style.position="fixed";area.style.opacity="0";document.body.appendChild(area);area.select();
  try{if(!document.execCommand("copy"))throw new Error("浏览器拒绝复制操作")}finally{area.remove()}
}
function pretty(value:unknown){return typeof value==="string"?value:JSON.stringify(value,null,2)}
function messageContent(message:AgentMessage){if(message.content)return readableText(message.content);if(message.parts?.length)return message.parts.map(part=>part.text?readableText(part.text):(part.json?pretty(part.json):part.uri??"")).filter(Boolean).join("\n");return ""}
function readableText(value:string){try{return pretty(JSON.parse(value))}catch{return value}}
function filteredPayload(value:unknown):unknown{
  if(Array.isArray(value))return value.map(filteredPayload);
  if(value&&typeof value==="object")return Object.fromEntries(Object.entries(value as Record<string,unknown>).filter(([key])=>!/(^id$|_id$|_ids$|hash$|traceparent|lease_token|worker_id|call_id)/i.test(key)).map(([key,item])=>[key,filteredPayload(item)]));
  return value;
}
function hasData(value:unknown){return Boolean(value&&typeof value==="object"&&Object.keys(value as object).length)}
function ContextSummary({value}:{value:any}){if(!value)return null;const summary={输入Token:value.input_tokens??value.estimated_tokens,压缩代次:value.compaction_generation,消息数量:value.message_count,记忆数量:Array.isArray(value.memory_ids)?value.memory_ids.length:undefined};const present=Object.fromEntries(Object.entries(summary).filter(([,item])=>item!==undefined&&item!==null));return Object.keys(present).length?<JsonBlock label="上下文统计" value={present}/>:<p className="agent-detail-note">本次上下文已按版本快照固定，内部引用仅保存在数据库中。</p>}

function EventDetail({event,modelRequest}:{event:AgentEvent;modelRequest?:AgentEvent}){
  const payload=event.payload||{};
  if(event.type==="RUN_CREATED")return <div className="agent-event-detail"><DetailTitle label="执行请求" meta={String(payload.trigger_type||"api")}/>{payload.description?<p className="agent-detail-note">{String(payload.description)}</p>:null}<JsonBlock label="原始输入" value={payload.input??{}}/></div>;
  if(event.type==="RUN_CLAIMED")return <PlainDetail title="调度完成">调度器已分配 Worker，并获得本次执行的独占租约。</PlainDetail>;
  if(event.type==="MODEL_RESOLVED")return <div className="agent-event-detail"><DetailTitle label="模型发现结果已冻结" meta={selectionLabel(String(payload.selection_policy||"pinned"))}/><div className="agent-failure-grid"><Info label="实际模型" value={String(payload.model_id||"未记录")}/><Info label="服务" value={String(payload.service_ref||"未记录")}/><Info label="Provider" value={String(payload.provider||"未记录")}/><Info label="上下文窗口" value={payload.context_window_tokens?`${payload.context_window_tokens} Token`:"服务未声明"}/><Info label="版本" value={String(payload.model_version||"上游未报告")}/><Info label="Artifact Digest" value={shortDigest(String(payload.artifact_digest||""))||"上游未报告"}/></div><p className="agent-detail-note">该解析结果属于当前 Run；Worker 恢复时会校验服务配置哈希并继续使用同一模型。实际消息预算还会扣除输出预留、Tool Schema 和模板安全余量。</p></div>;
  if(event.type==="IDENTITY_COMPILED")return <div className="agent-event-detail"><DetailTitle label="结构化角色身份已编译" meta={shortDigest(String(payload.digest||""))}/><div className="agent-failure-grid"><Info label="显示名称" value={String(payload.identity?.display_name||"未填写")}/><Info label="角色" value={String(payload.identity?.role||"未填写")}/><Info label="目标" value={String(payload.identity?.goal||"未填写")}/><Info label="沟通风格" value={String(payload.identity?.communication_style||"未填写")}/></div><JsonBlock label="责任与边界" value={{responsibilities:payload.identity?.responsibilities||[],boundaries:payload.identity?.boundaries||[]}}/><p className="agent-detail-note">该身份位于用户 Prompt 和 Skill 之前的不可裁剪系统上下文，并通过 Digest 固定到本次 Run。</p></div>;
  if(event.type==="SKILL_ACTIVATED")return <div className="agent-event-detail"><DetailTitle label="注入 Skill"/>{Array.isArray(payload.skills)&&payload.skills.length?<div className="agent-skill-list">{payload.skills.map((item:any,index:number)=><section key={`${item.key||item.name}-${index}`}><strong>{item.name||item.key||"Skill"} · v{item.version||"—"}</strong>{item.instructions?.map((instruction:string,i:number)=><pre key={i}>{instruction}</pre>)}</section>)}</div>:<p className="agent-detail-note">已注入该 Agent 版本固定的 Skill。此旧事件未记录可读名称，内部版本引用已保存在数据库。</p>}</div>;
  if(event.type==="MEMORY_RETRIEVED")return <div className="agent-event-detail"><DetailTitle label="召回分层记忆" meta={`${payload.count||0} 条`}/><p className="agent-detail-note">仅召回当前租户、Agent、用户与 Session 有权读取且未过期的记忆；内容按独立 token 预算注入，并被标记为事实而非指令。</p><div className="agent-recalled-memory-list">{Array.isArray(payload.memories)?payload.memories.map((memory:any)=><section key={memory.id}><strong>{memoryScopeLabel(String(memory.scope))} · {memoryKindLabel(String(memory.kind))}</strong><p>{String(memory.content||"")}</p><small>召回分数 {Number(memory.score||0).toFixed(3)}</small></section>):null}</div></div>;
  if(event.type==="TURN_STARTED")return <PlainDetail title="开始处理">开始处理当前用户请求，并建立本轮对话上下文。</PlainDetail>;
  if(event.type==="STEP_STARTED")return <PlainDetail title={`进入 ReAct 第 ${event.step||1} 步`}>准备构建上下文并决定调用模型或工具。</PlainDetail>;
  if(event.type==="EXECUTION_MODE_SELECTED")return <div className="agent-event-detail"><DetailTitle label={payload.mode==="planned"?"已进入规划执行":"已采用直接回答"} meta={planningPolicyLabel(String(payload.policy||"auto"))}/><p className="agent-detail-note">{payload.mode==="planned"?"模型已创建持久化 Plan；tool_hints 用于表达当前 Todo 意图，Runtime 会投影必要的读取、修复、执行与依赖申请工具，实际权限仍由 AgentVersion、审批策略和 Sandbox 约束。":"本次请求无需实质工具，运行时允许模型直接生成用户可见结果，不创建空 Plan。"}</p><JsonBlock label="判定依据" value={{source:payload.source||"runtime",reason:payload.reason||"未记录"}}/></div>;
  if(event.type==="PLAN_CREATED"||event.type==="PLAN_UPDATED")return <div className="agent-event-detail"><DetailTitle label={event.type==="PLAN_CREATED"?"已建立执行计划":"已更新执行计划"} meta={`revision ${payload.revision||1}`}/><p className="agent-detail-note">{String(payload.goal||"")}</p><JsonBlock label="计划步骤" value={payload.steps||[]}/>{payload.explanation?<p className="agent-detail-note">{String(payload.explanation)}</p>:null}</div>;
  if(event.type.startsWith("VERIFICATION_"))return <div className="agent-event-detail"><DetailTitle label={eventLabel(event.type)} meta={String(payload.provider_key||payload.verdict||payload.action||payload.reason_code||"")}/><div className="agent-failure-grid"><Info label="Plan Step" value={String(payload.plan_step_key||payload.step_id||"未记录")}/><Info label="验收项" value={String(payload.criterion_key||payload.criterion_id||"未记录")}/><Info label="Intent" value={shortID(String(payload.intent_id||""))||"未生成"}/><Info label="Spec" value={shortID(String(payload.spec_id||""))||"未生成"}/><Info label="Attempt" value={shortID(String(payload.attempt_id||""))||"未执行"}/><Info label="Evidence" value={shortID(String(payload.evidence_id||""))||"未生成"}/></div>{event.type==="VERIFICATION_LOOP_DETECTED"?<p className="agent-diagnosis-tip"><strong>熔断原因：</strong>模型连续两次忽略相同恢复契约；Runtime 已提前终止，避免耗尽整个模型调用预算。</p>:null}<JsonBlock label="验证事实" value={payload}/></div>;
  if(event.type==="PLAN_COMPLETION_BLOCKED"){
    const required=Array.isArray(payload.required_tools)?payload.required_tools:[];
    const declared=Array.isArray(payload.declared_tools)?payload.declared_tools:[];
    const injected=Array.isArray(payload.auto_injected_tools)?payload.auto_injected_tools:[];
    const available=Array.isArray(payload.available_tools)?payload.available_tools:[];
    return <div className="agent-event-detail"><DetailTitle label="运行时阻止提前结束" meta={String(payload.reason||"plan_guard")}/>{payload.criterion?<><p className="agent-detail-note"><strong>{String(payload.criterion)}</strong></p><div className="agent-failure-grid"><Info label="Plan Step" value={String(payload.step_id||"未记录")}/><Info label="验收条件" value={String(payload.criterion_id||"未记录")}/><Info label="验证类型" value={verificationLabel(String(payload.verification_kind||""))}/><Info label="目标" value={String(payload.target||"未记录")}/><Info label="必需工具" value={required.join("、")||"未识别"}/><Info label="Plan 声明工具" value={declared.join("、")||"未声明"}/><Info label="Runtime 有效工具" value={available.join("、")||"未记录"}/></div>{injected.length?<p className="agent-diagnosis-tip"><strong>Runtime 已自动补齐：</strong>{injected.join("、")}</p>:null}<p className="agent-diagnosis-tip"><strong>证据说明：</strong>工具回执由平台自动记录和绑定，模型不需要也不能填写回执 ID。</p><p className="agent-diagnosis-tip"><strong>下一步：</strong>{String(payload.required_action||"重新取得可验证的工具回执")}</p></>:<p className="agent-detail-note">{String(payload.required_action||"Plan 仍包含未完成工作，Runtime 已要求模型继续执行。")}</p>}<JsonBlock label="返回给模型的恢复指令" value={payload.instruction||"未记录"}/></div>;
  }
  if(event.type==="FINAL_OUTPUT_REJECTED")return <div className="agent-event-detail"><DetailTitle label="最终结果完整性校验未通过"/><p className="agent-detail-note">模型返回了上下文摘要、运行时控制块、私有推理标记或空结果。本次内容不会成为最终回答，执行内核已保存检查点并要求模型继续生成可交付结果。</p></div>;
  if(event.type==="USER_INPUT_REQUESTED")return <div className="agent-event-detail"><DetailTitle label="等待用户补充信息"/><p className="agent-detail-note">{String(payload.question||"")}</p>{payload.context?<p className="agent-detail-note">{String(payload.context)}</p>:null}{payload.options?<JsonBlock label="可选回答" value={payload.options}/>:null}</div>;
  if(event.type==="USER_INPUT_RECEIVED")return <div className="agent-event-detail"><DetailTitle label="已收到回答，任务将从检查点续跑"/><p className="agent-detail-note">{String(payload.answer||"")}</p></div>;
  if(event.type==="CONTEXT_COMPACTED"){
    const removedMessages=Number(payload.removed_messages||0);
    const readTimeProjection=payload.mode==="read_time_projection";
    const legacyProjection=removedMessages===0;
    return <div className="agent-event-detail"><DetailTitle label={readTimeProjection?"已生成 Context Collapse 视图":legacyProjection?"执行状态已投影（未压缩上下文）":"上下文已主动压缩"} meta={readTimeProjection||!legacyProjection?`generation ${payload.generation||1}`:"历史兼容事件"}/><div className="agent-failure-grid"><Info label="模型窗口" value={payload.context_window_tokens?`${payload.context_window_tokens} Token`:"未记录"}/><Info label="消息预算" value={payload.message_budget_tokens?`${payload.message_budget_tokens} Token`:"未记录"}/><Info label="Tool Schema" value={payload.tool_schema_tokens?`${payload.tool_schema_tokens} Token`:"未记录"}/><Info label="输出预留" value={payload.reserve_output_tokens?`${payload.reserve_output_tokens} Token`:"未记录"}/><Info label="摘要模式" value={String(payload.summary_mode||"—")}/><Info label="近期轮次" value={payload.recent_turn_tokens?`${payload.recent_turn_tokens} Token`:"未记录"}/><Info label="Tool 结果" value={payload.tool_result_tokens?`${payload.tool_result_tokens} Token`:"未记录"}/><Info label={legacyProjection?"投影前":"压缩前"} value={`${payload.before_tokens||0} Token`}/><Info label={legacyProjection?"投影后":"压缩后"} value={`${payload.after_tokens||0} Token`}/><Info label="移除历史" value={`${removedMessages} 条消息`}/></div><p className="agent-detail-note">{readTimeProjection?"这是模型可见的读时投影；完整 durable history 仍保存在 Checkpoint 中。":legacyProjection?"这是旧 Worker 将完整 Execution Ledger 的模型可见投影误记为压缩的兼容事件；没有历史消息被移除，完整执行状态仍保存在检查点中。":"系统身份与近期完整工具交互被保留，更早历史转换为有界摘要后继续执行。"}</p></div>;
  }
  if(event.type==="CONTEXT_BUILT")return <div className="agent-event-detail"><DetailTitle label="上下文构建完成"/><ContextSummary value={payload}/></div>;
  if(event.type==="MODEL_REQUESTED")return <div className="agent-event-detail"><DetailTitle label="发送给模型" meta={`${payload.model_id||"未记录"} · ${selectionLabel(String(payload.selection_policy||"pinned"))} · ${payload.messages?.length||0} 条消息 · ${payload.tools?.length||0} 个工具`}/><div className="agent-failure-grid"><Info label="上下文窗口" value={payload.context_window_tokens?`${payload.context_window_tokens} Token`:"未记录"}/><Info label="消息预算" value={payload.message_budget_tokens?`${payload.message_budget_tokens} Token`:"未记录"}/><Info label="Tool Schema 预算" value={payload.tool_schema_tokens?`${payload.tool_schema_tokens} Token`:"未记录"}/><Info label="最大输出" value={`${payload.max_tokens||0} Token`}/></div><MessageList messages={payload.messages||[]}/>{payload.tools?.length?<JsonBlock label="可用 Tool Schema" value={payload.tools}/>:null}</div>;
  if(event.type==="MODEL_COMPLETED")return <div className="agent-event-detail"><DetailTitle label="模型回复" meta={`${formatMS(Number(payload.latency_ms)||0)} · 输入 ${payload.usage?.input_tokens||0} / 输出 ${payload.usage?.output_tokens||0} Token`}/><MessageList messages={payload.message?[payload.message]:[]}/><ContextSummary value={payload.context_manifest}/></div>;
  if(event.type==="MODEL_FAILED"){
    const request=modelRequest?.payload||{};
    const kind=String(payload.error_kind||inferErrorKind(String(payload.error||"")));
    return <div className="agent-event-detail"><DetailTitle label="模型调用失败" meta={formatMS(Number(payload.latency_ms)||0)}/><div className="agent-failure-grid"><Info label="错误类型" value={errorKindLabel(kind)}/><Info label="失败阶段" value="连接并请求模型"/><Info label="模型" value={String(payload.model_id||request.model_id||"未记录")}/><Info label="模型服务" value={String(payload.service||request.service||"未记录")}/><Info label="Provider" value={String(payload.provider||request.provider||"未记录")}/><Info label="超时限制" value={formatMS(Number(payload.timeout_ms||request.timeout_ms)||0)}/></div><pre className="agent-detail-error">{String(payload.error||"未知错误")}</pre><p className="agent-diagnosis-tip"><strong>排障建议：</strong>{errorSuggestion(kind)}</p>{request.messages?.length?<><DetailTitle label="本次失败请求的完整模型输入" meta={`${request.messages.length} 条消息`}/><MessageList messages={request.messages}/></>:<p className="agent-detail-note">该旧事件未保存请求消息，请展开前一个“请求模型”节点查看。</p>}{request.tools?.length?<JsonBlock label="本次可用 Tool Schema" value={request.tools}/>:null}<JsonBlock label="请求参数" value={{max_tokens:request.max_tokens,temperature:request.temperature,timeout_ms:payload.timeout_ms||request.timeout_ms}}/><ContextSummary value={payload.context_manifest}/></div>;
  }
  if(event.type==="TOOL_CALLED")return <div className="agent-event-detail"><DetailTitle label={`Tool 输入 · ${payload.name||"未命名工具"}`}/><JsonBlock label="原始参数" value={payload.arguments??{}}/></div>;
  if(event.type==="TOOL_COMPLETED"||event.type==="TOOL_FAILED"){
    const result=payload.result||{}; const failureCode=String(payload.error_code||result.error_code||result.meta?.error_code||"");
    return <div className="agent-event-detail"><DetailTitle label={`Tool ${event.type==="TOOL_FAILED"?"失败":"输出"} · ${payload.name||"未命名工具"}`} meta={formatMS(Number(payload.latency_ms)||0)}/>{event.type==="TOOL_FAILED"?<div className="agent-failure-grid"><Info label="错误类型" value={failureCode||"TOOL_EXECUTION_FAILED"}/><Info label="可重试" value={String(payload.retryable??result.retryable??"未记录")}/><Info label="失败阶段" value={failureCode.startsWith("SANDBOX")?"Sandbox 策略":"工具执行"}/></div>:null}{payload.correction||result.correction?<p className="agent-diagnosis-tip"><strong>平台修复建议：</strong>{String(payload.correction||result.correction)}</p>:null}<JsonBlock label="原始参数" value={payload.arguments??{}}/><JsonBlock label={result?.meta?.content_artifact_id?"模型结果（完整内容已 Artifact 化）":"原始结果"} value={result}/></div>;
  }
  if(event.type==="STEP_COMPLETED")return <PlainDetail title="步骤完成" meta={formatMS(Number(payload.latency_ms)||0)}>当前步骤状态：{String(payload.status||"completed")}。</PlainDetail>;
  if(event.type==="STEP_FAILED")return <PlainDetail title="步骤失败" meta={formatMS(Number(payload.latency_ms)||0)}>当前步骤未完成，运行器将依据错误状态决定重试、暂停或终止。{payload.error?` ${String(payload.error)}`:""}</PlainDetail>;
  if(event.type==="TURN_COMPLETED")return <PlainDetail title="轮次完成">本轮模型与工具处理已经结束。</PlainDetail>;
  if(event.type==="CHECKPOINT_CREATED")return <PlainDetail title="检查点已保存">当前执行状态已持久化，可用于故障恢复和继续执行。</PlainDetail>;
  if(event.type==="RUN_COMPLETED")return <div className="agent-event-detail"><DetailTitle label="最终输出"/><JsonBlock label="原始结果" value={payload.output??payload}/></div>;
  if(event.type==="RUN_FAILED")return <div className="agent-event-detail"><DetailTitle label="执行失败"/><pre className="agent-detail-error">{String(payload.error||payload.message||"Agent 执行失败")}</pre></div>;
  const visible=filteredPayload(payload);
  return <div className="agent-event-detail">{hasData(visible)?<JsonBlock label="业务数据" value={visible}/>:<p className="agent-detail-note">该节点用于记录执行状态；内部标识已保存在数据库，不在界面展示。</p>}</div>;
}
function DetailTitle({label,meta}:{label:string;meta?:string}){return <header><strong>{label}</strong>{meta?<span>{meta}</span>:null}</header>}
function Info({label,value}:{label:string;value:string}){return <span><small>{label}</small><strong>{value}</strong></span>}
function inferErrorKind(message:string){const value=message.toLowerCase();if(value.includes("maximum context length")||value.includes("context length")||value.includes("input_tokens"))return "context_window_exceeded";if(value.includes("connection refused"))return "connection_refused";if(value.includes("timeout")||value.includes("deadline exceeded"))return "timeout";if(value.includes("401")||value.includes("unauthorized"))return "authentication";if(value.includes("404")||value.includes("not found"))return "endpoint_or_model_not_found";return "provider_error"}
function errorKindLabel(kind:string){return ({context_window_exceeded:"上下文窗口超限",connection_refused:"连接被拒绝",timeout:"请求超时",authentication:"认证失败",endpoint_or_model_not_found:"端点或模型不存在",cancelled:"请求被取消",provider_error:"模型服务错误"} as Record<string,string>)[kind]||kind}
function errorSuggestion(kind:string){return ({context_window_exceeded:"检查运行轨迹中的模型窗口、消息预算、Tool Schema 预算和输出预留。新 Run 会按服务声明的真实窗口提前压缩；旧 Run 的不可变版本配置不会被静默改写。",connection_refused:"模型进程没有监听目标端口。检查推理服务是否启动、端口映射是否正确，以及容器能否访问宿主机。",timeout:"模型在超时限制内没有响应。检查模型负载、显存、排队请求和超时配置。",authentication:"检查模型服务的 API Key 环境变量和鉴权配置。",endpoint_or_model_not_found:"检查 Base URL、/v1/chat/completions 路径以及模型名称是否与服务注册一致。",cancelled:"检查 Run 是否被取消、Worker 是否重启或上游请求是否提前断开。",provider_error:"查看模型服务日志和 HTTP 状态码，并核对请求参数是否受当前模型支持。"} as Record<string,string>)[kind]||"查看模型服务日志和请求参数。"}
function PlainDetail({title,meta,children}:{title:string;meta?:string;children:ReactNode}){return <div className="agent-event-detail"><DetailTitle label={title} meta={meta}/><p className="agent-detail-note">{children}</p></div>}
function MessageList({messages}:{messages:AgentMessage[]}){return <div className="agent-message-list">{messages.map((message,index)=><section key={`${message.role}-${index}`}><b>{message.role}{message.name?` · ${message.name}`:""}</b><pre>{messageContent(message)||"（无文本内容）"}</pre>{message.tool_calls?.length?<JsonBlock label="Tool Calls" value={message.tool_calls.map(call=>({name:call.name,arguments:call.arguments}))}/>:null}</section>)}</div>}
function JsonBlock({label,value}:{label:string;value:unknown}){if(value===undefined||value===null)return null;return <details className="agent-json-block" open><summary>{label}</summary><pre>{pretty(value)}</pre></details>}
