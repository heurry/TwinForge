// 统一 API 客户端
//
// 设计目标（企业级横切能力 §8.1）：
// - baseURL 走环境变量 VITE_API_BASE，支持 dev/staging/prod 多环境
// - 每个请求注入 x-request-id（trace id），便于和后端日志/审计串联
// - 统一超时（AbortController），避免请求悬挂
// - 类型化错误 ApiError（带 status / requestId / body），上层可按状态码分支
// - 401 统一交给可注册的处理钩子（后续接入登录态）

import type { K8sHPA, ScaleDeploymentInput, UpsertHpaInput } from "../types/platform";
import type { ModelRegistryList, RegisterModelInput, RegisteredModelVersion } from "../types/registry";
import type { ArchiveManifest, ArchiveRunResult, StorageTiers } from "../types/storage";
import type { CallGraph } from "../types/topology";
import type { RoutingPolicy, RoutingPolicyDetail, RoutingPolicyList, RoutingResourceTransition, RoutingStats, SavePolicyInput } from "../types/routing";
import type { DocSignalList, FeedbackDataset, RagEvalHistory, RagEvalResult } from "../types/feedback";
import type { SubmitTrainingInput, TrainingJobsList, TrainingKubernetesDetail, TrainingLogs } from "../types/training";
import type { A2AStreamEvent, A2ATask, AgentApproval, AgentArtifact, AgentArtifactPromotion, AgentDefinition, AgentDependencyInstall, AgentEnvironmentTemplate, AgentEvent, AgentExecutableVersion, AgentMemory, AgentMemoryLifecycleEvent, AgentMemoryRetrievalRecord, AgentMemoryRevision, AgentMemorySource, AgentMemoryTimelineEntry, AgentObservabilitySummary, AgentPlatformCapabilities, AgentPromptVersion, AgentRun, AgentRunManifest, AgentScore, AgentSession, AgentSessionAudit, AgentSkillSetVersion, AgentSkillVersion, AgentStorageSummary, AgentTaskPlan, AgentToolSetVersion, AgentToolVersion, AgentTrajectoryPage, AgentUserQuestion, AgentVerificationRecord, AgentVersion, AgentVersionRef, AgentVersionSpecInput, AgentWorkflow, MCPServerVersion } from "../types/agent";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");
const DEFAULT_TIMEOUT_MS = 30_000;

export class ApiError extends Error {
  readonly status: number;
  readonly requestId: string;
  readonly body: string;

  constructor(message: string, options: { status: number; requestId: string; body: string }) {
    super(message);
    this.name = "ApiError";
    this.status = options.status;
    this.requestId = options.requestId;
    this.body = options.body;
  }
}

// 401 处理钩子：登录态接入后由 auth 模块注册（如跳转登录页）。
let unauthorizedHandler: (() => void) | null = null;
export function setUnauthorizedHandler(handler: (() => void) | null): void {
  unauthorizedHandler = handler;
}

// D2：JWT 令牌（认证开启时由登录写入；持久化到 localStorage，刷新不丢）。
const AUTH_TOKEN_KEY = "cip_auth_token";
let authToken: string | null = typeof localStorage !== "undefined" ? localStorage.getItem(AUTH_TOKEN_KEY) : null;
export function setAuthToken(token: string | null): void {
  authToken = token;
  if (typeof localStorage !== "undefined") {
    if (token) localStorage.setItem(AUTH_TOKEN_KEY, token);
    else localStorage.removeItem(AUTH_TOKEN_KEY);
  }
}
export function getAuthToken(): string | null {
  return authToken;
}
function authHeaders(): Record<string, string> {
  return authToken ? { Authorization: `Bearer ${authToken}` } : {};
}

function makeRequestId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `req-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export interface ApiOptions extends Omit<RequestInit, "headers"> {
  headers?: Record<string, string>;
  /** 请求超时（毫秒），默认 30s。传入 0 关闭超时。 */
  timeoutMs?: number;
}

export async function api<T>(path: string, init?: ApiOptions): Promise<T> {
  const requestId = makeRequestId();
  const timeoutMs = init?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const url = path.startsWith("http") ? path : `${API_BASE}${path}`;

  // 外部可传入自己的 signal；同时叠加超时 signal。
  const controller = new AbortController();
  const timer = timeoutMs > 0 ? window.setTimeout(() => controller.abort(), timeoutMs) : undefined;
  if (init?.signal) {
    init.signal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  let response: Response;
  try {
    const contentHeaders: Record<string, string> = init?.body instanceof FormData ? {} : { "Content-Type": "application/json" };
    response = await fetch(url, {
      ...init,
      signal: controller.signal,
      headers: {
        ...contentHeaders,
        "x-request-id": requestId,
        ...authHeaders(),
        ...(init?.headers ?? {})
      }
    });
  } catch (error) {
    if (timer) window.clearTimeout(timer);
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError(`请求超时（${timeoutMs}ms）`, { status: 0, requestId, body: "" });
    }
    throw new ApiError(error instanceof Error ? error.message : "网络请求失败", {
      status: 0,
      requestId,
      body: ""
    });
  } finally {
    if (timer) window.clearTimeout(timer);
  }

  if (response.status === 401) {
    unauthorizedHandler?.();
  }

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new ApiError(body || response.statusText || `请求失败（${response.status}）`, {
      status: response.status,
      requestId: response.headers.get("x-request-id") || requestId,
      body
    });
  }

  // 204 / 空响应体安全返回
  if (response.status === 204) {
    return undefined as T;
  }
  const text = await response.text();
  if (!text) return undefined as T;
  return JSON.parse(text) as T;
}

/** Normalize nullable list payloads at the transport boundary. Go encodes a nil
 * slice as null; UI callers must always receive an array. */
export function normalizeListResponse<T>(value: { data?: T[] | null } | null | undefined): { data: T[] } {
  return { data: Array.isArray(value?.data) ? value.data : [] };
}

async function apiList<T>(path: string, init?: ApiOptions): Promise<{ data: T[] }> {
  return normalizeListResponse<T>(await api<{ data?: T[] | null }>(path, init));
}

const AGENT_TENANT = import.meta.env.VITE_AGENT_TENANT ?? "demo";
export function listAgentRuns(limit = 50): Promise<{ data: AgentRun[] }> {
  return apiList(`/agent-api/api/v1/runs?limit=${limit}`, { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function listAgentWorkflows(sessionID?: string, limit = 50): Promise<{ data: AgentWorkflow[] }> {
  const query = new URLSearchParams({ limit: String(limit) });
  if (sessionID) query.set("session_id", sessionID);
  return apiList(`/agent-api/api/v1/workflows?${query.toString()}`, { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function listAgentWorkflowRuns(workflowID: string, limit = 100): Promise<{ data: AgentRun[] }> {
  return apiList(`/agent-api/api/v1/workflows/${encodeURIComponent(workflowID)}/runs?limit=${limit}`, { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function getAgentRunEvents(runID: string): Promise<{ data: AgentEvent[] }> {
  return apiList(`/agent-api/api/v1/runs/${encodeURIComponent(runID)}/events?limit=1000`, { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function getAgentRunTrajectory(runID: string, after = 0, limit = 500): Promise<{ data: AgentTrajectoryPage }> {
	return api<{ data?: AgentTrajectoryPage | null }>(`/agent-api/api/v1/runs/${encodeURIComponent(runID)}/trajectory?after=${after}&limit=${limit}`, { headers: { "X-Tenant-ID": AGENT_TENANT } }).then((response) => ({
		data: {
			records: Array.isArray(response?.data?.records) ? response.data.records : [],
			next_cursor: response?.data?.next_cursor || 0,
			has_more: Boolean(response?.data?.has_more),
			total: Number(response?.data?.total || 0),
		},
	}));
}
export function listAgentRunScores(runID: string): Promise<{ data: AgentScore[] }> {
  return apiList(`/agent-api/api/v1/runs/${encodeURIComponent(runID)}/scores?limit=500`, { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function listAgentRunArtifacts(runID: string): Promise<{ data: AgentArtifact[] }> {
  return apiList(`/agent-api/api/v1/runs/${encodeURIComponent(runID)}/artifacts?limit=500`, { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function getAgentRunManifest(runID: string): Promise<{ data: AgentRunManifest }> {
  return api(`/agent-api/api/v1/runs/${encodeURIComponent(runID)}/manifest`, { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function promoteAgentArtifact(artifactID: string, targetPath: string, expectedTargetSHA256 = ""): Promise<{ data: AgentArtifactPromotion }> {
  return api(`/agent-api/api/v1/artifacts/${encodeURIComponent(artifactID)}:promote`, {
    method: "POST",
    headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" },
    body: JSON.stringify({ target_path: targetPath, ...(expectedTargetSHA256 ? { expected_target_sha256: expectedTargetSHA256 } : {}) })
  });
}
export function listAgentRunChildren(runID: string): Promise<{ data: AgentRun[] }> {
  return apiList(`/agent-api/api/v1/runs/${encodeURIComponent(runID)}/children`, { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function getAgentRunPlan(runID: string): Promise<{ data: AgentTaskPlan | null }> {
  return api(`/agent-api/api/v1/runs/${encodeURIComponent(runID)}/plan`, { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function listAgentRunVerifications(runID: string): Promise<{ data: AgentVerificationRecord[] }> {
  return apiList(`/agent-api/api/v1/runs/${encodeURIComponent(runID)}/verifications`, { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function getAgentRunQuestion(runID: string): Promise<{ data: AgentUserQuestion | null }> {
  return api(`/agent-api/api/v1/runs/${encodeURIComponent(runID)}/question`, { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function answerAgentRunQuestion(questionID: string, answer: string): Promise<{ data: AgentUserQuestion }> {
  return api(`/agent-api/api/v1/questions/${encodeURIComponent(questionID)}:answer`, { method: "POST", headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" }, body: JSON.stringify({ answer }) });
}
export function cancelAgentRun(runID: string): Promise<void> {
  return api(`/agent-api/api/v1/runs/${encodeURIComponent(runID)}:cancel`, { method: "POST", headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" } });
}
export function listAgentApprovals(status = ""): Promise<{ data: AgentApproval[] }> {
  const query = new URLSearchParams({ limit: "500" });
  if (status) query.set("status", status);
  return api(`/agent-api/api/v1/approvals?${query}`, { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function listAgentEnvironmentTemplates(): Promise<{ data: AgentEnvironmentTemplate[] }> {
  return apiList("/agent-api/api/v1/environment-templates", { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function listAgentDependencyInstalls(runID = ""): Promise<{ data: AgentDependencyInstall[] }> {
  const query = new URLSearchParams({ limit: "200" });
  if (runID) query.set("run_id", runID);
  return apiList(`/agent-api/api/v1/dependency-installs?${query}`, { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function decideAgentApproval(approvalID: string, approved: boolean, reason: string): Promise<{ data: AgentApproval }> {
  return api(`/agent-api/api/v1/approvals/${encodeURIComponent(approvalID)}:${approved ? "approve" : "reject"}`, { method: "POST", headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" }, body: JSON.stringify({ reason }) });
}
export async function getAgentArtifactContent(artifactID: string): Promise<Blob> {
  const response = await fetch(`${API_BASE}/agent-api/api/v1/artifacts/${encodeURIComponent(artifactID)}/content`, { headers: { ...authHeaders(), "X-Tenant-ID": AGENT_TENANT, "x-request-id": makeRequestId() } });
  if (!response.ok) throw new ApiError(await response.text().catch(() => "读取 Artifact 失败"), { status: response.status, requestId: response.headers.get("x-request-id") || "", body: "" });
  return response.blob();
}
export function getAgentObservabilitySummary(): Promise<{ data: AgentObservabilitySummary }> {
  return api("/agent-api/api/v1/observability/summary", { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function listExecutableAgentVersions(): Promise<{ data: AgentExecutableVersion[] }> {
  return api("/agent-api/api/v1/agent-versions?limit=100", { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function getAgentPlatformCapabilities(): Promise<{ data: AgentPlatformCapabilities }> {
  return api("/agent-api/api/v1/capabilities", { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function listAgentDefinitions(): Promise<{ data: AgentDefinition[] }> {
  return apiList("/agent-api/api/v1/agents?limit=200", { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function createAgentDefinition(input: { key: string; name: string; description?: string; owner?: string }): Promise<{ data: AgentDefinition }> {
  return api("/agent-api/api/v1/agents", {
    method: "POST",
    headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" },
    body: JSON.stringify(input)
  });
}
export function listAgentVersions(agentID: string): Promise<{ data: AgentVersion[] }> {
  return api(`/agent-api/api/v1/agents/${encodeURIComponent(agentID)}/versions`, { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function createAgentVersion(agentID: string, spec: AgentVersionSpecInput): Promise<{ data: AgentVersion }> {
  return api(`/agent-api/api/v1/agents/${encodeURIComponent(agentID)}/versions`, {
    method: "POST",
    headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" },
    body: JSON.stringify({ spec })
  });
}
export function releaseAgentVersion(versionID: string): Promise<{ data: AgentVersion }> {
  return api(`/agent-api/api/v1/agent-versions/${encodeURIComponent(versionID)}:release`, {
    method: "POST",
    headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" }
  });
}
export function listAgentPromptVersions(): Promise<{ data: AgentPromptVersion[] }> {
  return apiList("/agent-api/api/v1/prompt-versions?limit=500", { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function createAgentPromptVersion(input: { key: string; name: string; content: string }): Promise<{ data: AgentPromptVersion }> {
  return api("/agent-api/api/v1/prompt-versions", {
    method: "POST",
    headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" },
    body: JSON.stringify(input)
  });
}
export function listAgentSkillVersions(): Promise<{ data: AgentSkillVersion[] }> {
  return apiList("/agent-api/api/v1/skill-versions?limit=500", { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function uploadAgentSkillVersion(file: File): Promise<{ data: AgentSkillVersion }> {
  const form = new FormData();
  form.append("file", file);
  return api("/agent-api/api/v1/skill-versions:upload", {
    method: "POST",
    headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" },
    body: form,
    timeoutMs: 60_000
  });
}
export function listAgentSkillSetVersions(): Promise<{ data: AgentSkillSetVersion[] }> {
  return apiList("/agent-api/api/v1/skillset-versions?limit=500", { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function createAgentSkillSetVersion(input: { key: string; name: string; skills: AgentVersionRef[] }): Promise<{ data: AgentSkillSetVersion }> {
  return api("/agent-api/api/v1/skillset-versions", {
    method: "POST",
    headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" },
    body: JSON.stringify({ key: input.key, name: input.name, spec: { skills: input.skills } })
  });
}
export function listAgentToolVersions(): Promise<{ data: AgentToolVersion[] }> {
  return apiList("/agent-api/api/v1/tool-versions?limit=500", { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function createAgentToolVersion(input: { key: string; name: string; spec: Record<string, unknown> }): Promise<{ data: AgentToolVersion }> {
  return api("/agent-api/api/v1/tool-versions", {
    method: "POST",
    headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" },
    body: JSON.stringify(input)
  });
}
export function listAgentToolSetVersions(): Promise<{ data: AgentToolSetVersion[] }> {
  return apiList("/agent-api/api/v1/toolset-versions?limit=500", { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function createAgentToolSetVersion(input: { key: string; name: string; tools: AgentVersionRef[] }): Promise<{ data: AgentToolSetVersion }> {
  return api("/agent-api/api/v1/toolset-versions", {
    method: "POST",
    headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" },
    body: JSON.stringify({ key: input.key, name: input.name, spec: { tools: input.tools } })
  });
}
export function listMCPServerVersions(): Promise<{ data: MCPServerVersion[] }> {
  return apiList("/agent-api/api/v1/mcp-servers?limit=200", { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function createMCPServerVersion(input: { key: string; name: string; endpoint: string; header_environment?: Record<string, string> }): Promise<{ data: MCPServerVersion }> {
  return api("/agent-api/api/v1/mcp-servers", { method: "POST", headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" }, body: JSON.stringify({ key: input.key, name: input.name, spec: { transport: "streamable_http", endpoint: input.endpoint, protocol_version: "2025-06-18", header_environment: input.header_environment || {}, timeout: 10_000_000_000, max_response_bytes: 2_097_152 } }) });
}
export function testMCPServerVersion(id: string): Promise<{ data: MCPServerVersion }> {
  return api(`/agent-api/api/v1/mcp-server-versions/${encodeURIComponent(id)}:test`, { method: "POST", headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function syncMCPServerTools(id: string): Promise<{ data: MCPServerVersion }> {
  return api(`/agent-api/api/v1/mcp-server-versions/${encodeURIComponent(id)}:sync-tools`, { method: "POST", headers: { "X-Tenant-ID": AGENT_TENANT }, timeoutMs: 60_000 });
}
export function getA2AAgentCard(agentID: string): Promise<Record<string, unknown>> {
  return api(`/agent-api/api/v1/a2a/agents/${encodeURIComponent(agentID)}/card`, { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function sendA2AMessage(agentID: string, text: string, messageID = makeRequestId()): Promise<{ task: A2ATask }> {
  return api(`/agent-api/api/v1/a2a/agents/${encodeURIComponent(agentID)}/message:send`, {
    method: "POST",
    headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" },
    body: JSON.stringify({ message: { messageId: messageID, role: "user", parts: [{ text }] } }),
    timeoutMs: 60_000
  });
}
export function getA2ATask(taskID: string): Promise<A2ATask> {
  return api(`/agent-api/api/v1/a2a/tasks/${encodeURIComponent(taskID)}`, { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function cancelA2ATask(taskID: string): Promise<A2ATask> {
  return api(`/agent-api/api/v1/a2a/tasks/${encodeURIComponent(taskID)}:cancel`, {
    method: "POST",
    headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" }
  });
}
export async function subscribeA2ATask(taskID: string, onEvent: (event: A2AStreamEvent) => void, signal?: AbortSignal): Promise<void> {
  const response = await fetch(`${API_BASE}/agent-api/api/v1/a2a/tasks/${encodeURIComponent(taskID)}/subscribe`, {
    headers: { Accept: "text/event-stream", "X-Tenant-ID": AGENT_TENANT, "x-request-id": makeRequestId(), ...authHeaders() },
    signal
  });
  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new ApiError(body || "A2A 订阅失败", { status: response.status, requestId: response.headers.get("x-request-id") || "", body });
  }
  if (!response.body) throw new ApiError("浏览器未提供流式响应体", { status: 0, requestId: "", body: "" });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      let event = "message";
      const data: string[] = [];
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      if (data.length) onEvent({ event, data: JSON.parse(data.join("\n")) as Record<string, unknown> });
    }
    if (done) break;
  }
}
export function listAgentSessions(agentID: string, limit = 50): Promise<{ data: AgentSession[] }> {
  const query = new URLSearchParams({ agent_id: agentID, limit: String(limit) });
  return api(`/agent-api/api/v1/sessions?${query}`, { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function getAgentStorageSummary(): Promise<{ data: AgentStorageSummary }> {
  return api("/agent-api/api/v1/storage/summary", { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function createAgentSession(agentID: string, title = "新对话"): Promise<{ data: AgentSession }> {
  return api("/agent-api/api/v1/sessions", {
    method: "POST",
    headers: { "X-Tenant-ID": AGENT_TENANT },
    body: JSON.stringify({ agent_id: agentID, user_id: "web-console", metadata: { title, channel: "web" } })
  });
}
export function listAgentSessionRuns(sessionID: string, limit = 100): Promise<{ data: AgentRun[] }> {
  return api(`/agent-api/api/v1/sessions/${encodeURIComponent(sessionID)}/runs?limit=${limit}`, { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function getAgentSessionAudit(sessionID: string): Promise<{ data: AgentSessionAudit }> {
  return api(`/agent-api/api/v1/sessions/${encodeURIComponent(sessionID)}/audit`, { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function createAgentRun(agentVersionID: string, input: Record<string, unknown>, sessionID?: string, triggerType = "web", workflowID?: string, newWorkflow = false, routingIntent = "auto"): Promise<{ data: AgentRun }> {
  return api("/agent-api/api/v1/runs", {
    method: "POST",
    headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" },
    body: JSON.stringify({ agent_version_id: agentVersionID, session_id: sessionID, workflow_id: workflowID, new_workflow: newWorkflow, routing_intent: routingIntent, trigger_type: triggerType, input })
  });
}
export function listAgentMemories(input: { agentID: string; sessionID?: string; userID?: string }): Promise<{ data: AgentMemory[] }> {
  const query = new URLSearchParams({ agent_id: input.agentID, limit: "100" });
  if (input.sessionID) query.set("session_id", input.sessionID);
  if (input.userID) query.set("user_id", input.userID);
  return api(`/agent-api/api/v1/memories?${query}`, { headers: { "X-Tenant-ID": AGENT_TENANT } });
}
export function createAgentMemory(input: {
  scope: AgentMemory["scope"];
  agent_id?: string;
  user_id?: string;
  session_id?: string;
  kind: AgentMemory["kind"];
  content: string;
  importance: number;
  ttl_seconds?: number;
}): Promise<{ data: AgentMemory }> {
  return api("/agent-api/api/v1/memories", {
    method: "POST",
    headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" },
    body: JSON.stringify(input)
  });
}
export function deleteAgentMemory(memoryID: string): Promise<void> {
  return api(`/agent-api/api/v1/memories/${encodeURIComponent(memoryID)}`, {
    method: "DELETE",
    headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" }
  });
}

export function getAgentMemory(memoryID: string): Promise<{ data: AgentMemory }> {
  return api(`/agent-api/api/v1/memories/${encodeURIComponent(memoryID)}`, { headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" } });
}

export function updateAgentMemory(memoryID: string, input: Record<string, unknown>, idempotencyKey = crypto.randomUUID()): Promise<{ data: AgentMemory }> {
  return api(`/agent-api/api/v1/memories/${encodeURIComponent(memoryID)}`, {
    method: "PATCH",
    headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console", "Idempotency-Key": idempotencyKey },
    body: JSON.stringify(input)
  });
}

export function listAgentMemoryRevisions(memoryID: string, limit = 100): Promise<{ data: AgentMemoryRevision[] }> {
  return api(`/agent-api/api/v1/memories/${encodeURIComponent(memoryID)}/revisions?limit=${limit}`, { headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" } });
}

export function listAgentMemoryLifecycleEvents(memoryID: string, limit = 100): Promise<{ data: AgentMemoryLifecycleEvent[] }> {
  return api(`/agent-api/api/v1/memories/${encodeURIComponent(memoryID)}/events?limit=${limit}`, { headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" } });
}

export function listAgentMemorySources(input: { sourceLayer?: string; projectKey?: string; teamID?: string; limit?: number } = {}): Promise<{ data: AgentMemorySource[] }> {
  const query = new URLSearchParams({ limit: String(input.limit ?? 100) });
  if (input.sourceLayer) query.set("source_layer", input.sourceLayer);
  if (input.projectKey) query.set("project_key", input.projectKey);
  if (input.teamID) query.set("team_id", input.teamID);
  return api(`/agent-api/api/v1/memory-sources?${query}`, { headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" } });
}

export function listAgentMemoryRetrievals(runID: string, limit = 100): Promise<{ data: AgentMemoryRetrievalRecord[] }> {
  return api(`/agent-api/api/v1/runs/${encodeURIComponent(runID)}/memory-retrievals?limit=${limit}`, { headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" } });
}

export function listAgentMemoryTimeline(runID: string, limit = 100): Promise<{ data: AgentMemoryTimelineEntry[] }> {
  return api(`/agent-api/api/v1/runs/${encodeURIComponent(runID)}/memory-timeline?limit=${limit}`, { headers: { "X-Tenant-ID": AGENT_TENANT, "X-Actor-ID": "web-console" } });
}

// ===== A2：K8s 弹性扩缩容写操作 =====
//
// 写操作受后端双重约束：feature flag（ALLOW_K8S_WRITES）+ 命名空间允许名单 + serving 组件硬禁。
// 未开启写时后端返回 403（code: k8s_writes_disabled）；命中保护命名空间返回 403（k8s_namespace_protected）。

/** 手动扩缩 Deployment 副本数；返回写入后的期望副本。 */
export async function scaleK8sDeployment(input: ScaleDeploymentInput): Promise<{ replicas: number }> {
  return api(`/api/kubernetes/deployments/${encodeURIComponent(input.name)}/scale`, {
    method: "POST",
    body: JSON.stringify({ namespace: input.namespace, replicas: input.replicas })
  });
}

/** 创建或更新 HPA（按 CPU 利用率水平伸缩 Deployment）；幂等。 */
export async function upsertK8sHpa(input: UpsertHpaInput): Promise<K8sHPA> {
  return api(`/api/kubernetes/hpa`, { method: "PUT", body: JSON.stringify(input) });
}

/** 删除 HPA；幂等（不存在视为成功）。 */
export async function deleteK8sHpa(namespace: string, name: string): Promise<void> {
  await api(`/api/kubernetes/hpa/${encodeURIComponent(name)}?namespace=${encodeURIComponent(namespace)}`, {
    method: "DELETE"
  });
}

// ===== C1：模型注册中心 =====

export function listModelRegistry(): Promise<ModelRegistryList> {
  return api<ModelRegistryList>("/api/models/registry");
}

// ===== C2：分层存储生命周期 =====
export function storageTiers(): Promise<StorageTiers> {
  return api<StorageTiers>("/api/storage/tiers");
}
export function listArchives(limit = 50): Promise<{ archives: ArchiveManifest[] }> {
  return api<{ archives: ArchiveManifest[] }>(`/api/storage/archives?limit=${limit}`);
}
export function runArchive(): Promise<{ results: ArchiveRunResult[] }> {
  return api("/api/storage/archive", { method: "POST", body: JSON.stringify({}) });
}
export function archiveDownloadURL(id: string): Promise<{ key: string; download_url?: string; note?: string }> {
  return api(`/api/storage/archives/${encodeURIComponent(id)}`);
}

// ===== C3：真实服务拓扑（trace 派生的调用图） =====
export function topologyGraph(): Promise<CallGraph> {
  return api<CallGraph>("/api/topology/graph");
}

// ===== E3：模型路由 / A-B / 影子流量 =====
export function listRoutingPolicies(): Promise<RoutingPolicyList> {
  return api<RoutingPolicyList>("/api/routing/policies");
}
export function getRoutingPolicy(name: string): Promise<RoutingPolicyDetail> {
  return api<RoutingPolicyDetail>(`/api/routing/policies/${encodeURIComponent(name)}`);
}
export function routingPolicyStats(name: string, window = 3600): Promise<RoutingStats> {
  return api<RoutingStats>(`/api/routing/policies/${encodeURIComponent(name)}/stats?window=${window}`);
}
export function createRoutingPolicy(input: SavePolicyInput): Promise<{ policy: RoutingPolicy }> {
  return api("/api/routing/policies", { method: "POST", body: JSON.stringify(input) });
}
export function updateRoutingPolicy(name: string, input: SavePolicyInput): Promise<{ policy: RoutingPolicy }> {
  return api(`/api/routing/policies/${encodeURIComponent(name)}`, { method: "PATCH", body: JSON.stringify(input) });
}
export async function deleteRoutingPolicy(name: string): Promise<void> {
  await api(`/api/routing/policies/${encodeURIComponent(name)}`, { method: "DELETE" });
}
export function promoteRoutingVariant(name: string, label: string): Promise<{ policy: RoutingPolicy; resource_transition?: RoutingResourceTransition }> {
  return api(`/api/routing/policies/${encodeURIComponent(name)}/promote`, { method: "POST", body: JSON.stringify({ label }) });
}
export function rollbackRoutingPolicy(name: string): Promise<{ policy: RoutingPolicy; resource_transition?: RoutingResourceTransition }> {
  return api(`/api/routing/policies/${encodeURIComponent(name)}/rollback`, { method: "POST", body: JSON.stringify({}) });
}

// 路由候选/影子目标的下拉来源：已注册的 serving 实例名（service_instances）。
export async function serviceInstanceNames(): Promise<string[]> {
  const res = await api<{ instances: Array<{ name: string }> }>("/api/service-instances");
  return res.instances.map((i) => i.name);
}

// ===== E2：RAG 评测体系 + 在线反馈回流 =====
export function ragDataset(): Promise<FeedbackDataset> {
  return api<FeedbackDataset>("/api/rag/dataset");
}
export function ragSignal(): Promise<DocSignalList> {
  return api<DocSignalList>("/api/rag/signal");
}
export function runRagEval(): Promise<RagEvalResult> {
  return api<RagEvalResult>("/api/rag/eval", { method: "POST", body: JSON.stringify({}), timeoutMs: 120_000 });
}
export function ragEvalHistory(limit = 20): Promise<RagEvalHistory> {
  return api<RagEvalHistory>(`/api/rag/eval/history?limit=${limit}`);
}

export function registerModelVersion(input: RegisterModelInput): Promise<{ id: string; model_id: string; version: string }> {
  return api("/api/models/registry", { method: "POST", body: JSON.stringify(input) });
}

export function updateModelStatus(id: string, status: string): Promise<RegisteredModelVersion> {
  return api(`/api/models/registry/${encodeURIComponent(id)}/status`, { method: "PATCH", body: JSON.stringify({ status }) });
}

export async function deleteModelVersion(id: string): Promise<void> {
  await api(`/api/models/registry/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export function modelArtifactURL(id: string): Promise<{ download_url?: string; external?: boolean; key?: string; note?: string }> {
  return api(`/api/models/registry/${encodeURIComponent(id)}/artifact`);
}

/** 上传模型产物到 MinIO（multipart）。不能复用 api()——FormData 需浏览器自动设 multipart boundary。 */
export async function uploadModelArtifact(id: string, file: File): Promise<{ artifact_uri: string; size: number }> {
  const requestId = makeRequestId();
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API_BASE}/api/models/registry/${encodeURIComponent(id)}/artifact`, {
    method: "POST",
    body: fd,
    headers: { "x-request-id": requestId, ...authHeaders() }
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(body || res.statusText || `上传失败（${res.status}）`, {
      status: res.status,
      requestId: res.headers.get("x-request-id") || requestId,
      body
    });
  }
  return (await res.json()) as { artifact_uri: string; size: number };
}

// ===== Phase F：分布式训练（Kubeflow PyTorchJob） =====
export function listTrainingJobs(): Promise<TrainingJobsList> {
  return api<TrainingJobsList>("/api/training/jobs");
}
export function submitTrainingJob(
  input: SubmitTrainingInput
): Promise<{ id: string; name: string; namespace: string; status: string; registers_as?: string }> {
  return api("/api/training/jobs", { method: "POST", body: JSON.stringify(input) });
}
export async function cancelTrainingJob(id: string): Promise<void> {
  await api(`/api/training/jobs/${encodeURIComponent(id)}`, { method: "DELETE" });
}
export function trainingJobLogs(id: string): Promise<TrainingLogs> {
  return api<TrainingLogs>(`/api/training/jobs/${encodeURIComponent(id)}/logs`);
}
export function trainingJobKubernetes(id: string): Promise<TrainingKubernetesDetail> {
  return api<TrainingKubernetesDetail>(`/api/training/jobs/${encodeURIComponent(id)}/kubernetes`, { timeoutMs: 12_000 });
}

// ===== SSE 流式对话（AI Copilot） =====
//
// /api/ai/chat:stream 是 Go 单一入口反向代理到 Python AI 服务（FlushInterval=-1 逐块 flush）。
// 上游按 `event: <name>\ndata: <json>\n\n` 推送：start{mode} / token{text} / notice{message}
// / error{error} / done{}。AI 服务不可达时 Go 返回 502 错误信封（非 SSE），这里解析后回调 onError。

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatStreamHandlers {
  onStart?: (mode: string) => void;
  onToken: (text: string) => void;
  onNotice?: (message: string) => void;
  onError?: (message: string) => void;
  onDone?: () => void;
}

export interface ChatStreamOptions {
  signal?: AbortSignal;
  maxTokens?: number;
  temperature?: number;
}

export async function streamAIChat(
  messages: ChatMessage[],
  handlers: ChatStreamHandlers,
  options?: ChatStreamOptions
): Promise<void> {
  const requestId = makeRequestId();
  const url = `${API_BASE}/api/ai/chat:stream`;

  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      signal: options?.signal,
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
        "x-request-id": requestId,
        ...authHeaders()
      },
      body: JSON.stringify({
        messages,
        max_tokens: options?.maxTokens ?? 1024,
        temperature: options?.temperature ?? 0.2
      })
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") return;
    handlers.onError?.(error instanceof Error ? error.message : "网络请求失败");
    handlers.onDone?.();
    return;
  }

  if (response.status === 401) unauthorizedHandler?.();

  if (!response.ok || !response.body) {
    const body = await response.text().catch(() => "");
    let message = body || response.statusText || `请求失败（${response.status}）`;
    try {
      const parsed = JSON.parse(body) as { error?: { message?: string }; message?: string };
      message = parsed?.error?.message || parsed?.message || message;
    } catch {
      /* 非 JSON：保留原文 */
    }
    handlers.onError?.(message);
    handlers.onDone?.();
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;

  try {
    while (!finished) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const { event, data } = parseSSEBlock(block);
        if (!event) continue;
        switch (event) {
          case "start":
            handlers.onStart?.(String((data as { mode?: unknown })?.mode ?? ""));
            break;
          case "token": {
            const text = (data as { text?: unknown })?.text;
            if (typeof text === "string") handlers.onToken(text);
            break;
          }
          case "notice":
            handlers.onNotice?.(String((data as { message?: unknown })?.message ?? ""));
            break;
          case "error":
            handlers.onError?.(String((data as { error?: unknown })?.error ?? "AI 服务错误"));
            break;
          case "done":
            finished = true;
            break;
        }
      }
    }
  } catch (error) {
    if (!(error instanceof DOMException && error.name === "AbortError")) {
      handlers.onError?.(error instanceof Error ? error.message : "流式连接中断");
    }
  } finally {
    handlers.onDone?.();
    try {
      reader.releaseLock();
    } catch {
      /* 已释放 */
    }
  }
}

// ===== SSE 流式客服 RAG 对话（/api/chat/sessions/{id}/messages:stream） =====
//
// Go 原生 RAG 管线逐事件推送（`event: <name>\ndata: <json>\n\n`）：
// retrieval{documents,query,retrieval_ms,memory_turns,request_id} / route{...}
// / token{text} / fallback{reason,error?} / citation{doc_ids} / metrics{ttft_ms,total_ms,target_pod,...} / done{}。
// 会话不存在 / content 为空时后端返回非 SSE 错误信封，这里解析后回调 onError。

export interface ChatCitation {
  doc_id: string;
  title?: string;
  category?: string;
  version?: string;
  score?: number;
}

export interface ChatSessionMetrics {
  request_id?: string;
  retrieval_ms?: number | null;
  ttft_ms?: number | null;
  generation_ms?: number | null;
  total_ms?: number | null;
  target_pod?: string;
  fallback_reason?: string;
  status?: string;
  error?: string;
}

export interface ChatSessionStreamHandlers {
  onRetrieval?: (docs: ChatCitation[], info: { query?: string; retrievalMs?: number; memoryTurns?: number; requestId?: string }) => void;
  onRoute?: (info: { endpointId?: string; selectedEndpointId?: string; routingStrategy?: string }) => void;
  onToken: (text: string) => void;
  onFallback?: (reason: string, error?: string) => void;
  onMetrics?: (metrics: ChatSessionMetrics) => void;
  onError?: (message: string) => void;
  onDone?: () => void;
}

export interface ChatSessionStreamOptions {
  signal?: AbortSignal;
  endpointId?: string;
  maxTokens?: number;
  temperature?: number;
}

export async function streamChatSession(
  sessionId: string,
  content: string,
  handlers: ChatSessionStreamHandlers,
  options?: ChatSessionStreamOptions
): Promise<void> {
  const requestId = makeRequestId();
  const url = `${API_BASE}/api/chat/sessions/${encodeURIComponent(sessionId)}/messages:stream`;

  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      signal: options?.signal,
      headers: { "Content-Type": "application/json", Accept: "text/event-stream", "x-request-id": requestId, ...authHeaders() },
      body: JSON.stringify({
        content,
        endpoint_id: options?.endpointId ?? "",
        max_tokens: options?.maxTokens ?? 1024,
        ...(options?.temperature != null ? { temperature: options.temperature } : {})
      })
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") return;
    handlers.onError?.(error instanceof Error ? error.message : "网络请求失败");
    handlers.onDone?.();
    return;
  }

  if (response.status === 401) unauthorizedHandler?.();

  if (!response.ok || !response.body) {
    const body = await response.text().catch(() => "");
    let message = body || response.statusText || `请求失败（${response.status}）`;
    try {
      const parsed = JSON.parse(body) as { error?: { message?: string }; message?: string };
      message = parsed?.error?.message || parsed?.message || message;
    } catch {
      /* 非 JSON：保留原文 */
    }
    handlers.onError?.(message);
    handlers.onDone?.();
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;

  try {
    while (!finished) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const { event, data } = parseSSEBlock(block);
        if (!event) continue;
        const obj = (data ?? {}) as Record<string, unknown>;
        switch (event) {
          case "retrieval":
            handlers.onRetrieval?.(Array.isArray(obj.documents) ? (obj.documents as ChatCitation[]) : [], {
              query: asStr(obj.query),
              retrievalMs: asNum(obj.retrieval_ms),
              memoryTurns: asNum(obj.memory_turns),
              requestId: asStr(obj.request_id)
            });
            break;
          case "route":
            handlers.onRoute?.({
              endpointId: asStr(obj.endpoint_id),
              selectedEndpointId: asStr(obj.selected_endpoint_id),
              routingStrategy: asStr(obj.routing_strategy)
            });
            break;
          case "token": {
            const text = obj.text;
            if (typeof text === "string") handlers.onToken(text);
            break;
          }
          case "fallback":
            handlers.onFallback?.(asStr(obj.reason) ?? "fallback", asStr(obj.error));
            break;
          case "metrics":
            handlers.onMetrics?.(obj as ChatSessionMetrics);
            break;
          case "citation":
            /* doc_ids 已随 retrieval/metrics 提供，这里忽略以保持事件兼容 */
            break;
          case "error":
            handlers.onError?.(asStr(obj.error) ?? "对话服务错误");
            break;
          case "done":
            finished = true;
            break;
        }
      }
    }
  } catch (error) {
    if (!(error instanceof DOMException && error.name === "AbortError")) {
      handlers.onError?.(error instanceof Error ? error.message : "流式连接中断");
    }
  } finally {
    handlers.onDone?.();
    try {
      reader.releaseLock();
    } catch {
      /* 已释放 */
    }
  }
}

function asStr(v: unknown): string | undefined {
  return typeof v === "string" ? v : undefined;
}
function asNum(v: unknown): number | undefined {
  return typeof v === "number" ? v : undefined;
}

function parseSSEBlock(block: string): { event: string; data: unknown } {
  let event = "";
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice("event:".length).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice("data:".length).trim());
  }
  let data: unknown;
  if (dataLines.length) {
    const joined = dataLines.join("\n");
    try {
      data = JSON.parse(joined);
    } catch {
      data = joined;
    }
  }
  return { event, data };
}
