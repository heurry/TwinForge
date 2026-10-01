import type { AgentEvent } from "../types/agent";

export type AgentActivityKind = "runtime" | "plan" | "model" | "tool" | "context" | "agent" | "input";
export type AgentActivityStatus = "running" | "completed" | "failed" | "waiting" | "cancelled";

export interface AgentActivityItem {
  key: string;
  sequence: number;
  lastSequence: number;
  eventSequences: number[];
  kind: AgentActivityKind;
  status: AgentActivityStatus;
  title: string;
  detail?: string;
  content?: string;
  createdAt: string;
  turn?: number;
  step?: number;
}

// These tools emit richer first-class events; showing both would duplicate one
// logical action in the conversation stream.
const CONTROL_TOOLS = new Set(["update_plan", "update_plan_step", "ask_user", "delegate_agent"]);

/**
 * Build the user-facing execution stream from the immutable Event Ledger.
 * Request/completion event pairs are folded into one row so the conversation
 * shows real progress without dumping transport payloads or private reasoning.
 */
export function buildAgentActivity(events: AgentEvent[]): AgentActivityItem[] {
  const items: AgentActivityItem[] = [];
  const indexes = new Map<string, number>();
  const ordered = [...events].sort((left, right) => left.sequence - right.sequence);
  const blockedModelSteps = new Set(
    ordered
      .filter(event => event.type === "PLAN_COMPLETION_BLOCKED" || event.type === "FINAL_OUTPUT_REJECTED")
      .map(event => eventPositionKey(event))
  );

  const add = (item: AgentActivityItem) => {
    indexes.set(item.key, items.length);
    items.push(item);
  };
  const merge = (key: string, event: AgentEvent, update: Partial<AgentActivityItem>) => {
    const index = indexes.get(key);
    if (index === undefined) return false;
    const current = items[index];
    items[index] = {
      ...current,
      ...update,
      lastSequence: event.sequence,
      eventSequences: [...current.eventSequences, event.sequence]
    };
    return true;
  };

  for (const event of ordered) {
    const payload = event.payload || {};
    const position = eventPosition(event);
    if (event.type === "RUN_CLAIMED") {
      add(activity(event, "runtime", "running", "Worker 已接管任务", "正在准备身份、Skill、记忆与执行环境"));
      continue;
    }
    if (event.type === "RUN_RESUMED") {
      add(activity(event, "runtime", "running", "已从检查点恢复执行", "继续未完成的模型或工具步骤"));
      continue;
    }
    if (event.type === "MODEL_RESOLVED") {
      const policy = payload.selection_policy === "auto" ? "自动探测后已冻结" : "固定版本";
      add(activity(event, "runtime", "completed", `已选择模型 · ${text(payload.model_id, "未命名模型")}`, `${policy}${payload.service_ref ? ` · ${payload.service_ref}` : ""}`));
      continue;
    }
    if (event.type === "PLAN_CREATED" || event.type === "PLAN_UPDATED") {
      const steps = Array.isArray(payload.steps) ? payload.steps : [];
      const completed = steps.filter((step: any) => step?.status === "completed").length;
      add(activity(event, "plan", "completed", event.type === "PLAN_CREATED" ? "已制定执行计划" : "已更新执行计划", `${text(payload.goal, "当前任务")} · ${completed}/${steps.length} 已完成`));
      continue;
    }
    if (event.type === "MODEL_REQUESTED") {
      const key = modelKey(event);
      add({ ...activity(event, "model", "running", "正在调用模型", `${text(payload.model_id, "模型")} · ${position}`), key });
      continue;
    }
    if (event.type === "MODEL_COMPLETED") {
      const key = modelKey(event);
      const message = object(payload.message);
      const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
      const names = calls.map((call: any) => text(call?.name || call?.function?.name, "")).filter(Boolean);
      const visible = visibleModelText(typeof message.content === "string" ? message.content : "");
      const completionRejected = names.length === 0 && blockedModelSteps.has(eventPositionKey(event));
      const detail = names.length
        ? `决定调用 ${names.join("、")} · ${tokenUsage(payload.usage)}`
        : `${completionRejected ? "候选结果未通过 Runtime 验收" : "模型返回阶段性结果"} · ${tokenUsage(payload.usage)}`;
      const update = { status: "completed" as const, title: names.length ? "模型已完成本步决策" : completionRejected ? "模型提交的完成结果未通过验收" : "模型已返回阶段性结果", detail, content: visible || undefined };
      if (!merge(key, event, update)) add({ ...activity(event, "model", "completed", update.title, detail, visible || undefined), key });
      continue;
    }
    if (event.type === "MODEL_FAILED") {
      const key = modelKey(event);
      const update = { status: "failed" as const, title: "模型调用失败", detail: compactText(payload.error, 220) || "模型服务返回错误" };
      if (!merge(key, event, update)) add({ ...activity(event, "model", "failed", update.title, update.detail), key });
      continue;
    }
    if (event.type === "TOOL_CALLED") {
      const name = text(payload.name, "未命名工具");
      if (CONTROL_TOOLS.has(name)) continue;
      const key = toolKey(event);
      add({ ...activity(event, "tool", "running", `正在调用工具 · ${name}`, toolArgumentSummary(name, payload.arguments)), key });
      continue;
    }
    if (event.type === "TOOL_APPROVAL_REQUESTED") {
      const key = toolKey(event);
      const update = { status: "waiting" as const, title: `等待工具审批 · ${text(payload.name, "写操作")}`, detail: "批准后将从当前检查点继续" };
      if (!merge(key, event, update)) add({ ...activity(event, "tool", "waiting", update.title, update.detail), key });
      continue;
    }
    if (event.type === "TOOL_COMPLETED" || event.type === "TOOL_FAILED") {
      const name = text(payload.name, "未命名工具");
      if (CONTROL_TOOLS.has(name)) continue;
      const key = toolKey(event);
      const failed = event.type === "TOOL_FAILED";
      const update = {
        status: failed ? "failed" as const : "completed" as const,
        title: `${failed ? "工具执行失败" : "工具执行完成"} · ${name}`,
        detail: toolResultSummary(payload.result, failed)
      };
      if (!merge(key, event, update)) add({ ...activity(event, "tool", update.status, update.title, update.detail), key });
      continue;
    }
    if (event.type === "CONTEXT_COMPACTED") {
      const removedMessages = number(payload.removed_messages);
      if (payload.mode === "read_time_projection") {
        add(activity(event, "context", "completed", "已生成 Context Collapse 视图", `${number(payload.before_tokens)} → ${number(payload.after_tokens)} Token · durable history 保留`));
      } else if (removedMessages === 0) {
        add(activity(event, "context", "completed", "已生成模型可见执行状态", `${number(payload.before_tokens)} → ${number(payload.after_tokens)} Token · 未移除历史消息`));
      } else {
        add(activity(event, "context", "completed", "已压缩上下文并继续执行", `${number(payload.before_tokens)} → ${number(payload.after_tokens)} Token · 移除 ${removedMessages} 条历史消息`));
      }
      continue;
    }
    if (event.type === "USER_INPUT_REQUESTED") {
      add(activity(event, "input", "waiting", "等待你的补充信息", compactText(payload.question, 220) || "Agent 需要你的决定"));
      continue;
    }
    if (event.type === "USER_INPUT_RECEIVED") {
      add(activity(event, "input", "completed", "已收到你的回答", "正在从检查点恢复任务"));
      continue;
    }
    if (event.type === "DELEGATION_REQUESTED") {
      add(activity(event, "agent", "running", "已委派子 Agent", delegationSummary(payload)));
      continue;
    }
    if (event.type === "DELEGATION_COMPLETED" || event.type === "DELEGATION_FAILED") {
      const failed = event.type === "DELEGATION_FAILED";
      add(activity(event, "agent", failed ? "failed" : "completed", failed ? "子 Agent 执行失败" : "子 Agent 已返回结果", delegationSummary(payload)));
      continue;
    }
    if (event.type === "PLAN_COMPLETION_BLOCKED") {
      const criterion = text(payload.criterion, "");
      const target = text(payload.target, "");
      const required = stringList(payload.required_tools);
      const title = criterion
        ? `验收未通过 · ${criterion}`
        : payload.reason === "missing_plan"
          ? "尚未建立执行计划"
          : payload.reason === "artifact_task_requires_plan"
            ? "产物任务正在转入规划执行"
            : "计划尚未完成，继续执行";
		const mismatch = String(payload.instruction || "").includes("found ") && String(payload.instruction || "").includes("receipt(s)");
		const detail = criterion
			? mismatch
				? `${target || "当前交付物"} · 已有成功回执，但参数、结果类型或 Todo 归属不匹配`
				: `${target || "当前交付物"} · 缺少 ${required.length ? required.join("、") : "有效工具"} 的成功回执`
			: compactText(payload.required_action || payload.instruction, 220) || "Runtime 已给出恢复动作并继续执行";
      add(activity(event, "plan", "running", title, detail));
      continue;
    }
    if (event.type === "VERIFICATION_SPEC_REVISED") {
      add(activity(event, "plan", "completed", "已修订验收规范", `${text(payload.criterion_key, "验收项")} · ${text(payload.reason, "已切换恢复契约")}`));
      continue;
    }
    if (event.type === "VERIFICATION_FAILED") {
      add(activity(event, "plan", "failed", "验证尝试未通过", `${text(payload.criterion_key, "验收项")} · ${text(payload.reason_code, "assertion_failed")}`));
      continue;
    }
		if (event.type === "VERIFICATION_COMPLETED") {
			add(activity(event, "plan", "completed", "平台验证通过", text(payload.criterion_key, "验收项")));
      continue;
    }
    if (event.type === "VERIFICATION_LOOP_DETECTED") {
      add(activity(event, "plan", "failed", "已熔断重复验收循环", "同一恢复契约连续被忽略两次，Runtime 已提前终止"));
      continue;
    }
    if (event.type === "FINAL_OUTPUT_REJECTED") {
      add(activity(event, "runtime", "running", "最终结果未通过完整性校验", "候选内容不是可交付结果，Runtime 已要求模型继续执行并重新生成"));
      continue;
    }
    if (event.type === "RUN_CANCEL_REQUESTED") {
      add(activity(event, "runtime", "cancelled", "正在停止任务", "等待当前模型或工具调用安全退出"));
    }
  }

  return items;
}

export function visibleModelText(value: string): string {
  let visible = value.trim();
  if (!visible) return "";
  // Never reveal an unfinished or completed private reasoning block. Providers
  // that expose only visible assistant content pass through unchanged.
  const lower = visible.toLowerCase();
  const lastOpen = lower.lastIndexOf("<think>");
  const lastClose = lower.lastIndexOf("</think>");
  if (lastOpen >= 0 && lastClose < lastOpen) return "";
  visible = visible.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  if (!visible) return "";
  try {
    const decoded = JSON.parse(visible) as Record<string, unknown>;
    for (const key of ["answer", "content", "message", "response"]) {
      if (typeof decoded?.[key] === "string") return decoded[key].trim();
    }
  } catch {
    // Normal assistant prose is not JSON.
  }
  return visible;
}

function activity(event: AgentEvent, kind: AgentActivityKind, status: AgentActivityStatus, title: string, detail?: string, content?: string): AgentActivityItem {
  return {
    key: `event-${event.sequence}`,
    sequence: event.sequence,
    lastSequence: event.sequence,
    eventSequences: [event.sequence],
    kind,
    status,
    title,
    detail,
    content,
    createdAt: event.created_at,
    turn: event.turn,
    step: event.step
  };
}

function modelKey(event: AgentEvent) { return `model-${event.turn || 0}-${event.step || 0}`; }
function toolKey(event: AgentEvent) { return `tool-${event.call_id || event.sequence}`; }
function eventPosition(event: AgentEvent) { return event.turn && event.step ? `Turn ${event.turn} · Step ${event.step}` : "任务级"; }
function eventPositionKey(event: AgentEvent) { return `${event.run_id}:${event.turn || 0}:${event.step || 0}`; }
function object(value: unknown): Record<string, any> { return value && typeof value === "object" ? value as Record<string, any> : {}; }
function text(value: unknown, fallback: string) { return typeof value === "string" && value.trim() ? value.trim() : fallback; }
function number(value: unknown) { const result = Number(value); return Number.isFinite(result) ? result : 0; }
function stringList(value: unknown) { return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string" && Boolean(item.trim())) : []; }

function tokenUsage(value: unknown) {
  const usage = object(value);
  const input = number(usage.input_tokens);
  const output = number(usage.output_tokens);
  return `${input}+${output} Token`;
}

function toolArgumentSummary(name: string, value: unknown) {
  const args = object(value);
  const path = text(args.path, "");
  if (name === "read_file" && path) {
    const start = number(args.start_line);
    const count = number(args.line_count);
    return count ? `读取 ${path} · 第 ${start || 1}-${(start || 1) + count - 1} 行` : `读取 ${path}`;
  }
  if (["write_file", "append_file", "edit_file"].includes(name) && path) return `${name === "write_file" ? "写入" : name === "append_file" ? "追加" : "编辑"} ${path}`;
  if (name === "list_files") return `查看 ${path || "."}`;
  if (name === "search_files") return `搜索 ${text(args.query || args.pattern, "工作区内容")}${path ? ` · ${path}` : ""}`;
  if (name === "run_command") return `运行 ${compactText(args.command || args.args, 150) || "受控命令"}`;
  if (name === "install_dependency") {
    const packages = Array.isArray(args.packages) ? args.packages.map((item: any) => `${item?.name || "?"}==${item?.version || "?"}`).join(", ") : "依赖";
    return `安装 ${compactText(packages, 150)} · ${compactText(args.source, 40) || "受信源"} · Run 作用域`;
  }
  if (name === "delegate_agent") return `委派 ${text(args.agent_version_id || args.agent, "子 Agent")} · ${compactText(args.task || args.input, 140)}`;
  const fields = Object.entries(args)
    .filter(([, item]) => ["string", "number", "boolean"].includes(typeof item))
    .slice(0, 2)
    .map(([key, item]) => `${key}=${compactText(item, 80)}`);
  return fields.join(" · ") || "参数已经校验并记录";
}

function toolResultSummary(value: unknown, failed: boolean) {
  const result = object(value);
  if (failed) return compactText(result.error || object(result.content).error || result.content, 220) || "工具返回失败";
  let content: unknown = result.content;
  if (typeof content === "string") {
    try { content = JSON.parse(content); } catch { return compactText(content, 180) || "工具已成功返回"; }
  }
  const data = object(content);
  if (typeof data.path === "string") return `已处理 ${data.path}${data.bytes ? ` · ${data.bytes} bytes` : ""}`;
  if (Array.isArray(data.entries)) return `返回 ${data.entries.length} 个条目`;
  if (typeof data.stdout === "string" && data.stdout.trim()) return `输出：${compactText(data.stdout, 180)}`;
  if (typeof data.status === "string") return `状态：${data.status}`;
  if (typeof data.ok === "boolean") return data.ok ? "执行成功" : "工具报告未完成";
  return "工具已成功返回，结果已写入执行上下文";
}

function delegationSummary(payload: Record<string, any>) {
  const agent = text(payload.agent_name || payload.agent_version_id, "子 Agent");
  const task = compactText(payload.task || payload.description || payload.result || payload.error, 180);
  return task ? `${agent} · ${task}` : agent;
}

function compactText(value: unknown, limit: number) {
  let result = "";
  if (typeof value === "string") result = value;
  else if (value !== undefined && value !== null) {
    try { result = JSON.stringify(value); } catch { result = String(value); }
  }
  result = result.replace(/\s+/g, " ").trim();
  return result.length > limit ? `${result.slice(0, limit)}…` : result;
}
