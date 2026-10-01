import type { AgentEvent, AgentPlanningPolicy, AgentRun } from "../types/agent";

export function planningPolicyLabel(policy?: string): string {
  if (policy === "required") return "强制规划";
  if (policy === "disabled") return "仅对话";
  return "自动判断";
}

export function executionModeLabel(run: AgentRun, events: AgentEvent[]): string {
  const selected = [...events].reverse().find((event) => event.type === "EXECUTION_MODE_SELECTED");
  if (selected?.payload?.mode === "planned") return "规划执行";
  if (selected?.payload?.mode === "conversational") return "直接回答";
  if (events.some((event) => event.type === "PLAN_CREATED" || event.type === "PLAN_UPDATED")) return "规划执行";
  const policy: AgentPlanningPolicy = run.binding_snapshot?.spec?.planning?.policy || "auto";
  if (policy === "required") return "等待规划";
  if (policy === "disabled") return "仅对话";
  return "自动判定中";
}
