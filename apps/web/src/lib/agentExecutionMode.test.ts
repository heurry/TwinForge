import { describe, expect, it } from "vitest";

import type { AgentEvent, AgentRun } from "../types/agent";
import { executionModeLabel, planningPolicyLabel } from "./agentExecutionMode";

const run = (policy: "auto" | "required" | "disabled" = "auto"): AgentRun => ({
  id: "run-1",
  agent_version_id: "version-1",
  status: "running",
  input: { question: "test" },
  binding_snapshot: { spec: { planning: { policy } } },
  current_turn: 1,
  current_step: 1,
  created_at: "2026-09-11T00:00:00Z"
});

const modeEvent = (mode: "planned" | "conversational"): AgentEvent => ({
  run_id: "run-1",
  type: "EXECUTION_MODE_SELECTED",
  sequence: 3,
  created_at: "2026-09-11T00:00:01Z",
  payload: { mode, policy: "auto" }
});

describe("Agent execution mode projection", () => {
  it("shows an undecided auto run without pretending a Plan exists", () => {
    expect(executionModeLabel(run(), [])).toBe("自动判定中");
  });

  it("uses the durable selection event as the source of truth", () => {
    expect(executionModeLabel(run(), [modeEvent("conversational")])).toBe("直接回答");
    expect(executionModeLabel(run(), [modeEvent("planned")])).toBe("规划执行");
  });

  it("falls back to legacy Plan events and frozen policy", () => {
    expect(executionModeLabel(run(), [{ ...modeEvent("planned"), type: "PLAN_CREATED" }])).toBe("规划执行");
    expect(executionModeLabel(run("required"), [])).toBe("等待规划");
    expect(executionModeLabel(run("disabled"), [])).toBe("仅对话");
    expect(planningPolicyLabel("auto")).toBe("自动判断");
  });
});
