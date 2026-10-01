import { describe, expect, it } from "vitest";

import type { AgentEvent } from "../types/agent";
import { buildAgentActivity, visibleModelText } from "./agentActivity";

function event(sequence: number, type: string, payload: Record<string, any> = {}, extra: Partial<AgentEvent> = {}): AgentEvent {
  return { run_id: "run-1", sequence, type, payload, created_at: `2026-09-09T00:00:${String(sequence).padStart(2, "0")}Z`, ...extra };
}

describe("agent conversation activity projection", () => {
  it("folds model and tool lifecycle pairs while preserving their real order", () => {
    const items = buildAgentActivity([
      event(1, "MODEL_REQUESTED", { model_id: "qwen" }, { turn: 1, step: 1 }),
      event(2, "MODEL_COMPLETED", { usage: { input_tokens: 10, output_tokens: 4 }, message: { content: "我先读取配置。", tool_calls: [{ name: "read_file" }] } }, { turn: 1, step: 1 }),
      event(3, "TOOL_CALLED", { name: "read_file", arguments: { path: "config.yaml", start_line: 1, line_count: 20 } }, { turn: 1, step: 1, call_id: "call-1" }),
      event(4, "TOOL_COMPLETED", { name: "read_file", result: { content: "{\"path\":\"config.yaml\"}" } }, { turn: 1, step: 1, call_id: "call-1" }),
      event(5, "MODEL_REQUESTED", { model_id: "qwen" }, { turn: 1, step: 2 }),
      event(6, "MODEL_COMPLETED", { usage: { input_tokens: 20, output_tokens: 5 }, message: { content: "配置已确认，继续修改。" } }, { turn: 1, step: 2 })
    ]);

    expect(items).toHaveLength(3);
    expect(items.map(item => item.kind)).toEqual(["model", "tool", "model"]);
    expect(items[0]).toMatchObject({ status: "completed", content: "我先读取配置。", eventSequences: [1, 2] });
    expect(items[1]).toMatchObject({ status: "completed", title: "工具执行完成 · read_file", eventSequences: [3, 4] });
    expect(items[2].content).toBe("配置已确认，继续修改。");
  });

  it("never exposes think blocks as intermediate output", () => {
    expect(visibleModelText("<think>private reasoning</think>可以安全展示")).toBe("可以安全展示");
    expect(visibleModelText("<think>unfinished private reasoning")).toBe("");
    expect(visibleModelText('{"answer":"最终可见结果"}')).toBe("最终可见结果");
  });

  it("hides runtime control tools because their durable plan events are clearer", () => {
    const items = buildAgentActivity([
      event(1, "TOOL_CALLED", { name: "update_plan", arguments: {} }, { turn: 1, step: 1, call_id: "plan-call" }),
      event(2, "PLAN_CREATED", { goal: "实现游戏", revision: 1, steps: [{ status: "in_progress" }] }),
      event(3, "TOOL_COMPLETED", { name: "update_plan", result: { content: "{}" } }, { turn: 1, step: 1, call_id: "plan-call" })
    ]);

    expect(items).toHaveLength(1);
    expect(items[0]).toMatchObject({ kind: "plan", title: "已制定执行计划" });
  });

  it("surfaces rejected final output as continued runtime work", () => {
    const items = buildAgentActivity([
      event(1, "FINAL_OUTPUT_REJECTED", { reason: "invalid_final_output" }, { turn: 2, step: 3 })
    ]);

    expect(items).toHaveLength(1);
    expect(items[0]).toMatchObject({
      kind: "runtime",
      status: "running",
      title: "最终结果未通过完整性校验"
    });
  });

  it("surfaces the exact failed completion criterion and required tool", () => {
    const items = buildAgentActivity([
      event(1, "PLAN_COMPLETION_BLOCKED", {
        reason: "invalid_evidence",
        criterion: "snake.py 的 Python 语法正确",
        target: "snake.py",
        required_tools: ["run_command"],
        declared_tools: ["read_file", "search_files"],
        auto_injected_tools: ["run_command"],
        required_action: "执行 python3 -m py_compile snake.py"
      })
    ]);

    expect(items).toHaveLength(1);
    expect(items[0]).toMatchObject({
      title: "验收未通过 · snake.py 的 Python 语法正确",
      detail: "snake.py · 缺少 run_command 的成功回执"
    });
  });

  it("marks a model completion claim as rejected when runtime blocks the same step", () => {
    const items = buildAgentActivity([
      event(1, "MODEL_REQUESTED", {}, { turn: 2, step: 3 }),
      event(2, "MODEL_COMPLETED", { message: { content: "All plan steps are completed." }, usage: { input_tokens: 10, output_tokens: 4 } }, { turn: 2, step: 3 }),
      event(3, "PLAN_COMPLETION_BLOCKED", { reason: "invalid_evidence", criterion: "syntax" }, { turn: 2, step: 3 })
    ]);

    expect(items[0]).toMatchObject({
      title: "模型提交的完成结果未通过验收",
      detail: "候选结果未通过 Runtime 验收 · 10+4 Token"
    });
  });

  it("does not present a legacy zero-removal ledger projection as context compaction", () => {
    const items = buildAgentActivity([
      event(1, "CONTEXT_COMPACTED", {
        generation: 5,
        before_tokens: 5843,
        after_tokens: 5550,
        removed_messages: 0,
        message_budget_tokens: 20123
      })
    ]);

    expect(items).toHaveLength(1);
    expect(items[0]).toMatchObject({
      kind: "context",
      title: "已生成模型可见执行状态",
      detail: "5843 → 5550 Token · 未移除历史消息"
    });
  });

  it("keeps real history compaction visible when messages were removed", () => {
    const items = buildAgentActivity([
      event(1, "CONTEXT_COMPACTED", {
        before_tokens: 22000,
        after_tokens: 17000,
        removed_messages: 12
      })
    ]);

    expect(items[0]).toMatchObject({
      title: "已压缩上下文并继续执行",
      detail: "22000 → 17000 Token · 移除 12 条历史消息"
    });
  });
});
