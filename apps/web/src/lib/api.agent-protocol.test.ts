import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  answerAgentRunQuestion,
  cancelAgentRun,
  cancelA2ATask,
  getA2ATask,
  getAgentRunPlan,
  getAgentRunQuestion,
  listMCPServerVersions,
  normalizeListResponse,
  sendA2AMessage,
  subscribeA2ATask,
  syncMCPServerTools,
  testMCPServerVersion
} from "./api";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("window", globalThis);
});

describe("MCP frontend transport", () => {
  it("normalizes a nullable list response to an empty array", () => {
    expect(normalizeListResponse(null)).toEqual({ data: [] });
    expect(normalizeListResponse({ data: null })).toEqual({ data: [] });
  });

  it("does not expose a null MCP list to the UI", async () => {
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ data: null }), {
      status: 200,
      headers: { "Content-Type": "application/json" }
    }));

    await expect(listMCPServerVersions()).resolves.toEqual({ data: [] });
    expect(fetchMock.mock.calls[0][0]).toBe("/agent-api/api/v1/mcp-servers?limit=200");
  });

  it("calls both MCP initialize test and tool discovery actions", async () => {
    const server = { id: "server-1", name: "Filesystem MCP", health: { status: "healthy" }, tools: [] };
    fetchMock
      .mockResolvedValueOnce(new Response(JSON.stringify({ data: server }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ data: { ...server, tools: [{ name: "read_file" }] } }), { status: 200 }));

    await testMCPServerVersion("server-1");
    await syncMCPServerTools("server-1");

    expect(fetchMock.mock.calls[0][0]).toBe("/agent-api/api/v1/mcp-server-versions/server-1:test");
    expect(fetchMock.mock.calls[1][0]).toBe("/agent-api/api/v1/mcp-server-versions/server-1:sync-tools");
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: "POST" });
  });
});

describe("A2A frontend transport", () => {
  const task = {
    id: "task-1",
    contextId: "context-1",
    status: { state: "working", timestamp: "2026-09-07T00:00:00Z" },
    metadata: { runId: "run-1" }
  };

  it("sends a message and supports get/cancel task lifecycle", async () => {
    fetchMock
      .mockResolvedValueOnce(new Response(JSON.stringify({ task }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(task), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ ...task, status: { ...task.status, state: "canceled" } }), { status: 200 }));

    await expect(sendA2AMessage("agent-1", "检查部署", "message-1")).resolves.toEqual({ task });
    await expect(getA2ATask("task-1")).resolves.toEqual(task);
    await expect(cancelA2ATask("task-1")).resolves.toMatchObject({ status: { state: "canceled" } });

    const sendInit = fetchMock.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(String(sendInit.body))).toEqual({
      message: { messageId: "message-1", role: "user", parts: [{ text: "检查部署" }] }
    });
    expect(fetchMock.mock.calls.map(call => call[0])).toEqual([
      "/agent-api/api/v1/a2a/agents/agent-1/message:send",
      "/agent-api/api/v1/a2a/tasks/task-1",
      "/agent-api/api/v1/a2a/tasks/task-1:cancel"
    ]);
  });

  it("parses chunked A2A SSE status updates in order", async () => {
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode("event: status-update\ndata: {\"statusUpdate\":{\"status\":{\"state\":\"working\"}}}\n"));
        controller.enqueue(encoder.encode("\nevent: status-update\ndata: {\"statusUpdate\":{\"status\":{\"state\":\"completed\"},\"final\":true}}\n\n"));
        controller.close();
      }
    });
    fetchMock.mockResolvedValueOnce(new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } }));
    const events: Array<{ event: string; data: Record<string, unknown> }> = [];

    await subscribeA2ATask("task-1", event => events.push(event));

    expect(events.map(event => (event.data.statusUpdate as { status: { state: string } }).status.state)).toEqual(["working", "completed"]);
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ headers: expect.objectContaining({ Accept: "text/event-stream", "X-Tenant-ID": "demo" }) });
  });
});

describe("default autonomous run transport", () => {
  it("sends an authenticated stop request for an active run", async () => {
    fetchMock.mockResolvedValueOnce(new Response(null, { status: 202 }));

    await expect(cancelAgentRun("run-1")).resolves.toBeUndefined();

    expect(fetchMock.mock.calls[0][0]).toBe("/agent-api/api/v1/runs/run-1:cancel");
    expect(fetchMock.mock.calls[0][1]).toMatchObject({
      method: "POST",
      headers: expect.objectContaining({ "X-Actor-ID": "web-console", "X-Tenant-ID": "demo" })
    });
  });

  it("loads the durable plan and answers a suspended run question", async () => {
    const plan = { run_id: "run-1", revision: 2, goal: "ship", steps: [] };
    const question = { id: "question-1", run_id: "run-1", status: "pending", question: "继续吗？" };
    fetchMock
      .mockResolvedValueOnce(new Response(JSON.stringify({ data: plan }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ data: question }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ data: { ...question, status: "answered", answer: "继续" } }), { status: 200 }));

    await expect(getAgentRunPlan("run-1")).resolves.toEqual({ data: plan });
    await expect(getAgentRunQuestion("run-1")).resolves.toEqual({ data: question });
    await expect(answerAgentRunQuestion("question-1", "继续")).resolves.toMatchObject({ data: { status: "answered" } });

    expect(fetchMock.mock.calls.map(call => call[0])).toEqual([
      "/agent-api/api/v1/runs/run-1/plan",
      "/agent-api/api/v1/runs/run-1/question",
      "/agent-api/api/v1/questions/question-1:answer"
    ]);
    expect(fetchMock.mock.calls[2][1]).toMatchObject({ method: "POST", headers: expect.objectContaining({ "X-Actor-ID": "web-console" }) });
  });
});
