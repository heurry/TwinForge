import { describe, expect, it } from "vitest";

import { buildStorageOverview } from "./storageOverview";
import type { AgentStorageSummary } from "../types/agent";
import type { StorageTiers } from "../types/storage";

const formatBytes = (value: number) => `${value} B`;

describe("buildStorageOverview", () => {
  it("merges control-plane and Agent storage without hiding scope boundaries", () => {
    const platform: StorageTiers = {
      retention: {}, archive_enabled: false,
      tiers: [
        { id: "hot", label: "Redis", kind: "cache", enabled: true, detail: "cache-aside" },
        { id: "relational", label: "PostgreSQL", kind: "relational", enabled: true, total_bytes: 2048, tables: [] },
        { id: "object", label: "MinIO", kind: "object", enabled: true, archived_bytes: 512, manifests: 3 }
      ]
    };
    const agent: AgentStorageSummary = {
      postgresql: { name: "PostgreSQL", role: "事实", status: "ready", metrics: { sessions: 4, messages: 12, runs: 7, events: 80, current_run_states: 2, pending_outbox: 1, inline_artifact_bytes: 64 } },
      redis: { name: "Redis", role: "通知", status: "ready", detail: "Run wakeup" },
	  minio: { name: "MinIO", role: "对象", status: "ready", metrics: { object_artifacts: 5, inline_artifacts: 1, artifact_bytes: 128 } },
	  pgvector: { name: "pgvector", role: "向量", status: "ready", metrics: { embedded_memories: 3, pending_embeddings: 1 } }
    };

    const rows = buildStorageOverview(platform, agent, formatBytes);
    expect(rows).toHaveLength(4);
    expect(rows[0].platform.status).toBe("ready");
    expect(rows[0].metrics).toContain("Agent 4 Session / 12 消息");
    expect(rows[1].metrics).toEqual(["1 条 Agent Outbox 待发布"]);
    expect(rows[2].platform.status).toBe("ready");
	expect(rows[2].agent.status).toBe("ready");
	expect(rows[2].metrics).toContain("5 个 Agent 对象产物");
	expect(rows[3].metrics).toContain("3 条已向量化 Memory");
  });

  it("returns explicit unknown states for missing responses instead of throwing on null data", () => {
    const rows = buildStorageOverview(undefined, undefined, formatBytes);
    expect(rows).toHaveLength(4);
    expect(rows[0].platform.status).toBe("unknown");
    expect(rows[0].agent.status).toBe("unknown");
    expect(rows[0].metrics).toContain("Agent 0 Session / 0 消息");
  });
});
