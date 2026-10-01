// 实时存储总览：统一展示控制面和 Agent 运行时的真实数据落点。
import { Archive, RefreshCw } from "lucide-react";

import { PageHeader, PanelHeader } from "../components/common/PlatformPrimitives";
import { StorageArchiveTable } from "../components/storage/StorageArchiveTable";
import { StorageRuntimeOverview } from "../components/storage/StorageRuntimeOverview";
import { useStorage } from "../lib/useStorage";
import { useGoToPage } from "../lib/useGoToPage";

export function StoragePage() {
  const storage = useStorage();
  const goTo = useGoToPage();

  return (
    <section className="infra-page storage-page">
      <PageHeader
        title="存储总览"
        subtitle="实时查看 PostgreSQL、Redis、MinIO 与 pgvector 在控制面和 Agent 运行时中的连接状态、数据规模与职责边界"
        actions={
          <div className="storage-actions">
            <button
              className="console-refresh"
              type="button"
              onClick={() => {
                storage.tiers.refetch();
                storage.agent.refetch();
                storage.archives.refetch();
              }}
            >
              <RefreshCw className={storage.tiers.isFetching || storage.agent.isFetching ? "spinning" : undefined} size={14} /> 刷新实时状态
            </button>
          </div>
        }
      />

      <StorageRuntimeOverview storage={storage} />

      <section className="infra-panel storage-archive-panel">
        <PanelHeader title="控制面冷数据归档" action={<div className="storage-archive-actions"><button className="link-btn" type="button" onClick={() => goTo("config")}>配置保留期</button><button className="console-refresh" type="button" disabled={storage.run.isPending} onClick={() => storage.run.mutate()}><Archive size={14}/>{storage.run.isPending ? "归档中…" : "执行控制面归档"}</button></div>} />
        <div className="storage-archive-scope">
          <p>这里只处理控制面中超过保留期的数据并写入 MinIO，不会归档 Agent Session、Run State 或待发布 Outbox。</p>
          <span>自动归档：{storage.tiers.data?.archive_enabled ? "已开启" : "未开启"}</span>
          {Object.entries(storage.tiers.data?.retention ?? {}).map(([table, days]) => <span key={table}>{table} · {days} 天</span>)}
        </div>
        <StorageArchiveTable storage={storage} />
      </section>
    </section>
  );
}
