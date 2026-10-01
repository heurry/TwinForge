// C2：分层存储状态机 hook —— 各层占用 + 归档清单 + 手动归档触发。
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { describeError } from "../components/common/FeedbackStates";
import { getAgentStorageSummary, listArchives, runArchive, storageTiers } from "./api";

export type Storage = ReturnType<typeof useStorage>;

export function useStorage() {
  const qc = useQueryClient();
  const invalidate = () => {
    void qc.invalidateQueries({ queryKey: ["storage", "tiers"] });
    void qc.invalidateQueries({ queryKey: ["storage", "archives"] });
    void qc.invalidateQueries({ queryKey: ["storage", "agent"] });
  };

  const tiers = useQuery({
    queryKey: ["storage", "tiers"],
    queryFn: storageTiers,
    refetchInterval: 5000
  });

  const agent = useQuery({
    queryKey: ["storage", "agent"],
    queryFn: getAgentStorageSummary,
    refetchInterval: 5000,
    retry: false
  });

  const archives = useQuery({
    queryKey: ["storage", "archives"],
    queryFn: () => listArchives(50),
    refetchInterval: 30000
  });

  const run = useMutation({
    mutationFn: runArchive,
    onSuccess: (res) => {
      const archived = res.results.filter((r) => r.row_count > 0);
      const moved = archived.reduce((sum, r) => sum + r.row_count, 0);
      if (moved > 0) {
        toast.success(`已归档 ${moved} 行至对象层`, { description: archived.map((r) => `${r.table}:${r.row_count}`).join(" · ") });
      } else {
        const skipped = res.results.find((r) => r.skipped);
        toast.info(skipped?.skipped ? `无可归档数据（${skipped.skipped}）` : "无早于保留期的数据可归档");
      }
      invalidate();
    },
    onError: (e) => toast.error(`归档失败：${describeError(e)}`)
  });

  return { tiers, agent, archives, run };
}
