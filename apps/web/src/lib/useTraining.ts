// Phase F：分布式训练 hook —— 任务列表（5s 轮询）+ 提交 / 取消。
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { describeError } from "../components/common/FeedbackStates";
import { cancelTrainingJob, listTrainingJobs, submitTrainingJob } from "./api";
import type { SubmitTrainingInput } from "../types/training";

export type Training = ReturnType<typeof useTraining>;

export function useTraining() {
  const qc = useQueryClient();
  const invalidate = () => void qc.invalidateQueries({ queryKey: ["training", "jobs"] });

  const jobs = useQuery({
    queryKey: ["training", "jobs"],
    queryFn: listTrainingJobs,
    refetchInterval: 5000
  });

  const submit = useMutation({
    mutationFn: (input: SubmitTrainingInput) => submitTrainingJob(input),
    onSuccess: (res) => {
      toast.success(`训练任务已提交：${res.name}`, {
        description: res.registers_as ? `成功后注册为 ${res.registers_as}` : undefined
      });
      invalidate();
    },
    onError: (e) => toast.error(`提交失败：${describeError(e)}`)
  });

  const cancel = useMutation({
    mutationFn: (id: string) => cancelTrainingJob(id),
    onSuccess: () => {
      toast.success("训练任务已取消");
      invalidate();
    },
    onError: (e) => toast.error(`取消失败：${describeError(e)}`)
  });

  return { jobs, submit, cancel };
}
