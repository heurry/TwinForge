// Phase F：模型训练页（薄组合）。提交分布式训练 → 实时状态 → 成功自动注册模型版本。
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Boxes, CheckCircle2, Database, Plus, RefreshCw } from "lucide-react";

import { PageHeader, PanelHeader } from "../components/common/PlatformPrimitives";
import { SubmitTrainingDrawer } from "../components/training/SubmitTrainingDrawer";
import { TrainingJobsPanel } from "../components/training/TrainingJobsPanel";
import { useTraining } from "../lib/useTraining";
import { useGoToPage } from "../lib/useGoToPage";
import { useDeliveryContext } from "../lib/useDeliveryContext";
import { api } from "../lib/api";
import type { ConfigItem } from "../types/ops";

type InferenceRuntime = { status?: string; model?: string; profile?: string };

export function TrainingPage() {
  const training = useTraining();
  const goTo = useGoToPage();
  const { context, update } = useDeliveryContext();
  const [submitting, setSubmitting] = useState(Boolean(context.deliveryKind === "training" && context.configItemId));
  const runtime = useQuery({
    queryKey: ["inference", "runtime"],
    queryFn: () => api<InferenceRuntime>("/api/inference/runtime"),
    retry: false,
    refetchInterval: 5000,
  });
  const configs = useQuery({
    queryKey: ["config", "items", "training"],
    queryFn: () => api<{ items: ConfigItem[] }>("/api/config/items"),
    select: (payload) => payload.items.filter((item) => item.config_key.toLowerCase().includes("train"))
  });
  const inferenceBusy = runtime.data?.status === "ready" || runtime.data?.status === "starting";
  const jobs = training.jobs.data?.jobs ?? [];
  const activeJobs = jobs.filter((job) => job.status === "pending" || job.status === "running");

  return (
    <section className="infra-page training-page">
      <PageHeader
        title="训练任务控制面"
        subtitle="Qwen3.5-4B 客服 LoRA 微调：数据准备 → PyTorchJob → 日志与 GPU 观测 → 版本自动归档"
        actions={
          <div className="training-actions">
            <button className="console-refresh" type="button" onClick={() => training.jobs.refetch()}>
              <RefreshCw className={training.jobs.isFetching ? "spinning" : undefined} size={14} /> 刷新
            </button>
            <button className="console-refresh primary" disabled={inferenceBusy} title={inferenceBusy ? "请先在推理服务页停止 vLLM 服务" : "提交训练任务"} type="button" onClick={() => setSubmitting(true)}>
              <Plus size={14} /> 提交训练
            </button>
          </div>
        }
      />

      <section className={`training-lane-banner ${inferenceBusy ? "blocked" : "available"}`}>
        {inferenceBusy ? <AlertTriangle size={18} /> : <CheckCircle2 size={18} />}
        <span><strong>{inferenceBusy ? "GPU 通道正在运行推理服务" : activeJobs.length ? "训练任务正在使用 GPU 通道" : "GPU 训练通道可用"}</strong><small>{inferenceBusy ? `${runtime.data?.model || "Qwen3.6-27B-FP8"} · ${runtime.data?.profile || "vLLM"}；停止服务后才可提交训练` : "单机双卡按阶段分时使用，控制面与后端都会执行资源互斥检查"}</small></span>
        {inferenceBusy ? <button type="button" onClick={() => goTo("benchmarks")}>前往停止推理服务</button> : null}
      </section>

      <div className="training-contract-strip">
        <div><Boxes size={17} /><span><small>基座模型</small><strong>Qwen3.5-4B</strong><em>本地路径已配置</em></span></div>
        <div><Database size={17} /><span><small>客服数据</small><strong>DianJin-CSC-Data</strong><em>原始数据已下载，SFT 切分待构建</em></span></div>
        <div><CheckCircle2 size={17} /><span><small>产出</small><strong>LoRA Adapter</strong><em>成功后自动登记模型版本与血缘</em></span></div>
      </div>

      <section className="infra-panel">
        <PanelHeader title="训练任务" action={`${activeJobs.length} 个运行中 · 5s 轮询`} />
        <TrainingJobsPanel
          training={training}
          onOpenModel={(job) => goTo("models", { deliveryKind: "training", trainingJobId: job.id, modelVersionId: job.model_version_id, benchmarkRunId: null, deploymentId: null })}
          onDiagnose={(job) => goTo("aiOps", { deliveryKind: "training", trainingJobId: job.id, benchmarkRunId: null, deploymentId: null })}
        />
      </section>

      {submitting ? <SubmitTrainingDrawer
        configs={configs.data ?? []}
        initialConfigItemID={context.deliveryKind === "training" ? context.configItemId : undefined}
        initialConfigVersion={context.deliveryKind === "training" ? context.configVersion : undefined}
        blockedReason={inferenceBusy ? "vLLM 推理服务正在占用 GPU，请先停止推理服务" : undefined}
        training={training}
        onClose={() => setSubmitting(false)}
        onSubmitted={(result) => update({ deliveryKind: "training", trainingJobId: result.id ?? null, benchmarkRunId: null, deploymentId: null })}
      /> : null}
    </section>
  );
}
