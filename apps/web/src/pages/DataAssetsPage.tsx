import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, CheckCircle2, Clock3, Database, RefreshCw } from "lucide-react";

import { EmptyState, ErrorState, Skeleton } from "../components/common/FeedbackStates";
import { PageHeader, PanelHeader, StatusBadge } from "../components/common/PlatformPrimitives";
import { api } from "../lib/api";
import { useGoToPage } from "../lib/useGoToPage";
import type { TrainingJob, TrainingJobsList } from "../types/training";

type DatasetAsset = {
  id: string;
  name: string;
  version: string;
  purpose: string;
  location: string;
  source: string;
  scale: string;
  state: "configured" | "observed" | "planned";
  target: "training" | "benchmarks";
};

type InferenceEvidence = {
  inference?: { benchmark?: { workload?: string; updated_at?: string; run_id?: string } };
};

const TRAIN_DATASET = "/mnt/nvme-data/LLM/llm_train_platform_miniv2/data/raw/dianjin_csc_train.parquet";
const BENCH_DATASET = "data/cleaned/dianjin_csc_benchmark.jsonl";

export function DataAssetsPage() {
  const goTo = useGoToPage();
  const jobs = useQuery({
    queryKey: ["training", "jobs", "dataset-assets"],
    queryFn: () => api<TrainingJobsList>("/api/training/jobs"),
    retry: false,
  });
  const evidence = useQuery({
    queryKey: ["ai", "inference", "evidence", "dataset-assets"],
    queryFn: () => api<InferenceEvidence>("/api/ai/inference/evidence"),
    retry: false,
  });

  const assets = useMemo(() => buildAssets(jobs.data?.jobs ?? [], evidence.data), [jobs.data, evidence.data]);
  const observed = assets.filter((item) => item.state === "observed").length;
  const planned = assets.filter((item) => item.state === "planned").length;

  return (
    <section className="infra-page data-assets-page">
      <PageHeader
        title="数据资产"
        subtitle="客服场景的训练、压测与质量回归数据契约；项目预设可选，任务提交时记录实际版本"
        actions={<button className="console-refresh" type="button" onClick={() => { jobs.refetch(); evidence.refetch(); }}><RefreshCw size={14} /> 刷新状态</button>}
      />

      <div className="asset-summary-strip">
        <Summary icon={Database} label="已配置" value={assets.length - planned} detail="当前项目可直接选择" />
        <Summary icon={CheckCircle2} label="已有运行证据" value={observed} detail="训练台账或压测证据已引用" />
        <Summary icon={Clock3} label="待构建" value={planned} detail="量化与质量回归下一阶段" />
      </div>

      <section className="infra-panel asset-table-panel">
        <PanelHeader title="受控数据集版本" action="DianJin-CSC-Data · 中文客服" />
        {jobs.isLoading && evidence.isLoading ? <Skeleton rows={4} /> : (
          <div className="asset-table">
            <div className="asset-table-row header">
              <span>数据集 / 版本</span><span>用途</span><span>规模</span><span>状态</span><span>来源</span><span>操作</span>
            </div>
            {assets.map((asset) => (
              <div className="asset-table-row" key={asset.id}>
                <span className="asset-name"><strong>{asset.name} · {asset.version}</strong><small title={asset.location}>{asset.location}</small></span>
                <span>{asset.purpose}</span>
                <span>{asset.scale}</span>
                <StatusBadge status={stateLabel(asset.state)} />
                <span>{asset.source}</span>
                <button className="asset-open-btn" type="button" onClick={() => goTo(asset.target)}>{asset.target === "training" ? "用于微调" : "用于压测"}<ArrowRight size={13} /></button>
              </div>
            ))}
          </div>
        )}
        {(jobs.isError || evidence.isError) && !jobs.data && !evidence.data ? (
          <ErrorState error={jobs.error ?? evidence.error} onRetry={() => { jobs.refetch(); evidence.refetch(); }} />
        ) : null}
      </section>

      <div className="asset-contract-grid">
        <section>
          <h2>训练数据契约</h2>
          <p>默认选择 Qwen3.5-4B 与 DianJin 客服 SFT 数据。数据集不是写死在训练实现中，而是受控白名单中的默认项；后续注册新版本后可切换并保留任务溯源。</p>
        </section>
        <section>
          <h2>推理数据契约</h2>
          <p>压测集与训练集分离，固定覆盖 1K/2K 上下文和 1/2/4/8/16 并发。质量回归集与 AWQ 校准集仍需独立划分，界面明确标记为待构建。</p>
        </section>
      </div>
    </section>
  );
}

function buildAssets(jobs: TrainingJob[], evidence?: InferenceEvidence): DatasetAsset[] {
  const trainObserved = jobs.some((job) => job.dataset_uri === TRAIN_DATASET || job.dataset_uri?.includes("dianjin"));
  const benchmark = evidence?.inference?.benchmark;
  return [
    { id: "dianjin-raw-v1", name: "DianJin-CSC-Raw", version: "v1", purpose: "SFT 数据构建来源", location: TRAIN_DATASET, source: "DianJin-CSC-Data", scale: "约 1.3 万条原始对话", state: trainObserved ? "observed" : "configured", target: "training" },
    { id: "dianjin-sft-v1", name: "DianJin-CSC-SFT", version: "规划中", purpose: "LoRA 客服微调", location: "待清洗、模板化并划分 train/validation", source: "DianJin-CSC-Data", scale: "待构建", state: "planned", target: "training" },
    { id: "dianjin-bench-v1", name: "DianJin-CSC-Benchmark", version: "v1", purpose: "推理性能与正确性", location: BENCH_DATASET, source: benchmark?.run_id ? `Benchmark ${benchmark.run_id.slice(0, 8)}` : "项目压测预设", scale: "1K / 2K 上下文", state: benchmark ? "observed" : "configured", target: "benchmarks" },
    { id: "dianjin-quality-v1", name: "DianJin-CSC-Quality-Holdout", version: "规划中", purpose: "微调/量化质量回归", location: "待从原始数据中隔离", source: "DianJin-CSC-Data", scale: "待确定", state: "planned", target: "benchmarks" },
    { id: "dianjin-awq-v1", name: "DianJin-CSC-AWQ-Calib", version: "规划中", purpose: "INT4 / AWQ 校准", location: "待构建校准子集", source: "训练集无泄漏抽样", scale: "待确定", state: "planned", target: "benchmarks" },
  ];
}

function stateLabel(state: DatasetAsset["state"]): string {
  if (state === "observed") return "已验证";
  if (state === "configured") return "已配置";
  return "待构建";
}

function Summary({ icon: Icon, label, value, detail }: { icon: typeof Database; label: string; value: number; detail: string }) {
  return <div><Icon size={18} /><span><small>{label}</small><strong>{value}</strong><em>{detail}</em></span></div>;
}
