// Phase F：提交训练任务抽屉（PyTorchJob spec + 成功后注册到 C1 的目标）。
import { useState, type ChangeEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { Drawer } from "../common/Drawer";
import type { Training } from "../../lib/useTraining";
import { api } from "../../lib/api";
import { describeError } from "../common/FeedbackStates";
import type { ConfigItem } from "../../types/ops";

const BASE_MODEL = "/opt/twinforge/models/Qwen3.5-4B";
const MODEL_OPTIONS = [
  { path: "/opt/twinforge/models/Qwen3.5-4B", label: "Qwen3.5-4B · 本地基座", id: "qwen35-4b-customer" },
  { path: "/opt/twinforge/models/Qwen3-4B", label: "Qwen3-4B · 本地基座", id: "qwen3-4b-customer" },
];
const SFT_DATASET = "/opt/twinforge/data/dianjin_csc_sft_train.jsonl";

type SubmittedTraining = { id?: string; name?: string; modelVersionId?: string };

export function SubmitTrainingDrawer({ training, configs, onClose, blockedReason, initialConfigItemID = "", initialConfigVersion, onSubmitted }: { training: Training; configs: ConfigItem[]; onClose: () => void; blockedReason?: string; initialConfigItemID?: string; initialConfigVersion?: string; onSubmitted?: (result: SubmittedTraining) => void }) {
  const qc = useQueryClient();
  const [configItemID, setConfigItemID] = useState(initialConfigItemID);
  const [configVersion, setConfigVersion] = useState(initialConfigVersion ?? "");
  const [form, setForm] = useState({
    name: "qwen35-4b-customer-lora-v1",
    base_model: BASE_MODEL,
    image: "local/train:qwen35-v1",
    namespace: "training",
    dataset_uri: SFT_DATASET,
    workers: "1",
    gpus_per_worker: "1",
    model_id: "qwen35-4b-customer",
    version: "lora-v1",
    base_version: "",
    learning_rate: "0.0002",
    epochs: "3",
    lora_rank: "16",
    lora_alpha: "32",
    batch_size: "1",
    gradient_accumulation: "8",
    max_seq_length: "1024",
    max_samples: "512",
    precision: "bf16",
    deepspeed: "zero2",
    gradient_checkpointing: true,
  });
  const set = (k: keyof typeof form) => (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm((f) => ({ ...f, [k]: e.target.value }));

  const configLaunch = useMutation({
    mutationFn: ({ itemID, overrides }: { itemID: string; overrides: Record<string, unknown> }) => api<{ id?: string; name?: string }>(`/api/config/items/${itemID}/launch`, {
      method: "POST", body: JSON.stringify({ kind: "training", version: configVersion ? Number(configVersion) : undefined, overrides, operator: "frontend" }), timeoutMs: 60_000
    }),
    onSuccess: (payload) => {
      toast.success(`配置训练任务已提交：${payload.name || payload.id?.slice(0, 8) || "training"}`);
      qc.invalidateQueries({ queryKey: ["training", "jobs"] });
      qc.invalidateQueries({ queryKey: ["deployments"] });
      onSubmitted?.({ id: payload.id, name: payload.name });
      onClose();
    },
    onError: (error) => toast.error(`配置启动失败：${describeError(error)}`)
  });

  function submit() {
    const hyperparams: Record<string, unknown> = {
      learning_rate: Number(form.learning_rate),
      epochs: Number(form.epochs),
      lora_rank: Number(form.lora_rank),
      lora_alpha: Number(form.lora_alpha),
      per_device_train_batch_size: Number(form.batch_size),
      gradient_accumulation_steps: Number(form.gradient_accumulation),
      max_seq_length: Number(form.max_seq_length),
      max_samples: Number(form.max_samples),
      precision: form.precision,
      deepspeed: form.deepspeed,
      gradient_checkpointing: form.gradient_checkpointing,
    };
    const input = {
        name: form.name.trim(),
        base_model: form.base_model.trim(),
        image: form.image.trim(),
        namespace: form.namespace.trim() || undefined,
        dataset_uri: form.dataset_uri.trim() || undefined,
        workers: Number(form.workers) || 0,
        gpus_per_worker: Number(form.gpus_per_worker) || 0,
        model_id: form.model_id.trim() || undefined,
        version: form.version.trim() || undefined,
        base_version: form.base_version.trim() || undefined,
        hyperparams
      };
    if (configItemID) {
      configLaunch.mutate({ itemID: configItemID, overrides: input });
      return;
    }
    training.submit.mutate(input, {
      onSuccess: (payload) => {
        onSubmitted?.({ id: payload.id, name: payload.name });
        onClose();
      }
    });
  }

  const valid = form.name.trim() && form.base_model.trim() && form.image.trim() && !blockedReason;

  return (
    <Drawer open title="提交客服模型微调" subtitle="Qwen3.5-4B · DianJin-CSC · LoRA / PEFT" onClose={onClose}>
      <div className="drawer-section training-form">
        {blockedReason ? <p className="training-form-blocked">{blockedReason}</p> : null}
        {configs.length ? <label>
          配置中心启动模板
          <select className="drawer-input" value={configItemID} onChange={(event) => {
            const itemID = event.target.value;
            const item = configs.find((candidate) => candidate.id === itemID);
            setConfigItemID(itemID);
            setConfigVersion(itemID ? String(item?.active_version ?? "") : "");
          }}>
            <option value="">不使用模板（表单直接提交）</option>
            {configs.map((item) => <option key={item.id} value={item.id}>{item.config_key} · v{item.id === configItemID && configVersion ? configVersion : item.active_version} · {item.env}</option>)}
          </select>
          <small className="training-input-note">选择后以该不可变版本为基线，并用下方表单作为显式 overrides；任务会记录 config_ref。</small>
        </label> : null}
        <label>
          任务名
          <input className="drawer-input" value={form.name} onChange={set("name")} />
        </label>
        <label>
          基座模型
          <select className="drawer-input" value={form.base_model} onChange={(event) => {
            const option = MODEL_OPTIONS.find((item) => item.path === event.target.value);
            setForm((current) => ({ ...current, base_model: event.target.value, model_id: option?.id || current.model_id }));
          }}>
            {MODEL_OPTIONS.map((item) => <option key={item.path} value={item.path}>{item.label}</option>)}
          </select>
          <small className="training-input-note">切换基座会同步产出模型 ID；模型文件需已挂载到训练节点对应路径。</small>
        </label>
        <label>
          客服数据集
          <select className="drawer-input" value={form.dataset_uri} onChange={set("dataset_uri")}>
            <option value={SFT_DATASET}>DianJin-CSC-Data · 客服多轮 SFT JSONL</option>
          </select>
        </label>
        <label>
          训练镜像
          <input className="drawer-input" value={form.image} onChange={set("image")} />
          <small className="training-input-note">内置 Transformers + PEFT + DeepSpeed，支持 Qwen3.5 文本 LoRA。</small>
        </label>
        <div className="training-form-grid">
          <label>
            Worker 副本
            <select className="drawer-input" value={form.workers} onChange={set("workers")}><option value="0">单卡 · Master</option><option value="1">双卡 · Master + Worker</option></select>
          </label>
          <label>
            GPU / 副本
            <select className="drawer-input" value={form.gpus_per_worker} onChange={set("gpus_per_worker")}><option value="1">1 GPU</option></select>
          </label>
        </div>
        <label>
          命名空间
          <select className="drawer-input" value={form.namespace} onChange={set("namespace")}><option value="training">training</option></select>
        </label>

        <hr className="training-form-sep" />
        <p className="training-form-hint">LoRA 与显存优化参数</p>
        <div className="training-form-grid">
          <label>学习率<input className="drawer-input" type="number" step="0.00001" value={form.learning_rate} onChange={set("learning_rate")} /></label>
          <label>Epochs<input className="drawer-input" type="number" min={1} max={10} value={form.epochs} onChange={set("epochs")} /></label>
          <label>LoRA Rank<select className="drawer-input" value={form.lora_rank} onChange={set("lora_rank")}><option value="8">8</option><option value="16">16</option><option value="32">32</option></select></label>
          <label>LoRA Alpha<select className="drawer-input" value={form.lora_alpha} onChange={set("lora_alpha")}><option value="16">16</option><option value="32">32</option><option value="64">64</option></select></label>
          <label>Micro Batch<select className="drawer-input" value={form.batch_size} onChange={set("batch_size")}><option value="1">1</option><option value="2">2</option></select></label>
          <label>梯度累积<select className="drawer-input" value={form.gradient_accumulation} onChange={set("gradient_accumulation")}><option value="4">4</option><option value="8">8</option><option value="16">16</option></select></label>
          <label>最大序列长度<select className="drawer-input" value={form.max_seq_length} onChange={set("max_seq_length")}><option value="512">512</option><option value="1024">1024</option><option value="2048">2048</option></select></label>
          <label>训练样本上限<input className="drawer-input" type="number" min={0} step={128} value={form.max_samples} onChange={set("max_samples")} /></label>
          <label>精度<select className="drawer-input" value={form.precision} onChange={set("precision")}><option value="bf16">BF16</option><option value="fp16">FP16</option></select></label>
          <label>DeepSpeed<select className="drawer-input" value={form.deepspeed} onChange={set("deepspeed")}><option value="zero2">ZeRO-2</option><option value="zero3">ZeRO-3</option><option value="off">关闭</option></select></label>
        </div>
        <label className="training-checkbox"><input type="checkbox" checked={form.gradient_checkpointing} onChange={(event) => setForm((current) => ({ ...current, gradient_checkpointing: event.target.checked }))} />梯度检查点</label>

        <hr className="training-form-sep" />
        <p className="training-form-hint">成功后注册到模型与版本中心</p>
        <div className="training-form-grid">
          <label>
            model_id
            <input className="drawer-input" value={form.model_id} onChange={set("model_id")} />
          </label>
          <label>
            version
            <input className="drawer-input" value={form.version} onChange={set("version")} />
          </label>
        </div>
        <label>血缘父版本<input className="drawer-input" placeholder="可选" value={form.base_version} onChange={set("base_version")} /></label>

        <button className="infra-action-btn" type="button" disabled={!valid || training.submit.isPending || configLaunch.isPending} onClick={submit}>
          {training.submit.isPending || configLaunch.isPending ? "提交中..." : configItemID ? "按配置版本启动训练" : "提交训练任务"}
        </button>
      </div>
    </Drawer>
  );
}
