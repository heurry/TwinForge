// C1：注册模型版本抽屉（元数据；产物可在此填外部 URI，或注册后在列表上传至 MinIO）。
import { useState } from "react";

import { Drawer } from "../common/Drawer";
import type { ModelRegistry } from "../../lib/useModelRegistry";

export function RegisterVersionDrawer({ registry, onClose }: { registry: ModelRegistry; onClose: () => void }) {
  const [form, setForm] = useState({
    model_id: "qwen35-4b-customer",
    version: "lora-v1",
    base_model: "/mnt/nvme-data/LLM/llm_train_platform_miniv2/model/Qwen3.5-4B",
    lora_adapter: "",
    parent_version: "",
    tags: "",
    artifact_uri: ""
  });
  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm((f) => ({ ...f, [k]: e.target.value }));

  const valid = form.model_id.trim() !== "" && form.version.trim() !== "";
  const pending = registry.register.isPending;

  const submit = () => {
    registry.register.mutate(
      {
        model_id: form.model_id.trim(),
        version: form.version.trim(),
        base_model: form.base_model.trim() || undefined,
        lora_adapter: form.lora_adapter.trim() || undefined,
        parent_version: form.parent_version.trim() || undefined,
        artifact_uri: form.artifact_uri.trim() || undefined,
        tags: form.tags.split(",").map((t) => t.trim()).filter(Boolean)
      },
      { onSuccess: onClose }
    );
  };

  return (
    <Drawer open title="注册模型版本" subtitle="独立模型注册中心" onClose={onClose}>
      <div className="drawer-section">
        <input className="drawer-input" placeholder="model_id（如 qwen35-4b-customer）" value={form.model_id} onChange={set("model_id")} />
        <input className="drawer-input" placeholder="版本（如 v1 / v2）" value={form.version} onChange={set("version")} />
        <input className="drawer-input" placeholder="基座模型路径" value={form.base_model} onChange={set("base_model")} />
        <input className="drawer-input" placeholder="LoRA 适配器（可空）" value={form.lora_adapter} onChange={set("lora_adapter")} />
        <input className="drawer-input" placeholder="父版本 parent_version（血缘，可空）" value={form.parent_version} onChange={set("parent_version")} />
        <input className="drawer-input" placeholder="标签（逗号分隔，可空）" value={form.tags} onChange={set("tags")} />
        <input className="drawer-input" placeholder="产物 URI（可空；或注册后上传至 MinIO）" value={form.artifact_uri} onChange={set("artifact_uri")} />
        <p className="registry-hint">产物可在此填外部 URI，或注册后在列表里「上传产物」存入 MinIO。</p>
        <button className="infra-action-btn" type="button" disabled={!valid || pending} onClick={submit}>
          {pending ? "注册中…" : "注册版本"}
        </button>
      </div>
    </Drawer>
  );
}
