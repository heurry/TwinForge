// C1：模型注册中心面板（元数据管理 tab 的真实数据）。
//
// 展示注册的模型版本：状态 / 基座+LoRA / 血缘(parent_version) / 标签 / 运行时绑定(来自 service_instances) /
// 产物(上传至 MinIO 或下载) / 状态切换 / 注销。纯展示 + 调用 useModelRegistry 的写操作。
import { useRef } from "react";
import { toast } from "sonner";
import { CheckCircle2, Download, Gauge, Rocket, Trash2, UploadCloud } from "lucide-react";

import { StatusBadge } from "../common/PlatformPrimitives";
import { describeError, EmptyState, ErrorState, Skeleton } from "../common/FeedbackStates";
import { modelArtifactURL } from "../../lib/api";
import type { ModelRegistry } from "../../lib/useModelRegistry";
import type { RegisteredModelVersion } from "../../types/registry";

const inferenceModelIDs = new Set(["qwen38-27b-fp8", "qwen36-27b-fp8", "qwen36-27b-awq", "qwen35-4b-customer", "qwen3-4b-customer"]);

export function ModelRegistryPanel({ registry, selectedVersionID, onSelect, onBenchmark, onRelease }: { registry: ModelRegistry; selectedVersionID?: string; onSelect?: (version: RegisteredModelVersion) => void; onBenchmark?: (version: RegisteredModelVersion) => void; onRelease?: (version: RegisteredModelVersion) => void }) {
  const { list } = registry;
  const versions = list.data?.versions ?? [];
  const bindings = list.data?.bindings ?? {};

  async function download(id: string) {
    try {
      const res = await modelArtifactURL(id);
      if (res.download_url) window.open(res.download_url, "_blank", "noopener");
      else toast.info(res.note || "暂无可下载产物");
    } catch (e) {
      toast.error(`获取下载地址失败：${describeError(e)}`);
    }
  }

  if (list.isLoading) return <Skeleton rows={4} />;
  if (list.isError) return <ErrorState error={list.error} onRetry={list.refetch} />;
  if (versions.length === 0) {
    return <EmptyState title="暂无注册模型版本" description="点击「注册模型」登记第一个版本（model_id + 版本 + 基座/LoRA + 血缘）" />;
  }

  return (
    <table className="infra-table models-tab-table registry-table">
      <thead>
        <tr>{["模型 / 版本", "状态", "基座 / LoRA", "血缘", "标签", "运行实例", "产物", "操作"].map((c) => <th key={c}>{c}</th>)}</tr>
      </thead>
      <tbody>
        {versions.map((v) => {
          const bound = bindings[v.model_id] ?? [];
          return (
            <tr className={selectedVersionID === v.id ? "registry-row-selected" : undefined} key={v.id}>
              <td>
                <strong>{v.model_id}</strong>
                <span className="cell-subtle">{v.version}{v.created_by ? ` · ${v.created_by}` : ""}</span>
              </td>
              <td><StatusBadge status={v.status} /></td>
              <td>
                {v.base_model ?? "—"}
                {v.lora_adapter ? <span className="cell-subtle">+ LoRA {v.lora_adapter}</span> : null}
              </td>
              <td>{v.parent_version ? `← ${v.parent_version}` : "—"}</td>
              <td>{v.tags.length ? v.tags.join(", ") : "—"}</td>
              <td>
                {bound.length ? <strong>{bound.length} 实例</strong> : <span className="cell-subtle" style={{ marginTop: 0 }}>未绑定</span>}
                {bound.length ? <span className="cell-subtle">{bound.map((b) => b.name).join(", ")}</span> : null}
              </td>
              <td>
                {v.artifact_uri ? (
                  <button className="link-btn" type="button" onClick={() => void download(v.id)}><Download size={13} /> 下载</button>
                ) : (
                  <ArtifactUpload id={v.id} registry={registry} />
                )}
              </td>
              <td className="registry-actions">
                {onSelect ? <button className="link-btn" type="button" title="设为当前交付版本" onClick={() => onSelect(v)}><CheckCircle2 size={13} /></button> : null}
                {inferenceModelIDs.has(v.model_id) && onBenchmark ? <button className="link-btn" type="button" title="进入推理验收" onClick={() => onBenchmark(v)}><Gauge size={13} /></button> : null}
                {inferenceModelIDs.has(v.model_id) && onRelease ? <button className="link-btn" type="button" title="进入发布" onClick={() => onRelease(v)}><Rocket size={13} /></button> : null}
                <select
                  value={v.status}
                  disabled={registry.setStatus.isPending}
                  onChange={(e) => registry.setStatus.mutate({ id: v.id, status: e.target.value })}
                >
                  <option value="registered">registered</option>
                  <option value="active">active</option>
                  <option value="deprecated">deprecated</option>
                </select>
                <button
                  className="link-btn danger"
                  type="button"
                  disabled={registry.remove.isPending}
                  title="注销版本"
                  onClick={() => {
                    if (window.confirm(`注销 ${v.model_id} ${v.version}？`)) registry.remove.mutate(v.id);
                  }}
                >
                  <Trash2 size={13} />
                </button>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function ArtifactUpload({ id, registry }: { id: string; registry: ModelRegistry }) {
  const ref = useRef<HTMLInputElement>(null);
  return (
    <>
      <button className="link-btn" type="button" disabled={registry.uploadArtifact.isPending} onClick={() => ref.current?.click()}>
        <UploadCloud size={13} /> {registry.uploadArtifact.isPending ? "上传中…" : "上传产物"}
      </button>
      <input
        ref={ref}
        type="file"
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) registry.uploadArtifact.mutate({ id, file });
          e.target.value = "";
        }}
      />
    </>
  );
}
