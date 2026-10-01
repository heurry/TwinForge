// E3：路由策略创建/编辑抽屉。多候选加权（A/B 灰度）+ 可选影子目标。
import { useState } from "react";
import { Plus, Trash2 } from "lucide-react";

import { Drawer } from "../common/Drawer";
import type { Routing } from "../../lib/useRouting";
import type { RoutingPolicy, RoutingVariant, SavePolicyInput } from "../../types/routing";

type VariantForm = { label: string; endpoint: string; model: string; weight: string };

function toForm(v: RoutingVariant): VariantForm {
  return { label: v.label, endpoint: v.endpoint, model: v.model ?? "", weight: String(v.weight) };
}

export function PolicyDrawer({ routing, editing, onClose }: { routing: Routing; editing: RoutingPolicy | null; onClose: () => void }) {
  const isEdit = editing !== null;
  const endpoints = routing.endpoints.data ?? [];

  const [name, setName] = useState(editing?.name ?? "");
  const [description, setDescription] = useState(editing?.description ?? "");
  const [enabled, setEnabled] = useState(editing?.enabled ?? true);
  const [variants, setVariants] = useState<VariantForm[]>(
    editing?.variants.map(toForm) ?? [
      { label: "stable", endpoint: "", model: "", weight: "90" },
      { label: "canary", endpoint: "", model: "", weight: "10" }
    ]
  );
  const [shadowOn, setShadowOn] = useState(Boolean(editing?.shadow));
  const [shadow, setShadow] = useState({
    label: editing?.shadow?.label ?? "shadow",
    endpoint: editing?.shadow?.endpoint ?? "",
    model: editing?.shadow?.model ?? ""
  });

  const setVariant = (i: number, k: keyof VariantForm, val: string) =>
    setVariants((vs) => vs.map((v, idx) => (idx === i ? { ...v, [k]: val } : v)));
  const addVariant = () => setVariants((vs) => [...vs, { label: "", endpoint: "", model: "", weight: "0" }]);
  const removeVariant = (i: number) => setVariants((vs) => vs.filter((_, idx) => idx !== i));

  const weightTotal = variants.reduce((sum, v) => sum + (Number(v.weight) || 0), 0);
  const valid =
    name.trim() !== "" &&
    variants.length > 0 &&
    variants.every((v) => v.label.trim() !== "" && v.endpoint.trim() !== "" && Number(v.weight) >= 0) &&
    weightTotal > 0 &&
    (!shadowOn || shadow.endpoint.trim() !== "");

  const submit = () => {
    const input: SavePolicyInput = {
      name: name.trim(),
      description: description.trim(),
      enabled,
      variants: variants.map((v) => ({
        label: v.label.trim(),
        endpoint: v.endpoint.trim(),
        model: v.model.trim() || undefined,
        weight: Number(v.weight) || 0
      })),
      shadow: shadowOn && shadow.endpoint.trim() ? { label: shadow.label.trim() || "shadow", endpoint: shadow.endpoint.trim(), model: shadow.model.trim() || undefined } : null
    };
    routing.save.mutate({ input, isEdit }, { onSuccess: onClose });
  };

  return (
    <Drawer open title={isEdit ? `编辑策略 ${editing!.name}` : "新建路由策略"} subtitle="模型版本与服务实例 / A-B 灰度 / 影子流量" onClose={onClose}>
      <datalist id="routing-endpoints">
        {endpoints.map((e) => <option key={e} value={e} />)}
      </datalist>

      <div className="drawer-section">
        <input className="drawer-input" placeholder="策略名（如 qwen3-ab）" value={name} disabled={isEdit} onChange={(e) => setName(e.target.value)} />
        <input className="drawer-input" placeholder="描述（可空）" value={description} onChange={(e) => setDescription(e.target.value)} />
        <label className="routing-toggle">
          <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} /> 启用（关闭后数据面返回 503）
        </label>
      </div>

      <div className="drawer-section">
        <div className="routing-section-head">
          <strong>候选版本 / 服务实例（A/B 与灰度）</strong>
          <span className={weightTotal > 0 ? "cell-subtle" : "routing-warn"}>权重合计 {weightTotal}</span>
        </div>
        {variants.map((v, i) => (
          <div className="routing-variant-row" key={i}>
            <input className="drawer-input" placeholder="候选标识（stable/canary）" value={v.label} onChange={(e) => setVariant(i, "label", e.target.value)} />
            <input className="drawer-input" list="routing-endpoints" placeholder="服务实例 endpoint" value={v.endpoint} onChange={(e) => setVariant(i, "endpoint", e.target.value)} />
            <input className="drawer-input" placeholder="模型版本（可选）" value={v.model} onChange={(e) => setVariant(i, "model", e.target.value)} />
            <input className="drawer-input routing-weight" type="number" min={0} placeholder="权重" value={v.weight} onChange={(e) => setVariant(i, "weight", e.target.value)} />
            <button className="link-btn danger" type="button" title="移除候选" disabled={variants.length <= 1} onClick={() => removeVariant(i)}>
              <Trash2 size={13} />
            </button>
          </div>
        ))}
        <button className="link-btn" type="button" onClick={addVariant}><Plus size={13} /> 添加候选版本 / 实例</button>
      </div>

      <div className="drawer-section">
        <label className="routing-toggle">
          <input type="checkbox" checked={shadowOn} onChange={(e) => setShadowOn(e.target.checked)} /> 影子流量（镜像同一请求、丢弃响应、只采指标）
        </label>
        {shadowOn && (
          <div className="routing-shadow-row">
            <input className="drawer-input" placeholder="标签" value={shadow.label} onChange={(e) => setShadow((s) => ({ ...s, label: e.target.value }))} />
            <input className="drawer-input" list="routing-endpoints" placeholder="endpoint" value={shadow.endpoint} onChange={(e) => setShadow((s) => ({ ...s, endpoint: e.target.value }))} />
            <input className="drawer-input" placeholder="model（可空）" value={shadow.model} onChange={(e) => setShadow((s) => ({ ...s, model: e.target.value }))} />
          </div>
        )}
        {shadowOn && !routing.list.data?.shadow_enabled && (
          <p className="routing-hint">注意：平台 <code>ROUTING_SHADOW_ENABLED=false</code>，镜像暂不会真正发起（策略仍会保存）。</p>
        )}
      </div>

      <div className="drawer-section">
        <button className="infra-action-btn" type="button" disabled={!valid || routing.save.isPending} onClick={submit}>
          {routing.save.isPending ? "保存中…" : isEdit ? "保存修改" : "创建策略"}
        </button>
      </div>
    </Drawer>
  );
}
