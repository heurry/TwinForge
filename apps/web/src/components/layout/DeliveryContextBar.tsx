import { Boxes, FileCheck2, Gauge, GraduationCap, Settings2, X } from "lucide-react";

import { useDeliveryContext } from "../../lib/useDeliveryContext";
import { useGoToPage } from "../../lib/useGoToPage";

export function DeliveryContextBar() {
  const { context, clear } = useDeliveryContext();
  const goTo = useGoToPage();
  const values = [
    context.modelId ? { label: "模型", value: context.modelId, icon: Boxes } : null,
    context.modelVersionId ? { label: "版本", value: context.modelVersionId, icon: FileCheck2 } : null,
    context.configItemId ? { label: "配置", value: `${context.configItemId}${context.configVersion ? ` · v${context.configVersion}` : ""}`, icon: Settings2 } : null,
    context.trainingJobId ? { label: "训练", value: context.trainingJobId, icon: GraduationCap } : null,
    context.benchmarkRunId ? { label: "压测", value: context.benchmarkRunId, icon: Gauge } : null,
    context.deploymentId ? { label: "发布", value: context.deploymentId, icon: FileCheck2 } : null,
  ].filter(Boolean) as Array<{ label: string; value: string; icon: typeof Boxes }>;

  if (!values.length) return null;
  return (
    <section className="delivery-context-bar" aria-label="当前交付上下文">
      <div className="delivery-context-title">
        <span>{context.deliveryKind === "training" ? "训练交付" : context.deliveryKind === "inference" ? "推理交付" : "当前交付"}</span>
        <strong>上下文已在页面间保留</strong>
      </div>
      <div className="delivery-context-values">
        {values.map(({ label, value, icon: Icon }) => (
          <span key={label} title={value}><Icon size={13} /><small>{label}</small><b>{shortID(value)}</b></span>
        ))}
      </div>
      <div className="delivery-context-actions">
        {context.deliveryKind === "training" ? <button type="button" onClick={() => goTo("training")}>训练任务</button> : null}
        {context.deliveryKind === "inference" ? <button type="button" onClick={() => goTo("benchmarks")}>推理验收</button> : null}
        <button type="button" onClick={() => goTo("aiOps")}>诊断</button>
        <button className="clear" type="button" title="清除当前交付上下文" onClick={clear}><X size={13} /></button>
      </div>
    </section>
  );
}

function shortID(value: string): string {
  if (value.length <= 30) return value;
  return `${value.slice(0, 14)}…${value.slice(-8)}`;
}
