// E3：路由策略卡片列表。每卡展示加权候选（配置权重 vs 实时份额）+ 全量/回滚/编辑/删除 + A/B 对比。
import { useState } from "react";
import { ChevronDown, ChevronRight, Pencil, RotateCcw, Trash2 } from "lucide-react";

import { StatusBadge } from "../common/PlatformPrimitives";
import { EmptyState, ErrorState, Skeleton } from "../common/FeedbackStates";
import { ABComparison } from "./ABComparison";
import type { Routing } from "../../lib/useRouting";
import type { RoutingPolicy, VariantStat } from "../../types/routing";

export function RoutingPolicyList({ routing, onEdit }: { routing: Routing; onEdit: (p: RoutingPolicy) => void }) {
  const { list } = routing;
  if (list.isLoading) return <Skeleton rows={4} />;
  if (list.isError) return <ErrorState error={list.error} onRetry={list.refetch} />;
  const policies = list.data?.policies ?? [];
  if (policies.length === 0) {
    return <EmptyState title="暂无模型或服务路由策略" description="点击「新建路由策略」，添加稳定版、灰度版等候选实例并设置分流权重" />;
  }
  return (
    <div className="routing-card-list">
      {policies.map((p) => <PolicyCard key={p.name} policy={p} routing={routing} onEdit={onEdit} />)}
    </div>
  );
}

function PolicyCard({ policy, routing, onEdit }: { policy: RoutingPolicy; routing: Routing; onEdit: (p: RoutingPolicy) => void }) {
  const [open, setOpen] = useState(false);
  const liveByLabel = new Map<string, VariantStat>((policy.live ?? []).map((s) => [s.label, s]));
  const totalWeight = policy.variants.reduce((sum, v) => sum + v.weight, 0) || 1;
  const hasModelVersions = policy.variants.some((variant) => Boolean(variant.model));
  const releaseManaged = policy.metadata?.source === "model_release";
  const aibrixManaged = releaseManaged && policy.metadata?.release_target === "aibrix";
  const resourceTransition = policy.metadata?.resource_transition;
  const rolloutPhase = policy.metadata?.rollout_phase === "full" ? "全量" : policy.metadata?.rollout_phase === "canary" ? "灰度" : "";

  return (
    <article className="routing-card">
      <header className="routing-card-head">
        <div>
          <strong>{policy.name}</strong>
          <StatusBadge status={policy.enabled ? "enabled" : "disabled"} />
          <span className="routing-policy-kind">{hasModelVersions ? "模型多版本" : "服务实例"}</span>
          {releaseManaged ? <span className="routing-release-tag">发布中心同步{rolloutPhase ? ` · ${rolloutPhase}` : ""}</span> : null}
          {policy.shadow ? <span className="routing-shadow-tag">影子流量 → {policy.shadow.endpoint}</span> : null}
          {policy.description ? <p className="cell-subtle">{policy.description}</p> : null}
        </div>
        <div className="routing-card-actions">
          <button className="link-btn" type="button" onClick={() => onEdit(policy)}><Pencil size={13} /> 编辑</button>
          <button
            className="link-btn"
            type="button"
            disabled={routing.rollback.isPending}
            title="回滚到全量前的权重"
            onClick={() => routing.rollback.mutate(policy.name)}
          >
            <RotateCcw size={13} /> 回滚
          </button>
          <button
            className="link-btn danger"
            type="button"
            disabled={routing.remove.isPending}
            title="删除策略"
            onClick={() => {
              if (window.confirm(`删除策略 ${policy.name}？（样本保留）`)) routing.remove.mutate(policy.name);
            }}
          >
            <Trash2 size={13} />
          </button>
        </div>
      </header>

      <div className="routing-variants">
        <div className="routing-variant header">
          <span>候选版本 / 服务实例</span><span>配置权重与实时流量（近 1 小时）</span><span>发布操作</span>
        </div>
        {policy.variants.map((v) => {
          const live = liveByLabel.get(v.label);
          const cfgShare = (v.weight / totalWeight) * 100;
          const liveShare = live ? live.share * 100 : 0;
          const hasLive = Boolean(live && live.count > 0);
          const alreadyFull = v.weight === totalWeight;
          const needsResourceCleanup = Boolean(aibrixManaged && alreadyFull && resourceTransition?.cleanup_complete !== true);
          return (
            <div className="routing-variant" key={v.label}>
              <div className="routing-variant-meta">
                <strong>{v.label} <StatusBadge status={policy.endpoint_status?.[v.endpoint] || "unknown"} /></strong>
                <span>{v.model || "继承请求模型"}</span>
                <small>{v.endpoint}</small>
              </div>
              <div className="routing-share">
                <div className="routing-share-line">
                  <span>配置</span><div className="routing-share-bar"><i className="cfg" style={{ width: `${cfgShare}%` }} /></div><strong>{cfgShare.toFixed(0)}%</strong>
                </div>
                <div className="routing-share-line">
                  <span>实时</span><div className="routing-share-bar"><i className="live" style={{ width: `${liveShare}%` }} /></div><strong>{hasLive ? `${liveShare.toFixed(0)}%` : "暂无"}</strong>
                </div>
                <small className="routing-live-meta">权重 {v.weight}{hasLive && live ? ` · ${live.count} 请求 · P95 ${live.p95_ms}ms · 错误率 ${(live.error_rate * 100).toFixed(2)}%` : " · 暂无真实请求样本"}</small>
              </div>
              <button
                className="link-btn"
                type="button"
                disabled={routing.promote.isPending || (alreadyFull && !needsResourceCleanup)}
                title={needsResourceCleanup ? "权重已全量，缩容零权重旧模型并释放 GPU" : "全量到该候选，然后回收旧稳定模型"}
                onClick={() => routing.promote.mutate({ name: policy.name, label: v.label })}
              >
                {needsResourceCleanup ? "回收旧版本" : "切到全量"}
              </button>
            </div>
          );
        })}
      </div>

      <button className="routing-cmp-toggle" type="button" onClick={() => setOpen((o) => !o)}>
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />} 查看 A/B 与影子指标对比
      </button>
      {open && <ABComparison policyName={policy.name} />}
    </article>
  );
}
