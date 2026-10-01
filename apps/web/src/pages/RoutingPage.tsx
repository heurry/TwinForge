// E3：模型与服务路由策略控制台。管理多版本权重、A/B 灰度、影子流量和实时分布。
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Activity, GitBranch, Layers3, Plus, Radio, RefreshCw } from "lucide-react";

import { PageHeader } from "../components/common/PlatformPrimitives";
import { RoutingPolicyList } from "../components/routing/RoutingPolicyList";
import { PolicyDrawer } from "../components/routing/PolicyDrawer";
import { api } from "../lib/api";
import { useRouting } from "../lib/useRouting";
import type { Metrics } from "../types/platform";
import type { RoutingPolicy } from "../types/routing";

export function RoutingPage() {
  const routing = useRouting();
  const metrics = useQuery({
    queryKey: ["metrics", "current", "routing"],
    queryFn: () => api<Metrics>("/api/metrics/current"),
    refetchInterval: 5000,
  });
  const [drawer, setDrawer] = useState<{ open: boolean; editing: RoutingPolicy | null }>({ open: false, editing: null });
  const policies = routing.list.data?.policies ?? [];
  const shadowEnabled = routing.list.data?.shadow_enabled ?? false;
  const enabledPolicies = policies.filter((policy) => policy.enabled).length;
  const variantCount = policies.reduce((count, policy) => count + policy.variants.length, 0);
  const liveSamples = policies.reduce((count, policy) => count + (policy.live ?? []).reduce((sum, item) => sum + item.count, 0), 0);
  const gatewayPods = (metrics.data?.target_pod_stats ?? []).filter((item) => item.request_count > 0);
  const gatewaySamples = gatewayPods.reduce((sum, item) => sum + item.request_count, 0);
  const metricsWindow = Math.max(1, Math.round((metrics.data?.window_seconds ?? 600) / 60));

  return (
    <section className="infra-page routing-page">
      <PageHeader
        title="流量策略"
        subtitle="统一管理模型与服务的路由策略，配置多版本权重、A/B、灰度和影子流量，并对照配置权重与最近 1 小时实际分布"
        actions={
          <div className="storage-actions">
            <button className="console-refresh" type="button" onClick={() => { routing.list.refetch(); metrics.refetch(); }}>
              <RefreshCw className={routing.list.isFetching || metrics.isFetching ? "spinning" : undefined} size={14} /> 刷新
            </button>
            <button className="console-refresh primary" type="button" onClick={() => setDrawer({ open: true, editing: null })}>
              <Plus size={14} /> 新建路由策略
            </button>
          </div>
        }
      />

      <section className="routing-overview-grid" aria-label="路由策略概况">
        <article><GitBranch size={18} /><span><small>路由策略</small><strong>{policies.length}</strong><em>{enabledPolicies} 条已启用</em></span></article>
        <article><Layers3 size={18} /><span><small>候选版本 / 实例</small><strong>{variantCount}</strong><em>参与权重分流</em></span></article>
        <article><Activity size={18} /><span><small>近 1 小时请求样本</small><strong>{liveSamples}</strong><em>用于计算实时占比</em></span></article>
        <article><Radio size={18} /><span><small>AIBrix 实际请求</small><strong>{gatewaySamples}</strong><em>近 {metricsWindow} 分钟 · {gatewayPods.length} 个目标 Pod</em></span></article>
      </section>

      {!shadowEnabled ? (
        <p className="routing-banner">
          影子流量全局开关 <code>ROUTING_SHADOW_ENABLED=false</code>。当前可以保存影子目标，但不会复制真实请求；
          开启后才会镜像同一请求、丢弃影子响应并单独采集延迟和错误率。A/B、灰度、全量与回滚不受影响。
        </p>
      ) : null}

      {liveSamples === 0 ? (
        <p className="routing-banner">
          当前 1 小时没有路由代理样本：业务若直接调用 <code>http://127.0.0.1:8020/v1</code>，请求会直达 vLLM，不经过这里的权重分配。
          只有调用 <code>POST /api/routing/&lt;策略名&gt;/v1/chat/completions</code> 才会按候选权重随机分流并记录实时占比。
        </p>
      ) : null}

      {gatewaySamples > 0 && gatewaySamples !== liveSamples ? (
        <p className="routing-banner routing-banner-info">
          当前数据面已有 <strong>{gatewaySamples}</strong> 个 AIBrix 请求，但只有 <strong>{liveSamples}</strong> 个请求经过策略代理。
          AIBrix/auto-router 的流量会显示在下方“网关实际分布”，只有 <code>/api/routing/&lt;策略名&gt;/...</code> 才改变策略候选的实时占比。
        </p>
      ) : null}

      <section className="infra-panel routing-gateway-panel">
        <header className="routing-policy-toolbar">
          <div><strong>AIBrix 网关实际 Pod 分布</strong><small>直接来自推理 Pod 指标，与策略代理样本分开统计；可用于检查负载是否均衡</small></div>
          <span className="routing-gateway-total">{gatewaySamples} 请求 / {metricsWindow} 分钟</span>
        </header>
        {gatewayPods.length ? <div className="routing-gateway-list">
          {gatewayPods.map((pod) => {
            const share = gatewaySamples ? (pod.request_count / gatewaySamples) * 100 : 0;
            return <div className="routing-gateway-row" key={pod.name}>
              <div><strong>{pod.name}</strong><small>{pod.request_count} 请求 · P95 {Math.round(pod.p95_latency_ms ?? 0)}ms · 错误率 {(pod.error_rate * 100).toFixed(2)}%</small></div>
              <div className="routing-gateway-share"><i style={{ width: `${share}%` }} /><span>{share.toFixed(1)}%</span></div>
            </div>;
          })}
        </div> : <p className="routing-gateway-empty">当前窗口没有 AIBrix Pod 请求；通过网关或自动路由发起请求后会自动出现。</p>}
      </section>

      <section className="infra-panel routing-policy-panel">
        <header className="routing-policy-toolbar">
          <div><strong>模型与服务路由策略</strong><small>模型发布会在这里创建或更新生产策略；AIBrix 全量后会回收旧模型 GPU，回滚时先预热旧模型再切换流量</small></div>
          <div className="routing-weight-legend" aria-label="流量分布图例"><span><i className="cfg" />配置权重</span><span><i className="live" />实时流量</span></div>
        </header>
        <RoutingPolicyList routing={routing} onEdit={(p) => setDrawer({ open: true, editing: p })} />
      </section>

      {drawer.open && <PolicyDrawer routing={routing} editing={drawer.editing} onClose={() => setDrawer({ open: false, editing: null })} />}
    </section>
  );
}
