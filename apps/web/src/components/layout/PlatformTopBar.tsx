import { useEffect, useRef, useState, type FormEvent } from "react";
import { useQuery } from "@tanstack/react-query";
import { Activity, AlertTriangle, ChevronDown, HelpCircle, Search, ShieldCheck, Sparkles, UserRound } from "lucide-react";

import { api } from "../../lib/api";
import { relativeTime } from "../../lib/format";
import { cn } from "../../lib/utils";
import { useAuth } from "../../lib/useAuth";
import { useGoToPage } from "../../lib/useGoToPage";
import { navigationPages, pageLabels, type Page } from "../../types/navigation";
import type { Incident } from "../../types/ops";

type AlertItem = {
  id: string;
  severity: string;
  name: string;
  service: string;
  metric: string;
  value: string;
  threshold: string;
  status: string;
};
type AuditItem = {
  id: string;
  action: string;
  resource_type: string | null;
  resource_id: string | null;
  actor_id: string | null;
  created_at: string;
};

// 审计资源类型 → 跳转页面。
const RESOURCE_PAGE: Record<string, Page> = {
  config_item: "config",
  deployment: "release",
  incident: "aiOps",
  service_instance: "services"
};

const ACTION_LABEL: Record<string, string> = {
  "deployment.trigger": "触发部署",
  "deployment.finish": "完成部署",
  "deployment.rollback": "回滚部署",
  "config.publish": "发布配置",
  "config.rollback": "回滚配置",
  "config.create": "新建配置",
  "incident.create": "创建故障",
  "incident.ack": "确认故障",
  "incident.resolve": "解决故障",
  "service_instance.register": "注册服务",
  "service_instance.healthcheck": "健康检查",
  "ai.diagnose": "AI 诊断"
};

function sevColor(severity: string): string {
  const s = severity.toLowerCase();
  if (s.includes("crit") || s.includes("严重") || s.includes("danger")) return "#dc2626";
  if (s.includes("warn") || s.includes("警告") || s.includes("major")) return "#d97706";
  return "#2563eb";
}

export function PlatformTopBar() {
  const goTo = useGoToPage();
  const auth = useAuth();
  const [openMenu, setOpenMenu] = useState<null | "notif" | "activity">(null);
  const [searchText, setSearchText] = useState("");
  const actionsRef = useRef<HTMLDivElement | null>(null);
  const searchRef = useRef<HTMLInputElement | null>(null);

  // 全局搜索 = 客户端页面跳转（无后端搜索服务）：匹配导航标签/键，回车跳转首个命中。
  function onSearch(event: FormEvent) {
    event.preventDefault();
    const q = searchText.trim().toLowerCase();
    if (!q) return;
    const hit = navigationPages.find(
      (page) => pageLabels[page].toLowerCase().includes(q) || page.toLowerCase().includes(q)
    );
    if (hit) {
      goTo(hit);
      setSearchText("");
    }
  }

  const alertsQuery = useQuery({
    queryKey: ["alerts"],
    queryFn: () => api<{ alerts: AlertItem[]; summary: Record<string, number> }>("/api/alerts"),
    refetchInterval: 15000
  });
  const platformQuery = useQuery({
    queryKey: ["metrics", "current"],
    queryFn: () => api<{ gpu?: unknown[]; kubernetes?: { available?: boolean } }>("/api/metrics/current"),
    refetchInterval: 15000
  });
  const incidentsQuery = useQuery({
    queryKey: ["incidents", "topbar"],
    queryFn: () => api<{ incidents: Incident[] }>("/api/incidents"),
    refetchInterval: 15000
  });
  const auditQuery = useQuery({
    queryKey: ["audit", "recent"],
    queryFn: () => api<{ events: AuditItem[] }>("/api/audit/events?limit=12"),
    refetchInterval: 20000,
    enabled: openMenu === "activity"
  });

  const alerts = alertsQuery.data?.alerts ?? [];
  const openIncidents = (incidentsQuery.data?.incidents ?? []).filter((item) => item.status !== "resolved");
  const notifCount = alerts.length + openIncidents.length;
  const events = auditQuery.data?.events ?? [];

  // 点击面板外 / Esc 关闭。
  useEffect(() => {
    if (!openMenu) return;
    const onClick = (event: MouseEvent) => {
      if (actionsRef.current && !actionsRef.current.contains(event.target as Node)) setOpenMenu(null);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpenMenu(null);
    };
    document.addEventListener("mousedown", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [openMenu]);

  useEffect(() => {
    const focusSearch = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        searchRef.current?.focus();
      }
    };
    document.addEventListener("keydown", focusSearch);
    return () => document.removeEventListener("keydown", focusSearch);
  }, []);

  function navTo(page: Page) {
    setOpenMenu(null);
    goTo(page);
  }

  return (
    <header className="infra-topbar">
      <div className="infra-topbar-brand">
        <span className="infra-brand-mark"><Sparkles size={16} /></span>
        <strong>TwinForge</strong>
      </div>
      <form className="infra-global-search" onSubmit={onSearch}>
        <Search size={15} />
        <input ref={searchRef} onChange={(event) => setSearchText(event.target.value)} placeholder="搜索：训练 / 推理服务 / AIOps / 配置中心…" value={searchText} />
        <kbd>⌘ K</kbd>
      </form>
      <div className="infra-topbar-actions" ref={actionsRef}>
        <div className="infra-env-select" title="当前部署环境">
          <span>本地 Minikube · {platformQuery.data?.gpu?.length ? `${platformQuery.data.gpu.length} GPU` : "GPU 检测中"}</span>
        </div>
        <button className="infra-icon-action" onClick={() => goTo("settings")} title="平台状态与安全设置" type="button"><ShieldCheck size={16} /></button>

        <div className="topbar-menu-anchor">
          <button
            className={cn("infra-icon-action", openMenu === "activity" && "active")}
            onClick={() => setOpenMenu((m) => (m === "activity" ? null : "activity"))}
            title="活动"
            type="button"
          >
            <Activity size={16} />
          </button>
          {openMenu === "activity" && (
            <div className="topbar-menu">
              <header>最近活动</header>
              <div className="topbar-menu-list">
                {auditQuery.isLoading ? (
                  <p className="topbar-menu-empty">加载中…</p>
                ) : events.length === 0 ? (
                  <p className="topbar-menu-empty">暂无平台操作记录</p>
                ) : (
                  events.map((event) => (
                    <button
                      className="topbar-menu-item"
                      key={event.id}
                      onClick={() => navTo(RESOURCE_PAGE[event.resource_type ?? ""] ?? "dashboard")}
                      type="button"
                    >
                      <span className="topbar-menu-dot" style={{ background: "#2563eb" }} />
                      <span className="topbar-menu-text">
                        <strong>{ACTION_LABEL[event.action] ?? event.action}</strong>
                        <small>{[event.resource_id, event.actor_id].filter(Boolean).join(" · ") || event.resource_type || "—"} · {relativeTime(event.created_at)}</small>
                      </span>
                    </button>
                  ))
                )}
              </div>
            </div>
          )}
        </div>

        <div className="topbar-menu-anchor">
          <button
            className={cn("infra-icon-action", notifCount > 0 && "has-dot", openMenu === "notif" && "active")}
            onClick={() => setOpenMenu((m) => (m === "notif" ? null : "notif"))}
            title="通知"
            type="button"
          >
            <AlertTriangle size={16} />
          </button>
          {openMenu === "notif" && (
            <div className="topbar-menu">
              <header>通知 · {notifCount} 条</header>
              <div className="topbar-menu-list">
                {notifCount === 0 ? (
                  <p className="topbar-menu-empty">暂无未处理告警或故障</p>
                ) : (
                  <>
                    {openIncidents.map((incident) => (
                      <button className="topbar-menu-item" key={`inc-${incident.id}`} onClick={() => navTo("aiOps")} type="button">
                        <span className="topbar-menu-dot" style={{ background: sevColor(incident.severity) }} />
                        <span className="topbar-menu-text">
                          <strong>{incident.title}</strong>
                          <small>故障 · {incident.summary ?? incident.status} · {relativeTime(incident.created_at)}</small>
                        </span>
                      </button>
                    ))}
                    {alerts.map((alert) => (
                      <button className="topbar-menu-item" key={`alert-${alert.id}`} onClick={() => navTo("observability")} type="button">
                        <span className="topbar-menu-dot" style={{ background: sevColor(alert.severity) }} />
                        <span className="topbar-menu-text">
                          <strong>{alert.name}</strong>
                          <small>{alert.service} · {alert.metric} {alert.value}/{alert.threshold}</small>
                        </span>
                      </button>
                    ))}
                  </>
                )}
              </div>
            </div>
          )}
        </div>

        <button className="infra-icon-action" onClick={() => goTo("knowledge")} title="知识库与帮助文档" type="button"><HelpCircle size={16} /></button>
        {auth.authEnabled && auth.authenticated ? (
          <span className="infra-user-chip" title={`${auth.subject} · ${auth.role}`}>
            <UserRound size={14} />
            <span className="infra-user-name">{auth.subject}</span>
            <em className={cn("infra-user-role", auth.role)}>{auth.role}</em>
            <button className="infra-user-logout" type="button" title="退出登录" onClick={auth.logout}>退出</button>
          </span>
        ) : (
          <>
            <span className="infra-user-avatar"><UserRound size={16} /><small>{auth.authenticated ? (auth.subject?.[0]?.toUpperCase() ?? "U") : "A"}</small></span>
            <ChevronDown className="infra-user-caret" size={14} />
          </>
        )}
      </div>
    </header>
  );
}
