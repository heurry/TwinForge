import {
  Activity,
  Bot,
  Boxes,
  Workflow,
  Cpu,
  Database,
  FlaskConical,
  Gauge,
  GraduationCap,
  HardDrive,
  Home,
  Library,
  Network,
  Rocket,
  Server,
  Settings,
  Split,
  type LucideIcon
} from "lucide-react";

import { cn } from "../../lib/utils";
import type { Page } from "../../types/navigation";

type NavItem = { id: Page; label: string; icon: LucideIcon };
type NavSection = { id: string; label: string; items: NavItem[] };

const navSections: NavSection[] = [
  { id: "workbench", label: "交付工作台", items: [
    { id: "dashboard", label: "平台总览", icon: Home },
    { id: "training", label: "训练任务", icon: GraduationCap },
    { id: "benchmarks", label: "推理服务", icon: Gauge },
  ] },
  { id: "foundation", label: "基础管理", items: [
    { id: "config", label: "配置中心", icon: Database },
    { id: "datasets", label: "数据资产", icon: FlaskConical },
    { id: "models", label: "模型与版本", icon: Cpu },
    { id: "storage", label: "存储总览", icon: HardDrive },
  ] },
  { id: "governance", label: "服务治理", items: [
    { id: "services", label: "服务目录", icon: Server },
    { id: "routing", label: "流量策略", icon: Split },
    { id: "kubernetes", label: "集群与资源", icon: Network },
  ] },
  { id: "delivery", label: "可观测与发布", items: [
    { id: "observability", label: "可观测中心", icon: Activity },
    { id: "release", label: "发布中心", icon: Rocket },
  ] },
  { id: "aiops", label: "AIOps 分析", items: [
    { id: "aiOps", label: "智能诊断", icon: Bot },
    { id: "agents", label: "Agent 运行", icon: Workflow },
    { id: "agentStudio", label: "Agent Studio", icon: Boxes },
    { id: "knowledge", label: "诊断知识库", icon: Library },
  ] },
  { id: "system", label: "系统", items: [
    { id: "settings", label: "平台状态", icon: Settings },
  ] },
];

export function AppSidebar({ page, setPage }: { page: Page; setPage: (page: Page) => void }) {
  return (
    <aside className="infra-sidebar">
      <nav className="infra-nav" aria-label="TwinForge 云原生基础设施平台导航">
        {navSections.map((section) => (
          <section className="infra-nav-section" key={section.id}>
            <div className="infra-nav-section-header"><span className="infra-nav-section-label">{section.label}</span></div>
            {section.items.map((item) => {
              const Glyph = item.icon;
              return (
                <button
                  key={item.id}
                  className={cn("infra-nav-button", page === item.id && "active")}
                  onClick={() => setPage(item.id)}
                  type="button"
                >
                  <Glyph size={17} />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </section>
        ))}
      </nav>
    </aside>
  );
}
