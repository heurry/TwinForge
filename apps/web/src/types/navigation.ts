export type Page =
  | "dashboard"
  | "services"
  | "kubernetes"
  | "observability"
  | "aiOps"
  | "agents"
  | "agentStudio"
  | "knowledge"
  | "support"
  | "evals"
  | "feedback"
  | "benchmarks"
  | "release"
  | "config"
  | "pipelines"
  | "datasets"
  | "models"
  | "routing"
  | "storage"
  | "training"
  | "settings";

export const pages: Page[] = [
  "dashboard",
  "services",
  "kubernetes",
  "observability",
  "aiOps",
  "agents",
  "agentStudio",
  "knowledge",
  "support",
  "evals",
  "feedback",
  "benchmarks",
  "release",
  "config",
  "pipelines",
  "datasets",
  "models",
  "routing",
  "storage",
  "training",
  "settings"
];

// 旧 AI 应用页面继续支持直链，但不再出现在基础设施平台的主导航和全局搜索中。
export const navigationPages: Page[] = pages.filter(
  (page) => !(["pipelines", "support", "evals", "feedback"] as Page[]).includes(page)
);

export const pageLabels: Record<Page, string> = {
  dashboard: "平台总览",
  services: "服务目录",
  kubernetes: "集群与资源",
  config: "配置中心",
  pipelines: "CI/CD 流水线",
  observability: "可观测中心",
  aiOps: "智能诊断",
  agents: "Agent 运行",
  agentStudio: "Agent Studio",
  knowledge: "诊断知识库",
  support: "智能客服",
  evals: "检索评测",
  feedback: "反馈回流",
  benchmarks: "推理服务",
  release: "发布中心 / CI/CD",
  datasets: "数据资产",
  models: "模型与版本",
  routing: "流量策略",
  storage: "存储总览",
  training: "训练任务",
  settings: "平台状态"
};

// Page <-> URL path 映射（react-router）。
// 真实路由让深链、浏览器前进/后退、面包屑成为可能；UI 仍以 Page 为内部标识。
export const pagePaths: Record<Page, string> = {
  dashboard: "/dashboard",
  services: "/services",
  kubernetes: "/kubernetes",
  config: "/config",
  pipelines: "/pipelines",
  observability: "/observability",
  aiOps: "/ai-ops",
  agents: "/agents",
  agentStudio: "/agent-studio",
  knowledge: "/knowledge",
  support: "/support",
  evals: "/evals",
  feedback: "/feedback",
  benchmarks: "/benchmarks",
  release: "/release",
  datasets: "/datasets",
  models: "/models",
  routing: "/routing",
  storage: "/storage",
  training: "/training",
  settings: "/settings"
};

const pathToPageEntries = Object.entries(pagePaths) as Array<[Page, string]>;

/** 由 URL pathname 解析出 Page；兼容旧 ?page= 取值；无匹配回退 dashboard。 */
export function pageFromPath(pathname: string, legacyPageParam?: string | null): Page {
  const normalized = pathname.replace(/\/+$/, "") || "/";
  const matched = pathToPageEntries.find(([, path]) => path === normalized);
  if (matched) return matched[0];

  if (legacyPageParam) {
    if (pages.includes(legacyPageParam as Page)) return legacyPageParam as Page;
    const legacyMap: Record<string, Page> = {
      chat: "aiOps",
      knowledge: "knowledge",
      benchmarks: "benchmarks",
      metrics: "observability",
      evaluations: "models"
    };
    if (legacyMap[legacyPageParam]) return legacyMap[legacyPageParam];
  }
  return "dashboard";
}
