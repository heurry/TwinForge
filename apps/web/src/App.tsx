import { useEffect, useRef } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { DeliveryContextBar } from "./components/layout/DeliveryContextBar";
import { SettingsStatusPanel } from "./components/layout/SettingsStatusPanel";
import { AppSidebar } from "./components/layout/AppSidebar";
import { PlatformTopBar } from "./components/layout/PlatformTopBar";
import { LoginModal } from "./components/auth/LoginModal";
import { useAuth } from "./lib/useAuth";
import { AIOpsPage } from "./pages/AIOpsPage";
import { AgentOperationsPage } from "./pages/AgentOperationsPage";
import { AgentStudioPage } from "./pages/AgentStudioPage";
import { BenchmarksPage } from "./pages/BenchmarksPage";
import { KubernetesPage } from "./pages/KubernetesPage";
import { ConfigCenterPage } from "./pages/ConfigCenterPage";
import { KnowledgePage } from "./pages/KnowledgePage";
import { CustomerSupportPage } from "./pages/CustomerSupportPage";
import { RetrievalEvalPage } from "./pages/RetrievalEvalPage";
import { FeedbackReflowPage } from "./pages/FeedbackReflowPage";
import { ObservabilityPage } from "./pages/ObservabilityPage";
import { ModelsPage } from "./pages/ModelsPage";
import { RoutingPage } from "./pages/RoutingPage";
import { StoragePage } from "./pages/StoragePage";
import { TrainingPage } from "./pages/TrainingPage";
import { DataAssetsPage } from "./pages/DataAssetsPage";
import { ModelReleasePage } from "./pages/ModelReleasePage";
import { PlatformOverviewPage } from "./pages/PlatformOverviewPage";
import { ServicesPage } from "./pages/ServicesPage";
import { SettingsPage } from "./pages/SettingsPage";
import { pageFromPath, pagePaths, type Page } from "./types/navigation";

export function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const auth = useAuth();
  const mainRef = useRef<HTMLElement>(null);

  // URL 是导航的唯一真实来源；Page 仅作 UI 内部标识。
  const page = pageFromPath(location.pathname, new URLSearchParams(location.search).get("page"));

  // 根路径或旧 ?page= 链接：规范化到真实路由路径。
  useEffect(() => {
    const normalized = pagePaths[page];
    if (location.pathname !== normalized) {
      const params = new URLSearchParams(location.search);
      params.delete("page");
      navigate({ pathname: normalized, search: params.toString(), hash: location.hash }, { replace: true });
    }
    // 仅在首次/路径与解析结果不一致时纠正，避免循环。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 每个工作台页面都应从顶部开始；同页查询参数变化则保留操作位置。
  useEffect(() => {
    mainRef.current?.scrollTo({ top: 0, left: 0, behavior: "auto" });
  }, [location.pathname]);

  function setPage(nextPage: Page) {
    const params = new URLSearchParams(location.search);
    params.delete("page");
    navigate({ pathname: pagePaths[nextPage], search: params.toString() });
  }

  return (
    <div className="infra-platform-shell">
      <PlatformTopBar />
      <AppSidebar page={page} setPage={setPage} />
      <main ref={mainRef} className="infra-main">
        <DeliveryContextBar />
        {renderPage(page, setPage)}
      </main>
      {page === "settings" && <SettingsStatusPanel />}
      {auth.authEnabled && !auth.authenticated ? <LoginModal auth={auth} /> : null}
    </div>
  );
}

function renderPage(page: Page, setPage: (page: Page) => void) {
  switch (page) {
    case "dashboard": return <PlatformOverviewPage setPage={setPage} />;
    case "services": return <ServicesPage />;
    case "kubernetes": return <KubernetesPage />;
    case "config": return <ConfigCenterPage />;
    case "pipelines": return <ModelReleasePage initialTab="pipeline" />;
    case "observability": return <ObservabilityPage />;
    case "aiOps": return <AIOpsPage />;
    case "agents": return <AgentOperationsPage />;
    case "agentStudio": return <AgentStudioPage />;
    case "knowledge": return <KnowledgePage />;
    case "support": return <CustomerSupportPage />;
    case "evals": return <RetrievalEvalPage />;
    case "feedback": return <FeedbackReflowPage />;
    case "benchmarks": return <BenchmarksPage />;
    case "release": return <ModelReleasePage />;
    case "datasets": return <DataAssetsPage />;
    case "models": return <ModelsPage />;
    case "routing": return <RoutingPage />;
    case "storage": return <StoragePage />;
    case "training": return <TrainingPage />;
    case "settings": return <SettingsPage />;
  }
}
