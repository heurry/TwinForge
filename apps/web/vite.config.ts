import { defineConfig } from "vite";

// 开发代理：默认指向 Go 控制面（:8081，单一入口）。Go 原生服务 ops 端点，并把
// Python 单体仍拥有的 AI/服务面端点（aiops/knowledge/benchmarks/proxy/…）反向代理透传。
// 迁移期可通过 VITE_PROXY_TARGET 切回旧后端对照：
// - 旧控制面（已退役）：       VITE_PROXY_TARGET=http://127.0.0.1:8080
// - 旧 FastAPI 单体（直连对照）：VITE_PROXY_TARGET=http://127.0.0.1:8088
const proxyTarget = process.env.VITE_PROXY_TARGET || "http://127.0.0.1:8081";
const agentProxyTarget = process.env.VITE_AGENT_PROXY_TARGET || "http://127.0.0.1:8180";

export default defineConfig({
  server: {
    proxy: {
      "/api": proxyTarget,
      "/agent-api": { target: agentProxyTarget, rewrite: (path) => path.replace(/^\/agent-api/, "") }
    }
  }
});
