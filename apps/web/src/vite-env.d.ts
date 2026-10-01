/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 后端 API 基础地址，留空表示同源（开发走 vite 代理）。 */
  readonly VITE_API_BASE?: string;
  readonly VITE_AGENT_TENANT?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
