#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

show_help() {
  cat <<'EOF'
TwinForge 一键启动入口

用法：
  ./start.sh                 构建并启动完整应用栈
  ./start.sh up              同上
  ./start.sh restart         重启完整应用栈
  ./start.sh status          查看容器与端到端健康状态
  ./start.sh logs            持续查看所有服务日志
  ./start.sh down            停止服务，保留数据卷

常用环境变量：
  SKIP_BUILD=1               直接使用已有镜像快速启动
                              默认先构建；仓库网络失败且镜像齐全时自动回退
  ENABLE_OBSERVABILITY=1     同时启动 Grafana、Prometheus、Tempo
  ALLOW_MINIKUBE_DOWN=1      明确允许无 minikube 的降级模式
  HEALTH_TIMEOUT=300         启动健康检查超时（秒）
  AGENT_WORKSPACE_ALLOW_WRITE=true
                              本地 Agent 是否启用 write/edit（生产应结合审批）
  AGENT_WORKSPACE_UID/GID     Workspace Worker 的宿主文件 UID/GID，默认 1000
  AGENT_MCP_ALLOWED_HOSTS     MCP Server host:port 白名单，API 注册与 Worker 执行共用
  AGENT_A2A_DEFAULT_AGENT_ID  多个已发布 Agent 时用于标准 well-known 发现的默认 Agent UUID

Serving 说明：
  正式应用链路固定使用 minikube NodePort :30080；宿主 :8010 只是可选调试
  port-forward。`./start.sh status` 会分别核验 Workload、Gateway 和模型路由。
EOF
}

case "${1:-up}" in
  -h|--help|help)
    show_help
    ;;
  up|start|down|stop|restart|status|logs)
    exec bash scripts/run_full_stack.sh "${1:-up}"
    ;;
  *)
    show_help >&2
    exit 2
    ;;
esac
