# D3：web 镜像。多阶段——Node 构建 SPA → nginx 提供静态资源 + 反代 /api 到 go-server。
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
# npm 10 may rarely emit "Exit handler never called" yet return success before
# installing binaries on ephemeral CI runners. Verify tsc exists and retry from
# a clean node_modules directory so an incomplete install cannot pass this layer.
RUN set -eu; \
    attempt=1; \
    while [ "$attempt" -le 3 ]; do \
      rm -rf node_modules; \
      if npm ci --no-audit --no-fund && test -x node_modules/.bin/tsc; then \
        exit 0; \
      fi; \
      echo "npm ci attempt ${attempt} failed or incomplete; retrying"; \
      attempt=$((attempt + 1)); \
      sleep 5; \
    done; \
    exit 1
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
# /api 反代上游由 helm 经 GO_SERVER_UPSTREAM 注入；NGINX_ENVSUBST_FILTER 限定只替换该变量，
# 不动 nginx 自身的 $host/$uri 等。
ENV GO_SERVER_UPSTREAM="localhost:8081" \
	AGENT_API_UPSTREAM="localhost:8180" \
	NGINX_ENVSUBST_FILTER="GO_SERVER_UPSTREAM|AGENT_API_UPSTREAM"
COPY nginx.conf /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
