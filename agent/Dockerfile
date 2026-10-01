# Go Infra Agent —— 多阶段构建，运行期为极小静态镜像。

# ---- build stage ----
FROM golang:1.22-alpine AS build
ARG GOPROXY=https://proxy.golang.org|https://goproxy.cn|direct
ENV GOPROXY=${GOPROXY}
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY *.go ./
RUN CGO_ENABLED=0 GOOS=linux go build -o /out/infra-agent .

# ---- runtime stage ----
# 用 Debian 系运行期，配合 compose 挂载宿主 nvidia-smi/NVML；不依赖远端 CUDA base。
# 无 GPU/无挂载时 gpu.go 会优雅降级并返回诊断原因。
FROM golang:1.22
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=utility
WORKDIR /app
COPY --from=build /out/infra-agent .
EXPOSE 8090
ENTRYPOINT ["/app/infra-agent"]
