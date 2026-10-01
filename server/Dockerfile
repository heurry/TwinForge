# 多阶段构建：编译期用 golang 镜像，运行期用 distroless/static。
# 构建上下文为 server/（compose 里 context: ../../server），独立模块构建，不受 go.work 影响。
# ---- build ----
FROM golang:1.22 AS build
ARG GOPROXY=https://proxy.golang.org|https://goproxy.cn|direct
ENV GOPROXY=${GOPROXY}
WORKDIR /src
COPY go.mod go.sum* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /out/server ./cmd/server

# ---- run ----
FROM gcr.io/distroless/static-debian12
COPY --from=build /out/server /server
EXPOSE 8081
ENV SERVER_ADDR=:8081
ENTRYPOINT ["/server"]
