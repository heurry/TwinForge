ARG RUNTIME_IMAGE=gcr.io/distroless/static-debian12:latest

FROM golang:1.22 AS build
ARG GOPROXY=https://proxy.golang.org|https://goproxy.cn|direct
ENV GOPROXY=${GOPROXY}
WORKDIR /src
COPY go.mod go.sum ./
RUN /usr/local/go/bin/go mod download
COPY . .
RUN CGO_ENABLED=0 /usr/local/go/bin/go build -o /out/agent-api ./cmd/agent-api && \
    CGO_ENABLED=0 /usr/local/go/bin/go build -o /out/agent-worker ./cmd/agent-worker && \
	CGO_ENABLED=0 /usr/local/go/bin/go build -o /out/agent-sandbox ./cmd/agent-sandbox && \
	CGO_ENABLED=0 /usr/local/go/bin/go build -o /out/agent-dependency-installer ./cmd/agent-dependency-installer && \
	CGO_ENABLED=0 /usr/local/go/bin/go build -o /out/agent-project-broker ./cmd/agent-project-broker && \
    CGO_ENABLED=0 /usr/local/go/bin/go build -o /out/agent-migrate ./cmd/agent-migrate

FROM ${RUNTIME_IMAGE}
WORKDIR /app
COPY --from=build /out/agent-api /out/agent-worker /out/agent-sandbox /out/agent-dependency-installer /out/agent-project-broker /out/agent-migrate /app/
COPY migrations /app/migrations
USER 65532:65532
ENTRYPOINT ["/app/agent-api"]
