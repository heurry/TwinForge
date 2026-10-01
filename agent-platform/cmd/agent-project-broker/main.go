package main

import (
	"crypto/subtle"
	"encoding/json"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/provider/projectaccess"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/provider/sandbox"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/provider/workspace"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "--healthcheck" {
		client := http.Client{Timeout: 2 * time.Second}
		response, err := client.Get("http://127.0.0.1:8192/health/ready")
		if err != nil || response.StatusCode != http.StatusOK {
			os.Exit(1)
		}
		_ = response.Body.Close()
		return
	}
	root := strings.TrimSpace(os.Getenv("AGENT_PROJECT_ROOT"))
	token := strings.TrimSpace(os.Getenv("AGENT_PROJECT_ACCESS_TOKEN"))
	if root == "" || token == "" {
		slog.Error("AGENT_PROJECT_ROOT and AGENT_PROJECT_ACCESS_TOKEN are required")
		os.Exit(2)
	}

	authorized := func(request *http.Request) bool {
		provided := strings.TrimPrefix(request.Header.Get("Authorization"), "Bearer ")
		return len(provided) == len(token) && subtle.ConstantTimeCompare([]byte(provided), []byte(token)) == 1
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health/ready", func(writer http.ResponseWriter, _ *http.Request) {
		writeJSON(writer, http.StatusOK, map[string]any{"status": "ready", "capabilities": []string{"approved_read", "artifact_promotion"}})
	})
	mux.HandleFunc("POST /v1/read", func(writer http.ResponseWriter, request *http.Request) {
		if !authorized(request) {
			writeJSON(writer, http.StatusUnauthorized, projectaccess.ReadResponse{Error: "unauthorized"})
			return
		}
		request.Body = http.MaxBytesReader(writer, request.Body, 64<<10)
		var input projectaccess.ReadRequest
		decoder := json.NewDecoder(request.Body)
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&input); err != nil {
			writeJSON(writer, http.StatusBadRequest, projectaccess.ReadResponse{Error: "invalid request: " + err.Error()})
			return
		}
		if strings.TrimSpace(input.RunID) == "" {
			writeJSON(writer, http.StatusBadRequest, projectaccess.ReadResponse{Error: "run_id is required"})
			return
		}
		if err := input.ValidateModelInput(); err != nil {
			writeJSON(writer, http.StatusUnprocessableEntity, projectaccess.ReadResponse{Error: err.Error()})
			return
		}
		snapshot, err := workspace.ReadProjectFileSnapshot(root, input.Path, input.StartLine, input.LineCount, 1<<20)
		if err != nil {
			writeJSON(writer, http.StatusUnprocessableEntity, projectaccess.ReadResponse{Error: err.Error()})
			return
		}
		writeJSON(writer, http.StatusOK, projectaccess.ReadResponse{
			Path: snapshot.Path, Bytes: snapshot.Bytes, SHA256: snapshot.SHA256,
			LineCount: snapshot.LineCount, StartLine: snapshot.StartLine,
			Content: snapshot.Content, MediaType: snapshot.MediaType,
			SnapshotContent: snapshot.SnapshotContent,
		})
	})
	mux.HandleFunc("POST /v1/promote", func(writer http.ResponseWriter, request *http.Request) {
		if !authorized(request) {
			writeJSON(writer, http.StatusUnauthorized, sandbox.PromoteResponse{Error: "unauthorized"})
			return
		}
		request.Body = http.MaxBytesReader(writer, request.Body, 1<<20)
		var input sandbox.PromoteRequest
		decoder := json.NewDecoder(request.Body)
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&input); err != nil {
			writeJSON(writer, http.StatusBadRequest, sandbox.PromoteResponse{Error: "invalid request: " + err.Error()})
			return
		}
		previousHash, resultHash, created, err := workspace.PromoteRunFile(root, input.RunID, input.SourcePath, input.SourceHash, input.TargetPath, input.ExpectedTargetSHA256)
		result := sandbox.PromoteResponse{TargetPath: input.TargetPath, PreviousTargetSHA256: previousHash, ResultTargetSHA256: resultHash, Created: created}
		if err != nil {
			result.Error = err.Error()
			writeJSON(writer, http.StatusConflict, result)
			return
		}
		writeJSON(writer, http.StatusOK, result)
	})

	server := &http.Server{Addr: ":8192", Handler: mux, ReadHeaderTimeout: 3 * time.Second, ReadTimeout: 30 * time.Second, WriteTimeout: 30 * time.Second, IdleTimeout: 30 * time.Second, MaxHeaderBytes: 16 << 10}
	slog.Info("agent project broker listening", "addr", server.Addr)
	if err := server.ListenAndServe(); err != nil {
		slog.Error("agent project broker stopped", "err", err)
		os.Exit(1)
	}
}

func writeJSON(writer http.ResponseWriter, status int, value any) {
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(status)
	_ = json.NewEncoder(writer).Encode(value)
}
