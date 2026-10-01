package main

import (
	"crypto/subtle"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/provider/sandbox"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/provider/workspace"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "--healthcheck" {
		client := http.Client{Timeout: 2 * time.Second}
		response, err := client.Get("http://127.0.0.1:8190/health/ready")
		if err != nil || response.StatusCode != 200 {
			os.Exit(1)
		}
		_ = response.Body.Close()
		return
	}
	root := strings.TrimSpace(os.Getenv("AGENT_SANDBOX_WORKSPACE_ROOT"))
	token := strings.TrimSpace(os.Getenv("AGENT_SANDBOX_TOKEN"))
	if root == "" || token == "" {
		slog.Error("AGENT_SANDBOX_WORKSPACE_ROOT and AGENT_SANDBOX_TOKEN are required")
		os.Exit(2)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health/ready", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"status":"ready","isolation":"container"}`)
	})
	mux.HandleFunc("POST /v1/execute", func(w http.ResponseWriter, r *http.Request) {
		provided := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		if len(provided) != len(token) || subtle.ConstantTimeCompare([]byte(provided), []byte(token)) != 1 {
			write(w, 401, sandbox.ExecuteResponse{Error: "unauthorized"})
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, 2<<20)
		var input sandbox.ExecuteRequest
		decoder := json.NewDecoder(r.Body)
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&input); err != nil {
			write(w, 400, sandbox.ExecuteResponse{Error: "invalid request: " + err.Error()})
			return
		}
		call := input.Call.ToolCall()
		workspaceID := call.WorkspaceID
		if strings.TrimSpace(workspaceID) == "" {
			workspaceID = call.RunID
		}
		runRoot, rootErr := workspace.EnsureRunRoot(root, workspaceID)
		if rootErr != nil {
			write(w, 422, sandbox.ExecuteResponse{Error: rootErr.Error()})
			return
		}
		var handler tool.Handler
		var err error
		if input.Preview {
			handler, err = workspace.NewPreviewHandler(input.Spec, workspace.Config{Root: runRoot, AllowWrite: true})
		} else {
			handler, err = workspace.NewHandler(input.Spec, workspace.Config{Root: runRoot, AllowWrite: true})
		}
		if err != nil {
			write(w, 422, sandbox.ExecuteResponse{Error: err.Error()})
			return
		}
		result, err := handler(r.Context(), call)
		if err != nil {
			write(w, 422, sandbox.ExecuteResponse{Error: err.Error()})
			return
		}
		if result.Meta == nil {
			result.Meta = make(map[string]string)
		}
		result.Meta["workspace_scope"] = "run"
		result.Meta["workspace_id"] = workspaceID
		out := sandbox.ExecuteResponse{Result: result}
		for _, item := range result.Artifacts {
			out.Artifacts = append(out.Artifacts, sandbox.WireArtifact{Kind: item.Kind, Name: item.Name, MediaType: item.MediaType, Content: item.Content, Metadata: item.Metadata})
		}
		out.Result.Artifacts = nil
		write(w, 200, out)
	})
	mux.HandleFunc("POST /v1/promote", func(w http.ResponseWriter, r *http.Request) {
		provided := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		if len(provided) != len(token) || subtle.ConstantTimeCompare([]byte(provided), []byte(token)) != 1 {
			write(w, 401, sandbox.PromoteResponse{Error: "unauthorized"})
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
		var input sandbox.PromoteRequest
		decoder := json.NewDecoder(r.Body)
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&input); err != nil {
			write(w, 400, sandbox.PromoteResponse{Error: "invalid request: " + err.Error()})
			return
		}
		previousHash, resultHash, created, err := workspace.PromoteRunFile(root, input.RunID, input.SourcePath, input.SourceHash, input.TargetPath, input.ExpectedTargetSHA256)
		result := sandbox.PromoteResponse{TargetPath: input.TargetPath, PreviousTargetSHA256: previousHash, ResultTargetSHA256: resultHash, Created: created}
		if err != nil {
			result.Error = err.Error()
			write(w, 409, result)
			return
		}
		write(w, 200, result)
	})
	server := &http.Server{Addr: ":8190", Handler: mux, ReadHeaderTimeout: 3 * time.Second, ReadTimeout: 30 * time.Second, WriteTimeout: 30 * time.Second, IdleTimeout: 30 * time.Second, MaxHeaderBytes: 16 << 10}
	slog.Info("agent sandbox listening", "addr", server.Addr)
	if err := server.ListenAndServe(); err != nil {
		slog.Error("agent sandbox stopped", "err", err)
		os.Exit(1)
	}
}
func write(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
