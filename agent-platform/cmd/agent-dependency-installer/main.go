package main

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/provider/dependency"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/provider/workspace"
)

type installer struct {
	root    string
	token   string
	sources map[string]string
	locks   sync.Map
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "--healthcheck" {
		response, err := (&http.Client{Timeout: 2 * time.Second}).Get("http://127.0.0.1:8191/health/ready")
		if err != nil || response.StatusCode != http.StatusOK {
			os.Exit(1)
		}
		_ = response.Body.Close()
		return
	}
	root := strings.TrimSpace(os.Getenv("AGENT_DEPENDENCY_WORKSPACE_ROOT"))
	token := strings.TrimSpace(os.Getenv("AGENT_DEPENDENCY_INSTALLER_TOKEN"))
	sources, err := parseSources(os.Getenv("AGENT_DEPENDENCY_SOURCES"))
	if root == "" || token == "" || err != nil {
		slog.Error("invalid dependency installer configuration", "root_configured", root != "", "token_configured", token != "", "err", err)
		os.Exit(2)
	}
	service := &installer{root: root, token: token, sources: sources}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health/ready", service.ready)
	mux.HandleFunc("POST /v1/dependencies/install", service.install)
	server := &http.Server{Addr: ":8191", Handler: mux, ReadHeaderTimeout: 3 * time.Second, ReadTimeout: 130 * time.Second, WriteTimeout: 130 * time.Second, IdleTimeout: 30 * time.Second, MaxHeaderBytes: 16 << 10}
	slog.Info("agent dependency installer listening", "addr", server.Addr, "sources", sourceNames(sources))
	if err := server.ListenAndServe(); err != nil {
		slog.Error("dependency installer stopped", "err", err)
		os.Exit(1)
	}
}

func parseSources(raw string) (map[string]string, error) {
	var sources map[string]string
	if err := json.Unmarshal([]byte(strings.TrimSpace(raw)), &sources); err != nil {
		return nil, errors.New("AGENT_DEPENDENCY_SOURCES must be a JSON object")
	}
	if len(sources) == 0 {
		return nil, errors.New("AGENT_DEPENDENCY_SOURCES must not be empty")
	}
	normalized := make(map[string]string, len(sources))
	for name, value := range sources {
		name = strings.ToLower(strings.TrimSpace(name))
		parsed, err := url.Parse(strings.TrimSpace(value))
		if name == "" || err != nil || parsed.Scheme != "https" || parsed.Host == "" || parsed.User != nil {
			return nil, errors.New("every dependency source must have a name and an absolute https URL")
		}
		normalized[name] = parsed.String()
	}
	return normalized, nil
}

func sourceNames(sources map[string]string) []string {
	result := make([]string, 0, len(sources))
	for name := range sources {
		result = append(result, name)
	}
	return result
}

func (i *installer) ready(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ready", "isolation": "run", "sources": sourceNames(i.sources), "shell": false, "binary_only": true})
}

func (i *installer) install(w http.ResponseWriter, r *http.Request) {
	provided := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	if len(provided) != len(i.token) || subtle.ConstantTimeCompare([]byte(provided), []byte(i.token)) != 1 {
		writeJSON(w, http.StatusUnauthorized, dependency.Response{Error: "unauthorized"})
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 64<<10)
	var request dependency.Request
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&request); err != nil {
		writeJSON(w, http.StatusBadRequest, dependency.Response{Error: "invalid request: " + err.Error()})
		return
	}
	if err := request.NormalizeAndValidate(); err != nil {
		writeJSON(w, http.StatusUnprocessableEntity, dependency.Response{Error: err.Error()})
		return
	}
	indexURL, ok := i.sources[request.Source]
	if !ok {
		writeJSON(w, http.StatusUnprocessableEntity, dependency.Response{Error: "dependency source is not configured"})
		return
	}
	runRoot, err := workspace.EnsureRunRoot(i.root, request.RunID)
	if err != nil {
		writeJSON(w, http.StatusUnprocessableEntity, dependency.Response{Error: err.Error()})
		return
	}
	lockValue, _ := i.locks.LoadOrStore(request.RunID, &sync.Mutex{})
	lock := lockValue.(*sync.Mutex)
	lock.Lock()
	defer lock.Unlock()
	target := filepath.Join(runRoot, ".deps", "python")
	if err := os.MkdirAll(target, 0o750); err != nil {
		writeJSON(w, http.StatusInternalServerError, dependency.Response{Error: "create dependency target: " + err.Error()})
		return
	}
	// --no-deps is intentional: the approval enumerates every installed object.
	// A resolver-selected transitive package would otherwise bypass review.
	args := []string{"-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--no-color", "--only-binary=:all:", "--no-deps", "--target", target, "--index-url", indexURL}
	for _, item := range request.Packages {
		args = append(args, item.Name+"=="+item.Version)
	}
	ctx, cancel := context.WithTimeout(r.Context(), 120*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, "python3", args...)
	command.Env = []string{"PATH=/usr/local/bin:/usr/bin:/bin", "HOME=/tmp", "PIP_NO_INPUT=1", "PIP_DISABLE_PIP_VERSION_CHECK=1", "PYTHONDONTWRITEBYTECODE=1"}
	stdout, stderr := &limitedWriter{limit: 1 << 20}, &limitedWriter{limit: 1 << 20}
	command.Stdout, command.Stderr = stdout, stderr
	started := time.Now()
	err = command.Run()
	result := dependency.Result{Ecosystem: request.Ecosystem, Packages: request.Packages, Source: request.Source, Scope: request.Scope, Target: ".deps/python", Stdout: stdout.String(), Stderr: stderr.String(), DurationMS: time.Since(started).Milliseconds()}
	if ctx.Err() == context.DeadlineExceeded {
		writeJSON(w, http.StatusGatewayTimeout, dependency.Response{Result: result, Error: "dependency installation timed out"})
		return
	}
	if err != nil {
		writeJSON(w, http.StatusUnprocessableEntity, dependency.Response{Result: result, Error: "dependency installation failed: " + err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, dependency.Response{Result: result})
}

type limitedWriter struct {
	data  []byte
	limit int
}

func (w *limitedWriter) Write(value []byte) (int, error) {
	n := len(value)
	if remaining := w.limit - len(w.data); remaining > 0 {
		if len(value) > remaining {
			value = value[:remaining]
		}
		w.data = append(w.data, value...)
	}
	return n, nil
}
func (w *limitedWriter) String() string { return string(w.data) }

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

var _ io.Writer = (*limitedWriter)(nil)
