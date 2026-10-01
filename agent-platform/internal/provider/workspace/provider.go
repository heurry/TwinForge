// Package workspace provides deployment-scoped filesystem tools.
package workspace

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/resource"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

const (
	defaultMaxBytes = int64(256 << 10)
	maxEntries      = 500
)

var runWorkspaceID = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$`)

// Config is operator-owned policy. Root is deliberately absent from ToolSpec.
type Config struct {
	Root       string
	AllowWrite bool
}

// ProjectFileSnapshot is produced only by the trusted project-access broker.
// Content is the bounded model-visible range; SnapshotContent is the complete
// immutable input captured for Artifact storage and is never sent to the model.
type ProjectFileSnapshot struct {
	Path            string `json:"path"`
	Bytes           int64  `json:"bytes"`
	SHA256          string `json:"sha256"`
	LineCount       int    `json:"line_count"`
	StartLine       int    `json:"start_line,omitempty"`
	Content         string `json:"content"`
	MediaType       string `json:"media_type"`
	SnapshotContent []byte `json:"snapshot_content"`
}

// ReadProjectFileSnapshot reads one operator-bound project file without ever
// accepting an absolute path, traversal, internal Run storage or sensitive
// credential paths. The caller must enforce human approval before invoking it.
func ReadProjectFileSnapshot(root, requested string, startLine, lineCount int, maxBytes int64) (ProjectFileSnapshot, error) {
	if maxBytes <= 0 || maxBytes > 1<<20 {
		maxBytes = 1 << 20
	}
	clean := filepath.ToSlash(filepath.Clean(strings.TrimSpace(requested)))
	if clean == ".agent-workspaces" || strings.HasPrefix(clean, ".agent-workspaces/") {
		return ProjectFileSnapshot{}, errors.New("project access cannot read internal Run workspaces")
	}
	path, relative, err := confinedPath(root, requested, false)
	if err != nil {
		return ProjectFileSnapshot{}, err
	}
	data, err := readLimited(path, maxBytes)
	if err != nil {
		return ProjectFileSnapshot{}, err
	}
	if !utf8.Valid(data) {
		return ProjectFileSnapshot{}, errors.New("binary or non-UTF-8 project files cannot be read as text")
	}
	lines := strings.Split(string(data), "\n")
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	if startLine < 1 {
		startLine = 1
	}
	start := min(startLine-1, len(lines))
	end := len(lines)
	if lineCount > 0 {
		end = min(start+lineCount, len(lines))
	}
	selected := strings.Join(lines[start:end], "\n")
	if len(selected) > 256<<10 {
		return ProjectFileSnapshot{}, errors.New("selected project file range exceeds 256 KiB; request fewer lines")
	}
	mediaType := "text/plain"
	switch strings.ToLower(filepath.Ext(relative)) {
	case ".md", ".markdown":
		mediaType = "text/markdown"
	case ".json":
		mediaType = "application/json"
	case ".py":
		mediaType = "text/x-python"
	case ".go":
		mediaType = "text/x-go"
	case ".c", ".h", ".cc", ".cpp", ".hpp":
		mediaType = "text/x-c"
	}
	return ProjectFileSnapshot{
		Path: relative, Bytes: int64(len(data)), SHA256: digest(data), LineCount: len(lines),
		StartLine: startLine, Content: selected, MediaType: mediaType,
		SnapshotContent: append([]byte(nil), data...),
	}, nil
}

// EnsureRunRoot returns a deterministic, isolated workspace for one Run.
// The model never chooses this path: runID is supplied by the fenced runtime,
// while the operator owns the base directory. This prevents one Run from
// observing or overwriting another Run's partial artifacts.
func EnsureRunRoot(baseRoot, runID string) (string, error) {
	baseRoot = strings.TrimSpace(baseRoot)
	runID = strings.TrimSpace(runID)
	if baseRoot == "" {
		return "", errors.New("workspace base root is required")
	}
	if !runWorkspaceID.MatchString(runID) {
		return "", errors.New("run id is invalid for workspace isolation")
	}
	resolvedBase, err := filepath.EvalSymlinks(baseRoot)
	if err != nil {
		return "", fmt.Errorf("resolve workspace base root: %w", err)
	}
	resolvedBase, err = filepath.Abs(resolvedBase)
	if err != nil {
		return "", fmt.Errorf("resolve workspace base root: %w", err)
	}
	container := filepath.Join(resolvedBase, ".agent-workspaces")
	if err := os.MkdirAll(container, 0o750); err != nil {
		return "", fmt.Errorf("create Run workspace container: %w", err)
	}
	runRoot := filepath.Join(container, runID)
	if err := os.MkdirAll(runRoot, 0o750); err != nil {
		return "", fmt.Errorf("create Run workspace: %w", err)
	}
	resolvedRun, err := filepath.EvalSymlinks(runRoot)
	if err != nil {
		return "", fmt.Errorf("resolve Run workspace: %w", err)
	}
	if err := ensureWithin(resolvedBase, resolvedRun); err != nil {
		return "", errors.New("resolved Run workspace escapes operator root")
	}
	return resolvedRun, nil
}

// PromoteRunFile copies one immutable Run-workspace file into the operator's
// project workspace with optimistic concurrency. The caller supplies the
// source Artifact hash; an existing target additionally requires its exact
// current hash, preventing silent overwrite of user changes.
func PromoteRunFile(baseRoot, runID, sourcePath, sourceHash, targetPath, expectedTargetHash string) (string, string, bool, error) {
	runRoot, err := EnsureRunRoot(baseRoot, runID)
	if err != nil {
		return "", "", false, err
	}
	source, _, err := confinedPath(runRoot, sourcePath, false)
	if err != nil {
		return "", "", false, fmt.Errorf("resolve promotion source: %w", err)
	}
	content, err := readLimited(source, 10<<20)
	if err != nil {
		return "", "", false, fmt.Errorf("read promotion source: %w", err)
	}
	actualSourceHash := digest(content)
	if !strings.EqualFold(strings.TrimSpace(sourceHash), actualSourceHash) {
		return "", "", false, errors.New("Run workspace file no longer matches the selected Artifact; select the latest Artifact")
	}
	targetPath = strings.TrimSpace(targetPath)
	cleanTarget := filepath.ToSlash(filepath.Clean(targetPath))
	if cleanTarget == ".agent-workspaces" || strings.HasPrefix(cleanTarget, ".agent-workspaces/") {
		return "", "", false, errors.New("promotion target cannot be inside the internal Run workspace container")
	}
	target, _, err := confinedPath(baseRoot, targetPath, true)
	if err != nil {
		return "", "", false, fmt.Errorf("resolve promotion target: %w", err)
	}
	prior, readErr := readLimited(target, 10<<20)
	created := errors.Is(readErr, os.ErrNotExist)
	if readErr != nil && !created {
		return "", "", false, fmt.Errorf("read promotion target: %w", readErr)
	}
	previousHash := digest(prior)
	if !created {
		if strings.TrimSpace(expectedTargetHash) == "" {
			return previousHash, "", false, fmt.Errorf("target exists; retry with expected_target_sha256=%s after reviewing the Diff", previousHash)
		}
		if !strings.EqualFold(expectedTargetHash, previousHash) {
			return previousHash, "", false, fmt.Errorf("target changed; current_target_sha256=%s", previousHash)
		}
	}
	if err := atomicWrite(target, content); err != nil {
		return previousHash, "", false, fmt.Errorf("promote workspace file: %w", err)
	}
	return previousHash, actualSourceHash, created, nil
}

// NewHandler returns a path-confined in-process filesystem operation.
func NewHandler(spec resource.ToolSpec, config Config) (tool.Handler, error) {
	if err := spec.Validate(); err != nil {
		return nil, err
	}
	if spec.ProviderType != "workspace" || spec.Workspace == nil {
		return nil, errors.New("workspace provider configuration is required")
	}
	root := strings.TrimSpace(config.Root)
	if root == "" {
		return nil, errors.New("workspace tools are disabled: AGENT_WORKSPACE_ROOT is empty")
	}
	resolvedRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		return nil, fmt.Errorf("resolve workspace root: %w", err)
	}
	resolvedRoot, err = filepath.Abs(resolvedRoot)
	if err != nil {
		return nil, fmt.Errorf("resolve workspace root: %w", err)
	}
	operation := spec.Workspace.Operation
	if (operation == "write_file" || operation == "append_file" || operation == "edit_file" || operation == "promote_file" || operation == "create_directory" || operation == "run_command") && !config.AllowWrite {
		return nil, errors.New("workspace write tools are disabled by deployment policy")
	}
	limit := spec.Workspace.MaxBytes
	if limit == 0 {
		limit = defaultMaxBytes
	}
	return func(ctx context.Context, call tool.Call) (tool.Result, error) {
		if err := ctx.Err(); err != nil {
			return tool.Result{}, err
		}
		var value any
		var err error
		switch operation {
		case "read_file":
			value, err = readFile(resolvedRoot, limit, call.Arguments)
		case "list_files":
			value, err = listFiles(ctx, resolvedRoot, call.Arguments)
		case "search_files":
			value, err = searchFiles(ctx, resolvedRoot, limit, call.Arguments)
		case "write_file":
			var artifact tool.Artifact
			value, artifact, err = writeFile(resolvedRoot, limit, call.Arguments)
			if err == nil {
				snapshot, snapshotErr := workspaceFileArtifact(resolvedRoot, limit, value)
				if snapshotErr != nil {
					return tool.Result{}, snapshotErr
				}
				content, marshalErr := json.Marshal(value)
				if marshalErr != nil {
					return tool.Result{}, fmt.Errorf("encode workspace result: %w", marshalErr)
				}
				return tool.Result{Content: content, Meta: map[string]string{"provider": "workspace", "operation": operation}, Artifacts: []tool.Artifact{artifact, snapshot}}, nil
			}
		case "append_file":
			var artifact tool.Artifact
			value, artifact, err = appendFile(resolvedRoot, limit, call.Arguments)
			if err == nil {
				snapshot, snapshotErr := workspaceFileArtifact(resolvedRoot, limit, value)
				if snapshotErr != nil {
					return tool.Result{}, snapshotErr
				}
				content, marshalErr := json.Marshal(value)
				if marshalErr != nil {
					return tool.Result{}, fmt.Errorf("encode workspace result: %w", marshalErr)
				}
				return tool.Result{Content: content, Meta: map[string]string{"provider": "workspace", "operation": operation}, Artifacts: []tool.Artifact{artifact, snapshot}}, nil
			}
		case "edit_file":
			var artifact tool.Artifact
			value, artifact, err = editFile(resolvedRoot, limit, call.Arguments)
			if err == nil {
				snapshot, snapshotErr := workspaceFileArtifact(resolvedRoot, limit, value)
				if snapshotErr != nil {
					return tool.Result{}, snapshotErr
				}
				content, marshalErr := json.Marshal(value)
				if marshalErr != nil {
					return tool.Result{}, fmt.Errorf("encode workspace result: %w", marshalErr)
				}
				return tool.Result{Content: content, Meta: map[string]string{"provider": "workspace", "operation": operation}, Artifacts: []tool.Artifact{artifact, snapshot}}, nil
			}
		case "promote_file":
			var artifact tool.Artifact
			value, artifact, err = promoteFile(resolvedRoot, limit, call.Arguments)
			if err == nil {
				snapshot, snapshotErr := workspaceFileArtifact(resolvedRoot, limit, value)
				if snapshotErr != nil {
					return tool.Result{}, snapshotErr
				}
				content, marshalErr := json.Marshal(value)
				if marshalErr != nil {
					return tool.Result{}, fmt.Errorf("encode workspace result: %w", marshalErr)
				}
				return tool.Result{Content: content, Meta: map[string]string{"provider": "workspace", "operation": operation}, Artifacts: []tool.Artifact{artifact, snapshot}}, nil
			}
		case "create_directory":
			value, err = createDirectory(resolvedRoot, call.Arguments)
		case "run_command":
			result, commandErr := runCommand(ctx, resolvedRoot, limit, call.Arguments)
			if commandErr != nil {
				return tool.Result{}, commandErr
			}
			content, marshalErr := json.Marshal(result)
			if marshalErr != nil {
				return tool.Result{}, fmt.Errorf("encode command result: %w", marshalErr)
			}
			if result.ExitCode != 0 || result.TimedOut {
				message := fmt.Sprintf("command exited with code %d", result.ExitCode)
				if result.TimedOut {
					message = "command timed out"
				}
				correction := "The process ran and failed. Inspect diagnostic/stderr_tail, repair the referenced workspace code or change the invocation, then retry once; do not repeat the unchanged command against an unchanged workspace."
				if result.TimedOut {
					correction = "The process ran but timed out. Inspect available output, reduce or repair the workload, or change timeout_seconds within the offered limit; do not repeat the unchanged command against an unchanged workspace."
				}
				return tool.Result{Content: content, IsError: true, Error: message, Meta: map[string]string{"provider": "workspace", "operation": operation, "failure_kind": result.FailureKind, "retryable": "true", "correction": correction}}, nil
			}
			return tool.Result{Content: content, Meta: map[string]string{"provider": "workspace", "operation": operation}}, nil
		default:
			err = fmt.Errorf("unsupported workspace operation %q", operation)
		}
		if err != nil {
			return tool.Result{}, err
		}
		content, err := json.Marshal(value)
		if err != nil {
			return tool.Result{}, fmt.Errorf("encode workspace result: %w", err)
		}
		return tool.Result{Content: content, Meta: map[string]string{"provider": "workspace", "operation": operation}}, nil
	}, nil
}

type commandResult struct {
	Command string   `json:"command"`
	Args    []string `json:"args"`
	Profile string   `json:"execution_profile,omitempty"`
	// SandboxPolicy makes the distinction between an allowed process that
	// returned a non-zero exit code and a command rejected before execution
	// explicit to the model. The latter is represented by a structured
	// SANDBOX_COMMAND_REJECTED tool error, never by this result payload.
	SandboxPolicy string `json:"sandbox_policy,omitempty"`
	FailureKind   string `json:"failure_kind,omitempty"`
	// Diagnostic is a deterministic, language-agnostic extraction of the last
	// non-empty stderr/stdout line. It normally contains the root exception or
	// final compiler/test failure without asking an LLM to summarize logs.
	Diagnostic string `json:"diagnostic,omitempty"`
	StdoutTail string `json:"stdout_tail,omitempty"`
	StderrTail string `json:"stderr_tail,omitempty"`
	Guidance   string `json:"guidance,omitempty"`
	WorkingDir string `json:"working_dir"`
	ExitCode   int    `json:"exit_code"`
	Stdout     string `json:"stdout,omitempty"`
	Stderr     string `json:"stderr,omitempty"`
	TimedOut   bool   `json:"timed_out"`
	DurationMS int64  `json:"duration_ms"`
}

func runCommand(ctx context.Context, root string, limit int64, raw json.RawMessage) (commandResult, error) {
	var input struct {
		Command        string   `json:"command"`
		Args           []string `json:"args"`
		WorkingDir     string   `json:"working_directory"`
		TimeoutSeconds int      `json:"timeout_seconds"`
	}
	if err := json.Unmarshal(raw, &input); err != nil {
		return commandResult{}, fmt.Errorf("decode run_command arguments: %w", err)
	}
	if err := validateCommand(input.Command, input.Args); err != nil {
		return commandResult{}, err
	}
	directory, relative, err := confinedPath(root, defaultPath(input.WorkingDir), false)
	if err != nil {
		return commandResult{}, fmt.Errorf("resolve command working directory: %w", err)
	}
	timeout := input.TimeoutSeconds
	if timeout <= 0 {
		timeout = 10
	}
	if timeout > 30 {
		timeout = 30
	}
	profile := "workspace_script"
	sandboxPolicy := "workspace_script_allowed"
	if isInlineProbe(input.Args) {
		profile = "inline_probe"
		sandboxPolicy = "inline_probe_restricted"
		if timeout > 5 {
			timeout = 5
		}
	}
	commandCtx, cancel := context.WithTimeout(ctx, time.Duration(timeout)*time.Second)
	defer cancel()
	command := exec.CommandContext(commandCtx, input.Command, input.Args...)
	command.Dir = directory
	// Run-scoped approved dependencies are installed by the separate installer
	// into this reserved directory. The model cannot read, write, or select this
	// path, but subsequent Python processes in the same Run can import it.
	dependencyPath := filepath.Join(root, ".deps", "python")
	command.Env = []string{"PATH=/usr/local/bin:/usr/bin:/bin", "HOME=/tmp", "PYTHONPATH=" + dependencyPath, "PYTHONDONTWRITEBYTECODE=1", "PYTHONUNBUFFERED=1", "SDL_VIDEODRIVER=dummy", "SDL_AUDIODRIVER=dummy"}
	stdout := newLimitedBuffer(limit)
	stderr := newLimitedBuffer(limit)
	command.Stdout = stdout
	command.Stderr = stderr
	started := time.Now()
	runErr := command.Run()
	result := commandResult{Command: input.Command, Args: input.Args, Profile: profile, SandboxPolicy: sandboxPolicy, WorkingDir: relative, ExitCode: 0, Stdout: stdout.String(), Stderr: stderr.String(), DurationMS: time.Since(started).Milliseconds()}
	result.StdoutTail = commandOutputTail(result.Stdout, 1200)
	result.StderrTail = commandOutputTail(result.Stderr, 1200)
	result.Diagnostic = commandDiagnostic(result.Stderr, result.Stdout, 500)
	if commandCtx.Err() == context.DeadlineExceeded {
		result.ExitCode = -1
		result.TimedOut = true
		result.FailureKind = "timeout"
		result.Guidance = "The command was allowed to run in the Sandbox but timed out; inspect the workspace script and reduce its scope."
		return result, nil
	}
	if runErr == nil {
		return result, nil
	}
	var exitErr *exec.ExitError
	if errors.As(runErr, &exitErr) {
		result.ExitCode = exitErr.ExitCode()
		result.FailureKind = "process_exit"
		result.Guidance = "The workspace script was allowed to run; inspect stderr and edit the script before retrying. This is not a Sandbox policy rejection."
		return result, nil
	}
	return commandResult{}, fmt.Errorf("start command: %w", runErr)
}

func commandOutputTail(value string, limit int) string {
	runes := []rune(strings.TrimSpace(value))
	if len(runes) <= limit {
		return string(runes)
	}
	return "…" + string(runes[len(runes)-limit:])
}

func commandDiagnostic(stderr, stdout string, limit int) string {
	value := stderr
	if strings.TrimSpace(value) == "" {
		value = stdout
	}
	lines := strings.Split(strings.TrimSpace(value), "\n")
	for index := len(lines) - 1; index >= 0; index-- {
		line := strings.TrimSpace(lines[index])
		if line == "" {
			continue
		}
		runes := []rune(line)
		if len(runes) > limit {
			return string(runes[:limit]) + "…"
		}
		return line
	}
	return ""
}

// ValidateCommandArguments performs the side-effect-free policy check used by
// the Worker before it asks a human to approve command execution.
func ValidateCommandArguments(raw json.RawMessage) error {
	var input struct {
		Command string   `json:"command"`
		Args    []string `json:"args"`
	}
	if err := json.Unmarshal(raw, &input); err != nil {
		return fmt.Errorf("decode run_command arguments: %w", err)
	}
	return validateCommand(input.Command, input.Args)
}

func validateCommand(command string, args []string) error {
	if command != "python3" && command != "python" {
		return fmt.Errorf("command %q is not allowed; available commands: python3", command)
	}
	if len(args) == 0 || len(args) > 32 {
		return errors.New("python command requires 1-32 arguments")
	}
	for _, argument := range args {
		if len(argument) > 1024 || strings.ContainsRune(argument, '\x00') {
			return errors.New("command argument is invalid or too long")
		}
		if argument == "-" {
			return errors.New("stdin Python execution is not allowed")
		}
	}
	if args[0] == "-c" {
		return validateInlineProbe(args)
	}
	for _, argument := range args[1:] {
		if argument == "-c" {
			return errors.New("python -c is allowed only as a top-level restricted inline probe")
		}
	}
	if args[0] == "-m" {
		if len(args) < 2 {
			return errors.New("python -m requires an allowed module")
		}
		switch args[1] {
		case "py_compile", "compileall", "unittest":
		default:
			return fmt.Errorf("python module %q is not allowed", args[1])
		}
	}
	return nil
}

func isInlineProbe(args []string) bool { return len(args) != 0 && args[0] == "-c" }

func validateInlineProbe(args []string) error {
	if len(args) != 2 {
		return errors.New("restricted python -c requires exactly one code argument")
	}
	code := strings.TrimSpace(args[1])
	if code == "" || len(code) > 512 || strings.ContainsAny(code, "\r\n") {
		return errors.New("restricted python -c probe must be one line and at most 512 characters")
	}
	lower := strings.ToLower(code)
	denied := []string{
		"__", "open(", "exec(", "eval(", "compile(", "subprocess", "socket", "urllib", "requests",
		"http.client", "pathlib", "shutil", "tempfile", "multiprocessing", "ctypes", "importlib", "import os", "from os", "os.", "import io", "from io", "glob",
	}
	for _, token := range denied {
		if strings.Contains(lower, token) {
			return fmt.Errorf("restricted python -c probe contains denied capability %q; write an auditable workspace script if this operation is required", token)
		}
	}
	return nil
}

type limitedBuffer struct {
	data  []byte
	limit int64
}

func newLimitedBuffer(limit int64) *limitedBuffer {
	if limit <= 0 || limit > 1<<20 {
		limit = defaultMaxBytes
	}
	return &limitedBuffer{limit: limit}
}

func (b *limitedBuffer) Write(value []byte) (int, error) {
	written := len(value)
	remaining := int(b.limit) - len(b.data)
	if remaining > 0 {
		if len(value) > remaining {
			value = value[:remaining]
		}
		b.data = append(b.data, value...)
	}
	return written, nil
}

func (b *limitedBuffer) String() string { return string(b.data) }

// NewPreviewHandler computes the exact write/edit result and diff without
// mutating the workspace. It is used before a human approval is requested.
func NewPreviewHandler(spec resource.ToolSpec, config Config) (tool.Handler, error) {
	if err := spec.Validate(); err != nil {
		return nil, err
	}
	if spec.ProviderType != "workspace" || spec.Workspace == nil {
		return nil, errors.New("workspace provider configuration is required")
	}
	root := strings.TrimSpace(config.Root)
	if root == "" {
		return nil, errors.New("workspace tools are disabled: AGENT_WORKSPACE_ROOT is empty")
	}
	resolvedRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		return nil, fmt.Errorf("resolve workspace root: %w", err)
	}
	resolvedRoot, err = filepath.Abs(resolvedRoot)
	if err != nil {
		return nil, err
	}
	limit := spec.Workspace.MaxBytes
	if limit == 0 {
		limit = defaultMaxBytes
	}
	operation := spec.Workspace.Operation
	return func(ctx context.Context, call tool.Call) (tool.Result, error) {
		if err := ctx.Err(); err != nil {
			return tool.Result{}, err
		}
		var value any
		var artifact tool.Artifact
		var previewErr error
		switch operation {
		case "write_file":
			value, artifact, _, _, previewErr = prepareWriteFile(resolvedRoot, limit, call.Arguments)
		case "append_file":
			value, artifact, _, _, previewErr = prepareAppendFile(resolvedRoot, limit, call.Arguments)
		case "edit_file":
			value, artifact, _, _, previewErr = prepareEditFile(resolvedRoot, limit, call.Arguments)
		case "promote_file":
			value, artifact, _, _, previewErr = preparePromoteFile(resolvedRoot, limit, call.Arguments)
		default:
			return tool.Result{}, errors.New("preview is only available for workspace write_file, append_file, edit_file and promote_file")
		}
		if previewErr != nil {
			return tool.Result{}, previewErr
		}
		if artifact.Metadata == nil {
			artifact.Metadata = make(map[string]string)
		}
		artifact.Metadata["phase"] = "proposed"
		content, err := json.Marshal(value)
		return tool.Result{Content: content, Meta: map[string]string{"provider": "workspace", "preview": "true"}, Artifacts: []tool.Artifact{artifact}}, err
	}, nil
}

type pathInput struct {
	Path string `json:"path"`
}

func createDirectory(root string, raw json.RawMessage) (any, error) {
	var input pathInput
	if err := json.Unmarshal(raw, &input); err != nil {
		return nil, fmt.Errorf("decode create_directory arguments: %w", err)
	}
	if strings.TrimSpace(input.Path) == "" || input.Path == "." {
		return nil, errors.New("directory path must name a workspace-relative directory")
	}
	path, relative, err := confinedPath(root, input.Path, true)
	if err != nil {
		return nil, err
	}
	info, statErr := os.Stat(path)
	if statErr == nil {
		if !info.IsDir() {
			return nil, fmt.Errorf("workspace path %q exists and is not a directory", relative)
		}
		return map[string]any{"path": relative, "created": false, "exists": true}, nil
	}
	if !errors.Is(statErr, os.ErrNotExist) {
		return nil, fmt.Errorf("inspect directory: %w", statErr)
	}
	if err := os.MkdirAll(path, 0o750); err != nil {
		return nil, fmt.Errorf("create workspace directory: %w", err)
	}
	return map[string]any{"path": relative, "created": true, "exists": true}, nil
}

type fileReadResult struct {
	Path       string `json:"path"`
	Bytes      int64  `json:"bytes"`
	FileSHA256 string `json:"file_sha256"`
	LineCount  int    `json:"line_count"`
	Tail       string `json:"tail,omitempty"`
	StartLine  int    `json:"start_line,omitempty"`
	Content    string `json:"content"`
}

func readFile(root string, limit int64, raw json.RawMessage) (any, error) {
	var input struct {
		Path      string `json:"path"`
		StartLine int    `json:"start_line"`
		LineCount int    `json:"line_count"`
	}
	if err := json.Unmarshal(raw, &input); err != nil {
		return nil, fmt.Errorf("decode read_file arguments: %w", err)
	}
	path, relative, err := confinedPath(root, input.Path, false)
	if err != nil {
		return nil, err
	}
	if input.StartLine > 0 || input.LineCount > 0 {
		return readFileRange(path, relative, limit, input.StartLine, input.LineCount)
	}
	data, err := readLimited(path, limit)
	if err != nil {
		return nil, err
	}
	if !utf8.Valid(data) {
		return nil, errors.New("binary or non-UTF-8 files cannot be read as text")
	}
	return fileReadResult{Path: relative, Bytes: int64(len(data)), FileSHA256: digest(data), LineCount: lineCount(data), Tail: textTail(string(data), 240), Content: string(data)}, nil
}

func readFileRange(path, relative string, limit int64, start, count int) (any, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	stat, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if start < 1 {
		start = 1
	}
	hasher := sha256.New()
	scanner := bufio.NewScanner(io.TeeReader(file, hasher))
	// A selected line may be larger than the normal Scanner token size, but it
	// must still fit the tool's output policy.
	maxLine := int(limit) + 1
	if maxLine < 64<<10 {
		maxLine = 64 << 10
	}
	scanner.Buffer(make([]byte, 64<<10), maxLine)
	selected := make([]string, 0, max(count, 1))
	selectedBytes := int64(0)
	line := 0
	for scanner.Scan() {
		line++
		value := scanner.Text()
		if !utf8.ValidString(value) {
			return nil, errors.New("binary or non-UTF-8 files cannot be read as text")
		}
		if line < start || (count > 0 && line >= start+count) {
			continue
		}
		addition := int64(len(value))
		if len(selected) > 0 {
			addition++
		}
		if selectedBytes+addition > limit {
			return nil, fmt.Errorf("selected file range exceeds %d byte limit", limit)
		}
		selected = append(selected, value)
		selectedBytes += addition
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan file range: %w", err)
	}
	content := strings.Join(selected, "\n")
	return fileReadResult{Path: relative, Bytes: stat.Size(), FileSHA256: hex.EncodeToString(hasher.Sum(nil)), StartLine: start, LineCount: len(selected), Tail: textTail(content, 240), Content: content}, nil
}

func textTail(value string, limit int) string {
	runes := []rune(value)
	if len(runes) <= limit {
		return value
	}
	return string(runes[len(runes)-limit:])
}

func listFiles(ctx context.Context, root string, raw json.RawMessage) (any, error) {
	var input struct {
		Path          string `json:"path"`
		Recursive     bool   `json:"recursive"`
		IncludeHidden bool   `json:"include_hidden"`
		MaxEntries    int    `json:"max_entries"`
	}
	if err := json.Unmarshal(raw, &input); err != nil {
		return nil, fmt.Errorf("decode list_files arguments: %w", err)
	}
	path, relative, err := confinedPath(root, defaultPath(input.Path), false)
	if err != nil {
		return nil, err
	}
	limit := input.MaxEntries
	if limit <= 0 || limit > maxEntries {
		limit = maxEntries
	}
	entries := make([]map[string]any, 0, min(limit, 64))
	if !input.Recursive {
		children, err := os.ReadDir(path)
		if err != nil {
			return nil, fmt.Errorf("list workspace path: %w", err)
		}
		type listedFile struct {
			value    map[string]any
			modified time.Time
		}
		files := make([]listedFile, 0, len(children))
		directories := make([]map[string]any, 0, len(children))
		for _, entry := range children {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			if !input.IncludeHidden && strings.HasPrefix(entry.Name(), ".") {
				continue
			}
			if ignoredDirectory(entry) {
				continue
			}
			relRoot, _ := filepath.Rel(root, filepath.Join(path, entry.Name()))
			if sensitivePath(relRoot) || entry.Type()&os.ModeSymlink != 0 {
				continue
			}
			info, err := entry.Info()
			if err != nil {
				return nil, err
			}
			value := map[string]any{"path": filepath.ToSlash(relRoot), "type": entryType(entry), "bytes": info.Size()}
			if entry.IsDir() {
				directories = append(directories, value)
			} else {
				files = append(files, listedFile{value: value, modified: info.ModTime()})
			}
		}
		sort.SliceStable(files, func(left, right int) bool {
			if files[left].modified.Equal(files[right].modified) {
				return files[left].value["path"].(string) < files[right].value["path"].(string)
			}
			return files[left].modified.After(files[right].modified)
		})
		all := make([]map[string]any, 0, len(files)+len(directories))
		for _, file := range files {
			all = append(all, file.value)
		}
		all = append(all, directories...)
		truncated := len(all) > limit
		if truncated {
			all = all[:limit]
		}
		return map[string]any{"path": relative, "entries": all, "truncated": truncated, "hidden_included": input.IncludeHidden}, nil
	}
	err = filepath.WalkDir(path, func(current string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if current == path {
			return nil
		}
		relToPath, _ := filepath.Rel(path, current)
		if !input.IncludeHidden && strings.HasPrefix(filepath.Base(relToPath), ".") {
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if ignoredDirectory(entry) {
			return filepath.SkipDir
		}
		relRoot, _ := filepath.Rel(root, current)
		if sensitivePath(relRoot) {
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		entries = append(entries, map[string]any{"path": filepath.ToSlash(relRoot), "type": entryType(entry), "bytes": info.Size()})
		if len(entries) >= limit {
			return fs.SkipAll
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("list workspace path: %w", err)
	}
	return map[string]any{"path": relative, "entries": entries, "truncated": len(entries) >= limit}, nil
}

func searchFiles(ctx context.Context, root string, fileLimit int64, raw json.RawMessage) (any, error) {
	var input struct {
		Path       string `json:"path"`
		Query      string `json:"query"`
		MaxResults int    `json:"max_results"`
	}
	if err := json.Unmarshal(raw, &input); err != nil {
		return nil, fmt.Errorf("decode search_files arguments: %w", err)
	}
	if strings.TrimSpace(input.Query) == "" {
		return nil, errors.New("search query is required")
	}
	path, relative, err := confinedPath(root, defaultPath(input.Path), false)
	if err != nil {
		return nil, err
	}
	limit := input.MaxResults
	if limit <= 0 || limit > 200 {
		limit = 100
	}
	results := make([]map[string]any, 0, min(limit, 32))
	err = filepath.WalkDir(path, func(current string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if entry.IsDir() {
			if current != path && ignoredDirectory(entry) {
				return filepath.SkipDir
			}
			return nil
		}
		relRoot, _ := filepath.Rel(root, current)
		if sensitivePath(relRoot) {
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return nil
		}
		data, err := readLimited(current, fileLimit)
		if err != nil || !utf8.Valid(data) {
			return nil
		}
		scanner := bufio.NewScanner(strings.NewReader(string(data)))
		line := 0
		for scanner.Scan() {
			line++
			if strings.Contains(scanner.Text(), input.Query) {
				results = append(results, map[string]any{"path": filepath.ToSlash(relRoot), "line": line, "text": scanner.Text()})
				if len(results) >= limit {
					return fs.SkipAll
				}
			}
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("search workspace: %w", err)
	}
	return map[string]any{"path": relative, "query": input.Query, "results": results, "truncated": len(results) >= limit}, nil
}

func writeFile(root string, limit int64, raw json.RawMessage) (any, tool.Artifact, error) {
	value, artifact, path, data, err := prepareWriteFile(root, limit, raw)
	if err != nil {
		return nil, tool.Artifact{}, err
	}
	if err := atomicWrite(path, data); err != nil {
		return nil, tool.Artifact{}, err
	}
	return value, artifact, nil
}

func prepareWriteFile(root string, limit int64, raw json.RawMessage) (any, tool.Artifact, string, []byte, error) {
	var input struct {
		Path           string `json:"path"`
		Content        string `json:"content"`
		FilePath       string `json:"file_path"`
		Text           string `json:"text"`
		Overwrite      bool   `json:"overwrite"`
		ExpectedSHA256 string `json:"expected_sha256"`
	}
	if err := json.Unmarshal(raw, &input); err != nil {
		return nil, tool.Artifact{}, "", nil, fmt.Errorf("decode write_file arguments: %w", err)
	}
	if input.Path == "" {
		input.Path = input.FilePath
	}
	if input.Content == "" {
		input.Content = input.Text
	}
	if int64(len(input.Content)) > limit {
		return nil, tool.Artifact{}, "", nil, fmt.Errorf("content exceeds %d byte limit", limit)
	}
	path, relative, err := confinedPath(root, input.Path, true)
	if err != nil {
		return nil, tool.Artifact{}, "", nil, err
	}
	prior, readErr := readLimited(path, limit)
	if readErr == nil {
		if !input.Overwrite {
			return nil, tool.Artifact{}, "", nil, errors.New("file exists and overwrite is false; read it, continue it with append_file, or set overwrite=true only when replacement is intentional")
		}
		if input.ExpectedSHA256 != "" && !strings.EqualFold(input.ExpectedSHA256, digest(prior)) {
			return nil, tool.Artifact{}, "", nil, errors.New("file changed since expected_sha256 was observed")
		}
	} else if !errors.Is(readErr, os.ErrNotExist) {
		return nil, tool.Artifact{}, "", nil, fmt.Errorf("read target file: %w", readErr)
	}
	data := []byte(input.Content)
	created := errors.Is(readErr, os.ErrNotExist)
	nextAction := fmt.Sprintf("Write completed atomically. If the intended file has remaining content, preserve every requirement and continue path %q with append_file using expected_sha256=%q; do not compact or discard implementation merely to fit one call. When the file is complete, verify syntax and behavior; use edit_file only for a precise repair.", relative, digest(data))
	return fileMutationResult(relative, len(data), len(prior), digest(prior), data, map[string]any{"created": created, "tail": textTail(input.Content, 240), "next_action": nextAction, "diff": map[string]int{"additions": lineCount(data), "deletions": lineCount(prior)}}), diffArtifact(relative, prior, data, created), path, data, nil
}

func appendFile(root string, limit int64, raw json.RawMessage) (any, tool.Artifact, error) {
	value, artifact, path, data, err := prepareAppendFile(root, limit, raw)
	if err != nil {
		return nil, tool.Artifact{}, err
	}
	if err := atomicWrite(path, data); err != nil {
		return nil, tool.Artifact{}, err
	}
	return value, artifact, nil
}

func prepareAppendFile(root string, limit int64, raw json.RawMessage) (any, tool.Artifact, string, []byte, error) {
	var input struct {
		Path           string `json:"path"`
		Content        string `json:"content"`
		FilePath       string `json:"file_path"`
		Text           string `json:"text"`
		ExpectedSHA256 string `json:"expected_sha256"`
	}
	if err := json.Unmarshal(raw, &input); err != nil {
		return nil, tool.Artifact{}, "", nil, fmt.Errorf("decode append_file arguments: %w", err)
	}
	if input.Path == "" {
		input.Path = input.FilePath
	}
	if input.Content == "" {
		input.Content = input.Text
	}
	if input.Content == "" {
		return nil, tool.Artifact{}, "", nil, errors.New("append content is required")
	}
	path, relative, err := confinedPath(root, input.Path, false)
	if err != nil {
		return nil, tool.Artifact{}, "", nil, err
	}
	prior, err := readLimited(path, limit)
	if err != nil {
		return nil, tool.Artifact{}, "", nil, err
	}
	if !utf8.Valid(prior) || !utf8.ValidString(input.Content) {
		return nil, tool.Artifact{}, "", nil, errors.New("only UTF-8 text files can be appended")
	}
	if input.ExpectedSHA256 != "" && !strings.EqualFold(input.ExpectedSHA256, digest(prior)) {
		return nil, tool.Artifact{}, "", nil, errors.New("file changed since expected_sha256 was observed")
	}
	next := append(append([]byte(nil), prior...), []byte(input.Content)...)
	if int64(len(next)) > limit {
		return nil, tool.Artifact{}, "", nil, fmt.Errorf("appended file exceeds %d byte limit", limit)
	}
	nextAction := fmt.Sprintf("Append completed atomically. If content remains, continue path %q with another coherent append_file chunk using expected_sha256=%q. Otherwise verify syntax and behavior; a successful append alone does not prove the module is complete.", relative, digest(next))
	return fileMutationResult(relative, len(next), len(prior), digest(prior), next, map[string]any{"appended_bytes": len(input.Content), "tail": textTail(input.Content, 240), "next_action": nextAction}), diffArtifact(relative, prior, next, false), path, next, nil
}

func editFile(root string, limit int64, raw json.RawMessage) (any, tool.Artifact, error) {
	value, artifact, path, next, err := prepareEditFile(root, limit, raw)
	if err != nil {
		return nil, tool.Artifact{}, err
	}
	if err := atomicWrite(path, next); err != nil {
		return nil, tool.Artifact{}, err
	}
	return value, artifact, nil
}

func prepareEditFile(root string, limit int64, raw json.RawMessage) (any, tool.Artifact, string, []byte, error) {
	var input struct {
		Path           string `json:"path"`
		OldText        string `json:"old_text"`
		NewText        string `json:"new_text"`
		ExpectedSHA256 string `json:"expected_sha256"`
	}
	if err := json.Unmarshal(raw, &input); err != nil {
		return nil, tool.Artifact{}, "", nil, fmt.Errorf("decode edit_file arguments: %w", err)
	}
	if input.OldText == "" {
		return nil, tool.Artifact{}, "", nil, errors.New("old_text is required")
	}
	path, relative, err := confinedPath(root, input.Path, false)
	if err != nil {
		return nil, tool.Artifact{}, "", nil, err
	}
	prior, err := readLimited(path, limit)
	if err != nil {
		return nil, tool.Artifact{}, "", nil, err
	}
	if !utf8.Valid(prior) {
		return nil, tool.Artifact{}, "", nil, errors.New("binary or non-UTF-8 files cannot be edited")
	}
	if input.ExpectedSHA256 != "" && !strings.EqualFold(input.ExpectedSHA256, digest(prior)) {
		return nil, tool.Artifact{}, "", nil, errors.New("file changed since expected_sha256 was observed")
	}
	if count := strings.Count(string(prior), input.OldText); count != 1 {
		return nil, tool.Artifact{}, "", nil, fmt.Errorf("old_text must match exactly once; matched %d times", count)
	}
	next := []byte(strings.Replace(string(prior), input.OldText, input.NewText, 1))
	if int64(len(next)) > limit {
		return nil, tool.Artifact{}, "", nil, fmt.Errorf("edited file exceeds %d byte limit", limit)
	}
	return fileMutationResult(relative, len(next), len(prior), digest(prior), next, map[string]any{"diff": map[string]int{"additions": lineCount([]byte(input.NewText)), "deletions": lineCount([]byte(input.OldText))}}), diffArtifact(relative, prior, next, false), path, next, nil
}

// promoteFile publishes a completed staging file to its final workspace path.
// Both source and target hashes are checked before the atomic rename so a
// partially written or concurrently changed file cannot be published.
func promoteFile(root string, limit int64, raw json.RawMessage) (any, tool.Artifact, error) {
	value, artifact, target, content, err := preparePromoteFile(root, limit, raw)
	if err != nil {
		return nil, tool.Artifact{}, err
	}
	if err := atomicWrite(target, content); err != nil {
		return nil, tool.Artifact{}, fmt.Errorf("atomically publish staged file: %w", err)
	}
	return value, artifact, nil
}

func preparePromoteFile(root string, limit int64, raw json.RawMessage) (any, tool.Artifact, string, []byte, error) {
	var input struct {
		SourcePath           string `json:"source_path"`
		TargetPath           string `json:"target_path"`
		ExpectedSourceSHA256 string `json:"expected_source_sha256"`
		ExpectedTargetSHA256 string `json:"expected_target_sha256"`
	}
	if err := json.Unmarshal(raw, &input); err != nil {
		return nil, tool.Artifact{}, "", nil, fmt.Errorf("decode promote_file arguments: %w", err)
	}
	if strings.TrimSpace(input.SourcePath) == "" || strings.TrimSpace(input.TargetPath) == "" {
		return nil, tool.Artifact{}, "", nil, errors.New("source_path and target_path are required")
	}
	if strings.TrimSpace(input.ExpectedSourceSHA256) == "" {
		return nil, tool.Artifact{}, "", nil, errors.New("expected_source_sha256 is required to publish a staged file")
	}
	source, sourceRelative, err := confinedPath(root, input.SourcePath, false)
	if err != nil {
		return nil, tool.Artifact{}, "", nil, fmt.Errorf("resolve staging source: %w", err)
	}
	content, err := readLimited(source, limit)
	if err != nil {
		return nil, tool.Artifact{}, "", nil, fmt.Errorf("read staging source: %w", err)
	}
	if !utf8.Valid(content) {
		return nil, tool.Artifact{}, "", nil, errors.New("staging source must be UTF-8 text")
	}
	sourceHash := digest(content)
	if !strings.EqualFold(strings.TrimSpace(input.ExpectedSourceSHA256), sourceHash) {
		return nil, tool.Artifact{}, "", nil, fmt.Errorf("staging source changed; current_source_sha256=%s", sourceHash)
	}
	target, targetRelative, err := confinedPath(root, input.TargetPath, true)
	if err != nil {
		return nil, tool.Artifact{}, "", nil, fmt.Errorf("resolve promotion target: %w", err)
	}
	prior, readErr := readLimited(target, limit)
	created := errors.Is(readErr, os.ErrNotExist)
	if readErr != nil && !created {
		return nil, tool.Artifact{}, "", nil, fmt.Errorf("read promotion target: %w", readErr)
	}
	previousHash := digest(prior)
	if !created {
		if strings.TrimSpace(input.ExpectedTargetSHA256) == "" {
			return nil, tool.Artifact{}, "", nil, fmt.Errorf("target exists; retry with expected_target_sha256=%s after reviewing the Diff", previousHash)
		}
		if !strings.EqualFold(strings.TrimSpace(input.ExpectedTargetSHA256), previousHash) {
			return nil, tool.Artifact{}, "", nil, fmt.Errorf("target changed; current_target_sha256=%s", previousHash)
		}
	}
	value := fileMutationResult(targetRelative, len(content), len(prior), previousHash, content, map[string]any{
		"source_path": sourceRelative, "source_file_sha256": sourceHash, "atomic_published": true,
		"created": created, "next_action": "Atomic publish completed. Verify syntax or runtime behavior from the published target; do not rewrite the whole file after a localized failure.",
	})
	return value, diffArtifact(targetRelative, prior, content, created), target, content, nil
}

// fileMutationResult is the model-facing receipt for a workspace mutation.
// file_sha256 identifies the bytes currently installed at path. Artifact
// hashes are assigned by the persistence layer and are deliberately not
// conflated with this file hash.
func fileMutationResult(path string, bytes, previousBytes int, previousFileSHA256 string, data []byte, extra map[string]any) map[string]any {
	result := map[string]any{
		"path":                 path,
		"bytes":                bytes,
		"line_count":           lineCount(data),
		"file_sha256":          digest(data),
		"content_sha256":       digest(data),
		"previous_bytes":       previousBytes,
		"previous_file_sha256": previousFileSHA256,
		"syntax_status":        fileSyntaxStatus(path),
	}
	for key, value := range extra {
		result[key] = value
	}
	return result
}

// Writes never guess that a source file is valid. The actual compiler or
// interpreter remains authoritative; this status prevents a successful write
// receipt from being mistaken for a successful build.
func fileSyntaxStatus(path string) string {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".py", ".go", ".js", ".jsx", ".ts", ".tsx", ".rs", ".java", ".c", ".cc", ".cpp", ".h", ".hpp":
		return "unverified"
	default:
		return "not_applicable"
	}
}

func diffArtifact(path string, before, after []byte, created bool) tool.Artifact {
	oldName := "a/" + path
	if created {
		oldName = "/dev/null"
	}
	var builder strings.Builder
	fmt.Fprintf(&builder, "--- %s\n+++ b/%s\n@@ -1,%d +1,%d @@\n", oldName, path, lineCount(before), lineCount(after))
	if len(before) > 0 {
		for _, line := range strings.Split(strings.TrimSuffix(string(before), "\n"), "\n") {
			builder.WriteByte('-')
			builder.WriteString(line)
			builder.WriteByte('\n')
		}
	}
	if len(after) > 0 {
		for _, line := range strings.Split(strings.TrimSuffix(string(after), "\n"), "\n") {
			builder.WriteByte('+')
			builder.WriteString(line)
			builder.WriteByte('\n')
		}
	}
	return tool.Artifact{Kind: "file_diff", Name: path + ".diff", MediaType: "text/x-diff", Content: []byte(builder.String()), Metadata: map[string]string{"path": path, "before_file_sha256": digest(before), "after_file_sha256": digest(after)}}
}

func workspaceFileArtifact(root string, limit int64, value any) (tool.Artifact, error) {
	result, ok := value.(map[string]any)
	if !ok {
		return tool.Artifact{}, errors.New("workspace mutation did not return a file path")
	}
	relative, _ := result["path"].(string)
	path, relative, err := confinedPath(root, relative, false)
	if err != nil {
		return tool.Artifact{}, err
	}
	content, err := readLimited(path, limit)
	if err != nil {
		return tool.Artifact{}, fmt.Errorf("read mutated workspace artifact: %w", err)
	}
	mediaType := "text/plain"
	if strings.HasSuffix(strings.ToLower(relative), ".py") {
		mediaType = "text/x-python"
	}
	return tool.Artifact{
		Kind: "workspace_file", Name: relative, MediaType: mediaType, Content: content,
		Metadata: map[string]string{"path": relative, "file_sha256": digest(content), "line_count": fmt.Sprintf("%d", lineCount(content)), "syntax_status": fileSyntaxStatus(relative), "phase": "applied"},
	}, nil
}

func lineCount(content []byte) int {
	if len(content) == 0 {
		return 0
	}
	count := strings.Count(string(content), "\n")
	if content[len(content)-1] != '\n' {
		count++
	}
	return count
}

func confinedPath(root, requested string, allowMissing bool) (string, string, error) {
	requested = strings.TrimSpace(requested)
	if requested == "" || filepath.IsAbs(requested) {
		return "", "", errors.New("path must be a non-empty workspace-relative path")
	}
	clean := filepath.Clean(requested)
	if clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", "", errors.New("path escapes workspace root")
	}
	if sensitivePath(clean) {
		return "", "", errors.New("path is blocked by workspace secret policy")
	}
	target := filepath.Join(root, clean)
	resolved, err := filepath.EvalSymlinks(target)
	if err == nil {
		if err := ensureWithin(root, resolved); err != nil {
			return "", "", err
		}
		return resolved, filepath.ToSlash(clean), nil
	}
	if !allowMissing || !errors.Is(err, os.ErrNotExist) {
		return "", "", fmt.Errorf("resolve workspace path: %w", err)
	}
	// A model may create a nested relative file in an otherwise empty Run
	// workspace. Resolve the nearest existing ancestor now; atomicWrite creates
	// only the missing descendants after the confinement check succeeds.
	parent := filepath.Dir(target)
	for {
		resolvedParent, parentErr := filepath.EvalSymlinks(parent)
		if parentErr == nil {
			parent = resolvedParent
			break
		}
		if !errors.Is(parentErr, os.ErrNotExist) {
			return "", "", fmt.Errorf("resolve target parent: %w", parentErr)
		}
		next := filepath.Dir(parent)
		if next == parent {
			return "", "", fmt.Errorf("resolve target parent: %w", parentErr)
		}
		parent = next
	}
	if err := ensureWithin(root, parent); err != nil {
		return "", "", err
	}
	return target, filepath.ToSlash(clean), nil
}

func ensureWithin(root, target string) error {
	relative, err := filepath.Rel(root, target)
	if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return errors.New("resolved path escapes workspace root")
	}
	return nil
}

func readLimited(path string, limit int64) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open workspace file: %w", err)
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, limit+1))
	if err != nil {
		return nil, fmt.Errorf("read workspace file: %w", err)
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("file exceeds %d byte limit", limit)
	}
	return data, nil
}

func atomicWrite(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		return fmt.Errorf("create workspace parent directory: %w", err)
	}
	file, err := os.CreateTemp(filepath.Dir(path), ".agent-write-*")
	if err != nil {
		return fmt.Errorf("create temporary file: %w", err)
	}
	temporary := file.Name()
	defer os.Remove(temporary)
	if err := file.Chmod(0o644); err != nil {
		file.Close()
		return err
	}
	if _, err := file.Write(data); err != nil {
		file.Close()
		return fmt.Errorf("write temporary file: %w", err)
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return fmt.Errorf("sync temporary file: %w", err)
	}
	if err := file.Close(); err != nil {
		return err
	}
	if err := os.Rename(temporary, path); err != nil {
		return fmt.Errorf("replace workspace file: %w", err)
	}
	return nil
}

func digest(data []byte) string { sum := sha256.Sum256(data); return hex.EncodeToString(sum[:]) }
func defaultPath(value string) string {
	if strings.TrimSpace(value) == "" {
		return "."
	}
	return value
}
func ignoredDirectory(entry fs.DirEntry) bool {
	return entry.IsDir() && (entry.Name() == ".git" || entry.Name() == ".ssh" || entry.Name() == ".deps" || entry.Name() == "node_modules" || entry.Name() == "dist")
}
func sensitivePath(value string) bool {
	for _, part := range strings.Split(filepath.ToSlash(value), "/") {
		lower := strings.ToLower(part)
		if lower == ".git" || lower == ".ssh" || lower == ".deps" || lower == ".env" || strings.HasPrefix(lower, ".env.") || strings.HasSuffix(lower, ".pem") || strings.HasSuffix(lower, ".key") {
			return true
		}
	}
	return false
}
func entryType(entry fs.DirEntry) string {
	if entry.IsDir() {
		return "directory"
	}
	if entry.Type()&os.ModeSymlink != 0 {
		return "symlink"
	}
	return "file"
}
