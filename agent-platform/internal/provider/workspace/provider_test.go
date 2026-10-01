package workspace

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/resource"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

func TestReadFileConfinesPathsAndReportsDigest(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "note.txt"), []byte("one\ntwo\nthree"), 0o644); err != nil {
		t.Fatal(err)
	}
	handler, err := NewHandler(testSpec("read_file", tool.RiskRead), Config{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	result, err := handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"path":"note.txt","start_line":2,"line_count":1}`)})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(result.Content), `"content":"two"`) || !strings.Contains(string(result.Content), `"file_sha256"`) {
		t.Fatalf("unexpected result: %s", result.Content)
	}
	_, err = handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"path":"../secret"}`)})
	if err == nil || !strings.Contains(err.Error(), "escapes workspace") {
		t.Fatalf("expected traversal rejection, got %v", err)
	}
}

func TestCreateDirectoryIsConfinedAndIdempotent(t *testing.T) {
	root := t.TempDir()
	handler, err := NewHandler(testSpec("create_directory", tool.RiskLowWrite), Config{Root: root, AllowWrite: true})
	if err != nil {
		t.Fatal(err)
	}
	result, err := handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"path":"tmp/nested"}`)})
	if err != nil || !strings.Contains(string(result.Content), `"created":true`) {
		t.Fatalf("create result=%s err=%v", result.Content, err)
	}
	result, err = handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"path":"tmp/nested"}`)})
	if err != nil || !strings.Contains(string(result.Content), `"created":false`) {
		t.Fatalf("idempotent result=%s err=%v", result.Content, err)
	}
	if _, err := handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"path":"../escape"}`)}); err == nil {
		t.Fatal("directory traversal must be rejected")
	}
}

func TestReadFileRejectsEscapingSymlink(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	secret := filepath.Join(outside, "secret.txt")
	if err := os.WriteFile(secret, []byte("secret"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(secret, filepath.Join(root, "escape")); err != nil {
		t.Fatal(err)
	}
	handler, err := NewHandler(testSpec("read_file", tool.RiskRead), Config{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	_, err = handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"path":"escape"}`)})
	if err == nil || !strings.Contains(err.Error(), "escapes workspace") {
		t.Fatalf("expected symlink escape rejection, got %v", err)
	}
}

func TestReadFileRangeSupportsFilesLargerThanWholeFileLimit(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "large.txt")
	content := strings.Repeat(strings.Repeat("x", 900)+"\n", 400)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	handler, err := NewHandler(testSpec("read_file", tool.RiskRead), Config{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	result, err := handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"path":"large.txt","start_line":381,"line_count":20}`)})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(result.Content), `"start_line":381`) || !strings.Contains(string(result.Content), `"line_count":20`) {
		t.Fatalf("unexpected ranged result: %s", result.Content)
	}
}

func TestReadFilePlacesTailBeforeContentForCompaction(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "partial.py"), []byte("first\nsecond\nLAST_MARKER\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	handler, err := NewHandler(testSpec("read_file", tool.RiskRead), Config{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	result, err := handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"path":"partial.py"}`)})
	if err != nil {
		t.Fatal(err)
	}
	encoded := string(result.Content)
	if strings.Index(encoded, `"tail"`) < 0 || strings.Index(encoded, `"tail"`) > strings.Index(encoded, `"content"`) || !strings.Contains(encoded, "LAST_MARKER") {
		t.Fatalf("read result does not prioritize tail metadata: %s", encoded)
	}
}

func TestEditRequiresWritePolicyAndObservationMatch(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "note.txt")
	if err := os.WriteFile(path, []byte("before"), 0o644); err != nil {
		t.Fatal(err)
	}
	spec := testSpec("edit_file", tool.RiskLowWrite)
	if _, err := NewHandler(spec, Config{Root: root}); err == nil {
		t.Fatal("expected disabled write policy")
	}
	handler, err := NewHandler(spec, Config{Root: root, AllowWrite: true})
	if err != nil {
		t.Fatal(err)
	}
	_, err = handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"path":"note.txt","old_text":"before","new_text":"after","expected_sha256":"stale"}`)})
	if err == nil || !strings.Contains(err.Error(), "changed") {
		t.Fatalf("expected observation conflict, got %v", err)
	}
	_, err = handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"path":"note.txt","old_text":"before","new_text":"after"}`)})
	if err != nil {
		t.Fatal(err)
	}
	content, _ := os.ReadFile(path)
	if string(content) != "after" {
		t.Fatalf("content = %q", content)
	}
}

func TestPreviewProducesDiffWithoutMutatingFile(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "note.txt")
	if err := os.WriteFile(path, []byte("before\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	handler, err := NewPreviewHandler(testSpec("edit_file", tool.RiskLowWrite), Config{Root: root, AllowWrite: true})
	if err != nil {
		t.Fatal(err)
	}
	result, err := handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"path":"note.txt","old_text":"before","new_text":"after"}`)})
	if err != nil {
		t.Fatal(err)
	}
	content, _ := os.ReadFile(path)
	if string(content) != "before\n" {
		t.Fatalf("preview mutated file: %q", content)
	}
	if len(result.Artifacts) != 1 || result.Artifacts[0].MediaType != "text/x-diff" || !strings.Contains(string(result.Artifacts[0].Content), "+after") {
		t.Fatalf("unexpected preview artifact: %+v", result.Artifacts)
	}
}

func TestRunCommandPolicyAllowsRestrictedInlineProbe(t *testing.T) {
	if err := validateCommand("sh", []string{"-c", "true"}); err == nil {
		t.Fatal("expected shell command to be rejected")
	}
	if err := validateCommand("python3", []string{"-c", "import pygame; print(pygame.version.ver)"}); err != nil {
		t.Fatalf("safe inline probe was rejected: %v", err)
	}
	if err := validateCommand("python3", []string{"-c", "import subprocess; subprocess.run(['id'])"}); err == nil {
		t.Fatal("expected subprocess inline probe to be rejected")
	}
	if err := validateCommand("python3", []string{"-m", "py_compile", "-c", "print('unsafe')"}); err == nil {
		t.Fatal("expected nested inline Python flag to be rejected")
	}
	if err := validateCommand("python3", []string{"-m", "pytest"}); err == nil {
		t.Fatal("expected non-allowlisted module to be rejected")
	}
	if err := validateCommand("python3", []string{"-m", "py_compile", "game.py"}); err != nil {
		t.Fatalf("expected py_compile to be allowed: %v", err)
	}
}

func TestRunCommandExecutesRestrictedInlineProbe(t *testing.T) {
	handler, err := NewHandler(testSpec("run_command", tool.RiskHigh), Config{Root: t.TempDir(), AllowWrite: true})
	if err != nil {
		t.Fatal(err)
	}
	result, err := handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"command":"python3","args":["-c","print('PROBE_OK')"],"timeout_seconds":30}`)})
	if err != nil || result.IsError || !strings.Contains(string(result.Content), `"execution_profile":"inline_probe"`) || !strings.Contains(string(result.Content), "PROBE_OK") {
		t.Fatalf("unexpected inline probe result: result=%+v err=%v", result, err)
	}
}

func TestRunCommandRequiresWritePolicy(t *testing.T) {
	root := t.TempDir()
	if _, err := NewHandler(testSpec("run_command", tool.RiskHigh), Config{Root: root}); err == nil {
		t.Fatal("expected command execution to require workspace write policy")
	}
}

func TestRunCommandExecutesWorkspacePythonAndReturnsObservations(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "probe.py"), []byte("print('sandbox-ok')\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	handler, err := NewHandler(testSpec("run_command", tool.RiskHigh), Config{Root: root, AllowWrite: true})
	if err != nil {
		t.Fatal(err)
	}
	result, err := handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"command":"python3","args":["probe.py"],"timeout_seconds":5,"working_directory":"."}`)})
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError || !strings.Contains(string(result.Content), `"exit_code":0`) || !strings.Contains(string(result.Content), `sandbox-ok`) || !strings.Contains(string(result.Content), `"sandbox_policy":"workspace_script_allowed"`) {
		t.Fatalf("unexpected direct execution result: %+v content=%s", result, result.Content)
	}
}

func TestRunCommandReturnsPythonTracebackToAgent(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "broken.py"), []byte("raise RuntimeError('visible-failure')\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	handler, err := NewHandler(testSpec("run_command", tool.RiskHigh), Config{Root: root, AllowWrite: true})
	if err != nil {
		t.Fatal(err)
	}
	result, err := handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"command":"python3","args":["broken.py"],"timeout_seconds":5}`)})
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError || !strings.Contains(string(result.Content), `"exit_code":1`) || !strings.Contains(string(result.Content), `visible-failure`) || !strings.Contains(string(result.Content), `"failure_kind":"process_exit"`) || !strings.Contains(string(result.Content), `"diagnostic":"RuntimeError: visible-failure"`) || !strings.Contains(string(result.Content), `"stderr_tail"`) || !strings.Contains(string(result.Content), `not a Sandbox policy rejection`) || !strings.Contains(result.Meta["correction"], "unchanged workspace") {
		t.Fatalf("traceback was not returned to the agent: %+v content=%s", result, result.Content)
	}
}

func TestAppendFileUsesObservedDigest(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "game.py")
	if err := os.WriteFile(path, []byte("one\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	handler, err := NewHandler(testSpec("append_file", tool.RiskLowWrite), Config{Root: root, AllowWrite: true})
	if err != nil {
		t.Fatal(err)
	}
	result, err := handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"path":"game.py","content":"two\n","expected_sha256":"stale"}`)})
	if err == nil || result.Content != nil {
		t.Fatalf("expected stale digest rejection, result=%+v err=%v", result, err)
	}
	result, err = handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"path":"game.py","content":"two\n"}`)})
	if err != nil {
		t.Fatal(err)
	}
	content, _ := os.ReadFile(path)
	if string(content) != "one\ntwo\n" || len(result.Artifacts) != 2 || result.Artifacts[1].Kind != "workspace_file" || string(result.Artifacts[1].Content) != "one\ntwo\n" {
		t.Fatalf("content=%q artifacts=%d", content, len(result.Artifacts))
	}
	var receipt map[string]any
	if err := json.Unmarshal(result.Content, &receipt); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"file_sha256", "content_sha256", "line_count", "syntax_status"} {
		if _, ok := receipt[key]; !ok {
			t.Fatalf("append receipt missing %q: %s", key, result.Content)
		}
	}
	if receipt["syntax_status"] != "unverified" {
		t.Fatalf("unexpected syntax status: %#v", receipt["syntax_status"])
	}
	if _, ambiguous := receipt["sha256"]; ambiguous {
		t.Fatalf("mutation receipt still exposes ambiguous sha256: %s", result.Content)
	}
}

func TestPromoteFilePublishesHashCheckedStagingFile(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "module.py.part")
	target := filepath.Join(root, "module.py")
	content := []byte("print('complete')\n")
	if err := os.WriteFile(source, content, 0o644); err != nil {
		t.Fatal(err)
	}
	handler, err := NewHandler(testSpec("promote_file", tool.RiskLowWrite), Config{Root: root, AllowWrite: true})
	if err != nil {
		t.Fatal(err)
	}
	result, err := handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"source_path":"module.py.part","target_path":"module.py","expected_source_sha256":"` + digest(content) + `"}`)})
	if err != nil {
		t.Fatal(err)
	}
	published, _ := os.ReadFile(target)
	if string(published) != string(content) || len(result.Artifacts) != 2 {
		t.Fatalf("published=%q artifacts=%d", published, len(result.Artifacts))
	}
	var receipt map[string]any
	if err := json.Unmarshal(result.Content, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt["atomic_published"] != true || receipt["source_file_sha256"] != digest(content) || receipt["file_sha256"] != digest(content) {
		t.Fatalf("unexpected promotion receipt: %s", result.Content)
	}

	if err := os.WriteFile(target, []byte("changed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err = handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"source_path":"module.py.part","target_path":"module.py","expected_source_sha256":"` + digest(content) + `","expected_target_sha256":"stale"}`)})
	if err == nil || !strings.Contains(err.Error(), "target changed") {
		t.Fatalf("expected optimistic concurrency rejection, got %v", err)
	}
}

func TestPromotePreviewDoesNotPublish(t *testing.T) {
	root := t.TempDir()
	content := []byte("print('staged')\n")
	if err := os.WriteFile(filepath.Join(root, "module.py.part"), content, 0o644); err != nil {
		t.Fatal(err)
	}
	handler, err := NewPreviewHandler(testSpec("promote_file", tool.RiskLowWrite), Config{Root: root, AllowWrite: true})
	if err != nil {
		t.Fatal(err)
	}
	result, err := handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"source_path":"module.py.part","target_path":"module.py","expected_source_sha256":"` + digest(content) + `"}`)})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "module.py")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("preview published target: %v", err)
	}
	if len(result.Artifacts) != 1 || result.Artifacts[0].Metadata["phase"] != "proposed" {
		t.Fatalf("unexpected preview result: %+v", result.Artifacts)
	}
}

func TestWriteAndAppendAcceptSmallModelFieldAliases(t *testing.T) {
	root := t.TempDir()
	writeHandler, err := NewHandler(testSpec("write_file", tool.RiskLowWrite), Config{Root: root, AllowWrite: true})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := writeHandler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"file_path":"small.py","text":"print('one')\n"}`)}); err != nil {
		t.Fatal(err)
	}
	appendHandler, err := NewHandler(testSpec("append_file", tool.RiskLowWrite), Config{Root: root, AllowWrite: true})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := appendHandler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"file_path":"small.py","text":"print('two')\n"}`)}); err != nil {
		t.Fatal(err)
	}
	content, _ := os.ReadFile(filepath.Join(root, "small.py"))
	if string(content) != "print('one')\nprint('two')\n" {
		t.Fatalf("content=%q", content)
	}
}

func TestWriteReceiptUsesChunkContinuationContract(t *testing.T) {
	root := t.TempDir()
	handler, err := NewHandler(testSpec("write_file", tool.RiskLowWrite), Config{Root: root, AllowWrite: true})
	if err != nil {
		t.Fatal(err)
	}
	result, err := handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"path":"core.py","content":"def first():\n    return 1\n"}`)})
	if err != nil {
		t.Fatal(err)
	}
	var receipt struct {
		FileSHA256 string `json:"file_sha256"`
		NextAction string `json:"next_action"`
	}
	if err := json.Unmarshal(result.Content, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.FileSHA256 == "" || !strings.Contains(receipt.NextAction, "append_file") || !strings.Contains(receipt.NextAction, receipt.FileSHA256) || !strings.Contains(receipt.NextAction, "do not compact") {
		t.Fatalf("write receipt does not preserve the chunk continuation contract: %s", result.Content)
	}
	if strings.Contains(receipt.NextAction, "complete module in one call") {
		t.Fatalf("write receipt retained the conflicting one-call policy: %s", result.Content)
	}
}

func TestListFilesHidesInternalEntriesAndPrioritizesFiles(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"zeta.txt", "alpha.txt", ".secret"} {
		if err := os.WriteFile(filepath.Join(root, name), []byte(name), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Mkdir(filepath.Join(root, "aaa-directory"), 0o755); err != nil {
		t.Fatal(err)
	}
	handler, err := NewHandler(testSpec("list_files", tool.RiskRead), Config{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	result, err := handler(context.Background(), tool.Call{Arguments: json.RawMessage(`{"max_entries":2}`)})
	if err != nil {
		t.Fatal(err)
	}
	encoded := string(result.Content)
	if strings.Contains(encoded, ".secret") || strings.Contains(encoded, "aaa-directory") || !strings.Contains(encoded, "alpha.txt") || !strings.Contains(encoded, "zeta.txt") {
		t.Fatalf("unexpected prioritized listing: %s", encoded)
	}
}

func TestEnsureRunRootIsolatesRuns(t *testing.T) {
	base := t.TempDir()
	first, err := EnsureRunRoot(base, "run-one")
	if err != nil {
		t.Fatal(err)
	}
	second, err := EnsureRunRoot(base, "run-two")
	if err != nil {
		t.Fatal(err)
	}
	if first == second || filepath.Dir(first) != filepath.Dir(second) {
		t.Fatalf("Run roots are not isolated under one container: first=%q second=%q", first, second)
	}
	if err := os.WriteFile(filepath.Join(first, "artifact.txt"), []byte("one"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(second, "artifact.txt")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("second Run observed first Run artifact: %v", err)
	}
}

func TestEnsureRunRootRejectsUnsafeIdentifier(t *testing.T) {
	if _, err := EnsureRunRoot(t.TempDir(), "../escape"); err == nil {
		t.Fatal("unsafe Run id was accepted")
	}
}

func TestPromoteRunFileUsesOptimisticTargetHash(t *testing.T) {
	base := t.TempDir()
	runRoot, err := EnsureRunRoot(base, "run-promote")
	if err != nil {
		t.Fatal(err)
	}
	source := []byte("print('from agent')\n")
	if err := os.WriteFile(filepath.Join(runRoot, "game.py"), source, 0o600); err != nil {
		t.Fatal(err)
	}
	previous, result, created, err := PromoteRunFile(base, "run-promote", "game.py", digest(source), "game.py", "")
	if err != nil || !created || previous != digest(nil) || result != digest(source) {
		t.Fatalf("new target promotion previous=%q result=%q created=%v err=%v", previous, result, created, err)
	}
	userContent := []byte("user change\n")
	if err := os.WriteFile(filepath.Join(base, "game.py"), userContent, 0o600); err != nil {
		t.Fatal(err)
	}
	previous, _, _, err = PromoteRunFile(base, "run-promote", "game.py", digest(source), "game.py", "")
	if err == nil || previous != digest(userContent) {
		t.Fatalf("existing target without optimistic hash previous=%q err=%v", previous, err)
	}
	_, result, created, err = PromoteRunFile(base, "run-promote", "game.py", digest(source), "game.py", previous)
	if err != nil || created || result != digest(source) {
		t.Fatalf("approved overwrite result=%q created=%v err=%v", result, created, err)
	}
}

func TestReadProjectFileSnapshotReturnsRangeAndRejectsInternalPaths(t *testing.T) {
	root := t.TempDir()
	content := []byte("first\nsecond\nthird\n")
	if err := os.WriteFile(filepath.Join(root, "notes.md"), content, 0o600); err != nil {
		t.Fatal(err)
	}
	snapshot, err := ReadProjectFileSnapshot(root, "notes.md", 2, 1, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Path != "notes.md" || snapshot.Content != "second" || snapshot.SHA256 != digest(content) || string(snapshot.SnapshotContent) != string(content) {
		t.Fatalf("unexpected project snapshot: %#v", snapshot)
	}
	if _, err := ReadProjectFileSnapshot(root, "../notes.md", 1, 1, 1<<20); err == nil {
		t.Fatal("project path traversal was accepted")
	}
	if _, err := ReadProjectFileSnapshot(root, ".agent-workspaces/run/secret", 1, 1, 1<<20); err == nil {
		t.Fatal("internal Run workspace path was accepted")
	}
}

func testSpec(operation string, risk tool.Risk) resource.ToolSpec {
	return resource.ToolSpec{
		Definition: tool.Definition{
			Name: operation, Version: "1", Description: operation,
			InputSchema: json.RawMessage(`{"type":"object"}`), Risk: risk, ExecutionMode: tool.ExecutionSerial,
		},
		ProviderType: "workspace", Workspace: &resource.WorkspaceProvider{Operation: operation},
	}
}
