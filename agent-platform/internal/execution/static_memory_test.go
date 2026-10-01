package execution

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
)

func TestResolveStaticMemoryFilesDiscoversLayersAndHashesRevisions(t *testing.T) {
	root := t.TempDir()
	home := t.TempDir()
	config := filepath.Join(home, ".config", "agent")
	if err := os.MkdirAll(config, 0o755); err != nil {
		t.Fatal(err)
	}
	write := func(path, content string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	managed := filepath.Join(root, "managed.md")
	write(managed, "managed rule")
	write(filepath.Join(config, "AGENT.md"), "user preference")
	write(filepath.Join(root, "CLAUDE.md"), "project rule")
	write(filepath.Join(root, "CLAUDE.local.md"), "local rule")
	write(filepath.Join(root, ".agent", "memory", "auto", "gateway.md"), "auto fact")
	write(filepath.Join(root, ".agent", "memory", "team", "team.md"), "team fact")
	lookup := func(key string) (string, bool) {
		switch key {
		case "AGENT_MANAGED_MEMORY_FILE":
			return managed, true
		case "HOME":
			return home, true
		case "AGENT_PROJECT_KEY":
			return "project-a", true
		case "AGENT_TEAM_ID":
			return "team-a", true
		default:
			return "", false
		}
	}
	documents, err := resolveStaticMemoryFiles(root, lookup)
	if err != nil {
		t.Fatal(err)
	}
	if len(documents) != 6 {
		t.Fatalf("documents = %d, want 6", len(documents))
	}
	seen := map[string]bool{}
	for _, document := range documents {
		seen[document.SourceLayer] = true
		if document.ContentHash == "" || document.ID == "" || document.Content == "" {
			t.Fatalf("incomplete document: %+v", document)
		}
		if document.SourceLayer == agent.MemoryLayerProject && document.Path != "CLAUDE.md" {
			t.Fatalf("project path = %q", document.Path)
		}
		if document.ProjectKey != "project-a" || document.TeamID != "team-a" {
			t.Fatalf("static source scope = %+v", document)
		}
	}
	for _, layer := range []string{agent.MemoryLayerManaged, agent.MemoryLayerUser, agent.MemoryLayerProject, agent.MemoryLayerLocal, agent.MemoryLayerAuto, agent.MemoryLayerTeam} {
		if !seen[layer] {
			t.Fatalf("missing layer %q: %+v", layer, seen)
		}
	}
}
