package execution

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
)

const (
	staticMemoryFileLimit  = 128 << 10
	staticMemoryTotalLimit = 512 << 10
)

// resolveStaticMemoryFiles discovers operator/user/project/local documents
// plus optional exported Auto/Team directories. The returned snapshots are
// immutable for the lifetime of a Run and carry content hashes for replay.
// Database-backed Auto/Team memories still use the dynamic retrieval path.
func resolveStaticMemoryFiles(root string, lookupEnv func(string) (string, bool)) ([]agent.StaticMemoryDocument, error) {
	root = strings.TrimSpace(root)
	if root == "" {
		return []agent.StaticMemoryDocument{}, nil
	}
	root, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("resolve static memory root: %w", err)
	}
	if lookupEnv == nil {
		lookupEnv = os.LookupEnv
	}
	projectKey, _ := lookupEnv("AGENT_PROJECT_KEY")
	teamID, _ := lookupEnv("AGENT_TEAM_ID")
	type candidate struct {
		layer string
		path  string
	}
	candidates := make([]candidate, 0, 16)
	add := func(layer, path string) {
		if strings.TrimSpace(path) != "" {
			candidates = append(candidates, candidate{layer: layer, path: path})
		}
	}
	if managed, ok := lookupEnv("AGENT_MANAGED_MEMORY_FILE"); ok {
		add(agent.MemoryLayerManaged, managed)
	}
	userFiles := userStaticMemoryFiles(lookupEnv)
	for _, path := range userFiles {
		add(agent.MemoryLayerUser, path)
	}
	for _, name := range []string{"AGENTS.md", "CLAUDE.md", ".claude/CLAUDE.md"} {
		add(agent.MemoryLayerProject, filepath.Join(root, filepath.FromSlash(name)))
	}
	for _, name := range []string{"CLAUDE.local.md", ".claude/CLAUDE.local.md"} {
		add(agent.MemoryLayerLocal, filepath.Join(root, filepath.FromSlash(name)))
	}
	for _, layer := range []string{agent.MemoryLayerAuto, agent.MemoryLayerTeam} {
		matches, globErr := filepath.Glob(filepath.Join(root, ".agent", "memory", layer, "*.md"))
		if globErr != nil {
			return nil, fmt.Errorf("discover %s static memories: %w", layer, globErr)
		}
		sort.Strings(matches)
		for _, path := range matches {
			add(layer, path)
		}
	}

	seen := make(map[string]struct{}, len(candidates))
	documents := make([]agent.StaticMemoryDocument, 0, len(candidates))
	totalBytes := 0
	for _, item := range candidates {
		path, err := filepath.Abs(item.path)
		if err != nil {
			return nil, fmt.Errorf("resolve static memory path: %w", err)
		}
		if _, exists := seen[path]; exists {
			continue
		}
		seen[path] = struct{}{}
		data, readErr := readStaticMemoryFile(path)
		if errors.Is(readErr, os.ErrNotExist) {
			continue
		}
		if readErr != nil {
			return nil, fmt.Errorf("read %s static memory %s: %w", item.layer, path, readErr)
		}
		if totalBytes+len(data) > staticMemoryTotalLimit {
			return nil, fmt.Errorf("static memory snapshot exceeds %d bytes", staticMemoryTotalLimit)
		}
		totalBytes += len(data)
		hash := sha256.Sum256(data)
		hashText := hex.EncodeToString(hash[:])
		relative := path
		if rel, relErr := filepath.Rel(root, path); relErr == nil && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && rel != ".." {
			relative = filepath.ToSlash(rel)
		}
		documents = append(documents, agent.StaticMemoryDocument{
			ID:          "static-" + hashText,
			SourceLayer: item.layer,
			Path:        relative,
			ProjectKey:  strings.TrimSpace(projectKey),
			TeamID:      strings.TrimSpace(teamID),
			ContentHash: hashText,
			ObservedAt:  time.Now().UTC(),
			Content:     string(data),
		})
	}
	return documents, nil
}

func userStaticMemoryFiles(lookupEnv func(string) (string, bool)) []string {
	paths := make([]string, 0, 3)
	configHome, configOK := lookupEnv("XDG_CONFIG_HOME")
	if !configOK || strings.TrimSpace(configHome) == "" {
		if home, homeOK := lookupEnv("HOME"); homeOK && strings.TrimSpace(home) != "" {
			configHome = filepath.Join(home, ".config")
		}
	}
	if strings.TrimSpace(configHome) != "" {
		paths = append(paths, filepath.Join(configHome, "agent", "AGENT.md"))
	}
	if home, ok := lookupEnv("HOME"); ok && strings.TrimSpace(home) != "" {
		paths = append(paths, filepath.Join(home, "AGENTS.md"))
	}
	return paths
}

func readStaticMemoryFile(path string) ([]byte, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return nil, errors.New("static memory path must be a regular file")
	}
	if info.Size() > staticMemoryFileLimit {
		return nil, fmt.Errorf("static memory file exceeds %d bytes", staticMemoryFileLimit)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if !utf8.Valid(data) {
		return nil, errors.New("static memory file is not valid UTF-8")
	}
	return data, nil
}
