package skill

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"gopkg.in/yaml.v3"
)

var skillKeyPattern = regexp.MustCompile(`^[a-z][a-z0-9._-]{0,127}$`)

// Import is the normalized immutable Skill revision produced from an uploaded
// SKILL.md or JSON document.
type Import struct {
	Key  string `json:"key"`
	Name string `json:"name"`
	Spec Spec   `json:"spec"`
}

type markdownFrontMatter struct {
	Key             string             `yaml:"key"`
	Name            string             `yaml:"name"`
	Description     string             `yaml:"description"`
	InstructionName string             `yaml:"instruction_name"`
	Priority        int                `yaml:"priority"`
	RequiredTools   []agent.VersionRef `yaml:"required_tools"`
}

// ParseUpload accepts a UTF-8 Markdown skill or a structured JSON export.
func ParseUpload(filename string, content []byte) (Import, error) {
	if len(content) == 0 {
		return Import{}, errors.New("skill file is empty")
	}
	if !utf8.Valid(content) {
		return Import{}, errors.New("skill file must be UTF-8 text")
	}
	extension := strings.ToLower(filepath.Ext(filename))
	switch extension {
	case ".md", ".markdown":
		return parseMarkdownUpload(filename, string(bytes.TrimPrefix(content, []byte{0xef, 0xbb, 0xbf})))
	case ".json":
		var imported Import
		if err := json.Unmarshal(content, &imported); err != nil {
			return Import{}, fmt.Errorf("decode skill JSON: %w", err)
		}
		return validateImport(imported)
	default:
		return Import{}, errors.New("skill file must use .md, .markdown, or .json")
	}
}

func parseMarkdownUpload(filename, content string) (Import, error) {
	normalized := strings.ReplaceAll(content, "\r\n", "\n")
	front := markdownFrontMatter{}
	body := normalized
	if strings.HasPrefix(normalized, "---\n") {
		remaining := normalized[4:]
		end := strings.Index(remaining, "\n---\n")
		if end < 0 {
			return Import{}, errors.New("SKILL.md front matter is not terminated by ---")
		}
		if err := yaml.Unmarshal([]byte(remaining[:end]), &front); err != nil {
			return Import{}, fmt.Errorf("decode SKILL.md front matter: %w", err)
		}
		body = remaining[end+5:]
	}
	body = strings.TrimSpace(body)
	if body == "" {
		return Import{}, errors.New("SKILL.md instructions are empty")
	}
	base := strings.TrimSuffix(filepath.Base(filename), filepath.Ext(filename))
	name := strings.TrimSpace(front.Name)
	if name == "" && !strings.EqualFold(base, "SKILL") {
		name = strings.ReplaceAll(base, "-", " ")
	}
	key := strings.TrimSpace(front.Key)
	if key == "" && !strings.EqualFold(base, "SKILL") {
		key = slugKey(base)
	}
	instructionName := strings.TrimSpace(front.InstructionName)
	if instructionName == "" {
		instructionName = "SKILL.md"
	}
	return validateImport(Import{
		Key:  key,
		Name: name,
		Spec: Spec{
			Description:   strings.TrimSpace(front.Description),
			Instructions:  []InstructionBlock{{Name: instructionName, Content: body, Priority: front.Priority}},
			RequiredTools: front.RequiredTools,
		},
	})
}

func validateImport(imported Import) (Import, error) {
	imported.Key = strings.TrimSpace(imported.Key)
	imported.Name = strings.TrimSpace(imported.Name)
	if !skillKeyPattern.MatchString(imported.Key) {
		return Import{}, errors.New("skill key must match ^[a-z][a-z0-9._-]{0,127}$")
	}
	if imported.Name == "" {
		return Import{}, errors.New("skill name is required; set front matter name or upload form name")
	}
	if err := imported.Spec.Validate(); err != nil {
		return Import{}, err
	}
	return imported, nil
}

func slugKey(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	var builder strings.Builder
	lastSeparator := false
	for _, char := range value {
		valid := char >= 'a' && char <= 'z' || char >= '0' && char <= '9' || char == '.' || char == '_'
		if valid {
			builder.WriteRune(char)
			lastSeparator = false
			continue
		}
		if !lastSeparator && builder.Len() > 0 {
			builder.WriteByte('-')
			lastSeparator = true
		}
	}
	return strings.Trim(builder.String(), "-")
}
