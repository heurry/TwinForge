package skill

import "testing"

func TestParseUploadMarkdownFrontMatter(t *testing.T) {
	t.Parallel()
	content := []byte("---\nname: Evidence Reviewer\nkey: evidence-reviewer\ndescription: Reviews observable facts\npriority: 20\nrequired_tools:\n  - id: 2b7c7424-32ba-4697-85ef-82258624240e\n    version: \"1\"\n---\n# Workflow\n\nCollect evidence before conclusions.\n")
	imported, err := ParseUpload("SKILL.md", content)
	if err != nil {
		t.Fatalf("ParseUpload() error = %v", err)
	}
	if imported.Key != "evidence-reviewer" || imported.Name != "Evidence Reviewer" {
		t.Fatalf("identity = %q/%q", imported.Key, imported.Name)
	}
	if len(imported.Spec.Instructions) != 1 || imported.Spec.Instructions[0].Priority != 20 {
		t.Fatalf("spec = %+v", imported.Spec)
	}
	if len(imported.Spec.RequiredTools) != 1 || imported.Spec.RequiredTools[0].Version != "1" {
		t.Fatalf("required tools = %+v", imported.Spec.RequiredTools)
	}
}

func TestParseUploadMarkdownUsesFilenameDefaults(t *testing.T) {
	t.Parallel()
	imported, err := ParseUpload("release-review.md", []byte("Review the release evidence."))
	if err != nil {
		t.Fatalf("ParseUpload() error = %v", err)
	}
	if imported.Key != "release-review" || imported.Name != "release review" {
		t.Fatalf("identity = %q/%q", imported.Key, imported.Name)
	}
}

func TestParseUploadRejectsAmbiguousSkillFile(t *testing.T) {
	t.Parallel()
	_, err := ParseUpload("SKILL.md", []byte("Instructions without identity."))
	if err == nil {
		t.Fatal("ParseUpload() expected missing identity error")
	}
}

func TestParseUploadJSON(t *testing.T) {
	t.Parallel()
	imported, err := ParseUpload("skill.json", []byte(`{"key":"json-skill","name":"JSON Skill","spec":{"instructions":[{"name":"workflow","content":"Do the work."}]}}`))
	if err != nil || imported.Key != "json-skill" {
		t.Fatalf("ParseUpload() = %+v, %v", imported, err)
	}
}
