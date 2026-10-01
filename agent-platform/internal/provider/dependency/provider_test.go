package dependency

import (
	"encoding/json"
	"testing"
)

func TestRequestAllowsOnlyExactRunScopedPackages(t *testing.T) {
	t.Parallel()
	valid := Request{Ecosystem: "python", Packages: []Package{{Name: "requests", Version: "2.32.3"}}, Source: "pypi", Scope: "run", Reason: "HTTP client required by the generated program"}
	if err := valid.NormalizeAndValidate(); err != nil {
		t.Fatalf("valid request: %v", err)
	}
	for name, request := range map[string]Request{
		"unpinned": {Ecosystem: "python", Packages: []Package{{Name: "requests", Version: "*"}}, Source: "pypi", Scope: "run", Reason: "needed"},
		"url":      {Ecosystem: "python", Packages: []Package{{Name: "https://evil.invalid/pkg.whl", Version: "1.0"}}, Source: "pypi", Scope: "run", Reason: "needed"},
		"flag":     {Ecosystem: "python", Packages: []Package{{Name: "--target", Version: "1.0"}}, Source: "pypi", Scope: "run", Reason: "needed"},
		"global":   {Ecosystem: "python", Packages: []Package{{Name: "requests", Version: "2.32.3"}}, Source: "pypi", Scope: "global", Reason: "needed"},
	} {
		t.Run(name, func(t *testing.T) {
			if err := request.NormalizeAndValidate(); err == nil {
				t.Fatal("unsafe request was accepted")
			}
		})
	}
}

func TestDefinitionContainsNoShellOrURLFields(t *testing.T) {
	t.Parallel()
	definition := Definition([]string{"internal", "pypi"})
	var schema map[string]any
	if err := json.Unmarshal(definition.InputSchema, &schema); err != nil {
		t.Fatal(err)
	}
	properties := schema["properties"].(map[string]any)
	for _, forbidden := range []string{"command", "args", "url", "target", "index_url"} {
		if _, ok := properties[forbidden]; ok {
			t.Fatalf("schema exposes forbidden field %q", forbidden)
		}
	}
}
