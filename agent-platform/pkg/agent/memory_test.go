package agent

import (
	"encoding/json"
	"testing"
)

func TestMemoryTaxonomyConstants(t *testing.T) {
	for _, layer := range []string{MemoryLayerManaged, MemoryLayerUser, MemoryLayerProject, MemoryLayerLocal, MemoryLayerAuto, MemoryLayerTeam} {
		if !ValidMemoryLayer(layer) {
			t.Fatalf("layer %q was rejected", layer)
		}
	}
	for _, semanticType := range []string{MemoryTypeUser, MemoryTypeFeedback, MemoryTypeProject, MemoryTypeReference} {
		if !ValidMemoryType(semanticType) {
			t.Fatalf("semantic type %q was rejected", semanticType)
		}
	}
	if SemanticTypeForLegacyKind("preference") != MemoryTypeFeedback {
		t.Fatal("preference must map to feedback")
	}
	if SemanticTypeForLegacyKind("episodic") != MemoryTypeProject {
		t.Fatal("episodic must map to project")
	}
}

func TestValidateMemoryExtractionCandidatesRejectsSecretsAndAcceptsFacts(t *testing.T) {
	valid := MemoryExtractionCandidate{
		SemanticType: MemoryTypeProject, Title: "API Gateway", Description: "The project uses Kong",
		Body: "Use Kong for gateway diagnostics.", StructuredData: json.RawMessage(`{"fact":"Use Kong for gateway diagnostics","why":"The project standardizes on Kong","how_to_apply":"Check Kong routes and plugins before suggesting nginx changes."}`),
		Confidence: 0.9, Importance: 0.8, FreshnessClass: MemoryFreshnessVolatile, SuggestedAction: "create",
	}
	if err := ValidateMemoryExtractionCandidates([]MemoryExtractionCandidate{valid}); err != nil {
		t.Fatal(err)
	}
	secret := valid
	secret.Body = "api_key=should-never-become-memory"
	if err := ValidateMemoryExtractionCandidates([]MemoryExtractionCandidate{secret}); err == nil {
		t.Fatal("secret-bearing candidate was accepted")
	}
}

func TestMemoryTaxonomyValidationNormalizesCaseAndWhitespace(t *testing.T) {
	if !ValidMemoryLayer("  AUTO ") || !ValidMemoryType(" FEEDBACK ") {
		t.Fatal("taxonomy validation should normalize case and whitespace")
	}
	if ValidMemoryLayer("global") || ValidMemoryType("semantic") {
		t.Fatal("legacy names must not be accepted as new taxonomy values")
	}
}
