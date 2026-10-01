package httpx

import "testing"

func TestMergeLaunchConfigKeepsNestedDefaultsAndOverridesValues(t *testing.T) {
	dst := map[string]any{
		"name":        "base",
		"hyperparams": map[string]any{"epochs": 3.0, "learning_rate": 0.001},
	}
	mergeLaunchConfig(dst, map[string]any{
		"name":        "override",
		"hyperparams": map[string]any{"epochs": 5.0},
	})
	if dst["name"] != "override" {
		t.Fatalf("top-level override was not applied: %+v", dst)
	}
	hyperparams := dst["hyperparams"].(map[string]any)
	if hyperparams["epochs"] != 5.0 || hyperparams["learning_rate"] != 0.001 {
		t.Fatalf("nested merge lost defaults or override: %+v", hyperparams)
	}
}

func TestDetectLogLevel(t *testing.T) {
	for line, want := range map[string]string{
		"INFO runtime ready":       "info",
		"WARNING queue saturated":  "warning",
		"CUDA out of memory error": "error",
	} {
		if got := detectLogLevel(line); got != want {
			t.Fatalf("detectLogLevel(%q)=%q, want %q", line, got, want)
		}
	}
}

func TestInferenceReleaseSLOGateBlocksLatencyRegression(t *testing.T) {
	profile, _ := inferenceReleaseProfileByKey("balanced")
	evidence := inferenceBenchmarkEvidence{Summary: map[string]any{"scenarios": []any{
		map[string]any{
			"success_rate": 1.0, "quality_gate_pass_rate": 1.0,
			"p95_ttft_ms": inferenceReleaseMaxP95TTFTMs + 1,
			"p95_tpot_ms": 50.0, "output_tokens_per_second": 80.0,
		},
	}}}
	candidate := summarizeReleaseCandidate(profile, evidence)
	if candidate.GatePassed {
		t.Fatalf("latency over SLO must block release: %+v", candidate)
	}
}
