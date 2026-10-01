package httpx

import "testing"

func TestQwen38InferenceModelSpec(t *testing.T) {
	spec, ok := inferenceModelSpecFor("qwen38-27b-fp8")
	if !ok {
		t.Fatal("qwen38-27b-fp8 must be a supported inference model")
	}
	if spec.Endpoint != "qwen38-27b-fp8-vllm" || spec.GPUCount != 2 || spec.TensorParallel != 2 || spec.MaxModelLen != 24576 || spec.MaxNumSeqs != 4 || spec.MaxBatchedTokens != 8192 || spec.GPUMemory != 0.88 {
		t.Fatalf("unexpected Qwen3.8 inference spec: %+v", spec)
	}
}

func TestQwen38BalancedProfileMatchesServingContract(t *testing.T) {
	profile, ok := inferenceReleaseProfileByKeyForModel("balanced", "qwen38-27b-fp8")
	if !ok || profile.MaxNumSeqs != 4 || profile.MaxBatchedTokens != 8192 {
		t.Fatalf("unexpected Qwen3.8 balanced profile: %+v", profile)
	}
	request := profile.RuntimeRequest
	if request["max_model_len"] != 24576 || request["gpu_memory_utilization"] != 0.88 || request["kv_cache_dtype"] != "auto" {
		t.Fatalf("Qwen3.8 serving contract drifted: %+v", request)
	}
}

func TestSupportedInferenceModelsContainQwen38(t *testing.T) {
	for _, spec := range supportedInferenceModelSpecs() {
		if spec.ModelID == "qwen38-27b-fp8" {
			return
		}
	}
	t.Fatal("supported inference models do not contain qwen38-27b-fp8")
}
