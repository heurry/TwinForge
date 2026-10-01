package main

import (
	"encoding/binary"
	"testing"
)

func TestNormalizeInferenceStartDefaults(t *testing.T) {
	config, err := normalizeInferenceStart(inferenceStartRequest{Profile: "scheduler"})
	if err != nil {
		t.Fatal(err)
	}
	if config.TensorParallelSize != 2 || config.PipelineParallelSize != 1 || config.MaxNumSeqs != 8 || config.MaxNumBatchedTokens != 4096 {
		t.Fatalf("unexpected scheduler defaults: %+v", config)
	}
	if config.AsyncScheduling == nil || !*config.AsyncScheduling {
		t.Fatalf("async scheduling should be explicit and enabled by default: %+v", config)
	}
}

func TestNormalizeInferenceStartSchedulerOverrides(t *testing.T) {
	disabled := false
	config, err := normalizeInferenceStart(inferenceStartRequest{
		Profile:                  "scheduler",
		TensorParallelSize:       1,
		PipelineParallelSize:     2,
		PipelineLayerPartition:   "34,30",
		DBODecodeTokenThreshold:  4,
		DBOPrefillTokenThreshold: 512,
		MaxNumSeqs:               16,
		MaxNumBatchedTokens:      8192,
		AsyncScheduling:          &disabled,
		MaxNumPartialPrefills:    1,
		MaxLongPartialPrefills:   1,
		SchedulerReserveFullISL:  &disabled,
		Profiling:                true,
		GPUMemoryUtilization:     0.92,
		MaxModelLen:              3072,
		KVCacheDType:             "fp8",
		SpeculativeDecoding:      "ngram",
	})
	if err != nil {
		t.Fatal(err)
	}
	if config.AsyncScheduling == nil || *config.AsyncScheduling {
		t.Fatalf("async scheduling override was not preserved: %+v", config)
	}
	if config.TensorParallelSize != 1 || config.PipelineParallelSize != 2 || config.PipelineLayerPartition != "34,30" || config.EnableDBO || config.DBODecodeTokenThreshold != 4 || config.SchedulerReserveFullISL || !config.Profiling || config.GPUMemoryUtilization != 0.92 || config.MaxModelLen != 3072 || config.KVCacheDType != "fp8" || config.SpeculativeDecoding != "ngram" {
		t.Fatalf("scheduler overrides were not preserved: %+v", config)
	}
}

func TestNormalizeInferenceStartAcceptsCoLocatedMemoryFraction(t *testing.T) {
	config, err := normalizeInferenceStart(inferenceStartRequest{
		Profile:              "scheduler",
		GPUMemoryUtilization: 0.4,
	})
	if err != nil || config.GPUMemoryUtilization != 0.4 {
		t.Fatalf("co-located memory fraction was rejected: %+v, %v", config, err)
	}
}

func TestNormalizeInferenceStartAcceptsQwen38ContextWindow(t *testing.T) {
	config, err := normalizeInferenceStart(inferenceStartRequest{
		Profile:     "scheduler",
		MaxModelLen: 24576,
		MaxNumSeqs:  4,
	})
	if err != nil || config.MaxModelLen != 24576 || config.MaxNumSeqs != 4 {
		t.Fatalf("Qwen3.8 24K context window was rejected: %+v, %v", config, err)
	}
}

func TestSameInferenceConfigIgnoresSelectedDevices(t *testing.T) {
	left := inferenceLaunchConfig{TensorParallelSize: 1, GPUDeviceIDs: []string{"1"}}
	right := inferenceLaunchConfig{TensorParallelSize: 1}
	if !sameInferenceConfig(left, right) {
		t.Fatal("live GPU placement must not change configuration identity")
	}
}

func boolPointer(value bool) *bool { return &value }

func TestNormalizeInferenceStartRejectsDBOWithoutDeepEP(t *testing.T) {
	_, err := normalizeInferenceStart(inferenceStartRequest{
		Profile:   "scheduler",
		EnableDBO: boolPointer(true),
	})
	if err == nil {
		t.Fatal("expected DBO to be rejected by the fixed RTX 3090 runtime")
	}
}

func TestNormalizeInferenceStartRejectsLayerPartitionWithoutPP(t *testing.T) {
	_, err := normalizeInferenceStart(inferenceStartRequest{
		Profile:                "scheduler",
		TensorParallelSize:     2,
		PipelineParallelSize:   1,
		PipelineLayerPartition: "34,30",
	})
	if err == nil {
		t.Fatal("expected a PP layer partition on TP-only topology to be rejected")
	}
}

func TestNormalizeInferenceStartRejectsInvalidParallelTopology(t *testing.T) {
	_, err := normalizeInferenceStart(inferenceStartRequest{
		Profile:              "scheduler",
		TensorParallelSize:   2,
		PipelineParallelSize: 2,
	})
	if err == nil {
		t.Fatal("expected a four-rank topology to be rejected on the fixed dual-GPU runtime")
	}
}

func TestNormalizeInferenceStartRejectsSpeculationWithAsyncScheduling(t *testing.T) {
	_, err := normalizeInferenceStart(inferenceStartRequest{Profile: "scheduler", SpeculativeDecoding: "ngram"})
	if err == nil {
		t.Fatal("expected speculative decoding with async scheduling to be rejected")
	}
}

func TestNormalizeInferenceStartRejectsUnsupportedConcurrentPartialPrefill(t *testing.T) {
	_, err := normalizeInferenceStart(inferenceStartRequest{
		Profile:                "scheduler",
		MaxNumPartialPrefills:  4,
		MaxLongPartialPrefills: 1,
	})
	if err == nil {
		t.Fatal("expected concurrent partial prefill to be rejected for the fixed Qwen3.6 runtime")
	}
}

func TestDecodeDockerLogStream(t *testing.T) {
	payload := []byte("warning line\n")
	frame := make([]byte, 8+len(payload))
	frame[0] = 2
	binary.BigEndian.PutUint32(frame[4:8], uint32(len(payload)))
	copy(frame[8:], payload)
	if got := decodeDockerLogStream(frame); got != string(payload) {
		t.Fatalf("decoded log = %q, want %q", got, payload)
	}
}
