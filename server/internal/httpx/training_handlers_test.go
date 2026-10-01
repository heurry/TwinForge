package httpx

import (
	"testing"

	"github.com/heurry/cloudnative-infra-platform/server/internal/k8s"
)

func TestPickTrainingPod(t *testing.T) {
	pods := []k8s.PodSnapshot{
		{Namespace: "training", Name: "lora-001-worker-0"},
		{Namespace: "training", Name: "lora-001-master-0"},
		{Namespace: "other", Name: "lora-001-master-0"},
		{Namespace: "training", Name: "unrelated-pod"},
	}
	// master 优先
	if got := pickTrainingPod(pods, "training", "lora-001"); got != "lora-001-master-0" {
		t.Fatalf("want master pod, got %q", got)
	}
	// 无 master 时取名字最小的 worker
	noMaster := []k8s.PodSnapshot{
		{Namespace: "training", Name: "lora-001-worker-1"},
		{Namespace: "training", Name: "lora-001-worker-0"},
	}
	if got := pickTrainingPod(noMaster, "training", "lora-001"); got != "lora-001-worker-0" {
		t.Fatalf("want worker-0, got %q", got)
	}
	// 无匹配 → 空
	if got := pickTrainingPod(pods, "training", "nonexistent"); got != "" {
		t.Fatalf("want empty, got %q", got)
	}
	// 命名空间隔离：other 命名空间的 master 不被选
	onlyOther := []k8s.PodSnapshot{{Namespace: "other", Name: "lora-001-master-0"}}
	if got := pickTrainingPod(onlyOther, "training", "lora-001"); got != "" {
		t.Fatalf("namespace isolation failed, got %q", got)
	}
}

func TestFilterTrainingKubernetesResources(t *testing.T) {
	pods := []k8s.PodSnapshot{
		{Namespace: "training", Name: "lora-001-master-0"},
		{Namespace: "training", Name: "lora-001-worker-0"},
		{Namespace: "training", Name: "other-master-0"},
		{Namespace: "other", Name: "lora-001-worker-1"},
	}
	if got := filterTrainingPods(pods, "training", "lora-001"); len(got) != 2 {
		t.Fatalf("expected 2 matching pods, got %d", len(got))
	}
	events := []k8s.EventSnapshot{
		{Namespace: "training", ResourceName: "lora-001", Reason: "Created"},
		{Namespace: "training", ResourceName: "lora-001-master-0", Reason: "Scheduled"},
		{Namespace: "training", ResourceName: "other-master-0", Reason: "Pulled"},
		{Namespace: "other", ResourceName: "lora-001", Reason: "Created"},
	}
	if got := filterTrainingEvents(events, "training", "lora-001"); len(got) != 2 {
		t.Fatalf("expected 2 matching events, got %d", len(got))
	}
}

func TestCopyTrainingHyperparams(t *testing.T) {
	env := map[string]string{}
	copyTrainingHyperparams(env, map[string]any{
		"learning_rate":          0.0002,
		"max_samples":            16,
		"gradient_checkpointing": true,
		"ignored":                "value",
	})
	if env["LEARNING_RATE"] != "0.0002" || env["MAX_SAMPLES"] != "16" || env["GRADIENT_CHECKPOINTING"] != "true" {
		t.Fatalf("unexpected training env: %#v", env)
	}
	if _, exists := env["ignored"]; exists {
		t.Fatalf("unknown hyperparameter leaked into environment: %#v", env)
	}
}
