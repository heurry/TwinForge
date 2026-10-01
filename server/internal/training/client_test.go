package training

import (
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func TestBuildPyTorchJob(t *testing.T) {
	obj := buildPyTorchJob(JobSpec{
		Namespace:      "training",
		Name:           "lora-qwen-001",
		Image:          "local/train:dev",
		Command:        []string{"torchrun"},
		Args:           []string{"train.py", "--lora"},
		Env:            map[string]string{"DATASET_URI": "s3://x", "BASE_MODEL": "Qwen3-1.7B"},
		Workers:        2,
		GPUsPerWorker:  1,
		HostPathMounts: []HostPathMount{{Name: "training-data", HostPath: "/mnt/data", MountPath: "/mnt/data"}},
		EnvFromSecrets: []string{"training-artifacts"},
	})

	if got := obj.GetAPIVersion(); got != pytorchJobAPIVersion {
		t.Fatalf("apiVersion: got %q", got)
	}
	if got := obj.GetKind(); got != pytorchJobKind {
		t.Fatalf("kind: got %q", got)
	}
	if obj.GetName() != "lora-qwen-001" || obj.GetNamespace() != "training" {
		t.Fatalf("meta: %s/%s", obj.GetNamespace(), obj.GetName())
	}

	if r, found, err := unstructured.NestedInt64(obj.Object, "spec", "pytorchReplicaSpecs", "Master", "replicas"); err != nil || !found || r != 1 {
		t.Fatalf("master replicas: got=%d found=%v err=%v", r, found, err)
	}
	if r, found, _ := unstructured.NestedInt64(obj.Object, "spec", "pytorchReplicaSpecs", "Worker", "replicas"); !found || r != 2 {
		t.Fatalf("worker replicas: got=%d found=%v", r, found)
	}

	containers, found, _ := unstructured.NestedSlice(obj.Object, "spec", "pytorchReplicaSpecs", "Worker", "template", "spec", "containers")
	if !found || len(containers) != 1 {
		t.Fatalf("want 1 worker container, got %d (found=%v)", len(containers), found)
	}
	c0, ok := containers[0].(map[string]interface{})
	if !ok {
		t.Fatal("container is not a map")
	}
	if gpu, _, _ := unstructured.NestedString(c0, "resources", "limits", "nvidia.com/gpu"); gpu != "1" {
		t.Fatalf("gpu limit: got %q want \"1\"", gpu)
	}
	if env, _, _ := unstructured.NestedSlice(c0, "env"); len(env) != 2 {
		t.Fatalf("env: want 2, got %d", len(env))
	}
	if envFrom, _, _ := unstructured.NestedSlice(c0, "envFrom"); len(envFrom) != 1 {
		t.Fatalf("envFrom: want 1, got %d", len(envFrom))
	}
	if mounts, _, _ := unstructured.NestedSlice(c0, "volumeMounts"); len(mounts) != 2 {
		t.Fatalf("volume mounts: want host data + dshm, got %d", len(mounts))
	}
	if volumes, _, _ := unstructured.NestedSlice(obj.Object, "spec", "pytorchReplicaSpecs", "Worker", "template", "spec", "volumes"); len(volumes) != 2 {
		t.Fatalf("volumes: want host data + dshm, got %d", len(volumes))
	}
	if restart, _, _ := unstructured.NestedString(obj.Object, "spec", "pytorchReplicaSpecs", "Worker", "restartPolicy"); restart != "OnFailure" {
		t.Fatalf("restartPolicy: got %q", restart)
	}
}

func TestBuildPyTorchJobSingleNode(t *testing.T) {
	obj := buildPyTorchJob(JobSpec{Namespace: "training", Name: "single", Image: "img", Workers: 0})
	if _, found, _ := unstructured.NestedMap(obj.Object, "spec", "pytorchReplicaSpecs", "Worker"); found {
		t.Fatal("Workers=0 must not produce a Worker replica spec")
	}
	if _, found, _ := unstructured.NestedMap(obj.Object, "spec", "pytorchReplicaSpecs", "Master"); !found {
		t.Fatal("Master replica spec missing")
	}
	// 未请求 GPU 时不应有 resources.limits
	c, _, _ := unstructured.NestedSlice(obj.Object, "spec", "pytorchReplicaSpecs", "Master", "template", "spec", "containers")
	c0 := c[0].(map[string]interface{})
	if _, found := c0["resources"]; found {
		t.Fatal("GPUsPerWorker=0 must not set container resources")
	}
}

func TestParseJobStatus(t *testing.T) {
	u := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": pytorchJobAPIVersion,
		"kind":       pytorchJobKind,
		"metadata":   map[string]interface{}{"name": "j", "namespace": "training"},
		"status": map[string]interface{}{
			"conditions": []interface{}{
				map[string]interface{}{"type": "Created", "status": "True"},
				map[string]interface{}{"type": "Running", "status": "False"},
				map[string]interface{}{"type": "Succeeded", "status": "True", "reason": "JobSucceeded", "message": "done"},
			},
			"replicaStatuses": map[string]interface{}{
				"Master": map[string]interface{}{"succeeded": int64(1)},
				"Worker": map[string]interface{}{"succeeded": int64(2), "failed": int64(0)},
			},
			"startTime":      "2026-06-09T00:00:00Z",
			"completionTime": "2026-06-09T00:10:00Z",
		},
	}}

	st := parseJobStatus(u)
	if st.Phase != "Succeeded" {
		t.Fatalf("phase: got %q want Succeeded", st.Phase)
	}
	if st.Reason != "JobSucceeded" || st.Message != "done" {
		t.Fatalf("reason/message: %q / %q", st.Reason, st.Message)
	}
	if st.ReplicaStatuses["Worker"].Succeeded != 2 {
		t.Fatalf("worker succeeded: got %d want 2", st.ReplicaStatuses["Worker"].Succeeded)
	}
	if st.StartTime == "" || st.CompletionTime == "" {
		t.Fatalf("times missing: start=%q completion=%q", st.StartTime, st.CompletionTime)
	}
}

func TestParseJobStatusRunning(t *testing.T) {
	u := &unstructured.Unstructured{Object: map[string]interface{}{
		"status": map[string]interface{}{
			"conditions": []interface{}{
				map[string]interface{}{"type": "Created", "status": "True"},
				map[string]interface{}{"type": "Running", "status": "True"},
			},
		},
	}}
	if st := parseJobStatus(u); st.Phase != "Running" {
		t.Fatalf("phase: got %q want Running", st.Phase)
	}
}

func TestParseJobStatusEmpty(t *testing.T) {
	// 无 status（刚创建未被 operator 处理）→ Unknown，不 panic。
	st := parseJobStatus(&unstructured.Unstructured{Object: map[string]interface{}{
		"metadata": map[string]interface{}{"name": "x", "namespace": "training"},
	}})
	if st.Phase != "Unknown" {
		t.Fatalf("phase: got %q want Unknown", st.Phase)
	}
	if st.Name != "x" || st.Namespace != "training" {
		t.Fatalf("meta: %s/%s", st.Namespace, st.Name)
	}
}
