// Go Infra Agent —— 云原生平台的节点侧采集组件。
//
// 职责（02-system-architecture.md §3.3）：
//   - 读取 Kubernetes Pod / Deployment / Service / Event
//   - 采集节点 CPU/内存/磁盘/网络/GPU 摘要（后续）
//   - 向 Go 控制面上报，或提供拉取接口
//   - 提供 /healthz 与 /metrics
//
// 无 kubeconfig / 集群不可达时优雅降级（available=false），不阻塞启动。
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"
)

type agentConfig struct {
	port                    string
	k8sEnabled              bool
	kubeconfig              string
	inferenceControlEnabled bool
	inferenceImage          string
	inferenceModelPath      string
	inferenceCachePath      string
	inferenceContainerName  string
}

func loadConfig() agentConfig {
	return agentConfig{
		port:                    envOr("AGENT_PORT", "8090"),
		k8sEnabled:              envOr("AGENT_K8S_ENABLED", "false") == "true",
		kubeconfig:              os.Getenv("KUBECONFIG"),
		inferenceControlEnabled: envOr("INFERENCE_CONTROL_ENABLED", "false") == "true",
		inferenceImage:          envOr("INFERENCE_VLLM_IMAGE", "local/vllm-openai:qwen35-v0.19.1"),
		inferenceModelPath:      envOr("INFERENCE_MODEL_PATH", "/mnt/nvme-data/models/LLM_model/Qwen3.6-27B-FP8"),
		inferenceCachePath:      envOr("INFERENCE_CACHE_PATH", "/mnt/nvme-data/LLM/llm_train_platform_miniv2/.cache/vllm/managed"),
		inferenceContainerName:  envOr("INFERENCE_CONTAINER_NAME", "twinforge-vllm-managed"),
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// ---- 快照结构（与后端 k8s_resources / 前端 KubernetesPage 字段对齐） ----

type podSnapshot struct {
	Namespace string `json:"namespace"`
	Name      string `json:"name"`
	Phase     string `json:"phase"`
	Ready     string `json:"ready"`
	Restarts  int    `json:"restarts"`
	PodIP     string `json:"pod_ip"`
	Node      string `json:"node"`
	Component string `json:"component"`
}

type deploymentSnapshot struct {
	Namespace string `json:"namespace"`
	Name      string `json:"name"`
	Ready     string `json:"ready"`
	Available int    `json:"available"`
	Desired   int    `json:"desired"`
	Image     string `json:"image"`
}

type eventSnapshot struct {
	Namespace    string `json:"namespace"`
	ResourceKind string `json:"resource_kind"`
	ResourceName string `json:"resource_name"`
	Reason       string `json:"reason"`
	Message      string `json:"message"`
	Type         string `json:"type"`
	EventTime    string `json:"event_time"`
}

type kubernetesResponse struct {
	Available   bool                 `json:"available"`
	Disabled    bool                 `json:"disabled"`
	Error       string               `json:"error,omitempty"`
	Pods        []podSnapshot        `json:"pods"`
	Deployments []deploymentSnapshot `json:"deployments"`
	Events      []eventSnapshot      `json:"events"`
}

// agent 持有可选的 K8s collector（nil 表示未启用或初始化失败）以及
// 始终运行的主机/GPU 采集器（后台周期采样并缓存）。
type agent struct {
	cfg       agentConfig
	collector *kubeCollector
	hostc     *hostCollector
	gpuc      *gpuCollector
	inference *inferenceRuntime
	initErr   string
}

func main() {
	cfg := loadConfig()
	a := &agent{cfg: cfg}

	if cfg.k8sEnabled {
		c, err := newKubeCollector(cfg.kubeconfig)
		if err != nil {
			a.initErr = err.Error()
			log.Printf("kubernetes collector init failed (degraded): %v", err)
		} else {
			a.collector = c
			log.Printf("kubernetes collector initialized")
		}
	}

	// 主机指标始终采集（/proc，无外部依赖）；GPU 视 nvidia-smi 可用性优雅降级。
	a.hostc = newHostCollector()
	a.hostc.start(2 * time.Second)
	a.gpuc = newGPUCollector()
	a.gpuc.start(5 * time.Second)
	if cfg.inferenceControlEnabled {
		runtime, err := newInferenceRuntime(cfg)
		if err != nil {
			log.Printf("inference runtime init failed (degraded): %v", err)
		} else {
			a.inference = runtime
			log.Printf("inference runtime control initialized")
		}
	}

	mux := http.NewServeMux()

	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"status":      "ok",
			"k8s_enabled": cfg.k8sEnabled,
			"k8s_ready":   a.collector != nil,
			"time":        time.Now().UTC().Format(time.RFC3339),
		})
	})

	mux.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		fmt.Fprintf(w, "# HELP infra_agent_up Agent liveness.\n# TYPE infra_agent_up gauge\ninfra_agent_up 1\n")
		fmt.Fprintf(w, "infra_agent_k8s_enabled %d\n", boolToInt(cfg.k8sEnabled))
		fmt.Fprintf(w, "infra_agent_k8s_ready %d\n", boolToInt(a.collector != nil))
		_, hostOK := a.hostc.snapshot()
		_, gpuOK, _ := a.gpuc.snapshot()
		fmt.Fprintf(w, "infra_agent_host_ready %d\n", boolToInt(hostOK))
		fmt.Fprintf(w, "infra_agent_gpu_ready %d\n", boolToInt(gpuOK))
	})

	// 完整集群快照（pods + deployments + events）。
	mux.HandleFunc("/api/kubernetes/snapshot", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, a.snapshot(r.Context(), true, true, true))
	})
	// 单项接口，供后端按需拉取。
	mux.HandleFunc("/api/kubernetes/pods", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, a.snapshot(r.Context(), true, false, false))
	})
	mux.HandleFunc("/api/kubernetes/deployments", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, a.snapshot(r.Context(), false, true, false))
	})
	mux.HandleFunc("/api/kubernetes/events", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, a.snapshot(r.Context(), false, false, true))
	})

	// 主机资源（CPU/内存/磁盘/网络）：返回后台采样缓存。
	mux.HandleFunc("/api/host", func(w http.ResponseWriter, _ *http.Request) {
		hm, ok := a.hostc.snapshot()
		writeJSON(w, http.StatusOK, map[string]any{"available": ok, "host": hm})
	})
	// GPU：nvidia-smi 缓存；无 GPU/驱动时 available=false、gpu 为空。
	mux.HandleFunc("/api/gpu", func(w http.ResponseWriter, _ *http.Request) {
		gpus, ok, errText := a.gpuc.snapshot()
		writeJSON(w, http.StatusOK, map[string]any{"available": ok, "gpu": gpus, "error": errText})
	})
	a.registerInferenceHandlers(mux)

	addr := ":" + cfg.port
	log.Printf("infra-agent listening on %s (k8s_enabled=%v)", addr, cfg.k8sEnabled)
	srv := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	if err := srv.ListenAndServe(); err != nil {
		log.Fatalf("agent server error: %v", err)
	}
}

// snapshot 按需采集；未启用/未就绪/采集出错时返回降级结构（不编造数据）。
func (a *agent) snapshot(ctx context.Context, wantPods, wantDeploys, wantEvents bool) kubernetesResponse {
	resp := kubernetesResponse{
		Pods:        []podSnapshot{},
		Deployments: []deploymentSnapshot{},
		Events:      []eventSnapshot{},
	}
	if !a.cfg.k8sEnabled {
		resp.Disabled = true
		return resp
	}
	if a.collector == nil {
		resp.Error = "kubernetes collector unavailable: " + a.initErr
		return resp
	}

	ctx, cancel := context.WithTimeout(ctx, 8*time.Second)
	defer cancel()

	if wantPods {
		pods, err := a.collector.collectPods(ctx)
		if err != nil {
			resp.Error = "collect pods: " + err.Error()
			return resp
		}
		resp.Pods = pods
	}
	if wantDeploys {
		deps, err := a.collector.collectDeployments(ctx)
		if err != nil {
			resp.Error = "collect deployments: " + err.Error()
			return resp
		}
		resp.Deployments = deps
	}
	if wantEvents {
		events, err := a.collector.collectEvents(ctx, 50)
		if err != nil {
			resp.Error = "collect events: " + err.Error()
			return resp
		}
		resp.Events = events
	}
	resp.Available = true
	return resp
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
