// Package training 让 Go 控制面经 client-go dynamic client 管理 Kubeflow PyTorchJob
// （提交 / 查询 / 列举 / 删除 分布式训练任务）—— Phase F / F1。
//
// 设计与 internal/k8s 一致：连接 in-cluster（ServiceAccount）→ 回退 kubeconfig；
// CRD（kubeflow.org/v1 PyTorchJob）未安装 / 集群不可达时优雅降级——构造失败返回 nil + err，
// 调用失败返回 error，绝不编造数据、不阻塞控制面启动。
//
// 为何用 dynamic client：PyTorchJob 是 CRD，dynamic + unstructured 避免引入 training-operator
// 的大依赖树；本包做一层薄的 typed 映射（JobSpec ↔ unstructured ↔ JobStatus），与 k8s.Collector
// 把资源映射成 *Snapshot 同理。
package training

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

// PyTorchJob（kubeflow.org/v1）的 GroupVersionResource 与 GVK。
var pytorchJobGVR = schema.GroupVersionResource{Group: "kubeflow.org", Version: "v1", Resource: "pytorchjobs"}

const (
	pytorchJobAPIVersion = "kubeflow.org/v1"
	pytorchJobKind       = "PyTorchJob"
	containerName        = "pytorch" // training-operator 约定的容器名
)

// JobSpec 是提交一次分布式训练任务的入参（LoRA 微调等）。
// Master 恒 1 副本；Workers>0 时再加 Worker 角色（单机训练用 Workers=0）。
// 数据集 / 产物路径由调用方经 Env 注入（训练容器从 MinIO 读写）。
type JobSpec struct {
	Namespace      string
	Name           string
	Image          string
	Command        []string
	Args           []string
	Env            map[string]string
	Workers        int32 // Worker 副本数（0 = 仅 Master 单机）
	GPUsPerWorker  int32 // 每副本 GPU 数（Master 同值；0 = 不请求 GPU）
	HostPathMounts []HostPathMount
	EnvFromSecrets []string
}

// HostPathMount 用于本地单节点实验集群读取宿主模型、数据并写训练产物。
// 云上环境应替换为 PVC/对象存储 CSI，控制面 API 不接受用户直接传入此字段。
type HostPathMount struct {
	Name      string
	HostPath  string
	MountPath string
	ReadOnly  bool
}

// ReplicaStatus 是某副本角色（Master/Worker）的运行计数。
type ReplicaStatus struct {
	Active    int32 `json:"active"`
	Succeeded int32 `json:"succeeded"`
	Failed    int32 `json:"failed"`
}

// JobStatus 概括一个 PyTorchJob 的运行状态（对外统一外形）。
type JobStatus struct {
	Namespace       string                   `json:"namespace"`
	Name            string                   `json:"name"`
	Phase           string                   `json:"phase"` // Created|Running|Succeeded|Failed|Restarting|Unknown
	Reason          string                   `json:"reason,omitempty"`
	Message         string                   `json:"message,omitempty"`
	ReplicaStatuses map[string]ReplicaStatus `json:"replica_statuses"`
	StartTime       string                   `json:"start_time,omitempty"`
	CompletionTime  string                   `json:"completion_time,omitempty"`
}

// Client 经 dynamic client 操作 PyTorchJob。
type Client struct{ dyn dynamic.Interface }

// NewClient 按 in-cluster → kubeconfig 顺序建立 dynamic client（与 k8s.NewCollector 一致）。
func NewClient(kubeconfigPath string) (*Client, error) {
	cfg, err := rest.InClusterConfig()
	if err != nil {
		loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
		if kubeconfigPath != "" {
			loadingRules.ExplicitPath = kubeconfigPath
		}
		cfg, err = clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
			loadingRules, &clientcmd.ConfigOverrides{}).ClientConfig()
		if err != nil {
			return nil, fmt.Errorf("load kubeconfig: %w", err)
		}
	}
	cfg.Timeout = 10 * time.Second
	dyn, err := dynamic.NewForConfig(cfg)
	if err != nil {
		return nil, fmt.Errorf("build dynamic client: %w", err)
	}
	return &Client{dyn: dyn}, nil
}

// Available 探测 PyTorchJob CRD 是否可达（operator 已装 + 集群可达）。
// 供 handler/runner 决定是否降级（CRD 未装时 List 报 "no matches for kind"）。
func (c *Client) Available(ctx context.Context) bool {
	_, err := c.dyn.Resource(pytorchJobGVR).Namespace("").List(ctx, metav1.ListOptions{Limit: 1})
	return err == nil
}

// SubmitJob 创建一个 PyTorchJob，返回创建后的状态。调用方负责 feature flag + 命名空间守卫。
func (c *Client) SubmitJob(ctx context.Context, spec JobSpec) (JobStatus, error) {
	created, err := c.dyn.Resource(pytorchJobGVR).Namespace(spec.Namespace).
		Create(ctx, buildPyTorchJob(spec), metav1.CreateOptions{})
	if err != nil {
		return JobStatus{}, err
	}
	return parseJobStatus(created), nil
}

// GetJob 读取单个 PyTorchJob 状态。
func (c *Client) GetJob(ctx context.Context, namespace, name string) (JobStatus, error) {
	got, err := c.dyn.Resource(pytorchJobGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return JobStatus{}, err
	}
	return parseJobStatus(got), nil
}

// ListJobs 列举某命名空间下的 PyTorchJob（namespace 为空 = 全部命名空间）。
func (c *Client) ListJobs(ctx context.Context, namespace string) ([]JobStatus, error) {
	list, err := c.dyn.Resource(pytorchJobGVR).Namespace(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	out := make([]JobStatus, 0, len(list.Items))
	for i := range list.Items {
		out = append(out, parseJobStatus(&list.Items[i]))
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Namespace != out[j].Namespace {
			return out[i].Namespace < out[j].Namespace
		}
		return out[i].Name < out[j].Name
	})
	return out, nil
}

// DeleteJob 删除一个 PyTorchJob（不存在视为成功，幂等）。
func (c *Client) DeleteJob(ctx context.Context, namespace, name string) error {
	err := c.dyn.Resource(pytorchJobGVR).Namespace(namespace).Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil && apierrors.IsNotFound(err) {
		return nil
	}
	return err
}

// buildPyTorchJob 把 JobSpec 组装成 kubeflow.org/v1 PyTorchJob 的 unstructured 对象。
// 注意 unstructured 只接受 JSON 基础类型：string / int64 / bool / []interface{} / map[string]interface{}，
// 资源量（GPU）用字符串表示。
func buildPyTorchJob(spec JobSpec) *unstructured.Unstructured {
	makeReplica := func(replicas int32) map[string]interface{} {
		container := map[string]interface{}{
			"name":            containerName,
			"image":           spec.Image,
			"imagePullPolicy": "IfNotPresent",
		}
		if len(spec.Command) > 0 {
			container["command"] = toIfaceSlice(spec.Command)
		}
		if len(spec.Args) > 0 {
			container["args"] = toIfaceSlice(spec.Args)
		}
		if len(spec.Env) > 0 {
			container["env"] = envList(spec.Env)
		}
		if len(spec.EnvFromSecrets) > 0 {
			envFrom := make([]interface{}, 0, len(spec.EnvFromSecrets))
			for _, name := range spec.EnvFromSecrets {
				envFrom = append(envFrom, map[string]interface{}{"secretRef": map[string]interface{}{"name": name}})
			}
			container["envFrom"] = envFrom
		}
		volumes := make([]interface{}, 0, len(spec.HostPathMounts)+1)
		volumeMounts := make([]interface{}, 0, len(spec.HostPathMounts)+1)
		for _, mount := range spec.HostPathMounts {
			volumes = append(volumes, map[string]interface{}{
				"name":     mount.Name,
				"hostPath": map[string]interface{}{"path": mount.HostPath, "type": "Directory"},
			})
			volumeMounts = append(volumeMounts, map[string]interface{}{
				"name": mount.Name, "mountPath": mount.MountPath, "readOnly": mount.ReadOnly,
			})
		}
		volumes = append(volumes, map[string]interface{}{"name": "dshm", "emptyDir": map[string]interface{}{"medium": "Memory"}})
		volumeMounts = append(volumeMounts, map[string]interface{}{"name": "dshm", "mountPath": "/dev/shm"})
		container["volumeMounts"] = volumeMounts
		if spec.GPUsPerWorker > 0 {
			gpu := strconv.Itoa(int(spec.GPUsPerWorker))
			container["resources"] = map[string]interface{}{
				"limits": map[string]interface{}{"nvidia.com/gpu": gpu},
			}
		}
		podSpec := map[string]interface{}{
			"containers": []interface{}{container},
			"volumes":    volumes,
		}
		return map[string]interface{}{
			"replicas":      int64(replicas),
			"restartPolicy": "OnFailure",
			"template": map[string]interface{}{
				"spec": podSpec,
			},
		}
	}

	replicaSpecs := map[string]interface{}{"Master": makeReplica(1)}
	if spec.Workers > 0 {
		replicaSpecs["Worker"] = makeReplica(spec.Workers)
	}

	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": pytorchJobAPIVersion,
		"kind":       pytorchJobKind,
		"metadata": map[string]interface{}{
			"name":      spec.Name,
			"namespace": spec.Namespace,
		},
		"spec": map[string]interface{}{
			"pytorchReplicaSpecs": replicaSpecs,
		},
	}}
}

// parseJobStatus 从 PyTorchJob unstructured 提取统一状态外形。
// 相位 = status.conditions 中末个 status=True 的条件类型（Created→Running→Succeeded/Failed）。
func parseJobStatus(u *unstructured.Unstructured) JobStatus {
	st := JobStatus{
		Namespace:       u.GetNamespace(),
		Name:            u.GetName(),
		Phase:           "Unknown",
		ReplicaStatuses: map[string]ReplicaStatus{},
	}

	if conds, found, _ := unstructured.NestedSlice(u.Object, "status", "conditions"); found {
		for _, c := range conds {
			cm, ok := c.(map[string]interface{})
			if !ok {
				continue
			}
			if status, _, _ := unstructured.NestedString(cm, "status"); status != "True" {
				continue
			}
			if ctype, _, _ := unstructured.NestedString(cm, "type"); ctype != "" {
				st.Phase = ctype
				st.Reason, _, _ = unstructured.NestedString(cm, "reason")
				st.Message, _, _ = unstructured.NestedString(cm, "message")
			}
		}
	}

	if repl, found, _ := unstructured.NestedMap(u.Object, "status", "replicaStatuses"); found {
		for role, v := range repl {
			rm, ok := v.(map[string]interface{})
			if !ok {
				continue
			}
			st.ReplicaStatuses[role] = ReplicaStatus{
				Active:    nestedInt32(rm, "active"),
				Succeeded: nestedInt32(rm, "succeeded"),
				Failed:    nestedInt32(rm, "failed"),
			}
		}
	}

	st.StartTime, _, _ = unstructured.NestedString(u.Object, "status", "startTime")
	st.CompletionTime, _, _ = unstructured.NestedString(u.Object, "status", "completionTime")
	return st
}

func toIfaceSlice(ss []string) []interface{} {
	out := make([]interface{}, len(ss))
	for i, s := range ss {
		out[i] = s
	}
	return out
}

// envList 把 env map 转成 [{name,value},...]，按 key 排序（稳定、可测）。
func envList(env map[string]string) []interface{} {
	keys := make([]string, 0, len(env))
	for k := range env {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]interface{}, 0, len(env))
	for _, k := range keys {
		out = append(out, map[string]interface{}{"name": k, "value": env[k]})
	}
	return out
}

// nestedInt32 读 unstructured 中的整数字段（兼容 int64 / float64 两种解码来源）。
func nestedInt32(m map[string]interface{}, key string) int32 {
	if v, found, err := unstructured.NestedInt64(m, key); found && err == nil {
		return int32(v)
	}
	if f, ok := m[key].(float64); ok {
		return int32(f)
	}
	return 0
}
