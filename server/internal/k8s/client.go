// Package k8s 让 Go 控制面经 client-go 直读集群级 Kubernetes（Phase 5 / 5B.1）。
//
// 迁移前：控制面把 /api/kubernetes/* 反代给 Agent，由 Agent 持 client-go。
// 迁移后：控制面自身直读（pods/deployments/events/nodes），Agent 退成纯节点采集
// （host/gpu）。连接方式：in-cluster（ServiceAccount）→ 回退 kubeconfig。
// 集群不可达 / 无 kubeconfig 时优雅降级（available=false），绝不编造数据、不阻塞启动。
package k8s

import (
	"context"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

// 快照结构与 Agent / 前端 KubernetesPage 字段逐字对齐（json tag 不可改）。
type PodSnapshot struct {
	Namespace  string   `json:"namespace"`
	Name       string   `json:"name"`
	Phase      string   `json:"phase"`
	Ready      string   `json:"ready"`
	Restarts   int      `json:"restarts"`
	PodIP      string   `json:"pod_ip"`
	Node       string   `json:"node"`
	Component  string   `json:"component"`
	Containers []string `json:"containers"`
}

type DeploymentSnapshot struct {
	Namespace string `json:"namespace"`
	Name      string `json:"name"`
	Ready     string `json:"ready"`
	Available int    `json:"available"`
	Desired   int    `json:"desired"`
	Image     string `json:"image"`
}

type EventSnapshot struct {
	Namespace    string `json:"namespace"`
	ResourceKind string `json:"resource_kind"`
	ResourceName string `json:"resource_name"`
	Reason       string `json:"reason"`
	Message      string `json:"message"`
	Type         string `json:"type"`
	EventTime    string `json:"event_time"`
}

// NodeSnapshot：5B.1 新增（Agent/Java 此前不暴露节点）。
type NodeSnapshot struct {
	Name             string `json:"name"`
	Status           string `json:"status"`
	Roles            string `json:"roles"`
	Version          string `json:"version"`
	InternalIP       string `json:"internal_ip"`
	OSImage          string `json:"os_image"`
	ContainerRuntime string `json:"container_runtime"`
	CPUCapacity      string `json:"cpu_capacity"`
	MemoryBytes      int64  `json:"memory_bytes"`
	GPUCapacity      string `json:"gpu_capacity"`
}

// HPASnapshot：5B.3 HorizontalPodAutoscaler 真实状态（副本/目标阈值/是否在缩放）。
type HPASnapshot struct {
	Namespace       string `json:"namespace"`
	Name            string `json:"name"`
	TargetKind      string `json:"target_kind"`
	TargetName      string `json:"target_name"`
	MinReplicas     int    `json:"min_replicas"`
	MaxReplicas     int    `json:"max_replicas"`
	CurrentReplicas int    `json:"current_replicas"`
	DesiredReplicas int    `json:"desired_replicas"`
	Metric          string `json:"metric"` // 如 "cpu 45%/70%" 或 "cpu <unknown>/70%"
	ScalingActive   bool   `json:"scaling_active"`
}

// Snapshot 是 /api/kubernetes/* 的统一外形（available=false 时各数组为空）。
type Snapshot struct {
	Available   bool                 `json:"available"`
	Disabled    bool                 `json:"disabled"`
	Error       string               `json:"error,omitempty"`
	Pods        []PodSnapshot        `json:"pods"`
	Deployments []DeploymentSnapshot `json:"deployments"`
	Events      []EventSnapshot      `json:"events"`
	Nodes       []NodeSnapshot       `json:"nodes"`
	Hpas        []HPASnapshot        `json:"hpas"`
	// A2：写能力暴露给前端，用于灰显/启用扩缩容控件（由 handler 据 config 填充）。
	WritesEnabled   bool     `json:"writes_enabled"`
	WriteNamespaces []string `json:"write_namespaces"`
}

type Collector struct{ clientset *kubernetes.Clientset }

// NewCollector 按 in-cluster → kubeconfig 顺序建立连接（与 Agent 一致）。
func NewCollector(kubeconfigPath string) (*Collector, error) {
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
	cfg.Timeout = 8 * time.Second
	clientset, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		return nil, fmt.Errorf("build clientset: %w", err)
	}
	return &Collector{clientset: clientset}, nil
}

// CollectSnapshot 按需采集（任一块出错只记 Error、不影响其他块）。
func (c *Collector) CollectSnapshot(ctx context.Context, wantPods, wantDeploys, wantEvents, wantNodes, wantHPAs bool) Snapshot {
	resp := Snapshot{
		Available:   true,
		Pods:        []PodSnapshot{},
		Deployments: []DeploymentSnapshot{},
		Events:      []EventSnapshot{},
		Nodes:       []NodeSnapshot{},
		Hpas:        []HPASnapshot{},
	}
	var firstErr error
	if wantPods {
		if pods, err := c.collectPods(ctx); err != nil {
			firstErr = err
		} else {
			resp.Pods = pods
		}
	}
	if wantDeploys {
		if deps, err := c.collectDeployments(ctx); err != nil {
			firstErr = err
		} else {
			resp.Deployments = deps
		}
	}
	if wantEvents {
		if events, err := c.collectEvents(ctx, 50); err != nil {
			firstErr = err
		} else {
			resp.Events = events
		}
	}
	if wantNodes {
		if nodes, err := c.collectNodes(ctx); err != nil {
			firstErr = err
		} else {
			resp.Nodes = nodes
		}
	}
	if wantHPAs {
		if hpas, err := c.collectHPAs(ctx); err != nil {
			firstErr = err
		} else {
			resp.Hpas = hpas
		}
	}
	if firstErr != nil {
		resp.Available = false
		resp.Error = firstErr.Error()
	}
	return resp
}

// Pods 导出 collectPods，供 serving 抓取器筛选 vLLM Pod。
func (c *Collector) Pods(ctx context.Context) ([]PodSnapshot, error) {
	return c.collectPods(ctx)
}

// ProxyGetPodMetrics 经 kube-apiserver 的 pod proxy 抓取某 Pod 的 /metrics（5B.3 拓展/Option A）。
// 走 API server 代理，无需直连 Pod CIDR——控制面在集群内或经 API 都能用。
func (c *Collector) ProxyGetPodMetrics(ctx context.Context, namespace, name, port string) ([]byte, error) {
	return c.clientset.CoreV1().RESTClient().Get().
		Namespace(namespace).
		Resource("pods").
		Name(name + ":" + port).
		SubResource("proxy").
		Suffix("metrics").
		DoRaw(ctx)
}

type PodLogOptions struct {
	TailLines  int64
	Container  string
	SinceTime  *time.Time
	Previous   bool
	Timestamps bool
}

// PodLogs 返回某 Pod 末 tailLines 行日志（兼容训练任务等旧调用）。
func (c *Collector) PodLogs(ctx context.Context, namespace, name string, tailLines int64) (string, error) {
	return c.PodLogsWithOptions(ctx, namespace, name, PodLogOptions{TailLines: tailLines})
}

// PodLogsWithOptions 支持多容器、时间窗口、时间戳和上一个容器实例。
func (c *Collector) PodLogsWithOptions(ctx context.Context, namespace, name string, options PodLogOptions) (string, error) {
	tailLines := options.TailLines
	if tailLines <= 0 {
		tailLines = 200
	}
	var sinceTime *metav1.Time
	if options.SinceTime != nil {
		value := metav1.NewTime(*options.SinceTime)
		sinceTime = &value
	}
	stream, err := c.clientset.CoreV1().Pods(namespace).
		GetLogs(name, &corev1.PodLogOptions{
			TailLines: &tailLines, Container: options.Container, SinceTime: sinceTime,
			Previous: options.Previous, Timestamps: options.Timestamps,
		}).Stream(ctx)
	if err != nil {
		return "", err
	}
	defer stream.Close()
	b, err := io.ReadAll(stream)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func (c *Collector) collectPods(ctx context.Context) ([]PodSnapshot, error) {
	list, err := c.clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	pods := make([]PodSnapshot, 0, len(list.Items))
	for i := range list.Items {
		p := &list.Items[i]
		containers := make([]string, 0, len(p.Spec.Containers))
		for _, container := range p.Spec.Containers {
			containers = append(containers, container.Name)
		}
		pods = append(pods, PodSnapshot{
			Namespace:  p.Namespace,
			Name:       p.Name,
			Phase:      string(p.Status.Phase),
			Ready:      readyString(p),
			Restarts:   totalRestarts(p),
			PodIP:      p.Status.PodIP,
			Node:       p.Spec.NodeName,
			Component:  guessComponent(p),
			Containers: containers,
		})
	}
	sort.Slice(pods, func(i, j int) bool {
		if pods[i].Namespace != pods[j].Namespace {
			return pods[i].Namespace < pods[j].Namespace
		}
		return pods[i].Name < pods[j].Name
	})
	return pods, nil
}

func (c *Collector) collectDeployments(ctx context.Context) ([]DeploymentSnapshot, error) {
	list, err := c.clientset.AppsV1().Deployments("").List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	out := make([]DeploymentSnapshot, 0, len(list.Items))
	for i := range list.Items {
		d := &list.Items[i]
		out = append(out, DeploymentSnapshot{
			Namespace: d.Namespace,
			Name:      d.Name,
			Ready:     fmt.Sprintf("%d/%d", d.Status.ReadyReplicas, d.Status.Replicas),
			Available: int(d.Status.AvailableReplicas),
			Desired:   int(d.Status.Replicas),
			Image:     primaryImage(d.Spec.Template.Spec.Containers),
		})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Namespace != out[j].Namespace {
			return out[i].Namespace < out[j].Namespace
		}
		return out[i].Name < out[j].Name
	})
	return out, nil
}

func (c *Collector) collectEvents(ctx context.Context, limit int) ([]EventSnapshot, error) {
	list, err := c.clientset.CoreV1().Events("").List(ctx, metav1.ListOptions{Limit: int64(limit)})
	if err != nil {
		return nil, err
	}
	out := make([]EventSnapshot, 0, len(list.Items))
	for i := range list.Items {
		e := &list.Items[i]
		out = append(out, EventSnapshot{
			Namespace:    e.Namespace,
			ResourceKind: e.InvolvedObject.Kind,
			ResourceName: e.InvolvedObject.Name,
			Reason:       e.Reason,
			Message:      e.Message,
			Type:         e.Type,
			EventTime:    eventTimestamp(e),
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].EventTime > out[j].EventTime })
	return out, nil
}

func (c *Collector) collectNodes(ctx context.Context) ([]NodeSnapshot, error) {
	list, err := c.clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	out := make([]NodeSnapshot, 0, len(list.Items))
	for i := range list.Items {
		n := &list.Items[i]
		cap := n.Status.Capacity
		out = append(out, NodeSnapshot{
			Name:             n.Name,
			Status:           nodeStatus(n),
			Roles:            nodeRoles(n),
			Version:          n.Status.NodeInfo.KubeletVersion,
			InternalIP:       nodeInternalIP(n),
			OSImage:          n.Status.NodeInfo.OSImage,
			ContainerRuntime: n.Status.NodeInfo.ContainerRuntimeVersion,
			CPUCapacity:      cap.Cpu().String(),
			MemoryBytes:      cap.Memory().Value(),
			GPUCapacity:      gpuCapacity(cap),
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

func (c *Collector) collectHPAs(ctx context.Context) ([]HPASnapshot, error) {
	list, err := c.clientset.AutoscalingV2().HorizontalPodAutoscalers("").List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	out := make([]HPASnapshot, 0, len(list.Items))
	for i := range list.Items {
		out = append(out, hpaToSnapshot(&list.Items[i]))
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Namespace != out[j].Namespace {
			return out[i].Namespace < out[j].Namespace
		}
		return out[i].Name < out[j].Name
	})
	return out, nil
}

// hpaToSnapshot 把一个 HPA 资源映射为对外 HPASnapshot（collectHPAs / UpsertHPA 共用）。
func hpaToSnapshot(h *autoscalingv2.HorizontalPodAutoscaler) HPASnapshot {
	minReplicas := int32(1)
	if h.Spec.MinReplicas != nil {
		minReplicas = *h.Spec.MinReplicas
	}
	return HPASnapshot{
		Namespace:       h.Namespace,
		Name:            h.Name,
		TargetKind:      h.Spec.ScaleTargetRef.Kind,
		TargetName:      h.Spec.ScaleTargetRef.Name,
		MinReplicas:     int(minReplicas),
		MaxReplicas:     int(h.Spec.MaxReplicas),
		CurrentReplicas: int(h.Status.CurrentReplicas),
		DesiredReplicas: int(h.Status.DesiredReplicas),
		Metric:          hpaMetric(h),
		ScalingActive:   hpaScalingActive(h),
	}
}

// ---- A2：写操作（弹性扩缩容真配）。调用方负责 feature flag + 命名空间守卫。 ----

// ScaleDeployment 经 scale 子资源设置 Deployment 副本数，返回写入后的期望副本。
func (c *Collector) ScaleDeployment(ctx context.Context, namespace, name string, replicas int32) (int32, error) {
	scale, err := c.clientset.AppsV1().Deployments(namespace).GetScale(ctx, name, metav1.GetOptions{})
	if err != nil {
		return 0, err
	}
	scale.Spec.Replicas = replicas
	updated, err := c.clientset.AppsV1().Deployments(namespace).UpdateScale(ctx, name, scale, metav1.UpdateOptions{})
	if err != nil {
		return 0, err
	}
	return updated.Spec.Replicas, nil
}

// HPASpec 是配置 HPA 的入参（按 CPU 平均利用率水平伸缩 Deployment）。
type HPASpec struct {
	Namespace     string
	Name          string
	TargetName    string // 被伸缩的 Deployment 名
	MinReplicas   int32
	MaxReplicas   int32
	TargetCPUUtil int32 // 目标 CPU 平均利用率（%）
}

// UpsertHPA 创建或更新一个 autoscaling/v2 HPA，返回写入后的快照（幂等：不存在则建，存在则覆盖 spec）。
func (c *Collector) UpsertHPA(ctx context.Context, spec HPASpec) (HPASnapshot, error) {
	client := c.clientset.AutoscalingV2().HorizontalPodAutoscalers(spec.Namespace)
	minReplicas := spec.MinReplicas
	targetCPU := spec.TargetCPUUtil
	desiredSpec := autoscalingv2.HorizontalPodAutoscalerSpec{
		ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
			APIVersion: "apps/v1",
			Kind:       "Deployment",
			Name:       spec.TargetName,
		},
		MinReplicas: &minReplicas,
		MaxReplicas: spec.MaxReplicas,
		Metrics: []autoscalingv2.MetricSpec{{
			Type: autoscalingv2.ResourceMetricSourceType,
			Resource: &autoscalingv2.ResourceMetricSource{
				Name: corev1.ResourceCPU,
				Target: autoscalingv2.MetricTarget{
					Type:               autoscalingv2.UtilizationMetricType,
					AverageUtilization: &targetCPU,
				},
			},
		}},
	}
	existing, err := client.Get(ctx, spec.Name, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			created, cErr := client.Create(ctx, &autoscalingv2.HorizontalPodAutoscaler{
				ObjectMeta: metav1.ObjectMeta{Name: spec.Name, Namespace: spec.Namespace},
				Spec:       desiredSpec,
			}, metav1.CreateOptions{})
			if cErr != nil {
				return HPASnapshot{}, cErr
			}
			return hpaToSnapshot(created), nil
		}
		return HPASnapshot{}, err
	}
	existing.Spec = desiredSpec
	updated, uErr := client.Update(ctx, existing, metav1.UpdateOptions{})
	if uErr != nil {
		return HPASnapshot{}, uErr
	}
	return hpaToSnapshot(updated), nil
}

// DeleteHPA 删除一个 HPA（不存在视为成功，幂等）。
func (c *Collector) DeleteHPA(ctx context.Context, namespace, name string) error {
	err := c.clientset.AutoscalingV2().HorizontalPodAutoscalers(namespace).Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil && apierrors.IsNotFound(err) {
		return nil
	}
	return err
}

// ---- A1：真实 rollout 执行（改 Deployment 镜像 + 轮询滚动发布状态）。调用方负责守卫。 ----

// PatchDeploymentImage 用 strategic-merge patch 把首个容器镜像改为 image，返回改之前的镜像（供回滚）。
func (c *Collector) PatchDeploymentImage(ctx context.Context, namespace, name, image string) (string, error) {
	dep, err := c.clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	containers := dep.Spec.Template.Spec.Containers
	if len(containers) == 0 {
		return "", fmt.Errorf("deployment %s/%s has no containers", namespace, name)
	}
	prev := containers[0].Image
	// 按容器名定位（strategic-merge 对 list 以 name 为 merge key）。
	patch := fmt.Sprintf(`{"spec":{"template":{"spec":{"containers":[{"name":%q,"image":%q}]}}}}`,
		containers[0].Name, image)
	if _, err := c.clientset.AppsV1().Deployments(namespace).Patch(
		ctx, name, k8stypes.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{}); err != nil {
		return "", err
	}
	return prev, nil
}

// RolloutSnapshot 概括一次滚动发布的进度（对齐 kubectl rollout status 的判定输入）。
type RolloutSnapshot struct {
	Desired   int32
	Updated   int32
	Ready     int32
	Available int32
	Complete  bool   // 已全量就绪且 observedGeneration 追平
	Failed    bool   // Progressing 条件为 ProgressDeadlineExceeded
	Reason    string // Progressing 条件 reason
	Message   string // Progressing 条件 message
}

// RolloutStatus 读取 Deployment 当前滚动发布状态。
func (c *Collector) RolloutStatus(ctx context.Context, namespace, name string) (RolloutSnapshot, error) {
	dep, err := c.clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return RolloutSnapshot{}, err
	}
	desired := int32(1)
	if dep.Spec.Replicas != nil {
		desired = *dep.Spec.Replicas
	}
	s := RolloutSnapshot{
		Desired:   desired,
		Updated:   dep.Status.UpdatedReplicas,
		Ready:     dep.Status.ReadyReplicas,
		Available: dep.Status.AvailableReplicas,
	}
	for _, cond := range dep.Status.Conditions {
		if cond.Type == appsv1.DeploymentProgressing {
			s.Reason = cond.Reason
			s.Message = cond.Message
			if cond.Reason == "ProgressDeadlineExceeded" {
				s.Failed = true
			}
		}
	}
	// kubectl rollout status 判定：观测代追平 + 新副本全部 updated/ready/available。
	if dep.Status.ObservedGeneration >= dep.Generation &&
		s.Updated == desired && s.Ready == desired && s.Available == desired {
		s.Complete = true
	}
	return s, nil
}

// hpaMetric 把首个 metric 的目标与当前值格式化为 "cpu 45%/70%"（当前未知时 <unknown>）。
func hpaMetric(h *autoscalingv2.HorizontalPodAutoscaler) string {
	if len(h.Spec.Metrics) == 0 {
		return "—"
	}
	m := h.Spec.Metrics[0]
	if m.Type != autoscalingv2.ResourceMetricSourceType || m.Resource == nil {
		return string(m.Type)
	}
	name := string(m.Resource.Name)
	target := "—"
	if m.Resource.Target.AverageUtilization != nil {
		target = fmt.Sprintf("%d%%", *m.Resource.Target.AverageUtilization)
	}
	current := "<unknown>"
	for _, cm := range h.Status.CurrentMetrics {
		if cm.Type == autoscalingv2.ResourceMetricSourceType && cm.Resource != nil &&
			string(cm.Resource.Name) == name && cm.Resource.Current.AverageUtilization != nil {
			current = fmt.Sprintf("%d%%", *cm.Resource.Current.AverageUtilization)
		}
	}
	return fmt.Sprintf("%s %s/%s", name, current, target)
}

func hpaScalingActive(h *autoscalingv2.HorizontalPodAutoscaler) bool {
	for _, cond := range h.Status.Conditions {
		if cond.Type == autoscalingv2.ScalingActive {
			return cond.Status == corev1.ConditionTrue
		}
	}
	return false
}

// ---- 辅助 ----

func readyString(p *corev1.Pod) string {
	ready := 0
	for _, cs := range p.Status.ContainerStatuses {
		if cs.Ready {
			ready++
		}
	}
	return fmt.Sprintf("%d/%d", ready, len(p.Status.ContainerStatuses))
}

func totalRestarts(p *corev1.Pod) int {
	total := 0
	for _, cs := range p.Status.ContainerStatuses {
		total += int(cs.RestartCount)
	}
	return total
}

func guessComponent(p *corev1.Pod) string {
	name := strings.ToLower(p.Name)
	switch {
	case strings.Contains(name, "vllm") || strings.Contains(name, "qwen"):
		return "vllm"
	case strings.Contains(p.Namespace, "aibrix"):
		return "aibrix"
	case strings.Contains(p.Namespace, "envoy"):
		return "gateway"
	case strings.Contains(name, "redis"):
		return "redis"
	case strings.Contains(name, "prometheus") || strings.Contains(name, "cadvisor"):
		return "observability"
	case p.Namespace == "kube-system":
		return "system"
	default:
		return p.Namespace
	}
}

func primaryImage(containers []corev1.Container) string {
	if len(containers) == 0 {
		return ""
	}
	return containers[0].Image
}

func eventTimestamp(e *corev1.Event) string {
	if !e.LastTimestamp.IsZero() {
		return e.LastTimestamp.UTC().Format(time.RFC3339)
	}
	if !e.EventTime.IsZero() {
		return e.EventTime.UTC().Format(time.RFC3339)
	}
	return e.CreationTimestamp.UTC().Format(time.RFC3339)
}

func nodeStatus(n *corev1.Node) string {
	for _, c := range n.Status.Conditions {
		if c.Type == corev1.NodeReady {
			if c.Status == corev1.ConditionTrue {
				return "Ready"
			}
			return "NotReady"
		}
	}
	return "Unknown"
}

func nodeRoles(n *corev1.Node) string {
	roles := []string{}
	for label := range n.Labels {
		if strings.HasPrefix(label, "node-role.kubernetes.io/") {
			role := strings.TrimPrefix(label, "node-role.kubernetes.io/")
			if role != "" {
				roles = append(roles, role)
			}
		}
	}
	if len(roles) == 0 {
		return "worker"
	}
	sort.Strings(roles)
	return strings.Join(roles, ",")
}

func nodeInternalIP(n *corev1.Node) string {
	for _, addr := range n.Status.Addresses {
		if addr.Type == corev1.NodeInternalIP {
			return addr.Address
		}
	}
	return ""
}

func gpuCapacity(cap corev1.ResourceList) string {
	if q, ok := cap["nvidia.com/gpu"]; ok {
		return q.String()
	}
	return "0"
}
