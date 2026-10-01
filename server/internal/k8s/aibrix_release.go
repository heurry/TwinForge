package k8s

import (
	"context"
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

// AIBrixModelSpec is the controlled Kubernetes serving contract used by the
// model release center. Image, PVC and command shape are selected by the
// control plane; callers cannot submit arbitrary Kubernetes objects.
type AIBrixModelSpec struct {
	Namespace      string
	DeploymentName string
	// ServiceName must match the served model name. AIBrix creates an
	// HTTPRoute backend reference from model.aibrix.ai/name; using the
	// versioned Deployment name here leaves that reference unresolved.
	ServiceName          string
	ModelID              string
	Version              string
	Track                string
	Image                string
	PVCName              string
	ModelPath            string
	TensorParallel       int
	PipelineParallel     int
	MaxModelLen          int
	MaxNumSeqs           int
	MaxNumBatchedTokens  int
	GPUMemoryUtilization float64
	PrefixCaching        bool
	AsyncScheduling      bool
	KVCacheDType         string
}

// EnsureAIBrixGatewayService creates the stable application-facing NodePort.
// The generated Envoy LoadBalancer service remains owned by AIBrix; this
// additional Service only removes the application's dependency on a local
// kubectl port-forward process.
func (c *Collector) EnsureAIBrixGatewayService(ctx context.Context) error {
	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "twinforge-aibrix-gateway",
			Namespace: "envoy-gateway-system",
			Labels:    map[string]string{"platform.twinforge.io/managed-by": "serving-bootstrap"},
		},
		Spec: corev1.ServiceSpec{
			Type: corev1.ServiceTypeNodePort,
			Selector: map[string]string{
				"app.kubernetes.io/component":                    "proxy",
				"gateway.envoyproxy.io/owning-gateway-name":      "aibrix-eg",
				"gateway.envoyproxy.io/owning-gateway-namespace": "aibrix-system",
			},
			Ports: []corev1.ServicePort{{
				Name: "http", Protocol: corev1.ProtocolTCP, Port: 80,
				TargetPort: intstr.FromInt(10080), NodePort: 30080,
			}},
		},
	}
	services := c.clientset.CoreV1().Services(service.Namespace)
	existing, err := services.Get(ctx, service.Name, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			_, err = services.Create(ctx, service, metav1.CreateOptions{})
		}
		return err
	}
	service.ResourceVersion = existing.ResourceVersion
	service.Spec.ClusterIP = existing.Spec.ClusterIP
	service.Spec.ClusterIPs = existing.Spec.ClusterIPs
	service.Spec.IPFamilies = existing.Spec.IPFamilies
	service.Spec.IPFamilyPolicy = existing.Spec.IPFamilyPolicy
	service.Spec.HealthCheckNodePort = existing.Spec.HealthCheckNodePort
	_, err = services.Update(ctx, service, metav1.UpdateOptions{})
	return err
}

// UpsertAIBrixModelDeployment creates or replaces one versioned vLLM workload
// and its discovery Service. A unique candidate Deployment allows the old
// stable pod and the new candidate pod to coexist during a real canary.
func (c *Collector) UpsertAIBrixModelDeployment(ctx context.Context, spec AIBrixModelSpec) error {
	if spec.ServiceName == "" {
		spec.ServiceName = spec.ModelID
	}
	labels := map[string]string{
		"app":                              spec.DeploymentName,
		"model.aibrix.ai/name":             spec.ModelID,
		"model.aibrix.ai/port":             "8000",
		"model.aibrix.ai/runtime":          "vllm",
		"platform.twinforge.io/managed-by": "model-release",
		"platform.twinforge.io/track":      spec.Track,
		"platform.twinforge.io/version":    spec.Version,
	}
	args := []string{
		spec.ModelPath,
		"--served-model-name", spec.ModelID,
		"--tensor-parallel-size", fmt.Sprint(spec.TensorParallel),
		"--pipeline-parallel-size", fmt.Sprint(spec.PipelineParallel),
		"--dtype", "auto",
		"--max-model-len", fmt.Sprint(spec.MaxModelLen),
		"--gpu-memory-utilization", fmt.Sprintf("%.2f", spec.GPUMemoryUtilization),
		"--max-num-seqs", fmt.Sprint(spec.MaxNumSeqs),
		"--max-num-batched-tokens", fmt.Sprint(spec.MaxNumBatchedTokens),
		"--kv-cache-dtype", spec.KVCacheDType,
		"--enable-chunked-prefill",
		"--language-model-only",
		"--reasoning-parser", "qwen3",
		"--enable-auto-tool-choice",
		"--tool-call-parser", "qwen3_coder",
		"--generation-config", "vllm",
		"--trust-remote-code",
		"--host", "0.0.0.0",
		"--port", "8000",
	}
	if spec.PrefixCaching {
		args = append(args, "--enable-prefix-caching")
	} else {
		args = append(args, "--no-enable-prefix-caching")
	}
	if spec.AsyncScheduling {
		args = append(args, "--async-scheduling")
	} else {
		args = append(args, "--no-async-scheduling")
	}
	one := int32(1)
	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: spec.DeploymentName, Namespace: spec.Namespace, Labels: labels},
		Spec: appsv1.DeploymentSpec{
			Replicas: &one,
			Strategy: appsv1.DeploymentStrategy{Type: appsv1.RecreateDeploymentStrategyType},
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": spec.DeploymentName}},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name: "vllm", Image: spec.Image, ImagePullPolicy: corev1.PullIfNotPresent,
						Command: []string{"vllm", "serve"}, Args: args,
						Ports: []corev1.ContainerPort{{Name: "http", ContainerPort: 8000}},
						Resources: corev1.ResourceRequirements{
							Requests: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse(fmt.Sprint(spec.TensorParallel * spec.PipelineParallel))},
							Limits:   corev1.ResourceList{"nvidia.com/gpu": resource.MustParse(fmt.Sprint(spec.TensorParallel * spec.PipelineParallel))},
						},
						StartupProbe:   &corev1.Probe{ProbeHandler: corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{Path: "/health", Port: intstr.FromInt(8000)}}, FailureThreshold: 90, PeriodSeconds: 10},
						ReadinessProbe: &corev1.Probe{ProbeHandler: corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{Path: "/health", Port: intstr.FromInt(8000)}}, InitialDelaySeconds: 10, PeriodSeconds: 5},
						LivenessProbe:  &corev1.Probe{ProbeHandler: corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{Path: "/health", Port: intstr.FromInt(8000)}}, InitialDelaySeconds: 60, PeriodSeconds: 20},
						VolumeMounts:   []corev1.VolumeMount{{Name: "model-store", MountPath: spec.ModelPath, ReadOnly: true}},
					}},
					Volumes: []corev1.Volume{{Name: "model-store", VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: spec.PVCName, ReadOnly: true}}}},
				},
			},
		},
	}
	deployments := c.clientset.AppsV1().Deployments(spec.Namespace)
	existing, err := deployments.Get(ctx, spec.DeploymentName, metav1.GetOptions{})
	if err != nil {
		if !apierrors.IsNotFound(err) {
			return err
		}
		if _, err = deployments.Create(ctx, deployment, metav1.CreateOptions{}); err != nil {
			return err
		}
	} else {
		deployment.ResourceVersion = existing.ResourceVersion
		if _, err = deployments.Update(ctx, deployment, metav1.UpdateOptions{}); err != nil {
			return err
		}
	}

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: spec.ServiceName, Namespace: spec.Namespace, Labels: labels},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{"app": spec.DeploymentName},
			Ports:    []corev1.ServicePort{{Name: "http", Port: 8000, TargetPort: intstr.FromInt(8000)}},
		},
	}
	services := c.clientset.CoreV1().Services(spec.Namespace)
	existingService, err := services.Get(ctx, spec.ServiceName, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			_, err = services.Create(ctx, service, metav1.CreateOptions{})
		}
		return err
	}
	service.ResourceVersion = existingService.ResourceVersion
	service.Spec.ClusterIP = existingService.Spec.ClusterIP
	service.Spec.ClusterIPs = existingService.Spec.ClusterIPs
	service.Spec.IPFamilies = existingService.Spec.IPFamilies
	service.Spec.IPFamilyPolicy = existingService.Spec.IPFamilyPolicy
	_, err = services.Update(ctx, service, metav1.UpdateOptions{})
	return err
}

// ScaleAIBrixModelDeployment keeps the release object and snapshot available
// for audit/rollback while releasing its GPUs.
func (c *Collector) ScaleAIBrixModelDeployment(ctx context.Context, namespace, name string, replicas int32) error {
	_, err := c.ScaleDeployment(ctx, namespace, name, replicas)
	return err
}

// SetAIBrixModelDeploymentTrack updates only Deployment metadata. Keeping the
// track label out of the Pod template update avoids an unnecessary vLLM
// restart when a warmed candidate is promoted to stable.
func (c *Collector) SetAIBrixModelDeploymentTrack(ctx context.Context, namespace, name, track string) error {
	deployments := c.clientset.AppsV1().Deployments(namespace)
	deployment, err := deployments.Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	if deployment.Labels == nil {
		deployment.Labels = map[string]string{}
	}
	deployment.Labels["platform.twinforge.io/track"] = track
	_, err = deployments.Update(ctx, deployment, metav1.UpdateOptions{})
	return err
}
