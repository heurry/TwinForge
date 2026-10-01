package httpx

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// inferenceRuntimeStatus 返回本机双卡 vLLM workload 的容器与健康状态。
func (a *API) inferenceRuntimeStatus(w http.ResponseWriter, r *http.Request) {
	result, status, err := a.Agent.RequestObject(r.Context(), http.MethodGet, "/api/inference/runtime", nil)
	if err != nil {
		if status == 0 {
			status = http.StatusServiceUnavailable
		}
		WriteError(w, r, status, "inference_runtime_unavailable", err.Error())
		return
	}
	WriteJSON(w, http.StatusOK, result)
}

// startInferenceRuntime 按请求中的 model_id 启动对应 vLLM workload；训练占用 GPU 时拒绝。
func (a *API) startInferenceRuntime(w http.ResponseWriter, r *http.Request) {
	var req map[string]any
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		a.badRequest(w, r, "invalid JSON body")
		return
	}
	operator := a.actor(r, stringValue(req["operator"]))
	configItemID := stringValue(req["config_item_id"])
	configVersion := intValue(req["config_version"])
	configKey := stringValue(req["config_key"])
	configEnv := stringValue(req["config_env"])
	delete(req, "operator")
	delete(req, "config_item_id")
	delete(req, "config_version")
	delete(req, "config_key")
	delete(req, "config_env")
	if trainingID, active, err := a.activeTrainingJob(r.Context()); err != nil {
		a.fail(w, r, err)
		return
	} else if active {
		WriteError(w, r, http.StatusConflict, "gpu_lane_busy", "训练任务 "+trainingID+" 正在占用单机 GPU 实验通道，请先停止训练")
		return
	}
	profile := orDefault(stringValue(req["profile"]), "baseline")
	modelID := orDefault(stringValue(req["model_id"]), "qwen36-27b-fp8")
	version := profile
	if configVersion > 0 {
		version = fmt.Sprintf("config-v%d", configVersion)
	}
	meta := map[string]any{
		"owner": operator, "mode": "inference_runtime_manual", "phase": "starting",
		"runtime_request": req, "config_ref": map[string]any{
			"item_id": configItemID, "version": configVersion, "key": configKey, "env": configEnv,
		},
	}
	deploymentID, err := a.Store.CreateDeploymentMeta(r.Context(), modelID+"-inference-runtime", version, orDefault(configEnv, "dev"), meta)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	a.deployEvent(r.Context(), deploymentID, meta, "starting", "starting inference runtime")
	result, status, err := a.Agent.RequestObject(r.Context(), http.MethodPost, "/api/inference/runtime", req)
	if err != nil {
		meta["error"] = err.Error()
		a.deployEvent(r.Context(), deploymentID, meta, "failed", err.Error())
		_, _ = a.Store.FinishDeployment(r.Context(), deploymentID, "failed")
		a.Store.Audit(r.Context(), operator, "operator", "inference.runtime.start.failed", "deployment", deploymentID,
			map[string]any{"config_item_id": configItemID, "config_version": configVersion, "error": err.Error()})
		if status == 0 {
			status = http.StatusServiceUnavailable
		}
		WriteError(w, r, status, "inference_start_failed", err.Error())
		return
	}
	meta["runtime"] = result
	a.persistDeployMeta(r.Context(), deploymentID, meta)
	// 运行时模型切换后同步服务目录，供压测和路由按实际模型选择 endpoint。
	if model, ok := result["model"].(string); ok && strings.TrimSpace(model) != "" {
		name := model + "-vllm"
		gpuID := gpuDeviceIDFromRuntime(result)
		// 单机只有一个 8020 workload；切换模型时将上一个受控绑定标为不可达，
		// 避免注册中心把历史模型误显示成当前健康实例。
		_, _ = a.Pool.Exec(r.Context(), `UPDATE service_instances SET status='unreachable', last_checked_at=now()
			WHERE metadata->>'managed_by'='inference-runtime' AND name<>$1`, name)
		_, _ = a.Pool.Exec(r.Context(), `INSERT INTO service_instances (name, base_url, model_id, kind, gpu_id, routing_role, status, metadata)
			VALUES ($1,'http://127.0.0.1:8020/v1',$2,'vllm',$3,'model','healthy',$4::jsonb)
			ON CONFLICT (name) DO UPDATE SET base_url=EXCLUDED.base_url, model_id=EXCLUDED.model_id,
				gpu_id=EXCLUDED.gpu_id, status='healthy', metadata=EXCLUDED.metadata`,
			name, model, gpuID, `{"managed_by":"inference-runtime","scope":"internal"}`)
	}
	a.Store.Audit(r.Context(), operator, "operator", "inference.runtime.start", "deployment", deploymentID,
		map[string]any{"profile": profile, "config_item_id": configItemID, "config_version": configVersion})
	if runtimeStatus, _ := result["status"].(string); runtimeStatus == "ready" {
		a.deployEvent(r.Context(), deploymentID, meta, "succeeded", "inference runtime ready")
		_, _ = a.Store.FinishDeployment(r.Context(), deploymentID, "success")
	} else {
		go a.monitorManualInferenceRuntime(deploymentID, operator, meta)
	}
	result["deployment_id"] = deploymentID
	result["config_ref"] = meta["config_ref"]
	WriteJSON(w, http.StatusAccepted, result)
}

func gpuDeviceIDFromRuntime(runtime map[string]any) string {
	config, _ := runtime["config"].(map[string]any)
	values, _ := config["gpu_device_ids"].([]any)
	ids := make([]string, 0, len(values))
	for _, value := range values {
		if id := stringValue(value); id != "" {
			ids = append(ids, id)
		}
	}
	if len(ids) == 0 {
		return "unknown"
	}
	return strings.Join(ids, ",")
}

// stopInferenceRuntime 释放两张 GPU；有压测流量时要求先取消压测，避免半途断流。
func (a *API) stopInferenceRuntime(w http.ResponseWriter, r *http.Request) {
	if benchmarkID, active, err := a.activeBenchmarkRun(r.Context()); err != nil {
		a.fail(w, r, err)
		return
	} else if active {
		WriteError(w, r, http.StatusConflict, "benchmark_active", "推理压测 "+benchmarkID+" 仍在运行，请先停止压测")
		return
	}
	result, status, err := a.Agent.RequestObject(r.Context(), http.MethodDelete, "/api/inference/runtime", nil)
	if err != nil {
		if status == 0 {
			status = http.StatusServiceUnavailable
		}
		WriteError(w, r, status, "inference_stop_failed", err.Error())
		return
	}
	operator := a.actor(r, "")
	rows, _ := a.Pool.Query(r.Context(), `SELECT id FROM deployments
		WHERE metadata->>'mode'='inference_runtime_manual' AND status IN ('running','success')`)
	if rows != nil {
		defer rows.Close()
		for rows.Next() {
			var id string
			if rows.Scan(&id) == nil {
				_, _ = a.Pool.Exec(r.Context(), `UPDATE deployments SET status='rolled_back', finished_at=now(),
					metadata=jsonb_set(metadata,'{phase}','"stopped"'::jsonb) WHERE id=$1::uuid`, id)
			}
		}
	}
	stoppedModel := stringValue(result["model"])
	if stoppedModel == "" { stoppedModel = "inference-runtime" }
	a.Store.Audit(r.Context(), operator, "operator", "inference.runtime.stop", "service_instance", stoppedModel+"-vllm", nil)
	WriteJSON(w, http.StatusOK, result)
}

func (a *API) monitorManualInferenceRuntime(deploymentID, operator string, meta map[string]any) {
	ctx := context.Background()
	deadline := time.Now().Add(10 * time.Minute)
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	for range ticker.C {
		var ledgerStatus string
		if err := a.Pool.QueryRow(ctx, `SELECT status FROM deployments WHERE id=$1::uuid`, deploymentID).Scan(&ledgerStatus); err == nil && ledgerStatus == "rolled_back" {
			return
		}
		result, _, err := a.Agent.RequestObject(ctx, http.MethodGet, "/api/inference/runtime", nil)
		if err == nil {
			meta["runtime"] = result
			a.persistDeployMeta(ctx, deploymentID, meta)
			status, _ := result["status"].(string)
			switch status {
			case "ready":
				a.deployEvent(ctx, deploymentID, meta, "succeeded", "inference runtime ready")
				_, _ = a.Store.FinishDeployment(ctx, deploymentID, "success")
				a.Store.Audit(ctx, operator, "operator", "inference.runtime.ready", "deployment", deploymentID, nil)
				return
			case "error", "stopped":
				message := "inference runtime entered " + status
				if logs, _, logErr := a.Agent.RequestObject(ctx, http.MethodGet, "/api/inference/runtime/logs", nil); logErr == nil {
					if excerpt := runtimeFailureExcerpt(logs); excerpt != "" {
						message += ": " + excerpt
						meta["runtime_error_excerpt"] = excerpt
					}
					if signatures, ok := logs["signatures"]; ok {
						meta["runtime_log_signatures"] = signatures
					}
				}
				a.deployEvent(ctx, deploymentID, meta, "failed", message)
				_, _ = a.Store.FinishDeployment(ctx, deploymentID, "failed")
				return
			}
		}
		if time.Now().After(deadline) {
			a.deployEvent(ctx, deploymentID, meta, "failed", "inference runtime readiness timed out")
			_, _ = a.Store.FinishDeployment(ctx, deploymentID, "failed")
			return
		}
	}
}

func stringValue(value any) string {
	result, _ := value.(string)
	return strings.TrimSpace(result)
}

func intValue(value any) int {
	switch typed := value.(type) {
	case float64:
		return int(typed)
	case int:
		return typed
	default:
		return 0
	}
}
