package httpx

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"github.com/go-chi/chi/v5"
	"github.com/heurry/cloudnative-infra-platform/server/internal/store"
)

const defaultOperator = "platform-admin"

// decodeBody 解析可选 JSON body 到 v；空 body 不报错（对齐 Java @RequestBody(required=false)）。
func decodeBody(r *http.Request, v any) error {
	if r.Body == nil {
		return nil
	}
	if err := json.NewDecoder(r.Body).Decode(v); err != nil && !errors.Is(err, io.EOF) {
		return err
	}
	return nil
}

func orDefault(v, def string) string {
	if v == "" {
		return def
	}
	return v
}

// ===== 配置中心 /api/config =====

func (a *API) configItems(w http.ResponseWriter, r *http.Request) {
	rows, err := a.Store.ConfigItems(r.Context())
	if err != nil {
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{"items": mapSlice(rows, toConfigItemDTO)})
}

func (a *API) configVersions(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	active, versions, err := a.Store.ConfigVersions(r.Context(), id)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{
		"active_version": active,
		"versions":       mapSlice(versions, toConfigVersionDTO),
	})
}

type createConfigReq struct {
	ConfigKey    string `json:"config_key"`
	Env          string `json:"env"`
	Namespace    string `json:"namespace"`
	ConfigType   string `json:"config_type"`
	Content      string `json:"content"`
	ChangeReason string `json:"change_reason"`
	Operator     string `json:"operator"`
}

func (a *API) createConfigItem(w http.ResponseWriter, r *http.Request) {
	var req createConfigReq
	if err := decodeBody(r, &req); err != nil {
		a.badRequest(w, r, "invalid body")
		return
	}
	if req.ConfigKey == "" {
		a.badRequest(w, r, "config_key required")
		return
	}
	operator := a.actor(r, req.Operator)
	env := orDefault(req.Env, "dev")
	ns := orDefault(req.Namespace, "default")
	id, err := a.Store.CreateConfigItem(r.Context(), env, ns, req.ConfigKey,
		orDefault(req.ConfigType, "yaml"), req.Content, orDefault(req.ChangeReason, "初始化"), operator)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{"id": id, "config_key": req.ConfigKey, "active_version": 1})
}

type publishConfigReq struct {
	Content      string `json:"content"`
	ChangeReason string `json:"change_reason"`
	Operator     string `json:"operator"`
}

func (a *API) publishConfigVersion(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	var req publishConfigReq
	if err := decodeBody(r, &req); err != nil {
		a.badRequest(w, r, "invalid body")
		return
	}
	operator := a.actor(r, req.Operator)
	reason := orDefault(req.ChangeReason, "更新配置")
	next, _, err := a.Store.PublishConfigVersion(r.Context(), id, req.Content, reason, operator)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{"id": id, "active_version": next})
}

type rollbackConfigReq struct {
	Version  int32  `json:"version"`
	Operator string `json:"operator"`
}

func (a *API) rollbackConfigVersion(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	var req rollbackConfigReq
	if err := decodeBody(r, &req); err != nil {
		a.badRequest(w, r, "invalid body")
		return
	}
	operator := a.actor(r, req.Operator)
	_, err := a.Store.RollbackConfigVersion(r.Context(), id, req.Version, operator)
	if err != nil {
		if errors.Is(err, store.ErrVersionNotFound) {
			a.badRequest(w, r, "version not found")
			return
		}
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{"id": id, "active_version": req.Version})
}

// ===== 发布流水线 /api/deployments =====

func (a *API) deployments(w http.ResponseWriter, r *http.Request) {
	rows, err := a.Store.Deployments(r.Context())
	if err != nil {
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{"deployments": mapSlice(rows, toDeploymentDTO)})
}

type triggerDeployReq struct {
	Name      string `json:"name"`
	Version   string `json:"version"`
	Env       string `json:"env"`
	Namespace string `json:"namespace"` // A1：真实 rollout 的目标命名空间（与 image 同时给出时启用）
	Image     string `json:"image"`     // A1：新镜像；给出则走真实 K8s rollout，否则记录态
	Operator  string `json:"operator"`
}

func (a *API) triggerDeployment(w http.ResponseWriter, r *http.Request) {
	var req triggerDeployReq
	if err := decodeBody(r, &req); err != nil {
		a.badRequest(w, r, "invalid body")
		return
	}
	if req.Name == "" {
		a.badRequest(w, r, "name required")
		return
	}
	operator := a.actor(r, req.Operator)
	version := orDefault(req.Version, "latest")
	env := orDefault(req.Env, "dev")

	// A1：给出 image 即请求真实 rollout——复用 A2 写权限守卫（未开启/受保护命名空间→403），
	// 后台 patch 镜像 + 轮询滚动发布，失败自动回滚。未给 image 时保持既有「记录态」行为。
	if req.Image != "" {
		if !a.k8sWriteGuard(w, r, req.Namespace, req.Name) {
			return
		}
		meta := map[string]any{
			"owner": operator, "mode": "k8s_rollout",
			"target_namespace": req.Namespace, "target_name": req.Name,
			"image": req.Image, "phase": "queued",
		}
		id, err := a.Store.CreateDeploymentMeta(r.Context(), req.Name, version, env, meta)
		if err != nil {
			a.fail(w, r, err)
			return
		}
		a.Store.Audit(r.Context(), operator, "operator", "deployment.trigger", "deployment", req.Name,
			map[string]any{"version": version, "env": env, "namespace": req.Namespace, "image": req.Image, "mode": "k8s_rollout"})
		go a.runDeploymentRollout(id, rolloutTarget{Namespace: req.Namespace, Name: req.Name, Image: req.Image, Operator: operator}, meta)
		WriteJSON(w, http.StatusOK, map[string]any{"id": id, "name": req.Name, "status": "running", "mode": "k8s_rollout"})
		return
	}

	id, err := a.Store.CreateDeployment(r.Context(), req.Name, version, env, operator)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	a.Store.Audit(r.Context(), operator, "operator", "deployment.trigger", "deployment", req.Name,
		map[string]any{"version": version, "env": env})
	WriteJSON(w, http.StatusOK, map[string]any{"id": id, "name": req.Name, "status": "running"})
}

type finishDeployReq struct {
	Status   string `json:"status"`
	Operator string `json:"operator"`
}

func (a *API) finishDeployment(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	var req finishDeployReq
	if err := decodeBody(r, &req); err != nil {
		a.badRequest(w, r, "invalid body")
		return
	}
	status := orDefault(req.Status, "success")
	if status != "success" && status != "failed" {
		a.badRequest(w, r, "status must be success|failed")
		return
	}
	operator := a.actor(r, req.Operator)
	key, err := a.Store.FinishDeployment(r.Context(), id, status)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	a.Store.Audit(r.Context(), operator, "operator", "deployment.finish", "deployment", key,
		map[string]any{"status": status})
	WriteJSON(w, http.StatusOK, map[string]any{"id": id, "status": status})
}

func (a *API) rollbackDeployment(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	var req struct {
		Operator string `json:"operator"`
	}
	if err := decodeBody(r, &req); err != nil {
		a.badRequest(w, r, "invalid body")
		return
	}
	operator := a.actor(r, req.Operator)

	// A recorded Kubernetes rollout carries enough immutable target metadata to
	// perform a real rollback. Patch the workload back to previous_image and
	// track the rollback as its own rollout instead of only changing DB status.
	var key, version, env *string
	var rawMeta []byte
	if err := a.Pool.QueryRow(r.Context(), `SELECT deployment_key, version, env, metadata
		FROM deployments WHERE id=$1::uuid`, id).Scan(&key, &version, &env, &rawMeta); err != nil {
		a.fail(w, r, err)
		return
	}
	originalMeta := map[string]any{}
	_ = json.Unmarshal(rawMeta, &originalMeta)
	if stringValue(originalMeta["mode"]) == "k8s_rollout" {
		namespace := stringValue(originalMeta["target_namespace"])
		name := stringValue(originalMeta["target_name"])
		previousImage := stringValue(originalMeta["previous_image"])
		if namespace == "" || name == "" || previousImage == "" {
			WriteError(w, r, http.StatusConflict, "rollback_target_missing", "deployment does not contain a previous Kubernetes image")
			return
		}
		if !a.k8sWriteGuard(w, r, namespace, name) {
			return
		}
		meta := map[string]any{
			"owner": operator, "mode": "k8s_rollout", "phase": "queued", "rollback_of": id,
			"target_namespace": namespace, "target_name": name, "image": previousImage,
			"rollback_from_image": originalMeta["image"],
		}
		newID, err := a.Store.CreateDeploymentMeta(r.Context(), derefOr(key, ""), derefOr(version, ""), derefOr(env, ""), meta)
		if err != nil {
			a.fail(w, r, err)
			return
		}
		if _, err := a.Pool.Exec(r.Context(), `UPDATE deployments SET status='rolled_back', finished_at=now() WHERE id=$1::uuid`, id); err != nil {
			a.fail(w, r, err)
			return
		}
		a.Store.Audit(r.Context(), operator, "operator", "deployment.rollback", "deployment", derefOr(key, ""),
			map[string]any{"from": id, "to_image": previousImage, "rollback_deployment_id": newID, "mode": "k8s_rollout"})
		go a.runDeploymentRollout(newID, rolloutTarget{Namespace: namespace, Name: name, Image: previousImage, Operator: operator}, meta)
		WriteJSON(w, http.StatusAccepted, map[string]any{"id": newID, "name": derefOr(key, ""), "status": "running", "rollback_of": id, "mode": "k8s_rollout", "image": previousImage})
		return
	}
	newID, rollbackKey, err := a.Store.RollbackDeployment(r.Context(), id, operator)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	a.Store.Audit(r.Context(), operator, "operator", "deployment.rollback", "deployment", rollbackKey,
		map[string]any{"from": id})
	WriteJSON(w, http.StatusOK, map[string]any{"id": newID, "name": rollbackKey, "status": "running", "rollback_of": id})
}

// ===== 故障事件 /api/incidents =====

func (a *API) incidents(w http.ResponseWriter, r *http.Request) {
	rows, err := a.Store.Incidents(r.Context(), r.URL.Query().Get("status"))
	if err != nil {
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{"incidents": mapSlice(rows, toIncidentDTO)})
}

func (a *API) incidentDetail(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	inc, events, err := a.Store.IncidentDetail(r.Context(), id)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{
		"incident": incidentDetailDTO(inc),
		"events":   mapSlice(events, toIncidentEventDTO),
	})
}

type createIncidentReq struct {
	Title    string `json:"title"`
	Severity string `json:"severity"`
	Summary  string `json:"summary"`
	Operator string `json:"operator"`
}

func (a *API) createIncident(w http.ResponseWriter, r *http.Request) {
	var req createIncidentReq
	if err := decodeBody(r, &req); err != nil {
		a.badRequest(w, r, "invalid body")
		return
	}
	if req.Title == "" {
		a.badRequest(w, r, "title required")
		return
	}
	operator := a.actor(r, req.Operator)
	severity := orDefault(req.Severity, "info")
	id, err := a.Store.CreateIncident(r.Context(), req.Title, severity, req.Summary)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	a.Store.Audit(r.Context(), operator, "operator", "incident.create", "incident", id,
		map[string]any{"severity": severity, "title": req.Title})
	WriteJSON(w, http.StatusOK, map[string]any{"id": id, "status": "open"})
}

func (a *API) ackIncident(w http.ResponseWriter, r *http.Request) {
	a.incidentTransition(w, r, "acknowledged", "incident.ack", a.Store.AckIncident)
}

func (a *API) resolveIncident(w http.ResponseWriter, r *http.Request) {
	a.incidentTransition(w, r, "resolved", "incident.resolve", a.Store.ResolveIncident)
}

// incidentTransition 复用 ack/resolve 公共流程：解析 operator → 调 store → 审计 → 返回。
func (a *API) incidentTransition(
	w http.ResponseWriter, r *http.Request, newStatus, action string,
	mutate func(ctx context.Context, id, operator string) error,
) {
	id := chi.URLParam(r, "id")
	var req struct {
		Operator string `json:"operator"`
	}
	if err := decodeBody(r, &req); err != nil {
		a.badRequest(w, r, "invalid body")
		return
	}
	operator := a.actor(r, req.Operator)
	if err := mutate(r.Context(), id, operator); err != nil {
		a.fail(w, r, err)
		return
	}
	a.Store.Audit(r.Context(), operator, "operator", action, "incident", id, map[string]any{})
	WriteJSON(w, http.StatusOK, map[string]any{"id": id, "status": newStatus})
}

// ===== 审计 /api/audit =====

func (a *API) auditEvents(w http.ResponseWriter, r *http.Request) {
	limit := clamp(queryInt(r, "limit", 50), 1, 500)
	rows, err := a.Store.AuditEvents(r.Context(), r.URL.Query().Get("resourceType"), int32(limit))
	if err != nil {
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{"events": mapSlice(rows, toAuditEventDTO)})
}

func (a *API) badRequest(w http.ResponseWriter, r *http.Request, msg string) {
	WriteError(w, r, http.StatusBadRequest, "bad_request", msg)
}
