package httpx

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"
	"sigs.k8s.io/yaml"
)

// launchConfigVersion turns an immutable config-center version into an actual
// training or inference launch. The same downstream handlers are reused, so
// resource guards, deployment tracking, audit and AIOps evidence stay unified.
func (a *API) launchConfigVersion(w http.ResponseWriter, r *http.Request) {
	itemID := chi.URLParam(r, "id")
	var req struct {
		Version   int            `json:"version"`
		Kind      string         `json:"kind"`
		Overrides map[string]any `json:"overrides"`
		Operator  string         `json:"operator"`
	}
	if err := decodeBody(r, &req); err != nil {
		a.badRequest(w, r, "invalid body")
		return
	}
	var key, env, content string
	var version int
	err := a.Pool.QueryRow(r.Context(), `SELECT c.config_key, c.env, v.version, COALESCE(v.content,'')
		FROM config_items c JOIN config_versions v ON v.config_item_id=c.id
		WHERE c.id=$1::uuid AND v.version=CASE WHEN $2::int > 0 THEN $2::int ELSE c.active_version END`,
		itemID, req.Version).Scan(&key, &env, &version, &content)
	if err != nil {
		if isNoRows(err) {
			WriteError(w, r, http.StatusNotFound, "config_version_not_found", "config item or version not found")
			return
		}
		a.fail(w, r, err)
		return
	}

	rawJSON, err := yaml.YAMLToJSON([]byte(content))
	if err != nil {
		a.badRequest(w, r, "config content must be valid JSON or YAML: "+err.Error())
		return
	}
	config := map[string]any{}
	if err := json.Unmarshal(rawJSON, &config); err != nil {
		a.badRequest(w, r, "config content must be an object")
		return
	}
	kind := strings.ToLower(strings.TrimSpace(req.Kind))
	if kind == "" {
		kind, _ = config["kind"].(string)
		kind = strings.ToLower(strings.TrimSpace(kind))
	}
	if kind == "" {
		lowerKey := strings.ToLower(key)
		switch {
		case strings.Contains(lowerKey, "train"):
			kind = "training"
		case strings.Contains(lowerKey, "infer"), strings.Contains(lowerKey, "serv"), strings.Contains(lowerKey, "vllm"):
			kind = "inference"
		}
	}
	if spec, ok := config["spec"].(map[string]any); ok {
		config = spec
	}
	delete(config, "kind")
	mergeLaunchConfig(config, req.Overrides)
	config["config_item_id"] = itemID
	config["config_version"] = version
	config["config_key"] = key
	config["config_env"] = env
	config["operator"] = a.actor(r, req.Operator)
	payload, _ := json.Marshal(config)
	forward := r.Clone(r.Context())
	forward.Body = ioNopCloser{Reader: bytes.NewReader(payload)}
	forward.ContentLength = int64(len(payload))

	switch kind {
	case "training", "train", "pytorchjob":
		a.submitTrainingJob(w, forward)
	case "inference", "serving", "vllm":
		a.startInferenceRuntime(w, forward)
	default:
		a.badRequest(w, r, "kind must be training or inference (or use a config_key containing train/infer/serve)")
	}
}

// ioNopCloser avoids importing io only for io.NopCloser while still satisfying
// http.Request.Body.
type ioNopCloser struct{ *bytes.Reader }

func (ioNopCloser) Close() error { return nil }

func mergeLaunchConfig(dst, src map[string]any) {
	for key, value := range src {
		if child, ok := value.(map[string]any); ok {
			if current, ok := dst[key].(map[string]any); ok {
				mergeLaunchConfig(current, child)
				continue
			}
		}
		dst[key] = value
	}
}
