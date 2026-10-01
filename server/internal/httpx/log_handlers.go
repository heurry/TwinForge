package httpx

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/heurry/cloudnative-infra-platform/server/internal/k8s"
)

type platformLogInput struct {
	Level        string         `json:"level"`
	Source       string         `json:"source"`
	ResourceType string         `json:"resource_type"`
	ResourceID   string         `json:"resource_id"`
	Message      string         `json:"message"`
	Attributes   map[string]any `json:"attributes"`
}

func (a *API) recordPlatformLog(ctx context.Context, in platformLogInput) error {
	level := strings.ToLower(strings.TrimSpace(in.Level))
	if level == "" {
		level = "info"
	}
	attributes, _ := json.Marshal(in.Attributes)
	_, err := a.Pool.Exec(ctx, `INSERT INTO platform_logs
		(level, source, resource_type, resource_id, message, attributes)
		VALUES ($1,$2,NULLIF($3,''),NULLIF($4,''),$5,$6::jsonb)`,
		level, in.Source, in.ResourceType, in.ResourceID, in.Message, attributes)
	return err
}

func (a *API) ingestPlatformLog(w http.ResponseWriter, r *http.Request) {
	var in platformLogInput
	if err := decodeBody(r, &in); err != nil || strings.TrimSpace(in.Source) == "" || strings.TrimSpace(in.Message) == "" {
		a.badRequest(w, r, "source and message are required")
		return
	}
	if err := a.recordPlatformLog(r.Context(), in); err != nil {
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusAccepted, map[string]any{"status": "accepted"})
}

func (a *API) platformLogs(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	limit := clamp(queryInt(r, "limit", 200), 1, 1000)

	// Direct pod log query is useful before a collector has indexed the stream.
	if pod := strings.TrimSpace(q.Get("pod")); pod != "" {
		if a.K8s == nil {
			WriteError(w, r, http.StatusServiceUnavailable, "kubernetes_unavailable", "Kubernetes collector is not configured")
			return
		}
		namespace := orDefault(strings.TrimSpace(q.Get("namespace")), "default")
		var sinceTime *time.Time
		if raw := q.Get("since"); raw != "" {
			if parsed, parseErr := time.Parse(time.RFC3339, raw); parseErr == nil {
				sinceTime = &parsed
			}
		}
		container := strings.TrimSpace(q.Get("container"))
		logs, err := a.K8s.PodLogsWithOptions(r.Context(), namespace, pod, k8s.PodLogOptions{
			TailLines: int64(limit), Container: container, SinceTime: sinceTime,
			Previous: q.Get("previous") == "true", Timestamps: true,
		})
		if err != nil {
			WriteError(w, r, http.StatusBadGateway, "pod_logs_failed", err.Error())
			return
		}
		lines := strings.Split(strings.TrimRight(logs, "\n"), "\n")
		out := make([]map[string]any, 0, len(lines))
		for i, line := range lines {
			if strings.TrimSpace(line) == "" {
				continue
			}
			if query := strings.ToLower(strings.TrimSpace(q.Get("q"))); query != "" && !strings.Contains(strings.ToLower(line), query) {
				continue
			}
			message, timestamp := splitKubernetesTimestamp(line)
			out = append(out, map[string]any{
				"id": "pod-" + strconv.Itoa(i), "timestamp": timestamp, "level": detectLogLevel(message),
				"source": "kubernetes", "resource_type": "pod", "resource_id": namespace + "/" + pod,
				"message": message, "attributes": map[string]any{"namespace": namespace, "pod": pod, "container": container},
			})
		}
		WriteJSON(w, http.StatusOK, map[string]any{"logs": out, "source": "kubernetes", "count": len(out)})
		return
	}

	since := time.Now().Add(-24 * time.Hour)
	if raw := q.Get("since"); raw != "" {
		if parsed, err := time.Parse(time.RFC3339, raw); err == nil {
			since = parsed
		}
	}
	// 检索结果短暂 cache，避免刷新面板时反复扫描历史日志；写入仍落 PostgreSQL，缓存失效不影响历史数据。
	cacheKey := "cache:logs:" + fmt.Sprintf("%x", sha256.Sum256([]byte(q.Encode())))
	if a.Cache != nil && a.Cache.Enabled() {
		var cached map[string]any
		if hit, _ := a.Cache.GetJSON(r.Context(), cacheKey, &cached); hit {
			w.Header().Set("X-Cache", "HIT")
			WriteJSON(w, http.StatusOK, cached)
			return
		}
		w.Header().Set("X-Cache", "MISS")
	}
	rows, err := a.Pool.Query(r.Context(), `SELECT id, timestamp, level, source,
		COALESCE(resource_type,''), COALESCE(resource_id,''), message, attributes
		FROM platform_logs
		WHERE timestamp >= $1
		  AND ($2='' OR source=$2)
		  AND ($3='' OR level=$3)
		  AND ($4='' OR resource_type=$4)
		  AND ($5='' OR resource_id=$5)
		  AND ($6='' OR message ILIKE '%%' || $6 || '%%' OR attributes::text ILIKE '%%' || $6 || '%%')
		ORDER BY timestamp DESC LIMIT $7`, since, q.Get("source"), q.Get("level"), q.Get("resource_type"), q.Get("resource_id"), q.Get("q"), limit)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	defer rows.Close()
	out := []map[string]any{}
	for rows.Next() {
		var id int64
		var at time.Time
		var level, source, resourceType, resourceID, message string
		var attributes []byte
		if err := rows.Scan(&id, &at, &level, &source, &resourceType, &resourceID, &message, &attributes); err != nil {
			a.fail(w, r, err)
			return
		}
		out = append(out, map[string]any{
			"id": id, "timestamp": at.UTC().Format(time.RFC3339Nano), "level": level, "source": source,
			"resource_type": resourceType, "resource_id": resourceID, "message": message, "attributes": jsonbObject(attributes),
		})
	}
	if err := rows.Err(); err != nil {
		a.fail(w, r, err)
		return
	}
	payload := map[string]any{"logs": out, "source": "platform", "count": len(out)}
	if a.Cache != nil && a.Cache.Enabled() { _ = a.Cache.SetJSON(r.Context(), cacheKey, payload, 5*time.Second) }
	WriteJSON(w, http.StatusOK, payload)
}

func splitKubernetesTimestamp(line string) (string, any) {
	parts := strings.SplitN(line, " ", 2)
	if len(parts) == 2 {
		if parsed, err := time.Parse(time.RFC3339Nano, parts[0]); err == nil {
			return parts[1], parsed.UTC().Format(time.RFC3339Nano)
		}
	}
	return line, nil
}

func detectLogLevel(line string) string {
	lower := strings.ToLower(line)
	// DNS 等日志中的 NOERROR 表示成功，不能因包含 "error" 被误判。
	lower = strings.ReplaceAll(lower, "noerror", "")
	if strings.Contains(lower, "fatal") || strings.Contains(lower, "panic") || strings.Contains(lower, "error") || strings.Contains(lower, "exception") {
		return "error"
	}
	if strings.Contains(lower, "warn") {
		return "warning"
	}
	return "info"
}
