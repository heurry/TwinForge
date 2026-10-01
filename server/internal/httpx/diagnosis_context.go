package httpx

import (
	"context"
	"time"
)

func (a *API) recentConfigChanges(ctx context.Context, limit int) any {
	rows, err := a.Pool.Query(ctx, `SELECT c.id, c.config_key, c.env, c.namespace, c.active_version,
		active.content, previous.version, previous.content, active.change_reason, active.operator, active.created_at
		FROM config_items c
		JOIN config_versions active ON active.config_item_id=c.id AND active.version=c.active_version
		LEFT JOIN LATERAL (
			SELECT version, content FROM config_versions
			WHERE config_item_id=c.id AND version < c.active_version ORDER BY version DESC LIMIT 1
		) previous ON true
		ORDER BY active.created_at DESC LIMIT $1`, limit)
	if err != nil {
		return map[string]any{"error": err.Error()}
	}
	defer rows.Close()
	out := []map[string]any{}
	for rows.Next() {
		var id, key, env string
		var namespace, activeContent, previousContent, reason, operator *string
		var activeVersion int
		var previousVersion *int
		var createdAt time.Time
		if err := rows.Scan(&id, &key, &env, &namespace, &activeVersion, &activeContent, &previousVersion, &previousContent, &reason, &operator, &createdAt); err != nil {
			return map[string]any{"error": err.Error()}
		}
		out = append(out, map[string]any{
			"item_id": id, "config_key": key, "env": env, "namespace": namespace,
			"active_version": activeVersion, "active_content": activeContent,
			"previous_version": previousVersion, "previous_content": previousContent,
			"change_reason": reason, "operator": operator, "changed_at": createdAt.UTC().Format(time.RFC3339Nano),
		})
	}
	return out
}

func (a *API) recentDiagnosisLogs(ctx context.Context, limit int) any {
	rows, err := a.Pool.Query(ctx, `SELECT timestamp, level, source, COALESCE(resource_type,''),
		COALESCE(resource_id,''), message FROM platform_logs ORDER BY timestamp DESC LIMIT $1`, limit)
	if err != nil {
		return map[string]any{"error": err.Error()}
	}
	defer rows.Close()
	out := []map[string]any{}
	for rows.Next() {
		var at time.Time
		var level, source, resourceType, resourceID, message string
		if err := rows.Scan(&at, &level, &source, &resourceType, &resourceID, &message); err != nil {
			return map[string]any{"error": err.Error()}
		}
		out = append(out, map[string]any{"timestamp": at.UTC().Format(time.RFC3339Nano), "level": level,
			"source": source, "resource_type": resourceType, "resource_id": resourceID, "message": message})
	}
	return out
}

func (a *API) recentDiagnosisTraces(ctx context.Context, limit int) any {
	rows, err := a.Metrics.RecentRequests(ctx, limit)
	if err != nil {
		return map[string]any{"error": err.Error()}
	}
	return rows
}

func (a *API) diagnosisKnowledge(ctx context.Context, question string, limit int) any {
	if a.AI == nil {
		return map[string]any{"available": false, "error": "AI embed service is not configured"}
	}
	vectors, err := a.AI.Embed(ctx, []string{question}, true)
	if err != nil || len(vectors) == 0 {
		if err != nil {
			return map[string]any{"available": false, "error": err.Error()}
		}
		return map[string]any{"available": false, "error": "empty query embedding"}
	}
	hits, err := a.Store.SearchDocuments(ctx, vectors[0], "", limit)
	if err != nil {
		return map[string]any{"available": false, "error": err.Error()}
	}
	out := make([]map[string]any, 0, len(hits))
	for _, hit := range hits {
		content := hit.Content
		if len(content) > 2000 {
			content = content[:2000]
		}
		out = append(out, map[string]any{"doc_id": hit.DocID, "title": hit.Title, "category": hit.Category,
			"version": hit.Version, "score": hit.Score, "content": content})
	}
	return map[string]any{"available": true, "documents": out}
}
