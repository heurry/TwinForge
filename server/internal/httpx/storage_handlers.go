package httpx

import (
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
)

// C2：分层存储可视化 + 归档操作。
//   GET  /api/storage/tiers      —— 各层占用（Redis 热 / PG 关系 / MinIO 对象）+ 当前保留策略
//   GET  /api/storage/archives   —— 归档清单列表（冷数据可回溯入口）
//   GET  /api/storage/archives/{id} —— 取某归档批次的预签名下载地址
//   POST /api/storage/archive    —— 立即触发一次归档（手动，落审计）

// pgTables 是「关系层」面板展示的关键表（行数 + 物理大小）。
var pgTables = []string{
	"metrics_samples", "audit_events", "benchmark_samples", "request_traces",
	"platform_logs", "models", "service_instances", "kb_chunks", "chat_messages", "training_jobs",
}

func (a *API) storageTiers(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	// 关系层（PG）：逐表行数 + pg_total_relation_size。
	pgRows := []map[string]any{}
	var pgTotalBytes int64
	for _, tbl := range pgTables {
		var rows, bytes int64
		// to_regclass 为空说明表不存在（跨迁移版本健壮）。
		if err := a.Pool.QueryRow(ctx,
			`SELECT COALESCE((SELECT count(*) FROM `+tbl+`),0),
			        COALESCE(pg_total_relation_size(to_regclass('`+tbl+`')),0)`).Scan(&rows, &bytes); err != nil {
			continue
		}
		pgRows = append(pgRows, map[string]any{"table": tbl, "rows": rows, "bytes": bytes})
		pgTotalBytes += bytes
	}

	// 对象层（MinIO）：归档清单聚合。
	var archCount, archRows, archBytes int64
	_ = a.Pool.QueryRow(ctx,
		`SELECT count(*), COALESCE(sum(row_count),0), COALESCE(sum(bytes),0) FROM archive_manifests`).
		Scan(&archCount, &archRows, &archBytes)

	cacheEnabled := a.Cache != nil && a.Cache.Enabled()
	blobEnabled := a.Blob != nil && a.Blob.Enabled()

	WriteJSON(w, http.StatusOK, map[string]any{
		"retention":       a.retentionDays(ctx),
		"archive_enabled": a.StorageArchiveEnabled,
		"tiers": []map[string]any{
			{
				"id": "hot", "label": "Redis 热缓存", "kind": "cache",
				"enabled": cacheEnabled,
				"detail":  ternaryStr(cacheEnabled, "cache-aside / 限流 / 幂等", "未启用（全降级）"),
			},
			{
				"id": "relational", "label": "PostgreSQL 关系层", "kind": "relational",
				"enabled": true, "total_bytes": pgTotalBytes, "tables": pgRows,
			},
			{
				"id": "object", "label": "MinIO 对象层", "kind": "object",
				"enabled": blobEnabled, "manifests": archCount,
				"archived_rows": archRows, "archived_bytes": archBytes,
				"detail": ternaryStr(blobEnabled, "归档冷数据 / 基准报告 / 模型产物", "未启用（无法归档）"),
			},
		},
	})
}

func (a *API) listArchives(w http.ResponseWriter, r *http.Request) {
	limit := clamp(queryInt(r, "limit", 50), 1, 500)
	rows, err := a.Pool.Query(r.Context(),
		`SELECT id, source_table, object_key, row_count, bytes, min_ts, max_ts, archived_by, archived_at
		   FROM archive_manifests ORDER BY archived_at DESC LIMIT $1`, limit)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	defer rows.Close()
	out := []map[string]any{}
	for rows.Next() {
		var id, table, key string
		var rowCount int
		var bytes int64
		var minTS, maxTS, archivedAt *time.Time
		var archivedBy *string
		if err := rows.Scan(&id, &table, &key, &rowCount, &bytes, &minTS, &maxTS, &archivedBy, &archivedAt); err != nil {
			a.fail(w, r, err)
			return
		}
		out = append(out, map[string]any{
			"id": id, "source_table": table, "object_key": key, "row_count": rowCount, "bytes": bytes,
			"min_ts": tsOrNil(minTS), "max_ts": tsOrNil(maxTS),
			"archived_by": archivedBy, "archived_at": tsOrNil(archivedAt),
		})
	}
	if err := rows.Err(); err != nil {
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{"archives": out})
}

func (a *API) getArchive(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	var key string
	if err := a.Pool.QueryRow(r.Context(), `SELECT object_key FROM archive_manifests WHERE id = $1::uuid`, id).Scan(&key); err != nil {
		if isNoRows(err) {
			WriteError(w, r, http.StatusNotFound, "not_found", "archive not found")
			return
		}
		a.fail(w, r, err)
		return
	}
	if a.Blob == nil || !a.Blob.Enabled() {
		WriteJSON(w, http.StatusOK, map[string]any{"key": key, "download_url": nil, "note": "object storage disabled"})
		return
	}
	url, err := a.Blob.PresignedGet(r.Context(), key)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{"key": key, "download_url": url})
}

func (a *API) runArchive(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Operator string `json:"operator"`
	}
	_ = decodeBody(r, &req)
	operator := a.actor(r, req.Operator)
	results, err := a.runArchiveSweep(r.Context(), operator)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{"results": results})
}

func ternaryStr(cond bool, a, b string) string {
	if cond {
		return a
	}
	return b
}
