package httpx

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"
)

// C2：分层存储生命周期。把 PG 中过期的高频数据（metrics / traces / logs / audit）序列化为 NDJSON
// 归档到 MinIO 对象层，并登记 archive_manifests（使冷数据可回溯），随后从 PG 删除——PG 体积下降、
// 冷数据下沉。保留期由配置中心 storage.retention 驱动（平台用自己的配置中心管自己）。
//
// 安全：① 仅归档「严格早于 cutoff」的行（created_at≈now，活数据不受影响）；② 先上传对象、后删 PG，
// 删除 + 写清单在同一事务；③ 对象存储不可用时跳过（绝不在无对象层时删 PG）。

const archiveBatchLimit = 100000 // 单表单次归档行数上限，防一次性吃满内存

// archiveTarget 是一张可归档表（按时间列 + 默认保留天数）。
type archiveTarget struct {
	table       string
	tsColumn    string
	defaultDays int
}

var archiveTargets = []archiveTarget{
	{table: "metrics_samples", tsColumn: "created_at", defaultDays: 7},
	{table: "request_traces", tsColumn: "created_at", defaultDays: 7},
	{table: "platform_logs", tsColumn: "timestamp", defaultDays: 14},
	{table: "audit_events", tsColumn: "created_at", defaultDays: 30},
}

// archiveResult 是一张表一次归档的结果（供 API 返回 / 日志）。
type archiveResult struct {
	Table         string `json:"table"`
	RowCount      int    `json:"row_count"`
	Bytes         int64  `json:"bytes"`
	ObjectKey     string `json:"object_key,omitempty"`
	RetentionDays int    `json:"retention_days"`
	Skipped       string `json:"skipped,omitempty"`
}

// retentionDays 读配置中心 storage.retention（JSON: {"<table>_days": N}）合并内置默认。
func (a *API) retentionDays(ctx context.Context) map[string]int {
	out := map[string]int{}
	for _, t := range archiveTargets {
		out[t.table] = t.defaultDays
	}
	var content *string
	err := a.Pool.QueryRow(ctx, `
		SELECT cv.content FROM config_items ci
		  JOIN config_versions cv ON cv.config_item_id = ci.id AND cv.version = ci.active_version
		 WHERE ci.config_key = 'storage.retention'
		 ORDER BY ci.updated_at DESC LIMIT 1`).Scan(&content)
	if err != nil || content == nil || *content == "" {
		return out // 未配置 → 用默认
	}
	var cfg map[string]int
	if json.Unmarshal([]byte(*content), &cfg) == nil {
		for _, t := range archiveTargets {
			if d, ok := cfg[t.table+"_days"]; ok && d > 0 {
				out[t.table] = d
			}
		}
	}
	return out
}

// runArchiveSweep 归档所有目标表，返回逐表结果。
func (a *API) runArchiveSweep(ctx context.Context, operator string) ([]archiveResult, error) {
	retention := a.retentionDays(ctx)
	results := make([]archiveResult, 0, len(archiveTargets))
	for _, t := range archiveTargets {
		res, err := a.archiveTable(ctx, t, retention[t.table], operator)
		if err != nil {
			slog.Warn("archive table failed", "table", t.table, "err", err)
			res = archiveResult{Table: t.table, RetentionDays: retention[t.table], Skipped: err.Error()}
		}
		results = append(results, res)
	}
	return results, nil
}

// archiveTable 归档单表早于 cutoff 的行：读 → 上传 NDJSON 到 MinIO → 事务内删 PG + 写清单。
func (a *API) archiveTable(ctx context.Context, t archiveTarget, days int, operator string) (archiveResult, error) {
	res := archiveResult{Table: t.table, RetentionDays: days}
	if a.Blob == nil || !a.Blob.Enabled() {
		res.Skipped = "object storage disabled"
		return res, nil
	}
	if days <= 0 {
		res.Skipped = "retention disabled (days<=0)"
		return res, nil
	}
	cutoff := time.Now().Add(-time.Duration(days) * 24 * time.Hour)

	// 1) 读出过期行（id 文本 + 时间 + 整行 jsonb）。
	rows, err := a.Pool.Query(ctx, fmt.Sprintf(
		`SELECT id::text, %s, to_jsonb(t) FROM %s t WHERE %s < $1 ORDER BY %s LIMIT %d`,
		t.tsColumn, t.table, t.tsColumn, t.tsColumn, archiveBatchLimit), cutoff)
	if err != nil {
		return res, err
	}
	var (
		buf          bytes.Buffer
		count        int
		minTS, maxTS time.Time
	)
	for rows.Next() {
		var id string
		var ts time.Time
		var raw []byte
		if err := rows.Scan(&id, &ts, &raw); err != nil {
			rows.Close()
			return res, err
		}
		if count == 0 || ts.Before(minTS) {
			minTS = ts
		}
		if ts.After(maxTS) {
			maxTS = ts
		}
		buf.Write(raw)
		buf.WriteByte('\n')
		count++
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return res, err
	}
	if count == 0 {
		return res, nil // 无过期数据
	}

	// 2) 上传对象（先于删 PG，保证不丢数据）。
	key := fmt.Sprintf("archive/%s/%s/%d.ndjson", t.table, maxTS.UTC().Format("20060102"), time.Now().UnixNano())
	storedKey, err := a.Blob.Put(ctx, key, bytes.NewReader(buf.Bytes()), int64(buf.Len()), "application/x-ndjson")
	if err != nil {
		return res, fmt.Errorf("upload archive: %w", err)
	}

	// 3) 事务内：删 PG 过期行 + 写清单（原子）。
	tx, err := a.Pool.Begin(ctx)
	if err != nil {
		return res, err
	}
	defer tx.Rollback(ctx)
	if _, err := tx.Exec(ctx, fmt.Sprintf(`DELETE FROM %s WHERE %s < $1`, t.table, t.tsColumn), cutoff); err != nil {
		return res, err
	}
	if _, err := tx.Exec(ctx,
		`INSERT INTO archive_manifests (source_table, object_key, row_count, bytes, min_ts, max_ts, archived_by)
		 VALUES ($1,$2,$3,$4,$5,$6,$7)`,
		t.table, storedKey, count, int64(buf.Len()), minTS, maxTS, operator); err != nil {
		return res, err
	}
	if err := tx.Commit(ctx); err != nil {
		return res, err
	}

	res.RowCount = count
	res.Bytes = int64(buf.Len())
	res.ObjectKey = storedKey
	a.Store.Audit(ctx, operator, "system", "storage.archive", "table", t.table,
		map[string]any{"rows": count, "bytes": buf.Len(), "object_key": storedKey, "cutoff": cutoff.UTC().Format(time.RFC3339)})
	return res, nil
}

// RunArchiveLoop 周期自动归档（main 在 STORAGE_ARCHIVE_ENABLED 时启动）。
func (a *API) RunArchiveLoop(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if results, _ := a.runArchiveSweep(ctx, "system"); len(results) > 0 {
				for _, r := range results {
					if r.RowCount > 0 {
						slog.Info("archived", "table", r.Table, "rows", r.RowCount, "bytes", r.Bytes, "key", r.ObjectKey)
					}
				}
			}
		}
	}
}
