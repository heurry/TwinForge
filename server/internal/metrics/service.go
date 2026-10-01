// Package metrics 从 request_traces / metrics_samples 聚合指标，逐字段复刻 Java
// MetricsService（即 FastAPI src/metrics/store.py），前端 Metrics 契约直接消费。
package metrics

import (
	"context"
	"encoding/json"
	"sort"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

const windowSeconds = 600 // 固定 10min 窗口（与 Java 一致）

type Service struct{ pool *pgxpool.Pool }

func NewService(pool *pgxpool.Pool) *Service { return &Service{pool: pool} }

type traceRow struct {
	pod, endpoint, fallback *string
	status                  string
	total, ttft, generation *float64
	inTok, outTok           *int64
}

// Current 复刻 currentMetrics()：扁平结构。gpu/host/cadvisor/service_instances/
// upstream_metrics/kubernetes 由 handler 追加（需 agent / 额外查询）。
func (s *Service) Current(ctx context.Context) (map[string]any, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT target_pod, endpoint_id, status,
		       total_ms::float8, ttft_ms::float8, generation_ms::float8,
		       input_tokens, output_tokens, metadata->>'fallback_reason'
		  FROM request_traces
		 WHERE created_at >= now() - ($1 * interval '1 second')
		 ORDER BY created_at DESC LIMIT 500`, windowSeconds)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var rs []traceRow
	for rows.Next() {
		var r traceRow
		if err := rows.Scan(&r.pod, &r.endpoint, &r.status, &r.total, &r.ttft, &r.generation, &r.inTok, &r.outTok, &r.fallback); err != nil {
			return nil, err
		}
		rs = append(rs, r)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	n := len(rs)
	latencies := collect(rs, func(r traceRow) *float64 { return r.total })
	ttfts := collect(rs, func(r traceRow) *float64 { return r.ttft })
	gens := collect(rs, func(r traceRow) *float64 { return r.generation })
	var inputTokens, outputTokens int64
	errorCount := 0
	for _, r := range rs {
		if r.inTok != nil {
			inputTokens += *r.inTok
		}
		if r.outTok != nil {
			outputTokens += *r.outTok
		}
		if r.status != "ok" {
			errorCount++
		}
	}

	errRate := 0.0
	if n > 0 {
		errRate = float64(errorCount) / float64(n)
	}

	return map[string]any{
		"window":            "10m",
		"window_seconds":    windowSeconds,
		"request_count":     n,
		"qps":               float64(n) / float64(windowSeconds),
		"error_count":       errorCount,
		"error_rate":        errRate,
		"mean_latency_ms":   Mean(latencies),
		"mean_ttft_ms":      Mean(ttfts),
		"p50_latency_ms":    Pct(latencies, 0.50),
		"p95_latency_ms":    Pct(latencies, 0.95),
		"p99_latency_ms":    Pct(latencies, 0.99),
		"p50_ttft_ms":       Pct(ttfts, 0.50),
		"p95_ttft_ms":       Pct(ttfts, 0.95),
		"p99_ttft_ms":       Pct(ttfts, 0.99),
		"p95_generation_ms": Pct(gens, 0.95),
		"input_tokens":      inputTokens,
		"output_tokens":     outputTokens,
		"total_tokens":      inputTokens + outputTokens,
		"tokens_per_second": float64(outputTokens) / float64(windowSeconds),
		"target_pod_counts": counts(rs, func(r traceRow) string { return orUnknown(r.pod) }),
		"endpoint_counts":   counts(rs, func(r traceRow) string { return orUnknown(r.endpoint) }),
		"status_counts":     counts(rs, func(r traceRow) string { return statusKey(r.status) }),
		"fallback_counts":   counts(rs, func(r traceRow) string { return fallbackKey(r.fallback) }),
		"target_pod_stats":  groupStats(rs, func(r traceRow) string { return orUnknown(r.pod) }),
		"endpoint_stats":    groupStats(rs, func(r traceRow) string { return orUnknown(r.endpoint) }),
	}, nil
}

// History 复刻 /history：读 metrics_samples 最近 limit 条 → {source, created_at, metrics}。
func (s *Service) History(ctx context.Context, limit int) ([]map[string]any, error) {
	return s.HistorySince(ctx, time.Time{}, limit)
}

// HistorySince 按真实时间窗口读取指标。zero since 保持旧调用的“最近 N 条”语义。
func (s *Service) HistorySince(ctx context.Context, since time.Time, limit int) ([]map[string]any, error) {
	rows, err := s.pool.Query(ctx,
		`SELECT source, metrics, created_at
		   FROM metrics_samples
		  WHERE ($1::timestamptz IS NULL OR created_at >= $1)
		  ORDER BY created_at DESC LIMIT $2`, nullableTime(since), limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []map[string]any{}
	for rows.Next() {
		var source *string
		var metricsJSON []byte
		var createdAt time.Time
		if err := rows.Scan(&source, &metricsJSON, &createdAt); err != nil {
			return nil, err
		}
		out = append(out, map[string]any{
			"source":     source,
			"created_at": createdAt.UTC().Format(time.RFC3339Nano),
			"metrics":    jsonObjectOrEmpty(metricsJSON),
		})
	}
	return out, rows.Err()
}

func nullableTime(value time.Time) any {
	if value.IsZero() {
		return nil
	}
	return value
}

// RecentRequests 复刻 recentRequests(limit)。
func (s *Service) RecentRequests(ctx context.Context, limit int) ([]map[string]any, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT request_id, session_id, endpoint_id, target_pod, model_id,
		       retrieval_ms::float8, generation_ms::float8, total_ms::float8, ttft_ms::float8,
		       input_tokens, output_tokens, status, error, metadata->>'fallback_reason', created_at
		  FROM request_traces ORDER BY created_at DESC LIMIT $1`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []map[string]any{}
	for rows.Next() {
		var requestID, status string
		var sessionID, endpoint, pod, modelID, errMsg, fallbackReason *string
		var retrieval, generation, total, ttft *float64
		var inTok, outTok *int64
		var createdAt time.Time
		if err := rows.Scan(&requestID, &sessionID, &endpoint, &pod, &modelID,
			&retrieval, &generation, &total, &ttft, &inTok, &outTok, &status, &errMsg, &fallbackReason, &createdAt); err != nil {
			return nil, err
		}
		out = append(out, map[string]any{
			"request_id": requestID, "session_id": sessionID, "endpoint_id": endpoint,
			"target_pod": pod, "model_id": modelID, "retrieval_ms": retrieval,
			"generation_ms": generation, "total_ms": total, "ttft_ms": ttft,
			"input_tokens": inTok, "output_tokens": outTok, "status": status,
			"error": errMsg, "fallback_reason": fallbackReason, "created_at": createdAt.UTC().Format(time.RFC3339Nano),
		})
	}
	return out, rows.Err()
}

// ServiceInstanceSummary 复刻 metrics 内嵌的 service_instances 摘要。
func (s *Service) ServiceInstanceSummary(ctx context.Context) ([]map[string]any, error) {
	rows, err := s.pool.Query(ctx,
		`SELECT name, kind, base_url, model_id, routing_role, status, last_checked_at FROM service_instances ORDER BY name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []map[string]any{}
	for rows.Next() {
		var name, kind, baseURL string
		var modelID, routingRole *string
		var status string
		var lastChecked *time.Time
		if err := rows.Scan(&name, &kind, &baseURL, &modelID, &routingRole, &status, &lastChecked); err != nil {
			return nil, err
		}
		out = append(out, map[string]any{
			"name": name, "kind": kind, "base_url": baseURL, "model_id": modelID,
			"routing_role": routingRole, "status": status, "last_checked_at": tsOrNil(lastChecked),
		})
	}
	return out, rows.Err()
}

// PersistSample best-effort 落一条 metrics_samples（供 history 趋势）。
func (s *Service) PersistSample(ctx context.Context, source string, metrics map[string]any) {
	b, err := json.Marshal(metrics)
	if err != nil {
		return
	}
	_, _ = s.pool.Exec(ctx, `INSERT INTO metrics_samples(source, metrics) VALUES ($1, $2::jsonb)`, source, string(b))
}

// ---- 聚合辅助 ----

func collect(rs []traceRow, f func(traceRow) *float64) []float64 {
	var out []float64
	for _, r := range rs {
		if v := f(r); v != nil {
			out = append(out, *v)
		}
	}
	return out
}

func counts(rs []traceRow, key func(traceRow) string) map[string]int {
	c := map[string]int{}
	for _, r := range rs {
		c[key(r)]++
	}
	return c
}

func groupStats(rs []traceRow, key func(traceRow) string) []map[string]any {
	groups := map[string][]traceRow{}
	for _, r := range rs {
		k := key(r)
		groups[k] = append(groups[k], r)
	}
	keys := make([]string, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Strings(keys) // 对齐 Java TreeMap 的字典序

	out := []map[string]any{}
	for _, k := range keys {
		items := groups[k]
		lat := collect(items, func(r traceRow) *float64 { return r.total })
		tt := collect(items, func(r traceRow) *float64 { return r.ttft })
		var outTok int64
		errs := 0
		for _, r := range items {
			if r.outTok != nil {
				outTok += *r.outTok
			}
			if r.status != "ok" {
				errs++
			}
		}
		rate := 0.0
		if len(items) > 0 {
			rate = float64(errs) / float64(len(items))
		}
		out = append(out, map[string]any{
			"name": k, "request_count": len(items), "error_count": errs, "error_rate": rate,
			"mean_latency_ms": Mean(lat), "p95_latency_ms": Pct(lat, 0.95),
			"mean_ttft_ms": Mean(tt), "p95_ttft_ms": Pct(tt, 0.95),
			"output_tokens": outTok,
		})
	}
	return out
}

func orUnknown(p *string) string {
	if p == nil || *p == "" {
		return "unknown"
	}
	return *p
}

func statusKey(s string) string {
	if s == "" {
		return "unknown"
	}
	return s
}

func fallbackKey(value *string) string {
	if value == nil || *value == "" {
		return "none"
	}
	return *value
}

func tsOrNil(t *time.Time) any {
	if t == nil {
		return nil
	}
	return t.UTC().Format(time.RFC3339Nano)
}

func jsonObjectOrEmpty(b []byte) any {
	if len(b) == 0 {
		return map[string]any{}
	}
	var v any
	if err := json.Unmarshal(b, &v); err != nil || v == nil {
		return map[string]any{}
	}
	return v
}
