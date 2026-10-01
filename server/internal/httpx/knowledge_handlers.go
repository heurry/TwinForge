package httpx

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/heurry/cloudnative-infra-platform/server/internal/store"
)

// 6A：knowledge 组绞杀——Go 原生 RAG（pgvector）。复刻 legacy /api/knowledge/* 契约，
// 但语料改为「基准测试日志」（方案 A）：rebuild-index 把 benchmark_runs 摘要灌入 kb_*（经
// ai-service /internal/embed 嵌入），search 走 pgvector 余弦检索。版本/文档 CRUD 走 kb_*。
// 招聘语料已彻底清除（见 chore/purge-recruiting），不再迁入。

const knowledgeVersion = "default"

// GET /api/knowledge/versions
func (a *API) knowledgeVersions(w http.ResponseWriter, r *http.Request) {
	vs, err := a.Store.ListVersions(r.Context())
	if err != nil {
		a.fail(w, r, err)
		return
	}
	out := make([]map[string]any, 0, len(vs))
	for _, v := range vs {
		out = append(out, map[string]any{
			"version": v.Version, "description": v.Description, "status": v.Status,
			"created_at": v.CreatedAt.UTC().Format(time.RFC3339Nano),
		})
	}
	WriteJSON(w, http.StatusOK, map[string]any{"versions": out})
}

// POST /api/knowledge/versions
func (a *API) createKnowledgeVersion(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Version     string `json:"version"`
		Description string `json:"description"`
		Status      string `json:"status"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Version == "" {
		a.badRequest(w, r, "version is required")
		return
	}
	if err := a.Store.UpsertVersion(r.Context(), req.Version, req.Description); err != nil {
		a.fail(w, r, err)
		return
	}
	status := req.Status
	if status == "" {
		status = "active"
	}
	WriteJSON(w, http.StatusOK, map[string]any{"version": req.Version, "status": status})
}

// GET /api/knowledge/documents?limit=
func (a *API) knowledgeDocuments(w http.ResponseWriter, r *http.Request) {
	docs, err := a.Store.ListDocuments(r.Context(), queryInt(r, "limit", 100))
	if err != nil {
		a.fail(w, r, err)
		return
	}
	out := make([]map[string]any, 0, len(docs))
	for _, d := range docs {
		out = append(out, map[string]any{
			"doc_id": d.DocID, "title": d.Title, "category": d.Category,
			"effective_from": d.EffectiveFrom, "version": d.Version, "source_uri": d.SourceURI,
			"created_at": d.CreatedAt.UTC().Format(time.RFC3339Nano),
		})
	}
	WriteJSON(w, http.StatusOK, map[string]any{"documents": out})
}

// POST /api/knowledge/documents（手动 upsert 一篇文档并嵌入入库）
func (a *API) createKnowledgeDocument(w http.ResponseWriter, r *http.Request) {
	var req struct {
		DocID         string `json:"doc_id"`
		Title         string `json:"title"`
		Content       string `json:"content"`
		Category      string `json:"category"`
		EffectiveFrom string `json:"effective_from"`
		Version       string `json:"version"`
		SourceURI     string `json:"source_uri"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.DocID == "" {
		a.badRequest(w, r, "doc_id is required")
		return
	}
	version := req.Version
	if version == "" {
		version = knowledgeVersion
	}
	doc := store.Document{
		DocID: req.DocID, Title: req.Title, Content: req.Content, Category: req.Category,
		Version: version, EffectiveFrom: nilIfEmpty(req.EffectiveFrom), SourceURI: nilIfEmpty(req.SourceURI),
	}
	if err := a.Store.UpsertDocument(r.Context(), doc); err != nil {
		a.fail(w, r, err)
		return
	}
	if err := a.embedAndStore(r, req.DocID, version, req.Title+"\n"+req.Content); err != nil {
		WriteError(w, r, http.StatusBadGateway, "ai_unavailable", "embed failed: "+err.Error())
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{"doc_id": req.DocID, "status": "upserted"})
}

// POST /api/knowledge/rebuild-index：方案 A——把 benchmark_runs 摘要灌成知识文档并嵌入。
func (a *API) rebuildKnowledgeIndex(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	_ = a.Store.UpsertVersion(ctx, knowledgeVersion, "benchmark serving logs")

	rows, err := a.Pool.Query(ctx, `
		SELECT run_id, endpoint_id, workload, routing_strategy, status, summary, created_at
		  FROM benchmark_runs ORDER BY created_at DESC LIMIT 500`)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	type runDoc struct{ docID, text, title, srcURI string }
	pending := []runDoc{}
	for rows.Next() {
		var runID, status string
		var endpoint, workload, routing *string
		var summary []byte
		var createdAt time.Time
		if err := rows.Scan(&runID, &endpoint, &workload, &routing, &status, &summary, &createdAt); err != nil {
			rows.Close()
			a.fail(w, r, err)
			return
		}
		ep, wl := derefOr(endpoint, "?"), derefOr(workload, "?")
		pending = append(pending, runDoc{
			docID:  "benchmark:" + runID,
			title:  fmt.Sprintf("Benchmark %s @ %s (%s)", wl, ep, status),
			srcURI: "benchmark_runs/" + runID,
			text:   renderBenchmarkDoc(runID, ep, wl, derefOr(routing, ""), status, summary, createdAt),
		})
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		a.fail(w, r, err)
		return
	}

	count := 0
	for _, d := range pending {
		if err := a.Store.UpsertDocument(ctx, store.Document{
			DocID: d.docID, Title: d.title, Content: d.text, Category: "benchmark",
			Version: knowledgeVersion, SourceURI: &d.srcURI,
		}); err != nil {
			a.fail(w, r, err)
			return
		}
		if err := a.embedAndStore(r, d.docID, knowledgeVersion, d.text); err != nil {
			WriteError(w, r, http.StatusBadGateway, "ai_unavailable", "embed failed: "+err.Error())
			return
		}
		count++
	}
	operationalCount, err := a.indexOperationalKnowledge(r)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	count += operationalCount
	WriteJSON(w, http.StatusOK, map[string]any{
		"status": "rebuilt", "index_type": "pgvector_hnsw_v1", "document_count": count,
		"operational_document_count": operationalCount,
		"note":                       "Corpus = benchmark evidence + historical incidents + config changes + training outcomes; embeddings via ai-service /internal/embed.",
	})
}

func (a *API) indexOperationalKnowledge(r *http.Request) (int, error) {
	ctx := r.Context()
	type operationalDoc struct{ id, title, category, text, source string }
	docs := []operationalDoc{}

	incidentRows, err := a.Pool.Query(ctx, `SELECT id, title, severity, status, COALESCE(summary,''), created_at, resolved_at
		FROM incidents ORDER BY created_at DESC LIMIT 300`)
	if err != nil {
		return 0, err
	}
	for incidentRows.Next() {
		var id, title, severity, status, summary string
		var createdAt time.Time
		var resolvedAt *time.Time
		if err := incidentRows.Scan(&id, &title, &severity, &status, &summary, &createdAt, &resolvedAt); err != nil {
			incidentRows.Close()
			return 0, err
		}
		text := fmt.Sprintf("Incident: %s\nSeverity: %s\nStatus: %s\nSummary/root cause: %s\nCreated: %s\nResolved: %v",
			title, severity, status, summary, createdAt.UTC().Format(time.RFC3339), resolvedAt)
		docs = append(docs, operationalDoc{"incident:" + id, title, "incident", text, "incidents/" + id})
	}
	incidentRows.Close()

	configRows, err := a.Pool.Query(ctx, `SELECT c.id, c.config_key, c.env, v.version, COALESCE(v.change_reason,''),
		COALESCE(v.operator,''), COALESCE(v.content,''), v.created_at
		FROM config_versions v JOIN config_items c ON c.id=v.config_item_id
		ORDER BY v.created_at DESC LIMIT 300`)
	if err != nil {
		return 0, err
	}
	for configRows.Next() {
		var id, key, env, reason, operator, content string
		var version int
		var createdAt time.Time
		if err := configRows.Scan(&id, &key, &env, &version, &reason, &operator, &content, &createdAt); err != nil {
			configRows.Close()
			return 0, err
		}
		if len(content) > 4000 {
			content = content[:4000]
		}
		text := fmt.Sprintf("Config change: %s v%d\nEnvironment: %s\nReason: %s\nOperator: %s\nChanged: %s\nContent:\n%s",
			key, version, env, reason, operator, createdAt.UTC().Format(time.RFC3339), content)
		docID := fmt.Sprintf("config:%s:v%d", id, version)
		docs = append(docs, operationalDoc{docID, key + fmt.Sprintf(" v%d", version), "config", text, "config_items/" + id})
	}
	configRows.Close()

	trainingRows, err := a.Pool.Query(ctx, `SELECT id, name, base_model, status, hyperparams, metadata, created_at
		FROM training_jobs ORDER BY created_at DESC LIMIT 200`)
	if err == nil {
		for trainingRows.Next() {
			var id, name, baseModel, status string
			var hyperparams, metadata []byte
			var createdAt time.Time
			if trainingRows.Scan(&id, &name, &baseModel, &status, &hyperparams, &metadata, &createdAt) == nil {
				text := fmt.Sprintf("Training job: %s\nBase model: %s\nStatus: %s\nHyperparameters: %s\nEvents and evidence: %s\nCreated: %s",
					name, baseModel, status, hyperparams, metadata, createdAt.UTC().Format(time.RFC3339))
				docs = append(docs, operationalDoc{"training:" + id, name, "training", text, "training_jobs/" + id})
			}
		}
		trainingRows.Close()
	}

	count := 0
	for _, doc := range docs {
		source := doc.source
		if err := a.Store.UpsertDocument(ctx, store.Document{DocID: doc.id, Title: doc.title, Content: doc.text,
			Category: doc.category, Version: knowledgeVersion, SourceURI: &source}); err != nil {
			return count, err
		}
		if err := a.embedAndStore(r, doc.id, knowledgeVersion, doc.text); err != nil {
			return count, err
		}
		count++
	}
	return count, nil
}

// GET /api/knowledge/search?q=&top_k=
func (a *API) knowledgeSearch(w http.ResponseWriter, r *http.Request) {
	q := strings.TrimSpace(r.URL.Query().Get("q"))
	if q == "" {
		a.badRequest(w, r, "q is required")
		return
	}
	topK := clamp(queryInt(r, "top_k", 4), 1, 20)
	vecs, err := a.AI.Embed(r.Context(), []string{q}, true)
	if err != nil || len(vecs) == 0 {
		WriteError(w, r, http.StatusBadGateway, "ai_unavailable", "query embed failed")
		return
	}
	hits, err := a.Store.SearchDocuments(r.Context(), vecs[0], "", topK)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	docs := make([]map[string]any, 0, len(hits))
	for _, h := range hits {
		docs = append(docs, map[string]any{
			"doc_id": h.DocID, "title": h.Title, "category": h.Category,
			"version": h.Version, "score": h.Score,
		})
	}
	WriteJSON(w, http.StatusOK, map[string]any{"query": q, "documents": docs})
}

// embedAndStore 嵌入单篇文档文本（1 切块）并落 kb_chunks。
func (a *API) embedAndStore(r *http.Request, docID, version, text string) error {
	vecs, err := a.AI.Embed(r.Context(), []string{text}, false)
	if err != nil {
		return err
	}
	var emb []float32
	if len(vecs) > 0 {
		emb = vecs[0]
	}
	return a.Store.ReplaceChunks(r.Context(), docID, version, []store.Chunk{{Ordinal: 0, Text: text, Embedding: emb}})
}

// renderBenchmarkDoc 把一条 benchmark_run 渲染成可检索的知识文本。
func renderBenchmarkDoc(runID, endpoint, workload, routing, status string, summary []byte, createdAt time.Time) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Benchmark run %s\nEndpoint: %s\nWorkload: %s\nRouting strategy: %s\nStatus: %s\nStarted: %s\n",
		runID, endpoint, workload, routing, status, createdAt.UTC().Format(time.RFC3339))
	var parsed struct {
		Scenarios []map[string]any `json:"scenarios"`
	}
	if err := json.Unmarshal(summary, &parsed); err == nil {
		for _, s := range parsed.Scenarios {
			fmt.Fprintf(&b, "- concurrency=%v requests=%v p50=%vms p95=%vms p99=%vms qps=%v tokens/s=%v error_rate=%v\n",
				s["concurrency"], s["requests"], s["p50_ms"], s["p95_ms"], s["p99_ms"],
				s["qps"], s["output_tokens_per_second"], s["error_rate"])
		}
	}
	return b.String()
}
