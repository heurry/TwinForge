package httpx

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/heurry/cloudnative-infra-platform/server/internal/aiclient"
	"github.com/heurry/cloudnative-infra-platform/server/internal/config"
	"github.com/heurry/cloudnative-infra-platform/server/internal/db"
	"github.com/heurry/cloudnative-infra-platform/server/internal/store"
	"github.com/jackc/pgx/v5/pgxpool"
)

// mockEmbedVec：确定性 1024 维单位向量（不同文本 → 不同分量）。
func mockEmbedVec(s string) []float32 {
	v := make([]float32, 1024)
	// Keep this test's query and its seeded benchmark in the same semantic
	// bucket. Hash-only one-hot vectors made unrelated documents tie at score 0,
	// so a shared integration database could decide top_k only by row order.
	lower := strings.ToLower(s)
	if strings.Contains(lower, "kn-run-1") {
		v[0] = 1
		return v
	}
	h := 0
	for _, c := range s {
		h = h*31 + int(c)
	}
	if h < 0 {
		h = -h
	}
	v[h%1024] = 1
	return v
}

// 集成测试：需带 pgvector 的 PG（TEST_DATABASE_URL）。覆盖 rebuild-index（benchmark_runs→kb_*，
// 经 mock embed）+ documents/search/versions 契约。
func TestKnowledgeRebuildSearch(t *testing.T) {
	url := os.Getenv("TEST_DATABASE_URL")
	if url == "" {
		t.Skip("set TEST_DATABASE_URL (pgvector) to run knowledge handlers test")
	}
	ctx := context.Background()
	if err := db.Migrate(config.Config{DatabaseURL: url}.MigrateURL()); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	pool, err := pgxpool.New(ctx, url)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer pool.Close()

	// mock ai-service /internal/embed。
	embedSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/internal/embed" {
			http.Error(w, "bad path", http.StatusNotFound)
			return
		}
		var req struct {
			Texts []string `json:"texts"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		embs := make([][]float32, len(req.Texts))
		for i, t := range req.Texts {
			embs[i] = mockEmbedVec(t)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"embeddings": embs, "model": "mock", "dim": 1024, "mode": "stub"})
	}))
	defer embedSrv.Close()

	a := &API{Pool: pool, Store: store.New(pool), AI: aiclient.New(embedSrv.URL, 10*time.Second)}

	// 一条 benchmark_run 作为知识语料。
	if _, err := pool.Exec(ctx,
		`INSERT INTO benchmark_runs (run_id, status, endpoint_id, workload, summary)
		 VALUES ('kn-run-1', 'completed', 'aibrix-gateway', 'faq_short', $1)
		 ON CONFLICT (run_id) DO UPDATE SET summary = EXCLUDED.summary`,
		`{"scenarios":[{"concurrency":2,"requests":4,"p95_ms":120,"qps":3.2}]}`); err != nil {
		t.Fatalf("seed benchmark_run: %v", err)
	}
	defer pool.Exec(ctx, `DELETE FROM benchmark_runs WHERE run_id='kn-run-1'`)
	defer pool.Exec(ctx, `DELETE FROM kb_chunks WHERE doc_id='benchmark:kn-run-1'`)
	defer pool.Exec(ctx, `DELETE FROM kb_documents WHERE doc_id='benchmark:kn-run-1'`)

	rt := chi.NewRouter()
	rt.Route("/api/knowledge", func(r chi.Router) {
		r.Get("/versions", a.knowledgeVersions)
		r.Get("/documents", a.knowledgeDocuments)
		r.Post("/rebuild-index", a.rebuildKnowledgeIndex)
		r.Get("/search", a.knowledgeSearch)
	})
	do := func(method, path string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		rt.ServeHTTP(rec, httptest.NewRequest(method, path, nil))
		return rec
	}

	// rebuild-index → 至少灌入 1 篇。
	rb := do(http.MethodPost, "/api/knowledge/rebuild-index")
	if rb.Code != http.StatusOK {
		t.Fatalf("rebuild status %d: %s", rb.Code, rb.Body.String())
	}
	var rbBody struct {
		DocumentCount int    `json:"document_count"`
		IndexType     string `json:"index_type"`
	}
	_ = json.Unmarshal(rb.Body.Bytes(), &rbBody)
	if rbBody.DocumentCount < 1 || rbBody.IndexType != "pgvector_hnsw_v1" {
		t.Fatalf("rebuild unexpected: %s", rb.Body.String())
	}

	// documents → 含 benchmark:kn-run-1。
	dr := do(http.MethodGet, "/api/knowledge/documents?limit=50")
	if !strings.Contains(dr.Body.String(), "benchmark:kn-run-1") {
		t.Fatalf("documents missing benchmark doc: %s", dr.Body.String())
	}

	// search → 返回该文档。
	sr := do(http.MethodGet, "/api/knowledge/search?q=kn-run-1+p95+latency&top_k=4")
	if sr.Code != http.StatusOK {
		t.Fatalf("search status %d: %s", sr.Code, sr.Body.String())
	}
	var srBody struct {
		Documents []map[string]any `json:"documents"`
	}
	_ = json.Unmarshal(sr.Body.Bytes(), &srBody)
	if len(srBody.Documents) == 0 || srBody.Documents[0]["doc_id"] != "benchmark:kn-run-1" {
		t.Fatalf("search did not return the benchmark doc: %s", sr.Body.String())
	}

	// versions → 含 default。
	vr := do(http.MethodGet, "/api/knowledge/versions")
	if !strings.Contains(vr.Body.String(), `"version":"default"`) {
		t.Fatalf("versions missing default: %s", vr.Body.String())
	}
}
