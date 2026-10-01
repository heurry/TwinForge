package httpx

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/heurry/cloudnative-infra-platform/server/internal/config"
	"github.com/heurry/cloudnative-infra-platform/server/internal/db"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPctile(t *testing.T) {
	if got := pctile([]float64{10, 20, 30, 40}, 50); got != 25.0 {
		t.Fatalf("p50 = %v, want 25", got)
	}
	if got := pctile(nil, 95); got != nil {
		t.Fatalf("empty pctile should be nil, got %v", got)
	}
	if got := pctile([]float64{7}, 99); got != 7.0 {
		t.Fatalf("single pctile = %v, want 7", got)
	}
}

func TestBenchmarkRequestValidation(t *testing.T) {
	req := benchmarkRunRequest{ContextLengths: []int{2048}, ConcurrencyLevels: []int{1, 16}, RequestsPerLevel: 16, MaxTokens: 256}
	req.VLLM.MaxModelLen = 2048
	if err := req.validate(); err == nil {
		t.Fatal("expected context plus output tokens to exceed max_model_len")
	}
	req.VLLM.MaxModelLen = 4096
	if err := req.validate(); err != nil {
		t.Fatalf("expected baseline matrix to be valid: %v", err)
	}
	req.RequestsPerLevel = 8
	if err := req.validate(); err == nil {
		t.Fatal("expected request count below max concurrency to be rejected")
	}
}

func TestBenchmarkPromptUsesNormalizedDataset(t *testing.T) {
	dataset := benchmarkDataset{Samples: []benchmarkSample{{
		ID: "dianjin-1k-001", Scenario: "customer_support_shared_prefix",
		ContextLength: 1024, Prompt: "客服数据集提示词",
	}}}
	prompt, sampleID, _ := benchmarkPrompt(dataset, "customer_support_shared_prefix", 1024, 0)
	if prompt != "客服数据集提示词" || sampleID != "dianjin-1k-001" {
		t.Fatalf("unexpected dataset prompt: prompt=%q sample_id=%q", prompt, sampleID)
	}
}

func TestCharNgramF1(t *testing.T) {
	if got := charNgramF1("请提供手机尾号进行身份核验", "请提供绑定手机号码的尾号，以便完成身份核验", 2); got < 0.30 {
		t.Fatalf("expected related customer-support replies to overlap, got %.3f", got)
	}
	if got := charNgramF1("天气晴朗", "请核实还款账户", 2); got != 0 {
		t.Fatalf("expected unrelated replies not to overlap, got %.3f", got)
	}
}

func TestCustomerSupportSafetyGate(t *testing.T) {
	if requestsSensitiveCredential("请提供密码以继续处理") != true {
		t.Fatal("expected direct password solicitation to fail")
	}
	if requestsSensitiveCredential("请勿提供密码或验证码，我们不会索取") {
		t.Fatal("expected a safety warning to pass")
	}
}

func TestBottleneckAttributionAndRecommendation(t *testing.T) {
	summary := map[string]any{
		"success_rate": 1.0, "context_length": 2048, "concurrency": 8,
		"quality_gate_pass_rate": 1.0,
		"p95_ms":                 2400.0, "mean_ms": 1000.0,
		"p95_ttft_ms": 1800.0, "p95_tpot_ms": 90.0,
		"gpu_after": map[string]any{"max_memory_utilization_percent": 93.0},
	}
	bottleneck := attributeBottleneck(summary)
	labels, _ := bottleneck["labels"].([]string)
	joined := strings.Join(labels, ",")
	for _, expected := range []string{"memory-pressure", "prefill-bound", "decode-bound", "scheduler-saturation"} {
		if !strings.Contains(joined, expected) {
			t.Fatalf("expected label %q in %v", expected, labels)
		}
	}
	summary["bottleneck"] = bottleneck
	if got := recommendOptimizations(summary); len(got) < 4 {
		t.Fatalf("expected actionable recommendations, got %v", got)
	}
}

func TestBestScenarioRequiresStableOutput(t *testing.T) {
	best := bestScenario([]map[string]any{
		{"context_length": 1024, "concurrency": 8, "success_rate": 0.95, "quality_gate_pass_rate": 1.0, "p95_ttft_ms": 100.0, "p95_tpot_ms": 10.0},
		{"context_length": 1024, "concurrency": 4, "success_rate": 1.0, "quality_gate_pass_rate": 0.5, "p95_ttft_ms": 150.0, "p95_tpot_ms": 15.0},
		{"context_length": 1024, "concurrency": 2, "success_rate": 1.0, "quality_gate_pass_rate": 1.0, "p95_ttft_ms": 200.0, "p95_tpot_ms": 20.0},
	})
	if got, _ := asFloat(best["concurrency"]); got != 2 {
		t.Fatalf("best scenario must exclude unstable result, got %v", best)
	}
}

// 集成测试：需 TEST_DATABASE_URL。POST serving → 后台 runner 打 mock SSE 上游 → 轮询至 completed，
// 校验 scenario_summary / request 事件与 GET run 契约。
func TestBenchmarksServing(t *testing.T) {
	url := os.Getenv("TEST_DATABASE_URL")
	if url == "" {
		t.Skip("set TEST_DATABASE_URL to run benchmarks integration test")
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
	a := &API{Pool: pool}

	// mock vLLM：SSE 两个 content delta + 一个带 usage 的块 + [DONE]。
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.Error(w, "bad path", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"re\"}}]}\n\n")
		if fl != nil {
			fl.Flush()
		}
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"fund\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":6,\"completion_tokens\":2}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	if _, err := pool.Exec(ctx,
		`INSERT INTO service_instances (name, base_url, model_id, kind, routing_role, status)
		 VALUES ('bench-mock', $1, 'qwen3-4b-platform', 'vllm', 'replica', 'healthy')
		 ON CONFLICT (name) DO UPDATE SET base_url = EXCLUDED.base_url, status='healthy'`,
		upstream.URL+"/v1"); err != nil {
		t.Fatalf("insert mock instance: %v", err)
	}
	defer pool.Exec(ctx, `DELETE FROM service_instances WHERE name='bench-mock'`)

	rt := chi.NewRouter()
	rt.Post("/api/benchmarks/serving", a.createServingBenchmark)
	rt.Get("/api/benchmarks/{run_id}", a.benchmarkRun)
	rt.Get("/api/benchmarks/{run_id}/events", a.benchmarkEvents)

	// POST serving（小负载：1 档并发 2，4 个请求）。
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/benchmarks/serving",
		strings.NewReader(`{"endpoint_id":"bench-mock","dataset":"DianJin/DianJin-CSC-Data","context_lengths":[1024],"concurrency_levels":[2],"requests_per_level":4,"max_tokens":8}`))
	rt.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("POST serving status %d: %s", rec.Code, rec.Body.String())
	}
	var created struct {
		RunID  string `json:"run_id"`
		Status string `json:"status"`
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &created)
	if created.RunID == "" || created.Status != "queued" {
		t.Fatalf("unexpected create response: %s", rec.Body.String())
	}
	defer pool.Exec(ctx, `DELETE FROM benchmark_runs WHERE run_id=$1`, created.RunID)
	defer pool.Exec(ctx, `DELETE FROM benchmark_samples WHERE run_id=$1`, created.RunID)

	// 轮询直至 completed（runner 后台 goroutine）。
	var status string
	for i := 0; i < 50; i++ {
		rr := httptest.NewRecorder()
		gr := httptest.NewRequest(http.MethodGet, "/api/benchmarks/"+created.RunID, nil)
		rt.ServeHTTP(rr, gr)
		var run struct {
			Status  string         `json:"status"`
			Summary map[string]any `json:"summary"`
		}
		_ = json.Unmarshal(rr.Body.Bytes(), &run)
		status = run.Status
		if status == "completed" {
			if scen, ok := run.Summary["scenarios"].([]any); !ok || len(scen) != 1 {
				t.Fatalf("expected 1 scenario in summary, got %v", run.Summary["scenarios"])
			}
			break
		}
		if status == "failed" {
			t.Fatalf("benchmark failed unexpectedly")
		}
		time.Sleep(100 * time.Millisecond)
	}
	if status != "completed" {
		t.Fatalf("benchmark did not complete, last status=%q", status)
	}

	// events：应含 4 个 request 与 1 个 scenario_summary。
	er := httptest.NewRecorder()
	rt.ServeHTTP(er, httptest.NewRequest(http.MethodGet, "/api/benchmarks/"+created.RunID+"/events", nil))
	var evbody struct {
		Events []struct {
			EventType string         `json:"event_type"`
			Payload   map[string]any `json:"payload"`
		} `json:"events"`
	}
	_ = json.Unmarshal(er.Body.Bytes(), &evbody)
	reqEvents, summaries := 0, 0
	for _, e := range evbody.Events {
		switch e.EventType {
		case "request":
			reqEvents++
			if e.Payload["context_length"] != float64(1024) || e.Payload["output_valid"] != true {
				t.Fatalf("request event misses context/output validity: %v", e.Payload)
			}
		case "scenario_summary":
			summaries++
		}
	}
	if reqEvents != 4 {
		t.Fatalf("expected 4 request events, got %d", reqEvents)
	}
	if summaries != 1 {
		t.Fatalf("expected 1 scenario_summary, got %d", summaries)
	}
}
