package httpx

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/google/uuid"
)

// 6A：进程内 serving 压测 runner（取代 legacy 起 Python 子进程跑 11_benchmark_serving.py）。
// 对解析出的端点做并发负载扫描（concurrency_levels × requests_per_level），流式请求测 TTFT，
// 由响应 usage 取 token；逐请求落 `request` 事件、逐档落 `scenario_summary`，汇总写 MinIO 报告。
// 无 live 模型时如实落 error 事件 + failed（与 legacy 行为一致）。

var benchmarkCancels sync.Map

// workloadPrompts 把 workload 名映射到提示词（覆盖 legacy 的 prompt_profile 常用档）。
var workloadPrompts = map[string]string{
	"faq_short":                      "What is the refund policy for a defective product?",
	"faq_long":                       "Explain in detail the full warranty, refund, and return process for an order that arrived damaged, including timelines and required documents.",
	"mixed_peak":                     "Summarize the order cancellation policy and how refunds are issued.",
	"chitchat":                       "Hi, can you help me with my recent order?",
	"customer_support_shared_prefix": "请根据以下客服历史案例和服务规范，回答最后一位客户的问题。",
	"ticket_long_context":            "Analyze the following support ticket timeline and answer the final question.",
}

func workloadPrompt(workload string) string {
	if p, ok := workloadPrompts[workload]; ok {
		return p
	}
	return "What is the refund policy?"
}

// runServingBenchmark 是后台压测主流程（脱离请求 context）。
func (a *API) runServingBenchmark(ctx context.Context, runID string, ep *resolvedEndpoint, req benchmarkRunRequest) {
	defer benchmarkCancels.Delete(runID)
	defer func() {
		if rec := recover(); rec != nil {
			slog.Error("benchmark runner panic", "run_id", runID, "err", rec)
			a.failBenchmark(runID, fmt.Sprintf("panic: %v", rec))
		}
	}()
	dataset := loadBenchmarkDataset(req.Dataset)
	a.appendBenchmarkEvent(ctx, runID, "started",
		map[string]any{"endpoint": ep.TargetPod, "base_url": ep.BaseURL, "workload": req.Workload, "dataset": dataset.info()})
	_, _ = a.Pool.Exec(ctx, `UPDATE benchmark_runs SET status='running', updated_at=now() WHERE run_id=$1`, runID)

	scenarios := make([]map[string]any, 0, len(req.ConcurrencyLevels))
	if len(req.ContextMix) > 0 {
		for _, c := range req.ConcurrencyLevels {
			if ctx.Err() != nil {
				a.finishCancelledBenchmark(runID)
				return
			}
			scenarios = append(scenarios, a.runScenario(ctx, runID, ep, dataset, req.Workload, req.ContextMix,
				req.MaxTokens, c, req.RequestsPerLevel, boolValue(req.VLLM.EnableThinking), req.PriorityByContext, req.VLLM.FrequencyPenalty, req.VLLM.Stop))
		}
	} else {
		for _, ctxLen := range req.ContextLengths {
			if ctx.Err() != nil {
				a.finishCancelledBenchmark(runID)
				return
			}
			if ctxLen <= 0 {
				continue
			}
			for _, c := range req.ConcurrencyLevels {
				if ctx.Err() != nil {
					a.finishCancelledBenchmark(runID)
					return
				}
				if c <= 0 {
					continue
				}
				scenarios = append(scenarios, a.runScenario(ctx, runID, ep, dataset, req.Workload, []int{ctxLen}, req.MaxTokens, c, req.RequestsPerLevel, boolValue(req.VLLM.EnableThinking), false, req.VLLM.FrequencyPenalty, req.VLLM.Stop))
			}
		}
	}
	if ctx.Err() != nil {
		a.finishCancelledBenchmark(runID)
		return
	}

	summary := map[string]any{
		"scenarios": scenarios,
		"optimization_profile": map[string]any{
			"goal":        "minimize TTFT and TPOT while preserving request success rate and non-empty output",
			"dataset":     dataset.info(),
			"vllm_params": req.VLLM,
			"matrix": map[string]any{
				"context_lengths":     req.ContextLengths,
				"context_mix":         req.ContextMix,
				"priority_by_context": req.PriorityByContext,
				"concurrency_levels":  req.ConcurrencyLevels,
				"requests_per_level":  req.RequestsPerLevel,
				"max_tokens":          req.MaxTokens,
			},
			"best": bestScenario(scenarios),
		},
	}
	// 请求级错误不能只停留在 request 事件里：如果整个矩阵都失败，
	// 之前仍会把运行标记为 completed，用户只能看到一张“成功完成但成功率为 0”的报告。
	// 汇总所有场景，给运行记录一个明确的失败状态和可读原因。
	totalRequests, totalErrors := 0, 0
	firstError := ""
	for _, scenario := range scenarios {
		totalRequests += intFromAny(scenario["requests"])
		totalErrors += intFromAny(scenario["errors"])
		if firstError == "" {
			if value, ok := scenario["first_error"].(string); ok {
				firstError = strings.TrimSpace(value)
			}
		}
	}
	allRequestsFailed := totalRequests > 0 && totalErrors == totalRequests
	errMsg := ""
	if allRequestsFailed {
		if firstError == "" {
			firstError = "endpoint 在压测期间不可达或上游运行时未就绪"
		}
		errMsg = fmt.Sprintf("所有压测请求均失败（%d/%d）：%s", totalErrors, totalRequests, firstError)
		summary["failure"] = map[string]any{
			"error":         errMsg,
			"request_count": totalRequests,
			"error_count":   totalErrors,
			"endpoint":      ep.TargetPod,
			"base_url":      ep.BaseURL,
		}
	}
	a.appendBenchmarkEvent(ctx, runID, "optimization_report", summary["optimization_profile"].(map[string]any))
	reportPath := a.uploadBenchmarkReport(ctx, runID, ep, req, summary)

	sb, _ := json.Marshal(summary)
	if allRequestsFailed {
		a.appendBenchmarkEvent(ctx, runID, "error", map[string]any{
			"error": errMsg, "endpoint": ep.TargetPod, "base_url": ep.BaseURL,
			"request_count": totalRequests, "error_count": totalErrors,
		})
		_, _ = a.Pool.Exec(ctx,
			`UPDATE benchmark_runs SET status='failed', summary=$2, report_path=$3, error=$4, updated_at=now() WHERE run_id=$1`,
			runID, sb, nullableStr(reportPath), errMsg)
		_ = a.recordPlatformLog(ctx, platformLogInput{
			Level: "error", Source: "benchmark-runner", ResourceType: "benchmark_run", ResourceID: runID,
			Message: "benchmark failed: " + errMsg,
			Attributes: map[string]any{"run_id": runID, "error": errMsg, "endpoint": ep.TargetPod, "base_url": ep.BaseURL},
		})
		a.appendBenchmarkEvent(ctx, runID, "process_exit", map[string]any{
			"status": "failed", "scenarios": len(scenarios), "request_count": totalRequests, "error_count": totalErrors,
		})
		return
	}
	_, _ = a.Pool.Exec(ctx,
		`UPDATE benchmark_runs SET status='completed', summary=$2, report_path=$3, updated_at=now() WHERE run_id=$1`,
		runID, sb, nullableStr(reportPath))
	a.appendBenchmarkEvent(ctx, runID, "process_exit", map[string]any{"status": "completed", "scenarios": len(scenarios)})
}

func (a *API) finishCancelledBenchmark(runID string) {
	ctx := context.Background()
	_, _ = a.Pool.Exec(ctx, `UPDATE benchmark_runs SET status='cancelled', updated_at=now() WHERE run_id=$1 AND status NOT IN ('completed','failed')`, runID)
}

// runScenario 跑一个并发档，落逐请求事件 + 一条 scenario_summary，返回该档汇总。
func (a *API) runScenario(ctx context.Context, runID string, ep *resolvedEndpoint, dataset benchmarkDataset, workload string, contextLengths []int, maxTokens, concurrency, total int, enableThinking, priorityByContext bool, frequencyPenalty float64, stop []string) map[string]any {
	scenarioContext := maxIntSlice(contextLengths)
	a.appendBenchmarkEvent(ctx, runID, "scenario_start",
		map[string]any{"concurrency": concurrency, "context_length": scenarioContext, "context_mix": contextLengths, "requests": total})

	var (
		mu                sync.Mutex
		latencies         []float64
		ttfts             []float64
		tpots             []float64
		referenceOverlaps []float64
		errCount          int
		okCount           int
		refCount          int
		refPassCount      int
		qualityPassCount  int
		completedCount    int
		truncatedCount    int
		safetyFailCount   int
		firstError        string
		outTokSum         float64
		inTokSum          float64
		classMetrics      = map[int]*benchmarkClassMetrics{}
		sem               = make(chan struct{}, concurrency)
		wg                sync.WaitGroup
	)
	gpuBefore := a.benchmarkGPUSnapshot(ctx)
	start := time.Now()
requestLoop:
	for i := 0; i < total; i++ {
		select {
		case sem <- struct{}{}:
		case <-ctx.Done():
			break requestLoop
		}
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			defer func() { <-sem }()
			contextLength := contextLengths[i%len(contextLengths)]
			prompt, sampleID, referenceAnswer := benchmarkPrompt(dataset, workload, contextLength, i)
			priority := 0
			if priorityByContext && contextLength < scenarioContext {
				priority = -1
			}
			res := a.runOneRequest(ctx, ep, prompt, referenceAnswer, maxTokens, enableThinking, priority, frequencyPenalty, stop)
			mu.Lock()
			class := classMetrics[contextLength]
			if class == nil {
				class = &benchmarkClassMetrics{}
				classMetrics[contextLength] = class
			}
			class.requests++
			if res.err != "" {
				errCount++
				class.errors++
				if firstError == "" {
					firstError = res.err
				}
			} else {
				okCount++
				latencies = append(latencies, res.totalMs)
				if res.ttftMs > 0 {
					ttfts = append(ttfts, res.ttftMs)
				}
				if res.tpotMs > 0 {
					tpots = append(tpots, res.tpotMs)
				}
				inTokSum += res.inputTokens
				outTokSum += res.outputTokens
				class.latencies = append(class.latencies, res.totalMs)
				class.ttfts = append(class.ttfts, res.ttftMs)
				class.tpots = append(class.tpots, res.tpotMs)
				class.outputTokens += res.outputTokens
				if res.hasReference {
					refCount++
					referenceOverlaps = append(referenceOverlaps, res.referenceOverlap)
					if res.referencePass {
						refPassCount++
					}
				}
				if res.finishReason == "stop" {
					completedCount++
				}
				if res.finishReason == "length" {
					truncatedCount++
				}
				if !res.safetyPass {
					safetyFailCount++
				}
				if res.qualityPass {
					qualityPassCount++
				}
			}
			mu.Unlock()
			payload := map[string]any{
				"request_id": res.requestID, "total_ms": res.totalMs, "ttft_ms": nilIfZero(res.ttftMs),
				"tpot_ms": nilIfZero(res.tpotMs), "input_tokens": res.inputTokens,
				"output_tokens": res.outputTokens, "target_pod": ep.TargetPod,
				"context_length": contextLength, "concurrency": concurrency,
				"output_valid": res.outputValid, "output_sha256": res.outputSHA256,
				"output_preview": res.outputPreview, "finish_reason": res.finishReason,
				"quality_pass": res.qualityPass, "safety_pass": res.safetyPass,
				"sample_id": sampleID,
			}
			if res.hasReference {
				payload["reference_overlap"] = res.referenceOverlap
				payload["reference_pass"] = res.referencePass
			}
			if res.err != "" {
				payload["error"] = res.err
			}
			a.appendBenchmarkEvent(ctx, runID, "request", payload)
			// 压测请求同时进入统一请求追踪。此前只写 benchmark_samples，导致
			// 可观测中心（request_traces/metrics_samples）在压测期间始终显示 0。
			a.appendBenchmarkTrace(ctx, runID, ep, res)
		}(i)
	}
	wg.Wait()
	wallSec := time.Since(start).Seconds()
	gpuAfter := a.benchmarkGPUSnapshot(ctx)

	summary := map[string]any{
		"context_length":           scenarioContext,
		"context_mix":              contextLengths,
		"priority_by_context":      priorityByContext,
		"context_classes":          summarizeContextClasses(classMetrics, wallSec),
		"concurrency":              concurrency,
		"requests":                 total,
		"errors":                   errCount,
		"error_rate":               ratio(errCount, total),
		"success_rate":             ratio(okCount, total),
		"output_valid_rate":        ratio(okCount, total),
		"completed_output_rate":    ratio(completedCount, total),
		"quality_gate_pass_rate":   ratio(qualityPassCount, total),
		"truncation_rate":          ratio(truncatedCount, total),
		"safety_violation_rate":    ratio(safetyFailCount, total),
		"reference_pass_rate":      ratio(refPassCount, refCount),
		"mean_reference_overlap":   mean(referenceOverlaps),
		"p50_ms":                   pctile(latencies, 50),
		"p95_ms":                   pctile(latencies, 95),
		"p99_ms":                   pctile(latencies, 99),
		"mean_ms":                  mean(latencies),
		"mean_ttft_ms":             mean(ttfts),
		"p95_ttft_ms":              pctile(ttfts, 95),
		"mean_tpot_ms":             mean(tpots),
		"p95_tpot_ms":              pctile(tpots, 95),
		"qps":                      rate(len(latencies), wallSec),
		"output_tokens_per_second": rate(int(outTokSum), wallSec),
		"input_tokens":             inTokSum,
		"total_output_tokens":      outTokSum,
		"gpu_before":               gpuBefore,
		"gpu_after":                gpuAfter,
	}
	if firstError != "" {
		summary["first_error"] = firstError
	}
	summary["bottleneck"] = attributeBottleneck(summary)
	summary["recommendations"] = recommendOptimizations(summary)
	a.appendBenchmarkEvent(ctx, runID, "scenario_summary", map[string]any{"concurrency": concurrency, "context_length": scenarioContext, "context_mix": contextLengths, "summary": summary})
	a.Metrics.PersistSample(ctx, "benchmark:"+runID, benchmarkMetricsSample(runID, ep, summary))
	return summary
}

// appendBenchmarkTrace 把压测数据接入平台统一 trace 契约，供实时指标、请求追踪和 AIOps 共用。
func (a *API) appendBenchmarkTrace(ctx context.Context, runID string, ep *resolvedEndpoint, res reqResult) {
	status := "ok"
	if res.err != "" {
		status = "error"
	}
	generationMs := math.Max(0, res.totalMs-res.ttftMs)
	metadata, _ := json.Marshal(map[string]any{"source": "benchmark", "benchmark_run_id": runID, "finish_reason": res.finishReason})
	_, _ = a.Pool.Exec(ctx, `
		INSERT INTO request_traces
			(request_id, session_id, endpoint_id, target_pod, model_id, ttft_ms, generation_ms,
			 total_ms, input_tokens, output_tokens, status, error, metadata)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13::jsonb)`,
		res.requestID, "benchmark:"+runID, ep.EndpointID, ep.TargetPod, ep.ModelID,
		nilIfZero(res.ttftMs), generationMs, res.totalMs, int(res.inputTokens), int(res.outputTokens),
		status, nullableStr(res.err), string(metadata))
}

func benchmarkMetricsSample(runID string, ep *resolvedEndpoint, summary map[string]any) map[string]any {
	requests := intFromAny(summary["requests"])
	errors := intFromAny(summary["errors"])
	return map[string]any{
		"window": "benchmark_scenario", "benchmark_run_id": runID, "source": "benchmark",
		"request_count": requests, "error_count": errors, "error_rate": summary["error_rate"],
		"qps": summary["qps"], "mean_latency_ms": summary["mean_ms"],
		"p50_latency_ms": summary["p50_ms"], "p95_latency_ms": summary["p95_ms"],
		"p99_latency_ms": summary["p99_ms"], "mean_ttft_ms": summary["mean_ttft_ms"],
		"p95_ttft_ms": summary["p95_ttft_ms"], "tokens_per_second": summary["output_tokens_per_second"],
		"input_tokens": summary["input_tokens"], "output_tokens": summary["total_output_tokens"],
		"endpoint_counts": map[string]int{ep.EndpointID: requests}, "target_pod_counts": map[string]int{ep.TargetPod: requests},
	}
}

type benchmarkClassMetrics struct {
	requests     int
	errors       int
	latencies    []float64
	ttfts        []float64
	tpots        []float64
	outputTokens float64
}

func summarizeContextClasses(classes map[int]*benchmarkClassMetrics, wallSec float64) map[string]any {
	result := make(map[string]any, len(classes))
	for contextLength, class := range classes {
		result[fmt.Sprint(contextLength)] = map[string]any{
			"requests":                       class.requests,
			"success_rate":                   ratio(class.requests-class.errors, class.requests),
			"p95_ms":                         pctile(class.latencies, 95),
			"p95_ttft_ms":                    pctile(class.ttfts, 95),
			"p95_tpot_ms":                    pctile(class.tpots, 95),
			"output_tokens_per_second_share": rate(int(class.outputTokens), wallSec),
		}
	}
	return result
}

func maxIntSlice(values []int) int {
	result := 0
	for _, value := range values {
		if value > result {
			result = value
		}
	}
	return result
}

type reqResult struct {
	requestID        string
	totalMs          float64
	ttftMs           float64
	inputTokens      float64
	outputTokens     float64
	tpotMs           float64
	outputValid      bool
	outputSHA256     string
	outputPreview    string
	finishReason     string
	qualityPass      bool
	safetyPass       bool
	hasReference     bool
	referenceOverlap float64
	referencePass    bool
	err              string
}

// runOneRequest 发一条流式 chat completion，测 TTFT/总时延，由 usage 取 token。
func (a *API) runOneRequest(ctx context.Context, ep *resolvedEndpoint, prompt, referenceAnswer string, maxTokens int, enableThinking bool, priority int, frequencyPenalty float64, stop []string) reqResult {
	res := reqResult{requestID: uuid.NewString(), hasReference: strings.TrimSpace(referenceAnswer) != ""}
	payload := map[string]any{
		"model":          ep.ModelID,
		"messages":       []map[string]string{{"role": "user", "content": prompt}},
		"max_tokens":     maxTokens,
		"temperature":    0,
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
		"chat_template_kwargs": map[string]any{
			"enable_thinking": enableThinking,
		},
	}
	if priority != 0 {
		payload["priority"] = priority
	}
	if frequencyPenalty != 0 {
		payload["frequency_penalty"] = frequencyPenalty
	}
	if len(stop) > 0 {
		payload["stop"] = stop
	}
	body, _ := json.Marshal(payload)
	rctx, cancel := context.WithTimeout(ctx, 120*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(rctx, http.MethodPost, upstreamBaseURL(ep.BaseURL)+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		res.err = err.Error()
		return res
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer EMPTY")
	req.Header.Set("model", ep.ModelID)
	if ep.RoutingStrategy != "" {
		req.Header.Set("routing-strategy", ep.RoutingStrategy)
	}
	start := time.Now()
	resp, err := proxyHTTPClient.Do(req)
	if err != nil {
		res.err = err.Error()
		return res
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		res.err = fmt.Sprintf("upstream status %d", resp.StatusCode)
		return res
	}

	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	deltas := 0
	var output strings.Builder
	var reasoning strings.Builder
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}
		var ev struct {
			Choices []struct {
				Delta struct {
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     float64 `json:"prompt_tokens"`
				CompletionTokens float64 `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			continue
		}
		if len(ev.Choices) > 0 {
			choice := ev.Choices[0]
			delta := choice.Delta
			if delta.Content != "" || delta.ReasoningContent != "" {
				if res.ttftMs == 0 {
					res.ttftMs = float64(time.Since(start).Milliseconds())
				}
				deltas++
			}
			output.WriteString(delta.Content)
			reasoning.WriteString(delta.ReasoningContent)
			if choice.FinishReason != "" {
				res.finishReason = choice.FinishReason
			}
		}
		if ev.Usage != nil {
			res.inputTokens = ev.Usage.PromptTokens
			res.outputTokens = ev.Usage.CompletionTokens
		}
	}
	res.totalMs = float64(time.Since(start).Milliseconds())
	if res.outputTokens == 0 {
		res.outputTokens = float64(deltas) // usage 缺失时以 delta 数近似
	}
	res.outputValid = strings.TrimSpace(output.String()) != ""
	if res.outputValid && utf8.ValidString(output.String()) {
		res.outputSHA256 = fmt.Sprintf("%x", sha256.Sum256([]byte(output.String())))
		res.outputPreview = runePrefix(strings.TrimSpace(output.String()), 240)
	} else if res.outputValid {
		res.outputValid = false
		res.err = "output is not valid UTF-8"
	}
	if !res.outputValid && res.err == "" {
		if reasoning.Len() > 0 {
			res.err = "empty final output after reasoning"
		} else {
			res.err = "empty output"
		}
	}
	if res.outputValid && res.hasReference {
		res.referenceOverlap = charNgramF1(output.String(), referenceAnswer, 2)
		res.referencePass = res.referenceOverlap >= 0.08
	}
	res.safetyPass = !requestsSensitiveCredential(output.String())
	res.qualityPass = res.outputValid && res.finishReason == "stop" && res.safetyPass
	if res.ttftMs > 0 && res.outputTokens > 1 && res.totalMs > res.ttftMs {
		res.tpotMs = (res.totalMs - res.ttftMs) / (res.outputTokens - 1)
	}
	if err := sc.Err(); err != nil {
		res.err = err.Error()
	}
	return res
}

// uploadBenchmarkReport 把汇总写 MinIO（benchmarks/{run_id}/report.json），返回对象 key；Blob 禁用→""。
func (a *API) uploadBenchmarkReport(ctx context.Context, runID string, ep *resolvedEndpoint, req benchmarkRunRequest, summary map[string]any) string {
	if a.Blob == nil || !a.Blob.Enabled() {
		return ""
	}
	report := map[string]any{
		"run_id": runID, "endpoint": ep.TargetPod, "workload": req.Workload,
		"routing_strategy": req.RoutingStrategy, "generated_at": time.Now().UTC().Format(time.RFC3339),
		"summary": summary,
	}
	b, _ := json.MarshalIndent(report, "", "  ")
	key := "benchmarks/" + runID + "/report.json"
	uri, err := a.Blob.Put(ctx, key, bytes.NewReader(b), int64(len(b)), "application/json")
	if err != nil {
		slog.Warn("benchmark report upload failed", "run_id", runID, "err", err)
		return ""
	}
	return uri
}

func (a *API) failBenchmark(runID, errMsg string) {
	ctx := context.Background()
	a.appendBenchmarkEvent(ctx, runID, "error", map[string]any{"error": errMsg})
	_, _ = a.Pool.Exec(ctx, `UPDATE benchmark_runs SET status='failed', error=$2, updated_at=now() WHERE run_id=$1`, runID, errMsg)
	_ = a.recordPlatformLog(ctx, platformLogInput{
		Level: "error", Source: "benchmark-runner", ResourceType: "benchmark_run", ResourceID: runID,
		Message: "benchmark failed: " + errMsg, Attributes: map[string]any{"run_id": runID, "error": errMsg},
	})
}

// ---- 小工具 ----

func pctile(v []float64, p float64) any {
	if len(v) == 0 {
		return nil
	}
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	if len(s) == 1 {
		return s[0]
	}
	rank := (p / 100.0) * float64(len(s)-1)
	lo := int(math.Floor(rank))
	hi := int(math.Ceil(rank))
	if lo == hi {
		return s[lo]
	}
	return s[lo] + (s[hi]-s[lo])*(rank-float64(lo))
}

func mean(v []float64) any {
	if len(v) == 0 {
		return nil
	}
	sum := 0.0
	for _, x := range v {
		sum += x
	}
	return sum / float64(len(v))
}

func ratio(n, d int) any {
	if d == 0 {
		return nil
	}
	return float64(n) / float64(d)
}

func rate(n int, sec float64) any {
	if sec <= 0 {
		return nil
	}
	return float64(n) / sec
}

func nilIfZero(v float64) any {
	if v == 0 {
		return nil
	}
	return v
}

func nullableStr(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func runePrefix(value string, limit int) string {
	runes := []rune(value)
	if len(runes) <= limit {
		return value
	}
	return string(runes[:limit])
}

func requestsSensitiveCredential(output string) bool {
	normalized := strings.Join(strings.Fields(output), "")
	for _, phrase := range []string{
		"请提供密码", "请告诉我密码", "提供您的密码", "发送您的密码",
		"请提供验证码", "请告诉我验证码", "提供您的验证码", "发送验证码给我",
	} {
		if strings.Contains(normalized, phrase) {
			return true
		}
	}
	return false
}

type benchmarkSample struct {
	ID               string   `json:"id"`
	Scenario         string   `json:"scenario"`
	ContextLength    int      `json:"context_length"`
	Prompt           string   `json:"prompt"`
	ExpectedKeywords []string `json:"expected_keywords,omitempty"`
	SharedPrefixID   string   `json:"shared_prefix_id,omitempty"`
	ReferenceAnswer  string   `json:"reference_answer,omitempty"`
}

type benchmarkDataset struct {
	Name     string
	Path     string
	Samples  []benchmarkSample
	Fallback bool
	Err      string
}

func (d benchmarkDataset) info() map[string]any {
	return map[string]any{
		"name": d.Name, "path": d.Path, "samples": len(d.Samples),
		"fallback": d.Fallback, "error": d.Err,
	}
}

// loadBenchmarkDataset reads the normalized benchmark slice produced by
// scripts/prepare_dianjin_csc.py. Raw third-party data stays under ignored data/raw.
func loadBenchmarkDataset(name string) benchmarkDataset {
	dataset := benchmarkDataset{Name: name, Fallback: true}
	paths := []string{}
	if configured := strings.TrimSpace(os.Getenv("BENCHMARK_DATASET_PATH")); configured != "" {
		paths = append(paths, configured)
	}
	paths = append(paths,
		"data/cleaned/dianjin_csc_benchmark.jsonl",
		"../data/cleaned/dianjin_csc_benchmark.jsonl",
	)
	for _, path := range paths {
		file, err := os.Open(path)
		if err != nil {
			continue
		}
		scanner := bufio.NewScanner(file)
		scanner.Buffer(make([]byte, 64*1024), 4<<20)
		samples := []benchmarkSample{}
		for scanner.Scan() {
			var sample benchmarkSample
			if err := json.Unmarshal(scanner.Bytes(), &sample); err == nil && sample.ID != "" && sample.Prompt != "" {
				samples = append(samples, sample)
			}
		}
		scanErr := scanner.Err()
		_ = file.Close()
		if scanErr != nil {
			dataset.Err = scanErr.Error()
			continue
		}
		if len(samples) > 0 {
			dataset.Path = path
			dataset.Samples = samples
			dataset.Fallback = false
			dataset.Err = ""
			return dataset
		}
	}
	dataset.Err = "normalized dataset not found; run scripts/prepare_dianjin_csc.py"
	return dataset
}

func benchmarkPrompt(dataset benchmarkDataset, workload string, contextLength, requestIndex int) (string, string, string) {
	candidates := make([]benchmarkSample, 0, len(dataset.Samples))
	for _, sample := range dataset.Samples {
		if sample.ContextLength != contextLength {
			continue
		}
		if workload == "customer_support_shared_prefix" && sample.Scenario != "customer_support_shared_prefix" {
			continue
		}
		candidates = append(candidates, sample)
	}
	if len(candidates) == 0 {
		for _, sample := range dataset.Samples {
			if sample.ContextLength == contextLength {
				candidates = append(candidates, sample)
			}
		}
	}
	if len(candidates) > 0 {
		sample := candidates[requestIndex%len(candidates)]
		return sample.Prompt, sample.ID, sample.ReferenceAnswer
	}

	base := workloadPrompt(workload)
	policy := "服务规范：先确认客户诉求和必要身份信息；不得索取完整密码或验证码；明确说明办理步骤、预计时效和异常升级渠道；无法确认时转人工客服并生成工单。\n"
	var prompt strings.Builder
	prompt.WriteString(base)
	prompt.WriteString("\n\n")
	targetRunes := max(contextLength*2-120, 200)
	for utf8.RuneCountInString(prompt.String()) < targetRunes {
		prompt.WriteString(policy)
	}
	prompt.WriteString(fmt.Sprintf("\n客户问题 %d：我申请提前结清后一直显示审核中，请说明当前处理步骤、时效和需要补充的材料。", requestIndex+1))
	return prompt.String(), fmt.Sprintf("fallback-%d-%d", contextLength, requestIndex+1), ""
}

func charNgramF1(generated, reference string, n int) float64 {
	normalize := func(value string) []rune {
		out := []rune{}
		for _, r := range strings.ToLower(value) {
			if unicode.IsLetter(r) || unicode.IsNumber(r) {
				out = append(out, r)
			}
		}
		return out
	}
	grams := func(runes []rune) map[string]int {
		out := map[string]int{}
		if len(runes) < n {
			return out
		}
		for i := 0; i <= len(runes)-n; i++ {
			out[string(runes[i:i+n])]++
		}
		return out
	}
	generatedGrams := grams(normalize(generated))
	referenceGrams := grams(normalize(reference))
	if len(generatedGrams) == 0 || len(referenceGrams) == 0 {
		return 0
	}
	overlap, generatedTotal, referenceTotal := 0, 0, 0
	for gram, count := range generatedGrams {
		generatedTotal += count
		if referenceCount := referenceGrams[gram]; referenceCount < count {
			overlap += referenceCount
		} else {
			overlap += count
		}
	}
	for _, count := range referenceGrams {
		referenceTotal += count
	}
	precision := float64(overlap) / float64(generatedTotal)
	recall := float64(overlap) / float64(referenceTotal)
	if precision+recall == 0 {
		return 0
	}
	return 2 * precision * recall / (precision + recall)
}

func (a *API) benchmarkGPUSnapshot(ctx context.Context) map[string]any {
	if a.Agent == nil {
		return map[string]any{"available": false, "gpu_count": 0, "error": "agent client is not configured"}
	}
	response := a.agentGpu(ctx)
	status, _ := response["status"].(map[string]any)
	devices := asSlice(response["gpu"])
	maxUsed, maxTotal, maxMemoryPct, maxUtil := 0.0, 0.0, 0.0, 0.0
	for _, device := range devices {
		gpu, _ := device.(map[string]any)
		used, _ := asFloat(gpu["memory_used_mb"])
		total, _ := asFloat(gpu["memory_total_mb"])
		memoryPct, _ := asFloat(gpu["memory_utilization_percent"])
		util, _ := asFloat(gpu["gpu_utilization_percent"])
		maxUsed = math.Max(maxUsed, used)
		maxTotal = math.Max(maxTotal, total)
		maxMemoryPct = math.Max(maxMemoryPct, memoryPct)
		maxUtil = math.Max(maxUtil, util)
	}
	available, _ := status["available"].(bool)
	return map[string]any{
		"available": available, "gpu_count": len(devices), "devices": devices,
		"max_memory_used_mb": maxUsed, "max_memory_total_mb": maxTotal,
		"max_memory_utilization_percent": maxMemoryPct,
		"max_gpu_utilization_percent":    maxUtil, "error": status["error"],
	}
}

func attributeBottleneck(summary map[string]any) map[string]any {
	labels := []string{}
	evidence := []string{}
	successRate, _ := asFloat(summary["success_rate"])
	qualityPassRate, hasQualityPassRate := asFloat(summary["quality_gate_pass_rate"])
	contextLength, _ := asFloat(summary["context_length"])
	concurrency, _ := asFloat(summary["concurrency"])
	p95, _ := asFloat(summary["p95_ms"])
	meanLatency, _ := asFloat(summary["mean_ms"])
	ttft, hasTTFT := asFloat(summary["p95_ttft_ms"])
	tpot, hasTPOT := asFloat(summary["p95_tpot_ms"])
	gpuAfter, _ := summary["gpu_after"].(map[string]any)
	memoryPct, _ := asFloat(gpuAfter["max_memory_utilization_percent"])

	if successRate < 0.99 {
		labels = append(labels, "stability-risk")
		evidence = append(evidence, fmt.Sprintf("success_rate=%.2f%%", successRate*100))
	}
	if hasQualityPassRate && qualityPassRate < 0.99 {
		labels = append(labels, "quality-check-failed")
		evidence = append(evidence, fmt.Sprintf("quality_gate_pass_rate=%.2f%%", qualityPassRate*100))
	}
	if memoryPct >= 90 {
		labels = append(labels, "memory-pressure")
		evidence = append(evidence, fmt.Sprintf("gpu_memory=%.1f%%", memoryPct))
	}
	if hasTTFT && contextLength >= 1024 && ttft > math.Max(500, tpot*8) {
		labels = append(labels, "prefill-bound")
		evidence = append(evidence, fmt.Sprintf("p95_ttft=%.0fms at %.0f tokens", ttft, contextLength))
	}
	if hasTPOT && tpot > 80 {
		labels = append(labels, "decode-bound")
		evidence = append(evidence, fmt.Sprintf("p95_tpot=%.1fms", tpot))
	}
	if concurrency >= 8 && meanLatency > 0 && p95 > meanLatency*1.5 {
		labels = append(labels, "scheduler-saturation")
		evidence = append(evidence, fmt.Sprintf("p95/mean=%.2f at concurrency %.0f", p95/meanLatency, concurrency))
	}
	if len(labels) == 0 {
		labels = append(labels, "balanced-or-insufficient-evidence")
		if !hasTTFT || !hasTPOT {
			evidence = append(evidence, "TTFT or TPOT sample is unavailable")
		}
	}
	return map[string]any{"labels": labels, "evidence": evidence}
}

func recommendOptimizations(summary map[string]any) []map[string]any {
	bottleneck, _ := summary["bottleneck"].(map[string]any)
	labels, _ := bottleneck["labels"].([]string)
	recommendations := []map[string]any{}
	seen := map[string]bool{}
	add := func(parameter, action, reason string) {
		if seen[parameter+action] {
			return
		}
		seen[parameter+action] = true
		recommendations = append(recommendations, map[string]any{"parameter": parameter, "action": action, "reason": reason})
	}
	for _, label := range labels {
		switch label {
		case "stability-risk":
			add("max_num_seqs", "decrease one sweep level", "restore success rate before comparing latency")
		case "quality-check-failed":
			add("quality_gate", "inspect truncated, empty, or credential-unsafe samples", "do not accept a faster profile that fails deterministic customer-service checks")
		case "memory-pressure":
			add("gpu_memory_utilization", "decrease by 0.03-0.05", "leave headroom for KV cache and runtime workspace")
			add("max_model_len", "cap to the tested 1K/2K workload", "avoid reserving unused KV-cache capacity")
		case "prefill-bound":
			add("max_num_batched_tokens", "sweep 2048/4096/8192 with chunked prefill enabled", "Qwen3.6 does not officially support disabling chunked prefill in vLLM 0.19.1")
			add("enable_prefix_caching", "enable for shared-prefix workload", "reuse repeated customer-service policy prefixes")
		case "decode-bound":
			add("max_num_seqs", "sweep around the stable baseline", "balance continuous batching and per-request TPOT")
		case "scheduler-saturation":
			add("max_num_batched_tokens", "sweep 2048/4096/8192", "find the latency-throughput knee under concurrency")
		}
	}
	if len(recommendations) == 0 {
		add("baseline", "repeat three times before tuning", "current rule set has no dominant bottleneck signal")
	}
	return recommendations
}

func bestScenario(scenarios []map[string]any) map[string]any {
	best := map[string]any{}
	bestScore := math.Inf(1)
	for _, scenario := range scenarios {
		success, okSuccess := asFloat(scenario["success_rate"])
		qualityPassRate, hasQualityPassRate := asFloat(scenario["quality_gate_pass_rate"])
		ttft, okTTFT := asFloat(scenario["p95_ttft_ms"])
		tpot, okTPOT := asFloat(scenario["p95_tpot_ms"])
		if !okSuccess || success < 0.99 || !hasQualityPassRate || qualityPassRate < 0.99 || !okTTFT || !okTPOT {
			continue
		}
		score := ttft + 10*tpot
		if score < bestScore {
			bestScore = score
			best = map[string]any{
				"context_length": scenario["context_length"], "concurrency": scenario["concurrency"],
				"p95_ttft_ms": ttft, "p95_tpot_ms": tpot,
				"success_rate": success, "quality_gate_pass_rate": qualityPassRate, "score": score,
			}
		}
	}
	return best
}
