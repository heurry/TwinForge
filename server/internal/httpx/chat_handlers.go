package httpx

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"github.com/heurry/cloudnative-infra-platform/server/internal/store"
)

// 6A：chat 组绞杀——Go 原生客服 RAG 对话（会话/消息→PG chat_*；messages:stream 是
// 检索→提示→流式→落库的完整 RAG 管线，复用 #51 检索 + #48 端点解析）。
// response_filter is intentionally omitted because thinking is disabled. The
// completed RAG request is persisted into request_traces for the console while
// aggregate serving metrics continue to come from the vLLM scraper.

// POST /api/chat/sessions
func (a *API) createChatSession(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Title    string `json:"title"`
		UserRole string `json:"user_role"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)
	title := req.Title
	if title == "" {
		title = "New customer support chat"
	}
	sessionID, err := a.Store.CreateChatSession(r.Context(), title, req.UserRole)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{
		"session_id": sessionID, "title": title, "created_at": time.Now().UTC().Format(time.RFC3339Nano),
	})
}

// GET /api/chat/sessions
func (a *API) listChatSessions(w http.ResponseWriter, r *http.Request) {
	sessions, err := a.Store.ListChatSessions(r.Context(), queryInt(r, "limit", 50))
	if err != nil {
		a.fail(w, r, err)
		return
	}
	out := make([]map[string]any, 0, len(sessions))
	for _, s := range sessions {
		out = append(out, sessionDTO(s))
	}
	WriteJSON(w, http.StatusOK, map[string]any{"sessions": out})
}

// GET /api/chat/sessions/{session_id}
func (a *API) getChatSession(w http.ResponseWriter, r *http.Request) {
	sessionID := chi.URLParam(r, "session_id")
	sess, err := a.Store.GetChatSession(r.Context(), sessionID)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	if sess == nil {
		WriteError(w, r, http.StatusNotFound, "not_found", "session not found")
		return
	}
	msgs, err := a.Store.ListChatMessages(r.Context(), sessionID)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	out := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, map[string]any{
			"message_id": m.MessageID, "role": m.Role, "content": m.Content,
			"metadata": orEmptyMap(m.Metadata), "created_at": m.CreatedAt.UTC().Format(time.RFC3339Nano),
		})
	}
	WriteJSON(w, http.StatusOK, map[string]any{"session": sessionDTO(*sess), "messages": out})
}

// POST /api/chat/messages/{message_id}/feedback
func (a *API) createMessageFeedback(w http.ResponseWriter, r *http.Request) {
	messageID := chi.URLParam(r, "message_id")
	sessionID, requestID, ok, err := a.Store.GetMessageSession(r.Context(), messageID)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	if !ok {
		WriteError(w, r, http.StatusNotFound, "not_found", "message not found")
		return
	}
	var req struct {
		Rating string `json:"rating"`
		Note   string `json:"note"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Rating == "" {
		a.badRequest(w, r, "rating is required")
		return
	}
	feedbackID, err := a.Store.CreateMessageFeedback(r.Context(), messageID, sessionID, requestID, req.Rating, req.Note)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{"feedback_id": feedbackID, "message_id": messageID, "rating": req.Rating})
}

// POST /api/chat/sessions/{session_id}/messages:stream —— RAG 流式问答（SSE）。
func (a *API) streamChatMessage(w http.ResponseWriter, r *http.Request) {
	sessionID := chi.URLParam(r, "session_id")
	var req struct {
		Content     string   `json:"content"`
		EndpointID  string   `json:"endpoint_id"`
		MaxTokens   int      `json:"max_tokens"`
		Temperature *float64 `json:"temperature"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || strings.TrimSpace(req.Content) == "" {
		a.badRequest(w, r, "content is required")
		return
	}
	ctx := r.Context()
	sess, err := a.Store.GetChatSession(ctx, sessionID)
	if err != nil {
		a.fail(w, r, err)
		return
	}
	if sess == nil {
		WriteError(w, r, http.StatusNotFound, "not_found", "session not found")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	fl, _ := w.(http.Flusher)

	requestID := uuid.NewString()
	totalStart := time.Now()
	memory, _ := a.Store.FetchMemory(ctx, sessionID, 8)
	retrievalQuery := buildMemoryAwareQuery(req.Content, memory)
	_, _ = a.Store.InsertChatMessage(ctx, sessionID, "user", req.Content, nil)

	// 检索（embed 失败 → 空 docs → 走 no_retrieval 兜底，不中断流）。
	// E2：retrieveDocs 默认等价 SearchDocuments；开启 RAG_RERANK_FEEDBACK 时按反馈净分微调排序。
	retrStart := time.Now()
	docs := []store.DocHit{}
	reranked := false
	if vecs, err := a.AI.Embed(ctx, []string{retrievalQuery}, true); err == nil && len(vecs) > 0 {
		docs, reranked, _ = a.retrieveDocs(ctx, vecs[0], 4)
	}
	retrievalMs := msSince(retrStart)
	citationDocIDs := make([]string, 0, len(docs))
	for _, d := range docs {
		citationDocIDs = append(citationDocIDs, d.DocID)
	}
	writeSSE(w, fl, "retrieval", map[string]any{
		"request_id": requestID, "query": retrievalQuery, "memory_turns": len(memory),
		"documents": docList(docs), "retrieval_ms": retrievalMs, "reranked": reranked,
	})

	var answer strings.Builder
	targetPod, streamErr := "", ""
	selectedEndpointID := req.EndpointID
	if selectedEndpointID == "" {
		selectedEndpointID = "auto-router"
	}
	ttftMs := 0.0
	fallbackReason := decideFallback(req.Content, docs)

	if fallbackReason == "" {
		endpointID := selectedEndpointID
		ep, _, rerr := a.resolveEndpoint(ctx, endpointID)
		if rerr != nil {
			fallbackReason, streamErr = "model_endpoint_error", rerr.Error()
		} else {
			targetPod = ep.TargetPod
			writeSSE(w, fl, "route", map[string]any{
				"request_id": requestID, "endpoint_id": endpointID, "selected_endpoint_id": ep.TargetPod,
				"routing_strategy": ep.RoutingStrategy, "target_hint": ep.TargetPod,
			})
			temp := 0.2
			if req.Temperature != nil {
				temp = *req.Temperature
			}
			maxTok := req.MaxTokens
			if maxTok <= 0 {
				maxTok = 1024
			}
			tt, serr := a.streamUpstreamChat(ctx, ep, buildChatMessages(req.Content, docs, memory, 6000), maxTok, temp, w, fl, requestID, &answer)
			if serr != nil {
				fallbackReason, streamErr = "model_endpoint_error", serr.Error()
			} else {
				ttftMs = tt
			}
		}
	}

	if fallbackReason != "" && answer.Len() == 0 {
		ans := buildFallbackAnswer(docs, fallbackReason)
		answer.WriteString(ans)
		fb := map[string]any{"request_id": requestID, "reason": fallbackReason}
		if streamErr != "" {
			fb["error"] = streamErr
		}
		writeSSE(w, fl, "fallback", fb)
		writeSSE(w, fl, "token", map[string]any{"request_id": requestID, "text": ans})
	}

	answerText := strings.TrimSpace(answer.String())
	_, _ = a.Store.InsertChatMessage(ctx, sessionID, "assistant", answerText, map[string]any{
		"request_id": requestID, "citation_doc_ids": citationDocIDs, "target_pod": targetPod,
		"fallback_reason": fallbackReason, "memory_turns": len(memory), "retrieval_query": retrievalQuery,
	})

	totalMs := msSince(totalStart)
	status := "ok"
	if streamErr != "" {
		status = "error"
	}
	trace := map[string]any{
		"request_id": requestID, "session_id": sessionID, "retrieval_ms": retrievalMs,
		"ttft_ms": nilIfZero(ttftMs), "generation_ms": maxF(totalMs-retrievalMs, 0), "total_ms": totalMs,
		"target_pod": targetPod, "fallback_reason": fallbackReason, "citation_doc_ids": citationDocIDs,
		"status": status, "error": streamErr,
	}
	go a.recordRequestTrace(requestTraceInput{
		RequestID: requestID, SessionID: sessionID, EndpointID: selectedEndpointID, TargetPod: targetPod,
		RetrievalMs: retrievalMs, TTFTMs: nilIfZero(ttftMs), GenerationMs: maxF(totalMs-retrievalMs, 0),
		TotalMs: totalMs, Status: status, Error: streamErr,
		Metadata: map[string]any{"plane": "rag_chat", "fallback_reason": fallbackReason, "citation_doc_ids": citationDocIDs},
	})
	writeSSE(w, fl, "citation", map[string]any{"request_id": requestID, "doc_ids": citationDocIDs})
	writeSSE(w, fl, "metrics", trace)
	writeSSE(w, fl, "done", map[string]any{"request_id": requestID})
}

// streamUpstreamChat 流式打上游 chat/completions，逐 token 发 SSE 并累计答案，返回 TTFT(ms)。
func (a *API) streamUpstreamChat(ctx context.Context, ep *resolvedEndpoint, messages []map[string]string, maxTokens int, temperature float64, w http.ResponseWriter, fl http.Flusher, requestID string, answer *strings.Builder) (float64, error) {
	body, _ := json.Marshal(map[string]any{
		"model": ep.ModelID, "messages": messages, "max_tokens": maxTokens, "temperature": temperature,
		"stream": true, "stream_options": map[string]any{"include_usage": true},
		"chat_template_kwargs": map[string]any{"enable_thinking": false},
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, upstreamBaseURL(ep.BaseURL)+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer EMPTY")
	req.Header.Set("model", ep.ModelID)
	if ep.RoutingStrategy != "" {
		req.Header.Set("routing-strategy", ep.RoutingStrategy)
	}
	resp, err := proxyHTTPClient.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return 0, fmt.Errorf("upstream status %d", resp.StatusCode)
	}
	start := time.Now()
	ttft := 0.0
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
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
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			continue
		}
		if len(ev.Choices) > 0 && ev.Choices[0].Delta.Content != "" {
			if ttft == 0 {
				ttft = msSince(start)
			}
			answer.WriteString(ev.Choices[0].Delta.Content)
			writeSSE(w, fl, "token", map[string]any{"request_id": requestID, "text": ev.Choices[0].Delta.Content})
		}
	}
	return ttft, sc.Err()
}

// ---- helpers ----

func sessionDTO(s store.ChatSession) map[string]any {
	return map[string]any{
		"session_id": s.SessionID, "title": s.Title, "user_role": s.UserRole, "kind": s.Kind,
		"created_at": s.CreatedAt.UTC().Format(time.RFC3339Nano), "updated_at": s.UpdatedAt.UTC().Format(time.RFC3339Nano),
	}
}

func docList(docs []store.DocHit) []map[string]any {
	out := make([]map[string]any, 0, len(docs))
	for _, d := range docs {
		out = append(out, map[string]any{
			"doc_id": d.DocID, "title": d.Title, "category": d.Category, "version": d.Version, "score": d.Score,
		})
	}
	return out
}

func writeSSE(w http.ResponseWriter, fl http.Flusher, event string, data map[string]any) {
	b, _ := json.Marshal(data)
	_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
	if fl != nil {
		fl.Flush()
	}
}

func orEmptyMap(m map[string]any) map[string]any {
	if m == nil {
		return map[string]any{}
	}
	return m
}

func msSince(t time.Time) float64 { return float64(time.Since(t).Microseconds()) / 1000.0 }
func maxF(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
