package httpx

import (
	"context"
	"encoding/json"
	"log/slog"
	"time"
)

type requestTraceInput struct {
	RequestID      string
	SessionID      string
	EndpointID     string
	TargetPod      string
	ModelID        string
	RetrievalMs    any
	QueueGatewayMs any
	TTFTMs         any
	GenerationMs   any
	TotalMs        any
	InputTokens    any
	OutputTokens   any
	Status         string
	Error          string
	Metadata       map[string]any
}

// recordRequestTrace is deliberately best-effort: telemetry persistence must
// never turn a successful inference response into a failed user request.
func (a *API) recordRequestTrace(in requestTraceInput) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	metadata, _ := json.Marshal(in.Metadata)
	status := in.Status
	if status == "" {
		status = "ok"
	}
	_, err := a.Pool.Exec(ctx, `INSERT INTO request_traces
		(request_id, session_id, endpoint_id, target_pod, model_id, retrieval_ms,
		 queue_or_gateway_ms, ttft_ms, generation_ms, total_ms, input_tokens,
		 output_tokens, status, error, metadata)
		VALUES ($1,NULLIF($2,''),NULLIF($3,''),NULLIF($4,''),NULLIF($5,''),$6,$7,$8,$9,$10,$11,$12,$13,NULLIF($14,''),$15::jsonb)`,
		in.RequestID, in.SessionID, in.EndpointID, in.TargetPod, in.ModelID,
		in.RetrievalMs, in.QueueGatewayMs, in.TTFTMs, in.GenerationMs, in.TotalMs,
		in.InputTokens, in.OutputTokens, status, in.Error, metadata)
	if err != nil {
		slog.Warn("persist request trace failed", "request_id", in.RequestID, "err", err)
	}
}
