// Package observability provides OpenTelemetry tracing and Prometheus metrics
// shared by the standalone Agent API and Worker processes.
package observability

import (
	"context"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"
)

var (
	httpRequests  = prometheus.NewCounterVec(prometheus.CounterOpts{Name: "agent_http_requests_total", Help: "Agent HTTP requests."}, []string{"method", "route", "status"})
	httpDuration  = prometheus.NewHistogramVec(prometheus.HistogramOpts{Name: "agent_http_request_duration_seconds", Help: "Agent HTTP latency.", Buckets: prometheus.DefBuckets}, []string{"method", "route"})
	modelCalls    = prometheus.NewCounterVec(prometheus.CounterOpts{Name: "agent_model_calls_total", Help: "Agent model calls."}, []string{"provider", "model", "status"})
	modelDuration = prometheus.NewHistogramVec(prometheus.HistogramOpts{Name: "agent_model_call_duration_seconds", Help: "Agent model call latency.", Buckets: prometheus.ExponentialBuckets(.05, 2, 12)}, []string{"provider", "model"})
	modelTokens   = prometheus.NewCounterVec(prometheus.CounterOpts{Name: "agent_model_tokens_total", Help: "Agent model tokens."}, []string{"provider", "model", "direction"})
	toolCalls     = prometheus.NewCounterVec(prometheus.CounterOpts{Name: "agent_tool_calls_total", Help: "Agent tool calls."}, []string{"tool", "status"})
	toolDuration  = prometheus.NewHistogramVec(prometheus.HistogramOpts{Name: "agent_tool_call_duration_seconds", Help: "Agent tool call latency.", Buckets: prometheus.DefBuckets}, []string{"tool"})
	uuidPath      = regexp.MustCompile(`/[0-9a-fA-F-]{36}`)
)

func init() {
	prometheus.MustRegister(httpRequests, httpDuration, modelCalls, modelDuration, modelTokens, toolCalls, toolDuration)
}

type Config struct {
	ServiceName, ServiceVersion, OTLPEndpoint string
	Insecure                                  bool
}

func Init(ctx context.Context, cfg Config) (func(context.Context) error, error) {
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{}))
	res, err := resource.Merge(resource.Default(), resource.NewWithAttributes(semconv.SchemaURL, semconv.ServiceName(cfg.ServiceName), semconv.ServiceVersion(cfg.ServiceVersion)))
	if err != nil {
		res = resource.Default()
	}
	opts := []sdktrace.TracerProviderOption{sdktrace.WithResource(res)}
	var initErr error
	if strings.TrimSpace(cfg.OTLPEndpoint) != "" {
		exporter, exportErr := otlptracehttp.New(ctx, otlptracehttp.WithEndpointURL(cfg.OTLPEndpoint))
		if exportErr != nil {
			initErr = exportErr
		} else {
			opts = append(opts, sdktrace.WithBatcher(exporter))
		}
	}
	tp := sdktrace.NewTracerProvider(opts...)
	otel.SetTracerProvider(tp)
	return tp.Shutdown, initErr
}
func MetricsHandler() http.Handler { return promhttp.Handler() }

type recorder struct {
	http.ResponseWriter
	status int
}

func (r *recorder) WriteHeader(code int) { r.status = code; r.ResponseWriter.WriteHeader(code) }
func (r *recorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}
func (r *recorder) Unwrap() http.ResponseWriter { return r.ResponseWriter }

func HTTPMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/metrics" {
			next.ServeHTTP(w, r)
			return
		}
		ctx := otel.GetTextMapPropagator().Extract(r.Context(), propagation.HeaderCarrier(r.Header))
		route := uuidPath.ReplaceAllString(r.URL.Path, "/{id}")
		ctx, span := otel.Tracer("agent-platform/http").Start(ctx, r.Method+" "+route, trace.WithSpanKind(trace.SpanKindServer))
		defer span.End()
		started := time.Now()
		rw := &recorder{ResponseWriter: w, status: 200}
		next.ServeHTTP(rw, r.WithContext(ctx))
		span.SetAttributes(attribute.String("http.request.method", r.Method), attribute.String("http.route", route), attribute.Int("http.response.status_code", rw.status))
		httpRequests.WithLabelValues(r.Method, route, strconv.Itoa(rw.status)).Inc()
		httpDuration.WithLabelValues(r.Method, route).Observe(time.Since(started).Seconds())
	})
}

func RecordModel(provider, model, status string, duration time.Duration, input, output int64) {
	if provider == "" {
		provider = "unknown"
	}
	if model == "" {
		model = "unknown"
	}
	modelCalls.WithLabelValues(provider, model, status).Inc()
	modelDuration.WithLabelValues(provider, model).Observe(duration.Seconds())
	modelTokens.WithLabelValues(provider, model, "input").Add(float64(input))
	modelTokens.WithLabelValues(provider, model, "output").Add(float64(output))
}
func RecordTool(name, status string, duration time.Duration) {
	toolCalls.WithLabelValues(name, status).Inc()
	toolDuration.WithLabelValues(name).Observe(duration.Seconds())
}
