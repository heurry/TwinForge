package realtime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	redis "github.com/redis/go-redis/v9"
)

// OutboxMessage is a committed PostgreSQL fact waiting to be fanned out.
type OutboxMessage struct {
	ID            string
	AggregateID   string
	EventType     string
	Payload       json.RawMessage
	Attempt       int
	DeliveryToken string
}

// ObservationQueueItem is the durable cursor handed to the asynchronous
// observation archiver. The event body is intentionally not copied into Redis;
// workers re-read it from PostgreSQL, keeping Redis bounded and PostgreSQL the
// source of truth.
type ObservationQueueItem struct {
	ID        string
	RunID     string
	Sequence  int64
	EventType string
}

const observationQueueGroup = "agent-observation-archive"

// OutboxStore owns delivery state. PostgreSQL remains the source of truth;
// Redis delivery is intentionally at-least-once and consumers use event seqs.
type OutboxStore interface {
	ClaimOutbox(context.Context, int, time.Duration) ([]OutboxMessage, error)
	MarkOutboxPublished(context.Context, string, string) error
	MarkOutboxFailed(context.Context, string, string, error, time.Duration) error
	PrunePublishedOutbox(context.Context, time.Time, int) (int64, error)
}

// RedisBus provides ephemeral wakeups only. Event contents are always re-read
// from PostgreSQL, so a Redis restart cannot lose Agent state.
type RedisBus struct {
	client *redis.Client
	prefix string
}

func NewRedisBus(rawURL, prefix string) (*RedisBus, error) {
	options, err := redis.ParseURL(strings.TrimSpace(rawURL))
	if err != nil {
		return nil, fmt.Errorf("parse Agent Redis URL: %w", err)
	}
	if strings.TrimSpace(prefix) == "" {
		prefix = "agent:v1:"
	}
	// Redis is a best-effort wakeup layer, never the source of truth. Bound its
	// network budget so a degraded Redis node cannot stall PostgreSQL outbox
	// delivery or Agent API shutdown for several seconds per operation.
	options.DialTimeout = time.Second
	options.ReadTimeout = time.Second
	options.WriteTimeout = time.Second
	options.PoolTimeout = 2 * time.Second
	options.MaxRetries = 1
	return &RedisBus{client: redis.NewClient(options), prefix: prefix}, nil
}

func (b *RedisBus) Ping(ctx context.Context) error { return b.client.Ping(ctx).Err() }

func (b *RedisBus) Close() error { return b.client.Close() }

func (b *RedisBus) Publish(ctx context.Context, message OutboxMessage) error {
	if strings.TrimSpace(message.AggregateID) == "" {
		return errors.New("outbox aggregate_id is required")
	}
	envelope, err := json.Marshal(map[string]string{
		"outbox_id": message.ID, "run_id": message.AggregateID, "event_type": message.EventType,
	})
	if err != nil {
		return fmt.Errorf("encode Redis wakeup: %w", err)
	}
	if err := b.client.Publish(ctx, b.runChannel(message.AggregateID), envelope).Err(); err != nil {
		return err
	}
	// The stream is a separate at-least-once delivery path for raw observation
	// archiving. A capped stream prevents Redis from becoming an archive.
	sequence := int64(0)
	var cursor struct {
		Sequence int64 `json:"sequence"`
	}
	if err := json.Unmarshal(message.Payload, &cursor); err == nil {
		sequence = cursor.Sequence
	}
	_, err = b.client.XAdd(ctx, &redis.XAddArgs{
		Stream: b.observationStream(),
		MaxLen: 100000,
		Approx: true,
		Values: map[string]any{"outbox_id": message.ID, "run_id": message.AggregateID, "sequence": sequence, "event_type": message.EventType},
	}).Result()
	return err
}

func (b *RedisBus) observationStream() string { return b.prefix + "observations:raw" }

// EnsureObservationQueue creates the consumer group idempotently.
func (b *RedisBus) EnsureObservationQueue(ctx context.Context) error {
	_, err := b.client.XGroupCreateMkStream(ctx, b.observationStream(), observationQueueGroup, "0").Result()
	if err != nil && !strings.Contains(err.Error(), "BUSYGROUP") {
		return err
	}
	return nil
}

func (b *RedisBus) ReadObservationQueue(ctx context.Context, consumer string, count int, block time.Duration) ([]ObservationQueueItem, error) {
	if count <= 0 || count > 100 {
		count = 25
	}
	if err := b.EnsureObservationQueue(ctx); err != nil {
		return nil, err
	}
	streams, err := b.client.XReadGroup(ctx, &redis.XReadGroupArgs{Group: observationQueueGroup, Consumer: consumer, Streams: []string{b.observationStream(), ">"}, Count: int64(count), Block: block}).Result()
	if err != nil && !errors.Is(err, redis.Nil) {
		return nil, err
	}
	return queueItems(streams), nil
}

// ReclaimObservationQueue makes pending entries owned by a crashed worker
// visible again after a bounded idle period.
func (b *RedisBus) ReclaimObservationQueue(ctx context.Context, consumer string, count int, minIdle time.Duration) ([]ObservationQueueItem, error) {
	if count <= 0 || count > 100 {
		count = 25
	}
	if err := b.EnsureObservationQueue(ctx); err != nil {
		return nil, err
	}
	claimed, _, err := b.client.XAutoClaim(ctx, &redis.XAutoClaimArgs{Stream: b.observationStream(), Group: observationQueueGroup, Consumer: consumer, MinIdle: minIdle, Start: "0-0", Count: int64(count)}).Result()
	if err != nil && !errors.Is(err, redis.Nil) {
		return nil, err
	}
	items := make([]ObservationQueueItem, 0, len(claimed))
	for _, item := range claimed {
		items = append(items, queueItem(item))
	}
	return items, nil
}

func (b *RedisBus) AckObservationQueue(ctx context.Context, ids ...string) error {
	if len(ids) == 0 {
		return nil
	}
	_, err := b.client.XAck(ctx, b.observationStream(), observationQueueGroup, ids...).Result()
	return err
}

func queueItems(streams []redis.XStream) []ObservationQueueItem {
	items := make([]ObservationQueueItem, 0)
	for _, stream := range streams {
		for _, message := range stream.Messages {
			items = append(items, queueItem(message))
		}
	}
	return items
}

func queueItem(message redis.XMessage) ObservationQueueItem {
	item := ObservationQueueItem{ID: message.ID}
	if value, ok := message.Values["run_id"].(string); ok {
		item.RunID = value
	}
	if value, ok := message.Values["event_type"].(string); ok {
		item.EventType = value
	}
	switch value := message.Values["sequence"].(type) {
	case int64:
		item.Sequence = value
	case string:
		_, _ = fmt.Sscan(value, &item.Sequence)
	case float64:
		item.Sequence = int64(value)
	}
	return item
}

// SubscribeRun returns a coalescing signal channel suitable for waking an SSE
// loop. The caller must still query PostgreSQL after every signal.
func (b *RedisBus) SubscribeRun(ctx context.Context, runID string) (<-chan struct{}, func(), error) {
	pubsub := b.client.Subscribe(ctx, b.runChannel(runID))
	// Receive() deliberately has no socket deadline in go-redis PubSub. Bound
	// only the subscription handshake; Channel() then uses deadline-free reads
	// so an idle subscription is not disconnected by the command ReadTimeout.
	if _, err := pubsub.ReceiveTimeout(ctx, b.client.Options().ReadTimeout); err != nil {
		_ = pubsub.Close()
		return nil, nil, err
	}
	signals := make(chan struct{}, 1)
	done := make(chan struct{})
	go func() {
		defer close(signals)
		for {
			select {
			case <-ctx.Done():
				return
			case <-done:
				return
			case _, ok := <-pubsub.Channel():
				if !ok {
					return
				}
				select {
				case signals <- struct{}{}:
				default:
				}
			}
		}
	}()
	return signals, func() {
		select {
		case <-done:
		default:
			close(done)
		}
		_ = pubsub.Close()
	}, nil
}

func (b *RedisBus) runChannel(runID string) string { return b.prefix + "run:" + runID + ":events" }

// Relay drains the transactional outbox. A failure is retried with bounded
// exponential backoff and never blocks the Agent execution transaction.
type Relay struct {
	store           OutboxStore
	bus             *RedisBus
	pollInterval    time.Duration
	batchSize       int
	pruneInterval   time.Duration
	pruneBatchSize  int
	pruneMaxBatches int
}

func NewRelay(store OutboxStore, bus *RedisBus) *Relay {
	return &Relay{
		store: store, bus: bus, pollInterval: 250 * time.Millisecond, batchSize: 25,
		pruneInterval: 5 * time.Minute, pruneBatchSize: 10000, pruneMaxBatches: 10,
	}
}

func (r *Relay) Run(ctx context.Context) error {
	ticker := time.NewTicker(r.pollInterval)
	defer ticker.Stop()
	nextPrune := time.Now()
	for {
		if err := r.dispatch(ctx); err != nil && !errors.Is(err, context.Canceled) {
			slog.Warn("Agent realtime relay degraded", "err", err)
		}
		if time.Now().After(nextPrune) {
			if err := r.prunePublished(ctx, time.Now().Add(-24*time.Hour)); err != nil && !errors.Is(err, context.Canceled) {
				slog.Warn("Agent outbox retention degraded", "err", err)
			}
			nextPrune = time.Now().Add(r.pruneInterval)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

func (r *Relay) prunePublished(ctx context.Context, before time.Time) error {
	var deleted int64
	for batch := 0; batch < r.pruneMaxBatches; batch++ {
		count, err := r.store.PrunePublishedOutbox(ctx, before, r.pruneBatchSize)
		if err != nil {
			return err
		}
		deleted += count
		if count < int64(r.pruneBatchSize) {
			return nil
		}
	}
	slog.Warn("Agent outbox retention catch-up cap reached", "deleted", deleted, "before", before)
	return nil
}

func (r *Relay) dispatch(ctx context.Context) error {
	// A batch is published serially. Keep the lease comfortably above the
	// bounded worst-case Redis timeout for the whole batch.
	messages, err := r.store.ClaimOutbox(ctx, r.batchSize, 2*time.Minute)
	if err != nil {
		return err
	}
	for _, message := range messages {
		if err := r.bus.Publish(ctx, message); err != nil {
			backoff := time.Second * time.Duration(1<<min(message.Attempt, 5))
			if markErr := r.store.MarkOutboxFailed(ctx, message.ID, message.DeliveryToken, err, backoff); markErr != nil {
				return fmt.Errorf("publish outbox: %v; mark retry: %w", err, markErr)
			}
			continue
		}
		if err := r.store.MarkOutboxPublished(ctx, message.ID, message.DeliveryToken); err != nil {
			return err
		}
	}
	return nil
}

func min(left, right int) int {
	if left < right {
		return left
	}
	return right
}
