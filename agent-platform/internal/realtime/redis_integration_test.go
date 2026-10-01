package realtime

import (
	"context"
	"encoding/json"
	"os"
	"strconv"
	"testing"
	"time"
)

func TestRedisBusPublishesRunWakeup(t *testing.T) {
	redisURL := os.Getenv("TEST_AGENT_REDIS_URL")
	if redisURL == "" {
		t.Skip("TEST_AGENT_REDIS_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	bus, err := NewRedisBus(redisURL, "agent:test:")
	if err != nil {
		t.Fatal(err)
	}
	defer bus.Close()
	if err := bus.Ping(ctx); err != nil {
		t.Fatal(err)
	}
	runID := "run-" + strconv.FormatInt(time.Now().UnixNano(), 10)
	signals, closeSubscription, err := bus.SubscribeRun(ctx, runID)
	if err != nil {
		t.Fatal(err)
	}
	defer closeSubscription()
	// Command reads are bounded to one second, but a long-lived PubSub channel
	// must remain connected while idle. Wait through multiple ReadTimeout
	// windows before publishing the first notification.
	idle := time.NewTimer(3 * time.Second)
	defer idle.Stop()
	select {
	case _, open := <-signals:
		if !open {
			t.Fatal("Redis Run subscription closed while idle")
		}
		t.Fatal("received an unexpected Redis Run wakeup while idle")
	case <-idle.C:
	case <-ctx.Done():
		t.Fatal("timed out while verifying idle Redis Run subscription")
	}
	if err := bus.Publish(ctx, OutboxMessage{
		AggregateID: runID, EventType: "TEST_EVENT", Payload: json.RawMessage(`{"sequence":1}`),
	}); err != nil {
		t.Fatal(err)
	}
	select {
	case <-signals:
	case <-ctx.Done():
		t.Fatal("timed out waiting for Redis Run wakeup")
	}
}
