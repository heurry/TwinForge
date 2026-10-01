package realtime

import (
	"context"
	"testing"
	"time"
)

func TestNewRedisBusBoundsNetworkTimeouts(t *testing.T) {
	bus, err := NewRedisBus("redis://localhost:6379/0", "")
	if err != nil {
		t.Fatal(err)
	}
	defer bus.Close()

	options := bus.client.Options()
	if options.DialTimeout != time.Second || options.ReadTimeout != time.Second || options.WriteTimeout != time.Second {
		t.Fatalf(
			"Redis timeouts = dial %s, read %s, write %s; want 1s each",
			options.DialTimeout, options.ReadTimeout, options.WriteTimeout,
		)
	}
	if options.PoolTimeout != 2*time.Second || options.MaxRetries != 1 {
		t.Fatalf("Redis retry budget = pool %s, retries %d; want 2s and 1", options.PoolTimeout, options.MaxRetries)
	}
}

func TestRelayPruneCatchesUpInBoundedBatches(t *testing.T) {
	store := &fakeOutboxStore{pruneResults: []int64{10000, 10000, 7}}
	relay := NewRelay(store, nil)
	if err := relay.prunePublished(context.Background(), time.Now()); err != nil {
		t.Fatal(err)
	}
	if store.pruneCalls != 3 {
		t.Fatalf("prune calls = %d, want 3", store.pruneCalls)
	}
}

type fakeOutboxStore struct {
	pruneResults []int64
	pruneCalls   int
}

func (*fakeOutboxStore) ClaimOutbox(context.Context, int, time.Duration) ([]OutboxMessage, error) {
	return nil, nil
}

func (*fakeOutboxStore) MarkOutboxPublished(context.Context, string, string) error { return nil }

func (*fakeOutboxStore) MarkOutboxFailed(context.Context, string, string, error, time.Duration) error {
	return nil
}

func (s *fakeOutboxStore) PrunePublishedOutbox(context.Context, time.Time, int) (int64, error) {
	index := s.pruneCalls
	s.pruneCalls++
	if index >= len(s.pruneResults) {
		return 0, nil
	}
	return s.pruneResults[index], nil
}

func TestNewRelayUsesBoundedBatch(t *testing.T) {
	relay := NewRelay(nil, nil)
	if relay.batchSize != 25 {
		t.Fatalf("relay batch size = %d, want 25", relay.batchSize)
	}
	if relay.pruneInterval != 5*time.Minute || relay.pruneBatchSize != 10000 || relay.pruneMaxBatches != 10 {
		t.Fatalf("relay prune policy = %s/%d/%d", relay.pruneInterval, relay.pruneBatchSize, relay.pruneMaxBatches)
	}
}
