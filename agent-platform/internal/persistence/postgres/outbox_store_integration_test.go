package postgres

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

func TestOutboxDeliveryTokenFencesStaleClaim(t *testing.T) {
	databaseURL := os.Getenv("TEST_AGENT_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("TEST_AGENT_DATABASE_URL is not set")
	}
	if err := requireDedicatedTestDatabase(databaseURL); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool, err := Open(ctx, databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	if err := Migrate(ctx, pool, "../../../migrations"); err != nil {
		t.Fatal(err)
	}
	// The dedicated database guard above makes this reset safe and ensures this
	// test controls which row ClaimOutbox sees first.
	if _, err := pool.Exec(ctx, `TRUNCATE agent_platform.agent_outbox`); err != nil {
		t.Fatal(err)
	}

	var outboxID string
	if err := pool.QueryRow(ctx, `
		INSERT INTO agent_platform.agent_outbox (
			aggregate_type,aggregate_id,event_type,payload
		) VALUES ('run',gen_random_uuid(),'TEST_EVENT','{}'::jsonb)
		RETURNING id::text`).Scan(&outboxID); err != nil {
		t.Fatal(err)
	}
	store := NewRunStore(pool)
	first, err := store.ClaimOutbox(ctx, 1, time.Minute)
	if err != nil || len(first) != 1 {
		t.Fatalf("first claim = %+v, error = %v", first, err)
	}
	if first[0].ID != outboxID || first[0].DeliveryToken == "" || first[0].Attempt != 1 {
		t.Fatalf("first claim did not return a fenced lease: %+v", first[0])
	}

	// Simulate a relay that exceeded its lease and a second replica reclaiming
	// the same row before the first replica sends its acknowledgement.
	if _, err := pool.Exec(ctx, `
		UPDATE agent_platform.agent_outbox SET available_at=now()-interval '1 second'
		WHERE id=$1::uuid`, outboxID); err != nil {
		t.Fatal(err)
	}
	second, err := store.ClaimOutbox(ctx, 1, time.Minute)
	if err != nil || len(second) != 1 {
		t.Fatalf("second claim = %+v, error = %v", second, err)
	}
	if second[0].DeliveryToken == "" || second[0].DeliveryToken == first[0].DeliveryToken || second[0].Attempt != 2 {
		t.Fatalf("reclaim did not rotate delivery token: first=%+v second=%+v", first[0], second[0])
	}

	// The schema invariant also fences an old binary during a rolling deploy:
	// its id-only acknowledgement leaves delivery_token populated and is
	// therefore rejected atomically.
	if _, err := pool.Exec(ctx, `
		UPDATE agent_platform.agent_outbox SET status='published'
		WHERE id=$1::uuid AND status='publishing'`, outboxID); err == nil {
		t.Fatal("legacy id-only acknowledgement bypassed delivery fencing")
	}
	assertOutboxClaim(t, ctx, pool, outboxID, "publishing", second[0].DeliveryToken)

	if err := store.MarkOutboxPublished(ctx, outboxID, first[0].DeliveryToken); err != nil {
		t.Fatal(err)
	}
	assertOutboxClaim(t, ctx, pool, outboxID, "publishing", second[0].DeliveryToken)
	if err := store.MarkOutboxFailed(ctx, outboxID, first[0].DeliveryToken, errors.New("stale failure"), time.Second); err != nil {
		t.Fatal(err)
	}
	assertOutboxClaim(t, ctx, pool, outboxID, "publishing", second[0].DeliveryToken)

	if err := store.MarkOutboxPublished(ctx, outboxID, second[0].DeliveryToken); err != nil {
		t.Fatal(err)
	}
	assertOutboxClaim(t, ctx, pool, outboxID, "published", "")
}

func assertOutboxClaim(t *testing.T, ctx context.Context, pool *pgxpool.Pool, id, wantStatus, wantToken string) {
	t.Helper()
	var status, token string
	if err := pool.QueryRow(ctx, `
		SELECT status,COALESCE(delivery_token::text,'')
		FROM agent_platform.agent_outbox WHERE id=$1::uuid`, id).Scan(&status, &token); err != nil {
		t.Fatal(err)
	}
	if status != wantStatus || token != wantToken {
		t.Fatalf("outbox state = status %q, token %q; want status %q, token %q", status, token, wantStatus, wantToken)
	}
}
