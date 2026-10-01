package postgres

import (
	"context"
	"fmt"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/realtime"
)

// ClaimOutbox leases a bounded delivery batch using SKIP LOCKED so multiple
// API replicas may safely relay events without serializing each other.
func (s *RunStore) ClaimOutbox(ctx context.Context, limit int, lease time.Duration) ([]realtime.OutboxMessage, error) {
	if limit <= 0 || limit > 500 {
		limit = 25
	}
	if lease <= 0 {
		lease = 30 * time.Second
	}
	rows, err := s.pool.Query(ctx, `
		WITH candidates AS (
			SELECT id FROM agent_platform.agent_outbox
			WHERE (status='pending' AND available_at<=now())
			   OR (status='publishing' AND available_at<=now())
			ORDER BY created_at
			FOR UPDATE SKIP LOCKED LIMIT $1
		)
		UPDATE agent_platform.agent_outbox AS item
		SET status='publishing', attempt=item.attempt+1, available_at=now()+$2::interval,
			delivery_token=gen_random_uuid(),last_error=NULL
		FROM candidates WHERE item.id=candidates.id
		RETURNING item.id::text,item.aggregate_id::text,item.event_type,item.payload,item.attempt,
			item.delivery_token::text`,
		limit, lease.String())
	if err != nil {
		return nil, fmt.Errorf("claim Agent outbox: %w", err)
	}
	defer rows.Close()
	messages := make([]realtime.OutboxMessage, 0)
	for rows.Next() {
		var message realtime.OutboxMessage
		if err := rows.Scan(
			&message.ID, &message.AggregateID, &message.EventType, &message.Payload,
			&message.Attempt, &message.DeliveryToken,
		); err != nil {
			return nil, fmt.Errorf("scan Agent outbox: %w", err)
		}
		messages = append(messages, message)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Agent outbox: %w", err)
	}
	return messages, nil
}

func (s *RunStore) MarkOutboxPublished(ctx context.Context, id, deliveryToken string) error {
	if _, err := s.pool.Exec(ctx, `
		UPDATE agent_platform.agent_outbox
		SET status='published',published_at=now(),delivery_token=NULL,last_error=NULL
		WHERE id=$1::uuid AND delivery_token=$2::uuid AND status='publishing'`, id, deliveryToken); err != nil {
		return fmt.Errorf("ack Agent outbox: %w", err)
	}
	return nil
}

func (s *RunStore) MarkOutboxFailed(ctx context.Context, id, deliveryToken string, cause error, retryAfter time.Duration) error {
	if retryAfter <= 0 {
		retryAfter = time.Second
	}
	lastError := ""
	if cause != nil {
		lastError = cause.Error()
	}
	if _, err := s.pool.Exec(ctx, `
		UPDATE agent_platform.agent_outbox
		SET status='pending',available_at=now()+$3::interval,delivery_token=NULL,last_error=$4::text
		WHERE id=$1::uuid AND delivery_token=$2::uuid AND status='publishing'`,
		id, deliveryToken, retryAfter.String(), lastError); err != nil {
		return fmt.Errorf("retry Agent outbox: %w", err)
	}
	return nil
}

func (s *RunStore) PrunePublishedOutbox(ctx context.Context, before time.Time, limit int) (int64, error) {
	if limit <= 0 || limit > 10000 {
		limit = 1000
	}
	result, err := s.pool.Exec(ctx, `
		WITH doomed AS (
			SELECT id FROM agent_platform.agent_outbox
			WHERE status='published' AND published_at<$1
			ORDER BY published_at LIMIT $2
		)
		DELETE FROM agent_platform.agent_outbox AS item
		USING doomed WHERE item.id=doomed.id`, before, limit)
	if err != nil {
		return 0, fmt.Errorf("prune Agent outbox: %w", err)
	}
	return result.RowsAffected(), nil
}
