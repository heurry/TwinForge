// Package postgres implements durable Agent Platform persistence.
package postgres

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
)

const migrationLockID int64 = 7_188_617_012

// Open creates and verifies a PostgreSQL connection pool.
func Open(ctx context.Context, databaseURL string) (*pgxpool.Pool, error) {
	pool, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		return nil, fmt.Errorf("create postgres pool: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("ping postgres: %w", err)
	}
	return pool, nil
}

// Migrate applies pending .up.sql files in lexical order. Each migration and
// its version record commit atomically under a process-independent advisory lock.
func Migrate(ctx context.Context, pool *pgxpool.Pool, directory string) error {
	connection, err := pool.Acquire(ctx)
	if err != nil {
		return fmt.Errorf("acquire migration connection: %w", err)
	}
	defer connection.Release()
	if _, err := connection.Exec(ctx, "SELECT pg_advisory_lock($1)", migrationLockID); err != nil {
		return fmt.Errorf("acquire migration lock: %w", err)
	}
	defer func() { _, _ = connection.Exec(context.Background(), "SELECT pg_advisory_unlock($1)", migrationLockID) }()

	if _, err := connection.Exec(ctx, `
		CREATE SCHEMA IF NOT EXISTS agent_platform;
		CREATE TABLE IF NOT EXISTS agent_platform.schema_migrations (
			version VARCHAR(255) PRIMARY KEY,
			applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
		)`); err != nil {
		return fmt.Errorf("initialize migration ledger: %w", err)
	}

	entries, err := os.ReadDir(directory)
	if err != nil {
		return fmt.Errorf("read migrations directory: %w", err)
	}
	files := make([]string, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".up.sql") {
			files = append(files, entry.Name())
		}
	}
	sort.Strings(files)
	for _, name := range files {
		var applied bool
		if err := connection.QueryRow(ctx,
			`SELECT EXISTS (SELECT 1 FROM agent_platform.schema_migrations WHERE version=$1)`, name,
		).Scan(&applied); err != nil {
			return fmt.Errorf("read migration %s state: %w", name, err)
		}
		if applied {
			continue
		}
		script, err := os.ReadFile(filepath.Join(directory, name))
		if err != nil {
			return fmt.Errorf("read migration %s: %w", name, err)
		}
		tx, err := connection.Begin(ctx)
		if err != nil {
			return fmt.Errorf("begin migration %s: %w", name, err)
		}
		if _, err := tx.Exec(ctx, string(script)); err != nil {
			_ = tx.Rollback(ctx)
			return fmt.Errorf("apply migration %s: %w", name, err)
		}
		if _, err := tx.Exec(ctx,
			`INSERT INTO agent_platform.schema_migrations(version) VALUES ($1)`, name,
		); err != nil {
			_ = tx.Rollback(ctx)
			return fmt.Errorf("record migration %s: %w", name, err)
		}
		if err := tx.Commit(ctx); err != nil {
			return fmt.Errorf("commit migration %s: %w", name, err)
		}
	}
	return nil
}
