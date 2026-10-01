package main

import (
	"context"
	"log/slog"
	"os"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/persistence/postgres"
)

func main() {
	databaseURL := os.Getenv("AGENT_DATABASE_URL")
	if databaseURL == "" {
		slog.Error("AGENT_DATABASE_URL is required")
		os.Exit(2)
	}
	directory := os.Getenv("AGENT_MIGRATIONS_DIR")
	if directory == "" {
		directory = "migrations"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	pool, err := postgres.Open(ctx, databaseURL)
	if err != nil {
		slog.Error("connect postgres", "err", err)
		os.Exit(1)
	}
	defer pool.Close()
	if err := postgres.Migrate(ctx, pool, directory); err != nil {
		slog.Error("migrate postgres", "err", err)
		os.Exit(1)
	}
	slog.Info("agent platform migrations applied")
}
