package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/harness"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/resource"
)

func TestContinueRunAfterDeadlineKeepsWorkflowCheckpoint(t *testing.T) {
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
	lock, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := lock.Exec(ctx, `SELECT pg_advisory_lock(7188617099)`); err != nil {
		t.Fatal(err)
	}
	defer func() {
		_, _ = lock.Exec(context.Background(), `SELECT pg_advisory_unlock(7188617099)`)
		lock.Release()
	}()
	if err := Migrate(ctx, pool, "../../../migrations"); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `TRUNCATE agent_platform.agent_definitions CASCADE`); err != nil {
		t.Fatal(err)
	}

	store := NewRunStore(pool)
	prompt, err := store.CreatePromptVersion(ctx, resource.CreatePromptVersion{
		TenantID: "deadline-tenant", Key: "deadline-prompt", Name: "Deadline Prompt", Content: "Continue safely.",
	})
	if err != nil {
		t.Fatal(err)
	}
	toolSet, err := store.CreateToolSetVersion(ctx, resource.CreateToolSetVersion{
		TenantID: "deadline-tenant", Key: "deadline-tools", Name: "Deadline Tools", Spec: resource.ToolSetSpec{},
	})
	if err != nil {
		t.Fatal(err)
	}
	definition, err := store.CreateDefinition(ctx, agent.CreateDefinition{TenantID: "deadline-tenant", Key: "deadline-agent", Name: "Deadline Agent"})
	if err != nil {
		t.Fatal(err)
	}
	version, err := store.CreateVersion(ctx, agent.CreateVersion{TenantID: "deadline-tenant", AgentID: definition.ID, Spec: integrationSpec(prompt, toolSet)})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.ReleaseVersion(ctx, "deadline-tenant", version.ID); err != nil {
		t.Fatal(err)
	}
	session, err := store.CreateSession(ctx, agent.CreateSession{TenantID: "deadline-tenant", AgentID: definition.ID})
	if err != nil {
		t.Fatal(err)
	}
	original, err := store.CreateRun(ctx, agent.CreateRun{
		TenantID: "deadline-tenant", SessionID: &session.ID, AgentVersionID: version.ID,
		Input: json.RawMessage(`{"question":"finish the original task"}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	claimed, err := store.ClaimNext(ctx, "deadline-worker", time.Minute)
	if err != nil || claimed.ID != original.ID {
		t.Fatalf("claim original = %+v, error=%v", claimed, err)
	}
	lease := agent.Lease{RunID: claimed.ID, Owner: "deadline-worker", Token: claimed.LeaseToken}
	checkpoint := harness.Checkpoint{
		RunID: claimed.ID, Turn: 3, NextStep: 7,
		Messages: []model.Message{model.TextMessage(model.RoleUser, "original task"), model.TextMessage(model.RoleAssistant, "working")},
	}
	if err := store.SaveCheckpointFenced(ctx, lease, checkpoint); err != nil {
		t.Fatal(err)
	}

	next, err := store.ContinueRunAfterDeadline(ctx, lease, event.Input{
		Type:    event.RunFailed,
		Payload: json.RawMessage(`{"error":"context deadline exceeded","error_code":"RUN_ATTEMPT_DEADLINE_EXCEEDED","retryable":true}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	if next.ID == original.ID || next.WorkflowID != original.WorkflowID || next.Status != agent.RunQueued || next.TriggerType != "automatic_retry" {
		t.Fatalf("automatic continuation = %+v, original=%+v", next, original)
	}
	closed, err := store.GetRun(ctx, original.ID)
	if err != nil || closed.Status != agent.RunFailed || closed.ErrorCode == nil || *closed.ErrorCode != "RUN_ATTEMPT_DEADLINE_EXCEEDED" {
		t.Fatalf("closed attempt = %+v, error=%v", closed, err)
	}
	restored, found, err := store.LoadLatestCheckpointForWorkflow(ctx, next.WorkflowID, next.ID)
	if err != nil || !found || restored.RunID != next.ID || restored.SourceRunID != original.ID || restored.Turn != 3 || restored.NextStep != 7 {
		t.Fatalf("restored checkpoint = %+v found=%v error=%v", restored, found, err)
	}
	var activeRunID, workflowStatus string
	if err := pool.QueryRow(ctx, `SELECT active_run_id::text,status FROM agent_platform.agent_workflows WHERE id=$1::uuid`, next.WorkflowID).Scan(&activeRunID, &workflowStatus); err != nil {
		t.Fatal(err)
	}
	if activeRunID != next.ID || workflowStatus != "active" {
		t.Fatalf("workflow projection = %s/%s, want %s/active", activeRunID, workflowStatus, next.ID)
	}
	var inputMessages int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM agent_platform.agent_session_messages WHERE session_id=$1::uuid AND message_kind='run_input'`, session.ID).Scan(&inputMessages); err != nil {
		t.Fatal(err)
	}
	if inputMessages != 1 {
		t.Fatalf("automatic continuation duplicated original user input: count=%d", inputMessages)
	}
	if _, err := store.ContinueRunAfterDeadline(ctx, lease, event.Input{Type: event.RunFailed}); !errors.Is(err, agent.ErrLeaseLost) {
		t.Fatalf("stale deadline continuation error=%v, want lease lost", err)
	}
}
