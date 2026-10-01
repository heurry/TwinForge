package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/embedding"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/harness"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/resource"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/taskplan"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

func TestRunStoreLeaseFencing(t *testing.T) {
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
	testLock, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := testLock.Exec(ctx, `SELECT pg_advisory_lock(7188617099)`); err != nil {
		t.Fatal(err)
	}
	defer func() {
		_, _ = testLock.Exec(context.Background(), `SELECT pg_advisory_unlock(7188617099)`)
		testLock.Release()
	}()
	if err := Migrate(ctx, pool, "../../../migrations"); err != nil {
		t.Fatal(err)
	}
	// Applying the same migration set twice must be a no-op.
	if err := Migrate(ctx, pool, "../../../migrations"); err != nil {
		t.Fatalf("idempotent migrate: %v", err)
	}
	// This test owns its dedicated database. Clear prior failed-test state so
	// ClaimNext cannot legitimately recover an older expired Run first.
	if _, err := pool.Exec(ctx, `TRUNCATE agent_platform.agent_definitions CASCADE`); err != nil {
		t.Fatalf("reset integration database: %v", err)
	}

	store := NewRunStore(pool)
	store.SetOutboxEnabled(true)
	prompt, err := store.CreatePromptVersion(ctx, resource.CreatePromptVersion{
		TenantID: "test-tenant", Key: "integration-prompt", Name: "Integration Prompt",
		Content: "You are an integration test agent.",
	})
	if err != nil {
		t.Fatal(err)
	}
	toolSet, err := store.CreateToolSetVersion(ctx, resource.CreateToolSetVersion{
		TenantID: "test-tenant", Key: "empty-tools", Name: "Empty Tools",
		Spec: resource.ToolSetSpec{},
	})
	if err != nil {
		t.Fatal(err)
	}
	definition, err := store.CreateDefinition(ctx, agent.CreateDefinition{
		TenantID: "test-tenant", Key: "integration-agent", Name: "Integration Agent",
	})
	if err != nil {
		t.Fatal(err)
	}
	session, err := store.CreateSession(ctx, agent.CreateSession{
		TenantID: "test-tenant", AgentID: definition.ID,
		Metadata: json.RawMessage(`{"channel":"integration"}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.GetSession(ctx, "other-tenant", session.ID); !errors.Is(err, agent.ErrSessionNotFound) {
		t.Fatalf("cross-tenant session error = %v, want not found", err)
	}
	draft, err := store.CreateVersion(ctx, agent.CreateVersion{
		TenantID: "test-tenant", AgentID: definition.ID, Spec: integrationSpec(prompt, toolSet),
	})
	if err != nil {
		t.Fatal(err)
	}
	secondDraft, err := store.CreateVersion(ctx, agent.CreateVersion{
		TenantID: "test-tenant", AgentID: definition.ID, Spec: integrationSpec(prompt, toolSet),
	})
	if err != nil || secondDraft.Version != 2 {
		t.Fatalf("second version = %+v, error = %v", secondDraft, err)
	}
	released, err := store.ReleaseVersion(ctx, "test-tenant", draft.ID)
	if err != nil || released.Status != "published" {
		t.Fatalf("released version = %+v, error = %v", released, err)
	}
	if _, err := store.ReleaseVersion(ctx, "other-tenant", draft.ID); !errors.Is(err, agent.ErrVersionNotFound) {
		t.Fatalf("cross-tenant release error = %v, want not found", err)
	}
	active, err := store.GetDefinition(ctx, "test-tenant", definition.ID)
	if err != nil || active.ActiveVersionID == nil || *active.ActiveVersionID != draft.ID {
		t.Fatalf("active definition = %+v, error = %v", active, err)
	}
	if _, err := store.CreateRun(ctx, agent.CreateRun{
		TenantID: "test-tenant", SessionID: &session.ID, AgentVersionID: secondDraft.ID,
		TriggerType: "studio_test", Input: json.RawMessage(`{"question":"draft without permission"}`),
	}); !errors.Is(err, agent.ErrRunBindingInvalid) {
		t.Fatalf("draft run without allow flag error = %v, want invalid binding", err)
	}
	versionID := draft.ID
	if _, err := store.CreateRun(ctx, agent.CreateRun{
		TenantID: "other-tenant", AgentVersionID: versionID,
		Input: json.RawMessage(`{}`),
	}); !errors.Is(err, agent.ErrRunBindingInvalid) {
		t.Fatalf("cross-tenant create error = %v, want invalid binding", err)
	}
	run, err := store.CreateRun(ctx, agent.CreateRun{
		TenantID: "test-tenant", SessionID: &session.ID, AgentVersionID: versionID,
		Input: json.RawMessage(`{"question":"hello"}`),
	})
	if err != nil {
		t.Fatal(err)
	}

	// Artifact metadata stays transactional in PostgreSQL while payload bytes
	// are stored and integrity-checked through the object store.
	objects := &integrationObjectStore{objects: map[string][]byte{}}
	store.SetArtifactObjectStore(objects)
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	artifactID, err := store.persistArtifactTx(ctx, tx, "test-tenant", run.ID, "integration-artifact", artifactWrite{Kind: "workspace_file", Name: "result.txt", MediaType: "text/plain", Content: []byte("artifact in MinIO")})
	if err != nil {
		_ = tx.Rollback(ctx)
		t.Fatal(err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	storedArtifact, artifactContent, err := store.GetArtifactContentForTenant(ctx, "test-tenant", artifactID)
	if err != nil || storedArtifact.StorageBackend != "minio" || string(artifactContent) != "artifact in MinIO" {
		t.Fatalf("object artifact = %+v/%q, error = %v", storedArtifact, artifactContent, err)
	}
	var inlineContentIsNull bool
	if err := pool.QueryRow(ctx, `SELECT content IS NULL FROM agent_platform.agent_artifacts WHERE id=$1::uuid`, artifactID).Scan(&inlineContentIsNull); err != nil || !inlineContentIsNull {
		t.Fatalf("artifact content must not remain inline: null=%v error=%v", inlineContentIsNull, err)
	}

	// Memory creation writes a 1024-dimensional pgvector value and recall uses
	// the vector path while preserving Session scope enforcement.
	store.SetMemoryEmbeddingProvider(integrationEmbeddings{model: "integration-embedding"})
	memory, err := store.CreateMemory(ctx, agent.CreateMemory{TenantID: "test-tenant", Scope: "session", SessionID: &session.ID, Content: "the release codename is helios", Importance: 0.8})
	if err != nil || memory.EmbeddingStatus != "ready" {
		t.Fatalf("vector memory = %+v, error = %v", memory, err)
	}
	recalled, err := store.RecallMemories(ctx, run, agent.MemoryPolicy{Enabled: true, ReadScopes: []string{"session"}, MaxRecall: 5, MinimumScore: 0.1}, "what is the release codename")
	if err != nil || len(recalled) == 0 || recalled[0].ID != memory.ID || recalled[0].RecallScore <= 0 {
		t.Fatalf("hybrid memory recall = %+v, error = %v", recalled, err)
	}
	store.SetMemoryEmbeddingProvider(integrationEmbeddings{model: "integration-embedding-v2"})
	reembedded, err := store.ReconcileMemoryEmbeddings(ctx, 10)
	if err != nil || reembedded.Embedded != 1 {
		t.Fatalf("embedding model rotation = %+v, error = %v", reembedded, err)
	}
	memories, err := store.ListMemories(ctx, agent.MemoryFilter{TenantID: "test-tenant", SessionID: session.ID, Limit: 10})
	if err != nil || len(memories) != 1 || memories[0].EmbeddingModel == nil || *memories[0].EmbeddingModel != "integration-embedding-v2" {
		t.Fatalf("rotated memory = %+v, error = %v", memories, err)
	}
	store.SetMemoryEmbeddingProvider(failingIntegrationEmbeddings{})
	pendingMemory, err := store.CreateMemory(ctx, agent.CreateMemory{TenantID: "test-tenant", Scope: "tenant", Content: "lexical fallback remains available", Importance: 0.5})
	if err != nil || pendingMemory.EmbeddingStatus != "pending" || pendingMemory.EmbeddingError == nil {
		t.Fatalf("degraded memory write = %+v, error = %v", pendingMemory, err)
	}
	var snapshot struct {
		AgentVersionID string     `json:"agent_version_id"`
		Spec           agent.Spec `json:"spec"`
	}
	if err := json.Unmarshal(run.BindingSnapshot, &snapshot); err != nil ||
		snapshot.AgentVersionID != versionID || snapshot.Spec.Model.ServiceRef != "test-model" {
		t.Fatalf("server binding snapshot = %s, error = %v", run.BindingSnapshot, err)
	}
	first, err := store.ClaimNext(ctx, "worker-1", 40*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	if first.ID != run.ID || first.LeaseToken != 1 {
		t.Fatalf("first claim = %+v", first)
	}
	time.Sleep(70 * time.Millisecond)
	second, err := store.ClaimNext(ctx, "worker-2", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if second.ID != run.ID || second.LeaseToken != 2 {
		t.Fatalf("second claim = %+v", second)
	}
	staleSink := NewFencedEventSink(store, agent.Lease{
		RunID: run.ID, Owner: "worker-1", Token: first.LeaseToken,
	})
	if _, err := staleSink.Append(ctx, event.Input{RunID: run.ID, Type: event.CheckpointCreated}); !errors.Is(err, agent.ErrLeaseLost) {
		t.Fatalf("stale event error = %v, want lease lost", err)
	}
	activeLease := agent.Lease{RunID: run.ID, Owner: "worker-2", Token: second.LeaseToken}
	// Delegated children share the Workflow but are not competing root attempts.
	// The root single-active guard must allow the child, and claiming/finishing
	// it must keep the Workflow cursor on the owning parent Run.
	var childRunID string
	if err := pool.QueryRow(ctx, `
		INSERT INTO agent_platform.agent_runs
			(tenant_id,workflow_id,agent_version_id,status,trigger_type,input,binding_snapshot,created_by,parent_run_id,root_run_id,delegation_depth)
		SELECT tenant_id,workflow_id,agent_version_id,'queued','delegation','{}'::jsonb,binding_snapshot,created_by,id,id,1
		FROM agent_platform.agent_runs WHERE id=$1::uuid
		RETURNING id::text`, run.ID).Scan(&childRunID); err != nil {
		t.Fatalf("active parent blocked delegated child: %v", err)
	}
	child, err := store.ClaimNext(ctx, "worker-child", time.Second)
	if err != nil || child.ID != childRunID {
		t.Fatalf("claim child = %+v, error = %v", child, err)
	}
	var childClaimWorkflowOwner string
	if err := pool.QueryRow(ctx, `SELECT active_run_id::text FROM agent_platform.agent_workflows WHERE id=$1::uuid`, run.WorkflowID).Scan(&childClaimWorkflowOwner); err != nil {
		t.Fatal(err)
	}
	if childClaimWorkflowOwner != run.ID {
		t.Fatalf("child claim replaced Workflow owner: got=%s want=%s", childClaimWorkflowOwner, run.ID)
	}
	if _, err := store.Transition(ctx, agent.Lease{RunID: child.ID, Owner: "worker-child", Token: child.LeaseToken}, agent.RunRunning, agent.RunCompleted, event.Input{Type: event.RunCompleted, Payload: json.RawMessage(`{"output":{"answer":"child"}}`)}); err != nil {
		t.Fatal(err)
	}
	createdPlan, err := store.UpsertTaskPlan(ctx, activeLease, "test-tenant", tool.Call{RunID: run.ID, Turn: 1, Step: 1, ID: "plan-call-1", Name: "update_plan"}, taskplan.Update{
		Goal: "verify adaptive planning",
		Steps: []taskplan.Step{{ID: "verify", Description: "verify persisted execution mode", Status: taskplan.StatusInProgress,
			AcceptanceCriteria: []taskplan.AcceptanceCriterion{{
				ID: "lookup", Description: "lookup returns record 7", Status: taskplan.CriterionPending,
				Verification: taskplan.VerificationSpec{
					Kind: "tool_receipt", Tool: "lookup", Arguments: json.RawMessage(`{"id":7}`),
					Assertions: []taskplan.VerificationAssertion{{Path: "name", Operator: "equals", Value: json.RawMessage(`"record-7"`)}},
				},
			}}}},
	})
	if err != nil || createdPlan.Revision != 1 {
		t.Fatalf("create task plan = %+v, error = %v", createdPlan, err)
	}
	if createdPlan.PlanID == "" {
		t.Fatal("created task plan has no stable plan_id")
	}
	planEvents, err := store.ListEventsForTenant(ctx, "test-tenant", run.ID, 0, 100)
	if err != nil {
		t.Fatal(err)
	}
	var modeSelected, planCreated bool
	for _, committed := range planEvents {
		if committed.WorkflowID != run.WorkflowID {
			t.Fatalf("event workflow projection=%s want=%s event=%s", committed.WorkflowID, run.WorkflowID, committed.Type)
		}
		if committed.Type == event.ExecutionModeSelected {
			var payload map[string]any
			if err := json.Unmarshal(committed.Payload, &payload); err != nil {
				t.Fatal(err)
			}
			modeSelected = payload["mode"] == "planned" && payload["policy"] == "required"
		}
		planCreated = planCreated || committed.Type == event.PlanCreated
	}
	if !modeSelected || !planCreated {
		t.Fatalf("plan creation events missing: mode=%v plan=%v events=%+v", modeSelected, planCreated, planEvents)
	}
	resolution := agent.ModelResolution{
		SelectionPolicy: agent.ModelSelectionAuto, Provider: "vllm", ServiceRef: "primary",
		ModelID: "resolved-model", ModelVersion: "r1", ArtifactDigest: "sha256:abc",
		ServiceConfigHash: "sha256:service", DiscoveredAt: time.Now().UTC(),
	}
	frozen, err := store.FreezeModelResolution(ctx, activeLease, resolution)
	if err != nil || frozen.ModelID != "resolved-model" {
		t.Fatalf("freeze resolution = %+v, error = %v", frozen, err)
	}
	other := resolution
	other.ModelID = "must-not-replace"
	frozen, err = store.FreezeModelResolution(ctx, activeLease, other)
	if err != nil || frozen.ModelID != "resolved-model" {
		t.Fatalf("second freeze replaced resolution: %+v, error = %v", frozen, err)
	}
	resolvedRun, err := store.GetRun(ctx, run.ID)
	if err != nil || resolvedRun.ModelResolution == nil || resolvedRun.ModelResolution.ModelVersion != "r1" {
		t.Fatalf("hydrated model resolution = %+v, error = %v", resolvedRun.ModelResolution, err)
	}
	if _, err := store.FreezeModelResolution(ctx, agent.Lease{
		RunID: run.ID, Owner: "worker-1", Token: first.LeaseToken,
	}, resolution); !errors.Is(err, agent.ErrLeaseLost) {
		t.Fatalf("stale resolution error = %v, want lease lost", err)
	}
	checkpoint := harness.Checkpoint{
		RunID: run.ID, Turn: 1, NextStep: 2,
		Messages: []model.Message{{Role: model.RoleUser, Content: "resume me"}},
	}
	if err := store.SaveCheckpointFenced(ctx, activeLease, checkpoint); err != nil {
		t.Fatal(err)
	}
	var currentStateCount, historicalCheckpointCount int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM agent_platform.agent_run_states WHERE run_id=$1::uuid`, run.ID).Scan(&currentStateCount); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM agent_platform.agent_checkpoints WHERE run_id=$1::uuid`, run.ID).Scan(&historicalCheckpointCount); err != nil {
		t.Fatal(err)
	}
	if currentStateCount != 1 || historicalCheckpointCount != 0 {
		t.Fatalf("checkpoint projection: current=%d historical=%d, want current=1 historical=0", currentStateCount, historicalCheckpointCount)
	}
	loaded, found, err := store.LoadLatestCheckpoint(ctx, run.ID)
	if err != nil || !found || loaded.NextStep != 2 || len(loaded.Messages) != 1 {
		t.Fatalf("loaded checkpoint = %+v, found = %v, error = %v", loaded, found, err)
	}
	if err := store.SaveCheckpointFenced(ctx, agent.Lease{
		RunID: run.ID, Owner: "worker-1", Token: first.LeaseToken,
	}, checkpoint); !errors.Is(err, agent.ErrLeaseLost) {
		t.Fatalf("stale checkpoint error = %v, want lease lost", err)
	}
	handlerCalls := 0
	call := tool.Call{
		RunID: run.ID, Turn: 1, Step: 1, ID: "idempotent-call", Name: "lookup", PlanStepID: "verify",
		Arguments: json.RawMessage(`{"id":7}`),
	}
	handler := func(context.Context, tool.Call) (tool.Result, error) {
		handlerCalls++
		return tool.Result{Content: json.RawMessage(`{"name":"record-7"}`)}, nil
	}
	for index := 0; index < 2; index++ {
		result, err := store.ExecuteToolIdempotent(
			ctx, activeLease, "test-tenant", "tool-version-1", "http", call, handler,
		)
		var content map[string]string
		decodeErr := json.Unmarshal(result.Content, &content)
		if err != nil || decodeErr != nil || content["name"] != "record-7" {
			t.Fatalf("idempotent result %d = %+v, error = %v", index, result, err)
		}
	}
	if handlerCalls != 1 {
		t.Fatalf("tool handler calls = %d, want 1", handlerCalls)
	}
	var storedPlanStep string
	if err := pool.QueryRow(ctx, `SELECT COALESCE(plan_step_key,'') FROM agent_platform.agent_tool_executions WHERE run_id=$1::uuid AND call_id=$2`, run.ID, call.ID).Scan(&storedPlanStep); err != nil || storedPlanStep != "verify" {
		t.Fatalf("stored plan step = %q, error = %v", storedPlanStep, err)
	}
	var storedWorkflowID string
	if err := pool.QueryRow(ctx, `SELECT workflow_id::text FROM agent_platform.agent_tool_executions WHERE run_id=$1::uuid AND call_id=$2`, run.ID, call.ID).Scan(&storedWorkflowID); err != nil || storedWorkflowID != run.WorkflowID {
		t.Fatalf("stored workflow id = %q, want %q, error = %v", storedWorkflowID, run.WorkflowID, err)
	}
	activeSink := NewFencedEventSink(store, activeLease)
	if _, err := activeSink.Append(ctx, event.Input{
		RunID: run.ID, Type: event.ToolCompleted, Turn: 1, Step: 1, CallID: call.ID, PlanNodeID: "verify",
		Payload: json.RawMessage(`{"name":"lookup","arguments":{"id":7},"result":{"content":{"name":"record-7"}}}`),
	}); err != nil {
		t.Fatal(err)
	}
	var actionStatus, actionKind, actionNode string
	var actionCycle int
	if err := pool.QueryRow(ctx, `
		SELECT status,action_kind,COALESCE(plan_node_id,''),COALESCE(decision_cycle,0)
		FROM agent_platform.agent_action_attempts
		WHERE run_id=$1::uuid AND action_id=$2`, run.ID, call.ID).Scan(
		&actionStatus, &actionKind, &actionNode, &actionCycle,
	); err != nil {
		t.Fatal(err)
	}
	if actionStatus != "completed" || actionKind != "tool" || actionNode != "verify" || actionCycle != 1 {
		t.Fatalf("action projection status=%s kind=%s node=%s cycle=%d", actionStatus, actionKind, actionNode, actionCycle)
	}
	var cycleStatus string
	var projectedActions int
	if err := pool.QueryRow(ctx, `
		SELECT status,action_count FROM agent_platform.agent_decision_cycles
		WHERE run_id=$1::uuid AND cycle_no=1`, run.ID).Scan(&cycleStatus, &projectedActions); err != nil {
		t.Fatal(err)
	}
	if cycleStatus != "running" {
		t.Fatalf("decision cycle status=%s actions=%d", cycleStatus, projectedActions)
	}
	hydratedPlan, err := store.GetTaskPlanForWorkflow(ctx, "test-tenant", run.WorkflowID)
	if err != nil || len(hydratedPlan.Steps) != 1 || hydratedPlan.Steps[0].State.LastRunID != run.ID || hydratedPlan.Steps[0].State.LastEventSeq == 0 || hydratedPlan.Steps[0].State.Attempts != 1 {
		t.Fatalf("hydrated plan node state = %+v, error = %v", hydratedPlan.Steps, err)
	}
	firstPlanNodeEventSequence := hydratedPlan.Steps[0].State.LastEventSeq
	if err := store.ValidateSuccessfulToolEvidence(ctx, "test-tenant", run.ID, []string{call.ID}); err != nil {
		t.Fatalf("successful same-Run Tool evidence was rejected: %v", err)
	}
	criterion := createdPlan.Steps[0].AcceptanceCriteria[0]
	resolvedCriterion, err := store.ResolveAcceptanceCriterionEvidence(ctx, "test-tenant", run.ID, "verify", criterion)
	if err != nil || resolvedCriterion.Status != taskplan.CriterionPassed || len(resolvedCriterion.EvidenceCallIDs) != 1 || resolvedCriterion.EvidenceCallIDs[0] != call.ID {
		t.Fatalf("platform-managed evidence = %+v, error = %v", resolvedCriterion, err)
	}
	// A newer successful receipt from the same tool/Plan step but different
	// arguments is selection noise, not a failed verification attempt.
	unrelatedCall := tool.Call{
		RunID: run.ID, Turn: 1, Step: 2, ID: "unrelated-lookup", Name: "lookup", PlanStepID: "verify",
		Arguments: json.RawMessage(`{"id":8}`),
	}
	if _, err := store.ExecuteToolIdempotent(ctx, activeLease, "test-tenant", "tool-version-1", "http", unrelatedCall, func(context.Context, tool.Call) (tool.Result, error) {
		return tool.Result{Content: json.RawMessage(`{"name":"record-8"}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := activeSink.Append(ctx, event.Input{
		RunID: run.ID, Type: event.ToolCompleted, Turn: 1, Step: 2, CallID: unrelatedCall.ID, PlanNodeID: "verify",
		Payload: json.RawMessage(`{"name":"lookup","arguments":{"id":8},"result":{"content":{"name":"record-8"}}}`),
	}); err != nil {
		t.Fatal(err)
	}
	resolvedAfterNoise, err := store.ResolveAcceptanceCriterionEvidence(ctx, "test-tenant", run.ID, "verify", criterion)
	if err != nil || len(resolvedAfterNoise.EvidenceCallIDs) != 1 || resolvedAfterNoise.EvidenceCallIDs[0] != call.ID {
		t.Fatalf("unrelated receipt displaced exact evidence: %+v, error=%v", resolvedAfterNoise, err)
	}
	var failedVerificationAttempts int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM agent_platform.agent_verification_attempts WHERE run_id=$1::uuid AND status='failed'`, run.ID).Scan(&failedVerificationAttempts); err != nil {
		t.Fatal(err)
	}
	if failedVerificationAttempts != 0 {
		t.Fatalf("unrelated receipt created %d failed verification attempt(s)", failedVerificationAttempts)
	}
	// Simulate a rolling-upgrade writer that predates plan_step_key. Because the
	// call happened after this Todo became active, bounded compatibility still
	// discovers it without trusting a model-supplied call ID.
	if _, err := pool.Exec(ctx, `UPDATE agent_platform.agent_tool_executions SET plan_step_key=NULL WHERE run_id=$1::uuid AND call_id=$2`, run.ID, call.ID); err != nil {
		t.Fatal(err)
	}
	resolvedLegacyCriterion, err := store.ResolveAcceptanceCriterionEvidence(ctx, "test-tenant", run.ID, "verify", criterion)
	if err != nil || len(resolvedLegacyCriterion.EvidenceCallIDs) != 1 || resolvedLegacyCriterion.EvidenceCallIDs[0] != call.ID {
		t.Fatalf("legacy platform-managed evidence = %+v, error = %v", resolvedLegacyCriterion, err)
	}
	if err := store.ValidateSuccessfulToolEvidence(ctx, "test-tenant", run.ID, []string{"missing-call"}); err == nil {
		t.Fatal("missing Tool evidence receipt was accepted")
	}
	takeoverCall := tool.Call{
		RunID: run.ID, Turn: 1, Step: 1, ID: "takeover-call", Name: "lookup",
		Arguments: json.RawMessage(`{"id":8}`),
	}
	takeoverHash, takeoverRequest, err := hashToolRequest("tool-version-1", takeoverCall)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
		INSERT INTO agent_platform.agent_tool_executions (
			tenant_id, run_id, call_id, tool_name, tool_version, provider,
			request_hash, idempotency_key, status, attempt, request,
			started_at, lease_owner, lease_token
		) VALUES ('test-tenant', $1::uuid, $2, 'lookup', 'tool-version-1', 'http',
			$3, $4, 'running', 1, $5::jsonb, now(), 'worker-1', $6)`,
		run.ID, takeoverCall.ID, takeoverHash, run.ID+":"+takeoverCall.ID,
		takeoverRequest, first.LeaseToken); err != nil {
		t.Fatal(err)
	}
	takeoverCalls := 0
	if _, err := store.ExecuteToolIdempotent(
		ctx, activeLease, "test-tenant", "tool-version-1", "http", takeoverCall,
		func(context.Context, tool.Call) (tool.Result, error) {
			takeoverCalls++
			return tool.Result{Content: json.RawMessage(`{"name":"record-8"}`)}, nil
		},
	); err != nil || takeoverCalls != 1 {
		t.Fatalf("tool takeover calls = %d, error = %v", takeoverCalls, err)
	}

	_, err = store.Transition(ctx,
		agent.Lease{RunID: run.ID, Owner: "worker-1", Token: first.LeaseToken},
		agent.RunRunning, agent.RunCompleted, event.Input{Type: event.RunCompleted},
	)
	if !errors.Is(err, agent.ErrLeaseLost) {
		t.Fatalf("stale transition error = %v, want lease lost", err)
	}
	completed, err := store.Transition(ctx,
		agent.Lease{RunID: run.ID, Owner: "worker-2", Token: second.LeaseToken},
		agent.RunRunning, agent.RunCompleted,
		event.Input{Type: event.RunCompleted, Payload: json.RawMessage(`{"output":{"answer":"ok"}}`)},
	)
	if err != nil {
		t.Fatal(err)
	}
	if completed.Status != agent.RunCompleted || string(completed.Output) != `{"answer": "ok"}` {
		// PostgreSQL may normalize JSON whitespace, so verify semantic content below.
		var output map[string]string
		if err := json.Unmarshal(completed.Output, &output); err != nil || output["answer"] != "ok" {
			t.Fatalf("completed run output = %s, error = %v", completed.Output, err)
		}
	}
	if _, err := store.GetRunForTenant(ctx, "other-tenant", run.ID); !errors.Is(err, agent.ErrRunNotFound) {
		t.Fatalf("cross-tenant get error = %v, want not found", err)
	}
	if err := store.RequestCancelForTenant(ctx, "test-tenant", run.ID, "tester"); !errors.Is(err, agent.ErrRunTerminal) {
		t.Fatalf("terminal cancel error = %v, want terminal", err)
	}

	rows, err := pool.Query(ctx, `
		SELECT seq, event_type FROM agent_platform.agent_events
		WHERE run_id=$1::uuid ORDER BY seq`, run.ID)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	var sequence int64
	var eventTypes []string
	for rows.Next() {
		var seq int64
		var eventType string
		if err := rows.Scan(&seq, &eventType); err != nil {
			t.Fatal(err)
		}
		sequence++
		if seq != sequence {
			t.Fatalf("event seq = %d, want %d", seq, sequence)
		}
		eventTypes = append(eventTypes, eventType)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	want := []string{
		string(event.RunCreated), string(event.WorkflowCreated), string(event.RunAttemptCreated), string(event.WorkflowRoutingDecided), string(event.RunClaimed), string(event.WorkflowResumed), string(event.RunResumed),
		string(event.ExecutionModeSelected), string(event.PlanCreated), string(event.VerificationIntentCreated), string(event.VerificationSpecCompiled),
		string(event.ModelResolved), string(event.CheckpointCreated), string(event.ToolCompleted), string(event.ToolCompleted),
		// The Run completed, but its Workflow-owned Plan is still open, so the
		// Workflow must not emit WORKFLOW_COMPLETED.
		string(event.RunCompleted),
	}
	if len(eventTypes) != len(want) {
		t.Fatalf("events = %v, want %v", eventTypes, want)
	}
	for index := range want {
		if eventTypes[index] != want[index] {
			t.Fatalf("events = %v, want %v", eventTypes, want)
		}
	}
	timeline, err := store.ListEventsForTenant(ctx, "test-tenant", run.ID, 2, 10)
	if err != nil || len(timeline) == 0 || timeline[0].Sequence != 3 {
		t.Fatalf("timeline after 2 = %+v, error = %v", timeline, err)
	}
	if _, err := store.ListEventsForTenant(ctx, "other-tenant", run.ID, 0, 10); !errors.Is(err, agent.ErrRunNotFound) {
		t.Fatalf("cross-tenant timeline error = %v, want not found", err)
	}
	history, err := store.ListSessionMessages(
		ctx, "test-tenant", session.ID, "00000000-0000-4000-8000-000000000001", 10,
	)
	if err != nil || len(history) != 2 || history[0].Role != model.RoleUser || history[1].Role != model.RoleAssistant {
		t.Fatalf("session history = %+v, error = %v", history, err)
	}
	sessions, err := store.ListSessions(ctx, "test-tenant", definition.ID, 10)
	if err != nil || len(sessions) != 1 || sessions[0].RunCount != 1 || sessions[0].MessageCount != 2 {
		t.Fatalf("session counters = %+v, error = %v", sessions, err)
	}
	storage, err := store.StorageSummary(ctx, "test-tenant")
	if err != nil || storage.PostgreSQL.Status != "ready" || storage.PostgreSQL.Metrics["messages"] != 2 || storage.PostgreSQL.Metrics["pending_outbox"] == 0 {
		t.Fatalf("storage summary = %+v, error = %v", storage, err)
	}
	testRun, err := store.CreateRun(ctx, agent.CreateRun{
		TenantID: "test-tenant", SessionID: &session.ID, AgentVersionID: secondDraft.ID,
		TriggerType: "studio_test", AllowDraft: true, Input: json.RawMessage(`{"question":"verify draft"}`),
	})
	if err != nil || testRun.TriggerType != "studio_test" {
		t.Fatalf("allowed draft test run = %+v, error = %v", testRun, err)
	}
	// A continuation must advance the Workflow-level checkpoint sequence even
	// though the new Run Attempt starts its own event sequence at one.
	if _, err := pool.Exec(ctx, `UPDATE agent_platform.agent_runs SET status='cancelled',finished_at=now(),updated_at=now() WHERE id=$1::uuid`, testRun.ID); err != nil {
		t.Fatal(err)
	}
	var originalStateSeq int64
	if err := pool.QueryRow(ctx, `SELECT state_seq FROM agent_platform.agent_run_states WHERE workflow_id=$1::uuid`, run.WorkflowID).Scan(&originalStateSeq); err != nil {
		t.Fatal(err)
	}
	workflowID := run.WorkflowID
	continuation, err := store.CreateRun(ctx, agent.CreateRun{
		TenantID: "test-tenant", SessionID: &session.ID, WorkflowID: &workflowID,
		RoutingIntent: "resume", AgentVersionID: versionID,
		Input: json.RawMessage(`{"question":"continue checkpoint"}`),
	})
	if err != nil {
		t.Fatalf("create continuation = %+v, error = %v", continuation, err)
	}
	var resumedWorkflowStatus, resumedActiveRunID string
	if err := pool.QueryRow(ctx, `SELECT status,active_run_id::text FROM agent_platform.agent_workflows WHERE id=$1::uuid`, workflowID).Scan(&resumedWorkflowStatus, &resumedActiveRunID); err != nil {
		t.Fatal(err)
	}
	if resumedWorkflowStatus != "active" || resumedActiveRunID != continuation.ID {
		t.Fatalf("queued continuation did not reactivate Workflow: status=%s active_run_id=%s", resumedWorkflowStatus, resumedActiveRunID)
	}
	continuationLease, err := store.ClaimNext(ctx, "worker-continuation", time.Second)
	if err != nil || continuationLease.ID != continuation.ID {
		t.Fatalf("claim continuation = %+v, error = %v", continuationLease, err)
	}
	// A continuation creates a new Run attempt under the same Workflow. The
	// Plan identity and creating Run remain stable while the current revision
	// records the continuation as its modification source; this previously violated
	// fk_agent_workflows_active_plan.
	continuedPlan, err := store.UpsertTaskPlan(ctx, agent.Lease{RunID: continuation.ID, Owner: "worker-continuation", Token: continuationLease.LeaseToken}, "test-tenant", tool.Call{
		RunID: continuation.ID, WorkflowID: workflowID, Turn: 1, Step: 1, ID: "plan-call-continuation", Name: "update_plan_step",
	}, taskplan.Update{Goal: "continued plan update", Steps: createdPlan.Steps})
	if err != nil {
		t.Fatalf("update plan from continuation run: %v", err)
	}
	if continuedPlan.PlanID != createdPlan.PlanID || continuedPlan.RunID != createdPlan.RunID || continuedPlan.LastModifiedRunID != continuation.ID {
		t.Fatalf("continuation plan identity = %+v, want stable plan_id=%s creator_run=%s modifier_run=%s", continuedPlan, createdPlan.PlanID, createdPlan.RunID, continuation.ID)
	}
	// Event sequence is Run-local. The first node observation in a continuation
	// must replace the old Run's sequence rather than compare unrelated counters.
	continuationSink := NewFencedEventSink(store, agent.Lease{RunID: continuation.ID, Owner: "worker-continuation", Token: continuationLease.LeaseToken})
	continuationNodeEvent, err := continuationSink.Append(ctx, event.Input{
		RunID: continuation.ID, WorkflowID: workflowID, Type: event.ToolCompleted,
		Turn: 1, Step: 2, CallID: "continuation-node-event", PlanNodeID: "verify",
		Payload: json.RawMessage(`{"name":"lookup","arguments":{"id":9},"result":{"content":{"name":"record-9"}}}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	continuedHydratedPlan, err := store.GetTaskPlanForWorkflow(ctx, "test-tenant", workflowID)
	if err != nil || len(continuedHydratedPlan.Steps) != 1 {
		t.Fatalf("continued hydrated plan = %+v, error = %v", continuedHydratedPlan, err)
	}
	continuedNodeState := continuedHydratedPlan.Steps[0].State
	if continuedNodeState.LastRunID != continuation.ID || continuedNodeState.LastEventSeq != continuationNodeEvent.Sequence || continuedNodeState.Attempts != 2 {
		t.Fatalf("continued node cursor = %+v, want run=%s seq=%d attempts=2 (old seq=%d)", continuedNodeState, continuation.ID, continuationNodeEvent.Sequence, firstPlanNodeEventSequence)
	}
	if _, err := continuationSink.Append(ctx, event.Input{
		RunID: continuation.ID, WorkflowID: workflowID, Type: event.ToolCompleted,
		Turn: 1, Step: 3, CallID: "unknown-node-event", PlanNodeID: "unknown-node",
		Payload: json.RawMessage(`{"name":"lookup","result":{"content":{"name":"ignored"}}}`),
	}); err != nil {
		t.Fatal(err)
	}
	var unknownNodeCount int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM agent_platform.agent_plan_node_states WHERE workflow_id=$1::uuid AND node_id='unknown-node'`, workflowID).Scan(&unknownNodeCount); err != nil || unknownNodeCount != 0 {
		t.Fatalf("unknown event-created Plan nodes = %d, error = %v", unknownNodeCount, err)
	}
	var activePlanID string
	if err := pool.QueryRow(ctx, `SELECT active_plan_id::text FROM agent_platform.agent_workflows WHERE id=$1::uuid`, workflowID).Scan(&activePlanID); err != nil {
		t.Fatal(err)
	}
	if activePlanID != createdPlan.PlanID {
		t.Fatalf("workflow active_plan_id = %s, want stable plan_id %s", activePlanID, createdPlan.PlanID)
	}
	workflowTimeline, err := store.ListWorkflowEventsForTenant(ctx, "test-tenant", workflowID, 0, 1000)
	if err != nil {
		t.Fatal(err)
	}
	seenOriginal, seenContinuation := false, false
	for index, committed := range workflowTimeline {
		if committed.WorkflowSequence != int64(index+1) {
			t.Fatalf("workflow sequence at %d = %d", index, committed.WorkflowSequence)
		}
		seenOriginal = seenOriginal || committed.RunID == run.ID
		seenContinuation = seenContinuation || committed.RunID == continuation.ID
	}
	if !seenOriginal || !seenContinuation {
		t.Fatalf("workflow timeline did not span attempts: original=%v continuation=%v", seenOriginal, seenContinuation)
	}
	var latestWorkflowSequence int64
	if err := pool.QueryRow(ctx, `SELECT latest_workflow_seq FROM agent_platform.agent_workflows WHERE id=$1::uuid`, workflowID).Scan(&latestWorkflowSequence); err != nil {
		t.Fatal(err)
	}
	if len(workflowTimeline) == 0 || workflowTimeline[len(workflowTimeline)-1].WorkflowSequence != latestWorkflowSequence {
		t.Fatalf("workflow cursor=%d timeline tail=%+v", latestWorkflowSequence, workflowTimeline)
	}
	continuationCheckpoint := harness.Checkpoint{RunID: continuation.ID, Turn: 2, NextStep: 4, Messages: []model.Message{{Role: model.RoleUser, Content: "continue checkpoint"}}}
	if err := store.SaveCheckpointFenced(ctx, agent.Lease{RunID: continuation.ID, Owner: "worker-continuation", Token: continuationLease.LeaseToken}, continuationCheckpoint); err != nil {
		t.Fatal(err)
	}
	var continuationStateSeq int64
	if err := pool.QueryRow(ctx, `SELECT state_seq FROM agent_platform.agent_run_states WHERE workflow_id=$1::uuid`, workflowID).Scan(&continuationStateSeq); err != nil || continuationStateSeq <= originalStateSeq {
		t.Fatalf("continuation state_seq=%d original=%d error=%v", continuationStateSeq, originalStateSeq, err)
	}
	loadedContinuation, found, err := store.LoadLatestCheckpointForWorkflow(ctx, workflowID, continuation.ID)
	if err != nil || !found || loadedContinuation.RunID != continuation.ID || loadedContinuation.NextStep != 4 {
		t.Fatalf("workflow checkpoint = %+v found=%v error=%v", loadedContinuation, found, err)
	}
	if _, err := pool.Exec(ctx, `UPDATE agent_platform.agent_runs SET status='cancelled',finished_at=now(),updated_at=now() WHERE id=$1::uuid`, continuation.ID); err != nil {
		t.Fatal(err)
	}

	// Simulate a rolling-release gap: an older writer committed canonical Run
	// facts without the explicit Session projection. A newer Run is already
	// projected, so repair must restore facts in chronological order rather
	// than append the older messages at the tail.
	if _, err := pool.Exec(ctx, `
		DELETE FROM agent_platform.agent_session_messages
		WHERE run_id=$1::uuid`, run.ID); err != nil {
		t.Fatal(err)
	}
	repairedHistory, err := store.ListSessionMessages(
		ctx, "test-tenant", session.ID, "00000000-0000-4000-8000-000000000001", 10,
	)
	if err != nil || len(repairedHistory) != 4 ||
		repairedHistory[0].Role != model.RoleUser || repairedHistory[1].Role != model.RoleAssistant ||
		repairedHistory[2].Role != model.RoleUser || repairedHistory[3].Role != model.RoleUser {
		t.Fatalf("read-repaired session history = %+v, error = %v", repairedHistory, err)
	}
	// Remove the same projection once more so the explicit reconciliation API
	// and its inserted-row accounting are covered independently of read-repair.
	if _, err := pool.Exec(ctx, `
		DELETE FROM agent_platform.agent_session_messages
		WHERE run_id=$1::uuid`, run.ID); err != nil {
		t.Fatal(err)
	}
	inserted, err := store.ReconcileSessionMessages(ctx, "test-tenant", session.ID)
	if err != nil || inserted != 2 {
		t.Fatalf("session message repair inserted = %d, error = %v, want 2", inserted, err)
	}
	rows, err = pool.Query(ctx, `
		SELECT run_id::text, message_kind, sequence
		FROM agent_platform.agent_session_messages
		WHERE session_id=$1::uuid ORDER BY sequence`, session.ID)
	if err != nil {
		t.Fatal(err)
	}
	var repaired []string
	var expectedSequence int64
	for rows.Next() {
		var repairedRunID, kind string
		var messageSequence int64
		if err := rows.Scan(&repairedRunID, &kind, &messageSequence); err != nil {
			rows.Close()
			t.Fatal(err)
		}
		expectedSequence++
		if messageSequence != expectedSequence {
			rows.Close()
			t.Fatalf("repaired message sequence = %d, want %d", messageSequence, expectedSequence)
		}
		repaired = append(repaired, repairedRunID+":"+kind)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		t.Fatal(err)
	}
	rows.Close()
	wantRepaired := []string{
		run.ID + ":run_input", run.ID + ":run_output", testRun.ID + ":run_input", continuation.ID + ":run_input",
	}
	if len(repaired) != len(wantRepaired) {
		t.Fatalf("repaired messages = %v, want %v", repaired, wantRepaired)
	}
	for index := range wantRepaired {
		if repaired[index] != wantRepaired[index] {
			t.Fatalf("repaired messages = %v, want %v", repaired, wantRepaired)
		}
	}
	if inserted, err := store.ReconcileSessionMessages(ctx, "test-tenant", session.ID); err != nil || inserted != 0 {
		t.Fatalf("idempotent session repair inserted = %d, error = %v", inserted, err)
	}
	if _, err := store.ReconcileSessionMessages(ctx, "other-tenant", session.ID); !errors.Is(err, agent.ErrSessionNotFound) {
		t.Fatalf("cross-tenant session repair error = %v, want not found", err)
	}

	// A rolling old writer can append a legacy checkpoint after the one-shot
	// latest-state backfill. Recovery must choose the higher event_seq before
	// periodic projection reconciliation has run.
	legacyCheckpoint := checkpoint
	legacyCheckpoint.NextStep = 91
	legacyState, err := json.Marshal(legacyCheckpoint)
	if err != nil {
		t.Fatal(err)
	}
	const legacyEventSequence int64 = 9001
	if _, err := pool.Exec(ctx, `
		INSERT INTO agent_platform.agent_checkpoints
			(run_id,workflow_id,event_seq,lease_token,state,context_summary)
		SELECT $1::uuid,workflow_id,$2,$3,$4::jsonb,'rolling release checkpoint'
		FROM agent_platform.agent_runs WHERE id=$1::uuid`,
		run.ID, legacyEventSequence, second.LeaseToken, legacyState); err != nil {
		t.Fatal(err)
	}
	loaded, found, err = store.LoadLatestCheckpoint(ctx, run.ID)
	if err != nil || !found || loaded.NextStep != legacyCheckpoint.NextStep {
		t.Fatalf("legacy-newer checkpoint = %+v, found = %v, error = %v", loaded, found, err)
	}
	reconciled, err := store.ReconcileStorageFoundation(ctx)
	if err != nil || reconciled.RunStatesUpserted != 1 || reconciled.SessionMessagesInserted != 0 {
		t.Fatalf("storage reconciliation = %+v, error = %v", reconciled, err)
	}
	var projectedRunID string
	var projectedEventSequence int64
	if err := pool.QueryRow(ctx, `
		SELECT run_id::text,event_seq FROM agent_platform.agent_run_states
		WHERE workflow_id=$1::uuid`, run.WorkflowID).Scan(&projectedRunID, &projectedEventSequence); err != nil {
		t.Fatal(err)
	}
	if projectedRunID != run.ID || projectedEventSequence != legacyEventSequence {
		t.Fatalf("projected run/event sequence = %s/%d, want %s/%d", projectedRunID, projectedEventSequence, run.ID, legacyEventSequence)
	}
	reconciled, err = store.ReconcileStorageFoundation(ctx)
	if err != nil || reconciled.RunStatesUpserted != 0 || reconciled.SessionMessagesInserted != 0 {
		t.Fatalf("idempotent storage reconciliation = %+v, error = %v", reconciled, err)
	}
	// Auto routing must stop rather than silently bind the newest task when a
	// Session contains multiple resumable Workflows.
	ambiguousA, err := store.CreateRun(ctx, agent.CreateRun{
		TenantID: "test-tenant", SessionID: &session.ID, AgentVersionID: versionID,
		NewWorkflow: true, RoutingIntent: "new_workflow", Input: json.RawMessage(`{"question":"ambiguous-a"}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	ambiguousB, err := store.CreateRun(ctx, agent.CreateRun{
		TenantID: "test-tenant", SessionID: &session.ID, AgentVersionID: versionID,
		NewWorkflow: true, RoutingIntent: "new_workflow", Input: json.RawMessage(`{"question":"ambiguous-b"}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.CreateRun(ctx, agent.CreateRun{
		TenantID: "test-tenant", SessionID: &session.ID, AgentVersionID: versionID,
		RoutingIntent: "auto", Input: json.RawMessage(`{"question":"continue an unspecified task"}`),
	}); err == nil {
		t.Fatal("auto routing silently selected one of multiple Workflows")
	} else {
		var ambiguous *agent.WorkflowAmbiguousError
		if !errors.As(err, &ambiguous) || len(ambiguous.Candidates) < 2 {
			t.Fatalf("ambiguous routing error = %v, want structured candidates", err)
		}
	}
	if _, err := pool.Exec(ctx, `UPDATE agent_platform.agent_runs SET status='cancelled',finished_at=now(),updated_at=now() WHERE id=ANY($1::uuid[])`, []string{ambiguousA.ID, ambiguousB.ID}); err != nil {
		t.Fatal(err)
	}

	// A store with no Redis configuration must not create undrainable outbox
	// rows. Concurrent Run creation in one Session must also serialize message
	// sequence allocation without row-lock upgrade deadlocks.
	disabledStore := NewRunStore(pool)
	var outboxBefore int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM agent_platform.agent_outbox`).Scan(&outboxBefore); err != nil {
		t.Fatal(err)
	}
	const concurrentRuns = 8
	type createResult struct {
		run agent.Run
		err error
	}
	results := make(chan createResult, concurrentRuns)
	for index := 0; index < concurrentRuns; index++ {
		go func(index int) {
			created, createErr := disabledStore.CreateRun(ctx, agent.CreateRun{
				TenantID: "test-tenant", SessionID: &session.ID, AgentVersionID: versionID,
				Input:       mustJSON(map[string]any{"question": fmt.Sprintf("parallel-%d", index)}),
				NewWorkflow: true, RoutingIntent: "new_workflow",
			})
			results <- createResult{run: created, err: createErr}
		}(index)
	}
	createdIDs := make([]string, 0, concurrentRuns)
	for index := 0; index < concurrentRuns; index++ {
		result := <-results
		if result.err != nil {
			t.Fatalf("concurrent CreateRun failed: %v", result.err)
		}
		createdIDs = append(createdIDs, result.run.ID)
	}
	var messageCount, distinctSequenceCount int
	if err := pool.QueryRow(ctx, `
		SELECT count(*),count(DISTINCT sequence)
		FROM agent_platform.agent_session_messages
		WHERE session_id=$1::uuid AND run_id::text=ANY($2::text[])`, session.ID, createdIDs,
	).Scan(&messageCount, &distinctSequenceCount); err != nil {
		t.Fatal(err)
	}
	if messageCount != concurrentRuns || distinctSequenceCount != concurrentRuns {
		t.Fatalf("concurrent session messages = %d/%d, want %d unique", messageCount, distinctSequenceCount, concurrentRuns)
	}
	var outboxAfter int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM agent_platform.agent_outbox`).Scan(&outboxAfter); err != nil {
		t.Fatal(err)
	}
	if outboxAfter != outboxBefore {
		t.Fatalf("disabled outbox grew from %d to %d", outboxBefore, outboxAfter)
	}

	// Exercise the mixed-version lock order explicitly. The simulated legacy
	// writer holds the Session FK KEY SHARE lock before asking for its former
	// row-level FOR UPDATE lock, while reconciliation holds the new advisory
	// lock and waits for that row. The legacy upgrade must still complete; once
	// it commits, reconciliation sees and projects the just-committed Run.
	legacyTx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = legacyTx.Rollback(context.Background()) }()
	var legacyRunID string
	if err := legacyTx.QueryRow(ctx, `
		INSERT INTO agent_platform.agent_runs
			(tenant_id,session_id,workflow_id,agent_version_id,input,binding_snapshot)
		VALUES ($1::text,$2::uuid,$6::uuid,$3::uuid,$4::jsonb,$5::jsonb)
		RETURNING id::text`, "test-tenant", session.ID, versionID,
		json.RawMessage(`{"question":"legacy rolling writer"}`), run.BindingSnapshot, run.WorkflowID,
	).Scan(&legacyRunID); err != nil {
		t.Fatal(err)
	}
	type reconcileResult struct {
		inserted int64
		err      error
	}
	repairContext, cancelRepair := context.WithTimeout(ctx, 5*time.Second)
	defer cancelRepair()
	repairDone := make(chan reconcileResult, 1)
	go func() {
		inserted, repairErr := store.ReconcileSessionMessages(repairContext, "test-tenant", session.ID)
		repairDone <- reconcileResult{inserted: inserted, err: repairErr}
	}()
	observer, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	for {
		var acquired bool
		if err := observer.QueryRow(ctx, `
			SELECT pg_try_advisory_lock(hashtextextended($1::text || ':' || $2::text,0))`,
			"test-tenant", session.ID).Scan(&acquired); err != nil {
			observer.Release()
			t.Fatal(err)
		}
		if !acquired {
			break
		}
		if _, err := observer.Exec(ctx, `
			SELECT pg_advisory_unlock(hashtextextended($1::text || ':' || $2::text,0))`,
			"test-tenant", session.ID); err != nil {
			observer.Release()
			t.Fatal(err)
		}
		select {
		case <-repairContext.Done():
			observer.Release()
			t.Fatal(repairContext.Err())
		case <-time.After(5 * time.Millisecond):
		}
	}
	observer.Release()
	var legacyLocked bool
	if err := legacyTx.QueryRow(ctx, `
		SELECT true FROM agent_platform.agent_sessions
		WHERE id=$1::uuid FOR UPDATE`, session.ID).Scan(&legacyLocked); err != nil {
		t.Fatal(err)
	}
	if err := legacyTx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	repair := <-repairDone
	if repair.err != nil || repair.inserted != 1 {
		t.Fatalf("mixed-version reconciliation inserted = %d, error = %v, want 1", repair.inserted, repair.err)
	}
	var legacyMessageCount int
	if err := pool.QueryRow(ctx, `
		SELECT count(*) FROM agent_platform.agent_session_messages
		WHERE run_id=$1::uuid AND message_kind='run_input'`, legacyRunID).Scan(&legacyMessageCount); err != nil {
		t.Fatal(err)
	}
	if legacyMessageCount != 1 {
		t.Fatalf("legacy Run projected messages = %d, want 1", legacyMessageCount)
	}
}

// requireDedicatedTestDatabase is a hard guard around the destructive reset
// below. Merely naming an environment variable TEST_* is not sufficient proof
// that the selected PostgreSQL database is disposable.
func requireDedicatedTestDatabase(databaseURL string) error {
	parsed, err := url.Parse(databaseURL)
	if err != nil {
		return err
	}
	database := strings.ToLower(strings.TrimSpace(path.Base(parsed.Path)))
	if database == "." || database == "/" || database == "" {
		return errors.New("TEST_AGENT_DATABASE_URL must name a dedicated database")
	}
	if !(strings.HasPrefix(database, "test_") || strings.HasSuffix(database, "_test") || strings.HasSuffix(database, "-test")) {
		return errors.New("refusing destructive integration test: database name must start with test_ or end with _test/-test")
	}
	return nil
}

func TestRequireDedicatedTestDatabaseRejectsRuntimeDatabase(t *testing.T) {
	if err := requireDedicatedTestDatabase("postgres://infra:infra@localhost:5432/infra_platform?sslmode=disable"); err == nil {
		t.Fatal("runtime database was accepted as a destructive integration target")
	}
	if err := requireDedicatedTestDatabase("postgres://infra:infra@localhost:5432/agent_platform_test?sslmode=disable"); err != nil {
		t.Fatalf("dedicated test database was rejected: %v", err)
	}
}

func TestValidateEvidenceFreshnessRejectsFileChangedAfterCommand(t *testing.T) {
	records := []successfulToolEvidence{{
		CallID: "compile-1", Sequence: 10, Name: "run_command",
		Arguments: json.RawMessage(`{"command":"python3","args":["-m","py_compile","snake_game.py"]}`),
	}}
	if err := validateEvidenceFreshness(records, map[string]int64{"snake_game.py": 11}); err == nil {
		t.Fatal("stale command evidence was accepted")
	}
	if err := validateEvidenceFreshness(records, map[string]int64{"snake_game.py": 9}); err != nil {
		t.Fatalf("fresh command evidence was rejected: %v", err)
	}
}

func TestValidateEvidenceSemanticsDoesNotTreatCompileAsRunnable(t *testing.T) {
	records := []successfulToolEvidence{{
		CallID: "compile-1", Sequence: 10, Name: "run_command",
		Arguments: json.RawMessage(`{"command":"python3","args":["-m","py_compile","snake_game.py"]}`),
		Result:    json.RawMessage(`{"command":"python3","args":["-m","py_compile","snake_game.py"],"exit_code":0}`),
	}}
	criterion := taskplan.AcceptanceCriterion{
		ID: "runnable", Description: "游戏可以直接运行", Status: taskplan.CriterionPassed,
		Verification:    taskplan.VerificationSpec{Kind: "command_exit_zero", Target: "snake_game.py"},
		EvidenceCallIDs: []string{"compile-1"},
	}
	if err := validateEvidenceSemantics(criterion, records); err == nil {
		t.Fatal("py_compile evidence was accepted as runnable proof")
	}
	criterion.Verification.Kind = "python_syntax"
	if err := validateEvidenceSemantics(criterion, records); err != nil {
		t.Fatalf("py_compile was rejected as syntax proof: %v", err)
	}
}

func TestValidateEvidenceSemanticsAcceptsLegacyPyCompileSyntaxContract(t *testing.T) {
	records := []successfulToolEvidence{{
		CallID: "compile-1", Sequence: 10, Name: "run_command",
		Arguments: json.RawMessage(`{"command":"python3","args":["-m","py_compile","snake.py"]}`),
		Result:    json.RawMessage(`{"command":"python3","args":["-m","py_compile","snake.py"],"exit_code":0}`),
	}}
	criterion := taskplan.AcceptanceCriterion{
		ID: "syntax", Description: "snake.py 语法正确，无错误", Status: taskplan.CriterionPassed,
		Verification:    taskplan.VerificationSpec{Kind: "command_exit_zero", Target: "python -m py_compile snake.py"},
		EvidenceCallIDs: []string{"compile-1"},
	}
	if err := validateEvidenceSemantics(criterion, records); err != nil {
		t.Fatalf("legacy py_compile syntax evidence was rejected: %v", err)
	}
}

func TestValidateEvidenceSemanticsRejectsEmptySearch(t *testing.T) {
	records := []successfulToolEvidence{{
		CallID: "search-1", Sequence: 3, Name: "search_files",
		Arguments: json.RawMessage(`{"path":".","query":"snake_game.py"}`),
		Result:    json.RawMessage(`{"path":".","query":"snake_game.py","results":[]}`),
	}}
	criterion := taskplan.AcceptanceCriterion{
		ID: "found", Description: "找到生成的游戏文件", Status: taskplan.CriterionPassed,
		Verification:    taskplan.VerificationSpec{Kind: "search_nonempty", Target: "."},
		EvidenceCallIDs: []string{"search-1"},
	}
	if err := validateEvidenceSemantics(criterion, records); err == nil {
		t.Fatal("empty search result was accepted as positive evidence")
	}
}

type integrationObjectStore struct {
	mu      sync.Mutex
	objects map[string][]byte
}

func (s *integrationObjectStore) Enabled() bool  { return true }
func (s *integrationObjectStore) Bucket() string { return "integration-artifacts" }
func (s *integrationObjectStore) Put(_ context.Context, key string, content []byte, _ string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.objects[key] = append([]byte(nil), content...)
	return nil
}
func (s *integrationObjectStore) Get(_ context.Context, key string) ([]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	content, ok := s.objects[key]
	if !ok {
		return nil, errors.New("object not found")
	}
	return append([]byte(nil), content...), nil
}

type integrationEmbeddings struct{ model string }

func (integrationEmbeddings) Enabled() bool { return true }
func (provider integrationEmbeddings) Status() embedding.Status {
	return embedding.Status{Configured: true, Ready: true, Model: provider.model, Mode: "live", Dim: 1024}
}

type failingIntegrationEmbeddings struct{}

func (failingIntegrationEmbeddings) Enabled() bool { return true }
func (failingIntegrationEmbeddings) Status() embedding.Status {
	return embedding.Status{Configured: true, Dim: 1024, LastError: "embedding unavailable"}
}
func (failingIntegrationEmbeddings) Embed(context.Context, []string, bool) (embedding.Result, error) {
	return embedding.Result{}, errors.New("embedding unavailable")
}
func (provider integrationEmbeddings) Embed(_ context.Context, texts []string, _ bool) (embedding.Result, error) {
	vectors := make([][]float32, len(texts))
	for index := range texts {
		vectors[index] = make([]float32, 1024)
		vectors[index][0] = 1
	}
	return embedding.Result{Vectors: vectors, Model: provider.model, Dim: 1024, Mode: "live"}, nil
}

func integrationSpec(prompt resource.PromptVersion, toolSet resource.ToolSetVersion) agent.Spec {
	return agent.Spec{
		Name:     "integration-agent",
		Harness:  agent.HarnessSpec{Name: "react-v1", MaxTurns: 1, MaxSteps: 8},
		Planning: agent.PlanningPolicy{Policy: agent.PlanningPolicyRequired},
		Model: agent.ModelBinding{
			Provider: "openai-compatible", ServiceRef: "test-model", Capability: "tool_calling", ModelID: "test-model-v1",
		},
		PromptRef:    agent.VersionRef{ID: prompt.ID, Version: strconv.Itoa(prompt.Version)},
		ToolSetRef:   agent.VersionRef{ID: toolSet.ID, Version: strconv.Itoa(toolSet.Version)},
		InputSchema:  json.RawMessage(`{"type":"object"}`),
		OutputSchema: json.RawMessage(`{"type":"object"}`),
		Context: agent.ContextPolicy{
			MaxInputTokens: 4096, ReserveOutputTokens: 512,
		},
		Runtime: agent.RuntimePolicy{
			RunTimeout: time.Minute, ModelTimeout: 20 * time.Second, ToolTimeout: 10 * time.Second,
			MaxModelCalls: 8, MaxToolCalls: 16,
		},
	}
}
