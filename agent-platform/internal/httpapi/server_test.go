package httpapi

import (
	"bytes"
	"context"
	"errors"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	providersandbox "github.com/heurry/cloudnative-infra-platform/agent-platform/internal/provider/sandbox"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/artifact"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/environment"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/resource"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/skill"
)

const (
	testRunID     = "c7d159b1-1eed-4cfc-a125-b946fedb70da"
	testVersionID = "2b7c7424-32ba-4697-85ef-82258624240e"
)

func TestHealthEndpoints(t *testing.T) {
	t.Parallel()

	handler := New().Handler()
	for _, path := range []string{"/health/live", "/health/ready"} {
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, path, nil))
		if recorder.Code != http.StatusOK {
			t.Fatalf("GET %s status = %d, want %d", path, recorder.Code, http.StatusOK)
		}
	}
}

func TestCapabilitiesExposeOnlyRegisteredHarnesses(t *testing.T) {
	t.Parallel()
	handler := NewWithRuns(&fakeRunStore{}, nil).Handler()
	request := httptest.NewRequest(http.MethodGet, "/api/v1/capabilities", nil)
	request.Header.Set("X-Tenant-ID", "tenant-a")
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK || !bytes.Contains(recorder.Body.Bytes(), []byte(`"name":"react-v1"`)) {
		t.Fatalf("capabilities status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
}

func TestRunAPIRequiresTenantAndCreatesRun(t *testing.T) {
	t.Parallel()
	store := &fakeRunStore{}
	handler := NewWithRuns(store, nil).Handler()
	body := []byte(`{"agent_version_id":"` + testVersionID + `","input":{"question":"hi"}}`)

	missingTenant := httptest.NewRecorder()
	handler.ServeHTTP(missingTenant, httptest.NewRequest(http.MethodPost, "/api/v1/runs", bytes.NewReader(body)))
	if missingTenant.Code != http.StatusBadRequest {
		t.Fatalf("missing tenant status = %d", missingTenant.Code)
	}

	request := httptest.NewRequest(http.MethodPost, "/api/v1/runs", bytes.NewReader(body))
	request.Header.Set("X-Tenant-ID", "tenant-a")
	request.Header.Set("X-Actor-ID", "alice")
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusCreated {
		t.Fatalf("create status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	if store.created.TenantID != "tenant-a" || store.created.AgentVersionID != testVersionID {
		t.Fatalf("create input = %+v", store.created)
	}
	if store.created.CreatedBy == nil || *store.created.CreatedBy != "alice" {
		t.Fatalf("created_by = %v", store.created.CreatedBy)
	}
}

func TestStudioTestRunRequiresActorAndAllowsDraft(t *testing.T) {
	t.Parallel()
	store := &fakeRunStore{}
	handler := NewWithRuns(store, nil).Handler()
	body := []byte(`{"agent_version_id":"` + testVersionID + `","trigger_type":"studio_test","input":{"question":"verify draft"}}`)

	missingActor := httptest.NewRequest(http.MethodPost, "/api/v1/runs", bytes.NewReader(body))
	missingActor.Header.Set("X-Tenant-ID", "tenant-a")
	missingActorRecorder := httptest.NewRecorder()
	handler.ServeHTTP(missingActorRecorder, missingActor)
	if missingActorRecorder.Code != http.StatusBadRequest {
		t.Fatalf("studio test without actor status = %d, body = %s", missingActorRecorder.Code, missingActorRecorder.Body.String())
	}

	request := httptest.NewRequest(http.MethodPost, "/api/v1/runs", bytes.NewReader(body))
	request.Header.Set("X-Tenant-ID", "tenant-a")
	request.Header.Set("X-Actor-ID", "studio-user")
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusCreated || !store.created.AllowDraft || store.created.TriggerType != "studio_test" {
		t.Fatalf("studio test status = %d, input = %+v, body = %s", recorder.Code, store.created, recorder.Body.String())
	}
}

func TestRunAPITenantScopeAndCancellationActor(t *testing.T) {
	t.Parallel()
	store := &fakeRunStore{run: agent.Run{ID: testRunID, TenantID: "tenant-a"}}
	handler := NewWithRuns(store, nil).Handler()

	get := httptest.NewRequest(http.MethodGet, "/api/v1/runs/"+testRunID, nil)
	get.Header.Set("X-Tenant-ID", "tenant-a")
	getRecorder := httptest.NewRecorder()
	handler.ServeHTTP(getRecorder, get)
	if getRecorder.Code != http.StatusOK || store.getTenant != "tenant-a" {
		t.Fatalf("get status = %d, tenant = %q", getRecorder.Code, store.getTenant)
	}

	cancel := httptest.NewRequest(http.MethodPost, "/api/v1/runs/"+testRunID+":cancel", nil)
	cancel.Header.Set("X-Tenant-ID", "tenant-a")
	cancelRecorder := httptest.NewRecorder()
	handler.ServeHTTP(cancelRecorder, cancel)
	if cancelRecorder.Code != http.StatusBadRequest {
		t.Fatalf("cancel without actor status = %d", cancelRecorder.Code)
	}

	cancel = httptest.NewRequest(http.MethodPost, "/api/v1/runs/"+testRunID+":cancel", nil)
	cancel.Header.Set("X-Tenant-ID", "tenant-a")
	cancel.Header.Set("X-Actor-ID", "operator")
	cancelRecorder = httptest.NewRecorder()
	handler.ServeHTTP(cancelRecorder, cancel)
	if cancelRecorder.Code != http.StatusAccepted || store.cancelActor != "operator" {
		t.Fatalf("cancel status = %d, actor = %q", cancelRecorder.Code, store.cancelActor)
	}
}

func TestRunEventStreamClosesWhenTerminalCursorIsCurrent(t *testing.T) {
	t.Parallel()
	store := &fakeRunStore{run: agent.Run{
		ID: testRunID, TenantID: "tenant-a", Status: agent.RunCompleted,
	}}
	handler := NewWithRuns(store, nil).Handler()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	request := httptest.NewRequest(
		http.MethodGet, "/api/v1/runs/"+testRunID+"/events:stream", nil,
	).WithContext(ctx)
	request.Header.Set("X-Tenant-ID", "tenant-a")
	request.Header.Set("Last-Event-ID", "14")
	recorder := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		handler.ServeHTTP(recorder, request)
		close(done)
	}()
	select {
	case <-done:
		if recorder.Code != http.StatusOK {
			t.Fatalf("terminal stream status = %d, body = %s", recorder.Code, recorder.Body.String())
		}
	case <-time.After(250 * time.Millisecond):
		cancel()
		<-done
		t.Fatal("terminal stream did not close after the durable ledger was exhausted")
	}
}

func TestArtifactPromotionRequiresActorAndAuditsSandboxResult(t *testing.T) {
	t.Parallel()
	store := &fakeArtifactStore{}
	server := NewWithRuns(store, nil)
	promoter := &fakeArtifactPromoter{response: providersandbox.PromoteResponse{
		TargetPath: "deliveries/game.py", ResultTargetSHA256: "result-hash", Created: true,
	}}
	server.SetArtifactPromoter(promoter)
	handler := server.Handler()
	body := []byte(`{"target_path":"deliveries/game.py"}`)

	missingActor := httptest.NewRequest(http.MethodPost, "/api/v1/artifacts/"+testVersionID+":promote", bytes.NewReader(body))
	missingActor.Header.Set("X-Tenant-ID", "tenant-a")
	missingRecorder := httptest.NewRecorder()
	handler.ServeHTTP(missingRecorder, missingActor)
	if missingRecorder.Code != http.StatusBadRequest {
		t.Fatalf("promotion without actor status = %d, body = %s", missingRecorder.Code, missingRecorder.Body.String())
	}

	request := httptest.NewRequest(http.MethodPost, "/api/v1/artifacts/"+testVersionID+":promote", bytes.NewReader(body))
	request.Header.Set("X-Tenant-ID", "tenant-a")
	request.Header.Set("X-Actor-ID", "operator")
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK || !bytes.Contains(recorder.Body.Bytes(), []byte(`"status":"promoted"`)) {
		t.Fatalf("promotion status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	if promoter.request.RunID != testRunID || promoter.request.SourcePath != "game.py" || promoter.request.TargetPath != "deliveries/game.py" {
		t.Fatalf("sandbox request = %+v", promoter.request)
	}
	if store.actor != "operator" || store.finished.ResultTargetSHA256 != "result-hash" || store.finishErr != nil {
		t.Fatalf("audit actor/result/error = %q/%+v/%v", store.actor, store.finished, store.finishErr)
	}
}

func TestReadyFailsWhenDatabaseIsUnavailable(t *testing.T) {
	t.Parallel()
	handler := NewWithRuns(&fakeRunStore{}, failingReadiness{}).Handler()
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/health/ready", nil))
	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("ready status = %d", recorder.Code)
	}
}

func TestCatalogCreateAndReleaseRoutes(t *testing.T) {
	t.Parallel()
	store := &fakePlatformStore{}
	handler := NewWithRuns(store, nil).Handler()

	create := httptest.NewRequest(http.MethodPost, "/api/v1/agents", bytes.NewBufferString(`{"key":"support","name":"Support Agent"}`))
	create.Header.Set("X-Tenant-ID", "tenant-a")
	createRecorder := httptest.NewRecorder()
	handler.ServeHTTP(createRecorder, create)
	if createRecorder.Code != http.StatusCreated || store.definitionInput.Key != "support" {
		t.Fatalf("create agent status = %d, input = %+v, body = %s", createRecorder.Code, store.definitionInput, createRecorder.Body.String())
	}

	release := httptest.NewRequest(http.MethodPost, "/api/v1/agent-versions/"+testVersionID+":release", nil)
	release.Header.Set("X-Tenant-ID", "tenant-a")
	releaseRecorder := httptest.NewRecorder()
	handler.ServeHTTP(releaseRecorder, release)
	if releaseRecorder.Code != http.StatusOK || store.releasedVersion != testVersionID || store.releaseTenant != "tenant-a" {
		t.Fatalf("release status = %d, tenant/version = %s/%s", releaseRecorder.Code, store.releaseTenant, store.releasedVersion)
	}

	prompt := httptest.NewRequest(http.MethodPost, "/api/v1/prompt-versions", bytes.NewBufferString(
		`{"key":"support-system","name":"Support System","content":"Return JSON."}`))
	prompt.Header.Set("X-Tenant-ID", "tenant-a")
	promptRecorder := httptest.NewRecorder()
	handler.ServeHTTP(promptRecorder, prompt)
	if promptRecorder.Code != http.StatusCreated || store.promptInput.Content != "Return JSON." {
		t.Fatalf("prompt status = %d, input = %+v", promptRecorder.Code, store.promptInput)
	}
	sessionRequest := httptest.NewRequest(http.MethodPost, "/api/v1/sessions", bytes.NewBufferString(
		`{"agent_id":"`+testRunID+`","user_id":"user-1","metadata":{"channel":"web"}}`))
	sessionRequest.Header.Set("X-Tenant-ID", "tenant-a")
	sessionRecorder := httptest.NewRecorder()
	handler.ServeHTTP(sessionRecorder, sessionRequest)
	if sessionRecorder.Code != http.StatusCreated || store.sessionInput.AgentID != testRunID {
		t.Fatalf("session status = %d, input = %+v", sessionRecorder.Code, store.sessionInput)
	}
}

func TestResourceListsAndSkillUpload(t *testing.T) {
	t.Parallel()
	store := &fakePlatformStore{}
	handler := NewWithRuns(store, nil).Handler()

	list := httptest.NewRequest(http.MethodGet, "/api/v1/skill-versions?limit=25", nil)
	list.Header.Set("X-Tenant-ID", "tenant-a")
	listRecorder := httptest.NewRecorder()
	handler.ServeHTTP(listRecorder, list)
	if listRecorder.Code != http.StatusOK || store.skillListLimit != 25 {
		t.Fatalf("list skills status=%d limit=%d body=%s", listRecorder.Code, store.skillListLimit, listRecorder.Body.String())
	}

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("file", "SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = part.Write([]byte("---\nname: Release Reviewer\nkey: release-reviewer\ndescription: Reviews releases\n---\nCheck health and benchmark evidence."))
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	upload := httptest.NewRequest(http.MethodPost, "/api/v1/skill-versions:upload", &body)
	upload.Header.Set("Content-Type", writer.FormDataContentType())
	upload.Header.Set("X-Tenant-ID", "tenant-a")
	upload.Header.Set("X-Actor-ID", "alice")
	uploadRecorder := httptest.NewRecorder()
	handler.ServeHTTP(uploadRecorder, upload)
	if uploadRecorder.Code != http.StatusCreated {
		t.Fatalf("upload status=%d body=%s", uploadRecorder.Code, uploadRecorder.Body.String())
	}
	if store.skillInput.Key != "release-reviewer" || store.skillInput.Name != "Release Reviewer" || len(store.skillInput.Spec.Instructions) != 1 {
		t.Fatalf("uploaded skill = %+v", store.skillInput)
	}
}

func TestSessionAuditIsTenantScopedAndReturnsDurableUsage(t *testing.T) {
	t.Parallel()
	store := &fakePlatformStore{}
	handler := NewWithRuns(store, nil).Handler()
	request := httptest.NewRequest(http.MethodGet, "/api/v1/sessions/"+testVersionID+"/audit", nil)
	request.Header.Set("X-Tenant-ID", "tenant-a")
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("audit status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	if store.auditTenant != "tenant-a" || store.auditSession != testVersionID {
		t.Fatalf("audit scope tenant/session=%s/%s", store.auditTenant, store.auditSession)
	}
	if !bytes.Contains(recorder.Body.Bytes(), []byte(`"total_tokens":30`)) ||
		!bytes.Contains(recorder.Body.Bytes(), []byte(`"tool_success_rate_percent":75`)) {
		t.Fatalf("audit body=%s", recorder.Body.String())
	}
}

func TestMemoryCRUDRequiresTenantOwnershipAndDeleteActor(t *testing.T) {
	t.Parallel()
	store := &fakePlatformStore{}
	handler := NewWithRuns(store, nil).Handler()
	create := httptest.NewRequest(http.MethodPost, "/api/v1/memories", bytes.NewBufferString(
		`{"scope":"agent","agent_id":"`+testRunID+`","kind":"preference","content":"Prefer stable routing","ttl_seconds":3600}`))
	create.Header.Set("X-Tenant-ID", "tenant-a")
	create.Header.Set("X-Actor-ID", "alice")
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, create)
	if recorder.Code != http.StatusCreated || store.memoryInput.TenantID != "tenant-a" || store.memoryInput.ExpiresAt == nil {
		t.Fatalf("create memory status=%d input=%+v body=%s", recorder.Code, store.memoryInput, recorder.Body.String())
	}

	missingActor := httptest.NewRequest(http.MethodDelete, "/api/v1/memories/"+testVersionID, nil)
	missingActor.Header.Set("X-Tenant-ID", "tenant-a")
	missingRecorder := httptest.NewRecorder()
	handler.ServeHTTP(missingRecorder, missingActor)
	if missingRecorder.Code != http.StatusBadRequest {
		t.Fatalf("delete without actor status=%d", missingRecorder.Code)
	}

	remove := httptest.NewRequest(http.MethodDelete, "/api/v1/memories/"+testVersionID, nil)
	remove.Header.Set("X-Tenant-ID", "tenant-a")
	remove.Header.Set("X-Actor-ID", "alice")
	removeRecorder := httptest.NewRecorder()
	handler.ServeHTTP(removeRecorder, remove)
	if removeRecorder.Code != http.StatusNoContent || store.deletedMemoryID != testVersionID || store.deletedMemoryActor != "alice" {
		t.Fatalf("delete status=%d id/actor=%s/%s", removeRecorder.Code, store.deletedMemoryID, store.deletedMemoryActor)
	}
}

type fakeRunStore struct {
	created      agent.CreateRun
	run          agent.Run
	getTenant    string
	cancelActor  string
	cancelTenant string
}

type fakeArtifactStore struct {
	fakeRunStore
	actor     string
	finished  artifact.PromotionResult
	finishErr error
}

func (s *fakeArtifactStore) ListArtifactsForTenant(context.Context, string, string, int) ([]artifact.Artifact, error) {
	return nil, nil
}

func (s *fakeArtifactStore) GetArtifactContentForTenant(context.Context, string, string) (artifact.Artifact, []byte, error) {
	return artifact.Artifact{ID: testVersionID, RunID: testRunID, Kind: "workspace_file", Name: "game.py", ContentHash: "source-hash"}, nil, nil
}

func (s *fakeArtifactStore) BeginArtifactPromotion(_ context.Context, tenantID, artifactID string, request artifact.PromotionRequest, actor string) (artifact.Promotion, error) {
	s.actor = actor
	return artifact.Promotion{ID: testVersionID, TenantID: tenantID, RunID: testRunID, ArtifactID: artifactID, TargetPath: request.TargetPath, SourceHash: "source-hash", Status: "requested", RequestedBy: actor, CreatedAt: time.Now()}, nil
}

func (s *fakeArtifactStore) FinishArtifactPromotion(_ context.Context, tenantID, promotionID string, result artifact.PromotionResult, promoteErr error) (artifact.Promotion, error) {
	s.finished, s.finishErr = result, promoteErr
	now := time.Now()
	return artifact.Promotion{ID: promotionID, TenantID: tenantID, RunID: testRunID, ArtifactID: testVersionID, TargetPath: result.TargetPath, SourceHash: "source-hash", ResultTargetSHA256: result.ResultTargetSHA256, Status: "promoted", RequestedBy: s.actor, CreatedAt: now, FinishedAt: &now}, nil
}

type fakeArtifactPromoter struct {
	request  providersandbox.PromoteRequest
	response providersandbox.PromoteResponse
	err      error
}

func (p *fakeArtifactPromoter) Promote(_ context.Context, request providersandbox.PromoteRequest) (providersandbox.PromoteResponse, error) {
	p.request = request
	return p.response, p.err
}

func (s *fakeRunStore) CreateRun(_ context.Context, input agent.CreateRun) (agent.Run, error) {
	s.created = input
	return agent.Run{ID: testRunID, TenantID: input.TenantID, AgentVersionID: input.AgentVersionID}, nil
}

func (s *fakeRunStore) GetRunForTenant(_ context.Context, tenantID, _ string) (agent.Run, error) {
	s.getTenant = tenantID
	if s.run.ID == "" {
		return agent.Run{}, agent.ErrRunNotFound
	}
	return s.run, nil
}

func (s *fakeRunStore) RequestCancelForTenant(_ context.Context, tenantID, _ string, actor string) error {
	s.cancelTenant = tenantID
	s.cancelActor = actor
	return nil
}

func (s *fakeRunStore) ListEventsForTenant(_ context.Context, _, _ string, _ int64, _ int) ([]event.Event, error) {
	return nil, nil
}

func (s *fakeRunStore) ListRunsForTenant(context.Context, string, int) ([]agent.Run, error) {
	return []agent.Run{}, nil
}

func (s *fakeRunStore) ObservabilitySummary(context.Context, string) (agent.ObservabilitySummary, error) {
	return agent.ObservabilitySummary{}, nil
}

type failingReadiness struct{}

func (failingReadiness) Ping(context.Context) error { return errors.New("unavailable") }

type fakePlatformStore struct {
	fakeRunStore
	definitionInput    agent.CreateDefinition
	releaseTenant      string
	releasedVersion    string
	promptInput        resource.CreatePromptVersion
	skillInput         skill.CreateVersion
	skillListLimit     int
	sessionInput       agent.CreateSession
	sessionListAgent   string
	auditTenant        string
	auditSession       string
	memoryInput        agent.CreateMemory
	deletedMemoryID    string
	deletedMemoryActor string
}

func (s *fakePlatformStore) CreateDefinition(_ context.Context, input agent.CreateDefinition) (agent.Definition, error) {
	s.definitionInput = input
	return agent.Definition{ID: testRunID, TenantID: input.TenantID, Key: input.Key, Name: input.Name}, nil
}

func (s *fakePlatformStore) ListDefinitions(context.Context, string, int) ([]agent.Definition, error) {
	return []agent.Definition{}, nil
}

func (s *fakePlatformStore) GetDefinition(context.Context, string, string) (agent.Definition, error) {
	return agent.Definition{ID: testRunID}, nil
}

func (s *fakePlatformStore) CreateVersion(context.Context, agent.CreateVersion) (agent.Version, error) {
	return agent.Version{ID: testVersionID}, nil
}

func (s *fakePlatformStore) ListVersions(context.Context, string, string) ([]agent.Version, error) {
	return []agent.Version{}, nil
}

func (s *fakePlatformStore) ReleaseVersion(_ context.Context, tenantID, versionID string) (agent.Version, error) {
	s.releaseTenant = tenantID
	s.releasedVersion = versionID
	return agent.Version{ID: versionID, Status: "published"}, nil
}

func (s *fakePlatformStore) ListExecutableVersions(context.Context, string, int) ([]agent.ExecutableVersion, error) {
	return []agent.ExecutableVersion{}, nil
}

func (s *fakePlatformStore) CreatePromptVersion(_ context.Context, input resource.CreatePromptVersion) (resource.PromptVersion, error) {
	s.promptInput = input
	return resource.PromptVersion{ID: testVersionID, TenantID: input.TenantID, Content: input.Content}, nil
}

func (s *fakePlatformStore) GetPromptVersion(context.Context, string, string) (resource.PromptVersion, error) {
	return resource.PromptVersion{ID: testVersionID}, nil
}
func (s *fakePlatformStore) ListPromptVersions(context.Context, string, int) ([]resource.PromptVersion, error) {
	return []resource.PromptVersion{}, nil
}

func (s *fakePlatformStore) CreateToolVersion(context.Context, resource.CreateToolVersion) (resource.ToolVersion, error) {
	return resource.ToolVersion{ID: testVersionID}, nil
}

func (s *fakePlatformStore) GetToolVersion(context.Context, string, string) (resource.ToolVersion, error) {
	return resource.ToolVersion{ID: testVersionID}, nil
}
func (s *fakePlatformStore) ListToolVersions(context.Context, string, int) ([]resource.ToolVersion, error) {
	return []resource.ToolVersion{}, nil
}

func (s *fakePlatformStore) CreateToolSetVersion(context.Context, resource.CreateToolSetVersion) (resource.ToolSetVersion, error) {
	return resource.ToolSetVersion{ID: testVersionID}, nil
}

func (s *fakePlatformStore) GetToolSetVersion(context.Context, string, string) (resource.ToolSetVersion, error) {
	return resource.ToolSetVersion{ID: testVersionID}, nil
}
func (s *fakePlatformStore) ListToolSetVersions(context.Context, string, int) ([]resource.ToolSetVersion, error) {
	return []resource.ToolSetVersion{}, nil
}

func (s *fakePlatformStore) CreateSkillVersion(_ context.Context, input skill.CreateVersion) (skill.Version, error) {
	s.skillInput = input
	return skill.Version{ID: testVersionID}, nil
}
func (s *fakePlatformStore) GetSkillVersion(context.Context, string, string) (skill.Version, error) {
	return skill.Version{ID: testVersionID}, nil
}
func (s *fakePlatformStore) ListSkillVersions(_ context.Context, _ string, limit int) ([]skill.Version, error) {
	s.skillListLimit = limit
	return []skill.Version{}, nil
}
func (s *fakePlatformStore) CreateSkillSetVersion(context.Context, skill.CreateSetVersion) (skill.Version, error) {
	return skill.Version{ID: testVersionID}, nil
}
func (s *fakePlatformStore) GetSkillSetVersion(context.Context, string, string) (skill.Version, error) {
	return skill.Version{ID: testVersionID}, nil
}
func (s *fakePlatformStore) ListSkillSetVersions(context.Context, string, int) ([]skill.Version, error) {
	return []skill.Version{}, nil
}
func (s *fakePlatformStore) ListEnvironmentTemplates(context.Context) ([]environment.Template, error) {
	return []environment.Template{}, nil
}
func (s *fakePlatformStore) ListDependencyInstalls(context.Context, string, string, int) ([]environment.Install, error) {
	return []environment.Install{}, nil
}

func (s *fakePlatformStore) CreateSession(_ context.Context, input agent.CreateSession) (agent.Session, error) {
	s.sessionInput = input
	return agent.Session{ID: testVersionID, TenantID: input.TenantID, AgentID: input.AgentID}, nil
}

func (s *fakePlatformStore) GetSession(context.Context, string, string) (agent.Session, error) {
	return agent.Session{ID: testVersionID}, nil
}

func (s *fakePlatformStore) ListSessions(_ context.Context, _ string, agentID string, _ int) ([]agent.Session, error) {
	s.sessionListAgent = agentID
	return []agent.Session{{ID: testVersionID, AgentID: agentID}}, nil
}

func (s *fakePlatformStore) ListSessionRuns(context.Context, string, string, int) ([]agent.Run, error) {
	return []agent.Run{{ID: testRunID, SessionID: stringPointer(testVersionID)}}, nil
}

func (s *fakePlatformStore) GetSessionAudit(_ context.Context, tenantID, sessionID string) (agent.SessionAudit, error) {
	s.auditTenant, s.auditSession = tenantID, sessionID
	return agent.SessionAudit{
		SessionID: sessionID, RunCount: 1, InputTokens: 20, OutputTokens: 10, TotalTokens: 30,
		ToolCalls: 4, SuccessfulToolCalls: 3, FailedToolCalls: 1, TerminalToolCalls: 4,
		ToolSuccessRatePercent: 75, Runs: []agent.RunAudit{},
	}, nil
}

func (s *fakePlatformStore) CreateMemory(_ context.Context, input agent.CreateMemory) (agent.Memory, error) {
	s.memoryInput = input
	return agent.Memory{ID: testVersionID, TenantID: input.TenantID, Scope: input.Scope, Kind: input.Kind, Content: input.Content, ExpiresAt: input.ExpiresAt}, nil
}

func (s *fakePlatformStore) ListMemories(context.Context, agent.MemoryFilter) ([]agent.Memory, error) {
	return []agent.Memory{}, nil
}

func (s *fakePlatformStore) DeleteMemory(_ context.Context, _ string, memoryID, actor string) error {
	s.deletedMemoryID = memoryID
	s.deletedMemoryActor = actor
	return nil
}

func stringPointer(value string) *string { return &value }
