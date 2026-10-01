package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/embedding"
	providersandbox "github.com/heurry/cloudnative-infra-platform/agent-platform/internal/provider/sandbox"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/a2a"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/approval"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/artifact"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/environment"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/harness"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/interaction"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/mcp"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/resource"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/score"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/skill"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/taskplan"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/trajectory"
	verificationdomain "github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/verification"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/workflow"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
)

const (
	maxRequestBytes     = 1 << 20
	maxSkillUploadBytes = 1 << 20
)

var uuidPattern = regexp.MustCompile(`^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$`)

// RunStore is the tenant-scoped persistence contract used by the HTTP API.
type RunStore interface {
	CreateRun(context.Context, agent.CreateRun) (agent.Run, error)
	ListRunsForTenant(context.Context, string, int) ([]agent.Run, error)
	GetRunForTenant(context.Context, string, string) (agent.Run, error)
	RequestCancelForTenant(context.Context, string, string, string) error
	ListEventsForTenant(context.Context, string, string, int64, int) ([]event.Event, error)
	ObservabilitySummary(context.Context, string) (agent.ObservabilitySummary, error)
}

type WorkflowStore interface {
	ListWorkflowsForTenant(context.Context, string, string, int) ([]workflow.Workflow, error)
}

type WorkflowRunStore interface {
	ListWorkflowRunsForTenant(context.Context, string, string, int) ([]agent.Run, error)
}

// CatalogStore persists tenant-scoped Agent definitions and immutable versions.
type CatalogStore interface {
	CreateDefinition(context.Context, agent.CreateDefinition) (agent.Definition, error)
	ListDefinitions(context.Context, string, int) ([]agent.Definition, error)
	GetDefinition(context.Context, string, string) (agent.Definition, error)
	CreateVersion(context.Context, agent.CreateVersion) (agent.Version, error)
	ListVersions(context.Context, string, string) ([]agent.Version, error)
	ReleaseVersion(context.Context, string, string) (agent.Version, error)
	ListExecutableVersions(context.Context, string, int) ([]agent.ExecutableVersion, error)
}

// ResourceStore persists immutable execution resources referenced by AgentSpec.
type ResourceStore interface {
	CreatePromptVersion(context.Context, resource.CreatePromptVersion) (resource.PromptVersion, error)
	GetPromptVersion(context.Context, string, string) (resource.PromptVersion, error)
	ListPromptVersions(context.Context, string, int) ([]resource.PromptVersion, error)
	CreateToolVersion(context.Context, resource.CreateToolVersion) (resource.ToolVersion, error)
	GetToolVersion(context.Context, string, string) (resource.ToolVersion, error)
	ListToolVersions(context.Context, string, int) ([]resource.ToolVersion, error)
	CreateToolSetVersion(context.Context, resource.CreateToolSetVersion) (resource.ToolSetVersion, error)
	GetToolSetVersion(context.Context, string, string) (resource.ToolSetVersion, error)
	ListToolSetVersions(context.Context, string, int) ([]resource.ToolSetVersion, error)
	CreateSkillVersion(context.Context, skill.CreateVersion) (skill.Version, error)
	GetSkillVersion(context.Context, string, string) (skill.Version, error)
	ListSkillVersions(context.Context, string, int) ([]skill.Version, error)
	CreateSkillSetVersion(context.Context, skill.CreateSetVersion) (skill.Version, error)
	GetSkillSetVersion(context.Context, string, string) (skill.Version, error)
	ListSkillSetVersions(context.Context, string, int) ([]skill.Version, error)
	ListEnvironmentTemplates(context.Context) ([]environment.Template, error)
	ListDependencyInstalls(context.Context, string, string, int) ([]environment.Install, error)
}

// SessionStore persists tenant-owned multi-Run conversations.
type SessionStore interface {
	CreateSession(context.Context, agent.CreateSession) (agent.Session, error)
	GetSession(context.Context, string, string) (agent.Session, error)
	ListSessions(context.Context, string, string, int) ([]agent.Session, error)
	ListSessionRuns(context.Context, string, string, int) ([]agent.Run, error)
}

// SessionAuditStore is optional during rolling upgrades. Audit reads are
// served only when the persistence implementation can derive them from the
// durable execution ledger.
type SessionAuditStore interface {
	GetSessionAudit(context.Context, string, string) (agent.SessionAudit, error)
}

// MemoryStore persists explicit, tenant-scoped long-term memory resources.
type MemoryStore interface {
	CreateMemory(context.Context, agent.CreateMemory) (agent.Memory, error)
	ListMemories(context.Context, agent.MemoryFilter) ([]agent.Memory, error)
	DeleteMemory(context.Context, string, string, string) error
}

type MemoryDetailStore interface {
	GetMemoryForTenant(context.Context, string, string, string) (agent.Memory, error)
}

type MemoryEditStore interface {
	UpdateMemory(context.Context, agent.MemoryPatch) (agent.Memory, error)
}

// MemoryTeamStore is optional so existing embedders can roll out Team ACLs
// without changing the legacy Memory CRUD contract in one release.
type MemoryTeamStore interface {
	UpsertMemoryTeamMembership(context.Context, string, string, string, string, bool) (agent.MemoryTeamMembership, error)
	DeleteMemoryTeamMembership(context.Context, string, string, string) error
	ListMemoryTeamMemberships(context.Context, string, string, int) ([]agent.MemoryTeamMembership, error)
}

type MemoryFeedbackStore interface {
	RecordMemoryFeedback(context.Context, agent.MemoryFeedback) (agent.MemoryFeedback, error)
	ListMemoryFeedback(context.Context, string, string, int) ([]agent.MemoryFeedback, error)
}

type MemoryLifecycleStore interface {
	ListMemoryLifecycleEvents(context.Context, string, string, string, int) ([]agent.MemoryLifecycleEvent, error)
}

type MemoryRunTimelineStore interface {
	ListMemoryLifecycleEventsForRun(context.Context, string, string, string, int) ([]agent.MemoryLifecycleEvent, error)
}

type MemorySourceStore interface {
	ListMemorySources(context.Context, string, string, string, string, string, int) ([]agent.MemorySource, error)
}

type MemorySourceSyncStore interface {
	SyncStaticMemorySources(context.Context, agent.Run, []agent.StaticMemoryDocument) error
}

// MemoryRetrievalStore exposes the body-free retrieval funnel used by the
// Run Detail/Memory operations views. It is optional for compatibility with
// embedders that only implement legacy Memory CRUD.
type MemoryRetrievalStore interface {
	ListMemoryRetrievals(context.Context, string, string, int) ([]agent.MemoryRetrievalRecord, error)
}

type MemoryGovernanceStore interface {
	ListMemoryRevisions(context.Context, string, string, int) ([]agent.MemoryRevision, error)
	VerifyMemory(context.Context, string, string, string) (agent.Memory, error)
	PromoteMemoryToTeam(context.Context, string, string, string, string) (agent.Memory, error)
	SupersedeMemory(context.Context, string, string, string) error
}

// MemoryManifestStore is optional so older embedders can keep the legacy CRUD
// contract while taxonomy-aware persistence rolls out.
type MemoryManifestStore interface {
	ListMemoryManifest(context.Context, agent.MemoryFilter, int) ([]agent.MemoryManifestEntry, error)
}

type ArtifactStore interface {
	ListArtifactsForTenant(context.Context, string, string, int) ([]artifact.Artifact, error)
	GetArtifactContentForTenant(context.Context, string, string) (artifact.Artifact, []byte, error)
	BeginArtifactPromotion(context.Context, string, string, artifact.PromotionRequest, string) (artifact.Promotion, error)
	FinishArtifactPromotion(context.Context, string, string, artifact.PromotionResult, error) (artifact.Promotion, error)
}

// RunManifestStore exposes the canonical delivery projection separately from
// append-only artifact history. It is optional during rolling upgrades; the
// artifact list endpoint remains available when manifests are not migrated.
type RunManifestStore interface {
	GetRunManifestForTenant(context.Context, string, string) (artifact.RunManifest, error)
}

// WorkflowArtifactStore is optional so existing embedders can upgrade without
// changing their Run artifact implementation. It exposes the immutable
// artifact history across all Run attempts in one long-lived Workflow.
type WorkflowArtifactStore interface {
	ListWorkflowArtifactsForTenant(context.Context, string, string, int) ([]artifact.Artifact, error)
}

type ArtifactPromoter interface {
	Promote(context.Context, providersandbox.PromoteRequest) (providersandbox.PromoteResponse, error)
}

type ApprovalStore interface {
	ListApprovalsForTenant(context.Context, string, string, int) ([]approval.Approval, error)
	DecideApproval(context.Context, string, string, approval.Decision) (approval.Approval, error)
}

type MCPStore interface {
	CreateMCPServerVersion(context.Context, mcp.CreateServerVersion) (mcp.ServerVersion, error)
	ListMCPServersForTenant(context.Context, string, int) ([]mcp.ServerVersion, error)
	GetMCPServerVersionForTenant(context.Context, string, string) (mcp.ServerVersion, error)
	ReplaceMCPToolSnapshots(context.Context, string, string, []mcp.ToolSnapshot) error
	SaveMCPHealth(context.Context, string, mcp.Health) error
}
type ChildRunStore interface {
	ListChildRunsForTenant(context.Context, string, string) ([]agent.Run, error)
}
type A2AStore interface {
	CreateA2ATask(context.Context, string, string, a2a.SendMessageRequest, *string, *string) (a2a.Task, error)
	GetA2ATaskForTenant(context.Context, string, string) (a2a.Task, error)
	CancelA2ATask(context.Context, string, string, string) (a2a.Task, error)
}
type TaskPlanStore interface {
	GetTaskPlanForWorkflow(context.Context, string, string) (taskplan.Plan, error)
	ListVerificationRecordsForTenant(context.Context, string, string) ([]verificationdomain.Record, error)
}
type UserInteractionStore interface {
	GetPendingQuestionForRun(context.Context, string, string) (interaction.Question, error)
	AnswerUserQuestion(context.Context, string, string, string, string) (interaction.Question, error)
}

// Readiness reports whether required infrastructure is reachable.
type Readiness interface {
	Ping(context.Context) error
}

type SemanticReadiness interface {
	Ping(context.Context) error
	Status() embedding.Status
}

// RunEventSubscriber is an optional low-latency wakeup path. Subscribers must
// re-read the durable Event Ledger; a missed signal never implies lost data.
type RunEventSubscriber interface {
	SubscribeRun(context.Context, string) (<-chan struct{}, func(), error)
}

type StorageStore interface {
	StorageSummary(context.Context, string) (agent.StorageSummary, error)
}

type ScoreStore interface {
	CreateScore(context.Context, score.Create) (score.Score, error)
	ListScoresForTenant(context.Context, string, string, int) ([]score.Score, error)
}

// Server exposes the standalone Agent Platform HTTP API.
type Server struct {
	mux                *http.ServeMux
	runs               RunStore
	workflows          WorkflowStore
	workflowRuns       WorkflowRunStore
	catalog            CatalogStore
	resources          ResourceStore
	sessions           SessionStore
	sessionAudits      SessionAuditStore
	memories           MemoryStore
	memoryDetails      MemoryDetailStore
	memoryEdit         MemoryEditStore
	memoryTeams        MemoryTeamStore
	memoryFeedback     MemoryFeedbackStore
	memoryLifecycle    MemoryLifecycleStore
	memoryRunTimeline  MemoryRunTimelineStore
	memorySources      MemorySourceStore
	memorySourceSync   MemorySourceSyncStore
	memoryRetrievals   MemoryRetrievalStore
	memoryGovernance   MemoryGovernanceStore
	memoryManifest     MemoryManifestStore
	artifacts          ArtifactStore
	manifests          RunManifestStore
	workflowArtifacts  WorkflowArtifactStore
	artifactPromoter   ArtifactPromoter
	approvals          ApprovalStore
	mcpStore           MCPStore
	children           ChildRunStore
	a2aStore           A2AStore
	planStore          TaskPlanStore
	interactionStore   UserInteractionStore
	mcpAllowedHosts    []string
	lookupEnv          func(string) (string, bool)
	readiness          Readiness
	runEvents          RunEventSubscriber
	storage            StorageStore
	scores             ScoreStore
	hotStore           Readiness
	coldStore          Readiness
	semanticStore      SemanticReadiness
	workerCapabilities WorkerCapabilityStore
	securityEnforced   bool
	a2aDefaultTenant   string
	a2aDefaultAgent    string
}

type WorkerCapabilityStore interface {
	ListWorkerCapabilities(context.Context) ([]agent.WorkerCapability, error)
}

// SetSecurityEnforced makes the effective identity boundary visible through
// capability discovery. Authentication itself is performed by outer middleware.
func (s *Server) SetSecurityEnforced(enforced bool) { s.securityEnforced = enforced }

func (s *Server) SetMCPPolicy(allowedHosts []string, lookup func(string) (string, bool)) {
	s.mcpAllowedHosts = append([]string(nil), allowedHosts...)
	s.lookupEnv = lookup
}

func (s *Server) SetArtifactPromoter(promoter ArtifactPromoter) { s.artifactPromoter = promoter }

func (s *Server) SetRunEventSubscriber(subscriber RunEventSubscriber) { s.runEvents = subscriber }

func (s *Server) SetHotStoreReadiness(readiness Readiness)              { s.hotStore = readiness }
func (s *Server) SetColdStoreReadiness(readiness Readiness)             { s.coldStore = readiness }
func (s *Server) SetSemanticStoreReadiness(readiness SemanticReadiness) { s.semanticStore = readiness }

// SetA2ADiscovery configures the public, unambiguous well-known Agent Card.
// Tenant-scoped Task operations remain protected by the normal auth boundary.
func (s *Server) SetA2ADiscovery(tenantID, agentID string) {
	s.a2aDefaultTenant = strings.TrimSpace(tenantID)
	s.a2aDefaultAgent = strings.TrimSpace(agentID)
}

// New creates a probe-only server, primarily useful for local smoke tests.
func New() *Server {
	return newServer(nil, nil)
}

// NewWithRuns creates the durable Run API and database-backed readiness probe.
func NewWithRuns(runs RunStore, readiness Readiness) *Server {
	return newServer(runs, readiness)
}

func newServer(runs RunStore, readiness Readiness) *Server {
	s := &Server{mux: http.NewServeMux(), runs: runs, readiness: readiness}
	if catalog, ok := runs.(CatalogStore); ok {
		s.catalog = catalog
	}
	if workflows, ok := runs.(WorkflowStore); ok {
		s.workflows = workflows
	}
	if workflowRuns, ok := runs.(WorkflowRunStore); ok {
		s.workflowRuns = workflowRuns
	}
	if resources, ok := runs.(ResourceStore); ok {
		s.resources = resources
	}
	if sessions, ok := runs.(SessionStore); ok {
		s.sessions = sessions
	}
	if audits, ok := runs.(SessionAuditStore); ok {
		s.sessionAudits = audits
	}
	if memories, ok := runs.(MemoryStore); ok {
		s.memories = memories
	}
	if memoryDetails, ok := runs.(MemoryDetailStore); ok {
		s.memoryDetails = memoryDetails
	}
	if memoryEdit, ok := runs.(MemoryEditStore); ok {
		s.memoryEdit = memoryEdit
	}
	if memoryTeams, ok := runs.(MemoryTeamStore); ok {
		s.memoryTeams = memoryTeams
	}
	if memoryFeedback, ok := runs.(MemoryFeedbackStore); ok {
		s.memoryFeedback = memoryFeedback
	}
	if memoryLifecycle, ok := runs.(MemoryLifecycleStore); ok {
		s.memoryLifecycle = memoryLifecycle
	}
	if memoryRunTimeline, ok := runs.(MemoryRunTimelineStore); ok {
		s.memoryRunTimeline = memoryRunTimeline
	}
	if memorySources, ok := runs.(MemorySourceStore); ok {
		s.memorySources = memorySources
	}
	if memorySourceSync, ok := runs.(MemorySourceSyncStore); ok {
		s.memorySourceSync = memorySourceSync
	}
	if memoryRetrievals, ok := runs.(MemoryRetrievalStore); ok {
		s.memoryRetrievals = memoryRetrievals
	}
	if memoryGovernance, ok := runs.(MemoryGovernanceStore); ok {
		s.memoryGovernance = memoryGovernance
	}
	if memoryManifest, ok := runs.(MemoryManifestStore); ok {
		s.memoryManifest = memoryManifest
	}
	if artifacts, ok := runs.(ArtifactStore); ok {
		s.artifacts = artifacts
	}
	if manifests, ok := runs.(RunManifestStore); ok {
		s.manifests = manifests
	}
	if workflowArtifacts, ok := runs.(WorkflowArtifactStore); ok {
		s.workflowArtifacts = workflowArtifacts
	}
	if approvals, ok := runs.(ApprovalStore); ok {
		s.approvals = approvals
	}
	if mcpStore, ok := runs.(MCPStore); ok {
		s.mcpStore = mcpStore
	}
	if children, ok := runs.(ChildRunStore); ok {
		s.children = children
	}
	if a2aStore, ok := runs.(A2AStore); ok {
		s.a2aStore = a2aStore
	}
	if planStore, ok := runs.(TaskPlanStore); ok {
		s.planStore = planStore
	}
	if interactionStore, ok := runs.(UserInteractionStore); ok {
		s.interactionStore = interactionStore
	}
	if storage, ok := runs.(StorageStore); ok {
		s.storage = storage
	}
	if scores, ok := runs.(ScoreStore); ok {
		s.scores = scores
	}
	if workers, ok := runs.(WorkerCapabilityStore); ok {
		s.workerCapabilities = workers
	}
	s.lookupEnv = os.LookupEnv
	s.mux.HandleFunc("GET /health/live", s.live)
	s.mux.HandleFunc("GET /health/ready", s.ready)
	if runs != nil {
		s.mux.HandleFunc("POST /api/v1/runs", s.createRun)
		s.mux.HandleFunc("GET /api/v1/runs", s.listRuns)
		if s.workflows != nil {
			s.mux.HandleFunc("GET /api/v1/workflows", s.listWorkflows)
		}
		if s.workflowRuns != nil {
			s.mux.HandleFunc("GET /api/v1/workflows/{workflow_id}/runs", s.listWorkflowRuns)
		}
		s.mux.HandleFunc("GET /api/v1/runs/{run_id}", s.getRun)
		s.mux.HandleFunc("POST /api/v1/runs/{run_action}", s.cancelRun)
		s.mux.HandleFunc("GET /api/v1/runs/{run_id}/events", s.listRunEvents)
		s.mux.HandleFunc("GET /api/v1/runs/{run_id}/events:stream", s.streamRunEvents)
		s.mux.HandleFunc("GET /api/v1/runs/{run_id}/trajectory", s.getRunTrajectory)
		if s.planStore != nil {
			s.mux.HandleFunc("GET /api/v1/runs/{run_id}/plan", s.getRunPlan)
			s.mux.HandleFunc("GET /api/v1/runs/{run_id}/verifications", s.listRunVerifications)
		}
		if s.interactionStore != nil {
			s.mux.HandleFunc("GET /api/v1/runs/{run_id}/question", s.getRunQuestion)
			s.mux.HandleFunc("POST /api/v1/questions/{question_action}", s.answerRunQuestion)
		}
		s.mux.HandleFunc("GET /api/v1/observability/summary", s.observabilitySummary)
		if s.storage != nil {
			s.mux.HandleFunc("GET /api/v1/storage/summary", s.storageSummary)
		}
		if s.scores != nil {
			s.mux.HandleFunc("GET /api/v1/runs/{run_id}/scores", s.listRunScores)
			s.mux.HandleFunc("POST /api/v1/scores", s.createScore)
		}
		s.mux.HandleFunc("GET /api/v1/capabilities", s.capabilities)
		if s.workerCapabilities != nil {
			s.mux.HandleFunc("GET /api/v1/worker-capabilities", s.listWorkerCapabilities)
		}
	}
	if s.artifacts != nil {
		s.mux.HandleFunc("GET /api/v1/runs/{run_id}/artifacts", s.listRunArtifacts)
		if manifests, ok := s.runs.(RunManifestStore); ok {
			s.manifests = manifests
			s.mux.HandleFunc("GET /api/v1/runs/{run_id}/manifest", s.getRunManifest)
		}
		s.mux.HandleFunc("GET /api/v1/artifacts/{artifact_id}/content", s.getArtifactContent)
		s.mux.HandleFunc("POST /api/v1/artifacts/{artifact_action}", s.promoteArtifact)
	}
	if s.workflowArtifacts != nil {
		s.mux.HandleFunc("GET /api/v1/workflows/{workflow_id}/artifacts", s.listWorkflowArtifacts)
	}
	if s.children != nil {
		s.mux.HandleFunc("GET /api/v1/runs/{run_id}/children", s.listRunChildren)
	}
	if s.approvals != nil {
		s.mux.HandleFunc("GET /api/v1/approvals", s.listApprovals)
		s.mux.HandleFunc("POST /api/v1/approvals/{approval_action}", s.decideApproval)
	}
	if s.mcpStore != nil {
		s.mux.HandleFunc("POST /api/v1/mcp-servers", s.createMCPServer)
		s.mux.HandleFunc("GET /api/v1/mcp-servers", s.listMCPServers)
		s.mux.HandleFunc("POST /api/v1/mcp-server-versions/{mcp_action}", s.mcpServerAction)
	}
	if s.a2aStore != nil && s.catalog != nil {
		s.mux.HandleFunc("GET /.well-known/agent-card.json", s.wellKnownAgentCard)
		s.mux.HandleFunc("GET /api/v1/a2a/agents/{agent_id}/card", s.a2aAgentCard)
		s.mux.HandleFunc("POST /api/v1/a2a/agents/{agent_id}/message:send", s.a2aSendMessage)
		s.mux.HandleFunc("GET /api/v1/a2a/tasks/{task_id}", s.a2aGetTask)
		s.mux.HandleFunc("POST /api/v1/a2a/tasks/{task_action}", s.a2aCancelTask)
		s.mux.HandleFunc("GET /api/v1/a2a/tasks/{task_id}/subscribe", s.a2aSubscribeTask)
	}
	if s.catalog != nil {
		s.mux.HandleFunc("POST /api/v1/agents", s.createDefinition)
		s.mux.HandleFunc("GET /api/v1/agents", s.listDefinitions)
		s.mux.HandleFunc("GET /api/v1/agents/{agent_id}", s.getDefinition)
		s.mux.HandleFunc("POST /api/v1/agents/{agent_id}/versions", s.createVersion)
		s.mux.HandleFunc("GET /api/v1/agents/{agent_id}/versions", s.listVersions)
		s.mux.HandleFunc("POST /api/v1/agent-versions/{version_action}", s.releaseVersion)
		s.mux.HandleFunc("GET /api/v1/agent-versions", s.listExecutableVersions)
	}
	if s.resources != nil {
		s.mux.HandleFunc("POST /api/v1/prompt-versions", s.createPromptVersion)
		s.mux.HandleFunc("GET /api/v1/prompt-versions", s.listPromptVersions)
		s.mux.HandleFunc("GET /api/v1/prompt-versions/{version_id}", s.getPromptVersion)
		s.mux.HandleFunc("POST /api/v1/tool-versions", s.createToolVersion)
		s.mux.HandleFunc("GET /api/v1/tool-versions", s.listToolVersions)
		s.mux.HandleFunc("GET /api/v1/tool-versions/{version_id}", s.getToolVersion)
		s.mux.HandleFunc("POST /api/v1/toolset-versions", s.createToolSetVersion)
		s.mux.HandleFunc("GET /api/v1/toolset-versions", s.listToolSetVersions)
		s.mux.HandleFunc("GET /api/v1/toolset-versions/{version_id}", s.getToolSetVersion)
		s.mux.HandleFunc("GET /api/v1/environment-templates", s.listEnvironmentTemplates)
		s.mux.HandleFunc("GET /api/v1/dependency-installs", s.listDependencyInstalls)
		s.mux.HandleFunc("POST /api/v1/skill-versions", s.createSkillVersion)
		s.mux.HandleFunc("POST /api/v1/skill-versions:upload", s.uploadSkillVersion)
		s.mux.HandleFunc("GET /api/v1/skill-versions", s.listSkillVersions)
		s.mux.HandleFunc("GET /api/v1/skill-versions/{version_id}", s.getSkillVersion)
		s.mux.HandleFunc("POST /api/v1/skillset-versions", s.createSkillSetVersion)
		s.mux.HandleFunc("GET /api/v1/skillset-versions", s.listSkillSetVersions)
		s.mux.HandleFunc("GET /api/v1/skillset-versions/{version_id}", s.getSkillSetVersion)
	}
	if s.sessions != nil {
		s.mux.HandleFunc("POST /api/v1/sessions", s.createSession)
		s.mux.HandleFunc("GET /api/v1/sessions", s.listSessions)
		s.mux.HandleFunc("GET /api/v1/sessions/{session_id}", s.getSession)
		s.mux.HandleFunc("GET /api/v1/sessions/{session_id}/runs", s.listSessionRuns)
		if s.sessionAudits != nil {
			s.mux.HandleFunc("GET /api/v1/sessions/{session_id}/audit", s.getSessionAudit)
		}
	}
	if s.memories != nil {
		s.mux.HandleFunc("POST /api/v1/memories", s.createMemory)
		s.mux.HandleFunc("GET /api/v1/memories", s.listMemories)
		if s.memoryDetails != nil {
			s.mux.HandleFunc("GET /api/v1/memories/{memory_id}", s.getMemory)
		}
		if s.memoryEdit != nil {
			s.mux.HandleFunc("PATCH /api/v1/memories/{memory_id}", s.updateMemory)
		}
		if s.memoryManifest != nil {
			s.mux.HandleFunc("GET /api/v1/memories/manifest", s.listMemoryManifest)
		}
		s.mux.HandleFunc("DELETE /api/v1/memories/{memory_id}", s.deleteMemory)
		if s.memoryTeams != nil {
			s.mux.HandleFunc("GET /api/v1/memory-teams/members", s.listMemoryTeamMemberships)
			s.mux.HandleFunc("PUT /api/v1/memory-teams/{team_id}/members/{user_id}", s.upsertMemoryTeamMembership)
			s.mux.HandleFunc("DELETE /api/v1/memory-teams/{team_id}/members/{user_id}", s.deleteMemoryTeamMembership)
		}
		if s.memoryFeedback != nil {
			s.mux.HandleFunc("POST /api/v1/memories/{memory_id}/feedback", s.recordMemoryFeedback)
			s.mux.HandleFunc("GET /api/v1/memories/{memory_id}/feedback", s.listMemoryFeedback)
		}
		if s.memoryLifecycle != nil {
			s.mux.HandleFunc("GET /api/v1/memories/{memory_id}/events", s.listMemoryLifecycleEvents)
		}
		if s.memorySources != nil {
			s.mux.HandleFunc("GET /api/v1/memory-sources", s.listMemorySources)
		}
		if s.memorySourceSync != nil {
			s.mux.HandleFunc("POST /api/v1/memory-sources:sync", s.syncMemorySources)
		}
		if s.memoryRetrievals != nil {
			s.mux.HandleFunc("GET /api/v1/runs/{run_id}/memory-retrievals", s.listMemoryRetrievals)
		}
		if s.memoryRunTimeline != nil {
			s.mux.HandleFunc("GET /api/v1/runs/{run_id}/memory-timeline", s.listMemoryTimeline)
		}
		if s.memoryGovernance != nil {
			s.mux.HandleFunc("GET /api/v1/memories/{memory_id}/revisions", s.listMemoryRevisions)
			s.mux.HandleFunc("POST /api/v1/memories/{memory_action}", s.memoryGovernanceAction)
		}
	}
	return s
}

func (s *Server) promoteArtifact(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	actor := strings.TrimSpace(r.Header.Get("X-Actor-ID"))
	if actor == "" {
		writeError(w, http.StatusBadRequest, "missing_actor", "X-Actor-ID header is required")
		return
	}
	id, matched := actionID(r.PathValue("artifact_action"), "promote")
	if !matched || !uuidPattern.MatchString(id) {
		writeError(w, 400, "invalid_request", "artifact action must contain a UUID and :promote")
		return
	}
	if s.artifactPromoter == nil {
		writeError(w, 503, "workspace_promotion_unavailable", "isolated workspace promotion is not configured")
		return
	}
	var request artifact.PromotionRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, 400, "invalid_request", err.Error())
		return
	}
	request.TargetPath = filepath.ToSlash(filepath.Clean(strings.TrimSpace(request.TargetPath)))
	if request.TargetPath == "." || filepath.IsAbs(request.TargetPath) || request.TargetPath == ".." || strings.HasPrefix(request.TargetPath, "../") {
		writeError(w, 400, "invalid_target_path", "target_path must stay inside the project workspace")
		return
	}
	item, _, err := s.artifacts.GetArtifactContentForTenant(r.Context(), tenantID, id)
	if err != nil {
		writeError(w, 404, "artifact_not_found", "artifact was not found")
		return
	}
	promotion, err := s.artifacts.BeginArtifactPromotion(r.Context(), tenantID, id, request, actor)
	if err != nil {
		writeError(w, 409, "artifact_not_promotable", err.Error())
		return
	}
	outcome, promoteErr := s.artifactPromoter.Promote(r.Context(), providersandbox.PromoteRequest{
		RunID: item.RunID, SourcePath: item.Name, SourceHash: item.ContentHash,
		TargetPath: request.TargetPath, ExpectedTargetSHA256: request.ExpectedTargetSHA256,
	})
	result := artifact.PromotionResult{TargetPath: outcome.TargetPath, PreviousTargetSHA256: outcome.PreviousTargetSHA256, ResultTargetSHA256: outcome.ResultTargetSHA256, Created: outcome.Created}
	finished, finishErr := s.artifacts.FinishArtifactPromotion(r.Context(), tenantID, promotion.ID, result, promoteErr)
	if finishErr != nil {
		writeError(w, 500, "artifact_promotion_audit_failed", finishErr.Error())
		return
	}
	if promoteErr != nil {
		writeJSON(w, 409, map[string]any{"error": map[string]any{"code": "artifact_promotion_conflict", "message": promoteErr.Error(), "current_target_sha256": outcome.PreviousTargetSHA256}, "data": finished})
		return
	}
	writeJSON(w, 200, map[string]any{"data": finished})
}

func (s *Server) listRunArtifacts(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	runID := r.PathValue("run_id")
	if !uuidPattern.MatchString(runID) {
		writeError(w, 400, "invalid_request", "run_id must be a UUID")
		return
	}
	items, err := s.artifacts.ListArtifactsForTenant(r.Context(), tenantID, runID, parseBoundedInt(r.URL.Query().Get("limit"), 100, 500))
	if err != nil {
		s.writeRunError(w, err)
		return
	}
	writeJSON(w, 200, map[string]any{"data": items})
}

func (s *Server) getRunManifest(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	runID := r.PathValue("run_id")
	if !uuidPattern.MatchString(runID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "run_id must be a UUID")
		return
	}
	manifest, err := s.manifests.GetRunManifestForTenant(r.Context(), tenantID, runID)
	if err != nil {
		s.writeRunError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": manifest})
}

func (s *Server) listWorkflowArtifacts(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	workflowID := r.PathValue("workflow_id")
	if !uuidPattern.MatchString(workflowID) {
		writeError(w, 400, "invalid_request", "workflow_id must be a UUID")
		return
	}
	items, err := s.workflowArtifacts.ListWorkflowArtifactsForTenant(r.Context(), tenantID, workflowID, parseBoundedInt(r.URL.Query().Get("limit"), 100, 500))
	if err != nil {
		s.writeRunError(w, err)
		return
	}
	writeJSON(w, 200, map[string]any{"data": items})
}

func (s *Server) listRunChildren(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	runID := r.PathValue("run_id")
	if !uuidPattern.MatchString(runID) {
		writeError(w, 400, "invalid_request", "run_id must be a UUID")
		return
	}
	items, err := s.children.ListChildRunsForTenant(r.Context(), tenantID, runID)
	if err != nil {
		writeError(w, 422, "children_list_failed", err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{"data": items})
}

func (s *Server) getArtifactContent(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	id := r.PathValue("artifact_id")
	if !uuidPattern.MatchString(id) {
		writeError(w, 400, "invalid_request", "artifact_id must be a UUID")
		return
	}
	item, content, err := s.artifacts.GetArtifactContentForTenant(r.Context(), tenantID, id)
	if err != nil {
		writeError(w, 404, "artifact_not_found", "artifact was not found")
		return
	}
	w.Header().Set("Content-Type", item.MediaType)
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", strings.ReplaceAll(item.Name, "\"", "")))
	w.Header().Set("ETag", `"`+item.ContentHash+`"`)
	w.WriteHeader(200)
	_, _ = w.Write(content)
}

func (s *Server) wellKnownAgentCard(w http.ResponseWriter, r *http.Request) {
	tenantID := s.a2aDefaultTenant
	if tenantID == "" {
		writeError(w, 503, "a2a_discovery_not_configured", "AGENT_AUTH_DEFAULT_TENANT is required for public A2A discovery")
		return
	}
	agentID := s.a2aDefaultAgent
	if agentID == "" {
		definitions, err := s.catalog.ListDefinitions(r.Context(), tenantID, 100)
		if err != nil {
			s.writeCatalogError(w, err)
			return
		}
		for _, definition := range definitions {
			if definition.ActiveVersionID == nil {
				continue
			}
			if agentID != "" {
				writeError(w, 409, "a2a_default_agent_ambiguous", "multiple published Agents exist; configure AGENT_A2A_DEFAULT_AGENT_ID")
				return
			}
			agentID = definition.ID
		}
	}
	if agentID == "" {
		writeError(w, 404, "a2a_agent_not_found", "no published Agent is available for discovery")
		return
	}
	r.Header.Set("X-Tenant-ID", tenantID)
	r.SetPathValue("agent_id", agentID)
	s.a2aAgentCard(w, r)
}

func (s *Server) a2aAgentCard(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	agentID := r.PathValue("agent_id")
	if !uuidPattern.MatchString(agentID) {
		writeError(w, 400, "invalid_request", "agent_id must be a UUID")
		return
	}
	definition, err := s.catalog.GetDefinition(r.Context(), tenantID, agentID)
	if err != nil {
		s.writeCatalogError(w, err)
		return
	}
	if definition.ActiveVersionID == nil {
		writeError(w, 409, "agent_not_published", "Agent has no active published version")
		return
	}
	versions, err := s.catalog.ListVersions(r.Context(), tenantID, agentID)
	if err != nil {
		s.writeCatalogError(w, err)
		return
	}
	var selected agent.Version
	for _, version := range versions {
		if version.ID == *definition.ActiveVersionID {
			selected = version
			break
		}
	}
	if selected.ID == "" {
		writeError(w, 409, "active_version_missing", "active AgentVersion could not be resolved")
		return
	}
	var spec agent.Spec
	if err := json.Unmarshal(selected.Spec, &spec); err != nil {
		writeError(w, 500, "invalid_agent_spec", err.Error())
		return
	}
	description := spec.Description
	if description == "" && definition.Description != nil {
		description = *definition.Description
	}
	scheme := "http"
	if forwarded := r.Header.Get("X-Forwarded-Proto"); forwarded != "" {
		scheme = forwarded
	}
	base := scheme + "://" + r.Host + "/agent-api/api/v1/a2a/agents/" + agentID
	securitySchemes := map[string]any{}
	requirements := []map[string][]string{}
	if s.securityEnforced {
		securitySchemes["bearerAuth"] = map[string]any{"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
		requirements = append(requirements, map[string][]string{"bearerAuth": {}})
	}
	card := a2a.AgentCard{Name: definition.Name, Description: description, SupportedInterfaces: []a2a.AgentInterface{{URL: base, ProtocolBinding: "HTTP+JSON", ProtocolVersion: a2a.ProtocolVersion}}, Provider: a2a.AgentProvider{Organization: "TwinForge", URL: scheme + "://" + r.Host}, Version: strconv.Itoa(selected.Version), Capabilities: a2a.AgentCapabilities{Streaming: true}, DefaultInputModes: []string{"text/plain", "application/json"}, DefaultOutputModes: []string{"text/plain", "application/json"}, Skills: []a2a.AgentSkill{{ID: definition.Key, Name: spec.Name, Description: description, Tags: []string{"agent", "tool-use", spec.Harness.Name}}}, SecuritySchemes: securitySchemes, SecurityRequirements: requirements}
	writeJSON(w, 200, card)
}

func (s *Server) a2aSendMessage(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	agentID := r.PathValue("agent_id")
	if !uuidPattern.MatchString(agentID) {
		writeError(w, 400, "invalid_request", "agent_id must be a UUID")
		return
	}
	var request a2a.SendMessageRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, 400, "invalid_request", err.Error())
		return
	}
	headers := http.Header{}
	otel.GetTextMapPropagator().Inject(r.Context(), propagation.HeaderCarrier(headers))
	task, err := s.a2aStore.CreateA2ATask(r.Context(), tenantID, agentID, request, optionalHeader(r, "X-Actor-ID"), optionalString(headers.Get("traceparent")))
	if err != nil {
		writeError(w, 422, "a2a_send_failed", err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{"task": task})
}

func (s *Server) a2aGetTask(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	id := r.PathValue("task_id")
	if !uuidPattern.MatchString(id) {
		writeError(w, 400, "invalid_request", "task_id must be a UUID")
		return
	}
	task, err := s.a2aStore.GetA2ATaskForTenant(r.Context(), tenantID, id)
	if err != nil {
		writeError(w, 404, "task_not_found", "A2A task was not found")
		return
	}
	writeJSON(w, 200, task)
}

func (s *Server) a2aCancelTask(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	id, matched := actionID(r.PathValue("task_action"), "cancel")
	if !matched || !uuidPattern.MatchString(id) {
		writeError(w, 400, "invalid_request", "task action must be <uuid>:cancel")
		return
	}
	actor := ""
	if value := optionalHeader(r, "X-Actor-ID"); value != nil {
		actor = *value
	}
	task, err := s.a2aStore.CancelA2ATask(r.Context(), tenantID, id, actor)
	if err != nil {
		writeError(w, 409, "a2a_cancel_failed", err.Error())
		return
	}
	writeJSON(w, 200, task)
}

func (s *Server) a2aSubscribeTask(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	id := r.PathValue("task_id")
	if !uuidPattern.MatchString(id) {
		writeError(w, 400, "invalid_request", "task_id must be a UUID")
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, 500, "stream_unsupported", "streaming is unavailable")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	lastState := ""
	for {
		task, err := s.a2aStore.GetA2ATaskForTenant(r.Context(), tenantID, id)
		if err != nil {
			return
		}
		if task.Status.State != lastState {
			payload, _ := json.Marshal(map[string]any{"statusUpdate": map[string]any{"taskId": task.ID, "contextId": task.ContextID, "status": task.Status, "final": a2aTerminal(task.Status.State)}})
			_, _ = fmt.Fprintf(w, "event: status-update\ndata: %s\n\n", payload)
			flusher.Flush()
			lastState = task.Status.State
		}
		if a2aTerminal(task.Status.State) {
			return
		}
		select {
		case <-r.Context().Done():
			return
		case <-ticker.C:
		}
	}
}

func a2aTerminal(status string) bool {
	return status == "completed" || status == "failed" || status == "canceled" || status == "rejected"
}

func (s *Server) getRunTrajectory(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	runID := r.PathValue("run_id")
	if !uuidPattern.MatchString(runID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "run_id must be a UUID")
		return
	}
	if _, err := s.runs.GetRunForTenant(r.Context(), tenantID, runID); err != nil {
		s.writeRunError(w, err)
		return
	}
	// A projection page cannot split request/completion pairs. Read the ledger
	// in bounded database pages, project complete records, then apply the cursor.
	// The hard ceiling prevents a malformed Run from exhausting API memory.
	var events []event.Event
	var eventCursor int64
	for len(events) < 50000 {
		batch, err := s.runs.ListEventsForTenant(r.Context(), tenantID, runID, eventCursor, 1000)
		if err != nil {
			s.writeRunError(w, err)
			return
		}
		events = append(events, batch...)
		if len(batch) < 1000 {
			break
		}
		eventCursor = batch[len(batch)-1].Sequence
	}
	after, _ := strconv.ParseInt(r.URL.Query().Get("after"), 10, 64)
	page := trajectory.Paginate(trajectory.Project(events), after, parseBoundedInt(r.URL.Query().Get("limit"), 100, 500))
	writeJSON(w, http.StatusOK, map[string]any{"data": page})
}

func (s *Server) listExecutableVersions(w http.ResponseWriter, r *http.Request) {
	tenant, ok := requireTenant(w, r)
	if !ok {
		return
	}
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	versions, err := s.catalog.ListExecutableVersions(r.Context(), tenant, limit)
	if err != nil {
		s.writeCatalogError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": versions})
}

type createSkillVersionRequest struct {
	Key  string     `json:"key"`
	Name string     `json:"name"`
	Spec skill.Spec `json:"spec"`
}

func (s *Server) createSkillVersion(w http.ResponseWriter, r *http.Request) {
	tenant, ok := requireTenant(w, r)
	if !ok {
		return
	}
	var request createSkillVersionRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, 400, "invalid_request", err.Error())
		return
	}
	version, err := s.resources.CreateSkillVersion(r.Context(), skill.CreateVersion{TenantID: tenant, Key: request.Key, Name: request.Name, Spec: request.Spec, CreatedBy: optionalHeader(r, "X-Actor-ID")})
	if err != nil {
		writeError(w, 422, "invalid_skill", err.Error())
		return
	}
	writeJSON(w, 201, map[string]any{"data": version})
}

func (s *Server) uploadSkillVersion(w http.ResponseWriter, r *http.Request) {
	tenant, ok := requireTenant(w, r)
	if !ok {
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxSkillUploadBytes+(64<<10))
	if err := r.ParseMultipartForm(maxSkillUploadBytes); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_skill_upload", "skill upload must be multipart/form-data and no larger than 1 MiB")
		return
	}
	if r.MultipartForm != nil {
		defer r.MultipartForm.RemoveAll()
	}
	file, header, err := r.FormFile("file")
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_skill_upload", "multipart field 'file' is required")
		return
	}
	defer file.Close()
	content, err := io.ReadAll(io.LimitReader(file, maxSkillUploadBytes+1))
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_skill_upload", "skill file could not be read")
		return
	}
	if len(content) > maxSkillUploadBytes {
		writeError(w, http.StatusRequestEntityTooLarge, "skill_upload_too_large", "skill file exceeds 1 MiB")
		return
	}
	imported, err := skill.ParseUpload(header.Filename, content)
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, "invalid_skill_upload", err.Error())
		return
	}
	version, err := s.resources.CreateSkillVersion(r.Context(), skill.CreateVersion{
		TenantID: tenant, Key: imported.Key, Name: imported.Name, Spec: imported.Spec,
		CreatedBy: optionalHeader(r, "X-Actor-ID"),
	})
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, "invalid_skill", err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"data": version})
}
func (s *Server) getSkillVersion(w http.ResponseWriter, r *http.Request) {
	tenant, ok := requireTenant(w, r)
	if !ok {
		return
	}
	id := r.PathValue("version_id")
	if !uuidPattern.MatchString(id) {
		writeError(w, 400, "invalid_request", "version_id must be a UUID")
		return
	}
	version, err := s.resources.GetSkillVersion(r.Context(), tenant, id)
	if err != nil {
		writeError(w, 404, "skill_not_found", "skill version was not found")
		return
	}
	writeJSON(w, 200, map[string]any{"data": version})
}

func (s *Server) listSkillVersions(w http.ResponseWriter, r *http.Request) {
	tenant, ok := requireTenant(w, r)
	if !ok {
		return
	}
	versions, err := s.resources.ListSkillVersions(r.Context(), tenant, parseBoundedInt(r.URL.Query().Get("limit"), 100, 500))
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, "invalid_skill", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": versions})
}

type createSkillSetVersionRequest struct {
	Key  string        `json:"key"`
	Name string        `json:"name"`
	Spec skill.SetSpec `json:"spec"`
}

func (s *Server) createSkillSetVersion(w http.ResponseWriter, r *http.Request) {
	tenant, ok := requireTenant(w, r)
	if !ok {
		return
	}
	var request createSkillSetVersionRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, 400, "invalid_request", err.Error())
		return
	}
	version, err := s.resources.CreateSkillSetVersion(r.Context(), skill.CreateSetVersion{TenantID: tenant, Key: request.Key, Name: request.Name, Spec: request.Spec, CreatedBy: optionalHeader(r, "X-Actor-ID")})
	if err != nil {
		writeError(w, 422, "invalid_skillset", err.Error())
		return
	}
	writeJSON(w, 201, map[string]any{"data": version})
}
func (s *Server) getSkillSetVersion(w http.ResponseWriter, r *http.Request) {
	tenant, ok := requireTenant(w, r)
	if !ok {
		return
	}
	id := r.PathValue("version_id")
	if !uuidPattern.MatchString(id) {
		writeError(w, 400, "invalid_request", "version_id must be a UUID")
		return
	}
	version, err := s.resources.GetSkillSetVersion(r.Context(), tenant, id)
	if err != nil {
		writeError(w, 404, "skillset_not_found", "skill set version was not found")
		return
	}
	writeJSON(w, 200, map[string]any{"data": version})
}

func (s *Server) listSkillSetVersions(w http.ResponseWriter, r *http.Request) {
	tenant, ok := requireTenant(w, r)
	if !ok {
		return
	}
	versions, err := s.resources.ListSkillSetVersions(r.Context(), tenant, parseBoundedInt(r.URL.Query().Get("limit"), 100, 500))
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, "invalid_skillset", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": versions})
}

func (s *Server) listRuns(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	runs, err := s.runs.ListRunsForTenant(r.Context(), tenantID, limit)
	if err != nil {
		s.writeRunError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": runs})
}

func (s *Server) listWorkflows(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	sessionID := strings.TrimSpace(r.URL.Query().Get("session_id"))
	if sessionID != "" && !uuidPattern.MatchString(sessionID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "session_id must be a UUID")
		return
	}
	items, err := s.workflows.ListWorkflowsForTenant(r.Context(), tenantID, sessionID, parseBoundedInt(r.URL.Query().Get("limit"), 50, 200))
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, "invalid_workflow", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": items})
}

func (s *Server) listWorkflowRuns(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	workflowID := r.PathValue("workflow_id")
	if !uuidPattern.MatchString(workflowID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "workflow_id must be a UUID")
		return
	}
	items, err := s.workflowRuns.ListWorkflowRunsForTenant(r.Context(), tenantID, workflowID, parseBoundedInt(r.URL.Query().Get("limit"), 100, 200))
	if err != nil {
		s.writeRunError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": items})
}

type createScoreRequest struct {
	RunID             string         `json:"run_id"`
	ObservationID     string         `json:"observation_id"`
	SessionID         string         `json:"session_id"`
	DatasetRunID      string         `json:"dataset_run_id"`
	Name              string         `json:"name"`
	ScoreType         score.Type     `json:"score_type"`
	Value             *float64       `json:"value"`
	StringValue       *string        `json:"string_value"`
	Source            string         `json:"source"`
	EvaluatorVersion  string         `json:"evaluator_version"`
	AgentVersionID    string         `json:"agent_version_id"`
	ModelResolutionID string         `json:"model_resolution_id"`
	PromptVersionID   string         `json:"prompt_version_id"`
	ToolSetVersionID  string         `json:"toolset_version_id"`
	SkillSetVersionID string         `json:"skillset_version_id"`
	Metadata          map[string]any `json:"metadata"`
}

func (s *Server) createScore(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	var request createScoreRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	createdBy := ""
	if actor := optionalHeader(r, "X-Actor-ID"); actor != nil {
		createdBy = *actor
	}
	item, err := s.scores.CreateScore(r.Context(), score.Create{TenantID: tenantID, RunID: request.RunID, ObservationID: request.ObservationID, SessionID: request.SessionID, DatasetRunID: request.DatasetRunID, Name: request.Name, ScoreType: request.ScoreType, Value: request.Value, StringValue: request.StringValue, Source: request.Source, EvaluatorVersion: request.EvaluatorVersion, AgentVersionID: request.AgentVersionID, ModelResolutionID: request.ModelResolutionID, PromptVersionID: request.PromptVersionID, ToolSetVersionID: request.ToolSetVersionID, SkillSetVersionID: request.SkillSetVersionID, Metadata: request.Metadata, CreatedBy: createdBy})
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, "score_invalid", err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"data": item})
}

func (s *Server) listRunScores(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	runID := r.PathValue("run_id")
	if !uuidPattern.MatchString(runID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "run_id must be a UUID")
		return
	}
	items, err := s.scores.ListScoresForTenant(r.Context(), tenantID, runID, parseBoundedInt(r.URL.Query().Get("limit"), 100, 500))
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, "score_list_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": items})
}

func (s *Server) observabilitySummary(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	summary, err := s.runs.ObservabilitySummary(r.Context(), tenantID)
	if err != nil {
		s.writeRunError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": summary})
}

func (s *Server) storageSummary(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	summary, err := s.storage.StorageSummary(r.Context(), tenantID)
	if err != nil {
		s.writeRunError(w, err)
		return
	}
	if s.hotStore != nil {
		pingCtx, cancel := context.WithTimeout(r.Context(), 500*time.Millisecond)
		err := s.hotStore.Ping(pingCtx)
		cancel()
		if err == nil {
			summary.Redis.Status = "ready"
			summary.Redis.Detail = "PostgreSQL outbox 以至少一次语义发布 Run 唤醒，并写入 observations:raw Stream 指针；SSE 用序号去重，归档 Worker 回读 PG 事实。"
		} else {
			summary.Redis.Status = "degraded"
			summary.Redis.Detail = "Redis 当前不可达；Agent 执行不受影响，SSE 已回退 PostgreSQL 轮询。"
		}
	}
	if s.coldStore != nil {
		pingCtx, cancel := context.WithTimeout(r.Context(), 800*time.Millisecond)
		err := s.coldStore.Ping(pingCtx)
		cancel()
		if err == nil {
			summary.MinIO.Status = "ready"
			summary.MinIO.Detail = "Artifact 与原始 Observation 以内容寻址对象写入 MinIO；PostgreSQL 保存租户归属、哈希、对象键和状态，下载与 Diff 校验长度及 SHA-256。"
		} else {
			summary.MinIO.Status = "degraded"
			summary.MinIO.Detail = "MinIO 当前不可达；已有对象不会伪装成可读，新 Artifact 写入会失败并由 Run 重试。"
		}
	}
	if s.semanticStore != nil {
		status := s.semanticStore.Status()
		if status.Ready {
			summary.PGVector.Status = "ready"
			summary.PGVector.Detail = fmt.Sprintf("Memory 使用 %s（%d 维，%s）写入 pgvector，并以余弦距离、词法、重要度和时效性混合召回。", status.Model, status.Dim, status.Mode)
		} else if status.Configured {
			summary.PGVector.Status = "degraded"
			summary.PGVector.Detail = "pgvector 表结构已启用，但 live embedding 当前不可用；未向量化数据进入补偿队列，召回自动退回 pg_trgm。"
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": summary})
}

type featureCapability struct {
	Key         string `json:"key"`
	Name        string `json:"name"`
	Status      string `json:"status"`
	Frontend    bool   `json:"frontend"`
	Description string `json:"description"`
}

func (s *Server) listWorkerCapabilities(w http.ResponseWriter, r *http.Request) {
	if _, ok := requireTenant(w, r); !ok {
		return
	}
	items, err := s.workerCapabilities.ListWorkerCapabilities(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, "worker_capabilities_unavailable", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": items})
}

// capabilities exposes an honest runtime contract. A declared or missing
// capability must never be presented as production-ready by the console.
func (s *Server) capabilities(w http.ResponseWriter, r *http.Request) {
	if _, ok := requireTenant(w, r); !ok {
		return
	}
	features := []featureCapability{
		{Key: "trusted_identity", Name: "可信身份与租户", Status: capabilityStatus(s.securityEnforced, "enforced"), Frontend: true, Description: identityDescription(s.securityEnforced)},
		{Key: "harness_registry", Name: "Harness 注册与校验", Status: "enforced", Frontend: true, Description: "AgentVersion 仅允许引用当前 Worker 已注册的 Harness。"},
		{Key: "durable_runs", Name: "持久化 Run 与恢复", Status: "enforced", Frontend: true, Description: "Run 快照、Worker 租约、fencing、checkpoint 与结构化执行账本已进入执行链路。"},
		{Key: "redis_realtime", Name: "Redis 实时通知", Status: capabilityStatus(s.hotStore != nil, "available"), Frontend: true, Description: "Event 与 outbox 在 PostgreSQL 同事务提交，Redis 仅负责低延迟唤醒；失败时自动退避重试并回退事实表轮询。"},
		{Key: "autonomous_planning", Name: "自适应自主规划", Status: capabilityStatus(s.planStore != nil, "available"), Frontend: true, Description: "默认由 Agent 判断直接回答或规划执行；实质工具任务必须先建立持久化 Plan，只允许更新活跃 Todo，按 tool_hints 最小暴露能力，并在 Todo 或验收条件未闭环时阻止完成。"},
		{Key: "evidence_gate", Name: "验收证据门禁", Status: capabilityStatus(s.planStore != nil, "enforced"), Frontend: true, Description: "模型只提交验收状态；Runtime 自动匹配同一 Run、同一 Todo 的成功 Tool 回执，并持续复核工具语义、目标路径和文件变更后的证据新鲜度。"},
		{Key: "final_output_integrity", Name: "最终结果完整性", Status: "enforced", Frontend: true, Description: "空结果、上下文压缩摘要、运行时控制块与私有推理标记不能完成 Run；拒绝事件会进入对话执行流和 Event Ledger。"},
		{Key: "user_interaction", Name: "执行中询问与续跑", Status: capabilityStatus(s.interactionStore != nil, "available"), Frontend: true, Description: "ask_user 将 Run 可靠挂起为 waiting_input；用户回答后从原 Tool Call 检查点幂等恢复。"},
		{Key: "context_compaction", Name: "运行时上下文压缩", Status: "enforced", Frontend: true, Description: "在输入预算硬上限前压缩旧执行历史，保留系统契约和近期完整工具交互，并记录 CONTEXT_COMPACTED 事件。"},
		{Key: "model_resolution", Name: "模型发现与 Run 冻结", Status: "enforced", Frontend: true, Description: "自动探测候选服务与 /v1/models；选中模型、版本和制品摘要按 Worker lease 冻结到 Run。"},
		{Key: "structured_identity", Name: "结构化角色身份", Status: "enforced", Frontend: true, Description: "角色、目标、责任、边界和沟通风格按确定顺序编译进不可裁剪上下文；Identity Digest 与可读快照写入 Run Event。"},
		{Key: "studio_test_runs", Name: "AgentVersion 试运行", Status: "available", Frontend: true, Description: "Studio 可对草稿或发布版本创建隔离 Session；草稿权限仅对显式 studio_test 开放，Run 仍冻结完整版本快照并进入真实 Worker 与轨迹链路。"},
		{Key: "sessions", Name: "多 Session 对话", Status: capabilityStatus(s.sessions != nil, "available"), Frontend: true, Description: "可创建、切换 Session，并查看同一 Session 的完整 Run 对话。"},
		{Key: "skill_context", Name: "Skill 上下文完整性", Status: "enforced", Frontend: true, Description: "Skill 指令按优先级注入不可裁剪区；示例仍可按预算裁剪。"},
		{Key: "tool_contracts", Name: "Schema 运行时契约", Status: "enforced", Frontend: true, Description: "Agent 输入输出与 Tool 参数结果均按 JSON Schema 2020-12 强制校验。"},
		{Key: "denied_tools", Name: "DENIED 工具阻断", Status: "enforced", Frontend: true, Description: "DENIED 工具不暴露给模型，直接调用也不会进入 handler。"},
		{Key: "sandbox", Name: "隔离 Sandbox 执行", Status: "available", Frontend: true, Description: "生产部署由独立、无外网 Sandbox 服务执行文件工具；每个 Run 使用独立目录，容器丢弃全部 Linux capabilities。"},
		{Key: "workspace_tools", Name: "工作区文件与产物", Status: "available", Frontend: true, Description: "支持 read/list/search/write/append/edit/run；拦截路径逃逸与敏感文件，写入使用 SHA-256 乐观锁，并持久化 Diff 和文件快照 Artifact。"},
		{Key: "artifact_object_store", Name: "Artifact 对象存储", Status: capabilityStatus(s.coldStore != nil, "available"), Frontend: true, Description: "Artifact 元数据与租户归属保存在 PostgreSQL，内容写入 MinIO 内容寻址对象；读取、Diff 与 Promotion 强制校验长度和 SHA-256。"},
		{Key: "approvals", Name: "风险工具审批", Status: capabilityStatus(s.approvals != nil, "available"), Frontend: true, Description: "LOW_WRITE/HIGH_RISK 可按 AgentVersion 策略持久化暂停；审批绑定原始 Call Hash，决策后从 checkpoint 恢复。"},
		{Key: "trajectory_projection", Name: "Trajectory 后端投影", Status: "available", Frontend: true, Description: "从不可变 Event Ledger 确定性合并模型和工具生命周期，提供稳定游标分页；前端使用虚拟化账本。"},
		{Key: "layered_memory", Name: "分层记忆", Status: capabilityStatus(s.memories != nil, "available"), Frontend: true, Description: memoryDescription(s.memories != nil)},
		{Key: "semantic_memory", Name: "pgvector 语义记忆", Status: semanticCapabilityStatus(s.semanticStore), Frontend: true, Description: "使用模型版本隔离的 1024 维 embedding、HNSW 候选检索与向量/词法/重要度/时效性混合排序；非 live 响应明确降级。"},
		{Key: "mcp", Name: "MCP 工具协议", Status: capabilityStatus(s.mcpStore != nil, "available"), Frontend: true, Description: "支持 MCP 2025-06-18 Streamable HTTP、会话协商、连接健康、tools/list Schema 快照与精确版本运行绑定。"},
		{Key: "internal_delegation", Name: "内部 Agent 委派", Status: capabilityStatus(s.children != nil, "available"), Frontend: true, Description: "delegate_agent 使用目标白名单和不可变版本，建立父子 Run；执行深度、单步 fan-out、子任务数量和模型/工具/Token 预留预算，并支持等待恢复与级联取消。"},
		{Key: "a2a", Name: "A2A 通信", Status: capabilityStatus(s.a2aStore != nil && s.catalog != nil, "available"), Frontend: true, Description: "提供 A2A 1.0 HTTP+JSON Agent Card、Message/Task、取消、SSE 状态订阅，并将 Run 输出映射为 Artifact。"},
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": map[string]any{
		"framework_version": "0.12.0", "harnesses": harness.Descriptors(), "features": features,
	}})
}

func memoryDescription(enabled bool) string {
	if enabled {
		return "租户/Agent/用户/Session 四级显式记忆支持 TTL、软删除、向量与词法可解释召回、六层来源/四类语义和独立上下文预算；Auto 抽取、revision、生命周期账本与治理入口按 Agent Memory policy 启用。"
	}
	return "分层记忆存储与召回不可用。"
}

func semanticCapabilityStatus(readiness SemanticReadiness) string {
	if readiness == nil {
		return "missing"
	}
	if readiness.Status().Ready {
		return "available"
	}
	return "declared"
}

func identityDescription(enforced bool) string {
	if enforced {
		return "JWT 派生 tenant/actor，客户端转发头会被覆盖。"
	}
	return "开发开放模式：仍信任 X-Tenant-ID，不允许用于生产。"
}

func capabilityStatus(enabled bool, value string) string {
	if enabled {
		return value
	}
	return "missing"
}

func parseBoundedInt(raw string, fallback, maximum int) int {
	value, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || value <= 0 {
		return fallback
	}
	if value > maximum {
		return maximum
	}
	return value
}

type createSessionRequest struct {
	AgentID  string          `json:"agent_id"`
	UserID   *string         `json:"user_id"`
	Metadata json.RawMessage `json:"metadata"`
}

func (s *Server) createSession(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	var request createSessionRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	if !uuidPattern.MatchString(request.AgentID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "agent_id must be a UUID")
		return
	}
	session, err := s.sessions.CreateSession(r.Context(), agent.CreateSession{
		TenantID: tenantID, AgentID: request.AgentID,
		UserID: request.UserID, Metadata: request.Metadata,
	})
	if err != nil {
		s.writeSessionError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"data": session})
}

func (s *Server) listSessions(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	agentID := strings.TrimSpace(r.URL.Query().Get("agent_id"))
	if agentID != "" && !uuidPattern.MatchString(agentID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "agent_id must be a UUID")
		return
	}
	limit := parseBoundedInt(r.URL.Query().Get("limit"), 50, 100)
	sessions, err := s.sessions.ListSessions(r.Context(), tenantID, agentID, limit)
	if err != nil {
		s.writeSessionError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": sessions})
}

func (s *Server) getSession(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	sessionID := r.PathValue("session_id")
	if !uuidPattern.MatchString(sessionID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "session_id must be a UUID")
		return
	}
	session, err := s.sessions.GetSession(r.Context(), tenantID, sessionID)
	if err != nil {
		s.writeSessionError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": session})
}

func (s *Server) listSessionRuns(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	sessionID := r.PathValue("session_id")
	if !uuidPattern.MatchString(sessionID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "session_id must be a UUID")
		return
	}
	limit := parseBoundedInt(r.URL.Query().Get("limit"), 100, 200)
	runs, err := s.sessions.ListSessionRuns(r.Context(), tenantID, sessionID, limit)
	if err != nil {
		s.writeSessionError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": runs})
}

func (s *Server) getSessionAudit(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	sessionID := r.PathValue("session_id")
	if !uuidPattern.MatchString(sessionID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "session_id must be a UUID")
		return
	}
	audit, err := s.sessionAudits.GetSessionAudit(r.Context(), tenantID, sessionID)
	if err != nil {
		s.writeSessionError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": audit})
}

func (s *Server) writeSessionError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, agent.ErrSessionNotFound), errors.Is(err, agent.ErrDefinitionNotFound):
		writeError(w, http.StatusNotFound, "session_not_found", "session or agent was not found")
	default:
		writeError(w, http.StatusUnprocessableEntity, "invalid_session", err.Error())
	}
}

type createPromptVersionRequest struct {
	Key     string `json:"key"`
	Name    string `json:"name"`
	Content string `json:"content"`
}

func (s *Server) createPromptVersion(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	var request createPromptVersionRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	version, err := s.resources.CreatePromptVersion(r.Context(), resource.CreatePromptVersion{
		TenantID: tenantID, Key: request.Key, Name: request.Name, Content: request.Content,
		CreatedBy: optionalHeader(r, "X-Actor-ID"),
	})
	if err != nil {
		s.writeResourceError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"data": version})
}

func (s *Server) getPromptVersion(w http.ResponseWriter, r *http.Request) {
	tenantID, versionID, ok := resourceRequest(w, r)
	if !ok {
		return
	}
	version, err := s.resources.GetPromptVersion(r.Context(), tenantID, versionID)
	if err != nil {
		s.writeResourceError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": version})
}

func (s *Server) listPromptVersions(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	versions, err := s.resources.ListPromptVersions(r.Context(), tenantID, parseBoundedInt(r.URL.Query().Get("limit"), 100, 500))
	if err != nil {
		s.writeResourceError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": versions})
}

type createToolVersionRequest struct {
	Key  string            `json:"key"`
	Name string            `json:"name"`
	Spec resource.ToolSpec `json:"spec"`
}

func (s *Server) createToolVersion(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	var request createToolVersionRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	version, err := s.resources.CreateToolVersion(r.Context(), resource.CreateToolVersion{
		TenantID: tenantID, Key: request.Key, Name: request.Name, Spec: request.Spec,
		CreatedBy: optionalHeader(r, "X-Actor-ID"),
	})
	if err != nil {
		s.writeResourceError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"data": version})
}

func (s *Server) getToolVersion(w http.ResponseWriter, r *http.Request) {
	tenantID, versionID, ok := resourceRequest(w, r)
	if !ok {
		return
	}
	version, err := s.resources.GetToolVersion(r.Context(), tenantID, versionID)
	if err != nil {
		s.writeResourceError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": version})
}

func (s *Server) listToolVersions(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	versions, err := s.resources.ListToolVersions(r.Context(), tenantID, parseBoundedInt(r.URL.Query().Get("limit"), 100, 500))
	if err != nil {
		s.writeResourceError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": versions})
}

type createToolSetVersionRequest struct {
	Key  string               `json:"key"`
	Name string               `json:"name"`
	Spec resource.ToolSetSpec `json:"spec"`
}

func (s *Server) createToolSetVersion(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	var request createToolSetVersionRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	version, err := s.resources.CreateToolSetVersion(r.Context(), resource.CreateToolSetVersion{
		TenantID: tenantID, Key: request.Key, Name: request.Name, Spec: request.Spec,
		CreatedBy: optionalHeader(r, "X-Actor-ID"),
	})
	if err != nil {
		s.writeResourceError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"data": version})
}

func (s *Server) getToolSetVersion(w http.ResponseWriter, r *http.Request) {
	tenantID, versionID, ok := resourceRequest(w, r)
	if !ok {
		return
	}
	version, err := s.resources.GetToolSetVersion(r.Context(), tenantID, versionID)
	if err != nil {
		s.writeResourceError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": version})
}

func (s *Server) listToolSetVersions(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	versions, err := s.resources.ListToolSetVersions(r.Context(), tenantID, parseBoundedInt(r.URL.Query().Get("limit"), 100, 500))
	if err != nil {
		s.writeResourceError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": versions})
}

func (s *Server) listEnvironmentTemplates(w http.ResponseWriter, r *http.Request) {
	if _, ok := requireTenant(w, r); !ok {
		return
	}
	items, err := s.resources.ListEnvironmentTemplates(r.Context())
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, "environment_catalog_unavailable", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": items})
}

func (s *Server) listDependencyInstalls(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	runID := strings.TrimSpace(r.URL.Query().Get("run_id"))
	if runID != "" && !uuidPattern.MatchString(runID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "run_id must be a UUID")
		return
	}
	items, err := s.resources.ListDependencyInstalls(r.Context(), tenantID, runID, parseBoundedInt(r.URL.Query().Get("limit"), 100, 200))
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, "dependency_audit_unavailable", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": items})
}

func resourceRequest(w http.ResponseWriter, r *http.Request) (string, string, bool) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return "", "", false
	}
	versionID := r.PathValue("version_id")
	if !uuidPattern.MatchString(versionID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "version_id must be a UUID")
		return "", "", false
	}
	return tenantID, versionID, true
}

func (s *Server) writeResourceError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, resource.ErrPromptVersionNotFound),
		errors.Is(err, resource.ErrToolVersionNotFound),
		errors.Is(err, resource.ErrToolSetVersionNotFound):
		writeError(w, http.StatusNotFound, "resource_not_found", "versioned resource was not found")
	default:
		writeError(w, http.StatusUnprocessableEntity, "invalid_resource", err.Error())
	}
}

func (s *Server) listRunEvents(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	runID := r.PathValue("run_id")
	if !uuidPattern.MatchString(runID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "run_id must be a UUID")
		return
	}
	after, _ := strconv.ParseInt(r.URL.Query().Get("after"), 10, 64)
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	events, err := s.runs.ListEventsForTenant(r.Context(), tenantID, runID, after, limit)
	if err != nil {
		s.writeRunError(w, err)
		return
	}
	if events == nil {
		events = []event.Event{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": events})
}

func (s *Server) streamRunEvents(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	runID := r.PathValue("run_id")
	if !uuidPattern.MatchString(runID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "run_id must be a UUID")
		return
	}
	openedRun, err := s.runs.GetRunForTenant(r.Context(), tenantID, runID)
	if err != nil {
		s.writeRunError(w, err)
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "stream_unsupported", "streaming is unavailable")
		return
	}
	after, _ := strconv.ParseInt(r.Header.Get("Last-Event-ID"), 10, 64)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	var notifications <-chan struct{}
	pollInterval := time.Second
	if s.runEvents != nil {
		if channel, closeSubscription, err := s.runEvents.SubscribeRun(r.Context(), runID); err == nil {
			notifications = channel
			pollInterval = 15 * time.Second
			defer closeSubscription()
		}
	}
	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()
	terminalAtOpen := openedRun.Status.Terminal()
	for {
		events, err := s.runs.ListEventsForTenant(r.Context(), tenantID, runID, after, 200)
		if err != nil {
			if after == 0 {
				// Headers may not have been committed yet on the first iteration.
				s.writeRunError(w, err)
			}
			return
		}
		terminal := false
		for _, committed := range events {
			encoded, err := json.Marshal(committed)
			if err != nil {
				return
			}
			_, _ = fmt.Fprintf(w, "id: %d\nevent: %s\ndata: %s\n\n", committed.Sequence, committed.Type, encoded)
			after = committed.Sequence
			terminal = terminal || terminalEvent(committed.Type)
		}
		if len(events) != 0 {
			flusher.Flush()
		}
		if terminal || (terminalAtOpen && len(events) < 200) {
			return
		}
		// Drain an existing backlog without waiting for a heartbeat or Redis
		// signal between pages.
		if len(events) == 200 {
			continue
		}
		select {
		case <-r.Context().Done():
			return
		case _, open := <-notifications:
			if !open {
				notifications = nil
				ticker.Reset(time.Second)
			}
		case <-ticker.C:
			_, _ = io.WriteString(w, ": heartbeat\n\n")
			flusher.Flush()
		}
	}
}

func terminalEvent(eventType event.Type) bool {
	return eventType == event.RunCompleted || eventType == event.RunFailed || eventType == event.RunCancelled
}

// Handler returns the complete HTTP handler.
func (s *Server) Handler() http.Handler {
	return s.mux
}

func (s *Server) live(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"service": "agent-platform", "status": "live"})
}

func (s *Server) ready(w http.ResponseWriter, r *http.Request) {
	if s.readiness != nil {
		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
		defer cancel()
		if err := s.readiness.Ping(ctx); err != nil {
			writeError(w, http.StatusServiceUnavailable, "not_ready", "database is unavailable")
			return
		}
	}
	writeJSON(w, http.StatusOK, map[string]string{"service": "agent-platform", "status": "ready"})
}

type createRunRequest struct {
	SessionID *string `json:"session_id"`
	// WorkflowID explicitly selects an existing long-lived task. Omitting it
	// keeps the default "continue the latest resumable Workflow in Session"
	// behavior; a new Workflow is created only when no resumable one exists.
	WorkflowID     *string         `json:"workflow_id,omitempty"`
	NewWorkflow    bool            `json:"new_workflow,omitempty"`
	RoutingIntent  string          `json:"routing_intent,omitempty"`
	AgentVersionID string          `json:"agent_version_id"`
	TriggerType    string          `json:"trigger_type"`
	Input          json.RawMessage `json:"input"`
}

func (s *Server) createRun(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	var request createRunRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	if !uuidPattern.MatchString(request.AgentVersionID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "agent_version_id must be a UUID")
		return
	}
	if request.SessionID != nil && !uuidPattern.MatchString(*request.SessionID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "session_id must be a UUID")
		return
	}
	if request.WorkflowID != nil && !uuidPattern.MatchString(*request.WorkflowID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "workflow_id must be a UUID")
		return
	}
	createdBy := optionalHeader(r, "X-Actor-ID")
	allowDraft := request.TriggerType == "studio_test"
	if allowDraft && createdBy == nil {
		writeError(w, http.StatusBadRequest, "actor_required", "X-Actor-ID is required for Studio test runs")
		return
	}
	traceHeaders := http.Header{}
	otel.GetTextMapPropagator().Inject(r.Context(), propagation.HeaderCarrier(traceHeaders))
	traceParent := optionalString(traceHeaders.Get("traceparent"))
	run, err := s.runs.CreateRun(r.Context(), agent.CreateRun{
		TenantID: tenantID, SessionID: request.SessionID,
		WorkflowID:     request.WorkflowID,
		NewWorkflow:    request.NewWorkflow,
		RoutingIntent:  request.RoutingIntent,
		AgentVersionID: request.AgentVersionID, TriggerType: request.TriggerType, AllowDraft: allowDraft,
		Input: request.Input, CreatedBy: createdBy, TraceParent: traceParent,
	})
	if err != nil {
		slog.Error("create agent run failed", "tenant_id", tenantID, "agent_version_id", request.AgentVersionID, "err", err)
		s.writeRunError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"data": run})
}

func (s *Server) getRun(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	runID := r.PathValue("run_id")
	if !uuidPattern.MatchString(runID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "run_id must be a UUID")
		return
	}
	run, err := s.runs.GetRunForTenant(r.Context(), tenantID, runID)
	if err != nil {
		s.writeRunError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": run})
}

func (s *Server) cancelRun(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	runID, ok := actionID(r.PathValue("run_action"), "cancel")
	if !ok {
		http.NotFound(w, r)
		return
	}
	if !uuidPattern.MatchString(runID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "run_id must be a UUID")
		return
	}
	actor := strings.TrimSpace(r.Header.Get("X-Actor-ID"))
	if actor == "" {
		writeError(w, http.StatusBadRequest, "missing_actor", "X-Actor-ID header is required")
		return
	}
	if err := s.runs.RequestCancelForTenant(r.Context(), tenantID, runID, actor); err != nil {
		s.writeRunError(w, err)
		return
	}
	w.WriteHeader(http.StatusAccepted)
}

type createDefinitionRequest struct {
	Key         string  `json:"key"`
	Name        string  `json:"name"`
	Description *string `json:"description"`
	Owner       *string `json:"owner"`
}

func (s *Server) createDefinition(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	var request createDefinitionRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	definition, err := s.catalog.CreateDefinition(r.Context(), agent.CreateDefinition{
		TenantID: tenantID, Key: request.Key, Name: request.Name,
		Description: request.Description, Owner: request.Owner,
	})
	if err != nil {
		s.writeCatalogError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"data": definition})
}

func (s *Server) listDefinitions(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	definitions, err := s.catalog.ListDefinitions(r.Context(), tenantID, limit)
	if err != nil {
		s.writeCatalogError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": definitions})
}

func (s *Server) getDefinition(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	definitionID := r.PathValue("agent_id")
	if !uuidPattern.MatchString(definitionID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "agent_id must be a UUID")
		return
	}
	definition, err := s.catalog.GetDefinition(r.Context(), tenantID, definitionID)
	if err != nil {
		s.writeCatalogError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": definition})
}

type createVersionRequest struct {
	Spec agent.Spec `json:"spec"`
}

func (s *Server) createVersion(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	definitionID := r.PathValue("agent_id")
	if !uuidPattern.MatchString(definitionID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "agent_id must be a UUID")
		return
	}
	var request createVersionRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	version, err := s.catalog.CreateVersion(r.Context(), agent.CreateVersion{
		TenantID: tenantID, AgentID: definitionID, Spec: request.Spec,
		CreatedBy: optionalHeader(r, "X-Actor-ID"),
	})
	if err != nil {
		s.writeCatalogError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"data": version})
}

func (s *Server) listVersions(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	definitionID := r.PathValue("agent_id")
	if !uuidPattern.MatchString(definitionID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "agent_id must be a UUID")
		return
	}
	versions, err := s.catalog.ListVersions(r.Context(), tenantID, definitionID)
	if err != nil {
		s.writeCatalogError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": versions})
}

func (s *Server) releaseVersion(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	versionID, ok := actionID(r.PathValue("version_action"), "release")
	if !ok {
		http.NotFound(w, r)
		return
	}
	if !uuidPattern.MatchString(versionID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "version_id must be a UUID")
		return
	}
	version, err := s.catalog.ReleaseVersion(r.Context(), tenantID, versionID)
	if err != nil {
		s.writeCatalogError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": version})
}

type createMCPServerRequest struct {
	Key  string         `json:"key"`
	Name string         `json:"name"`
	Spec mcp.ServerSpec `json:"spec"`
}

func (s *Server) createMCPServer(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	var request createMCPServerRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, 400, "invalid_request", err.Error())
		return
	}
	if strings.TrimSpace(request.Key) == "" || strings.TrimSpace(request.Name) == "" {
		writeError(w, 422, "invalid_mcp_server", "key and name are required")
		return
	}
	if err := request.Spec.Validate(s.mcpAllowedHosts); err != nil {
		writeError(w, 422, "invalid_mcp_server", err.Error())
		return
	}
	item, err := s.mcpStore.CreateMCPServerVersion(r.Context(), mcp.CreateServerVersion{TenantID: tenantID, Key: request.Key, Name: request.Name, Spec: request.Spec, CreatedBy: optionalHeader(r, "X-Actor-ID")})
	if err != nil {
		writeError(w, 422, "invalid_mcp_server", err.Error())
		return
	}
	writeJSON(w, 201, map[string]any{"data": item})
}

func (s *Server) listMCPServers(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	items, err := s.mcpStore.ListMCPServersForTenant(r.Context(), tenantID, parseBoundedInt(r.URL.Query().Get("limit"), 100, 200))
	if err != nil {
		writeError(w, 422, "mcp_list_failed", err.Error())
		return
	}
	if items == nil {
		items = []mcp.ServerVersion{}
	}
	writeJSON(w, 200, map[string]any{"data": items})
}

func (s *Server) mcpServerAction(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	value := r.PathValue("mcp_action")
	id, isTest := actionID(value, "test")
	action := "test"
	if !isTest {
		id, ok = actionID(value, "sync-tools")
		action = "sync"
	} else {
		ok = true
	}
	if !ok || !uuidPattern.MatchString(id) {
		writeError(w, 400, "invalid_request", "MCP action must be :test or :sync-tools")
		return
	}
	version, err := s.mcpStore.GetMCPServerVersionForTenant(r.Context(), tenantID, id)
	if err != nil {
		writeError(w, 404, "mcp_server_not_found", "MCP server version was not found")
		return
	}
	var spec mcp.ServerSpec
	if err := json.Unmarshal(version.Spec, &spec); err != nil {
		writeError(w, 422, "invalid_mcp_server", err.Error())
		return
	}
	if err := spec.Validate(s.mcpAllowedHosts); err != nil {
		writeError(w, 422, "invalid_mcp_server", err.Error())
		return
	}
	client, err := mcp.NewClient(spec, s.mcpAllowedHosts, s.lookupEnv, http.DefaultClient)
	started := time.Now()
	health := mcp.Health{Status: "healthy", ProtocolVersion: spec.EffectiveProtocolVersion(), CheckedAt: time.Now().UTC()}
	if err == nil {
		if action == "sync" {
			var tools []mcp.ToolSnapshot
			tools, err = client.ListTools(r.Context())
			if err == nil {
				err = s.mcpStore.ReplaceMCPToolSnapshots(r.Context(), tenantID, id, tools)
				version.Tools = tools
			}
		} else {
			err = client.Initialize(r.Context())
		}
	}
	health.LatencyMS = time.Since(started).Milliseconds()
	if err != nil {
		health.Status = "unhealthy"
		health.Error = err.Error()
	}
	_ = s.mcpStore.SaveMCPHealth(r.Context(), id, health)
	version.Health = &health
	if err != nil {
		writeJSON(w, 502, map[string]any{"data": version, "error": map[string]string{"code": "mcp_connection_failed", "message": err.Error()}})
		return
	}
	writeJSON(w, 200, map[string]any{"data": version})
}

func (s *Server) listApprovals(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	items, err := s.approvals.ListApprovalsForTenant(r.Context(), tenantID, strings.TrimSpace(r.URL.Query().Get("status")), parseBoundedInt(r.URL.Query().Get("limit"), 100, 500))
	if err != nil {
		writeError(w, 422, "approval_list_failed", err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{"data": items})
}

func (s *Server) decideApproval(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	value := r.PathValue("approval_action")
	id, matched := actionID(value, "approve")
	approved := matched
	if !matched {
		id, matched = actionID(value, "reject")
	}
	if !matched || !uuidPattern.MatchString(id) {
		writeError(w, 400, "invalid_request", "approval action must contain a UUID and :approve or :reject")
		return
	}
	var request struct {
		Reason string `json:"reason"`
	}
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, 400, "invalid_request", err.Error())
		return
	}
	actor := ""
	if value := optionalHeader(r, "X-Actor-ID"); value != nil {
		actor = *value
	}
	item, err := s.approvals.DecideApproval(r.Context(), tenantID, id, approval.Decision{Approved: approved, Reason: request.Reason, ActorID: actor})
	if err != nil {
		writeError(w, 409, "approval_decision_failed", err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{"data": item})
}

func actionID(value, action string) (string, bool) {
	suffix := ":" + action
	if !strings.HasSuffix(value, suffix) {
		return "", false
	}
	return strings.TrimSuffix(value, suffix), true
}

func (s *Server) writeCatalogError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, agent.ErrDefinitionNotFound), errors.Is(err, agent.ErrVersionNotFound):
		writeError(w, http.StatusNotFound, "catalog_not_found", "agent or version was not found")
	case errors.Is(err, agent.ErrDefinitionConflict):
		writeError(w, http.StatusConflict, "agent_key_conflict", "agent key already exists in this tenant")
	default:
		writeError(w, http.StatusUnprocessableEntity, "invalid_agent", err.Error())
	}
}

func (s *Server) writeRunError(w http.ResponseWriter, err error) {
	var ambiguous *agent.WorkflowAmbiguousError
	switch {
	case errors.Is(err, agent.ErrRunNotFound):
		writeError(w, http.StatusNotFound, "run_not_found", "run was not found")
	case errors.Is(err, agent.ErrRunBindingInvalid):
		writeError(w, http.StatusUnprocessableEntity, "invalid_binding", "published agent version or session is unavailable for this tenant")
	case errors.Is(err, agent.ErrRunInputInvalid):
		writeError(w, http.StatusUnprocessableEntity, "invalid_input", err.Error())
	case errors.Is(err, agent.ErrRunTerminal):
		writeError(w, http.StatusConflict, "run_terminal", "terminal run cannot be cancelled")
	case errors.Is(err, agent.ErrWorkflowBusy):
		writeError(w, http.StatusConflict, "workflow_busy", "workflow already has an active run; continue the existing run or choose another task")
	case errors.Is(err, agent.ErrWorkflowRoutingInvalid):
		writeError(w, http.StatusUnprocessableEntity, "workflow_routing_invalid", err.Error())
	case errors.As(err, &ambiguous):
		writeJSON(w, http.StatusConflict, map[string]any{
			"error": map[string]any{"code": "workflow_ambiguous", "message": ambiguous.Error()},
			"data":  map[string]any{"candidates": ambiguous.Candidates},
		})
	default:
		writeError(w, http.StatusInternalServerError, "internal_error", "request could not be completed")
	}
}

func requireTenant(w http.ResponseWriter, r *http.Request) (string, bool) {
	tenantID := strings.TrimSpace(r.Header.Get("X-Tenant-ID"))
	if tenantID == "" {
		writeError(w, http.StatusBadRequest, "missing_tenant", "X-Tenant-ID header is required")
		return "", false
	}
	return tenantID, true
}

func optionalHeader(r *http.Request, name string) *string {
	value := strings.TrimSpace(r.Header.Get(name))
	if value == "" {
		return nil
	}
	return &value
}

func optionalString(value string) *string {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil
	}
	return &value
}

func decodeJSON(w http.ResponseWriter, r *http.Request, destination any) error {
	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBytes)
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return fmt.Errorf("invalid JSON body: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return errors.New("request body must contain one JSON object")
	}
	return nil
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, map[string]any{"error": map[string]string{"code": code, "message": message}})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
