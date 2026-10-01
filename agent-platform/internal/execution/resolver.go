// Package execution resolves immutable Run bindings into executable providers.
package execution

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/persistence/postgres"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/provider/dependency"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/provider/httptool"
	openai "github.com/heurry/cloudnative-infra-platform/agent-platform/internal/provider/openai"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/provider/projectaccess"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/provider/sandbox"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/provider/workspace"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/delegation"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/harness"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/interaction"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/mcp"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/resource"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/skill"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/taskplan"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

// ModelService maps an Agent capability reference to deployment connection data.
// API key values remain in the named environment variable.
type ModelService struct {
	Endpoint string `json:"endpoint"`
	Model    string `json:"model"`
	// ContextWindowTokens is a compatibility fallback for gateways that do not
	// expose OpenAI /v1/models metadata. Direct vLLM services should omit it.
	ContextWindowTokens int      `json:"context_window_tokens,omitempty"`
	DiscoveryStrategy   string   `json:"discovery_strategy,omitempty"`
	APIKeyEnvironment   string   `json:"api_key_environment,omitempty"`
	Capabilities        []string `json:"capabilities,omitempty"`
}

// Config contains deployment policy, not Agent business configuration.
type Config struct {
	ModelServices               map[string]ModelService
	ToolAllowedHosts            []string
	WorkspaceRoot               string
	WorkspaceAllowWrite         bool
	WorkspaceMode               string
	SandboxEndpoint             string
	SandboxToken                string
	DependencyInstallerEndpoint string
	DependencyInstallerToken    string
	ProjectAccessEndpoint       string
	ProjectAccessToken          string
	DependencySources           map[string]string
	EnvironmentTemplate         string
	EnvironmentDependencies     []string
	HTTPClient                  *http.Client
	LookupEnv                   func(string) (string, bool)
}

// Resolver loads versioned resources from PostgreSQL and constructs providers.
type Resolver struct {
	store                   *postgres.RunStore
	modelServices           map[string]ModelService
	toolAllowedHosts        []string
	workspaceRoot           string
	workspaceAllowWrite     bool
	workspaceMode           string
	sandboxEndpoint         string
	sandboxToken            string
	dependencyInstaller     *dependency.Client
	projectAccess           *projectaccess.Client
	environmentTemplate     string
	environmentDependencies []string
	httpClient              *http.Client
	lookupEnv               func(string) (string, bool)
}

// RuntimeEnvironment describes only model-safe facts about the execution
// environment. Secrets, host paths, and installer credentials never cross
// this boundary.
func (r *Resolver) RuntimeEnvironment() (string, []string) {
	return r.environmentTemplate, append([]string(nil), r.environmentDependencies...)
}

// ResolveStaticMemory snapshots the six-layer file-backed instruction
// surface. The snapshot is resolved once per Run so a mid-run file mutation
// cannot silently change the model-visible prefix.
func (r *Resolver) ResolveStaticMemory(ctx context.Context, run agent.Run) ([]agent.StaticMemoryDocument, error) {
	documents, err := resolveStaticMemoryFiles(r.workspaceRoot, r.lookupEnv)
	if err != nil {
		return nil, err
	}
	if syncErr := r.store.SyncStaticMemorySources(ctx, run, documents); syncErr != nil {
		return nil, syncErr
	}
	return documents, nil
}

// NewResolver validates and creates a production execution resolver.
func NewResolver(store *postgres.RunStore, config Config) (*Resolver, error) {
	if store == nil {
		return nil, errors.New("execution resource store is required")
	}
	if len(config.ModelServices) == 0 {
		return nil, errors.New("at least one model service is required")
	}
	for name, service := range config.ModelServices {
		if strings.TrimSpace(name) == "" || strings.TrimSpace(service.Endpoint) == "" {
			return nil, errors.New("model service name and endpoint are required")
		}
		if service.ContextWindowTokens < 0 {
			return nil, fmt.Errorf("model service %q context_window_tokens must not be negative", name)
		}
		switch serviceDiscoveryStrategy(service) {
		case "models", "chat_probe":
		default:
			return nil, fmt.Errorf("model service %q has unsupported discovery_strategy %q", name, service.DiscoveryStrategy)
		}
	}
	lookup := config.LookupEnv
	if lookup == nil {
		lookup = os.LookupEnv
	}
	workspaceMode := strings.ToLower(strings.TrimSpace(config.WorkspaceMode))
	if workspaceMode == "" && strings.TrimSpace(config.WorkspaceRoot) != "" {
		workspaceMode = "local"
	}
	var dependencyInstaller *dependency.Client
	if strings.TrimSpace(config.DependencyInstallerEndpoint) != "" {
		var err error
		dependencyInstaller, err = dependency.NewClient(dependency.Config{Endpoint: config.DependencyInstallerEndpoint, Token: config.DependencyInstallerToken, Sources: config.DependencySources, HTTPClient: config.HTTPClient})
		if err != nil {
			return nil, fmt.Errorf("configure dependency installer: %w", err)
		}
	}
	var projectAccess *projectaccess.Client
	if strings.TrimSpace(config.ProjectAccessEndpoint) != "" {
		var err error
		projectAccess, err = projectaccess.NewClient(projectaccess.Config{Endpoint: config.ProjectAccessEndpoint, Token: config.ProjectAccessToken, HTTPClient: config.HTTPClient})
		if err != nil {
			return nil, fmt.Errorf("configure project access broker: %w", err)
		}
	}
	return &Resolver{
		store: store, modelServices: config.ModelServices,
		toolAllowedHosts:        append([]string(nil), config.ToolAllowedHosts...),
		workspaceRoot:           strings.TrimSpace(config.WorkspaceRoot),
		workspaceAllowWrite:     config.WorkspaceAllowWrite,
		workspaceMode:           workspaceMode,
		sandboxEndpoint:         strings.TrimSpace(config.SandboxEndpoint),
		sandboxToken:            strings.TrimSpace(config.SandboxToken),
		dependencyInstaller:     dependencyInstaller,
		projectAccess:           projectAccess,
		environmentTemplate:     strings.TrimSpace(config.EnvironmentTemplate),
		environmentDependencies: append([]string(nil), config.EnvironmentDependencies...),
		httpClient:              config.HTTPClient, lookupEnv: lookup,
	}, nil
}

// ParseModelServices decodes AGENT_MODEL_SERVICES JSON.
func ParseModelServices(raw string) (map[string]ModelService, error) {
	var services map[string]ModelService
	if err := json.Unmarshal([]byte(raw), &services); err != nil {
		return nil, fmt.Errorf("decode AGENT_MODEL_SERVICES: %w", err)
	}
	if len(services) == 0 {
		return nil, errors.New("AGENT_MODEL_SERVICES must contain at least one service")
	}
	return services, nil
}

// ResolvePrompt loads one exact tenant-owned prompt revision.
func (r *Resolver) ResolvePrompt(ctx context.Context, tenantID string, ref agent.VersionRef) (string, error) {
	version, err := r.store.GetPromptVersion(ctx, tenantID, ref.ID)
	if err != nil {
		return "", err
	}
	if strconv.Itoa(version.Version) != ref.Version {
		return "", fmt.Errorf("prompt revision changed: expected %s, got %d", ref.Version, version.Version)
	}
	return version.Content, nil
}

// RecallMemories resolves only scope-visible, non-expired memories for one Run.
func (r *Resolver) RecallMemories(ctx context.Context, run agent.Run, policy agent.MemoryPolicy, query string) ([]agent.Memory, error) {
	return r.store.RecallMemories(ctx, run, policy, query)
}

// RecallMemoryManifest resolves ranked memory candidates with bounded excerpts.
// The runtime uses this optional capability to decide suppression before
// loading any full memory body.
func (r *Resolver) RecallMemoryManifest(ctx context.Context, run agent.Run, policy agent.MemoryPolicy, query string) ([]agent.MemoryManifestEntry, error) {
	return r.store.RecallMemoryManifest(ctx, run, policy, query)
}

// ListVisibleMemoryManifest enumerates the policy-visible catalog without a
// relevance query. It is the final retrieval fallback: the model router must
// still choose the relevant IDs, but a weak lexical/vector query must never
// prevent the router from seeing memory previews.
func (r *Resolver) ListVisibleMemoryManifest(ctx context.Context, run agent.Run, policy agent.MemoryPolicy) ([]agent.MemoryManifestEntry, error) {
	return r.store.ListVisibleMemoryManifest(ctx, run, policy)
}

// LoadMemoriesForRun loads only the manifest IDs selected for injection.
func (r *Resolver) LoadMemoriesForRun(ctx context.Context, run agent.Run, policy agent.MemoryPolicy, ids []string) ([]agent.Memory, error) {
	return r.store.LoadMemoriesForRun(ctx, run, policy, ids)
}

// RecordMemoryRetrieval persists the bounded retrieval-decision projection.
func (r *Resolver) RecordMemoryRetrieval(ctx context.Context, run agent.Run, audit agent.MemoryRetrievalAudit) error {
	return r.store.RecordMemoryRetrieval(ctx, run, audit)
}

// EnqueueMemoryWriteJob schedules turn-complete or collapse-barrier memory
// extraction with idempotent database uniqueness.
func (r *Resolver) EnqueueMemoryWriteJob(ctx context.Context, run agent.Run, turnID, trigger, inputHash string, sourceFrom, sourceTo int64) error {
	return r.store.EnqueueMemoryWriteJob(ctx, run, turnID, trigger, inputHash, sourceFrom, sourceTo)
}

// LastMemoryWriteSequence returns the highest durable event sequence already
// incorporated by a completed memory-write job for this run. Runtime uses it
// as a monotonic cursor so a resumed turn does not reprocess the same event
// range after a worker retry or process restart.
func (r *Resolver) LastMemoryWriteSequence(ctx context.Context, run agent.Run) (int64, error) {
	return r.store.LastMemoryWriteSequence(ctx, run)
}

// ListMemoryIDsWrittenSince returns memory revisions already persisted after
// an event cursor. Extractors use this bounded identity list to avoid proposing
// a duplicate for a fact that an explicit or earlier automatic write handled.
func (r *Resolver) ListMemoryIDsWrittenSince(ctx context.Context, run agent.Run, afterSequence int64) ([]string, error) {
	return r.store.ListMemoryIDsWrittenSince(ctx, run, afterSequence)
}

// ClaimMemoryWriteJob leases one asynchronous extraction request.
func (r *Resolver) ClaimMemoryWriteJob(ctx context.Context, leaseSeconds int64) (agent.MemoryWriteJob, bool, error) {
	return r.store.ClaimMemoryWriteJob(ctx, leaseSeconds)
}

func (r *Resolver) CompleteMemoryWriteJob(ctx context.Context, tenantID, jobID, leaseToken string, resultSummary json.RawMessage) error {
	return r.store.CompleteMemoryWriteJob(ctx, tenantID, jobID, leaseToken, resultSummary)
}

func (r *Resolver) FailMemoryWriteJob(ctx context.Context, tenantID, jobID, leaseToken, message string, retryAt *time.Time, resultSummary json.RawMessage) error {
	return r.store.FailMemoryWriteJob(ctx, tenantID, jobID, leaseToken, message, retryAt, resultSummary)
}

func (r *Resolver) GetTaskPlanForWorkflow(ctx context.Context, tenantID, workflowID string) (taskplan.Plan, error) {
	return r.store.GetTaskPlanForWorkflow(ctx, tenantID, workflowID)
}

// GetLatestTaskPlanForSession restores the durable plan when a continuation
// creates a new Run in the same Session.
func (r *Resolver) GetLatestTaskPlanForSession(ctx context.Context, tenantID, sessionID string) (taskplan.Plan, error) {
	return r.store.GetLatestTaskPlanForSession(ctx, tenantID, sessionID)
}

// ValidateTaskPlanEvidence rechecks every terminal criterion immediately
// before completion. Evidence that was valid when a step closed may become
// stale after a later workspace mutation, so transition-time checks alone are
// insufficient for a trustworthy completed Run.
func (r *Resolver) ValidateTaskPlanEvidence(ctx context.Context, tenantID, runID string, plan taskplan.Plan) error {
	for _, step := range plan.Steps {
		for _, criterion := range step.AcceptanceCriteria {
			// Advisory observations inform the user but never reopen already
			// completed work. Required/release gates retain strict freshness.
			if criterion.Status != taskplan.CriterionPassed || !criterion.BlocksCompletion() {
				continue
			}
			if err := r.store.ValidateAcceptanceCriterionEvidence(ctx, tenantID, runID, criterion); err != nil {
				return &taskplan.EvidenceValidationError{
					StepID: step.ID, CriterionID: criterion.ID, Description: criterion.Description,
					Verification: criterion.Verification, Cause: err,
				}
			}
		}
	}
	return nil
}

// ResolveModel discovers (for auto policy), freezes, and constructs the exact
// model provider for one Run. Arbitrary URLs in AgentSpec are never accepted.
func (r *Resolver) ResolveModel(ctx context.Context, run agent.Run, binding agent.ModelBinding) (model.Provider, agent.ModelResolution, error) {
	switch strings.ToLower(strings.TrimSpace(binding.Provider)) {
	case "openai", "openai-compatible", "vllm", "aibrix":
	default:
		return nil, agent.ModelResolution{}, fmt.Errorf("model provider %q is not supported", binding.Provider)
	}
	resolution := agent.ModelResolution{}
	if run.ModelResolution != nil {
		resolution = *run.ModelResolution
	} else {
		var err error
		if binding.EffectiveSelectionPolicy() == agent.ModelSelectionAuto {
			resolution, err = r.discoverModel(ctx, binding)
		} else {
			resolution, err = r.resolvePinnedModel(binding)
			if err == nil {
				resolution, err = r.hydratePinnedModelMetadata(ctx, binding, resolution)
			}
		}
		if err != nil {
			return nil, agent.ModelResolution{}, err
		}
		lease, err := leaseFromRun(run)
		if err != nil {
			return nil, agent.ModelResolution{}, err
		}
		resolution, err = r.store.FreezeModelResolution(ctx, lease, resolution)
		if err != nil {
			return nil, agent.ModelResolution{}, err
		}
	}
	provider, err := r.providerForResolution(binding, resolution)
	if err != nil {
		return nil, agent.ModelResolution{}, err
	}
	return provider, resolution, nil
}

func (r *Resolver) providerForResolution(binding agent.ModelBinding, resolution agent.ModelResolution) (model.Provider, error) {
	service, exists := r.modelServices[resolution.ServiceRef]
	if !exists {
		return nil, fmt.Errorf("frozen model service_ref %q is no longer configured", resolution.ServiceRef)
	}
	if hashModelService(resolution.ServiceRef, service) != resolution.ServiceConfigHash {
		return nil, fmt.Errorf("frozen model service %q configuration changed; refusing endpoint drift", resolution.ServiceRef)
	}
	if !serviceSupports(service, binding.Capability) {
		return nil, fmt.Errorf("model service %q does not declare capability %q", resolution.ServiceRef, binding.Capability)
	}
	apiKey := ""
	if service.APIKeyEnvironment != "" {
		value, exists := r.lookupEnv(service.APIKeyEnvironment)
		if !exists || strings.TrimSpace(value) == "" {
			return nil, fmt.Errorf("model credential environment %q is unavailable", service.APIKeyEnvironment)
		}
		apiKey = value
	}
	return openai.New(openai.Config{
		Endpoint: service.Endpoint, Model: resolution.ModelID, APIKey: apiKey, HTTPClient: r.httpClient,
		// Qwen3.5 defaults to thinking mode and otherwise places its reasoning in
		// content. Keep the Agent's user-facing answer concise while preserving
		// full request/response observability at the provider boundary.
		DisableThinking: strings.Contains(strings.ToLower(resolution.ModelID), "qwen"),
	})
}

func (r *Resolver) resolvePinnedModel(binding agent.ModelBinding) (agent.ModelResolution, error) {
	service, exists := r.modelServices[binding.ServiceRef]
	if !exists {
		return agent.ModelResolution{}, fmt.Errorf("model service_ref %q is not configured", binding.ServiceRef)
	}
	if !serviceSupports(service, binding.Capability) {
		return agent.ModelResolution{}, fmt.Errorf("model service %q does not declare capability %q", binding.ServiceRef, binding.Capability)
	}
	modelID := strings.TrimSpace(binding.ModelID)
	if modelID == "" {
		return agent.ModelResolution{}, errors.New("pinned model_id is required")
	}
	return agent.ModelResolution{
		SelectionPolicy: agent.ModelSelectionPinned, Provider: binding.Provider,
		ServiceRef: binding.ServiceRef, ModelID: modelID, ModelVersion: binding.ModelVersion,
		ArtifactDigest: binding.ArtifactDigest, ContextWindowTokens: service.ContextWindowTokens,
		ServiceConfigHash: hashModelService(binding.ServiceRef, service),
		DiscoveredAt:      time.Now().UTC(),
	}, nil
}

type discoveredModel struct {
	ID           string          `json:"id"`
	Version      string          `json:"version"`
	Revision     string          `json:"revision"`
	Digest       string          `json:"digest"`
	SHA256       string          `json:"sha256"`
	MaxModelLen  int             `json:"max_model_len"`
	Capabilities json.RawMessage `json:"capabilities"`
}

func (r *Resolver) hydratePinnedModelMetadata(ctx context.Context, binding agent.ModelBinding, resolution agent.ModelResolution) (agent.ModelResolution, error) {
	service := r.modelServices[resolution.ServiceRef]
	if serviceDiscoveryStrategy(service) == "models" {
		models, err := r.listModels(ctx, service)
		if err != nil {
			if resolution.ContextWindowTokens <= 0 {
				return agent.ModelResolution{}, fmt.Errorf("discover pinned model metadata: %w", err)
			}
			return resolution, nil
		}
		var selected discoveredModel
		var found bool
		for _, candidate := range models {
			if candidate.ID == resolution.ModelID {
				selected, found = candidate, true
				break
			}
		}
		if !found {
			return agent.ModelResolution{}, fmt.Errorf("pinned model %q is not reported by service %q", resolution.ModelID, resolution.ServiceRef)
		}
		resolution.ContextWindowTokens = discoveredContextWindow(selected, service)
		if resolution.ModelVersion == "" {
			resolution.ModelVersion = firstNonEmpty(selected.Version, selected.Revision)
		}
		if resolution.ArtifactDigest == "" {
			resolution.ArtifactDigest = firstNonEmpty(selected.Digest, selected.SHA256)
		}
	}
	if resolution.ContextWindowTokens <= 0 {
		return agent.ModelResolution{}, fmt.Errorf("model service %q did not report max_model_len and has no context_window_tokens fallback", resolution.ServiceRef)
	}
	return resolution, nil
}

func (r *Resolver) discoverModel(ctx context.Context, binding agent.ModelBinding) (agent.ModelResolution, error) {
	services := uniqueStrings(append([]string{binding.ServiceRef}, binding.ServiceCandidates...))
	candidates := uniqueStrings(append([]string{binding.ModelID}, binding.ModelCandidates...))
	var discoveryErrors []error
	for _, serviceRef := range services {
		service, exists := r.modelServices[serviceRef]
		if !exists {
			discoveryErrors = append(discoveryErrors, fmt.Errorf("service %q is not configured", serviceRef))
			continue
		}
		if !serviceSupports(service, binding.Capability) {
			discoveryErrors = append(discoveryErrors, fmt.Errorf("service %q lacks capability %q", serviceRef, binding.Capability))
			continue
		}
		var selected discoveredModel
		var ok bool
		var err error
		if serviceDiscoveryStrategy(service) == "chat_probe" {
			selected, err = r.probeCandidateModels(ctx, service, candidates)
			ok = err == nil
		} else {
			var models []discoveredModel
			models, err = r.listModels(ctx, service)
			if err == nil {
				selected, ok = selectDiscoveredModel(models, candidates, binding.Capability)
				if ok && serviceSupports(service, "tool_calling") {
					_, err = r.probeCandidateModels(ctx, service, []string{selected.ID})
					ok = err == nil
				}
			}
		}
		if err != nil {
			discoveryErrors = append(discoveryErrors, fmt.Errorf("probe service %q: %w", serviceRef, err))
			continue
		}
		if !ok {
			discoveryErrors = append(discoveryErrors, fmt.Errorf("service %q has no eligible candidate", serviceRef))
			continue
		}
		version := strings.TrimSpace(selected.Version)
		if version == "" {
			version = strings.TrimSpace(selected.Revision)
		}
		digest := strings.TrimSpace(selected.Digest)
		if digest == "" {
			digest = strings.TrimSpace(selected.SHA256)
		}
		contextWindow := discoveredContextWindow(selected, service)
		if contextWindow <= 0 {
			discoveryErrors = append(discoveryErrors, fmt.Errorf("service %q model %q did not report max_model_len and has no context_window_tokens fallback", serviceRef, selected.ID))
			continue
		}
		return agent.ModelResolution{
			SelectionPolicy: agent.ModelSelectionAuto, Provider: binding.Provider,
			ServiceRef: serviceRef, ModelID: selected.ID, ModelVersion: version,
			ArtifactDigest: digest, ContextWindowTokens: contextWindow,
			ServiceConfigHash: hashModelService(serviceRef, service),
			DiscoveredAt:      time.Now().UTC(),
		}, nil
	}
	return agent.ModelResolution{}, fmt.Errorf("automatic model discovery failed: %w", errors.Join(discoveryErrors...))
}

func discoveredContextWindow(discovered discoveredModel, service ModelService) int {
	if discovered.MaxModelLen > 0 {
		return discovered.MaxModelLen
	}
	return service.ContextWindowTokens
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			return value
		}
	}
	return ""
}

func (r *Resolver) probeCandidateModels(ctx context.Context, service ModelService, candidates []string) (discoveredModel, error) {
	if len(candidates) == 0 {
		return discoveredModel{}, errors.New("chat_probe discovery requires explicit model_candidates")
	}
	probeCtx, cancel := context.WithTimeout(ctx, 8*time.Second)
	defer cancel()
	var probeErrors []error
	for _, candidate := range candidates {
		apiKey := ""
		if service.APIKeyEnvironment != "" {
			value, exists := r.lookupEnv(service.APIKeyEnvironment)
			if !exists || strings.TrimSpace(value) == "" {
				return discoveredModel{}, fmt.Errorf("model credential environment %q is unavailable", service.APIKeyEnvironment)
			}
			apiKey = value
		}
		provider, err := openai.New(openai.Config{
			Endpoint: service.Endpoint, Model: candidate, APIKey: apiKey,
			HTTPClient: r.httpClient, DisableThinking: true,
		})
		var response model.Response
		expectsStructuredTools := serviceSupports(service, "tool_calling")
		if err == nil {
			request := model.Request{Messages: []model.Message{{Role: model.RoleUser, Content: "Reply with pong."}}, MaxTokens: 8, Temperature: 0}
			if expectsStructuredTools {
				request.Messages[0].Content = "Call the health_probe tool with an empty object."
				request.Tools = []model.ToolSchema{{Name: "health_probe", Description: "Model tool-calling readiness probe", Parameters: json.RawMessage(`{"type":"object","additionalProperties":false}`)}}
			}
			response, err = provider.Complete(probeCtx, request)
		}
		if err == nil && expectsStructuredTools {
			if len(response.Message.ToolCalls) != 1 || response.Message.ToolCalls[0].Name != "health_probe" {
				err = fmt.Errorf("structured tool-calling probe returned no health_probe tool_calls")
			}
		}
		if err == nil && strings.TrimSpace(response.ModelID) != "" && response.ModelID != candidate {
			err = fmt.Errorf("endpoint returned model %q for candidate %q", response.ModelID, candidate)
		}
		if err == nil {
			return discoveredModel{ID: candidate}, nil
		}
		probeErrors = append(probeErrors, fmt.Errorf("model %q: %w", candidate, err))
		if probeCtx.Err() != nil {
			break
		}
	}
	return discoveredModel{}, fmt.Errorf("no candidate accepted a chat probe: %w", errors.Join(probeErrors...))
}

func serviceDiscoveryStrategy(service ModelService) string {
	strategy := strings.ToLower(strings.TrimSpace(service.DiscoveryStrategy))
	if strategy == "" {
		return "models"
	}
	return strategy
}

func (r *Resolver) listModels(ctx context.Context, service ModelService) ([]discoveredModel, error) {
	endpoint, err := modelsURL(service.Endpoint)
	if err != nil {
		return nil, err
	}
	probeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	request, err := http.NewRequestWithContext(probeCtx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("create models request: %w", err)
	}
	if service.APIKeyEnvironment != "" {
		apiKey, exists := r.lookupEnv(service.APIKeyEnvironment)
		if !exists || strings.TrimSpace(apiKey) == "" {
			return nil, fmt.Errorf("model credential environment %q is unavailable", service.APIKeyEnvironment)
		}
		request.Header.Set("Authorization", "Bearer "+apiKey)
	}
	client := r.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, fmt.Errorf("models endpoint returned HTTP %d", response.StatusCode)
	}
	var payload struct {
		Data []discoveredModel `json:"data"`
	}
	decoder := json.NewDecoder(io.LimitReader(response.Body, 2<<20))
	if err := decoder.Decode(&payload); err != nil {
		return nil, fmt.Errorf("decode models response: %w", err)
	}
	if len(payload.Data) == 0 {
		return nil, errors.New("models endpoint returned no models")
	}
	return payload.Data, nil
}

func modelsURL(endpoint string) (string, error) {
	parsed, err := url.Parse(strings.TrimRight(strings.TrimSpace(endpoint), "/"))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return "", errors.New("model endpoint must be an absolute URL")
	}
	path := strings.TrimSuffix(parsed.Path, "/chat/completions")
	if !strings.HasSuffix(path, "/v1") {
		path = strings.TrimRight(path, "/") + "/v1"
	}
	parsed.Path = path + "/models"
	return parsed.String(), nil
}

func selectDiscoveredModel(models []discoveredModel, candidates []string, capability string) (discoveredModel, bool) {
	eligible := make(map[string]discoveredModel)
	for _, current := range models {
		if strings.TrimSpace(current.ID) != "" && modelSupports(current, capability) {
			eligible[current.ID] = current
		}
	}
	for _, candidate := range candidates {
		if selected, ok := eligible[candidate]; ok {
			return selected, true
		}
	}
	if len(candidates) != 0 {
		return discoveredModel{}, false
	}
	ids := make([]string, 0, len(eligible))
	for id := range eligible {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	if len(ids) == 0 {
		return discoveredModel{}, false
	}
	return eligible[ids[0]], true
}

func modelSupports(discovered discoveredModel, capability string) bool {
	required := strings.ToLower(strings.TrimSpace(capability))
	if required == "" || required == "chat" || len(discovered.Capabilities) == 0 || string(discovered.Capabilities) == "null" {
		return true
	}
	var list []string
	if json.Unmarshal(discovered.Capabilities, &list) == nil {
		return containsFold(list, required)
	}
	var flags map[string]bool
	return json.Unmarshal(discovered.Capabilities, &flags) == nil && flags[required]
}

func serviceSupports(service ModelService, capability string) bool {
	required := strings.ToLower(strings.TrimSpace(capability))
	return required == "" || required == "chat" || containsFold(service.Capabilities, required)
}

func containsFold(values []string, expected string) bool {
	for _, value := range values {
		if strings.EqualFold(strings.TrimSpace(value), expected) {
			return true
		}
	}
	return false
}

func uniqueStrings(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	return result
}

func hashModelService(name string, service ModelService) string {
	payload, _ := json.Marshal(struct {
		Name, Endpoint, Model, DiscoveryStrategy, APIKeyEnvironment string
		ContextWindowTokens                                         int
		Capabilities                                                []string
	}{name, strings.TrimRight(service.Endpoint, "/"), service.Model, serviceDiscoveryStrategy(service), service.APIKeyEnvironment, service.ContextWindowTokens, service.Capabilities})
	digest := sha256.Sum256(payload)
	return fmt.Sprintf("sha256:%x", digest[:])
}

// ResolveTools constructs a Run-local registry from exact ToolVersion refs.
func (r *Resolver) ResolveTools(ctx context.Context, run agent.Run, ref agent.VersionRef) (tool.Executor, error) {
	return r.ResolveToolsWithExtra(ctx, run, ref, nil)
}

// ResolveToolsWithExtra merges the Agent ToolSet with Skill-required exact
// ToolVersions. Duplicate version IDs are removed before registration.
func (r *Resolver) ResolveToolsWithExtra(ctx context.Context, run agent.Run, ref agent.VersionRef, extra []agent.VersionRef) (tool.Executor, error) {
	lease, err := leaseFromRun(run)
	if err != nil {
		return nil, err
	}
	setVersion, err := r.store.GetToolSetVersion(ctx, run.TenantID, ref.ID)
	if err != nil {
		return nil, err
	}
	if strconv.Itoa(setVersion.Version) != ref.Version {
		return nil, fmt.Errorf("tool set revision changed: expected %s, got %d", ref.Version, setVersion.Version)
	}
	var set resource.ToolSetSpec
	if err := json.Unmarshal(setVersion.Spec, &set); err != nil {
		return nil, fmt.Errorf("decode tool set: %w", err)
	}
	if err := set.Validate(); err != nil {
		return nil, err
	}
	registry := tool.NewRegistry()
	refs := append([]resource.VersionRef(nil), set.Tools...)
	seen := make(map[string]struct{}, len(refs)+len(extra))
	for _, current := range refs {
		seen[current.ID] = struct{}{}
	}
	for _, current := range extra {
		if _, ok := seen[current.ID]; ok {
			continue
		}
		refs = append(refs, resource.VersionRef{ID: current.ID, Version: current.Version})
		seen[current.ID] = struct{}{}
	}
	for _, toolRef := range refs {
		version, err := r.store.GetToolVersion(ctx, run.TenantID, toolRef.ID)
		if err != nil {
			return nil, err
		}
		if strconv.Itoa(version.Version) != toolRef.Version {
			return nil, fmt.Errorf("tool revision changed: expected %s, got %d", toolRef.Version, version.Version)
		}
		var spec resource.ToolSpec
		if err := json.Unmarshal(version.Spec, &spec); err != nil {
			return nil, fmt.Errorf("decode tool version %s: %w", version.ID, err)
		}
		var handler tool.Handler
		var previewHandler tool.Handler
		switch spec.ProviderType {
		case "http":
			handler, err = httptool.NewHandler(spec, httptool.Config{
				AllowedHosts: r.toolAllowedHosts, HTTPClient: r.httpClient, LookupEnv: r.lookupEnv,
			})
		case "workspace":
			if r.workspaceMode == "local" {
				var runRoot string
				workspaceID := run.WorkflowID
				if strings.TrimSpace(workspaceID) == "" {
					workspaceID = run.ID
				}
				runRoot, err = workspace.EnsureRunRoot(r.workspaceRoot, workspaceID)
				if err == nil {
					handler, err = workspace.NewHandler(spec, workspace.Config{Root: runRoot, AllowWrite: r.workspaceAllowWrite})
				}
				if err == nil && (spec.Workspace.Operation == "write_file" || spec.Workspace.Operation == "append_file" || spec.Workspace.Operation == "edit_file" || spec.Workspace.Operation == "promote_file") {
					previewHandler, err = workspace.NewPreviewHandler(spec, workspace.Config{Root: runRoot, AllowWrite: r.workspaceAllowWrite})
				}
			} else if r.workspaceMode == "sandbox" {
				handler, err = sandbox.NewHandler(spec, sandbox.Config{Endpoint: r.sandboxEndpoint, Token: r.sandboxToken, HTTPClient: r.httpClient})
				if err == nil && (spec.Workspace.Operation == "write_file" || spec.Workspace.Operation == "append_file" || spec.Workspace.Operation == "edit_file" || spec.Workspace.Operation == "promote_file") {
					previewHandler, err = sandbox.NewPreviewHandler(spec, sandbox.Config{Endpoint: r.sandboxEndpoint, Token: r.sandboxToken, HTTPClient: r.httpClient})
				}
			} else {
				err = fmt.Errorf("unsupported workspace mode %q", r.workspaceMode)
			}
		case "mcp":
			if spec.MCP == nil {
				err = errors.New("MCP binding is required")
				break
			}
			serverSpec, snapshot, bindingErr := r.store.GetMCPBinding(ctx, run.TenantID, spec.MCP.ServerVersionID, spec.MCP.ToolName, spec.MCP.SchemaHash)
			if bindingErr != nil {
				err = bindingErr
				break
			}
			if string(snapshot.InputSchema) != string(spec.Definition.InputSchema) {
				err = errors.New("MCP ToolVersion input schema does not match pinned discovery snapshot")
				break
			}
			client, clientErr := mcp.NewClient(serverSpec, r.toolAllowedHosts, r.lookupEnv, r.httpClient)
			if clientErr != nil {
				err = clientErr
				break
			}
			handler = func(callCtx context.Context, call tool.Call) (tool.Result, error) {
				content, isError, callErr := client.CallTool(callCtx, spec.MCP.ToolName, call.Arguments)
				if callErr != nil {
					return tool.Result{}, callErr
				}
				return tool.Result{Content: content, IsError: isError, Meta: map[string]string{"provider": "mcp", "server_version_id": spec.MCP.ServerVersionID, "schema_hash": spec.MCP.SchemaHash}}, nil
			}
		default:
			err = fmt.Errorf("unsupported provider %q", spec.ProviderType)
		}
		if err != nil {
			return nil, fmt.Errorf("resolve tool %s: %w", version.ID, err)
		}
		if spec.ProviderType == "workspace" && spec.Workspace != nil && spec.Workspace.Operation == "run_command" {
			spec.Definition = exposeRestrictedInlineProbe(spec.Definition)
		}
		idempotentHandler := func(callCtx context.Context, call tool.Call) (tool.Result, error) {
			if spec.ProviderType == "workspace" && spec.Workspace != nil && spec.Workspace.Operation == "run_command" {
				if validationErr := workspace.ValidateCommandArguments(call.Arguments); validationErr != nil {
					return tool.Result{}, tool.NewContractError("SANDBOX_COMMAND_REJECTED", call.Name, "/args", "an allowed Python command profile", "rejected", validationErr.Error(), false)
				}
			}
			requiresApproval := runRequiresApproval(run, spec.Definition.Risk)
			if requiresApproval && r.workspaceMode == "sandbox" && spec.ProviderType == "workspace" && spec.Workspace != nil && spec.Workspace.Operation == "run_command" && runAutoApprovesSandboxCommand(run) {
				requiresApproval = false
			}
			if requiresApproval {
				var preview *tool.Artifact
				if previewHandler != nil {
					previewResult, previewErr := previewHandler(callCtx, call)
					if previewErr != nil {
						return tool.Result{}, fmt.Errorf("preview workspace change: %w", previewErr)
					}
					if len(previewResult.Artifacts) != 1 {
						return tool.Result{}, errors.New("workspace approval preview must return exactly one Diff Artifact")
					}
					preview = &previewResult.Artifacts[0]
				}
				expires := approvalExpiry(run)
				if approvalErr := r.store.RequireToolApproval(callCtx, lease, run.TenantID, version.ID, spec.Definition.Risk, call, expires, preview); approvalErr != nil {
					return tool.Result{}, approvalErr
				}
			}
			return r.store.ExecuteToolIdempotent(
				callCtx, lease, run.TenantID, version.ID, spec.ProviderType, call, handler,
			)
		}
		if err := registry.Register(spec.Definition, idempotentHandler); err != nil {
			return nil, err
		}
	}
	if r.dependencyInstaller != nil {
		definition := dependency.Definition(r.dependencyInstaller.SourceNames())
		if r.environmentTemplate != "" {
			definition.Description += " The active prebuilt environment template is " + r.environmentTemplate + "; use this tool only when the required package is not already available there."
			if len(r.environmentDependencies) != 0 {
				definition.Description += " Prebuilt dependencies: " + strings.Join(r.environmentDependencies, ", ") + "."
			}
		}
		handler := func(callCtx context.Context, call tool.Call) (tool.Result, error) {
			var request dependency.Request
			if err := json.Unmarshal(call.Arguments, &request); err != nil {
				return tool.Result{}, fmt.Errorf("decode dependency request: %w", err)
			}
			request.RunID = run.ID
			installed, installErr := r.dependencyInstaller.Install(callCtx, request)
			if err := r.store.RecordDependencyInstall(context.WithoutCancel(callCtx), run.TenantID, run.ID, call.ID, r.environmentTemplate, request, installed, installErr); err != nil {
				if installErr != nil {
					return tool.Result{}, errors.Join(installErr, err)
				}
				return tool.Result{}, err
			}
			if installErr != nil {
				return tool.Result{}, installErr
			}
			content, err := json.Marshal(installed)
			if err != nil {
				return tool.Result{}, err
			}
			return tool.Result{Content: content, Meta: map[string]string{"provider": "dependency_installer", "scope": "run", "environment_template": r.environmentTemplate}}, nil
		}
		wrapped := func(callCtx context.Context, call tool.Call) (tool.Result, error) {
			var request dependency.Request
			if err := json.Unmarshal(call.Arguments, &request); err != nil {
				return tool.Result{}, err
			}
			if err := request.NormalizeAndValidate(); err != nil {
				return tool.Result{}, err
			}
			// Dependency installation is never covered by the generic sandbox
			// auto-approval toggle. Every exact request needs one human decision.
			if err := r.store.RequireToolApproval(callCtx, lease, run.TenantID, "", tool.RiskHigh, call, approvalExpiry(run), nil); err != nil {
				return tool.Result{}, err
			}
			return r.store.ExecuteToolIdempotent(callCtx, lease, run.TenantID, "platform:install_dependency:v1", "dependency_installer", call, handler)
		}
		if err := registry.Register(definition, wrapped); err != nil {
			return nil, err
		}
	}
	if r.projectAccess != nil {
		definition := tool.Definition{
			Name: "read_project_file", Version: "1",
			Description: "Request human-approved read access to one text file in the operator-bound project. Use a relative project path and explain why it is needed. This never grants directory access or shell access. After approval the exact full file is stored as an immutable input snapshot, while only the requested line range is returned here. Use read_file for files already inside this Run workspace.",
			Risk:        tool.RiskRead, ExecutionMode: tool.ExecutionSerial,
			InputSchema: json.RawMessage(`{"type":"object","required":["path","reason","line_count"],"properties":{"path":{"type":"string","minLength":1,"maxLength":1024},"reason":{"type":"string","minLength":1,"maxLength":1000},"start_line":{"type":"integer","minimum":1},"line_count":{"type":"integer","minimum":1,"maximum":200}},"additionalProperties":false}`),
		}
		handler := func(callCtx context.Context, call tool.Call) (tool.Result, error) {
			var request projectaccess.ReadRequest
			if err := json.Unmarshal(call.Arguments, &request); err != nil {
				return tool.Result{}, fmt.Errorf("decode project read request: %w", err)
			}
			request.RunID = run.ID
			if err := request.ValidateModelInput(); err != nil {
				return tool.Result{}, err
			}
			return r.projectAccess.Read(callCtx, request)
		}
		wrapped := func(callCtx context.Context, call tool.Call) (tool.Result, error) {
			var request projectaccess.ReadRequest
			if err := json.Unmarshal(call.Arguments, &request); err != nil {
				return tool.Result{}, err
			}
			if err := request.ValidateModelInput(); err != nil {
				return tool.Result{}, err
			}
			// Project reads are never covered by generic READ auto-approval:
			// every exact path/range/reason is a separately durable decision.
			if err := r.store.RequireToolApproval(callCtx, lease, run.TenantID, "", tool.RiskRead, call, approvalExpiry(run), nil); err != nil {
				return tool.Result{}, err
			}
			return r.store.ExecuteToolIdempotent(callCtx, lease, run.TenantID, "platform:read_project_file:v1", "project_access", call, handler)
		}
		if err := registry.Register(definition, wrapped); err != nil {
			return nil, err
		}
	}
	var binding struct {
		Spec agent.Spec `json:"spec"`
	}
	if err := json.Unmarshal(run.BindingSnapshot, &binding); err != nil {
		return nil, fmt.Errorf("decode collaboration policy: %w", err)
	}
	if err := registerMemoryTools(ctx, r, run, binding.Spec, registry); err != nil {
		return nil, fmt.Errorf("register memory tools: %w", err)
	}
	actionToolNames := make([]string, 0)
	for _, definition := range registry.Definitions() {
		actionToolNames = append(actionToolNames, definition.Name)
	}
	sort.Strings(actionToolNames)
	planDescription := "Use first for a substantive task that will edit files, run commands, or require verifiable multi-step work. Create the Workflow-owned durable execution plan with one outcome-oriented step per meaningful phase and observable acceptance criteria. For an existing Plan, do not rebuild it for ordinary progress: use update_plan_step or revise_verification. Full mutation requires base_revision and change_mode=extend|replan; replan also requires replan_reason and must preserve unfinished nodes or explicitly retire each removed node with a reason. Do not put platform fields, receipt IDs, exit_code, working_directory, or UI state into this request. Available action tools: " + strings.Join(actionToolNames, ", ")
	planDefinition := tool.Definition{Name: "update_plan", Version: "12", Description: planDescription, Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial, InputSchema: json.RawMessage(`{"type":"object","required":["goal","steps"],"properties":{"goal":{"type":"string","minLength":1},"explanation":{"type":"string"},"change_mode":{"enum":["create","extend","replan"]},"base_revision":{"type":"integer","minimum":1},"replan_reason":{"type":"string","minLength":1,"maxLength":2000},"retired_steps":{"type":"array","maxItems":8,"items":{"type":"object","required":["id","reason"],"properties":{"id":{"type":"string","minLength":1},"reason":{"type":"string","minLength":1,"maxLength":1000}},"additionalProperties":false}},"steps":{"type":"array","minItems":1,"maxItems":8,"items":{"type":"object","required":["id","description","status","acceptance_criteria"],"properties":{"id":{"type":"string","minLength":1},"description":{"type":"string","minLength":1},"status":{"enum":["pending","in_progress"]},"assignee":{"type":"string"},"agent_version_id":{"type":"string"},"depends_on":{"type":"array","items":{"type":"string"}},"tool_hints":{"type":"array","maxItems":12,"uniqueItems":true,"items":{"type":"string","minLength":1}},"result":{"type":"string"},"acceptance_criteria":{"type":"array","minItems":1,"maxItems":8,"items":{"type":"object","required":["id","description","status","verification"],"properties":{"id":{"type":"string","minLength":1},"description":{"type":"string","minLength":1},"status":{"const":"pending"},"verification":{"type":"object","required":["kind"],"properties":{"kind":{"type":"string","minLength":1},"target":{"type":"string"},"match":{"type":"string"},"tool":{"type":"string"},"arguments":{"type":"object"},"assertions":{"type":"array","minItems":1,"items":{"type":"object","required":["path","operator"],"properties":{"path":{"type":"string","minLength":1},"operator":{"enum":["equals","contains","nonempty"]},"value":{}},"additionalProperties":false}}},"additionalProperties":false}},"additionalProperties":false}}},"additionalProperties":false}}},"additionalProperties":false}`)}
	// update_plan is a complete graph revision, not an append-only checklist.
	// Keep the public contract aligned with taskplan.Update.Validate so one
	// model response cannot grow an unbounded active graph.
	planDefinition.InputSchema = json.RawMessage(strings.Replace(string(planDefinition.InputSchema), `"maxItems":64`, `"maxItems":8`, 1))
	planDefinition.Description += " For command_exit_zero/test_pass, verification.arguments is authoritative. Use JSON number 0, not string \"0\", for run_command exit_code assertions."
	if err := registry.Register(planDefinition, func(callCtx context.Context, call tool.Call) (tool.Result, error) {
		var update taskplan.Update
		if err := json.Unmarshal(call.Arguments, &update); err != nil {
			return tool.Result{}, err
		}
		plan, err := r.store.UpsertTaskPlan(callCtx, lease, run.TenantID, call, update)
		if err != nil {
			return tool.Result{}, err
		}
		content, _ := json.Marshal(plan)
		return tool.Result{Content: content, Meta: map[string]string{"provider": "platform", "resource": "task_plan"}}, nil
	}); err != nil {
		return nil, err
	}
	planStepDefinition := tool.Definition{Name: "update_plan_step", Version: "4", Description: "Use after executing or verifying work to update exactly one existing Todo; do not create a new Plan and do not copy tool call IDs or platform receipts. Keep status in_progress when declared tools or evidence are insufficient, and replace tool_hints with the smallest corrected set before another action call. Mark completed only when the Runtime can match successful Tool evidence for the step and its criteria; mark blocked or skipped only with a truthful result. Updating a step activates the next ready Todo.", Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial, InputSchema: json.RawMessage(`{"type":"object","required":["step_id","status"],"properties":{"step_id":{"type":"string","minLength":1},"status":{"enum":["in_progress","completed","blocked","skipped"]},"result":{"type":"string"},"tool_hints":{"type":"array","maxItems":12,"uniqueItems":true,"items":{"type":"string","minLength":1}},"acceptance_criteria":{"type":"array","maxItems":8,"items":{"type":"object","required":["id","status"],"properties":{"id":{"type":"string","minLength":1},"status":{"enum":["pending","passed","failed","skipped"]}},"additionalProperties":false}}},"additionalProperties":false}`)}
	if err := registry.Register(planStepDefinition, func(callCtx context.Context, call tool.Call) (tool.Result, error) {
		var mutation taskplan.StepUpdate
		if err := json.Unmarshal(call.Arguments, &mutation); err != nil {
			return tool.Result{}, err
		}
		current, err := r.store.GetTaskPlanForWorkflow(callCtx, run.TenantID, run.WorkflowID)
		if err != nil {
			return tool.Result{}, err
		}
		mutation, err = r.resolveStepUpdateEvidence(callCtx, run, current, mutation)
		if err != nil {
			return tool.Result{}, err
		}
		update, err := taskplan.ApplyStepUpdate(current, mutation)
		if err != nil {
			return tool.Result{}, err
		}
		plan, err := r.store.UpsertTaskPlan(callCtx, lease, run.TenantID, call, update)
		if err != nil {
			return tool.Result{}, err
		}
		content, _ := json.Marshal(plan)
		return tool.Result{Content: content, Meta: map[string]string{"provider": "platform", "resource": "task_plan_step"}}, nil
	}); err != nil {
		return nil, err
	}
	revisionDefinition := tool.Definition{Name: "revise_verification", Version: "2", Description: "Use when exactly one acceptance criterion has an invalid, unsupported, failed, stale, or pending verification. Repair that criterion only; do not rebuild the Plan or alter unrelated progress. Use action=replace with a corrected registered verification contract, and provide only the fields allowed by its Schema. Use action=skip_advisory only when the check is genuinely unnecessary or unsupported; required and release gates cannot be skipped. base_revision is optional for ordinary repair and required when Runtime applies a Reviewer recommendation. Never repeat an unchanged failing Tool call when this repair tool is available.", Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial, InputSchema: json.RawMessage(`{"type":"object","required":["step_id","criterion_id","action","reason"],"properties":{"step_id":{"type":"string","minLength":1},"criterion_id":{"type":"string","minLength":1},"action":{"enum":["replace","skip_advisory"]},"reason":{"type":"string","minLength":1,"maxLength":2000},"base_revision":{"type":"integer","minimum":1},"verification":{"type":"object","properties":{"kind":{"type":"string","minLength":1},"target":{"type":"string"},"match":{"type":"string"},"tool":{"type":"string"},"arguments":{"type":"object"},"assertions":{"type":"array","items":{"type":"object","required":["path","operator"],"properties":{"path":{"type":"string"},"operator":{"enum":["equals","contains","nonempty"]},"value":{}},"additionalProperties":false}}},"additionalProperties":false}},"additionalProperties":false}`)}
	if err := registry.Register(revisionDefinition, func(callCtx context.Context, call tool.Call) (tool.Result, error) {
		var revision taskplan.CriterionRevision
		if err := json.Unmarshal(call.Arguments, &revision); err != nil {
			return tool.Result{}, err
		}
		current, err := r.store.GetTaskPlanForWorkflow(callCtx, run.TenantID, run.WorkflowID)
		if err != nil {
			return tool.Result{}, err
		}
		update, err := taskplan.ApplyCriterionRevision(current, revision)
		if err != nil {
			return tool.Result{}, err
		}
		plan, err := r.store.UpsertTaskPlan(callCtx, lease, run.TenantID, call, update)
		if err != nil {
			return tool.Result{}, err
		}
		content, _ := json.Marshal(plan)
		return tool.Result{Content: content, Meta: map[string]string{"provider": "platform", "resource": "verification_revision"}}, nil
	}); err != nil {
		return nil, err
	}
	questionDefinition := tool.Definition{Name: "ask_user", Version: "1", Description: "Use only when a required fact, permission, or user decision cannot be discovered from the workspace, Plan, or available tools. Do not use it to avoid a recoverable tool error or to ask for information already present in context. Ask one focused question; when choices can be enumerated, include 2-4 concise options and explain the decision context. The Run pauses durably and resumes after the user answers.", Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial, InputSchema: json.RawMessage(`{"type":"object","required":["question"],"properties":{"question":{"type":"string","minLength":1},"options":{"type":"array","maxItems":12,"items":{"type":"string"}},"context":{"type":"string"}},"additionalProperties":false}`)}
	if err := registry.Register(questionDefinition, func(callCtx context.Context, call tool.Call) (tool.Result, error) {
		var request interaction.Request
		if err := json.Unmarshal(call.Arguments, &request); err != nil {
			return tool.Result{}, err
		}
		answer, err := r.store.RequireUserInput(callCtx, lease, run.TenantID, call, request)
		if err != nil {
			return tool.Result{}, err
		}
		content, _ := json.Marshal(map[string]string{"answer": answer})
		return tool.Result{Content: content, Meta: map[string]string{"provider": "platform", "resource": "user_input"}}, nil
	}); err != nil {
		return nil, err
	}
	if len(binding.Spec.Collaboration.AllowedTargets) > 0 {
		targets, targetErr := r.store.DescribeDelegationTargets(ctx, run.TenantID, binding.Spec.Collaboration.AllowedTargets)
		if targetErr != nil {
			return nil, fmt.Errorf("describe collaboration targets: %w", targetErr)
		}
		modeSet := map[string]struct{}{}
		targetLabels := make([]string, 0, len(targets))
		for _, target := range targets {
			for _, mode := range target.Modes {
				modeSet[mode] = struct{}{}
			}
			label := target.AgentName
			if label == "" {
				label = target.AgentKey
			}
			if label == "" {
				label = target.AgentVersionID
			}
			availability := "不可用"
			if target.Available {
				availability = "已发布"
			}
			targetLabels = append(targetLabels, fmt.Sprintf("%s (%s) · version=%s · modes=%s · %s", label, target.AgentKey, target.AgentVersionID, strings.Join(target.Modes, ","), availability))
		}
		modes := make([]string, 0, len(modeSet))
		for mode := range modeSet {
			modes = append(modes, mode)
		}
		sort.Strings(modes)
		if len(modes) == 0 {
			modes = []string{"sync", "async"}
		}
		delegateSchema := map[string]any{
			"type":     "object",
			"required": []string{"mode", "input"},
			"anyOf": []any{
				map[string]any{"required": []string{"target_agent_version_id"}},
				map[string]any{"required": []string{"target_agent"}},
			},
			"properties": map[string]any{
				"target_agent_version_id": map[string]any{"type": "string", "description": "Use the exact published AgentVersion UUID. A unique allowed Agent key or display name is also accepted and is normalized by the platform."},
				"target_agent":            map[string]any{"type": "string", "description": "Optional human-readable alias; prefer target_agent_version_id."},
				"mode":                    map[string]any{"type": "string", "enum": modes},
				"input":                   map[string]any{"type": "object"},
			},
			"additionalProperties": false,
			"x-allowed-targets":    targets,
		}
		encodedDelegateSchema, _ := json.Marshal(delegateSchema)
		description := "Delegate a bounded, independently verifiable subtask to one explicitly allowed published AgentVersion. Use sync when the result is required before the parent can continue; use async when the parent can proceed and wait for a later completion event. Provide a focused input object, not the whole conversation, and never invent a target outside the allowed list. The child Run has its own checkpoint, tools, budget, and audit trail. Allowed targets: " + strings.Join(targetLabels, "; ")
		definition := tool.Definition{Name: "delegate_agent", Version: "2", Description: description, Risk: tool.RiskInternal, ExecutionMode: tool.ExecutionSerial, InputSchema: encodedDelegateSchema}
		handler := func(callCtx context.Context, call tool.Call) (tool.Result, error) {
			if runRequiresApproval(run, definition.Risk) {
				if err := r.store.RequireToolApproval(callCtx, lease, run.TenantID, "", definition.Risk, call, approvalExpiry(run), nil); err != nil {
					return tool.Result{}, err
				}
			}
			var request delegation.Request
			if err := json.Unmarshal(call.Arguments, &request); err != nil {
				return tool.Result{}, err
			}
			outcome, err := r.store.ResolveDelegation(callCtx, lease, run, call, request, binding.Spec.Collaboration)
			if err != nil {
				return tool.Result{}, err
			}
			content, _ := json.Marshal(outcome)
			return tool.Result{Content: content, Meta: map[string]string{"provider": "internal_agent", "child_run_id": outcome.ChildRunID, "delegation_id": outcome.DelegationID}}, nil
		}
		if err := registry.Register(definition, handler); err != nil {
			return nil, err
		}
	}
	return registry, nil
}

func exposeRestrictedInlineProbe(definition tool.Definition) tool.Definition {
	definition.Description = "Execute bounded Python inside the isolated Run Sandbox. Workspace scripts are allowed and expected: use python3 <relative-script.py> [args] for substantive validation, or python3 -m py_compile|compileall|unittest ... for checks. A restricted one-line python3 -c probe is available only for short environment/import/value inspection; inline probes are limited to 512 characters and 5 seconds and deny file, network, subprocess, native-loading, and dynamic-execution capabilities. A non-zero exit_code from a workspace script means the script ran and the code failed, not that Sandbox policy rejected it; inspect stderr and edit the script."
	var schema map[string]any
	if json.Unmarshal(definition.InputSchema, &schema) != nil {
		return definition
	}
	properties, _ := schema["properties"].(map[string]any)
	args, _ := properties["args"].(map[string]any)
	if args != nil {
		args["description"] = "Use [\"-c\",\"<one-line probe>\"] only for bounded environment/import/value inspection; otherwise pass a relative script or an allowed -m module invocation."
		if encoded, err := json.Marshal(schema); err == nil {
			definition.InputSchema = encoded
		}
	}
	return definition
}

func (r *Resolver) resolveStepUpdateEvidence(ctx context.Context, run agent.Run, plan taskplan.Plan, mutation taskplan.StepUpdate) (taskplan.StepUpdate, error) {
	var current *taskplan.Step
	for index := range plan.Steps {
		if plan.Steps[index].ID == mutation.StepID {
			current = &plan.Steps[index]
			break
		}
	}
	if current == nil {
		return mutation, errors.New("plan step was not found")
	}
	criteria := make(map[string]taskplan.AcceptanceCriterion, len(current.AcceptanceCriteria))
	for _, criterion := range current.AcceptanceCriteria {
		criteria[criterion.ID] = criterion
	}
	requested := make(map[string]struct{}, len(mutation.AcceptanceCriteria))
	for _, changed := range mutation.AcceptanceCriteria {
		if _, exists := criteria[changed.ID]; !exists {
			return mutation, fmt.Errorf("acceptance criterion %q was not found", changed.ID)
		}
		requested[changed.ID] = struct{}{}
	}
	// A completed intent means all remaining criteria need proof. The model is
	// not required to echo a checklist that PostgreSQL already owns.
	if mutation.Status == taskplan.StatusCompleted {
		for _, criterion := range current.AcceptanceCriteria {
			if criterion.Status == taskplan.CriterionPassed || criterion.Status == taskplan.CriterionSkipped {
				continue
			}
			if _, exists := requested[criterion.ID]; !exists {
				requested[criterion.ID] = struct{}{}
				mutation.AcceptanceCriteria = append(mutation.AcceptanceCriteria, taskplan.AcceptanceCriterion{ID: criterion.ID, Status: taskplan.CriterionPassed})
			}
		}
	}
	for index := range mutation.AcceptanceCriteria {
		changed := mutation.AcceptanceCriteria[index]
		if changed.Status != taskplan.CriterionPassed {
			continue
		}
		base := criteria[changed.ID]
		if !base.BlocksCompletion() && (base.Status == taskplan.CriterionInvalid || base.Status == taskplan.CriterionUnsupported) {
			mutation.AcceptanceCriteria[index] = base
			continue
		}
		if base.Status == taskplan.CriterionPassed {
			mutation.AcceptanceCriteria[index] = base
			continue
		}
		resolved, err := r.store.ResolveAcceptanceCriterionEvidence(ctx, run.TenantID, run.ID, mutation.StepID, base)
		if err != nil {
			if base.BlocksCompletion() {
				return mutation, fmt.Errorf("required acceptance criterion %q is not yet satisfied: %w", base.ID, err)
			}
			base.Status = taskplan.CriterionFailed
			if taskplan.VerificationFailureReason(err) == taskplan.VerificationReasonEvidenceMissing {
				base.Status = taskplan.CriterionPending
			}
			base.VerificationReason = taskplan.VerificationFailureReason(err)
			base.VerificationMessage = err.Error()
			base.Evidence = ""
			base.EvidenceCallIDs = nil
			mutation.AcceptanceCriteria[index] = base
			continue
		}
		mutation.AcceptanceCriteria[index] = resolved
	}
	return mutation, nil
}

// ReconcileTaskPlanAfterTool advances platform-verifiable Plan state directly
// from a committed successful Tool receipt. The model still authors goals,
// graph structure and human-readable results; it no longer has to remember a
// second update_plan_step call merely to copy facts PostgreSQL already owns.
func (r *Resolver) ReconcileTaskPlanAfterTool(ctx context.Context, run agent.Run, call tool.Call) error {
	planNodeID := strings.TrimSpace(call.PlanNodeID)
	if planNodeID == "" {
		planNodeID = strings.TrimSpace(call.PlanStepID)
	}
	if planNodeID == "" || run.LeaseOwner == nil || strings.TrimSpace(*run.LeaseOwner) == "" {
		return nil
	}
	plan, err := r.store.GetTaskPlanForWorkflow(ctx, run.TenantID, run.WorkflowID)
	if errors.Is(err, taskplan.ErrNotFound) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("load Plan for receipt reconciliation: %w", err)
	}
	var active *taskplan.Step
	for index := range plan.Steps {
		if plan.Steps[index].ID == planNodeID {
			active = &plan.Steps[index]
			break
		}
	}
	if active == nil || (active.Status != taskplan.StatusInProgress && active.Status != taskplan.StatusBlocked) {
		return nil
	}
	updatedCriteria := append([]taskplan.AcceptanceCriterion(nil), active.AcceptanceCriteria...)
	changed := make([]taskplan.AcceptanceCriterion, 0, len(updatedCriteria))
	for index := range updatedCriteria {
		criterion := updatedCriteria[index]
		if criterion.Status == taskplan.CriterionPassed || criterion.Status == taskplan.CriterionSkipped ||
			criterion.Status == taskplan.CriterionInvalid || criterion.Status == taskplan.CriterionUnsupported ||
			strings.TrimSpace(criterion.Verification.Kind) == "" {
			continue
		}
		if err := taskplan.ValidateVerification(criterion.Verification); err != nil {
			continue
		}
		eligible := criterion.Verification.EvidenceTools()
		if len(eligible) != 0 {
			matchedTool := false
			for _, name := range eligible {
				if name == call.Name {
					matchedTool = true
					break
				}
			}
			if !matchedTool {
				continue
			}
		}
		resolved, resolveErr := r.store.ResolveAcceptanceCriterionEvidence(ctx, run.TenantID, run.ID, planNodeID, criterion)
		if resolveErr != nil {
			if taskplan.VerificationFailureReason(resolveErr) != taskplan.VerificationReasonAssertionFailed {
				continue
			}
			criterion.Status = taskplan.CriterionFailed
			criterion.VerificationReason = taskplan.VerificationReasonAssertionFailed
			criterion.VerificationMessage = resolveErr.Error()
			criterion.Evidence = ""
			criterion.EvidenceCallIDs = nil
			updatedCriteria[index] = criterion
			changed = append(changed, criterion)
			continue
		}
		updatedCriteria[index] = resolved
		changed = append(changed, resolved)
	}
	if len(changed) == 0 {
		return nil
	}
	supported := 0
	allSatisfied := true
	for _, criterion := range updatedCriteria {
		if criterion.Status == taskplan.CriterionInvalid || criterion.Status == taskplan.CriterionUnsupported || strings.TrimSpace(criterion.Verification.Kind) == "" {
			if criterion.BlocksCompletion() {
				allSatisfied = false
			}
			continue
		}
		if err := taskplan.ValidateVerification(criterion.Verification); err != nil {
			if criterion.BlocksCompletion() {
				allSatisfied = false
			}
			continue
		}
		supported++
		if criterion.Status != taskplan.CriterionPassed && criterion.Status != taskplan.CriterionSkipped {
			allSatisfied = false
		}
	}
	status := active.Status
	if supported > 0 && allSatisfied {
		status = taskplan.StatusCompleted
	}
	mutation := taskplan.StepUpdate{
		StepID: active.ID, Status: status,
		Result:             fmt.Sprintf("Runtime reconciled committed %s receipt %s", call.Name, call.ID),
		AcceptanceCriteria: changed,
	}
	update, err := taskplan.ApplyStepUpdate(plan, mutation)
	if err != nil {
		return fmt.Errorf("apply receipt-backed Plan progress: %w", err)
	}
	_, err = r.store.UpsertTaskPlan(ctx, agent.Lease{RunID: run.ID, Owner: *run.LeaseOwner, Token: run.LeaseToken}, run.TenantID, tool.Call{
		RunID: run.ID, WorkflowID: run.WorkflowID, WorkspaceID: call.WorkspaceID,
		Turn: call.Turn, DecisionCycle: call.DecisionCycle, Step: call.Step,
		ID: "runtime-plan-sync:" + call.ID, ActionID: call.ActionID,
		Name: "runtime_plan_sync", PlanStepID: active.ID, PlanNodeID: active.ID,
	}, update)
	if err != nil {
		return fmt.Errorf("persist receipt-backed Plan progress: %w", err)
	}
	return nil
}

func runRequiresApproval(run agent.Run, risk tool.Risk) bool {
	var snapshot struct {
		Spec agent.Spec `json:"spec"`
	}
	if json.Unmarshal(run.BindingSnapshot, &snapshot) != nil {
		return false
	}
	for _, required := range snapshot.Spec.Approval.RequireFor {
		if strings.EqualFold(strings.TrimSpace(required), string(risk)) {
			return true
		}
	}
	return false
}

// runAutoApprovesSandboxCommand is deliberately narrower than a generic
// HIGH_RISK exemption. It applies only at the workspace-provider call site
// after that site has confirmed sandbox mode and operation=run_command.
func runAutoApprovesSandboxCommand(run agent.Run) bool {
	var snapshot struct {
		Spec agent.Spec `json:"spec"`
	}
	if json.Unmarshal(run.BindingSnapshot, &snapshot) != nil {
		return false
	}
	return snapshot.Spec.Approval.AutoApproveSandboxCommand
}

func approvalExpiry(run agent.Run) time.Duration {
	var snapshot struct {
		Spec agent.Spec `json:"spec"`
	}
	if json.Unmarshal(run.BindingSnapshot, &snapshot) == nil && snapshot.Spec.Approval.ExpiresIn > 0 {
		return snapshot.Spec.Approval.ExpiresIn
	}
	return time.Hour
}

// ResolveSkillSet loads exact Skill versions and produces their deterministic
// instruction/example messages and required Tool refs.
func (r *Resolver) ResolveSkillSet(ctx context.Context, tenantID string, ref agent.VersionRef) (skill.Resolved, error) {
	setVersion, err := r.store.GetSkillSetVersion(ctx, tenantID, ref.ID)
	if err != nil {
		return skill.Resolved{}, err
	}
	if strconv.Itoa(setVersion.Version) != ref.Version {
		return skill.Resolved{}, fmt.Errorf("skill set revision changed: expected %s, got %d", ref.Version, setVersion.Version)
	}
	var set skill.SetSpec
	if err := json.Unmarshal(setVersion.Spec, &set); err != nil {
		return skill.Resolved{}, fmt.Errorf("decode skill set: %w", err)
	}
	if err := set.Validate(); err != nil {
		return skill.Resolved{}, err
	}
	resolved := skill.Resolved{}
	for _, skillRef := range set.Skills {
		version, err := r.store.GetSkillVersion(ctx, tenantID, skillRef.ID)
		if err != nil {
			return skill.Resolved{}, err
		}
		if strconv.Itoa(version.Version) != skillRef.Version {
			return skill.Resolved{}, fmt.Errorf("skill revision changed: expected %s, got %d", skillRef.Version, version.Version)
		}
		var spec skill.Spec
		if err := json.Unmarshal(version.Spec, &spec); err != nil {
			return skill.Resolved{}, fmt.Errorf("decode skill %s: %w", version.ID, err)
		}
		if err := spec.Validate(); err != nil {
			return skill.Resolved{}, err
		}
		selection := skill.Selection{Name: version.Name, Key: version.Key, Version: version.Version}
		instructions := append([]skill.InstructionBlock(nil), spec.Instructions...)
		sort.SliceStable(instructions, func(i, j int) bool { return instructions[i].Priority > instructions[j].Priority })
		for _, block := range instructions {
			resolved.InstructionMessages = append(resolved.InstructionMessages, model.TextMessage(model.RoleSystem, "Skill "+version.Name+" / "+block.Name+":\n"+block.Content))
			selection.Instructions = append(selection.Instructions, block.Name+":\n"+block.Content)
		}
		if len(spec.Examples) > 0 {
			exampleMessage := model.TextMessage(model.RoleUser, renderSkillReferenceExamples(version.Name, spec.Examples))
			exampleMessage.Metadata = map[string]string{"runtime.context_kind": "skill_reference_examples"}
			resolved.ExampleMessages = append(resolved.ExampleMessages, exampleMessage)
		}
		resolved.RequiredTools = append(resolved.RequiredTools, spec.RequiredTools...)
		resolved.VersionIDs = append(resolved.VersionIDs, version.ID)
		resolved.Skills = append(resolved.Skills, selection)
	}
	return resolved, nil
}

func renderSkillReferenceExamples(name string, examples []skill.Example) string {
	payload, _ := json.Marshal(examples)
	return "<SKILL_REFERENCE_EXAMPLES name=" + strconv.Quote(name) + ">\n" +
		"The following input/output pairs are non-active demonstrations, not conversation history or current user requests. Never answer or continue an example input. Apply only the demonstrated pattern to the final user task message.\n" +
		string(payload) + "\n</SKILL_REFERENCE_EXAMPLES>"
}

// EventSink binds all execution events to the claimed Worker generation.
func (r *Resolver) EventSink(run agent.Run) (event.Sink, error) {
	lease, err := leaseFromRun(run)
	if err != nil {
		return nil, err
	}
	return postgres.NewFencedEventSink(r.store, lease), nil
}

// LoadCheckpoint returns the newest durable state for a claimed Run.
func (r *Resolver) LoadCheckpoint(ctx context.Context, run agent.Run) (harness.Checkpoint, bool, error) {
	workflowID := run.WorkflowID
	if strings.TrimSpace(workflowID) == "" {
		workflowID = run.ID
	}
	return r.store.LoadLatestCheckpointForWorkflow(ctx, workflowID, run.ID)
}

// CheckpointSink binds checkpoint commits to the current Worker generation.
func (r *Resolver) CheckpointSink(run agent.Run) (harness.CheckpointSink, error) {
	lease, err := leaseFromRun(run)
	if err != nil {
		return nil, err
	}
	return postgres.NewFencedCheckpointSink(r.store, lease), nil
}

// SessionHistory returns completed prior Run input/output pairs.
func (r *Resolver) SessionHistory(ctx context.Context, run agent.Run, limit int) ([]model.Message, error) {
	if run.SessionID == nil {
		return nil, nil
	}
	return r.store.ListSessionMessages(ctx, run.TenantID, *run.SessionID, run.ID, limit)
}

func leaseFromRun(run agent.Run) (agent.Lease, error) {
	if run.LeaseOwner == nil || strings.TrimSpace(*run.LeaseOwner) == "" || run.LeaseToken <= 0 {
		return agent.Lease{}, errors.New("claimed run has no valid lease owner/token")
	}
	return agent.Lease{RunID: run.ID, Owner: *run.LeaseOwner, Token: run.LeaseToken}, nil
}
