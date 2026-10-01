// Package bootstrap installs the small set of platform-owned Agent resources
// needed to demonstrate governed internal delegation.  It is deliberately
// opt-in by tenant: normal tenants must explicitly enable the bootstrap in
// their deployment configuration.
package bootstrap

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/resource"
	reviewcontract "github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/review"
)

const (
	ReviewerKey = "reviewer-agent"
	BuilderKey  = "autonomous-builder"
)

// Catalog is the subset of the Agent catalog used by the bootstrapper.
type Catalog interface {
	CreateDefinition(context.Context, agent.CreateDefinition) (agent.Definition, error)
	ListDefinitions(context.Context, string, int) ([]agent.Definition, error)
	ListVersions(context.Context, string, string) ([]agent.Version, error)
	CreateVersion(context.Context, agent.CreateVersion) (agent.Version, error)
	ReleaseVersion(context.Context, string, string) (agent.Version, error)
}

// Resources is the subset of the immutable resource catalog used by the
// bootstrapper.
type Resources interface {
	CreatePromptVersion(context.Context, resource.CreatePromptVersion) (resource.PromptVersion, error)
	ListPromptVersions(context.Context, string, int) ([]resource.PromptVersion, error)
	ListToolVersions(context.Context, string, int) ([]resource.ToolVersion, error)
	CreateToolSetVersion(context.Context, resource.CreateToolSetVersion) (resource.ToolSetVersion, error)
	ListToolSetVersions(context.Context, string, int) ([]resource.ToolSetVersion, error)
}

type Config struct {
	TenantID string
}

type Result struct {
	ReviewerAgentID       string `json:"reviewer_agent_id"`
	ReviewerVersionID     string `json:"reviewer_version_id"`
	BuilderVersionID      string `json:"builder_version_id"`
	ReviewerToolSetID     string `json:"reviewer_toolset_id"`
	ReviewerPromptID      string `json:"reviewer_prompt_id"`
	BuilderVersionCreated bool   `json:"builder_version_created"`
}

const reviewerPrompt = `You are the platform's read-only Reviewer Agent. Review the current Run workspace and the parent Agent's bounded review input. Use only the read-only tools offered to you. Do not modify files, run arbitrary commands, install dependencies, invent evidence, or emit legacy tool markup.

Your final response must be one JSON object matching the configured Output Schema, with exactly these top-level fields: verdict, summary, findings, recommended_plan_changes. verdict is pass, changes_required, or blocked. Every finding needs severity, summary, and concrete observed evidence; include path and line only when verified. Recommendations are advisory Plan changes, never claims that the parent Plan was already modified. Use pass only when no medium/high/critical finding remains. If the workspace is empty, a requested file is absent, or the available read-only evidence cannot support a conclusion, use blocked and explain the missing evidence.`

// Ensure installs an immutable Reviewer Agent and publishes a new parent
// version whose collaboration allowlist contains exactly that Reviewer. It is
// idempotent: restarting the API with the same tenant does not create another
// Reviewer version or another parent version.
func Ensure(ctx context.Context, catalog Catalog, resources Resources, cfg Config) (Result, error) {
	var result Result
	tenant := strings.TrimSpace(cfg.TenantID)
	if tenant == "" {
		return result, errors.New("collaboration bootstrap tenant is required")
	}
	definitions, err := catalog.ListDefinitions(ctx, tenant, 200)
	if err != nil {
		return result, fmt.Errorf("list agent definitions: %w", err)
	}
	var reviewer *agent.Definition
	var builder *agent.Definition
	for i := range definitions {
		switch definitions[i].Key {
		case ReviewerKey:
			reviewer = &definitions[i]
		case BuilderKey:
			builder = &definitions[i]
		}
	}
	if reviewer == nil {
		created, createErr := catalog.CreateDefinition(ctx, agent.CreateDefinition{TenantID: tenant, Key: ReviewerKey, Name: "Reviewer Agent", Description: stringPtr("只读代码与测试审查 Agent；通过 delegate_agent 被主 Agent 调用。"), Owner: stringPtr("platform")})
		if createErr != nil {
			return result, fmt.Errorf("create reviewer definition: %w", createErr)
		}
		reviewer = &created
	}
	if builder == nil {
		return result, fmt.Errorf("builder Agent %q is not registered in tenant %q", BuilderKey, tenant)
	}
	result.ReviewerAgentID = reviewer.ID

	prompt, err := ensurePrompt(ctx, resources, tenant)
	if err != nil {
		return result, err
	}
	result.ReviewerPromptID = prompt.ID
	toolset, err := ensureReviewerToolSet(ctx, resources, tenant)
	if err != nil {
		return result, err
	}
	result.ReviewerToolSetID = toolset.ID

	reviewerVersions, err := catalog.ListVersions(ctx, tenant, reviewer.ID)
	if err != nil {
		return result, fmt.Errorf("list reviewer versions: %w", err)
	}
	var reviewerVersion *agent.Version
	for i := range reviewerVersions {
		var spec agent.Spec
		if json.Unmarshal(reviewerVersions[i].Spec, &spec) == nil && spec.PromptRef.ID == prompt.ID && spec.ToolSetRef.ID == toolset.ID && sameJSON(spec.OutputSchema, reviewcontract.OutputSchema()) && reviewerVersions[i].Status == "published" {
			reviewerVersion = &reviewerVersions[i]
			break
		}
	}
	if reviewerVersion == nil {
		base, err := latestPublishedVersion(ctx, catalog, tenant, builder.ID)
		if err != nil {
			return result, fmt.Errorf("load builder model binding: %w", err)
		}
		var baseSpec agent.Spec
		if err := json.Unmarshal(base.Spec, &baseSpec); err != nil {
			return result, fmt.Errorf("decode builder spec: %w", err)
		}
		reviewerSpec := baseSpec
		reviewerSpec.Name = "Reviewer Agent"
		reviewerSpec.Description = "只读审查父 Agent 产物、测试证据和工作区状态。"
		reviewerSpec.Identity = agent.Identity{
			DisplayName:        "Reviewer Agent",
			Role:               "只读代码与测试审查员",
			Goal:               "为父 Agent 提供可验证、可定位的审查结论",
			Responsibilities:   []string{"检查工作区文件和实现一致性", "识别缺失的测试或验收证据", "给出严重级别和修复建议"},
			Boundaries:         []string{"不得修改工作区文件", "不得执行任意命令或安装依赖", "不得伪造工具结果"},
			CommunicationStyle: "简洁、基于证据、指出文件和行号",
		}
		reviewerSpec.PromptRef = agent.VersionRef{ID: prompt.ID, Version: strconv.Itoa(prompt.Version)}
		reviewerSpec.ToolSetRef = agent.VersionRef{ID: toolset.ID, Version: strconv.Itoa(toolset.Version)}
		reviewerSpec.OutputSchema = reviewcontract.OutputSchema()
		reviewerSpec.SkillSetRef = nil
		reviewerSpec.Planning = agent.PlanningPolicy{Policy: agent.PlanningPolicyAuto}
		reviewerSpec.Harness.MaxTurns = 4
		reviewerSpec.Harness.MaxSteps = 6
		reviewerSpec.Runtime.MaxModelCalls = 16
		reviewerSpec.Runtime.MaxToolCalls = 32
		reviewerSpec.Memory = agent.MemoryPolicy{Enabled: false}
		reviewerSpec.Collaboration = agent.CollaborationPolicy{}
		reviewerSpec.Metadata = map[string]string{"role": "reviewer", "provisioned_by": "platform-collaboration-bootstrap"}
		created, createErr := catalog.CreateVersion(ctx, agent.CreateVersion{TenantID: tenant, AgentID: reviewer.ID, Spec: reviewerSpec, CreatedBy: stringPtr("platform-bootstrap")})
		if createErr != nil {
			return result, fmt.Errorf("create reviewer version: %w", createErr)
		}
		released, releaseErr := catalog.ReleaseVersion(ctx, tenant, created.ID)
		if releaseErr != nil {
			return result, fmt.Errorf("publish reviewer version: %w", releaseErr)
		}
		reviewerVersion = &released
	}
	result.ReviewerVersionID = reviewerVersion.ID

	builderVersions, err := catalog.ListVersions(ctx, tenant, builder.ID)
	if err != nil {
		return result, fmt.Errorf("list builder versions: %w", err)
	}
	var active *agent.Version
	for i := range builderVersions {
		if builder.ActiveVersionID != nil && builderVersions[i].ID == *builder.ActiveVersionID {
			active = &builderVersions[i]
			break
		}
	}
	if active == nil {
		active, err = latestPublishedVersion(ctx, catalog, tenant, builder.ID)
		if err != nil {
			return result, fmt.Errorf("resolve active builder version: %w", err)
		}
	}
	var builderSpec agent.Spec
	if err := json.Unmarshal(active.Spec, &builderSpec); err != nil {
		return result, fmt.Errorf("decode active builder spec: %w", err)
	}
	if hasTarget(builderSpec.Collaboration, reviewer.ID, reviewerVersion.ID) {
		result.BuilderVersionID = active.ID
		return result, nil
	}
	builderSpec.Collaboration = agent.CollaborationPolicy{
		AllowedTargets: []agent.CollaborationTarget{{AgentID: reviewer.ID, AgentVersionID: reviewerVersion.ID, Modes: []string{"sync"}}},
		MaxDepth:       1, MaxFanOut: 1, MaxChildRuns: 1,
		ShareSessionMemory: false, PropagateUserIdentity: true,
		ChildTimeout: 10 * time.Minute,
		Budget:       agent.CollaborationBudget{MaxModelCalls: 16, MaxToolCalls: 32, MaxTokens: 100000},
	}
	builderSpec.Metadata = cloneMetadata(builderSpec.Metadata)
	builderSpec.Metadata["collaboration_bootstrap"] = "reviewer-agent"
	created, err := catalog.CreateVersion(ctx, agent.CreateVersion{TenantID: tenant, AgentID: builder.ID, Spec: builderSpec, CreatedBy: stringPtr("platform-bootstrap")})
	if err != nil {
		return result, fmt.Errorf("create builder collaboration version: %w", err)
	}
	released, err := catalog.ReleaseVersion(ctx, tenant, created.ID)
	if err != nil {
		return result, fmt.Errorf("publish builder collaboration version: %w", err)
	}
	result.BuilderVersionID = released.ID
	result.BuilderVersionCreated = true
	return result, nil
}

func sameJSON(left, right json.RawMessage) bool {
	var leftValue, rightValue any
	if json.Unmarshal(left, &leftValue) != nil || json.Unmarshal(right, &rightValue) != nil {
		return false
	}
	return reflect.DeepEqual(leftValue, rightValue)
}

func ensurePrompt(ctx context.Context, resources Resources, tenant string) (resource.PromptVersion, error) {
	items, err := resources.ListPromptVersions(ctx, tenant, 500)
	if err != nil {
		return resource.PromptVersion{}, fmt.Errorf("list prompts: %w", err)
	}
	for _, item := range items {
		if item.TenantID == tenant && item.Key == ReviewerKey && item.Content == reviewerPrompt {
			return item, nil
		}
	}
	item, err := resources.CreatePromptVersion(ctx, resource.CreatePromptVersion{TenantID: tenant, Key: ReviewerKey, Name: "Reviewer Agent System Prompt", Content: reviewerPrompt, CreatedBy: stringPtr("platform-bootstrap")})
	if err != nil {
		return resource.PromptVersion{}, fmt.Errorf("create reviewer prompt: %w", err)
	}
	return item, nil
}

func ensureReviewerToolSet(ctx context.Context, resources Resources, tenant string) (resource.ToolSetVersion, error) {
	tools, err := resources.ListToolVersions(ctx, tenant, 500)
	if err != nil {
		return resource.ToolSetVersion{}, fmt.Errorf("list tools: %w", err)
	}
	required := []string{"workspace-list-files", "workspace-read-file", "workspace-search-files"}
	refs := make([]resource.VersionRef, 0, len(required))
	for _, key := range required {
		var found *resource.ToolVersion
		for i := range tools {
			if tools[i].TenantID == tenant && tools[i].Key == key && tools[i].Status == "published" && (found == nil || tools[i].Version > found.Version) {
				found = &tools[i]
			}
		}
		if found == nil {
			return resource.ToolSetVersion{}, fmt.Errorf("published reviewer tool %q is missing", key)
		}
		refs = append(refs, resource.VersionRef{ID: found.ID, Version: strconv.Itoa(found.Version)})
	}
	spec := resource.ToolSetSpec{Tools: refs}
	sets, err := resources.ListToolSetVersions(ctx, tenant, 500)
	if err != nil {
		return resource.ToolSetVersion{}, fmt.Errorf("list toolsets: %w", err)
	}
	for _, item := range sets {
		if item.TenantID != tenant || item.Key != ReviewerKey || item.Status != "published" {
			continue
		}
		var existing resource.ToolSetSpec
		if json.Unmarshal(item.Spec, &existing) == nil && reflect.DeepEqual(existing, spec) {
			return item, nil
		}
	}
	item, err := resources.CreateToolSetVersion(ctx, resource.CreateToolSetVersion{TenantID: tenant, Key: ReviewerKey, Name: "Reviewer Read-only Workspace", Spec: spec, CreatedBy: stringPtr("platform-bootstrap")})
	if err != nil {
		return resource.ToolSetVersion{}, fmt.Errorf("create reviewer toolset: %w", err)
	}
	return item, nil
}

func latestPublishedVersion(ctx context.Context, catalog Catalog, tenant, agentID string) (*agent.Version, error) {
	versions, err := catalog.ListVersions(ctx, tenant, agentID)
	if err != nil {
		return nil, err
	}
	for i := range versions {
		if versions[i].Status == "published" {
			return &versions[i], nil
		}
	}
	return nil, fmt.Errorf("agent %s has no published version", agentID)
}

func hasTarget(policy agent.CollaborationPolicy, agentID, versionID string) bool {
	for _, target := range policy.AllowedTargets {
		if target.AgentID == agentID && target.AgentVersionID == versionID && len(target.Modes) == 1 && target.Modes[0] == "sync" {
			return true
		}
	}
	return false
}

func cloneMetadata(input map[string]string) map[string]string {
	output := make(map[string]string, len(input)+1)
	for key, value := range input {
		output[key] = value
	}
	return output
}

func stringPtr(value string) *string { return &value }
