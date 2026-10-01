package httpx

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

var ciHTTPClient = &http.Client{Timeout: 15 * time.Second}

type triggerCIInput struct {
	Ref      string            `json:"ref"`
	Inputs   map[string]string `json:"inputs"`
	Operator string            `json:"operator"`
}

type gitLabPipeline struct {
	ID        int64  `json:"id"`
	Name      string `json:"name"`
	Status    string `json:"status"`
	Ref       string `json:"ref"`
	SHA       string `json:"sha"`
	WebURL    string `json:"web_url"`
	CreatedAt string `json:"created_at"`
	UpdatedAt string `json:"updated_at"`
}

type ciRunDTO struct {
	ID           int64   `json:"id"`
	Name         string  `json:"name"`
	DisplayTitle string  `json:"display_title"`
	Status       string  `json:"status"`
	Conclusion   *string `json:"conclusion"`
	HTMLURL      string  `json:"html_url"`
	HeadBranch   string  `json:"head_branch"`
	HeadSHA      string  `json:"head_sha"`
	CreatedAt    string  `json:"created_at"`
	UpdatedAt    string  `json:"updated_at"`
}

func (a *API) ciProviderName() string {
	provider := strings.ToLower(strings.TrimSpace(a.CIProvider))
	if provider == "" || provider == "gitlab_ci" {
		return "gitlab"
	}
	return provider
}

func (a *API) gitLabConfigured() bool {
	return strings.TrimSpace(a.GitLabBaseURL) != "" && strings.TrimSpace(a.GitLabProjectID) != "" && strings.TrimSpace(a.GitLabToken) != ""
}

func (a *API) githubActionsConfigured() bool {
	return strings.TrimSpace(a.GitHubRepository) != "" && strings.TrimSpace(a.GitHubToken) != "" && strings.TrimSpace(a.GitHubWorkflow) != ""
}

func jsonRequest(r *http.Request, method, endpoint string, body any) (*http.Request, error) {
	var payload []byte
	if body != nil {
		payload, _ = json.Marshal(body)
	}
	req, err := http.NewRequestWithContext(r.Context(), method, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	return req, nil
}

func (a *API) gitLabRequest(r *http.Request, method, path string, body any) (*http.Response, error) {
	req, err := jsonRequest(r, method, strings.TrimRight(a.GitLabBaseURL, "/")+path, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("PRIVATE-TOKEN", a.GitLabToken)
	return ciHTTPClient.Do(req)
}

func (a *API) githubRequest(r *http.Request, method, path string, body any) (*http.Response, error) {
	req, err := jsonRequest(r, method, "https://api.github.com"+path, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("Authorization", "Bearer "+a.GitHubToken)
	req.Header.Set("X-GitHub-Api-Version", "2022-11-28")
	return ciHTTPClient.Do(req)
}

func gitLabPipelineState(value string) (string, *string) {
	switch strings.ToLower(value) {
	case "created", "waiting_for_resource", "preparing", "pending", "scheduled", "manual":
		return "queued", nil
	case "running":
		return "in_progress", nil
	case "success":
		result := "success"
		return "completed", &result
	case "failed":
		result := "failure"
		return "completed", &result
	case "canceled":
		result := "cancelled"
		return "completed", &result
	case "skipped":
		result := "skipped"
		return "completed", &result
	default:
		return value, nil
	}
}

func normalizeGitLabPipeline(pipeline gitLabPipeline) ciRunDTO {
	status, conclusion := gitLabPipelineState(pipeline.Status)
	title := strings.TrimSpace(pipeline.Name)
	if title == "" {
		title = fmt.Sprintf("Pipeline #%d", pipeline.ID)
	}
	return ciRunDTO{
		ID: pipeline.ID, Name: title, DisplayTitle: title, Status: status, Conclusion: conclusion,
		HTMLURL: pipeline.WebURL, HeadBranch: pipeline.Ref, HeadSHA: pipeline.SHA,
		CreatedAt: pipeline.CreatedAt, UpdatedAt: pipeline.UpdatedAt,
	}
}

func (a *API) ciRuns(w http.ResponseWriter, r *http.Request) {
	switch a.ciProviderName() {
	case "gitlab":
		a.gitLabCIRuns(w, r)
	case "github", "github_actions":
		a.githubCIRuns(w, r)
	default:
		WriteJSON(w, http.StatusOK, map[string]any{"configured": false, "provider": a.ciProviderName(), "runs": []any{},
			"message": "Unsupported CI_PROVIDER. Use gitlab or github_actions."})
	}
}

func (a *API) gitLabCIRuns(w http.ResponseWriter, r *http.Request) {
	ref := orDefault(strings.TrimSpace(a.GitLabRef), "main")
	if !a.gitLabConfigured() {
		WriteJSON(w, http.StatusOK, map[string]any{"configured": false, "provider": "gitlab", "runs": []any{}, "default_ref": ref,
			"repository": a.GitLabProjectID, "message": "请配置 GITLAB_PROJECT_ID 和具有 api 权限的 GITLAB_TOKEN。"})
		return
	}
	project := url.PathEscape(strings.TrimSpace(a.GitLabProjectID))
	path := fmt.Sprintf("/api/v4/projects/%s/pipelines?per_page=%d&ref=%s&order_by=id&sort=desc",
		project, clamp(queryInt(r, "limit", 10), 1, 50), url.QueryEscape(ref))
	resp, err := a.gitLabRequest(r, http.MethodGet, path, nil)
	if err != nil {
		WriteError(w, r, http.StatusBadGateway, "ci_provider_unavailable", err.Error())
		return
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		WriteError(w, r, http.StatusBadGateway, "ci_provider_error", fmt.Sprintf("GitLab returned HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw))))
		return
	}
	var pipelines []gitLabPipeline
	if err := json.Unmarshal(raw, &pipelines); err != nil {
		a.fail(w, r, err)
		return
	}
	runs := make([]ciRunDTO, 0, len(pipelines))
	for _, pipeline := range pipelines {
		runs = append(runs, normalizeGitLabPipeline(pipeline))
	}
	WriteJSON(w, http.StatusOK, map[string]any{"configured": true, "provider": "gitlab", "repository": a.GitLabProjectID,
		"workflow": ".gitlab-ci.yml", "default_ref": ref, "runs": runs})
}

func (a *API) githubCIRuns(w http.ResponseWriter, r *http.Request) {
	if !a.githubActionsConfigured() {
		WriteJSON(w, http.StatusOK, map[string]any{"configured": false, "provider": "github_actions", "runs": []any{}, "default_ref": "main",
			"message": "Set GITHUB_REPOSITORY, GITHUB_TOKEN and optionally GITHUB_WORKFLOW_FILE to connect CI."})
		return
	}
	path := fmt.Sprintf("/repos/%s/actions/workflows/%s/runs?per_page=%d", a.GitHubRepository, url.PathEscape(a.GitHubWorkflow), clamp(queryInt(r, "limit", 10), 1, 50))
	resp, err := a.githubRequest(r, http.MethodGet, path, nil)
	if err != nil {
		WriteError(w, r, http.StatusBadGateway, "ci_provider_unavailable", err.Error())
		return
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		WriteError(w, r, http.StatusBadGateway, "ci_provider_error", fmt.Sprintf("GitHub Actions returned HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw))))
		return
	}
	var payload struct {
		Runs []map[string]any `json:"workflow_runs"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{"configured": true, "provider": "github_actions", "repository": a.GitHubRepository,
		"workflow": a.GitHubWorkflow, "default_ref": "main", "runs": payload.Runs})
}

func (a *API) triggerCIRun(w http.ResponseWriter, r *http.Request) {
	var input triggerCIInput
	if err := decodeBody(r, &input); err != nil {
		a.badRequest(w, r, "invalid body")
		return
	}
	switch a.ciProviderName() {
	case "gitlab":
		a.triggerGitLabCIRun(w, r, input)
	case "github", "github_actions":
		a.triggerGitHubCIRun(w, r, input)
	default:
		WriteError(w, r, http.StatusServiceUnavailable, "ci_not_configured", "Unsupported CI_PROVIDER")
	}
}

func (a *API) triggerGitLabCIRun(w http.ResponseWriter, r *http.Request, input triggerCIInput) {
	if !a.gitLabConfigured() {
		WriteError(w, r, http.StatusServiceUnavailable, "ci_not_configured", "GitLab CI integration is not configured")
		return
	}
	input.Ref = orDefault(strings.TrimSpace(input.Ref), orDefault(strings.TrimSpace(a.GitLabRef), "main"))
	keys := make([]string, 0, len(input.Inputs))
	for key := range input.Inputs {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	variables := make([]map[string]string, 0, len(keys))
	for _, key := range keys {
		variables = append(variables, map[string]string{"key": key, "value": input.Inputs[key]})
	}
	project := url.PathEscape(strings.TrimSpace(a.GitLabProjectID))
	payload := map[string]any{"ref": input.Ref}
	if len(variables) > 0 {
		payload["variables"] = variables
	}
	resp, err := a.gitLabRequest(r, http.MethodPost, fmt.Sprintf("/api/v4/projects/%s/pipeline", project), payload)
	if err != nil {
		WriteError(w, r, http.StatusBadGateway, "ci_provider_unavailable", err.Error())
		return
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != http.StatusCreated {
		WriteError(w, r, http.StatusBadGateway, "ci_dispatch_failed", fmt.Sprintf("GitLab returned HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw))))
		return
	}
	operator := a.actor(r, input.Operator)
	a.Store.Audit(r.Context(), operator, "operator", "ci.run.trigger", "repository", a.GitLabProjectID,
		map[string]any{"provider": "gitlab", "ref": input.Ref, "variables": input.Inputs})
	_ = a.recordPlatformLog(r.Context(), platformLogInput{Level: "info", Source: "gitlab-ci", ResourceType: "repository", ResourceID: a.GitLabProjectID,
		Message: "GitLab pipeline created", Attributes: map[string]any{"ref": input.Ref}})
	WriteJSON(w, http.StatusAccepted, map[string]any{"status": "queued", "provider": "gitlab", "repository": a.GitLabProjectID, "ref": input.Ref})
}

func (a *API) triggerGitHubCIRun(w http.ResponseWriter, r *http.Request, input triggerCIInput) {
	if !a.githubActionsConfigured() {
		WriteError(w, r, http.StatusServiceUnavailable, "ci_not_configured", "GitHub Actions integration is not configured")
		return
	}
	input.Ref = orDefault(strings.TrimSpace(input.Ref), "main")
	path := fmt.Sprintf("/repos/%s/actions/workflows/%s/dispatches", a.GitHubRepository, url.PathEscape(a.GitHubWorkflow))
	payload := map[string]any{"ref": input.Ref}
	if len(input.Inputs) > 0 {
		payload["inputs"] = input.Inputs
	}
	resp, err := a.githubRequest(r, http.MethodPost, path, payload)
	if err != nil {
		WriteError(w, r, http.StatusBadGateway, "ci_provider_unavailable", err.Error())
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		WriteError(w, r, http.StatusBadGateway, "ci_dispatch_failed", fmt.Sprintf("GitHub Actions returned HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw))))
		return
	}
	operator := a.actor(r, input.Operator)
	a.Store.Audit(r.Context(), operator, "operator", "ci.run.trigger", "repository", a.GitHubRepository,
		map[string]any{"provider": "github_actions", "workflow": a.GitHubWorkflow, "ref": input.Ref, "inputs": input.Inputs})
	_ = a.recordPlatformLog(r.Context(), platformLogInput{Level: "info", Source: "github-actions", ResourceType: "repository", ResourceID: a.GitHubRepository,
		Message: "CI workflow dispatched", Attributes: map[string]any{"workflow": a.GitHubWorkflow, "ref": input.Ref}})
	WriteJSON(w, http.StatusAccepted, map[string]any{"status": "queued", "provider": "github_actions", "repository": a.GitHubRepository,
		"workflow": a.GitHubWorkflow, "ref": input.Ref})
}
