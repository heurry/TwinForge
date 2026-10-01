package httpx

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestGitLabPipelineState(t *testing.T) {
	tests := []struct {
		input      string
		status     string
		conclusion string
	}{
		{input: "pending", status: "queued"},
		{input: "running", status: "in_progress"},
		{input: "success", status: "completed", conclusion: "success"},
		{input: "failed", status: "completed", conclusion: "failure"},
		{input: "canceled", status: "completed", conclusion: "cancelled"},
	}
	for _, test := range tests {
		t.Run(test.input, func(t *testing.T) {
			status, conclusion := gitLabPipelineState(test.input)
			if status != test.status {
				t.Fatalf("status=%q, want %q", status, test.status)
			}
			if test.conclusion == "" && conclusion != nil {
				t.Fatalf("conclusion=%q, want nil", *conclusion)
			}
			if test.conclusion != "" && (conclusion == nil || *conclusion != test.conclusion) {
				t.Fatalf("conclusion=%v, want %q", conclusion, test.conclusion)
			}
		})
	}
}

func TestGitLabCIRunsNormalizesPipeline(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("PRIVATE-TOKEN") != "test-token" {
			t.Fatalf("missing GitLab token header")
		}
		if !strings.Contains(r.RequestURI, "/api/v4/projects/group%2Fproject/pipelines") {
			t.Fatalf("unexpected request URI: %s", r.RequestURI)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[{
          "id": 42,
          "status": "running",
          "ref": "main",
          "sha": "1234567890abcdef",
          "web_url": "https://gitlab.example/group/project/-/pipelines/42",
          "created_at": "2026-08-29T02:00:00Z",
          "updated_at": "2026-08-29T02:01:00Z"
        }]`))
	}))
	defer server.Close()

	a := &API{
		CIProvider: "gitlab", GitLabBaseURL: server.URL, GitLabProjectID: "group/project",
		GitLabToken: "test-token", GitLabRef: "main",
	}
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/api/ci/runs?limit=8", nil)
	a.ciRuns(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	var payload struct {
		Configured bool       `json:"configured"`
		Provider   string     `json:"provider"`
		Runs       []ciRunDTO `json:"runs"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if !payload.Configured || payload.Provider != "gitlab" || len(payload.Runs) != 1 {
		t.Fatalf("unexpected response: %+v", payload)
	}
	run := payload.Runs[0]
	if run.ID != 42 || run.Status != "in_progress" || run.HeadBranch != "main" || run.HeadSHA != "1234567890abcdef" {
		t.Fatalf("unexpected normalized pipeline: %+v", run)
	}
	if run.Conclusion != nil || run.DisplayTitle != "Pipeline #42" {
		t.Fatalf("unexpected display state: %+v", run)
	}
}

func TestGitLabCIRunsReportsMissingConfiguration(t *testing.T) {
	a := &API{CIProvider: "gitlab", GitLabBaseURL: "https://gitlab.com", GitLabProjectID: "group/project", GitLabRef: "main"}
	recorder := httptest.NewRecorder()
	a.ciRuns(recorder, httptest.NewRequest(http.MethodGet, "/api/ci/runs", nil))
	if recorder.Code != http.StatusOK || !strings.Contains(recorder.Body.String(), `"configured":false`) || !strings.Contains(recorder.Body.String(), `"provider":"gitlab"`) {
		t.Fatalf("unexpected response: %d %s", recorder.Code, recorder.Body.String())
	}
}
