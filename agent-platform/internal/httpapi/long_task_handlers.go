package httpapi

import (
	"errors"
	"net/http"
	"strings"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/interaction"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/taskplan"
	verificationdomain "github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/verification"
)

func (s *Server) getRunPlan(w http.ResponseWriter, r *http.Request) {
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
	if errors.Is(err, agent.ErrRunNotFound) {
		writeError(w, http.StatusNotFound, "run_not_found", "run was not found")
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "run_read_failed", err.Error())
		return
	}
	plan, err := s.planStore.GetTaskPlanForWorkflow(r.Context(), tenantID, run.WorkflowID)
	if errors.Is(err, taskplan.ErrNotFound) {
		writeJSON(w, http.StatusOK, map[string]any{"data": nil})
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "plan_read_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": plan})
}

func (s *Server) listRunVerifications(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	runID := r.PathValue("run_id")
	if !uuidPattern.MatchString(runID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "run_id must be a UUID")
		return
	}
	records, err := s.planStore.ListVerificationRecordsForTenant(r.Context(), tenantID, runID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "verification_read_failed", err.Error())
		return
	}
	if records == nil {
		records = make([]verificationdomain.Record, 0)
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": records})
}

func (s *Server) getRunQuestion(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	runID := r.PathValue("run_id")
	if !uuidPattern.MatchString(runID) {
		writeError(w, http.StatusBadRequest, "invalid_request", "run_id must be a UUID")
		return
	}
	question, err := s.interactionStore.GetPendingQuestionForRun(r.Context(), tenantID, runID)
	if errors.Is(err, interaction.ErrNotFound) {
		writeJSON(w, http.StatusOK, map[string]any{"data": nil})
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "question_read_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": question})
}

type answerQuestionRequest struct {
	Answer string `json:"answer"`
}

func (s *Server) answerRunQuestion(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requireTenant(w, r)
	if !ok {
		return
	}
	id, matched := actionID(r.PathValue("question_action"), "answer")
	if !matched || !uuidPattern.MatchString(id) {
		writeError(w, http.StatusBadRequest, "invalid_request", "question action must be <uuid>:answer")
		return
	}
	actor := strings.TrimSpace(r.Header.Get("X-Actor-ID"))
	if actor == "" {
		writeError(w, http.StatusBadRequest, "missing_actor", "X-Actor-ID header is required")
		return
	}
	var request answerQuestionRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	question, err := s.interactionStore.AnswerUserQuestion(r.Context(), tenantID, id, request.Answer, actor)
	if err != nil {
		writeError(w, http.StatusConflict, "question_answer_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": question})
}
