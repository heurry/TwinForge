package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/interaction"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

func (s *RunStore) RequireUserInput(ctx context.Context, lease agent.Lease, tenantID string, call tool.Call, request interaction.Request) (string, error) {
	request.Question = strings.TrimSpace(request.Question)
	if request.Question == "" || len(request.Options) > 12 {
		return "", errors.New("ask_user requires a question and at most 12 options")
	}
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return "", err
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	var valid bool
	if err := tx.QueryRow(ctx, `SELECT true FROM agent_platform.agent_runs WHERE id=$1::uuid AND tenant_id=$2 AND lease_owner=$3 AND lease_token=$4 AND status='running' FOR UPDATE`, lease.RunID, tenantID, lease.Owner, lease.Token).Scan(&valid); errors.Is(err, pgx.ErrNoRows) {
		return "", agent.ErrLeaseLost
	} else if err != nil {
		return "", err
	}
	var existingQuestion, status string
	var answer *string
	err = tx.QueryRow(ctx, `SELECT question,status,answer FROM agent_platform.agent_user_questions WHERE run_id=$1::uuid AND call_id=$2 FOR UPDATE`, lease.RunID, call.ID).Scan(&existingQuestion, &status, &answer)
	if err == nil {
		if existingQuestion != request.Question {
			return "", tool.ErrCallIDConflict
		}
		if err := tx.Commit(ctx); err != nil {
			return "", err
		}
		if status == "answered" && answer != nil {
			return *answer, nil
		}
		return "", interaction.ErrInputRequired
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return "", err
	}
	options, _ := json.Marshal(request.Options)
	var questionID string
	if err := tx.QueryRow(ctx, `INSERT INTO agent_platform.agent_user_questions(tenant_id,run_id,call_id,turn_no,step_no,question,options,context) VALUES($1,$2::uuid,$3,$4,$5,$6,$7::jsonb,NULLIF($8,'')) RETURNING id::text`, tenantID, lease.RunID, call.ID, call.Turn, call.Step, request.Question, options, strings.TrimSpace(request.Context)).Scan(&questionID); err != nil {
		return "", err
	}
	if _, err := s.appendEventTx(ctx, tx, tenantID, event.Input{RunID: lease.RunID, Type: event.UserInputRequested, Turn: call.Turn, Step: call.Step, CallID: call.ID, Payload: mustJSON(map[string]any{"question_id": questionID, "question": request.Question, "options": request.Options, "context": request.Context})}); err != nil {
		return "", err
	}
	if err := tx.Commit(ctx); err != nil {
		return "", err
	}
	return "", interaction.ErrInputRequired
}

func (s *RunStore) GetPendingQuestionForRun(ctx context.Context, tenantID, runID string) (interaction.Question, error) {
	return scanQuestion(s.pool.QueryRow(ctx, `SELECT id::text,run_id::text,call_id,turn_no,step_no,question,options,COALESCE(context,''),COALESCE(answer,''),status,answered_by,created_at,answered_at FROM agent_platform.agent_user_questions WHERE tenant_id=$1 AND run_id=$2::uuid AND status='pending' ORDER BY created_at DESC LIMIT 1`, tenantID, runID))
}

func (s *RunStore) AnswerUserQuestion(ctx context.Context, tenantID, questionID, answer, actor string) (interaction.Question, error) {
	answer = strings.TrimSpace(answer)
	if answer == "" {
		return interaction.Question{}, errors.New("answer is required")
	}
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return interaction.Question{}, err
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	question, err := scanQuestion(tx.QueryRow(ctx, `SELECT id::text,run_id::text,call_id,turn_no,step_no,question,options,COALESCE(context,''),COALESCE(answer,''),status,answered_by,created_at,answered_at FROM agent_platform.agent_user_questions WHERE tenant_id=$1 AND id=$2::uuid FOR UPDATE`, tenantID, questionID))
	if err != nil {
		return interaction.Question{}, err
	}
	if question.Status == "answered" && question.Answer == answer {
		if err := tx.Commit(ctx); err != nil {
			return interaction.Question{}, err
		}
		return question, nil
	}
	if question.Status != "pending" {
		return interaction.Question{}, errors.New("user question is not pending")
	}
	now := time.Now().UTC()
	if _, err := tx.Exec(ctx, `UPDATE agent_platform.agent_user_questions SET status='answered',answer=$3,answered_by=NULLIF($4,''),answered_at=$5 WHERE id=$1::uuid AND tenant_id=$2`, questionID, tenantID, answer, actor, now); err != nil {
		return interaction.Question{}, err
	}
	// The user can answer in the narrow window after the question commits but
	// before Worker changes running -> waiting_input. Requeue both states and
	// revoke the old lease so either ordering resumes from the same checkpoint.
	command, err := tx.Exec(ctx, `UPDATE agent_platform.agent_runs SET status='queued',next_wakeup_at=now(),lease_owner=NULL,lease_expires_at=NULL,updated_at=now() WHERE id=$1::uuid AND tenant_id=$2 AND status IN ('running','waiting_input')`, question.RunID, tenantID)
	if err != nil {
		return interaction.Question{}, err
	}
	if command.RowsAffected() != 1 {
		return interaction.Question{}, errors.New("question run is no longer resumable")
	}
	var workflowID string
	if err := tx.QueryRow(ctx, `SELECT workflow_id::text FROM agent_platform.agent_runs WHERE id=$1::uuid AND tenant_id=$2::text`, question.RunID, tenantID).Scan(&workflowID); err != nil {
		return interaction.Question{}, err
	}
	if err := projectWorkflowTx(ctx, tx, tenantID, workflowID, question.RunID, "active"); err != nil {
		return interaction.Question{}, err
	}
	if _, err := s.appendEventTx(ctx, tx, tenantID, event.Input{RunID: question.RunID, Type: event.UserInputReceived, Turn: question.Turn, Step: question.Step, CallID: question.CallID, Payload: mustJSON(map[string]any{"question_id": question.ID, "answer": answer, "answered_by": actor})}); err != nil {
		return interaction.Question{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return interaction.Question{}, err
	}
	question.Status = "answered"
	question.Answer = answer
	question.AnsweredBy = optionalStringPointer(actor)
	question.AnsweredAt = &now
	return question, nil
}

type questionRow interface{ Scan(...any) error }

func scanQuestion(row questionRow) (interaction.Question, error) {
	var result interaction.Question
	var options []byte
	if err := row.Scan(&result.ID, &result.RunID, &result.CallID, &result.Turn, &result.Step, &result.Question, &options, &result.Context, &result.Answer, &result.Status, &result.AnsweredBy, &result.CreatedAt, &result.AnsweredAt); errors.Is(err, pgx.ErrNoRows) {
		return interaction.Question{}, interaction.ErrNotFound
	} else if err != nil {
		return interaction.Question{}, err
	}
	if err := json.Unmarshal(options, &result.Options); err != nil {
		return interaction.Question{}, err
	}
	return result, nil
}

func optionalStringPointer(value string) *string {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	return &value
}
