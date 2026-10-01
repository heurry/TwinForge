package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/a2a"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/jackc/pgx/v5"
)

func (s *RunStore) CreateA2ATask(ctx context.Context, tenantID, agentID string, request a2a.SendMessageRequest, createdBy *string, traceparent *string) (a2a.Task, error) {
	if request.Message.MessageID == "" || request.Message.Role != "user" || len(request.Message.Parts) == 0 {
		return a2a.Task{}, errors.New("A2A messageId, user role and at least one part are required")
	}
	if existing, err := s.GetA2ATaskByMessage(ctx, tenantID, agentID, request.Message.MessageID); err == nil {
		return existing, nil
	}
	messageRaw, _ := json.Marshal(request.Message)
	var taskID, contextID string
	err := s.pool.QueryRow(ctx, `INSERT INTO agent_platform.a2a_tasks(tenant_id,agent_id,client_message_id,message) VALUES($1::text,$2::uuid,$3::text,$4::jsonb) ON CONFLICT(tenant_id,agent_id,client_message_id) DO UPDATE SET client_message_id=EXCLUDED.client_message_id RETURNING id::text,context_id::text`, tenantID, agentID, request.Message.MessageID, messageRaw).Scan(&taskID, &contextID)
	if err != nil {
		return a2a.Task{}, err
	}
	definition, err := s.GetDefinition(ctx, tenantID, agentID)
	if err != nil {
		return a2a.Task{}, err
	}
	if definition.ActiveVersionID == nil {
		return a2a.Task{}, errors.New("A2A target has no active published AgentVersion")
	}
	text := ""
	for _, part := range request.Message.Parts {
		if part.Text != "" {
			if text != "" {
				text += "\n"
			}
			text += part.Text
		}
	}
	input, _ := json.Marshal(map[string]any{"question": text})
	for _, part := range request.Message.Parts {
		var object map[string]any
		if len(part.Data) > 0 && json.Unmarshal(part.Data, &object) == nil {
			input = part.Data
			break
		}
	}
	run, err := s.CreateRun(ctx, agent.CreateRun{TenantID: tenantID, AgentVersionID: *definition.ActiveVersionID, TriggerType: "a2a", Input: input, CreatedBy: createdBy, TraceParent: traceparent})
	if err != nil {
		_, _ = s.pool.Exec(ctx, `UPDATE agent_platform.a2a_tasks SET status='rejected',error=$3::text,updated_at=now() WHERE id=$1::uuid AND tenant_id=$2::text`, taskID, tenantID, err.Error())
		return a2a.Task{}, err
	}
	_, err = s.pool.Exec(ctx, `UPDATE agent_platform.a2a_tasks SET run_id=$3::uuid,status='working',updated_at=now() WHERE id=$1::uuid AND tenant_id=$2::text`, taskID, tenantID, run.ID)
	if err != nil {
		return a2a.Task{}, err
	}
	request.Message.ContextID = contextID
	request.Message.TaskID = taskID
	return a2a.Task{ID: taskID, ContextID: contextID, Status: a2a.TaskStatus{State: "working", Timestamp: time.Now().UTC()}, History: []a2a.Message{request.Message}, Metadata: map[string]any{"runId": run.ID}}, nil
}

func (s *RunStore) GetA2ATaskByMessage(ctx context.Context, tenantID, agentID, messageID string) (a2a.Task, error) {
	var id string
	err := s.pool.QueryRow(ctx, `SELECT id::text FROM agent_platform.a2a_tasks WHERE tenant_id=$1::text AND agent_id=$2::uuid AND client_message_id=$3::text`, tenantID, agentID, messageID).Scan(&id)
	if err != nil {
		return a2a.Task{}, err
	}
	return s.GetA2ATaskForTenant(ctx, tenantID, id)
}

func (s *RunStore) GetA2ATaskForTenant(ctx context.Context, tenantID, taskID string) (a2a.Task, error) {
	var contextID, status string
	var message json.RawMessage
	var runID *string
	var storedError *string
	var updated time.Time
	err := s.pool.QueryRow(ctx, `SELECT task.context_id::text,task.status,task.message,task.run_id::text,task.error,task.updated_at FROM agent_platform.a2a_tasks task WHERE task.id=$1::uuid AND task.tenant_id=$2::text`, taskID, tenantID).Scan(&contextID, &status, &message, &runID, &storedError, &updated)
	if errors.Is(err, pgx.ErrNoRows) {
		return a2a.Task{}, errors.New("A2A task not found")
	}
	if err != nil {
		return a2a.Task{}, err
	}
	var original a2a.Message
	_ = json.Unmarshal(message, &original)
	original.TaskID = taskID
	original.ContextID = contextID
	task := a2a.Task{ID: taskID, ContextID: contextID, History: []a2a.Message{original}, Metadata: map[string]any{}}
	if runID != nil {
		run, runErr := s.GetRunForTenant(ctx, tenantID, *runID)
		if runErr == nil {
			status = mapRunToA2A(run.Status)
			updated = run.UpdatedAt
			task.Metadata["runId"] = run.ID
			if run.Status == agent.RunCompleted {
				part := a2a.Part{Data: run.Output}
				task.Artifacts = append(task.Artifacts, a2a.Artifact{ArtifactID: "run-output-" + run.ID, Name: "Agent result", Parts: []a2a.Part{part}})
			}
		}
	}
	task.Status = a2a.TaskStatus{State: status, Timestamp: updated}
	if storedError != nil {
		task.Status.Message = &a2a.Message{MessageID: "error-" + taskID, TaskID: taskID, ContextID: contextID, Role: "agent", Parts: []a2a.Part{{Text: *storedError}}}
	}
	return task, nil
}

func (s *RunStore) CancelA2ATask(ctx context.Context, tenantID, taskID, actor string) (a2a.Task, error) {
	task, err := s.GetA2ATaskForTenant(ctx, tenantID, taskID)
	if err != nil {
		return task, err
	}
	runID, _ := task.Metadata["runId"].(string)
	if runID == "" {
		return task, errors.New("A2A task has no Run")
	}
	if err := s.RequestCancelForTenant(ctx, tenantID, runID, actor); err != nil {
		return task, err
	}
	_, _ = s.pool.Exec(ctx, `UPDATE agent_platform.a2a_tasks SET status='canceled',updated_at=now() WHERE id=$1::uuid AND tenant_id=$2::text`, taskID, tenantID)
	return s.GetA2ATaskForTenant(ctx, tenantID, taskID)
}

func mapRunToA2A(status agent.RunStatus) string {
	switch status {
	case agent.RunQueued:
		return "submitted"
	case agent.RunRunning, agent.RunWaitingTool, agent.RunWaitingExternal:
		return "working"
	case agent.RunWaitingApproval, agent.RunWaitingInput, agent.RunSuspended:
		return "input-required"
	case agent.RunCompleted:
		return "completed"
	case agent.RunFailed:
		return "failed"
	case agent.RunCancelled:
		return "canceled"
	default:
		return fmt.Sprint(status)
	}
}
