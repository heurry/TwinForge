package httpx

import (
	"context"
	"errors"
	"net/http"
	"strings"

	"github.com/heurry/cloudnative-infra-platform/server/internal/store"
	"github.com/jackc/pgx/v5"
)

// gatherTrainingEvidence 聚合训练台账、实时 PyTorchJob、Pod 日志和节点 GPU 快照。
func (a *API) gatherTrainingEvidence(ctx context.Context, jobID string) (map[string]any, error) {
	var job store.TrainingJob
	var err error
	if jobID != "" {
		job, err = a.Store.GetTrainingJob(ctx, jobID)
	} else {
		jobs, listErr := a.Store.ListTrainingJobs(ctx)
		if listErr != nil {
			return nil, listErr
		}
		if len(jobs) == 0 {
			return nil, pgx.ErrNoRows
		}
		job = jobs[0]
	}
	if err != nil {
		return nil, err
	}

	evidence := map[string]any{
		"job": job,
		"gpu": a.Agent.FetchObject(ctx, "/api/gpu"),
	}
	if a.Training != nil {
		if live, liveErr := a.Training.GetJob(ctx, job.Namespace, job.Name); liveErr == nil {
			evidence["pytorch_job"] = live
		} else {
			evidence["pytorch_job"] = map[string]any{"available": false, "error": liveErr.Error()}
		}
	} else {
		evidence["pytorch_job"] = map[string]any{"available": false, "error": orDefault(a.TrainingErr, "training client unavailable")}
	}

	podEvidence := map[string]any{"pod": nil, "logs": "", "available": false}
	if a.K8s != nil {
		if pods, podsErr := a.K8s.Pods(ctx); podsErr == nil {
			pod := pickTrainingPod(pods, job.Namespace, job.Name)
			if pod != "" {
				podEvidence["pod"] = pod
				if logs, logsErr := a.K8s.PodLogs(ctx, job.Namespace, pod, trainingLogTail); logsErr == nil {
					podEvidence["available"] = true
					podEvidence["logs"] = truncateTrainingEvidence(logs, 16000)
				} else {
					podEvidence["error"] = logsErr.Error()
				}
			}
		} else {
			podEvidence["error"] = podsErr.Error()
		}
	}
	evidence["pod"] = podEvidence
	return evidence, nil
}

func truncateTrainingEvidence(value string, limit int) string {
	value = strings.TrimSpace(value)
	if len(value) <= limit {
		return value
	}
	return value[len(value)-limit:]
}

// GET /api/ai/training/evidence：预览训练专项诊断将使用的真实证据。
func (a *API) trainingDiagnosisEvidence(w http.ResponseWriter, r *http.Request) {
	evidence, err := a.gatherTrainingEvidence(r.Context(), r.URL.Query().Get("job_id"))
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			WriteError(w, r, http.StatusNotFound, "training_job_not_found", "没有可用于诊断的训练任务")
			return
		}
		a.fail(w, r, err)
		return
	}
	WriteJSON(w, http.StatusOK, map[string]any{"training": evidence})
}
