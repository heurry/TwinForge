package postgres

import (
	"context"
	"fmt"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
)

func (s *RunStore) StorageSummary(ctx context.Context, tenantID string) (agent.StorageSummary, error) {
	var sessions, messages, runs, activeRuns, currentRunStates int64
	var events, memories, embeddedMemories, pendingEmbeddings int64
	var artifacts, objectArtifacts, inlineArtifacts, artifactBytes, inlineArtifactBytes, pendingOutbox int64
	var verificationIntents, verificationSpecs, verificationAttempts, verificationEvidence int64
	var observations, scores, observationArchives int64
	err := s.pool.QueryRow(ctx, `
		SELECT
			(SELECT count(*) FROM agent_platform.agent_sessions WHERE tenant_id=$1),
			(SELECT count(*) FROM agent_platform.agent_session_messages WHERE tenant_id=$1),
			(SELECT count(*) FROM agent_platform.agent_runs WHERE tenant_id=$1),
			(SELECT count(*) FROM agent_platform.agent_runs WHERE tenant_id=$1 AND status IN ('queued','running','waiting_tool','waiting_approval','waiting_input','waiting_external')),
			(SELECT count(*) FROM agent_platform.agent_run_states AS state JOIN agent_platform.agent_runs AS run ON run.id=state.run_id WHERE run.tenant_id=$1),
			(SELECT count(*) FROM agent_platform.agent_events WHERE tenant_id=$1),
			(SELECT count(*) FROM agent_platform.agent_memories WHERE tenant_id=$1 AND deleted_at IS NULL),
			(SELECT count(*) FROM agent_platform.agent_memories WHERE tenant_id=$1 AND deleted_at IS NULL AND embedding_status='ready'),
			(SELECT count(*) FROM agent_platform.agent_memories WHERE tenant_id=$1 AND deleted_at IS NULL AND embedding_status<>'ready'),
			(SELECT count(*) FROM agent_platform.agent_artifacts WHERE tenant_id=$1),
			(SELECT count(*) FROM agent_platform.agent_artifacts WHERE tenant_id=$1 AND storage_backend='minio'),
			(SELECT count(*) FROM agent_platform.agent_artifacts WHERE tenant_id=$1 AND storage_backend='inline'),
			(SELECT COALESCE(sum(size_bytes),0) FROM agent_platform.agent_artifacts WHERE tenant_id=$1),
			(SELECT COALESCE(sum(size_bytes),0) FROM agent_platform.agent_artifacts WHERE tenant_id=$1 AND storage_backend='inline'),
			(SELECT count(*) FROM agent_platform.agent_outbox AS item JOIN agent_platform.agent_runs AS run ON run.id=item.aggregate_id WHERE run.tenant_id=$1 AND item.status<>'published'),
			(SELECT count(*) FROM agent_platform.agent_verification_intents WHERE tenant_id=$1),
			(SELECT count(*) FROM agent_platform.agent_verification_specs WHERE tenant_id=$1),
			(SELECT count(*) FROM agent_platform.agent_verification_attempts WHERE tenant_id=$1),
			(SELECT count(*) FROM agent_platform.agent_evidence_records WHERE tenant_id=$1),
			(SELECT count(*) FROM agent_platform.agent_observations WHERE tenant_id=$1),
			(SELECT count(*) FROM agent_platform.agent_scores WHERE tenant_id=$1),
			(SELECT count(*) FROM agent_platform.agent_observation_archives WHERE tenant_id=$1)`, tenantID).Scan(
		&sessions, &messages, &runs, &activeRuns, &currentRunStates,
		&events, &memories, &embeddedMemories, &pendingEmbeddings,
		&artifacts, &objectArtifacts, &inlineArtifacts, &artifactBytes, &inlineArtifactBytes, &pendingOutbox,
		&verificationIntents, &verificationSpecs, &verificationAttempts, &verificationEvidence,
		&observations, &scores, &observationArchives,
	)
	if err != nil {
		return agent.StorageSummary{}, fmt.Errorf("summarize Agent storage: %w", err)
	}
	metrics := map[string]int64{
		"sessions": sessions, "messages": messages, "runs": runs, "active_runs": activeRuns,
		"current_run_states": currentRunStates, "events": events, "memories": memories,
		"embedded_memories": embeddedMemories, "pending_embeddings": pendingEmbeddings,
		"artifacts": artifacts, "object_artifacts": objectArtifacts, "inline_artifacts": inlineArtifacts,
		"artifact_bytes": artifactBytes, "inline_artifact_bytes": inlineArtifactBytes, "pending_outbox": pendingOutbox,
		"verification_intents": verificationIntents, "verification_specs": verificationSpecs,
		"verification_attempts": verificationAttempts, "verification_evidence": verificationEvidence,
		"observations": observations, "scores": scores, "observation_archives": observationArchives,
	}
	return agent.StorageSummary{
		PostgreSQL: agent.StorageBackend{Name: "PostgreSQL", Role: "唯一事实、事务与恢复", Status: "ready", Metrics: metrics, Detail: "Session、消息、Run、事件、Observation、Score、计划、验证事实和最新恢复状态均为事务事实。"},
		Redis:      agent.StorageBackend{Name: "Redis", Role: "实时通知与观测队列指针", Status: "not_configured", Metrics: map[string]int64{"pending_outbox": pendingOutbox}, Detail: "Pub/Sub 负责 Run 唤醒，observations:raw Stream 只保存 run/sequence 指针；事件正文始终从 PostgreSQL 重读。"},
		MinIO:      agent.StorageBackend{Name: "MinIO", Role: "Artifact 与原始 Observation 归档", Status: "not_configured", Metrics: map[string]int64{"object_artifacts": objectArtifacts, "inline_artifacts": inlineArtifacts, "artifact_bytes": artifactBytes, "inline_artifact_bytes": inlineArtifactBytes, "observation_archives": observationArchives}, Detail: "Artifact 与异步原始观测归档优先写入 MinIO；未配置时保留可查询的 PostgreSQL 事实。"},
		PGVector:   agent.StorageBackend{Name: "pgvector", Role: "语义记忆索引", Status: "not_configured", Metrics: map[string]int64{"embedded_memories": embeddedMemories, "pending_embeddings": pendingEmbeddings}, Detail: "Embedding Provider 未注入；召回使用 pg_trgm。"},
	}, nil
}
