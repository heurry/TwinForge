UPDATE agent_platform.agent_task_plans AS plan
SET steps = backup.steps,
    revision = plan.revision + 1,
    updated_at = now()
FROM agent_platform.agent_plan_verification_migration_49_backup AS backup
WHERE plan.plan_id = backup.plan_id;

UPDATE agent_platform.agent_plan_node_states AS state
SET plan_revision = plan.revision,
    updated_at = now()
FROM agent_platform.agent_task_plans AS plan
WHERE state.workflow_id = plan.workflow_id
  AND plan.plan_id IN (SELECT plan_id FROM agent_platform.agent_plan_verification_migration_49_backup)
  AND state.plan_revision <> plan.revision;

DROP TABLE IF EXISTS agent_platform.agent_plan_verification_migration_49_backup;
