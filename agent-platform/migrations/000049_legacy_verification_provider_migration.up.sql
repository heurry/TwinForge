-- Preserve an exact rollback image before rewriting legacy, unregistered
-- advisory verification kinds to current registered providers.
CREATE TABLE IF NOT EXISTS agent_platform.agent_plan_verification_migration_49_backup (
    plan_id UUID PRIMARY KEY REFERENCES agent_platform.agent_task_plans(plan_id) ON DELETE CASCADE,
    steps JSONB NOT NULL,
    backed_up_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO agent_platform.agent_plan_verification_migration_49_backup(plan_id, steps)
SELECT plan_id, steps
FROM agent_platform.agent_task_plans
WHERE EXISTS (
    SELECT 1
    FROM jsonb_array_elements(steps) AS step,
         jsonb_array_elements(COALESCE(step->'acceptance_criteria','[]'::jsonb)) AS criterion
    WHERE COALESCE(criterion#>>'{verification,kind}','') IN (
        'tool_result_observation','tool_result','command','run_command',
        'read_file','file_assert','file_check','command_echo_stdout','distinct_tool_outcomes'
    )
)
ON CONFLICT(plan_id) DO NOTHING;

CREATE OR REPLACE FUNCTION agent_platform.migrate_legacy_plan_criterion_49(criterion JSONB)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    kind TEXT := COALESCE(criterion#>>'{verification,kind}','');
    verification JSONB := COALESCE(criterion->'verification','{}'::jsonb);
    target TEXT := COALESCE(criterion#>>'{verification,target}','');
    original_tool TEXT := COALESCE(criterion#>>'{verification,tool}','');
    migrated_assertions JSONB;
BEGIN
    IF kind IN ('tool_result_observation','tool_result') THEN
        verification := jsonb_set(verification, '{kind}', '"tool_success"'::jsonb, true);
    ELSIF kind IN ('command','run_command') THEN
        verification := jsonb_set(verification, '{kind}', '"command_exit_zero"'::jsonb, true);
        verification := jsonb_set(verification, '{tool}', '"run_command"'::jsonb, true);
    ELSIF kind IN ('read_file','file_assert','file_check') AND target <> '' THEN
        SELECT COALESCE(jsonb_agg(
            jsonb_strip_nulls(jsonb_build_object(
                'path','content',
                'operator',CASE WHEN item->>'operator'='nonempty' THEN 'nonempty' ELSE 'contains' END,
                'value',CASE WHEN item->>'operator'='nonempty' THEN NULL ELSE item->'value' END
            ))
        ), '[]'::jsonb)
        INTO migrated_assertions
        FROM jsonb_array_elements(COALESCE(verification->'assertions','[]'::jsonb)) AS item;
        IF jsonb_array_length(migrated_assertions) = 0 THEN
            verification := jsonb_build_object('kind','tool_success','tool','read_file','arguments',jsonb_build_object('path',target));
        ELSE
            verification := jsonb_build_object('kind','tool_receipt','tool','read_file','arguments',jsonb_build_object('path',target),'assertions',migrated_assertions);
        END IF;
    ELSIF kind = 'command_echo_stdout' THEN
        verification := jsonb_build_object(
            'kind','tool_receipt','tool','run_command',
            'arguments',COALESCE(verification->'arguments','{}'::jsonb),
            'assertions',jsonb_build_array(jsonb_build_object('path','stdout','operator','nonempty'))
        );
    ELSIF kind = 'distinct_tool_outcomes' THEN
        verification := jsonb_build_object(
            'kind','tool_receipt','tool','delegate_agent',
            'assertions',jsonb_build_array(jsonb_build_object('path','output','operator','nonempty'))
        );
    ELSE
        RETURN criterion;
    END IF;
    criterion := jsonb_set(criterion, '{verification}', verification, true);
    criterion := criterion - 'verification_reason' - 'verification_message';
    IF criterion->>'status' IN ('invalid','unsupported') THEN
        criterion := jsonb_set(criterion, '{status}', '"pending"'::jsonb, true);
    END IF;
    RETURN criterion;
END;
$$;

UPDATE agent_platform.agent_task_plans AS plan
SET steps = transformed.steps,
    revision = plan.revision + 1,
    updated_at = now()
FROM (
    SELECT source.plan_id,
           jsonb_agg(
               CASE
                   WHEN step ? 'acceptance_criteria' THEN
                       jsonb_set(step, '{acceptance_criteria}', (
                           SELECT COALESCE(jsonb_agg(
                               CASE
                                   -- Historical completed nodes must not be
                                   -- rewritten into the contradictory state
                                   -- "step completed / criterion pending".
                                   -- Keep their old advisory verdict for audit;
                                   -- only open nodes are made executable again.
                                   WHEN step->>'status' = 'completed'
                                    AND criterion->>'status' IN ('invalid','unsupported')
                                   THEN jsonb_set(
                                       agent_platform.migrate_legacy_plan_criterion_49(criterion),
                                       '{status}', to_jsonb(criterion->>'status'), true
                                   )
                                   ELSE agent_platform.migrate_legacy_plan_criterion_49(criterion)
                               END
                           ), '[]'::jsonb)
                           FROM jsonb_array_elements(COALESCE(step->'acceptance_criteria','[]'::jsonb)) AS criterion
                       ), true)
                   ELSE step
               END
               ORDER BY step_ordinality
           ) AS steps
    FROM agent_platform.agent_task_plans AS source
    CROSS JOIN LATERAL jsonb_array_elements(source.steps) WITH ORDINALITY AS expanded(step, step_ordinality)
    WHERE source.plan_id IN (SELECT plan_id FROM agent_platform.agent_plan_verification_migration_49_backup)
    GROUP BY source.plan_id
) AS transformed
WHERE plan.plan_id = transformed.plan_id;

-- Node execution projections are keyed by the Plan revision they observed.
-- Keep that optimistic-concurrency cursor aligned with the rewritten Plan so
-- a subsequent continuation does not look like it is executing revision N-1.
UPDATE agent_platform.agent_plan_node_states AS state
SET plan_revision = plan.revision,
    updated_at = now()
FROM agent_platform.agent_task_plans AS plan
WHERE state.workflow_id = plan.workflow_id
  AND plan.plan_id IN (SELECT plan_id FROM agent_platform.agent_plan_verification_migration_49_backup)
  AND state.plan_revision <> plan.revision;

DROP FUNCTION agent_platform.migrate_legacy_plan_criterion_49(JSONB);
