-- Down migration is intentionally guarded: existing native-tool approvals
-- cannot be represented by the old non-null contract.
DELETE FROM agent_platform.agent_tool_approvals WHERE tool_version_id IS NULL;
ALTER TABLE agent_platform.agent_tool_approvals
    ALTER COLUMN tool_version_id SET NOT NULL;
