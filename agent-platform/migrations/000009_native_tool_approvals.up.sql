-- Platform-native tools (for example delegate_agent) have no tenant-owned
-- ToolVersion row, but must still participate in the same durable HIGH_RISK
-- approval lifecycle.
ALTER TABLE agent_platform.agent_tool_approvals
    ALTER COLUMN tool_version_id DROP NOT NULL;
