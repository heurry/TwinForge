export type AgentRunStatus = "queued" | "running" | "waiting_tool" | "waiting_approval" | "waiting_input" | "waiting_external" | "suspended" | "completed" | "failed" | "cancelled";
export type AgentPlanningPolicy = "auto" | "required" | "disabled";

export interface AgentRun {
  id: string;
  workflow_id?: string;
  agent_version_id: string;
  session_id?: string;
  status: AgentRunStatus;
  input: unknown;
  output?: unknown;
  binding_snapshot?: { spec?: { planning?: { policy?: AgentPlanningPolicy } } };
  model_resolution?: AgentModelResolution;
  current_turn: number;
  current_step: number;
  error_message?: string;
  created_at: string;
  started_at?: string;
  finished_at?: string;
  cancel_requested_at?: string;
  parent_run_id?: string;
  root_run_id?: string;
  delegation_id?: string;
  delegation_depth?: number;
}

export interface AgentRunManifest {
  run_id: string;
  workflow_id: string;
  tenant_id: string;
  status: AgentRunStatus | string;
  canonical_artifacts: Array<{ artifact_id: string; name: string; kind: string; media_type: string; content_hash: string; size_bytes: number; metadata?: unknown; canonical: boolean; created_at: string }>;
  required_outputs?: string[];
  verification_summary?: unknown;
  child_runs?: string[];
  final_artifact_ids?: string[];
  final_output_hash?: string;
  final_output_present: boolean;
  updated_at: string;
}

export interface AgentWorkflow {
  workflow_id: string;
  session_id?: string;
  status: string;
  goal?: string;
  workspace_id?: string;
  active_run_id?: string;
  active_plan_id?: string;
  latest_checkpoint_id?: string;
  latest_state_sequence?: number;
  created_at: string;
  updated_at: string;
}

export interface AgentModelResolution {
  selection_policy: "pinned" | "auto";
  provider: string;
  service_ref: string;
  model_id: string;
  model_version?: string;
  artifact_digest?: string;
  discovered_at: string;
}

export interface AgentModelBinding {
  provider?: string;
  service_ref?: string;
  service_candidates?: string[];
  capability?: string;
  selection_policy?: "pinned" | "auto";
  model_id?: string;
  model_candidates?: string[];
  model_version?: string;
  artifact_digest?: string;
}

export interface AgentExecutableVersion {
  id: string;
  agent_id: string;
  agent_key: string;
  agent_name: string;
  version: number;
  spec: {
    description?: string;
    planning?: { policy?: AgentPlanningPolicy };
    model?: AgentModelBinding;
    collaboration?: {
      allowed_targets?: Array<{ agent_id: string; agent_version_id: string; modes: Array<"sync" | "async"> }>;
      max_depth?: number;
      max_fan_out?: number;
      max_child_runs?: number;
    };
    skillset_ref?: { id: string; version: string };
    memory?: { enabled: boolean; read_scopes?: string[]; write_scope?: string; read_layers?: string[]; write_layer?: string; read_types?: string[]; auto_extract?: boolean; team_memory_enabled?: boolean; max_recall?: number; candidate_limit?: number; router_enabled?: boolean; router_top_k?: number; minimum_score?: number; manifest_tokens?: number; repeat_suppression_turns?: number };
    context?: { memory_tokens?: number };
  };
  published_at?: string;
}

export interface AgentDefinition {
  id: string;
  tenant_id: string;
  key: string;
  name: string;
  description?: string;
  owner?: string;
  status: string;
  active_version_id?: string;
  created_at: string;
  updated_at: string;
}

export interface AgentVersion {
  id: string;
  agent_id: string;
  version: number;
  spec: Record<string, unknown>;
  spec_hash: string;
  status: string;
  created_by?: string;
  created_at: string;
  published_at?: string;
}

export interface AgentVersionRef { id: string; version: string }

export interface AgentPromptVersion {
  id: string;
  definition_id: string;
  tenant_id: string;
  key: string;
  name: string;
  version: number;
  content: string;
  content_hash: string;
  status: string;
  created_by?: string;
  created_at: string;
}

export interface AgentSkillSpec {
  description?: string;
  instructions: Array<{ name: string; content: string; priority?: number }>;
  examples?: Array<{ input: string; output: string }>;
  required_tools?: AgentVersionRef[];
}

export interface AgentSkillVersion {
  id: string;
  definition_id: string;
  tenant_id: string;
  key: string;
  name: string;
  version: number;
  spec: AgentSkillSpec;
  spec_hash: string;
  status: string;
  created_by?: string;
  created_at: string;
}

export interface AgentSkillSetVersion {
  id: string;
  definition_id: string;
  tenant_id: string;
  key: string;
  name: string;
  version: number;
  spec: { skills: AgentVersionRef[] };
  spec_hash: string;
  status: string;
  created_by?: string;
  created_at: string;
}

export interface AgentVersionSpecInput {
  name: string;
  description?: string;
  identity?: {
    display_name?: string;
    role?: string;
    goal?: string;
    responsibilities?: string[];
    boundaries?: string[];
    communication_style?: string;
  };
  harness: { name: string; max_turns: number; max_steps: number };
  planning?: { policy: AgentPlanningPolicy };
  model: AgentModelBinding;
  prompt_ref: AgentVersionRef;
  toolset_ref: AgentVersionRef;
  skillset_ref?: AgentVersionRef;
  input_schema: Record<string, unknown>;
  output_schema: Record<string, unknown>;
  context: {
    max_input_tokens?: number;
    reserve_output_tokens: number;
    recent_turn_tokens: number;
    memory_tokens: number;
    knowledge_tokens: number;
    tool_result_tokens: number;
    compaction: string;
    collapse_trigger_ratio?: number;
  };
  memory: { enabled: boolean; read_scopes?: string[]; write_scope?: string; read_layers?: string[]; write_layer?: string; read_types?: string[]; auto_extract?: boolean; team_memory_enabled?: boolean; default_ttl?: string; max_recall?: number; candidate_limit?: number; router_enabled?: boolean; router_top_k?: number; minimum_score?: number; manifest_tokens?: number; repeat_suppression_turns?: number };
  runtime: { run_timeout: string; model_timeout: string; tool_timeout: string; max_model_calls: number; max_tool_calls: number };
  approval: { require_for?: string[]; auto_approve_sandbox_command?: boolean; expires_in?: string };
  collaboration?: {
    allowed_targets?: Array<{ agent_id: string; agent_version_id: string; modes: Array<"sync" | "async"> }>;
    max_depth?: number;
    max_fan_out?: number;
    max_child_runs?: number;
    share_session_memory?: boolean;
    propagate_user_identity?: boolean;
    child_timeout?: number;
    budget?: { max_model_calls?: number; max_tool_calls?: number; max_tokens?: number };
  };
  metadata?: Record<string, string>;
}

export interface AgentToolVersion {
  id: string;
  definition_id: string;
  tenant_id: string;
  key: string;
  name: string;
  version: number;
  spec: {
    definition?: { name?: string; description?: string; version?: string; risk?: string; execution_mode?: string; input_schema?: Record<string, unknown>; output_schema?: Record<string, unknown> };
    provider_type?: string;
    http?: { endpoint?: string; method?: string; timeout?: number; max_response_bytes?: number };
    workspace?: { operation?: string; max_bytes?: number };
    mcp?: { server_version_id: string; tool_name: string; schema_hash: string };
  };
  spec_hash: string;
  status: string;
  created_at: string;
}

export interface AgentToolSetVersion {
  id: string;
  definition_id: string;
  tenant_id: string;
  key: string;
  name: string;
  version: number;
  spec: { tools: AgentVersionRef[] };
  spec_hash: string;
  status: string;
  created_at: string;
}

export interface AgentMemory {
  id: string;
  tenant_id: string;
  scope: "tenant" | "agent" | "user" | "session";
  agent_id?: string;
  user_id?: string;
  session_id?: string;
  kind: "semantic" | "episodic" | "preference";
  content: string;
  importance: number;
  source_run_id?: string;
  metadata: Record<string, unknown>;
  expires_at?: string;
  created_at: string;
  updated_at: string;
  recall_score?: number;
	embedding_status: "pending" | "ready" | "failed";
	embedding_model?: string;
	embedding_error?: string;
	embedded_at?: string;
	source_id?: string;
	source_layer: "managed" | "user" | "project" | "local" | "auto" | "team";
	semantic_type: "user" | "feedback" | "project" | "reference";
	project_key?: string;
	team_id?: string;
	title: string;
	description: string;
	body: string;
	structured_data: Record<string, unknown>;
	status: "active" | "review" | "superseded" | "expired" | "deleted";
	confidence: number;
	freshness_class: "stable" | "normal" | "volatile";
	valid_from?: string;
	valid_until?: string;
	last_verified_at?: string;
	verification_hint?: string;
	canonical_key: string;
	content_hash: string;
	pinned: boolean;
	supersedes_id?: string;
}

export interface AgentMemoryManifestEntry {
	id: string;
	title: string;
	description: string;
	source_layer: "managed" | "user" | "project" | "local" | "auto" | "team";
	semantic_type: "user" | "feedback" | "project" | "reference";
	project_key?: string;
	updated_at: string;
	last_verified_at?: string;
	freshness_class: "stable" | "normal" | "volatile";
	confidence: number;
	importance: number;
	pinned: boolean;
}

export interface AgentMemoryRevision {
  id: string;
  memory_id: string;
  revision: number;
  title: string;
  description: string;
  body: string;
  structured_data: Record<string, unknown>;
  source_message_ids?: string[];
  source_run_id?: string;
  source_event_from?: number;
  source_event_to?: number;
  reason: string;
  created_by_type: string;
  created_by?: string;
  created_at: string;
}

export interface AgentMemoryLifecycleEvent {
  id: string;
  tenant_id: string;
  memory_id?: string;
  source_id?: string;
  run_id?: string;
  event_type: string;
  actor?: string;
  idempotency_key?: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface AgentMemoryTimelineEntry {
  id: string;
  source: "run_event" | "memory_lifecycle";
  sequence?: number;
  run_id: string;
  memory_id?: string;
  event_type: string;
  actor?: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface AgentMemoryRetrievalRecord {
  id: string;
  tenant_id: string;
  run_id: string;
  turn_id: string;
  query_hash: string;
  candidate_ids?: string[];
  routed_ids?: string[];
  injected_ids?: string[];
  suppressed_ids?: string[];
  scores?: Record<string, number>;
  suppression_reasons?: Record<string, string>;
  router_model?: string;
  manifest_tokens: number;
  body_tokens: number;
  latency_ms: number;
  created_at: string;
}

export interface AgentMemorySource {
  id: string;
  tenant_id: string;
  source_layer: AgentMemory["source_layer"];
  agent_id?: string;
  user_id?: string;
  project_key?: string;
  team_id?: string;
  uri: string;
  display_name: string;
  content_hash: string;
  revision: number;
  authority: number;
  writable_by: string;
  enabled: boolean;
  git_commit?: string;
  observed_at: string;
  created_at: string;
  updated_at: string;
}

export interface AgentSession {
  id: string;
  tenant_id: string;
  agent_id: string;
  user_id?: string;
  status: string;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  run_count?: number;
  message_count?: number;
  active_run_count?: number;
  last_activity_at?: string;
}

export interface AgentRunAudit {
  run_id: string;
  workflow_id: string;
  status: AgentRun["status"];
  started_at?: string;
  finished_at?: string;
  duration_ms: number;
  model_calls: number;
  successful_model_calls: number;
  failed_model_calls: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  tool_calls: number;
  successful_tool_calls: number;
  failed_tool_calls: number;
  terminal_tool_calls: number;
  tool_success_rate_percent: number;
}

export interface AgentSessionAudit {
  session_id: string;
  run_count: number;
  completed_runs: number;
  failed_runs: number;
  active_runs: number;
  first_started_at?: string;
  last_activity_at?: string;
  wall_duration_ms: number;
  execution_duration_ms: number;
  model_calls: number;
  successful_model_calls: number;
  failed_model_calls: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  tool_calls: number;
  successful_tool_calls: number;
  failed_tool_calls: number;
  terminal_tool_calls: number;
  tool_success_rate_percent: number;
  runs: AgentRunAudit[];
}

export interface AgentStorageBackend {
  name: string;
  role: string;
  status: "ready" | "degraded" | "not_configured" | "not_connected" | string;
  metrics?: Record<string, number>;
  detail?: string;
}

export interface AgentStorageSummary {
  postgresql: AgentStorageBackend;
  redis: AgentStorageBackend;
  minio: AgentStorageBackend;
  pgvector: AgentStorageBackend;
}

export type AgentCapabilityStatus = "enforced" | "available" | "declared" | "missing";

export interface AgentPlatformCapabilities {
  framework_version: string;
  harnesses: Array<{
    name: string;
    display_name: string;
    supports_checkpoint: boolean;
    supports_tools: boolean;
    supports_multi_turn: boolean;
  }>;
  features: Array<{
    key: string;
    name: string;
    status: AgentCapabilityStatus;
    frontend: boolean;
    description: string;
  }>;
}

export interface AgentMessagePart { type: string; text?: string; json?: unknown; uri?: string; media_type?: string }
export interface AgentMessage {
  role: string;
  content?: string;
  parts?: AgentMessagePart[];
  name?: string;
  tool_call_id?: string;
  tool_calls?: Array<{ id: string; name: string; arguments: unknown }>;
}

export interface AgentEvent {
  run_id: string;
  workflow_id?: string;
  type: string;
  turn?: number;
  step?: number;
  plan_node_id?: string;
  decision_cycle?: number;
  action_id?: string;
  call_id?: string;
  payload?: Record<string, any>;
  sequence: number;
  created_at: string;
}

export type AgentTaskPlanStepStatus = "pending" | "in_progress" | "completed" | "blocked" | "skipped";

export interface AgentPlanGraphState {
  status?: "idle" | "ready" | "running" | "waiting" | "retry_required" | "completed" | string;
  active_node_ids?: string[];
  ready_node_ids?: string[];
  blocked_node_ids?: string[];
  retry_node_id?: string;
  next_node_id?: string;
  next_action?: "run" | "retry" | "wait" | "complete" | "none" | string;
}

export interface AgentTaskPlanStep {
  id: string;
  description: string;
  status: AgentTaskPlanStepStatus;
  assignee?: string;
  agent_version_id?: string;
  depends_on?: string[];
  result?: string;
  state?: {
    status?: AgentTaskPlanStepStatus;
    attempts?: number;
    last_run_id?: string;
    last_event_sequence?: number;
    output?: string;
    artifact_ids?: string[];
    tests?: Array<{ name: string; kind?: string; status: string; exit_code?: number; message?: string; tool_call_id?: string; event_sequence?: number }>;
    usage?: { input_tokens?: number; output_tokens?: number; total_tokens?: number; cost_usd?: number; duration_ms?: number };
    blocked_reason?: string;
    retry_from_node_id?: string;
    next_node_ids?: string[];
  };
  acceptance_criteria?: Array<{
    id: string;
    description: string;
    status: "pending" | "passed" | "failed" | "skipped" | "invalid" | "unsupported" | "stale";
    enforcement?: "informational" | "advisory" | "required" | "release_gate";
    origin?: "user_explicit" | "deployment_policy" | "agent_inferred" | "provider_required";
    verification_reason?: string;
    verification_message?: string;
    verification?: { kind?: string; target?: string; match?: string };
    evidence?: string;
    evidence_call_ids?: string[];
  }>;
}

export interface AgentTaskPlan {
  plan_id: string;
  run_id: string;
  workflow_id?: string;
  tenant_id: string;
  revision: number;
  goal: string;
  explanation?: string;
  graph_state?: AgentPlanGraphState;
  execution_outcome?: "running" | "completed" | "blocked";
  verification_outcome?: "verified" | "partially_verified" | "unverified" | "rejected";
  steps: AgentTaskPlanStep[];
  created_at: string;
  updated_at: string;
}

export interface AgentVerificationRecord {
  intent: {
    id: string;
    run_id: string;
    plan_revision: number;
    plan_step_key: string;
    criterion_key: string;
    description: string;
    kind?: string;
    enforcement: string;
    origin: string;
    parameters: Record<string, unknown>;
    status: string;
    diagnostic_code?: string;
    diagnostic_message?: string;
    created_at: string;
    updated_at: string;
  };
  spec?: {
    id: string;
    intent_id: string;
    provider_key: string;
    provider_version: string;
    subject: Record<string, unknown>;
    execution: Record<string, unknown>;
    assertions: Array<Record<string, unknown>>;
    spec_digest: string;
    status: string;
    created_at: string;
  };
  attempts: Array<{
    id: string;
    spec_id: string;
    tool_execution_id?: string;
    status: string;
    reason_code?: string;
    diagnostic?: string;
    result: Record<string, unknown>;
    started_at: string;
    finished_at?: string;
  }>;
  evidence: Array<{
    id: string;
    intent_id: string;
    spec_id: string;
    attempt_id: string;
    tool_execution_id?: string;
    verdict: string;
    evidence: Record<string, unknown>;
    result_digest?: string;
    workspace_revision?: string;
    created_at: string;
  }>;
}

export interface AgentUserQuestion {
  id: string;
  run_id: string;
  call_id: string;
  turn: number;
  step: number;
  question: string;
  options?: string[];
  context?: string;
  answer?: string;
  status: "pending" | "answered" | "cancelled";
  created_at: string;
  answered_by?: string;
  answered_at?: string;
}

export interface AgentTrajectoryRecord {
  id: string;
  trace_id?: string;
  workflow_id?: string;
  sequence: number;
  last_sequence: number;
  kind: string;
  parent_id?: string;
  root_id?: string;
  turn?: number;
  step?: number;
  plan_node_id?: string;
  decision_cycle?: number;
  action_id?: string;
  call_id?: string;
  delegation_id?: string;
  child_run_id?: string;
  agent_version_id?: string;
  model_resolution_id?: string;
  model_id?: string;
  tool_version_id?: string;
  prompt_version_id?: string;
  skillset_version_id?: string;
  toolset_version_id?: string;
  status: "running" | "completed" | "failed" | "cancelled" | "waiting";
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  input_tokens?: number;
  output_tokens?: number;
  total_cost?: number;
  summary: string;
  metrics?: Record<string, unknown>;
  detail_refs?: Record<string, unknown>;
  event_sequences: number[];
}

export interface AgentTrajectoryPage {
  records: AgentTrajectoryRecord[];
  next_cursor?: number;
  has_more: boolean;
  total: number;
}

export interface AgentScore {
  id: string;
  tenant_id: string;
  run_id?: string;
  observation_id?: string;
  session_id?: string;
  dataset_run_id?: string;
  name: string;
  score_type: "numeric" | "categorical" | "boolean" | "text";
  value?: number;
  string_value?: string;
  source: string;
  evaluator_version?: string;
  agent_version_id?: string;
  model_resolution_id?: string;
  prompt_version_id?: string;
  toolset_version_id?: string;
  skillset_version_id?: string;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export interface AgentArtifact {
  id: string;
  run_id: string;
  workflow_id: string;
  call_id?: string;
  kind: string;
  name: string;
  media_type: string;
  content_hash: string;
  size_bytes: number;
	storage_backend: "inline" | "minio";
	storage_status: "ready" | "migration_pending" | "failed";
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface AgentArtifactPromotion {
  id: string;
  run_id: string;
  artifact_id: string;
  target_path: string;
  source_hash: string;
  expected_target_sha256?: string;
  previous_target_sha256?: string;
  result_target_sha256?: string;
  status: "requested" | "promoted" | "failed";
  requested_by: string;
  error?: string;
  created_at: string;
  finished_at?: string;
}

export interface AgentApproval {
  id: string;
  run_id: string;
  call_id: string;
  turn: number;
  step: number;
  tool_version_id?: string;
  tool_name: string;
  risk: string;
  request_hash: string;
  request: Record<string, unknown>;
  diff_artifact_id?: string;
  status: "pending" | "approved" | "rejected" | "expired";
  decided_by?: string;
  decision_reason?: string;
  expires_at?: string;
  created_at: string;
  decided_at?: string;
}

export interface AgentEnvironmentTemplate {
  id: string;
  key: string;
  version: number;
  name: string;
  runtime: string;
  image_ref: string;
  spec_digest: string;
  dependencies: Array<{ ecosystem: string; name: string; version: string }>;
  capabilities: Record<string, unknown>;
  status: "active" | "deprecated" | "retired";
  created_at: string;
}

export interface AgentDependencyInstall {
  id: string;
  run_id: string;
  call_id: string;
  environment_template: string;
  ecosystem: string;
  packages: Array<{ name: string; version: string }>;
  source: string;
  scope: "run";
  status: "installed" | "failed";
  result?: Record<string, unknown>;
  error?: string;
  created_at: string;
  finished_at?: string;
}

export interface MCPToolSnapshot {
  id?: string;
  server_version_id: string;
  name: string;
  description?: string;
  input_schema: Record<string, unknown>;
  schema_hash: string;
  risk: string;
  enabled: boolean;
  synced_at: string;
}

export interface MCPServerVersion {
  id: string;
  definition_id: string;
  key: string;
  name: string;
  version: number;
  spec: {
    transport: "streamable_http";
    endpoint: string;
    protocol_version: string;
    header_environment?: Record<string, string>;
    timeout: number;
    max_response_bytes?: number;
  };
  spec_hash: string;
  status: string;
  created_at: string;
  health?: { status: string; protocol_version?: string; latency_ms?: number; error?: string; checked_at: string };
  tools?: MCPToolSnapshot[];
}

export interface A2APart {
  text?: string;
  data?: unknown;
  metadata?: Record<string, unknown>;
}

export interface A2AMessage {
  messageId: string;
  contextId?: string;
  taskId?: string;
  role: "user" | "agent" | string;
  parts: A2APart[];
  metadata?: Record<string, unknown>;
}

export interface A2ATaskStatus {
  state: string;
  message?: A2AMessage;
  timestamp: string;
}

export interface A2AArtifact {
  artifactId: string;
  name?: string;
  description?: string;
  parts: A2APart[];
  metadata?: Record<string, unknown>;
}

export interface A2ATask {
  id: string;
  contextId: string;
  status: A2ATaskStatus;
  history?: A2AMessage[];
  artifacts?: A2AArtifact[];
  metadata?: Record<string, unknown>;
}

export interface A2AStreamEvent {
  event: string;
  data: Record<string, unknown>;
}

export interface AgentObservabilitySummary {
  total_runs: number;
  completed_runs: number;
  failed_runs: number;
  active_runs: number;
  model_calls: number;
  tool_calls: number;
  failed_tool_calls: number;
  input_tokens: number;
  output_tokens: number;
  average_run_latency_ms: number;
  average_model_latency_ms: number;
  tool_failures_by_code?: Record<string, number>;
  model_protocol_failures?: number;
  verification_failures?: number;
  plan_failures?: number;
  compaction_count?: number;
  compaction_before_tokens?: number;
  compaction_after_tokens?: number;
}
