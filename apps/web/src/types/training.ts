// Phase F：分布式训练（Kubeflow PyTorchJob）类型 —— 对齐后端 training_handlers.go DTO。

export type TrainingReplicaStatus = { active?: number; succeeded?: number; failed?: number };

export type TrainingJobMeta = {
  phase?: string;
  message?: string;
  reason?: string;
  replica_statuses?: Record<string, TrainingReplicaStatus>;
  start_time?: string;
  completion_time?: string;
  events?: Array<{ at: string; phase: string; message: string }>;
  error?: string;
  [key: string]: unknown;
};

export type TrainingJob = {
  id: string;
  name: string;
  framework: string;
  namespace: string;
  base_model: string;
  dataset_uri: string | null;
  image: string;
  workers: number;
  gpus_per_worker: number;
  hyperparams: Record<string, unknown>;
  status: string; // pending|running|succeeded|failed|cancelled
  k8s_job_ref: string | null;
  output_artifact_uri: string | null;
  model_version_id: string | null;
  metadata: TrainingJobMeta;
  created_by: string | null;
  created_at: string;
  updated_at: string;
};

export type TrainingJobsList = { jobs: TrainingJob[] };

export type SubmitTrainingInput = {
  name: string;
  base_model: string;
  image: string;
  namespace?: string;
  dataset_uri?: string;
  workers?: number;
  gpus_per_worker?: number;
  hyperparams?: Record<string, unknown>;
  model_id?: string;
  version?: string;
  base_version?: string;
};

export type TrainingLogs = { pod: string | null; logs: string; note?: string };

export type TrainingKubernetesDetail = {
  resource: {
    api_version: string;
    kind: string;
    namespace: string;
    name: string;
    ref: string;
    available: boolean;
    phase?: string;
    reason?: string;
    message?: string;
    error?: string;
    replica_statuses?: Record<string, TrainingReplicaStatus>;
    start_time?: string;
    completion_time?: string;
  };
  cluster: { available: boolean; error?: string };
  pods: Array<{
    namespace: string;
    name: string;
    phase: string;
    ready: string;
    restarts: number;
    pod_ip: string;
    node: string;
    component: string;
  }>;
  events: Array<{
    namespace: string;
    resource_kind: string;
    resource_name: string;
    reason: string;
    message: string;
    type: string;
    event_time: string;
  }>;
};
