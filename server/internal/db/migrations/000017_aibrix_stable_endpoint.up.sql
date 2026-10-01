-- AIBrix 的正式应用入口使用 minikube 固定 NodePort。历史 8010 是宿主机
-- kubectl port-forward，只能用于调试，进程退出后会制造陈旧 healthy 状态。
UPDATE service_instances
SET base_url = 'http://minikube:30080/v1',
    model_id = CASE WHEN model_id = 'qwen3-4b-platform' THEN 'qwen3-4b-customer' ELSE model_id END,
    status = 'unknown',
    last_checked_at = NULL,
    metadata = (metadata - 'healthcheck') ||
      '{"connectivity":{"exposure":"nodeport","service":"twinforge-aibrix-gateway","node_port":30080}}'::jsonb,
    updated_at = now()
WHERE kind = 'aibrix' OR name = 'aibrix-gateway';
