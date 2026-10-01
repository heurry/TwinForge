-- 服务注册表（5B.2）：注册/心跳/注销 + TTL 清扫。

-- name: RegisterServiceInstance :one
-- 注册或重注册（按 name upsert）：刷新元数据、置 healthy、打心跳戳。
INSERT INTO service_instances (
    name, base_url, model_id, kind, gpu_id, routing_role, metadata,
    status, last_heartbeat_at, last_checked_at
) VALUES (
    sqlc.arg(name), sqlc.arg(base_url), sqlc.narg(model_id), sqlc.arg(kind),
    sqlc.narg(gpu_id), sqlc.narg(routing_role), sqlc.arg(metadata),
    'healthy', now(), now()
)
ON CONFLICT (name) DO UPDATE SET
    base_url        = EXCLUDED.base_url,
    model_id        = EXCLUDED.model_id,
    kind            = EXCLUDED.kind,
    gpu_id          = EXCLUDED.gpu_id,
    routing_role    = EXCLUDED.routing_role,
    metadata        = EXCLUDED.metadata,
    status          = 'healthy',
    last_heartbeat_at = now(),
    last_checked_at = now(),
    updated_at      = now()
RETURNING name, status;

-- name: HeartbeatServiceInstance :execrows
-- 心跳：刷新戳并（若曾被判 unreachable）复活为 healthy。
UPDATE service_instances
   SET last_heartbeat_at = now(), last_checked_at = now(), status = 'healthy', updated_at = now()
 WHERE name = sqlc.arg(name);

-- name: DeregisterServiceInstance :execrows
DELETE FROM service_instances WHERE name = sqlc.arg(name);

-- name: SweepStaleServiceInstances :many
-- TTL 清扫：超时未心跳者置 unreachable；NULL 心跳（静态种子）不动。
UPDATE service_instances
   SET status = 'unreachable', updated_at = now()
WHERE last_heartbeat_at IS NOT NULL
  -- External gateways and logical routers are checked through their target;
  -- they do not emit process heartbeats to the control plane.
  AND lower(kind) NOT IN ('aibrix', 'auto_router', 'client_round_robin')
  AND last_heartbeat_at < now() - make_interval(secs => sqlc.arg(ttl_seconds)::double precision)
   AND status <> 'unreachable'
RETURNING name;
