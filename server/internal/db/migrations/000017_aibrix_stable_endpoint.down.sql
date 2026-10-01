UPDATE service_instances
SET base_url = 'http://127.0.0.1:8010/v1',
    status = 'unknown',
    last_checked_at = NULL,
    metadata = metadata - 'connectivity',
    updated_at = now()
WHERE kind = 'aibrix' OR name = 'aibrix-gateway';
