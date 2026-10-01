CREATE TABLE platform_logs (
    id            BIGSERIAL PRIMARY KEY,
    timestamp     TIMESTAMPTZ NOT NULL DEFAULT now(),
    level         VARCHAR(16) NOT NULL DEFAULT 'info',
    source        VARCHAR(128) NOT NULL,
    resource_type VARCHAR(64),
    resource_id   VARCHAR(256),
    message       TEXT NOT NULL,
    attributes    JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX idx_platform_logs_time ON platform_logs (timestamp DESC);
CREATE INDEX idx_platform_logs_resource ON platform_logs (resource_type, resource_id, timestamp DESC);
CREATE INDEX idx_platform_logs_source ON platform_logs (source, timestamp DESC);
