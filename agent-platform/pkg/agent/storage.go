package agent

// StorageBackend describes one storage tier and its effective runtime state.
type StorageBackend struct {
	Name    string           `json:"name"`
	Role    string           `json:"role"`
	Status  string           `json:"status"`
	Metrics map[string]int64 `json:"metrics,omitempty"`
	Detail  string           `json:"detail,omitempty"`
}

// StorageSummary exposes actual Agent storage wiring instead of inferring it
// from deployment configuration.
type StorageSummary struct {
	PostgreSQL StorageBackend `json:"postgresql"`
	Redis      StorageBackend `json:"redis"`
	MinIO      StorageBackend `json:"minio"`
	PGVector   StorageBackend `json:"pgvector"`
}
