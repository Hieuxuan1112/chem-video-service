-- Job store (spec §4.2). Every statement is idempotent.

CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  query TEXT NOT NULL,
  topic_id TEXT NOT NULL,
  status TEXT NOT NULL,
  stage TEXT,
  attempt INTEGER NOT NULL DEFAULT 0,
  lease_claims INTEGER NOT NULL DEFAULT 0,
  scheduled_retries INTEGER NOT NULL DEFAULT 0,
  worker_id TEXT,
  lease_until REAL,
  not_before REAL,
  generation_source TEXT,
  fallback_reason TEXT,
  error_code TEXT,
  error_cause TEXT,
  error_message TEXT,
  retryable INTEGER,
  idempotency_key TEXT UNIQUE,
  request_fingerprint TEXT,
  pipeline_version TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  started_at REAL,
  finished_at REAL
);
CREATE INDEX IF NOT EXISTS jobs_claim ON jobs(status, not_before, created_at);

CREATE TABLE IF NOT EXISTS job_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  attempt INTEGER NOT NULL,
  ts REAL NOT NULL,
  type TEXT NOT NULL,
  stage TEXT,
  data_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS job_events_job ON job_events(job_id, id);

CREATE TABLE IF NOT EXISTS stage_checkpoints (
  job_id TEXT NOT NULL,
  stage_key TEXT NOT NULL,
  attempt INTEGER NOT NULL,
  input_hash TEXT NOT NULL,
  manifest_sha256 TEXT NOT NULL,
  versions_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY (job_id, stage_key)
);

-- Backs the /healthz field worker_heartbeat_age_s.
CREATE TABLE IF NOT EXISTS worker_heartbeats (
  worker_id TEXT PRIMARY KEY,
  last_seen REAL NOT NULL
);
