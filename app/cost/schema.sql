-- Project spend ledger (spec §4.2, §9.3). Every statement is idempotent.

CREATE TABLE IF NOT EXISTS cost_entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT,
  attempt INTEGER,
  ts REAL NOT NULL,
  settled_at REAL,
  stage TEXT,
  operation TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT,
  status TEXT NOT NULL,
  billable INTEGER NOT NULL,
  projected_usd REAL NOT NULL,
  usd REAL,
  cost_source TEXT,
  units_json TEXT NOT NULL DEFAULT '{}',
  error_code TEXT,
  error_message TEXT,
  pricing_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS cost_entries_ts ON cost_entries(ts);
CREATE INDEX IF NOT EXISTS cost_entries_job ON cost_entries(job_id);

CREATE TABLE IF NOT EXISTS budget_state (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  ts REAL NOT NULL
);
