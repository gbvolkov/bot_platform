"""Additive, versioned SQLite migrations. Never modify graph checkpoints."""
MIGRATIONS = [(1, """
CREATE TABLE IF NOT EXISTS conversations (
 id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, user_id TEXT NOT NULL,
 user_role TEXT NOT NULL, status TEXT NOT NULL, title TEXT, metadata JSON NOT NULL,
 created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
 updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
 last_message_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
 id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id),
 role TEXT NOT NULL, content JSON NOT NULL, raw_text TEXT NOT NULL, metadata JSON NOT NULL,
 created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL);
CREATE TABLE runtime_assignments (
 conversation_id TEXT PRIMARY KEY REFERENCES conversations(id), runtime TEXT NOT NULL,
 revision TEXT NOT NULL, execution_class TEXT NOT NULL, privacy_affinity INTEGER NOT NULL DEFAULT 0,
 generation TEXT, checkpoint_reference TEXT, next_sequence INTEGER NOT NULL DEFAULT 1, blocked_run_id TEXT,
 resume_run_id TEXT, closed INTEGER NOT NULL DEFAULT 0);
INSERT INTO runtime_assignments(conversation_id,runtime,revision,execution_class)
 SELECT id,'legacy','legacy','legacy' FROM conversations;
CREATE TABLE runs (
 id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id),
 sequence INTEGER NOT NULL, revision TEXT NOT NULL, operation TEXT NOT NULL,
 input JSON NOT NULL, input_hash TEXT NOT NULL, idempotency_key TEXT,
 status TEXT NOT NULL, resume_of TEXT UNIQUE, user_message_id TEXT,
 attempt_id TEXT, generation TEXT, lease_until REAL, cancel_requested INTEGER NOT NULL DEFAULT 0,
 event_sequence INTEGER NOT NULL DEFAULT 0, result JSON, error TEXT, completion_hash TEXT,
 created_at REAL NOT NULL, started_at REAL, finished_at REAL,
 UNIQUE(conversation_id,sequence), UNIQUE(conversation_id,idempotency_key));
CREATE INDEX runs_queue ON runs(status,created_at);
CREATE TABLE run_attempts (
 id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id), generation TEXT NOT NULL,
 claimed_at REAL NOT NULL, started_at REAL, finished_at REAL, outcome TEXT);
CREATE TABLE run_events (
 run_id TEXT NOT NULL REFERENCES runs(id), sequence INTEGER NOT NULL,
 attempt_id TEXT, source_sequence INTEGER, type TEXT NOT NULL, payload JSON NOT NULL,
 created_at REAL NOT NULL, PRIMARY KEY(run_id,sequence), UNIQUE(attempt_id,source_sequence));
CREATE TABLE workers (
 generation TEXT PRIMARY KEY, execution_class TEXT NOT NULL, heartbeat REAL NOT NULL,
 ready JSON NOT NULL, errors JSON NOT NULL);
CREATE TABLE artifacts (
 id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id),
 owner_id TEXT NOT NULL, media_type TEXT NOT NULL, filename TEXT NOT NULL,
 storage_key TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE platform_metrics (name TEXT PRIMARY KEY, value REAL NOT NULL);
""")]
