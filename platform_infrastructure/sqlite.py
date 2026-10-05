from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from platform_application.service import Conflict, NotFound
from platform_contracts import TERMINAL
from .migrations import MIGRATIONS


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def new_id():
    return str(uuid.uuid4())


class SQLiteRepository:
    def __init__(self, path: str, *, lease_seconds=30, clock=time.time):
        self.path = str(path)
        self.lease_seconds = lease_seconds
        self.clock = clock

    @contextmanager
    def connection(self, *, write=False):
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        except BaseException:
            if write:
                db.rollback()
            raise
        finally:
            db.close()

    def migrate(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS platform_migrations(version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)")
        for version, script in MIGRATIONS:
            with self.connection(write=True) as db:
                if db.execute("SELECT 1 FROM platform_migrations WHERE version=?", (version,)).fetchone():
                    continue
                # executescript implicitly commits: execute individual statements
                # to keep the version marker and DDL atomic together.
                for statement in script.split(";"):
                    if statement.strip():
                        db.execute(statement)
                db.execute("INSERT INTO platform_migrations VALUES (?,?)", (version, self.clock()))

    @staticmethod
    def _conversation(db, conversation_id, user_id=None):
        row = db.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        if row is None or (user_id is not None and row["user_id"] != user_id):
            raise NotFound("Conversation not found")
        value = dict(row)
        value["metadata"] = json.loads(value["metadata"] or "{}")
        for key in ("created_at", "updated_at", "last_message_at"):
            if isinstance(value.get(key), str):
                value[key] = value[key].replace(" ", "T", 1)
        return value

    @staticmethod
    def _run(db, run_id, user_id=None):
        row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise NotFound("Run not found")
        SQLiteRepository._conversation(db, row["conversation_id"], user_id)
        value = dict(row)
        for key in ("input", "result"):
            value[key] = json.loads(value[key]) if value[key] is not None else None
        return value

    @staticmethod
    def _message(db, message_id):
        row = db.execute("SELECT id,role,content,raw_text,metadata,created_at FROM messages WHERE id=?", (message_id,)).fetchone()
        value = dict(row)
        for key in ("content", "metadata"):
            value[key] = json.loads(value[key])
        value["created_at"] = value["created_at"].replace(" ", "T", 1)
        return value

    def conversation(self, conversation_id, user_id, detail=False):
        with self.connection() as db:
            value = self._conversation(db, conversation_id, user_id)
            if detail:
                value["messages"] = [self._message(db, r[0]) for r in db.execute(
                    "SELECT id FROM messages WHERE conversation_id=? ORDER BY created_at,rowid", (conversation_id,))]
            return value

    def conversations(self, user_id):
        with self.connection() as db:
            return [self._conversation(db, r[0]) for r in db.execute(
                "SELECT id FROM conversations WHERE user_id=? ORDER BY last_message_at DESC", (user_id,))]

    def create_conversation(self, descriptor, user_id, user_role, title=None, metadata=None, runtime="worker", conversation_id=None):
        cid = conversation_id or new_id()
        with self.connection(write=True) as db:
            db.execute("INSERT INTO conversations(id,agent_id,user_id,user_role,status,title,metadata) VALUES (?,?,?,?,?,?,?)",
                       (cid, descriptor.id, user_id, user_role, "active", title, encode(metadata or {})))
            db.execute("INSERT INTO runtime_assignments(conversation_id,runtime,revision,execution_class,privacy_affinity) VALUES (?,?,?,?,?)",
                       (cid, runtime, descriptor.revision, descriptor.execution_class, descriptor.privacy_affinity))
            return self._conversation(db, cid)

    def assignment(self, conversation_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM runtime_assignments WHERE conversation_id=?", (conversation_id,)).fetchone()
            if row is None:
                raise NotFound("Runtime assignment not found")
            return dict(row)

    def attach_legacy(self, conversation_id):
        with self.connection(write=True) as db:
            self._conversation(db, conversation_id)
            db.execute("INSERT OR IGNORE INTO runtime_assignments(conversation_id,runtime,revision,execution_class) VALUES (?,'legacy','legacy','legacy')", (conversation_id,))

    def _event(self, db, run_id, event_type, payload, attempt_id=None, source_sequence=None):
        if source_sequence is not None:
            prior = db.execute("SELECT * FROM run_events WHERE attempt_id=? AND source_sequence=?", (attempt_id, source_sequence)).fetchone()
            if prior:
                if prior["type"] != event_type or prior["payload"] != encode(payload):
                    raise Conflict("Event sequence reused with different content")
                return prior["sequence"]
        db.execute("UPDATE runs SET event_sequence=event_sequence+1 WHERE id=?", (run_id,))
        sequence = db.execute("SELECT event_sequence FROM runs WHERE id=?", (run_id,)).fetchone()[0]
        db.execute("INSERT INTO run_events VALUES (?,?,?,?,?,?,?)", (run_id, sequence, attempt_id, source_sequence, event_type, encode(payload), self.clock()))
        return sequence

    def submit(self, conversation_id, user_id, payload, idempotency_key=None, resume=None):
        fingerprint = hashlib.sha256(encode({"payload": payload, "resume": resume}).encode()).hexdigest()
        with self.connection(write=True) as db:
            conversation = self._conversation(db, conversation_id, user_id)
            assignment = db.execute("SELECT * FROM runtime_assignments WHERE conversation_id=?", (conversation_id,)).fetchone()
            if assignment is None:
                raise Conflict("Conversation has no runtime assignment")
            if idempotency_key is not None:
                prior = db.execute("SELECT id,input_hash FROM runs WHERE conversation_id=? AND idempotency_key=?", (conversation_id, idempotency_key)).fetchone()
                if prior:
                    if prior["input_hash"] != fingerprint:
                        raise Conflict("Idempotency key reused with different input")
                    return self._run(db, prior["id"])
            if assignment["closed"]:
                raise Conflict("Conversation is closed")
            pending = conversation["metadata"].get("pending_interrupt")
            blocked = assignment["blocked_run_id"]
            interrupted = bool(blocked and self._run(db, blocked)["status"] == "interrupted")
            operation = payload.get("type", "text")
            resume_of = None
            if resume is not None:
                if not interrupted or not pending or resume.get("interrupt_id") != pending.get("interrupt_id"):
                    raise Conflict("Interrupt is no longer pending")
            if pending and interrupted and not assignment["resume_run_id"]:
                operation = "resume"
                resume_of = blocked
            elif resume is not None:
                raise Conflict("Interrupt already has an accepted response")
            run_id, message_id = new_id(), new_id()
            request = dict(payload)
            if operation == "resume":
                request["pending_interrupt"] = pending
                if resume is not None:
                    request["text"] = resume["response"]
            # Input is durable with the run. The legacy host remains the only
            # projector for legacy messages; workers project accepted input here.
            if assignment["runtime"] == "legacy":
                message_id = None
            else:
                text = request.get("text") or ("RESET" if operation == "reset" else "")
                content = {"type": "segments", "parts": [{"type": "reset" if operation == "reset" else "text", "text": text}]}
                db.execute("INSERT INTO messages(id,conversation_id,role,content,raw_text,metadata) VALUES (?,?,'user',?,?,?)",
                           (message_id, conversation_id, encode(content), text, encode(request.get("metadata", {}))))
            db.execute("INSERT INTO runs(id,conversation_id,sequence,revision,operation,input,input_hash,idempotency_key,status,resume_of,user_message_id,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (run_id, conversation_id, assignment["next_sequence"], assignment["revision"], operation,
                        encode(request), fingerprint, idempotency_key, "queued", resume_of, message_id, self.clock()))
            db.execute("UPDATE runtime_assignments SET next_sequence=next_sequence+1 WHERE conversation_id=?", (conversation_id,))
            if resume_of:
                db.execute("UPDATE runtime_assignments SET resume_run_id=? WHERE conversation_id=?", (run_id, conversation_id))
            db.execute("UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (conversation_id,))
            self._event(db, run_id, "status", {"status": "queued"})
            return self._run(db, run_id)

    def get_run(self, run_id, user_id=None):
        with self.connection() as db:
            return self._run(db, run_id, user_id)

    def events(self, run_id, after=0):
        with self.connection() as db:
            return [{**dict(r), "payload": json.loads(r["payload"])} for r in db.execute(
                "SELECT * FROM run_events WHERE run_id=? AND sequence>? ORDER BY sequence", (run_id, after))]

    def register_worker(self, generation, execution_class, ready, errors):
        if execution_class not in {"interactive", "batch", "legacy"}:
            raise ValueError("Invalid worker class")
        with self.connection(write=True) as db:
            prior = db.execute("SELECT execution_class FROM workers WHERE generation=?", (generation,)).fetchone()
            if prior and prior[0] != execution_class:
                raise Conflict("Worker generation cannot change execution class")
            db.execute("INSERT INTO workers VALUES (?,?,?,?,?) ON CONFLICT(generation) DO UPDATE SET heartbeat=excluded.heartbeat,ready=excluded.ready,errors=excluded.errors",
                       (generation, execution_class, self.clock(), encode(ready), encode(errors)))

    def _expire(self, db):
        now = self.clock()
        expired = db.execute("SELECT * FROM runs WHERE status IN ('claimed','running') AND lease_until<?", (now,)).fetchall()
        for run in expired:
            started = run["status"] == "running"
            outcome = "recovery_required" if started else "queued"
            db.execute("UPDATE run_attempts SET finished_at=?,outcome=? WHERE id=?", (now, outcome, run["attempt_id"]))
            db.execute("UPDATE runs SET status=?,lease_until=NULL,finished_at=?,error=? WHERE id=?",
                       (outcome, now if started else None, "Worker lease expired after execution started" if started else None, run["id"]))
            if started:
                db.execute("UPDATE runtime_assignments SET blocked_run_id=?,resume_run_id=NULL WHERE conversation_id=?", (run["id"], run["conversation_id"]))
            self._event(db, run["id"], outcome if started else "status", {"status": outcome}, run["attempt_id"])
        # Privacy state belongs to a process, including between turns.
        lost = db.execute("""SELECT a.conversation_id FROM runtime_assignments a
          LEFT JOIN workers w ON w.generation=a.generation
          WHERE a.privacy_affinity=1 AND a.generation IS NOT NULL AND a.closed=0
          AND (w.heartbeat IS NULL OR w.heartbeat<?)""", (now - self.lease_seconds,)).fetchall()
        for assignment in lost:
            for run in db.execute("SELECT id FROM runs WHERE conversation_id=? AND status='queued'", (assignment[0],)).fetchall():
                db.execute("UPDATE runs SET status='recovery_required',finished_at=?,error=? WHERE id=?", (now, "Privacy runtime generation lost", run[0]))
                db.execute("UPDATE runtime_assignments SET blocked_run_id=?,resume_run_id=NULL WHERE conversation_id=?", (run[0], assignment[0]))
                self._event(db, run[0], "recovery_required", {"status": "recovery_required", "error": "Privacy runtime generation lost"})

    def claim(self, generation):
        with self.connection(write=True) as db:
            self._expire(db)
            worker = db.execute("SELECT * FROM workers WHERE generation=?", (generation,)).fetchone()
            if not worker or worker["heartbeat"] < self.clock() - self.lease_seconds:
                raise Conflict("Worker must register before claiming")
            if db.execute("SELECT 1 FROM runs WHERE generation=? AND status IN ('claimed','running')", (generation,)).fetchone():
                return None
            ready = json.loads(worker["ready"])
            candidates = db.execute("""SELECT r.*,c.agent_id,c.user_id,c.user_role,a.runtime,a.privacy_affinity,a.generation AS affinity,a.checkpoint_reference
              FROM runs r JOIN conversations c ON c.id=r.conversation_id
              JOIN runtime_assignments a ON a.conversation_id=r.conversation_id
              WHERE r.status='queued' AND a.closed=0 AND a.execution_class=?
              AND (a.blocked_run_id IS NULL OR a.resume_run_id=r.id)
              AND NOT EXISTS (SELECT 1 FROM runs x WHERE x.conversation_id=r.conversation_id AND x.status IN ('claimed','running'))
              AND (a.resume_run_id=r.id OR NOT EXISTS (SELECT 1 FROM runs x WHERE x.conversation_id=r.conversation_id AND x.status='queued' AND x.sequence<r.sequence))
              ORDER BY r.created_at,r.sequence""", (worker["execution_class"],)).fetchall()
            for candidate in candidates:
                if candidate["runtime"] != "legacy" and ready.get(candidate["agent_id"]) != candidate["revision"]:
                    continue
                if candidate["privacy_affinity"] and candidate["affinity"] not in (None, generation):
                    continue
                attempt = new_id()
                db.execute("UPDATE runs SET status='claimed',attempt_id=?,generation=?,lease_until=? WHERE id=?", (attempt, generation, self.clock()+self.lease_seconds, candidate["id"]))
                db.execute("INSERT INTO run_attempts(id,run_id,generation,claimed_at) VALUES (?,?,?,?)", (attempt, candidate["id"], generation, self.clock()))
                self._event(db, candidate["id"], "status", {"status": "claimed"}, attempt)
                return {**self._run(db, candidate["id"]), "agent_id": candidate["agent_id"], "user_id": candidate["user_id"], "user_role": candidate["user_role"], "runtime": candidate["runtime"], "expected_checkpoint": candidate["checkpoint_reference"]}
            return None

    def _owned(self, db, run_id, attempt_id, generation):
        run = self._run(db, run_id)
        if run["attempt_id"] != attempt_id or run["generation"] != generation or run["status"] not in {"claimed", "running"} or run["lease_until"] < self.clock():
            raise Conflict("Stale or expired execution claim")
        return run

    def start(self, run_id, attempt_id, generation):
        with self.connection(write=True) as db:
            run = self._owned(db, run_id, attempt_id, generation)
            if run["cancel_requested"]:
                raise Conflict("Run was cancelled before start")
            if run["status"] == "running":
                return run
            db.execute("UPDATE runs SET status='running',started_at=? WHERE id=?", (self.clock(), run_id))
            db.execute("UPDATE run_attempts SET started_at=? WHERE id=?", (self.clock(), attempt_id))
            db.execute("UPDATE runtime_assignments SET generation=? WHERE conversation_id=? AND privacy_affinity=1", (generation, run["conversation_id"]))
            self._event(db, run_id, "status", {"status": "running"}, attempt_id)
            return self._run(db, run_id)

    def heartbeat(self, run_id, attempt_id, generation):
        with self.connection(write=True) as db:
            run = self._owned(db, run_id, attempt_id, generation)
            db.execute("UPDATE runs SET lease_until=? WHERE id=?", (self.clock()+self.lease_seconds, run_id))
            db.execute("UPDATE workers SET heartbeat=? WHERE generation=?", (self.clock(), generation))
            return {"cancel_requested": bool(run["cancel_requested"])}

    def append_event(self, run_id, attempt_id, generation, source_sequence, event_type, payload):
        if event_type not in {"chunk", "custom"} or source_sequence < 1:
            raise ValueError("Invalid worker event")
        with self.connection(write=True) as db:
            run = self._owned(db, run_id, attempt_id, generation)
            if run["status"] != "running":
                raise Conflict("Execution has not started")
            return self._event(db, run_id, event_type, payload, attempt_id, source_sequence)

    def finish(self, run_id, attempt_id, generation, status, result=None, error=None):
        if status not in {"completed", "interrupted", "failed", "cancelled", "recovery_required"}:
            raise ValueError("Invalid terminal status")
        completion_hash = hashlib.sha256(encode([status, result, error]).encode()).hexdigest()
        with self.connection(write=True) as db:
            prior = self._run(db, run_id)
            if prior["status"] in TERMINAL and prior["attempt_id"] == attempt_id and prior["generation"] == generation:
                if prior["completion_hash"] != completion_hash:
                    raise Conflict("Conflicting terminal result")
                return prior
            run = self._owned(db, run_id, attempt_id, generation)
            if run["status"] != "running":
                raise Conflict("Execution has not started")
            cid = run["conversation_id"]
            assignment = db.execute("SELECT * FROM runtime_assignments WHERE conversation_id=?", (cid,)).fetchone()
            if status in {"completed", "interrupted"}:
                if not isinstance(result, dict) or not isinstance(result.get("agent_message"), dict):
                    raise ValueError("Completion requires a serialized agent message")
                message = result["agent_message"]
                metadata = dict(message.get("metadata", {}))
                metadata["agent_status"] = status
                conversation = self._conversation(db, cid)
                metadata["agent_id"] = conversation["agent_id"]
                if status == "interrupted" and not (metadata.get("interrupt_payload") or {}).get("interrupt_id"):
                    raise ValueError("Interrupted result requires an interrupt ID")
                if assignment["runtime"] != "legacy":
                    human = result.get("accepted_message")
                    if human is not None:
                        db.execute("UPDATE messages SET content=?,raw_text=?,metadata=? WHERE id=?",
                            (encode(human["content"]), human["raw_text"], encode(human["metadata"]), run["user_message_id"]))
                    mid = new_id()
                    db.execute("INSERT INTO messages(id,conversation_id,role,content,raw_text,metadata) VALUES (?,?,'assistant',?,?,?)",
                               (mid, cid, encode(message["content"]), message["raw_text"], encode(metadata)))
                    cm = conversation["metadata"]
                    if status == "interrupted":
                        cm["pending_interrupt"] = {key: metadata["interrupt_payload"].get(key) for key in
                            ("interrupt_id", "question", "content", "artifact_id", "artifact_name")}
                    else:
                        cm.pop("pending_interrupt", None)
                    db.execute("UPDATE conversations SET metadata=?,status=?,updated_at=CURRENT_TIMESTAMP,last_message_at=CURRENT_TIMESTAMP WHERE id=?",
                               (encode(cm), "waiting_user" if status == "interrupted" else "active", cid))
                    result = {**result, "conversation": self._conversation(db, cid), "user_message": self._message(db, run["user_message_id"]), "agent_message": self._message(db, mid)}
            blocked = run_id if status in {"interrupted", "failed", "recovery_required", "cancelled"} else None
            if result and result.get("error_category") == "artifact":
                db.execute("INSERT INTO platform_metrics VALUES ('artifact_failures',1) ON CONFLICT(name) DO UPDATE SET value=value+1")
            if status in {"completed", "interrupted"} and result is not None and "checkpoint_reference" in result:
                db.execute("UPDATE runtime_assignments SET checkpoint_reference=? WHERE conversation_id=?", (result["checkpoint_reference"], cid))
            db.execute("UPDATE runtime_assignments SET blocked_run_id=?,resume_run_id=NULL WHERE conversation_id=?", (blocked, cid))
            db.execute("UPDATE runs SET status=?,result=?,error=?,completion_hash=?,finished_at=?,lease_until=NULL WHERE id=?", (status, encode(result) if result is not None else None, error, completion_hash, self.clock(), run_id))
            db.execute("UPDATE run_attempts SET finished_at=?,outcome=? WHERE id=?", (self.clock(), status, attempt_id))
            self._event(db, run_id, "interrupt" if status == "interrupted" else status,
                        {"status": status, "result": result, "error": error}, attempt_id)
            return self._run(db, run_id)

    def cancel(self, run_id, user_id):
        with self.connection(write=True) as db:
            run = self._run(db, run_id, user_id)
            if run["status"] in TERMINAL:
                return run
            if run["status"] in {"queued", "claimed"}:
                db.execute("UPDATE runs SET status='cancelled',cancel_requested=1,finished_at=?,lease_until=NULL WHERE id=?", (self.clock(), run_id))
                db.execute("UPDATE run_attempts SET finished_at=?,outcome='cancelled' WHERE id=?", (self.clock(), run["attempt_id"]))
                if run["resume_of"]:
                    db.execute("UPDATE runtime_assignments SET resume_run_id=NULL WHERE conversation_id=?", (run["conversation_id"],))
                    db.execute("UPDATE runs SET resume_of=NULL WHERE id=?", (run_id,))
                self._event(db, run_id, "cancelled", {"status": "cancelled"})
            elif not run["cancel_requested"]:
                db.execute("UPDATE runs SET cancel_requested=1 WHERE id=?", (run_id,))
                self._event(db, run_id, "status", {"status": "cancelling"})
            return self._run(db, run_id)

    def close_conversation(self, conversation_id, user_id):
        with self.connection(write=True) as db:
            self._conversation(db, conversation_id, user_id)
            if db.execute("SELECT 1 FROM runs WHERE conversation_id=? AND status IN ('queued','claimed','running')", (conversation_id,)).fetchone():
                raise Conflict("Conversation has unfinished work")
            db.execute("UPDATE runtime_assignments SET closed=1 WHERE conversation_id=?", (conversation_id,))
            db.execute("UPDATE conversations SET status='closed',updated_at=CURRENT_TIMESTAMP WHERE id=?", (conversation_id,))
            return self._conversation(db, conversation_id)

    def reconcile(self, run_id, *, checkpoint_reference, generation):
        # An explicit null means the operator verified an empty checkpoint.
        if checkpoint_reference == "":
            raise ValueError("Checkpoint reference must be an ID or explicit null")
        with self.connection(write=True) as db:
            run = self._run(db, run_id)
            if run["status"] not in {"failed", "cancelled", "recovery_required"}:
                raise Conflict("Run does not require reconciliation")
            a = db.execute("SELECT * FROM runtime_assignments WHERE conversation_id=?", (run["conversation_id"],)).fetchone()
            if a["privacy_affinity"] and a["generation"] != generation:
                raise Conflict("Privacy mapping cannot be transferred; close this conversation")
            if a["blocked_run_id"] != run_id:
                raise Conflict("Run is not the conversation blocker")
            db.execute("UPDATE runtime_assignments SET blocked_run_id=NULL,checkpoint_reference=? WHERE conversation_id=?", (checkpoint_reference, run["conversation_id"]))
            self._event(db, run_id, "reconciled", {"checkpoint": checkpoint_reference, "generation": generation})
            return self._run(db, run_id)

    def maintenance(self):
        with self.connection(write=True) as db:
            self._expire(db)
            db.execute("DELETE FROM run_events WHERE type IN ('chunk','custom') AND run_id IN (SELECT id FROM runs WHERE finished_at<?)", (self.clock()-21600,))

    def readiness(self):
        with self.connection() as db:
            workers = [{**dict(w), "ready": json.loads(w["ready"]), "errors": json.loads(w["errors"])} for w in db.execute("SELECT * FROM workers")]
            counts = {r[0]: r[1] for r in db.execute("SELECT status,count(*) FROM runs GROUP BY status")}
            oldest = db.execute("SELECT min(created_at) FROM runs WHERE status='queued'").fetchone()[0]
            return {"application": "ready", "workers": workers, "runs": counts,
                    "initialization_failures": sum(len(w["errors"]) for w in workers if w["heartbeat"] >= self.clock()-self.lease_seconds),
                    "metrics": {"artifact_failures": 0, "event_delivery_lag_seconds": 0,
                                **dict(db.execute("SELECT name,value FROM platform_metrics"))},
                    "queue_age_seconds": max(0, self.clock()-oldest) if oldest else 0,
                    "active_claims": counts.get("claimed", 0)+counts.get("running", 0),
                    "recovery_required": counts.get("recovery_required", 0)}

    def observe(self, name, value=1, *, increment=False):
        with self.connection(write=True) as db:
            db.execute("INSERT INTO platform_metrics VALUES (?,?) ON CONFLICT(name) DO UPDATE SET value=" +
                       ("platform_metrics.value+excluded.value" if increment else "excluded.value"), (name, value))
