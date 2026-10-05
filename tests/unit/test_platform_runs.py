import asyncio
import concurrent.futures
import sqlite3

import pytest

from platform_application.service import Application, Conflict, NotFound
from platform_contracts import AgentDescriptor
from platform_infrastructure.sqlite import SQLiteRepository


@pytest.fixture
def setup(tmp_path):
    now = [1000.0]
    repo = SQLiteRepository(tmp_path / "app.sqlite", clock=lambda: now[0])
    repo.migrate()
    descriptor = AgentDescriptor("test", "Test", "Test agent", "fake", "r1")
    conversation = repo.create_conversation(descriptor, "user", "default")
    repo.register_worker("g1", "interactive", {"test": "r1"}, {})
    return repo, conversation["id"], now


def submit(repo, cid, text="hi", **kwargs):
    return repo.submit(cid, "user", {"type": "text", "text": text}, **kwargs)


def start(repo, generation="g1"):
    run = repo.claim(generation)
    assert run is not None
    repo.start(run["id"], run["attempt_id"], generation)
    return run


def finish(repo, run, status="completed", result=None):
    return repo.finish(run["id"], run["attempt_id"], run["generation"], status,
                       result or {"agent_message": {"content": {"type": "text", "text": "done"}, "raw_text": "done", "metadata": {}}})


def test_acceptance_is_atomic_and_idempotent_with_owner_check(setup):
    repo, cid, _ = setup
    first = submit(repo, cid, idempotency_key="k")
    assert first == submit(repo, cid, idempotency_key="k")
    with pytest.raises(Conflict):
        submit(repo, cid, "different", idempotency_key="k")
    assert submit(repo, cid)["id"] != first["id"]
    assert len(repo.conversation(cid, "user", True)["messages"]) == 2
    with pytest.raises(NotFound):
        repo.get_run(first["id"], "other")


def test_concurrent_submissions_and_claims_are_ordered(setup):
    repo, cid, _ = setup
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        submitted = list(pool.map(lambda x: submit(repo, cid, str(x)), range(20)))
    assert sorted(r["sequence"] for r in submitted) == list(range(1, 21))
    repo.register_worker("g2", "interactive", {"test": "r1"}, {})
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        claims = list(pool.map(repo.claim, ["g1", "g2"]))
    claimed = [r for r in claims if r]
    assert len(claimed) == 1 and claimed[0]["sequence"] == 1
    run = claimed[0]
    repo.start(run["id"], run["attempt_id"], run["generation"])
    finish(repo, run)
    assert repo.claim("g1")["sequence"] == 2


def test_completion_projection_and_event_delivery_are_durable(setup):
    repo, cid, _ = setup
    submit(repo, cid)
    run = start(repo)
    result = finish(repo, run)
    assert finish(repo, run) == result
    assert len(repo.conversation(cid, "user", True)["messages"]) == 2
    reopened = SQLiteRepository(repo.path)
    reopened.migrate()
    async def check():
        events = [e async for e in Application(reopened).follow(run["id"], "user")]
        assert events[-1]["type"] == "completed"
        assert len([e for e in events if e["type"] == "completed"]) == 1
    asyncio.run(check())


def test_unstarted_claim_requeues_but_started_loss_requires_recovery(setup):
    repo, cid, now = setup
    submit(repo, cid)
    old = repo.claim("g1")
    now[0] += 31
    repo.register_worker("g2", "interactive", {"test": "r1"}, {})
    run = start(repo, "g2")
    assert run["id"] == old["id"] and run["attempt_id"] != old["attempt_id"]
    with pytest.raises(Conflict):
        repo.start(old["id"], old["attempt_id"], "g1")
    submit(repo, cid)
    now[0] += 31
    repo.maintenance()
    repo.register_worker("g3", "interactive", {"test": "r1"}, {})
    assert repo.get_run(run["id"])["status"] == "recovery_required"
    assert repo.claim("g3") is None
    with pytest.raises(Conflict):
        finish(repo, run)
    repo.reconcile(run["id"], checkpoint_reference="operator-verified-checkpoint", generation="g3")
    assert repo.claim("g3")["sequence"] == 2


def test_interrupt_pauses_earlier_messages_and_resume_has_priority(setup):
    repo, cid, _ = setup
    submit(repo, cid)
    run = start(repo)
    earlier = submit(repo, cid, "queued before interrupt")
    finish(repo, run, "interrupted", {"agent_message": {"content": {"type": "text", "text": "Approve?"},
           "raw_text": "Approve?", "metadata": {"interrupt_payload": {"interrupt_id": "int-1", "question": "Approve?"}}}})
    assert repo.claim("g1") is None
    response = repo.submit(cid, "user", {"text": "yes"}, "resume-key", {"interrupt_id": "int-1", "response": "yes"})
    assert response["operation"] == "resume"
    with pytest.raises(Conflict):
        repo.submit(cid, "user", {"text": "yes"}, resume={"interrupt_id": "int-1", "response": "yes"})
    resumed = start(repo)
    assert resumed["id"] == response["id"]
    finish(repo, resumed)
    assert repo.claim("g1")["id"] == earlier["id"]


def test_cancel_and_close(setup):
    repo, cid, _ = setup
    run = submit(repo, cid)
    with pytest.raises(Conflict):
        repo.close_conversation(cid, "user")
    assert repo.cancel(run["id"], "user")["status"] == "cancelled"
    next_run = submit(repo, cid)
    claimed = start(repo)
    assert claimed["id"] == next_run["id"]
    assert repo.cancel(claimed["id"], "user")["status"] == "running"
    assert repo.heartbeat(claimed["id"], claimed["attempt_id"], "g1")["cancel_requested"]
    repo.finish(claimed["id"], claimed["attempt_id"], "g1", "cancelled")
    assert repo.close_conversation(cid, "user")["status"] == "closed"
    with pytest.raises(Conflict):
        submit(repo, cid)


def test_events_are_idempotent_and_deltas_expire(setup):
    repo, cid, now = setup
    submit(repo, cid)
    run = start(repo)
    args = (run["id"], run["attempt_id"], "g1", 1, "chunk", {"content": "hi"})
    assert repo.append_event(*args) == repo.append_event(*args)
    with pytest.raises(Conflict):
        repo.append_event(*args[:-1], {"content": "different"})
    finish(repo, run)
    now[0] += 21601
    repo.maintenance()
    assert not [e for e in repo.events(run["id"]) if e["type"] == "chunk"]
    assert repo.events(run["id"])[-1]["type"] == "completed"


def test_artifact_failure_metric_is_not_duplicated_by_completion_retry(setup):
    repo, cid, _ = setup
    submit(repo, cid)
    run = start(repo)
    finish(repo, run, "failed", {"error_category": "artifact"})
    finish(repo, run, "failed", {"error_category": "artifact"})
    assert repo.readiness()["metrics"]["artifact_failures"] == 1


def test_privacy_affinity_survives_between_turns(setup):
    repo, _, now = setup
    descriptor = AgentDescriptor("test", "Test", "Test", "fake", "r1", privacy_affinity=True)
    cid = repo.create_conversation(descriptor, "user", "default")["id"]
    submit(repo, cid)
    finish(repo, start(repo))
    repo.register_worker("g2", "interactive", {"test": "r1"}, {})
    later = submit(repo, cid)
    assert repo.claim("g2") is None
    now[0] += 31
    repo.maintenance()
    assert repo.get_run(later["id"])["status"] == "recovery_required"
    with pytest.raises(Conflict):
        repo.reconcile(later["id"], checkpoint_reference="checkpoint", generation="g2")


def test_migration_keeps_old_sessions_and_does_not_duplicate_legacy_results(tmp_path):
    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(path)
    db.executescript("""CREATE TABLE conversations(id TEXT PRIMARY KEY,agent_id TEXT,user_id TEXT,user_role TEXT,status TEXT,title TEXT,metadata JSON,created_at DATETIME,updated_at DATETIME,last_message_at DATETIME);
    INSERT INTO conversations VALUES ('old','agent','user','default','active',NULL,'{}',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP);""")
    db.close()
    repo = SQLiteRepository(path)
    repo.migrate()
    repo.migrate()
    assert repo.assignment("old")["runtime"] == "legacy"
    submit(repo, "old")
    repo.register_worker("bridge", "legacy", {}, {})
    finish(repo, start(repo, "bridge"))
    assert repo.conversation("old", "user", True)["messages"] == []
