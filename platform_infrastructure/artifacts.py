"""Artifact IDs cross process boundaries; paths stay in this adapter."""
import base64
import hashlib
import json
import os
import uuid
from pathlib import Path

from platform_application.service import NotFound
from platform_contracts import ArtifactRef
from dataclasses import asdict


class ArtifactStorage:
    def __init__(self, root, repository):
        self.root = Path(root).resolve()
        self.repository = repository

    def stage(self, conversation_id, user_id, payload):
        self.repository.conversation(conversation_id, user_id)
        value = dict(payload)
        attachments = value.pop("attachments", [])
        refs = []
        for attachment in attachments:
            # Store the original payload; decoding/parsing for the agent happens
            # on the worker with the original content-type behavior.
            raw = json.dumps(attachment, sort_keys=True, ensure_ascii=False).encode()
            if attachment.get("data"):
                base64.b64decode(attachment["data"], validate=True)
            key = hashlib.sha256((conversation_id+"\0"+user_id+"\0").encode()+raw).hexdigest()
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.root / key
            if not path.exists():
                temporary = self.root / (key+"."+uuid.uuid4().hex+".tmp")
                try:
                    temporary.write_bytes(raw)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            with self.repository.connection(write=True) as db:
                db.execute("INSERT OR IGNORE INTO artifacts VALUES (?,?,?,?,?,?,?)", (
                    key, conversation_id, user_id, attachment.get("content_type") or "application/octet-stream",
                    Path(attachment.get("filename") or attachment.get("name") or "artifact").name, key, self.repository.clock()))
            refs.append(key)
        value["artifact_refs"] = refs
        return value

    def get(self, artifact_id, user_id=None, conversation_id=None):
        with self.repository.connection() as db:
            row = db.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
            if row is None or (user_id is not None and row["owner_id"] != user_id) or (conversation_id is not None and row["conversation_id"] != conversation_id):
                raise NotFound("Artifact not found")
            path = (self.root / row["storage_key"]).resolve()
            if path.parent != self.root:
                raise ValueError("Invalid artifact storage key")
            return json.loads(path.read_text(encoding="utf-8"))

    def describe(self, artifact_id, conversation_id):
        with self.repository.connection() as db:
            row = db.execute("SELECT * FROM artifacts WHERE id=? AND conversation_id=?", (artifact_id, conversation_id)).fetchone()
            if row is None:
                raise NotFound("Artifact not found")
            return asdict(ArtifactRef(row["id"], row["owner_id"], row["conversation_id"],
                row["media_type"], row["storage_key"], row["filename"]))

    def upload(self, run_id, attempt_id, generation, payload):
        with self.repository.connection() as db:
            run = self.repository._owned(db, run_id, attempt_id, generation)
            owner = self.repository._conversation(db, run["conversation_id"])["user_id"]
        payload = dict(payload)
        staged = self.stage(run["conversation_id"], owner, {"attachments": [payload]})
        return self.describe(staged["artifact_refs"][0], run["conversation_id"])

    def materialize_result(self, conversation_id, result):
        """Restore the existing attachment wire shape inside the access layer."""
        if result is None:
            return None
        def expand(value):
            if isinstance(value, dict):
                if set(value) == {"artifact_id"}:
                    return self.get(value["artifact_id"], conversation_id=conversation_id)
                return {key: expand(item) for key, item in value.items()}
            if isinstance(value, list):
                return [expand(item) for item in value]
            return value
        output = expand(result)
        for reference in output.get("artifacts", []):
            if self.describe(reference["id"], conversation_id) != reference:
                raise ValueError("Artifact reference does not match stored ownership")
        return output
