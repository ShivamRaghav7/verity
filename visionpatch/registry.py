"""Transactional, restartable approval and exact rollback for projection patches.

SQLite is authoritative for the active chain. Preview never changes that chain,
and every activation checks the precise parent state. Artifacts are promoted
atomically before their database rows become visible; a failed transaction may
leave an unreferenced file, but never a partially published patch. This registry
is a local administrator workbench, not an authentication or authorization layer.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Iterator
import uuid

import torch
from torch import Tensor

from .editor import (
    ProjectionPatch, load_projection_patch, projection_sha256, save_projection_patch,
)


class RegistryConflictError(ValueError):
    """A requested transition is incompatible with the present active state."""


class RegistryIntegrityError(ValueError):
    """A persisted artifact or chain does not match its registered exact hash."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: dict[str, Any] | None) -> str:
    if value is None:
        value = {}
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("metadata must be a JSON object with string keys")
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain finite JSON values") from exc
    if len(encoded.encode("utf-8")) > 1024 * 1024:
        raise ValueError("metadata exceeds the one MiB limit")
    return encoded


def _identity(actor: str, reason: str) -> tuple[str, str]:
    if not isinstance(actor, str) or not actor.strip() or len(actor) > 512:
        raise ValueError("actor must be a nonempty string of at most 512 characters")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 4096:
        raise ValueError("reason must be a nonempty string of at most 4096 characters")
    return actor.strip(), reason.strip()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS registry_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    format_version INTEGER NOT NULL CHECK (format_version = 1),
    artifacts_dir TEXT NOT NULL,
    base_sha256 TEXT NOT NULL,
    base_artifact_sha256 TEXT NOT NULL,
    current_sha256 TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision >= 0)
);
CREATE TABLE IF NOT EXISTS patches (
    patch_id TEXT PRIMARY KEY,
    parent_patch_id TEXT REFERENCES patches(patch_id),
    before_sha256 TEXT NOT NULL,
    after_sha256 TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('candidate', 'approved', 'withdrawn')),
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_reason TEXT NOT NULL,
    approved_at TEXT,
    approved_by TEXT,
    approved_reason TEXT
);
CREATE TABLE IF NOT EXISTS active_chain (
    ordinal INTEGER PRIMARY KEY CHECK (ordinal >= 0),
    patch_id TEXT NOT NULL UNIQUE REFERENCES patches(patch_id)
);
CREATE TABLE IF NOT EXISTS audit_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    action TEXT NOT NULL,
    patch_id TEXT REFERENCES patches(patch_id),
    actor TEXT NOT NULL,
    reason TEXT NOT NULL,
    revision INTEGER NOT NULL,
    details_json TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END;
"""


class PatchRegistry:
    """Persist projection edits with optimistic activation and exact snapshots.

    The supplied ``base_weight`` is bound on first initialization and checked on
    every restart. Callers must serialize their own model tensor replacement
    with inference. Activation/rollback return the committed CPU weight; reading
    ``get_current_weight`` after a model copy failure recovers authoritative state.
    Revision numbers cover active-state changes, not draft or approval actions.
    """

    def __init__(
        self,
        db_path: str | Path,
        base_weight: Tensor,
        artifacts_dir: str | Path | None = None,
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.artifacts_dir = (
            Path(artifacts_dir).resolve() if artifacts_dir is not None
            else self.db_path.parent / "patches"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        base = base_weight.detach().cpu().contiguous().clone()
        base_hash = projection_sha256(base)
        connection = self._connect()
        try:
            connection.executescript(_SCHEMA)
        finally:
            connection.close()
        with self._transaction() as connection:
            state = connection.execute("SELECT * FROM registry_state").fetchone()
            if state is None:
                self._atomic_save(lambda path: torch.save(base, path), "base.pt")
                connection.execute(
                    "INSERT INTO registry_state VALUES (1, 1, ?, ?, ?, ?, 0)",
                    (str(self.artifacts_dir), base_hash,
                     _file_hash(self.artifacts_dir / "base.pt"), base_hash),
                )
                self._audit(connection, "initialize", None, "system", "Bind original projection",
                            {"base_sha256": base_hash})
            elif state["format_version"] != 1:
                raise RegistryIntegrityError("unsupported registry format")
            elif state["base_sha256"] != base_hash:
                raise RegistryIntegrityError("supplied base projection does not match registry")
            elif state["artifacts_dir"] != str(self.artifacts_dir):
                raise RegistryIntegrityError("registry is bound to a different artifacts directory")
            self._current(connection)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _atomic_save(self, writer: Any, filename: str) -> None:
        handle, temporary_name = tempfile.mkstemp(prefix=".writing-", suffix=".pt",
                                                  dir=self.artifacts_dir)
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            writer(temporary)
            # Windows FlushFileBuffers requires a handle opened for writing.
            with temporary.open("r+b") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, self.artifacts_dir / filename)
        finally:
            temporary.unlink(missing_ok=True)

    def _artifact_path(self, patch_id: str) -> Path:
        # Only database-generated UUIDs are ever interpreted as filenames.
        try:
            if str(uuid.UUID(patch_id)) != patch_id:
                raise ValueError()
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("invalid patch identifier") from exc
        path = self.artifacts_dir / f"{patch_id}.pt"
        if path.resolve().parent != self.artifacts_dir:
            raise RegistryIntegrityError("artifact path escapes registry directory")
        return path

    @staticmethod
    def _row(connection: sqlite3.Connection, patch_id: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM patches WHERE patch_id = ?", (patch_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown patch: {patch_id}")
        return row

    def _load(self, row: sqlite3.Row) -> ProjectionPatch:
        path = self._artifact_path(row["patch_id"])
        try:
            if _file_hash(path) != row["artifact_sha256"]:
                raise RegistryIntegrityError("patch artifact file hash mismatch")
            patch = load_projection_patch(path)
            if (patch.before_sha256 != row["before_sha256"]
                    or patch.after_sha256 != row["after_sha256"]
                    or _file_hash(path) != row["artifact_sha256"]):
                raise RegistryIntegrityError("patch artifact does not match registry hashes")
            return patch
        except RegistryIntegrityError:
            raise
        except Exception as exc:
            raise RegistryIntegrityError("cannot validate registered patch artifact") from exc

    def _current(self, connection: sqlite3.Connection) -> tuple[Tensor, list[str], sqlite3.Row]:
        state = connection.execute("SELECT * FROM registry_state WHERE singleton = 1").fetchone()
        if state is None:
            raise RegistryIntegrityError("registry has no base state")
        path = self.artifacts_dir / "base.pt"
        try:
            if (path.resolve().parent != self.artifacts_dir
                    or _file_hash(path) != state["base_artifact_sha256"]):
                raise RegistryIntegrityError("base artifact file hash mismatch")
            weight = torch.load(path, map_location="cpu", weights_only=True)
            if (projection_sha256(weight) != state["base_sha256"]
                    or _file_hash(path) != state["base_artifact_sha256"]):
                raise RegistryIntegrityError("base projection hash mismatch")
        except RegistryIntegrityError:
            raise
        except Exception as exc:
            raise RegistryIntegrityError("cannot validate base projection artifact") from exc
        active: list[str] = []
        for position, row in enumerate(connection.execute(
            "SELECT p.*, a.ordinal FROM active_chain a JOIN patches p USING (patch_id) "
            "ORDER BY a.ordinal"
        ).fetchall()):
            parent = active[-1] if active else None
            if (row["ordinal"] != position or row["status"] != "approved"
                    or row["parent_patch_id"] != parent
                    or row["before_sha256"] != projection_sha256(weight)):
                raise RegistryIntegrityError("active patch chain has invalid lineage")
            weight = self._load(row).edited_weight()
            active.append(row["patch_id"])
        if projection_sha256(weight) != state["current_sha256"]:
            raise RegistryIntegrityError("active chain and registered current projection disagree")
        return weight, active, state

    @staticmethod
    def _audit(connection: sqlite3.Connection, action: str, patch_id: str | None,
               actor: str, reason: str, details: dict[str, Any]) -> None:
        revision = connection.execute("SELECT revision FROM registry_state").fetchone()[0]
        connection.execute(
            "INSERT INTO audit_events (at, action, patch_id, actor, reason, revision, details_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (_now(), action, patch_id, actor, reason, revision, _json(details)),
        )

    @staticmethod
    def _record(connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        result["approval_status"] = result["status"]
        active = connection.execute("SELECT ordinal FROM active_chain WHERE patch_id = ?",
                                    (row["patch_id"],)).fetchone()
        result["active"] = active is not None
        result["active_order"] = active["ordinal"] if active else None
        if active:
            result["status"] = "active"
        return result

    def state(self) -> dict[str, Any]:
        with self._transaction() as connection:
            _, active, state = self._current(connection)
            return {"base_sha256": state["base_sha256"],
                    "current_sha256": state["current_sha256"],
                    "revision": state["revision"], "active_patch_ids": active}

    def get_current_weight(self) -> Tensor:
        with self._transaction() as connection:
            weight, _, _ = self._current(connection)
            return weight.clone()

    def propose(self, patch: ProjectionPatch, *, actor: str, reason: str,
                metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        actor, reason = _identity(actor, reason)
        patch.edited_weight()  # Validate mutable tensor storage before publication.
        encoded = _json(metadata if metadata is not None else patch.metadata)
        patch_id = str(uuid.uuid4())
        with self._transaction() as connection:
            _, active, state = self._current(connection)
            if patch.before_sha256 == state["current_sha256"]:
                parent = active[-1] if active else None
            elif patch.before_sha256 == state["base_sha256"]:
                parent = None
            else:
                parent_row = connection.execute(
                    "SELECT * FROM patches WHERE after_sha256 = ? ORDER BY rowid DESC LIMIT 1",
                    (patch.before_sha256,),
                ).fetchone()
                if parent_row is None:
                    raise RegistryIntegrityError("patch source is not a known registry state")
                self._load(parent_row)
                parent = parent_row["patch_id"]
            path = self._artifact_path(patch_id)
            self._atomic_save(lambda target: save_projection_patch(patch, target), path.name)
            connection.execute(
                "INSERT INTO patches (patch_id, parent_patch_id, before_sha256, after_sha256, "
                "artifact_sha256, status, metadata_json, created_at, created_by, created_reason) "
                "VALUES (?, ?, ?, ?, ?, 'candidate', ?, ?, ?, ?)",
                (patch_id, parent, patch.before_sha256, patch.after_sha256, _file_hash(path),
                 encoded, _now(), actor, reason),
            )
            self._audit(connection, "propose", patch_id, actor, reason,
                        {"before_sha256": patch.before_sha256, "after_sha256": patch.after_sha256,
                         "parent_patch_id": parent})
            return self._record(connection, self._row(connection, patch_id))

    def approve(self, patch_id: str, *, actor: str, reason: str) -> dict[str, Any]:
        actor, reason = _identity(actor, reason)
        with self._transaction() as connection:
            self._current(connection)
            row = self._row(connection, patch_id)
            if row["status"] != "candidate":
                raise RegistryConflictError("only a candidate patch can be approved")
            self._load(row)
            connection.execute(
                "UPDATE patches SET status = 'approved', approved_at = ?, approved_by = ?, "
                "approved_reason = ? WHERE patch_id = ?", (_now(), actor, reason, patch_id),
            )
            self._audit(connection, "approve", patch_id, actor, reason, {})
            return self._record(connection, self._row(connection, patch_id))

    def activate(self, patch_id: str, *, actor: str, reason: str,
                 expected_revision: int | None = None) -> Tensor:
        actor, reason = _identity(actor, reason)
        with self._transaction() as connection:
            _, active, state = self._current(connection)
            self._check_revision(state, expected_revision)
            row = self._row(connection, patch_id)
            if row["status"] != "approved":
                raise RegistryConflictError("patch must be explicitly approved before activation")
            if patch_id in active:
                raise RegistryConflictError("patch is already active")
            if (row["before_sha256"] != state["current_sha256"]
                    or row["parent_patch_id"] != (active[-1] if active else None)):
                raise RegistryConflictError("stale patch parent; fit a new patch on the current state")
            weight = self._load(row).edited_weight()
            connection.execute("INSERT INTO active_chain VALUES (?, ?)", (len(active), patch_id))
            connection.execute("UPDATE registry_state SET current_sha256 = ?, revision = revision + 1",
                               (row["after_sha256"],))
            self._audit(connection, "activate", patch_id, actor, reason,
                        {"before_sha256": row["before_sha256"], "after_sha256": row["after_sha256"]})
            return weight.clone()

    @staticmethod
    def _check_revision(state: sqlite3.Row, expected_revision: int | None) -> None:
        if expected_revision is not None:
            if type(expected_revision) is not int or expected_revision < 0:
                raise ValueError("expected_revision must be a nonnegative integer")
            if expected_revision != state["revision"]:
                raise RegistryConflictError("registry revision changed; refresh before modifying state")

    def rollback(self, *, actor: str, reason: str, to_patch_id: str | None = None,
                 expected_revision: int | None = None) -> Tensor:
        """Pop one edit, or retain a specified active ancestor (``'base'`` = none)."""
        actor, reason = _identity(actor, reason)
        with self._transaction() as connection:
            _, active, state = self._current(connection)
            self._check_revision(state, expected_revision)
            if not active:
                raise RegistryConflictError("there is no active patch to roll back")
            if to_patch_id is None:
                keep = len(active) - 1
            elif to_patch_id == "base":
                keep = 0
            elif to_patch_id in active:
                keep = active.index(to_patch_id) + 1
            else:
                raise RegistryConflictError("rollback target must be an active ancestor or 'base'")
            if keep == len(active):
                raise RegistryConflictError("rollback target is already the current state")
            removed = active[keep:]
            # Restore the first removed edit's saved parent bytes, not a subtraction.
            weight = self._load(self._row(connection, removed[0])).before_weight.clone()
            connection.execute("DELETE FROM active_chain WHERE ordinal >= ?", (keep,))
            restored_hash = projection_sha256(weight)
            connection.execute("UPDATE registry_state SET current_sha256 = ?, revision = revision + 1",
                               (restored_hash,))
            self._audit(connection, "rollback", removed[-1], actor, reason,
                        {"removed_patch_ids": removed, "restored_sha256": restored_hash,
                         "retained_patch_id": active[keep - 1] if keep else None})
            self._current(connection)
            return weight

    def withdraw(self, patch_id: str, *, actor: str, reason: str) -> dict[str, Any]:
        actor, reason = _identity(actor, reason)
        with self._transaction() as connection:
            _, active, _ = self._current(connection)
            row = self._row(connection, patch_id)
            if patch_id in active:
                raise RegistryConflictError("roll back an active patch before withdrawing it")
            if row["status"] == "withdrawn":
                raise RegistryConflictError("patch is already withdrawn")
            connection.execute("UPDATE patches SET status = 'withdrawn' WHERE patch_id = ?", (patch_id,))
            self._audit(connection, "withdraw", patch_id, actor, reason, {})
            return self._record(connection, self._row(connection, patch_id))

    def load_patch(self, patch_id: str) -> ProjectionPatch:
        with self._transaction() as connection:
            return self._load(self._row(connection, patch_id))

    def get_patch(self, patch_id: str) -> dict[str, Any]:
        with self._transaction() as connection:
            row = self._row(connection, patch_id)
            self._load(row)
            return self._record(connection, row)

    def list_patches(self) -> list[dict[str, Any]]:
        with self._transaction() as connection:
            return [self._record(connection, row) for row in connection.execute(
                "SELECT * FROM patches ORDER BY rowid DESC"
            ).fetchall()]

    def events(self) -> list[dict[str, Any]]:
        with self._transaction() as connection:
            results = []
            for row in connection.execute("SELECT * FROM audit_events ORDER BY event_id"):
                result = dict(row)
                result["details"] = json.loads(result.pop("details_json"))
                results.append(result)
            return results
