"""Bounded workspace snapshots with strict data validation, not executable backups."""
from __future__ import annotations

from datetime import datetime
import copy
import hashlib
import json
import re

from .store import StoreError, _json, correction_identity

LIMIT = 6 * 1024 * 1024
COLUMNS = {
    "chats": "id title created_at updated_at".split(),
    "messages": "id chat_id position role content evidence_json knowledge_revision image_path created_at".split(),
    "corrections": ("id question answer source scope question_key scope_key lineage_id version status supersedes "
                    "superseded_by approved_by approved_at created_at updated_at knowledge_revision identity_json").split(),
    "events": "sequence correction_id action revision actor created_at detail_json".split(),
}


def seal(snapshot):
    snapshot.pop("sha256", None)
    snapshot["sha256"] = hashlib.sha256(_json(snapshot).encode("utf-8")).hexdigest()
    return snapshot


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise StoreError("Snapshot contains an invalid record identifier")


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise StoreError("Snapshot contains an invalid integer")


def _date(value):
    try:
        if not isinstance(value, str) or len(value) > 50 or datetime.fromisoformat(value).utcoffset() is None:
            raise ValueError()
    except (ValueError, TypeError):
        raise StoreError("Snapshot contains an invalid timezone-aware timestamp") from None


def _read_json(value):
    if not isinstance(value, str) or len(value) > 200000:
        raise StoreError("Snapshot JSON field exceeds its limit")
    try:
        return json.loads(value, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, TypeError, RecursionError):
        raise StoreError("Snapshot contains invalid JSON data") from None


def validate_snapshot(snapshot):
    if not isinstance(snapshot, dict) or set(snapshot) != {"schema", "created_at", "revision", "tables", "sha256"}:
        raise StoreError("Choose an EditGuard workspace snapshot")
    if len(_json(snapshot).encode("utf-8")) > LIMIT:
        raise StoreError("Workspace snapshot exceeds 6 MiB", 413)
    value = copy.deepcopy(snapshot)
    expected = value.pop("sha256")
    if not isinstance(expected, str) or hashlib.sha256(_json(value).encode("utf-8")).hexdigest() != expected:
        raise StoreError("Workspace snapshot checksum does not match")
    value["sha256"] = expected
    if value["schema"] != "editguard-workspace-v1":
        raise StoreError("Unsupported workspace schema")
    _date(value["created_at"])
    _integer(value["revision"])
    if value["revision"] > 100000:
        raise StoreError("Snapshot revision exceeds the supported limit")
    tables = value["tables"]
    if not isinstance(tables, dict) or set(tables) != set(COLUMNS):
        raise StoreError("Snapshot tables do not match the workspace schema")
    for table, columns in COLUMNS.items():
        rows = tables[table]
        if not isinstance(rows, list) or len(rows) > 10000:
            raise StoreError("Snapshot table exceeds 10000 records")
        for row in rows:
            if not isinstance(row, dict) or set(row) != set(columns):
                raise StoreError("Snapshot record fields do not match the schema")
            for key, field in row.items():
                if key in ("created_at", "updated_at", "approved_at"):
                    _date(field)
                elif key in ("id", "chat_id", "lineage_id", "correction_id"):
                    _identifier(field)
                elif key in ("supersedes", "superseded_by") and field is not None:
                    _identifier(field)
                elif key in ("version", "position", "sequence"):
                    _integer(field, 1)
                elif key in ("revision", "knowledge_revision"):
                    _integer(field)
                    if field > value["revision"]:
                        raise StoreError("A record refers to a future knowledge revision")
                elif key.endswith("_json"):
                    _read_json(field)
                elif field is None and key not in ("supersedes", "superseded_by", "image_path"):
                    raise StoreError("Snapshot contains a missing required field")
                elif field is not None and (not isinstance(field, str) or len(field) > 16000):
                    raise StoreError("Snapshot contains an invalid text field")
            if "image_path" in row and row["image_path"] is not None:
                raise StoreError("Text workspace snapshots cannot restore local image paths. Use the private database backup for image conversations.")
    ids = {}
    for table in ("chats", "messages", "corrections"):
        ids[table] = {row["id"] for row in tables[table]}
        if len(ids[table]) != len(tables[table]):
            raise StoreError("Snapshot contains duplicate identifiers")
    positions = {}
    for row in tables["messages"]:
        if row["chat_id"] not in ids["chats"] or row["role"] not in ("user", "assistant", "system"):
            raise StoreError("Snapshot message has an invalid conversation or role")
        positions.setdefault(row["chat_id"], []).append(row["position"])
    if any(sorted(found) != list(range(1, len(found) + 1)) for found in positions.values()):
        raise StoreError("Snapshot message positions are not contiguous")
    active, versions, corrections = set(), set(), {r["id"]: r for r in tables["corrections"]}
    for row in corrections.values():
        identity = _read_json(row["identity_json"])
        key, scope, checked = correction_identity(row["question"], row["scope"], identity)
        if (key, scope) != (row["question_key"], row["scope_key"]) or identity != checked:
            raise StoreError("Snapshot canonical fact keys do not match their fields")
        for name, maximum in (("question", 512), ("answer", 2000), ("source", 2048), ("scope", 160)):
            if not row[name].strip() or len(row[name]) > maximum:
                raise StoreError("Snapshot correction fields exceed their limits")
        if row["status"] not in ("active", "withdrawn", "superseded") or row["approved_by"] != "local_research_operator":
            raise StoreError("Snapshot contains an invalid correction state or actor")
        fact_key = (key, scope)
        version_key = (*fact_key, row["version"])
        if version_key in versions or row["status"] == "active" and fact_key in active:
            raise StoreError("Snapshot contains competing active facts or duplicate versions")
        versions.add(version_key)
        if row["status"] == "active":
            active.add(fact_key)
        lineage = corrections.get(row["lineage_id"])
        if lineage is None or lineage["version"] != 1 or lineage["lineage_id"] != lineage["id"]:
            raise StoreError("Snapshot lineage must refer to its original version")
        for name in ("lineage_id", "supersedes", "superseded_by"):
            reference = row[name]
            if reference is not None:
                target = corrections.get(reference)
                if target is None or (target["question_key"], target["scope_key"]) != fact_key:
                    raise StoreError("Snapshot correction lineage is broken")
                if name != "lineage_id" and reference == row["id"]:
                    raise StoreError("Snapshot contains a self-referencing correction")
    sequences, revisions = set(), set()
    for row in tables["events"]:
        if (row["sequence"] in sequences or row["correction_id"] not in ids["corrections"]
                or row["action"] not in ("published", "superseded", "withdrawn", "restored")
                or row["actor"] != "local_research_operator" or row["revision"] < 1):
            raise StoreError("Snapshot audit history is invalid")
        sequences.add(row["sequence"])
        revisions.add(row["revision"])
    if revisions != set(range(1, value["revision"] + 1)):
        raise StoreError("Snapshot audit history omits a knowledge revision")
    return value
