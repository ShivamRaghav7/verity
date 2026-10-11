"""Transactional authoritative facts, approved corrections, and immutable audit events.

The initial domain uses exact identifiers, half-open effective dates, and numeric
facts. Pending edits have no authority. Contradictory pending edits cause explicit
conflict abstention only where their date intervals overlap.
"""

from contextlib import contextmanager
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sqlite3
from threading import RLock
from uuid import uuid4

from pydantic import ValidationError

from .models import CorrectionCreate, Fact, iso_date


class DomainError(Exception):
    def __init__(self, message: str, code: int = 400):
        super().__init__(message)
        self.code = code


_INFINITY = date.max.toordinal() + 1


def _interval(record):
    start = date.fromisoformat(record["effective_from"]).toordinal()
    end = date.fromisoformat(record["effective_to"]).toordinal() if record.get("effective_to") else _INFINITY
    return start, end


def _scope(record):
    return tuple(record[key] for key in ("entity_id", "relation", "plan", "version"))


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class Store:
    def __init__(self, path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False, timeout=10)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA busy_timeout=10000")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY, data_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS edits (
                edit_id TEXT PRIMARY KEY,
                fact_id TEXT NOT NULL REFERENCES facts(fact_id),
                supersedes TEXT REFERENCES edits(edit_id),
                status TEXT NOT NULL CHECK(status IN ('draft','approved','withdrawn')),
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                approved_at TEXT
            );
            CREATE INDEX IF NOT EXISTS edits_fact ON edits(fact_id);
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY, value INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO metadata(key,value) VALUES ('revision',0);
            CREATE TABLE IF NOT EXISTS events (
                revision INTEGER PRIMARY KEY,
                event_id TEXT UNIQUE NOT NULL,
                event_type TEXT NOT NULL,
                actor TEXT NOT NULL,
                created_at TEXT NOT NULL,
                details_json TEXT NOT NULL
            );
        """)

    def close(self):
        with self._lock:
            self._db.close()

    @contextmanager
    def _transaction(self):
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._db.rollback()
                raise
            else:
                self._db.commit()

    @contextmanager
    def snapshot(self):
        """Read a coherent facts/edits/events/revision snapshot across writers.

        Nested snapshots reuse the caller's current transaction, including an
        existing mutation transaction, and never commit their caller's work.
        Start ordinary mutation methods outside an outer read snapshot.
        """
        with self._lock:
            if self._db.in_transaction:
                yield
                return
            self._db.execute("BEGIN")
            try:
                yield
            except BaseException:
                self._db.rollback()
                raise
            else:
                self._db.commit()

    def _event(self, event_type, actor, details):
        self._db.execute("UPDATE metadata SET value=value+1 WHERE key='revision'")
        revision = self.revision()
        self._db.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?)",
            (revision, str(uuid4()), event_type, actor, _now(), _json(details)),
        )

    def revision(self):
        with self._lock:
            return self._db.execute("SELECT value FROM metadata WHERE key='revision'").fetchone()[0]

    def seed(self):
        path = Path(__file__).resolve().parent.parent / "data" / "catalogue.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = [Fact.model_validate(row).model_dump(mode="json") for row in payload["facts"]]
        with self._transaction():
            existing = {record["fact_id"]: record for record in self.facts()}
            combined = dict(existing)
            inserted = []
            for record in records:
                fact_id = record["fact_id"]
                if fact_id in existing:
                    if existing[fact_id] != record:
                        raise DomainError(f"Seed fact {fact_id} changed; an explicit data migration is required", 409)
                    continue
                for prior in combined.values():
                    a, b = _interval(record), _interval(prior)
                    if _scope(record) == _scope(prior) and max(a[0], b[0]) < min(a[1], b[1]):
                        raise DomainError("Catalogue contains overlapping facts in the same exact scope", 409)
                self._db.execute("INSERT INTO facts VALUES (?,?)", (fact_id, _json(record)))
                combined[fact_id] = record
                inserted.append(fact_id)
            if inserted:
                self._event("catalogue_seeded", "system", {"dataset_id": payload["dataset_id"], "fact_ids": inserted})

    def facts(self):
        with self._lock:
            return [json.loads(row[0]) for row in self._db.execute("SELECT data_json FROM facts ORDER BY fact_id")]

    def _fact(self, fact_id):
        row = self._db.execute("SELECT data_json FROM facts WHERE fact_id=?", (fact_id,)).fetchone()
        if row is None:
            raise DomainError(f"Unknown fact: {fact_id}", 404)
        return json.loads(row[0])

    @staticmethod
    def _decode_edit(row):
        record = json.loads(row["data_json"])
        record.update(edit_id=row["edit_id"], status=row["status"], created_at=row["created_at"], approved_at=row["approved_at"])
        return record

    def edits(self):
        with self._lock:
            return [self._decode_edit(row) for row in self._db.execute("SELECT * FROM edits ORDER BY created_at,edit_id")]

    def _edit(self, edit_id):
        row = self._db.execute("SELECT * FROM edits WHERE edit_id=?", (edit_id,)).fetchone()
        if row is None:
            raise DomainError(f"Unknown correction: {edit_id}", 404)
        return self._decode_edit(row)

    def events(self):
        with self._lock:
            records = []
            for row in self._db.execute("SELECT * FROM events ORDER BY revision"):
                record = dict(row)
                record["details"] = json.loads(record.pop("details_json"))
                records.append(record)
            return records

    @staticmethod
    def _administrator(actor):
        if actor != "administrator":
            raise DomainError("Only the administrator workflow may mutate approved product information", 403)

    @staticmethod
    def _descendant(candidate, ancestor, by_id):
        current = candidate.get("supersedes")
        seen = set()
        while current and current not in seen:
            if current == ancestor["edit_id"]:
                return True
            seen.add(current)
            current = by_id.get(current, {}).get("supersedes")
        return False

    def _segments(self, edit, edits):
        segments = [_interval(edit)]
        by_id = {row["edit_id"]: row for row in edits}
        for descendant in edits:
            if descendant["status"] != "approved" or not self._descendant(descendant, edit, by_id):
                continue
            lo, hi = _interval(descendant)
            remaining = []
            for start, end in segments:
                if hi <= start or lo >= end:
                    remaining.append((start, end))
                else:
                    if start < lo:
                        remaining.append((start, lo))
                    if hi < end:
                        remaining.append((hi, end))
            segments = remaining
        return segments

    def _conflicts(self, edits):
        live = [edit for edit in edits if edit["status"] != "withdrawn"]
        by_id = {edit["edit_id"]: edit for edit in edits}
        conflicts = []
        for index, left in enumerate(live):
            for right in live[index + 1:]:
                if _scope(left) != _scope(right) or left["corrected_value"] == right["corrected_value"]:
                    continue
                # An explicit pending successor is an intentional future change,
                # not a contradictory competing authority with its predecessor.
                if self._descendant(left, right, by_id) or self._descendant(right, left, by_id):
                    continue
                for a in self._segments(left, edits):
                    for b in self._segments(right, edits):
                        lo, hi = max(a[0], b[0]), min(a[1], b[1])
                        if lo >= hi:
                            continue
                        conflicts.append({
                            "edit_ids": [left["edit_id"], right["edit_id"]],
                            "fact_id": left["fact_id"],
                            "reason": "Overlapping corrections endorse different values in the same exact scope",
                            "effective_from": date.fromordinal(lo).isoformat(),
                            "effective_to": None if hi == _INFINITY else date.fromordinal(hi).isoformat(),
                            "scope": dict(zip(("entity_id", "relation", "plan", "version"), _scope(left))),
                        })
        return conflicts

    def conflicts(self):
        with self._lock:
            return self._conflicts(self.edits())

    def _authoritative(self, fact, as_of, edits):
        ordinal = date.fromisoformat(as_of).toordinal()
        active = [edit for edit in edits if edit["fact_id"] == fact["fact_id"] and edit["status"] == "approved"
                  and any(start <= ordinal < end for start, end in self._segments(edit, edits))]
        values = {edit["corrected_value"] for edit in active}
        if len(values) > 1:
            raise DomainError("Approved facts have contradictory authorities; administrator resolution required", 409)
        return (next(iter(values)) if values else fact["value"]), active

    def _validate_correction(self, record, fact, edits):
        start, end = _interval(record)
        fact_start, fact_end = _interval(fact)
        if start < fact_start or end > fact_end:
            raise DomainError("Correction interval must stay within the selected fact's effective interval")
        try:
            Fact.model_validate({**fact, "value": record["corrected_value"]})
        except ValidationError as exc:
            raise DomainError(str(exc)) from exc
        current, _ = self._authoritative(fact, record["effective_from"], edits)
        predecessor_id = record.get("supersedes")
        if predecessor_id:
            predecessor = next((edit for edit in edits if edit["edit_id"] == predecessor_id), None)
            if predecessor is None:
                raise DomainError("Superseded correction does not exist", 404)
            if predecessor["status"] != "approved":
                raise DomainError("Supersession requires an approved predecessor", 409)
            if predecessor["fact_id"] != fact["fact_id"] or _scope(predecessor) != _scope(fact):
                raise DomainError("Supersession requires exactly the same fact, product, property, plan and version")
            predecessor_start, predecessor_end = _interval(predecessor)
            if start <= predecessor_start:
                raise DomainError("A successor must start later than its predecessor")
            if end > predecessor_end:
                raise DomainError("A successor must stay within its predecessor's effective interval")
            if not any(lo <= start < hi for lo, hi in self._segments(predecessor, edits)):
                raise DomainError("The selected predecessor is not active at the successor's effective date", 409)
            if record["original_value"] != predecessor["corrected_value"]:
                raise DomainError("original_value must match the approved predecessor's corrected value")
        if record["original_value"] != current:
            raise DomainError(f"original_value must match the authoritative value at effective_from ({current})", 409)

    def create_correction(self, correction, actor="administrator"):
        self._administrator(actor)
        try:
            correction = CorrectionCreate.model_validate(correction)
        except ValidationError as exc:
            raise DomainError(str(exc)) from exc
        record = correction.model_dump(mode="json")
        with self._transaction():
            fact = self._fact(record["fact_id"])
            self._validate_correction(record, fact, self.edits())
            for key in ("entity_id", "entity_name", "aliases", "relation", "unit", "plan", "version", "lineage_id"):
                record[key] = fact[key]
            edit_id = str(uuid4())
            self._db.execute(
                "INSERT INTO edits VALUES (?,?,?,?,?,?,?)",
                (edit_id, record["fact_id"], record["supersedes"], "draft", _json(record), _now(), None),
            )
            self._event("correction_created", actor, {"edit_id": edit_id, "fact_id": record["fact_id"]})
            return self._edit(edit_id)

    def approve(self, edit_id, actor="administrator"):
        self._administrator(actor)
        failure = None
        with self._transaction():
            edit = self._edit(edit_id)
            if edit["status"] == "approved":
                return edit
            if edit["status"] != "draft":
                raise DomainError("A withdrawn correction cannot be reactivated; create a new reviewed correction", 409)
            edits = self.edits()
            related = [conflict for conflict in self._conflicts(edits) if edit_id in conflict["edit_ids"]]
            if related:
                failure = DomainError("Contradictory overlapping corrections require administrator resolution; draft retained", 409)
            else:
                try:
                    self._validate_correction(edit, self._fact(edit["fact_id"]), edits)
                except DomainError as exc:
                    failure = DomainError(str(exc), 409)
            if failure:
                self._event("approval_rejected", actor, {"edit_id": edit_id, "reason": str(failure)})
            else:
                self._db.execute("UPDATE edits SET status='approved',approved_at=? WHERE edit_id=?", (_now(), edit_id))
                self._event("correction_approved", actor, {"edit_id": edit_id})
                result = self._edit(edit_id)
        if failure:
            raise failure
        return result

    def withdraw(self, edit_id, actor="administrator"):
        self._administrator(actor)
        with self._transaction():
            edit = self._edit(edit_id)
            if edit["status"] == "withdrawn":
                return edit
            edits = self.edits()
            by_id = {row["edit_id"]: row for row in edits}
            descendants = [row["edit_id"] for row in edits if row["status"] != "withdrawn" and self._descendant(row, edit, by_id)]
            if descendants:
                raise DomainError("Withdraw active descendants first: " + ", ".join(descendants), 409)
            self._db.execute("UPDATE edits SET status='withdrawn' WHERE edit_id=?", (edit_id,))
            self._event("correction_withdrawn", actor, {"edit_id": edit_id, "prior_status": edit["status"]})
            return self._edit(edit_id)

    def resolve_fact(self, fact_id, as_of):
        try:
            as_of = iso_date(as_of).isoformat()
        except ValueError as exc:
            raise DomainError(str(exc)) from exc
        with self.snapshot():
            fact = self._fact(fact_id)
            ordinal = date.fromisoformat(as_of).toordinal()
            start, end = _interval(fact)
            if not start <= ordinal < end:
                raise DomainError("No approved fact covers this product/version and requested date", 404)
            edits = self.edits()
            value, active = self._authoritative(fact, as_of, edits)
            relevant = [conflict for conflict in self._conflicts(edits) if conflict["fact_id"] == fact_id
                        and _interval(conflict)[0] <= ordinal < _interval(conflict)[1]]
            return {"fact": fact, "value": None if relevant else value, "edits": active,
                    "conflict": relevant or None, "revision": self.revision()}
