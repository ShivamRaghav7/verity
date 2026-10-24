"""Transactional conversation history and explicitly approved correction memory.

This store contains no model and never derives authoritative facts from a chat.
Conflict detection is deliberately exact: case/whitespace-normalized question
and scope identify a correction lineage. Semantic overlap is not certified.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
import uuid


class StoreError(Exception):
    def __init__(self, message, status=400, conflicts=None):
        super().__init__(message)
        self.status = status
        self.conflicts = list(conflicts or [])


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _text(value, name, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise StoreError(f"{name} must contain 1–{limit} characters")
    return value.strip()


def _key(text):
    return " ".join(text.casefold().split())


def _json(value):
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise StoreError("Evidence must be finite JSON data") from exc


class ChatStore:
    """One thread-safe connection with SQLite transactions and durable audit events."""

    def __init__(self, path):
        self.path = Path(path) if str(path) != ":memory:" else ":memory:"
        if isinstance(self.path, Path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), timeout=20, isolation_level=None,
                                   check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=20000")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            INSERT OR IGNORE INTO metadata VALUES ('knowledge_revision', '0');
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY, title TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, chat_id TEXT NOT NULL REFERENCES chats(id),
                position INTEGER NOT NULL, role TEXT NOT NULL
                    CHECK(role IN ('user','assistant','system')),
                content TEXT NOT NULL, evidence_json TEXT NOT NULL,
                knowledge_revision INTEGER NOT NULL, image_path TEXT,
                created_at TEXT NOT NULL, UNIQUE(chat_id, position)
            );
            CREATE TABLE IF NOT EXISTS corrections (
                id TEXT PRIMARY KEY, question TEXT NOT NULL, answer TEXT NOT NULL,
                source TEXT NOT NULL, scope TEXT NOT NULL, question_key TEXT NOT NULL,
                scope_key TEXT NOT NULL, lineage_id TEXT NOT NULL, version INTEGER NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('active','withdrawn','superseded')),
                supersedes TEXT REFERENCES corrections(id), superseded_by TEXT,
                approved_by TEXT NOT NULL, approved_at TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                knowledge_revision INTEGER NOT NULL,
                UNIQUE(question_key, scope_key, version)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_correction_per_key
                ON corrections(question_key, scope_key) WHERE status='active';
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                correction_id TEXT NOT NULL REFERENCES corrections(id),
                action TEXT NOT NULL, revision INTEGER NOT NULL,
                actor TEXT NOT NULL, created_at TEXT NOT NULL, detail_json TEXT NOT NULL
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
                yield self._db
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def get_revision(self):
        with self._lock:
            return self._revision(self._db)

    @property
    def revision(self):
        return self.get_revision()

    @staticmethod
    def _revision(db):
        return int(db.execute("SELECT value FROM metadata WHERE key='knowledge_revision'").fetchone()[0])

    def _bump(self, db):
        revision = self._revision(db) + 1
        db.execute("UPDATE metadata SET value=? WHERE key='knowledge_revision'", (str(revision),))
        return revision

    @staticmethod
    def _correction(row):
        if row is None:
            raise StoreError("Correction not found", 404)
        result = dict(row)
        result.pop("question_key", None)
        result.pop("scope_key", None)
        result["approved"] = True
        return result

    @staticmethod
    def _message(row):
        result = dict(row)
        result["evidence"] = json.loads(result.pop("evidence_json"))
        return result

    @staticmethod
    def _require_chat(db, chat_id):
        if not isinstance(chat_id, str):
            raise StoreError("Conversation identifier must be text")
        row = db.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
        if row is None:
            raise StoreError("Conversation not found", 404)
        return row

    def create_chat(self):
        now, chat_id = _now(), uuid.uuid4().hex
        with self._transaction() as db:
            db.execute("INSERT INTO chats VALUES (?, ?, ?, ?)", (chat_id, "New chat", now, now))
        return {"id": chat_id, "title": "New chat", "created_at": now, "updated_at": now}

    def list_chats(self):
        with self._lock:
            return [dict(row) for row in self._db.execute("SELECT * FROM chats ORDER BY updated_at DESC, id")]

    def get_chat(self, chat_id):
        with self._lock:
            chat = dict(self._require_chat(self._db, chat_id))
            chat["messages"] = self.list_messages(chat_id)
            return chat

    def list_messages(self, chat_id):
        with self._lock:
            self._require_chat(self._db, chat_id)
            return [self._message(row) for row in self._db.execute(
                "SELECT * FROM messages WHERE chat_id=? ORDER BY position", (chat_id,))]

    def add_message(self, chat_id, role, content, evidence=None, knowledge_revision=None, image_path=None):
        if role not in ("user", "assistant", "system"):
            raise StoreError("Message role must be user, assistant or system")
        if role == "user":
            content = _text(content, "Message", 16000)
        elif not isinstance(content, str) or len(content) > 16000:
            raise StoreError("Message must be text up to 16000 characters")
        if image_path is not None:
            image_path = _text(image_path, "Image path", 2048)
        encoded_evidence = _json([] if evidence is None else evidence)
        if len(encoded_evidence) > 200_000:
            raise StoreError("Evidence exceeds the storage limit")
        if knowledge_revision is not None and (isinstance(knowledge_revision, bool)
                or not isinstance(knowledge_revision, int) or knowledge_revision < 0):
            raise StoreError("Knowledge revision must be a nonnegative integer")
        now, message_id = _now(), uuid.uuid4().hex
        with self._transaction() as db:
            chat = self._require_chat(db, chat_id)
            position = db.execute("SELECT COALESCE(MAX(position), 0) + 1 FROM messages WHERE chat_id=?",
                                  (chat_id,)).fetchone()[0]
            revision = self._revision(db) if knowledge_revision is None else knowledge_revision
            db.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (message_id, chat_id, position, role, content, encoded_evidence,
                        revision, image_path, now))
            title = chat["title"]
            if role == "user" and not db.execute(
                    "SELECT 1 FROM messages WHERE chat_id=? AND role='user' AND id<>? LIMIT 1",
                    (chat_id, message_id)).fetchone():
                collapsed = " ".join(content.split())
                title = collapsed[:57] + "…" if len(collapsed) > 60 else collapsed
            db.execute("UPDATE chats SET title=?, updated_at=? WHERE id=?", (title, now, chat_id))
            row = db.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
        return self._message(row)

    def list_corrections(self):
        with self._lock:
            return [self._correction(row) for row in self._db.execute(
                "SELECT * FROM corrections ORDER BY created_at DESC, id")]

    def _event(self, db, correction_id, action, revision, now, detail):
        db.execute("INSERT INTO events(correction_id,action,revision,actor,created_at,detail_json) "
                   "VALUES (?,?,?,?,?,?)", (correction_id, action, revision,
                   "local_research_operator", now, _json(detail)))

    def publish_correction(self, question, answer, source, scope, supersedes=None):
        question = _text(question, "Question", 512)
        answer = _text(answer, "Answer", 2000)
        source = _text(source, "Source", 2048)
        scope = _text(scope, "Scope", 160)
        if supersedes is not None:
            supersedes = _text(supersedes, "Superseded correction identifier", 32)
        question_key, scope_key = _key(question), _key(scope)
        now, correction_id = _now(), uuid.uuid4().hex
        with self._transaction() as db:
            active = db.execute("SELECT * FROM corrections WHERE question_key=? AND scope_key=? AND status='active'",
                                (question_key, scope_key)).fetchone()
            if active is not None and supersedes != active["id"]:
                raise StoreError("An active correction already exists for this question and scope. "
                                 "Review it and explicitly replace it, or choose a different scope.",
                                 409, [self._correction(active)])
            if supersedes is not None and (active is None or supersedes != active["id"]):
                raise StoreError("Replacement must identify the current active correction in the same question and scope", 409)
            version = db.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM corrections WHERE question_key=? AND scope_key=?",
                                 (question_key, scope_key)).fetchone()[0]
            previous = db.execute("SELECT lineage_id FROM corrections WHERE question_key=? AND scope_key=? ORDER BY version LIMIT 1",
                                  (question_key, scope_key)).fetchone()
            lineage = previous["lineage_id"] if previous else correction_id
            revision = self._bump(db)
            if active is not None:
                db.execute("UPDATE corrections SET status='superseded', superseded_by=?, updated_at=?, knowledge_revision=? WHERE id=?",
                           (correction_id, now, revision, active["id"]))
                self._event(db, active["id"], "superseded", revision, now, {"superseded_by": correction_id})
            db.execute("INSERT INTO corrections VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (correction_id, question, answer, source, scope, question_key, scope_key,
                        lineage, version, "active", supersedes, None,
                        "local_research_operator", now, now, now, revision))
            self._event(db, correction_id, "published", revision, now,
                        {"supersedes": supersedes, "version": version,
                         "notice": "Operator approval does not independently verify factual truth"})
            row = db.execute("SELECT * FROM corrections WHERE id=?", (correction_id,)).fetchone()
        return self._correction(row)

    def withdraw(self, correction_id):
        if not isinstance(correction_id, str):
            raise StoreError("Correction identifier must be text")
        now = _now()
        with self._transaction() as db:
            row = db.execute("SELECT * FROM corrections WHERE id=?", (correction_id,)).fetchone()
            self._correction(row)
            if row["status"] != "active":
                return self._correction(row)
            revision = self._bump(db)
            db.execute("UPDATE corrections SET status='withdrawn', updated_at=?, knowledge_revision=? WHERE id=?",
                       (now, revision, correction_id))
            self._event(db, correction_id, "withdrawn", revision, now, {})
            row = db.execute("SELECT * FROM corrections WHERE id=?", (correction_id,)).fetchone()
        return self._correction(row)

    def restore(self, correction_id):
        if not isinstance(correction_id, str):
            raise StoreError("Correction identifier must be text")
        now = _now()
        with self._transaction() as db:
            row = db.execute("SELECT * FROM corrections WHERE id=?", (correction_id,)).fetchone()
            self._correction(row)
            if row["status"] == "active":
                return self._correction(row)
            active = db.execute("SELECT * FROM corrections WHERE question_key=? AND scope_key=? AND status='active'",
                                (row["question_key"], row["scope_key"])).fetchone()
            if active is not None:
                raise StoreError("Withdraw the active correction in this scope before restoring an older version",
                                 409, [self._correction(active)])
            revision = self._bump(db)
            previous_state = row["status"]
            db.execute("UPDATE corrections SET status='active', superseded_by=NULL, updated_at=?, knowledge_revision=?, "
                       "approved_at=?, approved_by='local_research_operator' WHERE id=?",
                       (now, revision, now, correction_id))
            self._event(db, correction_id, "restored", revision, now, {"previous_status": previous_state})
            row = db.execute("SELECT * FROM corrections WHERE id=?", (correction_id,)).fetchone()
        return self._correction(row)

    def list_events(self, correction_id=None):
        with self._lock:
            if correction_id is None:
                rows = self._db.execute("SELECT * FROM events ORDER BY sequence").fetchall()
            else:
                rows = self._db.execute("SELECT * FROM events WHERE correction_id=? ORDER BY sequence",
                                        (correction_id,)).fetchall()
            result = []
            for row in rows:
                event = dict(row)
                event["detail"] = json.loads(event.pop("detail_json"))
                result.append(event)
            return result
