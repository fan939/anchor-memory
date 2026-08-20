"""
Anchor Memory System — SQLite layer for graph-structured memory.

Handles: memory storage, tiered decay, synaptic edges (Hebbian learning),
emotion scoring, citation tracking, and graph operations.
"""

import json
import sqlite3
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone


def _utc_now() -> datetime:
    """Backward-compatible UTC clock without deprecated utcnow()."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


class AnchorDB:
    """SQLite storage with graph layer for memory synapses."""

    MAX_EDGE_WEIGHT = 10.0  # Synaptic saturation

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_tables()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_tables(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id   TEXT PRIMARY KEY,
                    text        TEXT NOT NULL,
                    timestamp   TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    last_used   TEXT,
                    tag         TEXT DEFAULT 'general',
                    tier        TEXT DEFAULT 'short',
                    pinned      INTEGER DEFAULT 0,
                    emotion_score REAL DEFAULT 0.5
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    source_id   TEXT NOT NULL,
                    target_id   TEXT NOT NULL,
                    weight      REAL DEFAULT 1.0,
                    created     TEXT NOT NULL,
                    last_fired  TEXT NOT NULL,
                    PRIMARY KEY (source_id, target_id),
                    FOREIGN KEY (source_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES memories(memory_id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
            # Comments table — memory as conversation space (design: Veille & 吱吱)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS comments (
                    comment_id  TEXT PRIMARY KEY,
                    memory_id   TEXT NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
                    content     TEXT NOT NULL,
                    author      TEXT DEFAULT 'ai',
                    reply_to    TEXT REFERENCES comments(comment_id),
                    read_by_ai  INTEGER DEFAULT 0,
                    read_by_human INTEGER DEFAULT 0,
                    created_at  TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_comments_memory ON comments(memory_id)")
            # Annotations — append-only notes on memories (design: Altair)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS annotations (
                    annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id   TEXT NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
                    text        TEXT NOT NULL,
                    created_at  TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_annotations_memory ON annotations(memory_id)")
            # Event log — immutable record of all operations (inspired by event sourcing)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id   TEXT,
                    event_type  TEXT NOT NULL,
                    detail      TEXT DEFAULT '',
                    created_at  TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_memory ON events(memory_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
            # Reflections are interpretations of event memories, never memory
            # rows themselves. Association tables keep provenance queryable and
            # prevent source events from being hard-deleted out from under them.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reflections (
                    reflection_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    selection_reason TEXT NOT NULL,
                    previous_interpretation TEXT NOT NULL DEFAULT '',
                    current_interpretation TEXT NOT NULL,
                    change_or_tension TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK(confidence >= 0.0 AND confidence <= 1.0),
                    open_questions TEXT NOT NULL DEFAULT '[]',
                    provenance TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'draft',
                    supersedes TEXT DEFAULT '',
                    retracted_by TEXT DEFAULT '',
                    cooldown_until TEXT DEFAULT '',
                    FOREIGN KEY (supersedes) REFERENCES reflections(reflection_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reflection_sources (
                    reflection_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    source_order INTEGER NOT NULL DEFAULT 0,
                    source_status_at_creation TEXT NOT NULL,
                    PRIMARY KEY (reflection_id, memory_id),
                    FOREIGN KEY (reflection_id) REFERENCES reflections(reflection_id) ON DELETE CASCADE,
                    FOREIGN KEY (memory_id) REFERENCES memories(memory_id) ON DELETE RESTRICT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reflection_sources_memory ON reflection_sources(memory_id)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reflection_evidence (
                    evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reflection_id TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK(kind IN ('support', 'counterevidence', 'outcome')),
                    content TEXT NOT NULL,
                    source_ref TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (reflection_id) REFERENCES reflections(reflection_id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reflection_evidence_reflection ON reflection_evidence(reflection_id)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_feedback (
                    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    relevant INTEGER NOT NULL CHECK(relevant IN (0, 1)),
                    reason TEXT NOT NULL DEFAULT '',
                    source_ref TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories(memory_id) ON DELETE RESTRICT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_feedback_memory ON memory_feedback(memory_id)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    input_summary TEXT NOT NULL,
                    output_summary TEXT NOT NULL,
                    outcome_status TEXT NOT NULL DEFAULT 'unknown',
                    provenance TEXT NOT NULL DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reflection_effects (
                    decision_id TEXT NOT NULL,
                    reflection_id TEXT NOT NULL,
                    effect_note TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (decision_id, reflection_id),
                    FOREIGN KEY (decision_id) REFERENCES decisions(decision_id) ON DELETE CASCADE,
                    FOREIGN KEY (reflection_id) REFERENCES reflections(reflection_id) ON DELETE RESTRICT
                )
            """)
            conn.commit()
        self._ensure_context_column()
        self._ensure_visual_column()
        self._ensure_provenance_columns()

    def _ensure_context_column(self):
        """Add context column if missing. text = search summary, context = full original."""
        with self._conn() as conn:
            try:
                conn.execute("SELECT context FROM memories LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE memories ADD COLUMN context TEXT DEFAULT ''")
                conn.commit()

    def _ensure_visual_column(self):
        """Add visual_embedding column if missing. For Anchor Vision integration."""
        with self._conn() as conn:
            try:
                conn.execute("SELECT visual_embedding FROM memories LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE memories ADD COLUMN visual_embedding TEXT DEFAULT ''")
                conn.commit()

    def _ensure_provenance_columns(self):
        """Add conservative, backward-compatible provenance metadata."""
        columns = {
            "perspective": "TEXT NOT NULL DEFAULT 'mixed'",
            "epistemic_status": "TEXT NOT NULL DEFAULT 'reported'",
            "confidence": "REAL NOT NULL DEFAULT 0.5",
            "source_turn": "TEXT DEFAULT ''",
            "created_at": "TEXT DEFAULT ''",
            "updated_at": "TEXT DEFAULT ''",
            "supersedes": "TEXT DEFAULT ''",
            "retracted_by": "TEXT DEFAULT ''",
            "memory_layer": "TEXT NOT NULL DEFAULT 'event'",
            "source_ref": "TEXT DEFAULT ''",
            "provenance": "TEXT NOT NULL DEFAULT '{}'",
        }
        with self._conn() as conn:
            existing = {
                row["name"] for row in conn.execute("PRAGMA table_info(memories)").fetchall()
            }
            for name, declaration in columns.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE memories ADD COLUMN {name} {declaration}")
            # Existing timestamps are the safest available creation time.
            conn.execute(
                "UPDATE memories SET created_at = timestamp "
                "WHERE created_at IS NULL OR created_at = ''"
            )
            conn.execute(
                "UPDATE memories SET updated_at = timestamp "
                "WHERE updated_at IS NULL OR updated_at = ''"
            )
            conn.execute(
                "UPDATE memories SET memory_layer = 'core' "
                "WHERE tier = 'core' AND memory_layer = 'event'"
            )

    # ── Event Log (immutable) ──

    def log_event(self, memory_id: str, event_type: str, detail: str = ""):
        """Log an immutable event. Types: created, updated, searched, connected, annotated, deleted, visual_stored."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO events (memory_id, event_type, detail, created_at) VALUES (?, ?, ?, ?)",
                (memory_id, event_type, detail, _utc_now_iso()),
            )
            conn.commit()

    def get_events(self, memory_id: str, limit: int = 50) -> list:
        """Get event history for a memory."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT event_id, event_type, detail, created_at FROM events "
                "WHERE memory_id = ? ORDER BY created_at DESC LIMIT ?",
                (memory_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_events(self, limit: int = 20, event_type: str = None) -> list:
        """Get recent events across all memories."""
        with self._conn() as conn:
            if event_type:
                rows = conn.execute(
                    "SELECT event_id, memory_id, event_type, detail, created_at FROM events "
                    "WHERE event_type = ? ORDER BY created_at DESC LIMIT ?",
                    (event_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT event_id, memory_id, event_type, detail, created_at FROM events "
                    "ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    # ── Annotations (append-only) ──

    def annotate(self, memory_id: str, text: str) -> int:
        """Add an annotation to a memory. Append-only — never delete or edit."""
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO annotations (memory_id, text, created_at) VALUES (?, ?, ?)",
                (memory_id, text, _utc_now_iso()),
            )
            conn.commit()
        self.log_event(memory_id, "annotated", text[:100])
        return cur.lastrowid

    def get_annotations(self, memory_id: str) -> list:
        """Get all annotations for a memory, oldest first."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT annotation_id, text, created_at FROM annotations "
                "WHERE memory_id = ? ORDER BY created_at ASC",
                (memory_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def search_annotations(self, query: str, limit: int = 5) -> list:
        """Search annotations text. Returns matching memory_ids."""
        words = query.strip().split()
        if not words:
            return []
        where = " AND ".join(["a.text LIKE ?"] * len(words))
        params = [f"%{w}%" for w in words]
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT a.memory_id, a.text, a.created_at FROM annotations a "
                f"WHERE {where} ORDER BY a.created_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Visual Embedding (Anchor Vision integration) ──

    def set_visual_embedding(self, memory_id: str, embedding_json: str):
        """Store a visual embedding (CLIP vector as JSON string) for a memory."""
        self._ensure_visual_column()
        with self._conn() as conn:
            conn.execute(
                "UPDATE memories SET visual_embedding = ? WHERE memory_id = ?",
                (embedding_json, memory_id),
            )
            conn.commit()

    def get_visual_embedding(self, memory_id: str) -> str:
        """Get visual embedding for a memory. Returns JSON string or empty."""
        self._ensure_visual_column()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT visual_embedding FROM memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        return row["visual_embedding"] if row and row["visual_embedding"] else ""

    def find_visual_memories(self) -> list:
        """Get all memories that have visual embeddings."""
        self._ensure_visual_column()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT memory_id, text, visual_embedding FROM memories "
                "WHERE visual_embedding != '' AND visual_embedding IS NOT NULL"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Memory CRUD ──

    def insert(self, memory_id: str, text: str, tag: str = "general",
               tier: str = "short", emotion_score: float = 0.5,
               context: str = "", perspective: str = "mixed",
               epistemic_status: str = "reported", confidence: float = 0.5,
               source_turn: str = "", supersedes: str = "",
               memory_layer: str = "event", source_ref: str = "",
               provenance: dict = None):
        """Insert a memory; re-inserting an existing id UPDATES it in place.

        First write = INSERT. Re-store of an existing id = UPSERT that updates
        content fields WITHOUT deleting the row.

        ⚠️ Was `INSERT OR REPLACE`, which on a key conflict DELETEs the old row
        then INSERTs — and the DELETE fires the edges table's ON DELETE CASCADE,
        wiping EVERY edge (Hebbian history + manual links) of any memory that
        gets re-stored. Verified: connect(A,B,2.0) → insert(A,...) → edge A↔B
        becomes None. Native UPSERT (ON CONFLICT DO UPDATE) updates fields with
        no DELETE, so edges survive. Two fields with independent lifecycles are
        PRESERVED via COALESCE (only the first write sets them):
          - timestamp:     a revision must not make an old memory look new
                           (would break recency ordering);
          - emotion_score: emotion propagation / dream-pass own it; a text edit
                           must not reset it.
        """
        if memory_layer not in {"event", "dynamic", "core"}:
            raise ValueError("memory_layer must be event, dynamic, or core")
        if memory_layer == "core" and epistemic_status != "confirmed":
            raise ValueError("core memories require epistemic_status='confirmed'")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self._ensure_context_column()
        now = _utc_now_iso()
        provenance_json = json.dumps(provenance or {}, ensure_ascii=False, sort_keys=True)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    memory_id, text, timestamp, tag, tier, emotion_score, context,
                    perspective, epistemic_status, confidence, source_turn,
                    created_at, updated_at, supersedes, memory_layer,
                    source_ref, provenance
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    text          = excluded.text,
                    tag           = excluded.tag,
                    tier          = excluded.tier,
                    context       = excluded.context,
                    perspective   = excluded.perspective,
                    epistemic_status = excluded.epistemic_status,
                    confidence    = excluded.confidence,
                    source_turn   = excluded.source_turn,
                    updated_at    = excluded.updated_at,
                    supersedes    = excluded.supersedes,
                    memory_layer  = excluded.memory_layer,
                    source_ref    = excluded.source_ref,
                    provenance    = excluded.provenance,
                    timestamp     = COALESCE(memories.timestamp, excluded.timestamp),
                    emotion_score = COALESCE(memories.emotion_score, excluded.emotion_score)
                """,
                (memory_id, text, now, tag, tier, emotion_score, context,
                 perspective, epistemic_status, confidence, source_turn,
                 now, now, supersedes, memory_layer, source_ref, provenance_json),
            )
            conn.commit()
        self.log_event(memory_id, "created", f"tag={tag} tier={tier}")

    def retract(self, memory_id: str, retracted_by: str = "") -> bool:
        """Mark a memory retracted while preserving it for audit."""
        with self._conn() as conn:
            cursor = conn.execute(
                "UPDATE memories SET epistemic_status = 'retracted', "
                "retracted_by = ?, updated_at = ? WHERE memory_id = ?",
                (retracted_by, _utc_now_iso(), memory_id),
            )
        if cursor.rowcount:
            self.log_event(memory_id, "retracted", f"retracted_by={retracted_by}")
        return bool(cursor.rowcount)

    def get(self, memory_id: str) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["provenance"] = self._json_value(result.get("provenance"), {})
        return result

    def delete(self, memory_id: str):
        with self._conn() as conn:
            # Application-level cascade: some SQLite builds (notably Windows
            # default) silently fail to enforce PRAGMA foreign_keys = ON, leaving
            # ON DELETE CASCADE inert and raising FK constraint violations later
            # when consolidate/dream-pass runs. Belt-and-suspenders: delete edges
            # explicitly before the memory itself. No-op on platforms where the
            # cascade fired.
            conn.execute(
                "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
                (memory_id, memory_id),
            )
            conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
            conn.commit()
        self.log_event(memory_id, "deleted")

    def list_all(self, limit: int = 50, offset: int = 0) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT memory_id, text, timestamp, tag, tier, memory_layer, "
                "perspective, epistemic_status, confidence, source_ref FROM memories "
                "ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            if "provenance" in item:
                item["provenance"] = self._json_value(item.get("provenance"), {})
            result.append(item)
        return result

    @staticmethod
    def _json_value(value, default):
        if value is None or value == "":
            return default
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default

    # ── Reflection: append-only interpretation and decision provenance ──

    def create_reflection(
        self, reflection_id: str, source_event_ids: list,
        trigger_type: str, selection_reason: str,
        previous_interpretation: str, current_interpretation: str,
        change_or_tension: str, confidence: float,
        open_questions: list, counterevidence: list,
        provenance: dict, status: str = "draft", supersedes: str = "",
        cooldown_until: str = "",
    ) -> dict:
        """Persist a reflection only after validating every event source."""
        if not source_event_ids:
            raise ValueError("source_event_ids must contain at least one event memory")
        unique_ids = list(dict.fromkeys(source_event_ids))
        now = _utc_now_iso()
        with self._conn() as conn:
            placeholders = ",".join("?" for _ in unique_ids)
            rows = conn.execute(
                f"SELECT memory_id, memory_layer, epistemic_status FROM memories "
                f"WHERE memory_id IN ({placeholders})", unique_ids,
            ).fetchall()
            found = {row["memory_id"]: row for row in rows}
            missing = [memory_id for memory_id in unique_ids if memory_id not in found]
            if missing:
                raise ValueError(f"Unknown source event IDs: {', '.join(missing)}")
            non_events = [
                memory_id for memory_id in unique_ids
                if found[memory_id]["memory_layer"] != "event"
            ]
            if non_events:
                raise ValueError(
                    "Reflections may only reference event-layer memories: "
                    + ", ".join(non_events)
                )
            if supersedes:
                prior = conn.execute(
                    "SELECT reflection_id FROM reflections WHERE reflection_id = ?",
                    (supersedes,),
                ).fetchone()
                if not prior:
                    raise ValueError(f"Unknown superseded reflection: {supersedes}")
            conn.execute(
                """
                INSERT INTO reflections (
                    reflection_id, created_at, updated_at, trigger_type,
                    selection_reason, previous_interpretation,
                    current_interpretation, change_or_tension, confidence,
                    open_questions, provenance, status, supersedes,
                    cooldown_until
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reflection_id, now, now, trigger_type, selection_reason,
                    previous_interpretation, current_interpretation,
                    change_or_tension, confidence,
                    json.dumps(open_questions or [], ensure_ascii=False),
                    json.dumps(provenance or {}, ensure_ascii=False, sort_keys=True),
                    status, supersedes or None, cooldown_until,
                ),
            )
            for order, memory_id in enumerate(unique_ids):
                conn.execute(
                    "INSERT INTO reflection_sources "
                    "(reflection_id, memory_id, source_order, source_status_at_creation) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        reflection_id, memory_id, order,
                        found[memory_id]["epistemic_status"],
                    ),
                )
            for item in counterevidence or []:
                content = item.get("content", "") if isinstance(item, dict) else str(item)
                source_ref = item.get("source_ref", "") if isinstance(item, dict) else ""
                if content.strip():
                    conn.execute(
                        "INSERT INTO reflection_evidence "
                        "(reflection_id, kind, content, source_ref, created_at) "
                        "VALUES (?, 'counterevidence', ?, ?, ?)",
                        (reflection_id, content.strip(), source_ref, now),
                    )
        self.log_event(reflection_id, "reflection_created", f"trigger={trigger_type}")
        return self.get_reflection(reflection_id)

    def get_reflection(self, reflection_id: str) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM reflections WHERE reflection_id = ?",
                (reflection_id,),
            ).fetchone()
            if not row:
                return None
            sources = conn.execute(
                """
                SELECT rs.memory_id, rs.source_order, rs.source_status_at_creation,
                       m.epistemic_status AS current_source_status,
                       m.text AS source_text, m.source_ref, m.provenance
                FROM reflection_sources rs
                JOIN memories m ON m.memory_id = rs.memory_id
                WHERE rs.reflection_id = ? ORDER BY rs.source_order
                """,
                (reflection_id,),
            ).fetchall()
            evidence = conn.execute(
                "SELECT evidence_id, kind, content, source_ref, created_at "
                "FROM reflection_evidence WHERE reflection_id = ? "
                "ORDER BY evidence_id",
                (reflection_id,),
            ).fetchall()
            decisions = conn.execute(
                "SELECT decision_id, effect_note, created_at FROM reflection_effects "
                "WHERE reflection_id = ? ORDER BY created_at",
                (reflection_id,),
            ).fetchall()
        result = dict(row)
        result["source_event_ids"] = [source["memory_id"] for source in sources]
        result["sources"] = []
        for source in sources:
            item = dict(source)
            item["provenance"] = self._json_value(item.get("provenance"), {})
            result["sources"].append(item)
        result["open_questions"] = self._json_value(result.get("open_questions"), [])
        result["provenance"] = self._json_value(result.get("provenance"), {})
        result["evidence"] = [dict(item) for item in evidence]
        result["counterevidence"] = [
            item for item in result["evidence"] if item["kind"] == "counterevidence"
        ]
        result["affected_decision_ids"] = [item["decision_id"] for item in decisions]
        result["decision_effects"] = [dict(item) for item in decisions]
        result["source_integrity"] = (
            "retracted_source" if any(
                source["current_source_status"] == "retracted" for source in sources
            ) else "intact"
        )
        return result

    def get_reflections_for_event(self, memory_id: str, limit: int = 5) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT r.reflection_id FROM reflections r "
                "JOIN reflection_sources rs ON rs.reflection_id = r.reflection_id "
                "WHERE rs.memory_id = ? ORDER BY r.created_at DESC LIMIT ?",
                (memory_id, limit),
            ).fetchall()
        return [self.get_reflection(row["reflection_id"]) for row in rows]

    def record_memory_feedback(
        self, memory_id: str, relevant: bool, reason: str = "", source_ref: str = "",
    ) -> dict:
        if not self.get(memory_id):
            raise ValueError(f"Unknown memory: {memory_id}")
        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO memory_feedback "
                "(memory_id, relevant, reason, source_ref, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (memory_id, int(bool(relevant)), reason, source_ref, _utc_now_iso()),
            )
        return {
            "feedback_id": cursor.lastrowid,
            "memory_id": memory_id,
            "relevant": bool(relevant),
            "deleted": False,
        }

    def get_memory_feedback(self, memory_id: str) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT feedback_id, relevant, reason, source_ref, created_at "
                "FROM memory_feedback WHERE memory_id = ? ORDER BY feedback_id",
                (memory_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["relevant"] = bool(item["relevant"])
            result.append(item)
        return result

    def get_feedback_penalty(self, memory_id: str) -> float:
        """Small bounded rank penalty; a single report never destroys data."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT SUM(CASE WHEN relevant = 0 THEN 1 ELSE 0 END) AS negative, "
                "SUM(CASE WHEN relevant = 1 THEN 1 ELSE 0 END) AS positive "
                "FROM memory_feedback WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        negative = row["negative"] or 0
        positive = row["positive"] or 0
        return min(max(negative - positive, 0) * 0.01, 0.10)

    def search_reflections(
        self, query: str = "", source_event_id: str = "",
        status: str = "", trigger_type: str = "", limit: int = 10,
    ) -> list:
        clauses = []
        params = []
        if query.strip():
            clauses.append(
                "(r.selection_reason LIKE ? OR r.previous_interpretation LIKE ? "
                "OR r.current_interpretation LIKE ? OR r.change_or_tension LIKE ?)"
            )
            like = f"%{query.strip()}%"
            params.extend([like, like, like, like])
        if source_event_id:
            clauses.append(
                "EXISTS (SELECT 1 FROM reflection_sources rs "
                "WHERE rs.reflection_id = r.reflection_id AND rs.memory_id = ?)"
            )
            params.append(source_event_id)
        if status:
            clauses.append("r.status = ?")
            params.append(status)
        if trigger_type:
            clauses.append("r.trigger_type = ?")
            params.append(trigger_type)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT r.reflection_id FROM reflections r" + where
                + " ORDER BY r.created_at DESC LIMIT ?",
                params + [max(1, min(limit, 100))],
            ).fetchall()
        return [self.get_reflection(row["reflection_id"]) for row in rows]

    def list_reflection_candidate_rows(self, limit: int = 50) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT m.*, COUNT(rs.reflection_id) AS reflection_count,
                       MAX(r.created_at) AS last_reflection_at,
                       MAX(r.cooldown_until) AS cooldown_until
                FROM memories m
                LEFT JOIN reflection_sources rs ON rs.memory_id = m.memory_id
                LEFT JOIN reflections r ON r.reflection_id = rs.reflection_id
                    AND r.status != 'abandoned'
                WHERE m.memory_layer = 'event'
                  AND m.epistemic_status != 'retracted'
                GROUP BY m.memory_id
                ORDER BY m.timestamp DESC LIMIT ?
                """,
                (max(1, min(limit, 500)),),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["provenance"] = self._json_value(item.get("provenance"), {})
            result.append(item)
        return result

    def reflection_opportunity_stats(self) -> dict:
        """Count new event seeds and active days since the latest Reflection."""
        with self._conn() as conn:
            latest_row = conn.execute(
                "SELECT MAX(created_at) AS latest FROM reflections "
                "WHERE status != 'abandoned'"
            ).fetchone()
            latest = latest_row["latest"] or ""
            if latest:
                row = conn.execute(
                    "SELECT COUNT(*) AS event_count, "
                    "COUNT(DISTINCT substr(timestamp, 1, 10)) AS active_days "
                    "FROM memories WHERE memory_layer = 'event' "
                    "AND epistemic_status != 'retracted' AND timestamp > ?",
                    (latest.replace("Z", ""),),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS event_count, "
                    "COUNT(DISTINCT substr(timestamp, 1, 10)) AS active_days "
                    "FROM memories WHERE memory_layer = 'event' "
                    "AND epistemic_status != 'retracted'"
                ).fetchone()
        return {
            "latest_reflection_at": latest,
            "new_event_count": row["event_count"],
            "active_days_since_reflection": row["active_days"],
        }

    def append_reflection_evidence(
        self, reflection_id: str, kind: str, content: str, source_ref: str = "",
    ) -> dict:
        if kind not in {"support", "counterevidence", "outcome"}:
            raise ValueError("kind must be support, counterevidence, or outcome")
        if not content or not content.strip():
            raise ValueError("evidence content cannot be empty")
        if not self.get_reflection(reflection_id):
            raise ValueError(f"Unknown reflection: {reflection_id}")
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO reflection_evidence "
                "(reflection_id, kind, content, source_ref, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (reflection_id, kind, content.strip(), source_ref, _utc_now_iso()),
            )
            conn.execute(
                "UPDATE reflections SET updated_at = ? WHERE reflection_id = ?",
                (_utc_now_iso(), reflection_id),
            )
        return self.get_reflection(reflection_id)

    def record_reflection_effect(
        self, decision_id: str, reflection_ids: list, input_summary: str,
        output_summary: str, effect_note: str, outcome_status: str = "unknown",
        provenance: dict = None,
    ) -> dict:
        if not reflection_ids:
            raise ValueError("reflection_ids must list reflections actually used")
        required_provenance = {"model", "thread_id", "context_ref", "extractor_version"}
        if not isinstance(provenance, dict):
            raise ValueError("provenance must be an object")
        missing_provenance = sorted(
            key for key in required_provenance if not provenance.get(key)
        )
        if missing_provenance:
            raise ValueError(
                "decision provenance is missing: " + ", ".join(missing_provenance)
            )
        if not input_summary.strip() or not output_summary.strip() or not effect_note.strip():
            raise ValueError("decision summaries and effect_note cannot be empty")
        unique_ids = list(dict.fromkeys(reflection_ids))
        now = _utc_now_iso()
        with self._conn() as conn:
            placeholders = ",".join("?" for _ in unique_ids)
            rows = conn.execute(
                f"SELECT reflection_id FROM reflections WHERE reflection_id IN ({placeholders})",
                unique_ids,
            ).fetchall()
            found = {row["reflection_id"] for row in rows}
            missing = [item for item in unique_ids if item not in found]
            if missing:
                raise ValueError(f"Unknown reflection IDs: {', '.join(missing)}")
            conn.execute(
                "INSERT INTO decisions "
                "(decision_id, created_at, input_summary, output_summary, outcome_status, provenance) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    decision_id, now, input_summary, output_summary, outcome_status,
                    json.dumps(provenance or {}, ensure_ascii=False, sort_keys=True),
                ),
            )
            for reflection_id in unique_ids:
                conn.execute(
                    "INSERT INTO reflection_effects "
                    "(decision_id, reflection_id, effect_note, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (decision_id, reflection_id, effect_note, now),
                )
                conn.execute(
                    "UPDATE reflections SET status = 'adopted', updated_at = ? "
                    "WHERE reflection_id = ? AND status IN ('draft', 'tentative')",
                    (now, reflection_id),
                )
        return {
            "decision_id": decision_id,
            "used_reflection_ids": unique_ids,
            "outcome_status": outcome_status,
        }

    def retract_reflection(self, reflection_id: str, retracted_by: str = "") -> bool:
        with self._conn() as conn:
            cursor = conn.execute(
                "UPDATE reflections SET status = 'abandoned', retracted_by = ?, "
                "updated_at = ? WHERE reflection_id = ?",
                (retracted_by, _utc_now_iso(), reflection_id),
            )
        if cursor.rowcount:
            self.log_event(reflection_id, "reflection_retracted", retracted_by)
        return bool(cursor.rowcount)

    def _tokenize_query(self, query: str) -> list:
        """Tokenize a search query for multi-keyword matching.

        Uses jieba (Chinese segmentation) if installed — handles Chinese without
        spaces by recognizing word boundaries via prefix dictionary. Mixed
        Chinese/English text is also handled correctly.

        Falls back to whitespace+punctuation split if jieba is not installed.
        Single Chinese characters are kept (semantically meaningful in CJK);
        non-Chinese tokens require >= 2 chars to filter out noise like "I", "a".
        """
        import re
        chinese_re = re.compile(r'[一-鿿]')

        try:
            import jieba
            raw_tokens = list(jieba.cut(query))
        except ImportError:
            raw_tokens = re.split(r'[\s,;:!?/\-+()]+', query)

        keywords = []
        for t in raw_tokens:
            t = t.strip()
            if not t:
                continue
            if not re.search(r'\w|[一-鿿]', t):
                continue
            if chinese_re.search(t) or len(t) >= 2:
                keywords.append(t)
        return keywords

    def keyword_search(self, query: str, limit: int = 5, tag: str = None) -> list:
        """Search memories + annotations by keyword.

        Tokenizes query (Chinese-aware via jieba if installed), then runs
        multi-keyword OR LIKE matching. A search like "记忆质量问题" gets
        segmented to ["记忆", "质量", "问题"] and any memory containing any
        of those tokens matches.
        """
        keywords = self._tokenize_query(query)
        if not keywords:
            # Fallback: treat whole query as a single token
            keywords = [query.strip()] if query.strip() else []
            if not keywords:
                return []

        like_clauses = " OR ".join(["text LIKE ?"] * len(keywords))
        like_params = [f"%{kw}%" for kw in keywords]

        with self._conn() as conn:
            if tag:
                rows = conn.execute(
                    f"SELECT memory_id, text, timestamp, tag FROM memories "
                    f"WHERE ({like_clauses}) AND tag = ? LIMIT ?",
                    like_params + [tag, limit]
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT memory_id, text, timestamp, tag FROM memories "
                    f"WHERE ({like_clauses}) LIMIT ?",
                    like_params + [limit]
                ).fetchall()
            results = [dict(r) for r in rows]
            found_ids = {r["memory_id"] for r in results}

            # Also search in annotations
            ann_like = " OR ".join(["a.text LIKE ?"] * len(keywords))
            ann_rows = conn.execute(
                f"SELECT DISTINCT a.memory_id FROM annotations a "
                f"WHERE ({ann_like}) LIMIT ?",
                like_params + [limit]
            ).fetchall()
            for ar in ann_rows:
                mid = ar["memory_id"]
                if mid not in found_ids:
                    mem = conn.execute(
                        "SELECT memory_id, text, timestamp, tag FROM memories "
                        "WHERE memory_id = ?", (mid,)
                    ).fetchone()
                    if mem:
                        results.append(dict(mem))
                        found_ids.add(mid)

        return results[:limit]

    # ── Tier management ──

    def set_tier(self, memory_id: str, tier: str):
        with self._conn() as conn:
            conn.execute("UPDATE memories SET tier = ? WHERE memory_id = ?", (tier, memory_id))
            conn.commit()

    def get_expired_short_ids(self, days: int = 14) -> list:
        """Return expired short-tier IDs without changing either store."""
        cutoff = (_utc_now() - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT memory_id FROM memories "
                "WHERE tier = 'short' AND timestamp < ? ORDER BY memory_id",
                (cutoff,)
            ).fetchall()
        return [row["memory_id"] for row in rows]

    def decay_short(self, days: int = 14) -> int:
        """Delete expired SQLite rows (legacy low-level API).

        New code should enumerate with ``get_expired_short_ids`` and delete
        through ``AnchorMemory.delete`` so Chroma is updated as well.
        """
        expired_ids = self.get_expired_short_ids(days=days)
        for memory_id in expired_ids:
            self.delete(memory_id)
        return len(expired_ids)

    # ── Citation tracking ──

    def get_citation_count(self, memory_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT usage_count FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return row["usage_count"] if row else 0

    def cite(self, memory_id: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE memories SET usage_count = usage_count + 1, last_used = ? WHERE memory_id = ?",
                (_utc_now_iso(), memory_id),
            )
            conn.commit()

    # ── Emotion scoring ──

    def get_context(self, memory_id: str) -> str:
        """Get the full context field for a memory."""
        self._ensure_context_column()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT context FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return row["context"] if row and row["context"] else ""

    def get_emotion_score(self, memory_id: str) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT emotion_score FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        if row and row["emotion_score"] is not None:
            return row["emotion_score"]
        return 0.5

    def set_emotion_score(self, memory_id: str, score: float):
        with self._conn() as conn:
            conn.execute(
                "UPDATE memories SET emotion_score = ? WHERE memory_id = ?",
                (max(0.0, min(1.0, score)), memory_id)
            )
            conn.commit()

    def equalize_emotion_scores(self, nudge: float = 0.05, threshold: float = 0.2) -> int:
        """Bidirectional emotion score equilibration across connected memories."""
        updated = 0
        with self._conn() as conn:
            memories = conn.execute(
                "SELECT memory_id, emotion_score FROM memories WHERE emotion_score IS NOT NULL"
            ).fetchall()

            for m in memories:
                mid = m["memory_id"]
                my_score = (m["emotion_score"] if m["emotion_score"] is not None
                            else 0.5)

                neighbors = conn.execute("""
                    SELECT m.emotion_score FROM memories m
                    INNER JOIN edges e ON (e.target_id = m.memory_id AND e.source_id = ?)
                       OR (e.source_id = m.memory_id AND e.target_id = ?)
                    WHERE m.emotion_score IS NOT NULL AND e.weight >= 0.5
                """, (mid, mid)).fetchall()

                if not neighbors:
                    continue

                avg_neighbor = sum(
                    n["emotion_score"] if n["emotion_score"] is not None else 0.5
                    for n in neighbors
                ) / len(neighbors)
                diff = avg_neighbor - my_score

                if abs(diff) > threshold:
                    new_score = my_score + nudge * (1 if diff > 0 else -1)
                    new_score = max(0.0, min(1.0, new_score))
                    conn.execute(
                        "UPDATE memories SET emotion_score = ? WHERE memory_id = ?",
                        (new_score, mid)
                    )
                    updated += 1

            conn.commit()
        return updated

    # ── Graph layer: synaptic edges ──

    def _upsert_edge(self, conn, source_id: str, target_id: str,
                     weight: float, now: str):
        # Self-loop guard: a memory must never have an edge to itself. Without
        # this, connect(A,A) writes an A→A edge and get_neighbors(A) returns A
        # as its own neighbor (associative recall would re-feed a memory to
        # itself). Dedup-merge edge migration is another source (a survivor↔dup
        # edge becomes survivor→survivor). All edge writes funnel through here,
        # so one guard covers every path. source == target → no-op.
        if source_id == target_id:
            return
        conn.execute("""
            INSERT INTO edges (source_id, target_id, weight, created, last_fired)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id) DO UPDATE SET
                weight = MIN(edges.weight + excluded.weight, ?),
                last_fired = excluded.last_fired
        """, (source_id, target_id, weight, now, now, self.MAX_EDGE_WEIGHT))

    def connect(self, source_id: str, target_id: str, weight: float = 1.0):
        self.log_event(source_id, "connected", f"to={target_id} weight={weight}")
        """Create or strengthen a bidirectional edge between memories."""
        now = _utc_now_iso()
        with self._conn() as conn:
            self._upsert_edge(conn, source_id, target_id, weight, now)
            self._upsert_edge(conn, target_id, source_id, weight, now)
            conn.commit()

    def connect_batch(self, pairs: list, weight: float = 0.2):
        """Batch connect pairs of memories (for Hebbian learning)."""
        now = _utc_now_iso()
        with self._conn() as conn:
            for source_id, target_id in pairs:
                self._upsert_edge(conn, source_id, target_id, weight, now)
                self._upsert_edge(conn, target_id, source_id, weight, now)
            conn.commit()

    def get_neighbors(self, memory_id: str, min_weight: float = 0.5,
                      limit: int = 5) -> list:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT target_id as memory_id, weight FROM edges
                WHERE source_id = ? AND weight >= ?
                ORDER BY weight DESC LIMIT ?
            """, (memory_id, min_weight, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_edge_weight(self, source_id: str, target_id: str):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT weight FROM edges WHERE source_id = ? AND target_id = ?",
                (source_id, target_id)
            ).fetchone()
        return row["weight"] if row else None

    def migrate_edges(self, from_id: str, to_id: str) -> int:
        """Move all of `from_id`'s edges onto `to_id` (the survivor of a merge).
        Each (from_id → X) becomes (to_id → X) and (X → from_id) becomes
        (X → to_id) via _upsert_edge, which handles the tricky cases for free:
        colliding edges saturate-add (MIN(sum, MAX_EDGE_WEIGHT)); a self-loop
        (an existing to_id↔from_id edge becoming to_id→to_id) is dropped by the
        _upsert_edge self-loop guard. The old from_id rows are then removed.
        Returns the number of source edge rows migrated."""
        now = _utc_now_iso()
        with self._conn() as conn:
            outgoing = conn.execute(
                "SELECT target_id, weight FROM edges WHERE source_id = ?", (from_id,)
            ).fetchall()
            incoming = conn.execute(
                "SELECT source_id, weight FROM edges WHERE target_id = ?", (from_id,)
            ).fetchall()
            migrated = 0
            for r in outgoing:
                self._upsert_edge(conn, to_id, r["target_id"], r["weight"], now)
                migrated += 1
            for r in incoming:
                self._upsert_edge(conn, r["source_id"], to_id, r["weight"], now)
                migrated += 1
            conn.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?",
                         (from_id, from_id))
            conn.commit()
        return migrated

    def decay_edges(self, min_weight: float = 0.1, decay_factor: float = 0.9) -> int:
        """Weaken all edges by decay_factor. Delete edges below min_weight."""
        with self._conn() as conn:
            conn.execute("UPDATE edges SET weight = weight * ?", (decay_factor,))
            cursor = conn.execute("DELETE FROM edges WHERE weight < ?", (min_weight,))
            conn.commit()
        return cursor.rowcount

    def decay_strong_edges(self, min_weight: float = 1.5, decay_factor: float = 0.95) -> int:
        """Slowly decay strong manual edges so they don't permanently dominate."""
        with self._conn() as conn:
            cursor = conn.execute(
                "UPDATE edges SET weight = weight * ? WHERE weight >= ?",
                (decay_factor, min_weight)
            )
            conn.commit()
        return cursor.rowcount

    def decay_edges_once(self, strong_threshold: float = 1.5,
                         weak_decay_factor: float = 0.9,
                         strong_decay_factor: float = 0.95,
                         prune_below: float = 0.1) -> dict:
        """Decay each edge exactly once using its pre-pass strength bucket."""
        with self._conn() as conn:
            weak_count = conn.execute(
                "SELECT COUNT(*) AS n FROM edges WHERE weight < ?",
                (strong_threshold,),
            ).fetchone()["n"]
            strong_count = conn.execute(
                "SELECT COUNT(*) AS n FROM edges WHERE weight >= ?",
                (strong_threshold,),
            ).fetchone()["n"]
            conn.execute(
                "UPDATE edges SET weight = CASE "
                "WHEN weight >= ? THEN weight * ? ELSE weight * ? END",
                (strong_threshold, strong_decay_factor, weak_decay_factor),
            )
            deleted = conn.execute(
                "DELETE FROM edges WHERE weight < ?", (prune_below,)
            ).rowcount
            conn.commit()
        return {
            "weak_decayed": weak_count,
            "strong_decayed": strong_count,
            "pruned": deleted,
        }

    # ── Pinning ──

    def pin(self, memory_id: str):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT memory_layer, epistemic_status FROM memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"memory not found: {memory_id}")
            if row["memory_layer"] != "core" or row["epistemic_status"] != "confirmed":
                raise ValueError(
                    "only confirmed core memories can be pinned; explicitly confirm "
                    "and store the core memory before pinning"
                )
            conn.execute("UPDATE memories SET pinned = 1 WHERE memory_id = ?", (memory_id,))
            conn.commit()

    def unpin(self, memory_id: str):
        with self._conn() as conn:
            conn.execute("UPDATE memories SET pinned = 0 WHERE memory_id = ?", (memory_id,))
            conn.commit()

    def get_pinned(self) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT memory_id, text, timestamp, tag FROM memories WHERE pinned = 1"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Comments: memory as conversation space (design: Veille & 吱吱) ──

    def insert_comment(self, memory_id: str, content: str,
                       author: str = "ai", reply_to: str = None) -> str:
        import uuid
        comment_id = f"comment_{uuid.uuid4().hex[:12]}"
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO comments (comment_id, memory_id, content, author, reply_to, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (comment_id, memory_id, content, author, reply_to,
                 _utc_now_iso()),
            )
            conn.commit()
        return comment_id

    def get_comments(self, memory_id: str) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM comments WHERE memory_id = ? ORDER BY created_at",
                (memory_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_unread_comments(self, reader: str = "ai") -> list:
        col = "read_by_ai" if reader == "ai" else "read_by_human"
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT c.*, m.text as memory_text "
                f"FROM comments c JOIN memories m ON c.memory_id = m.memory_id "
                f"WHERE c.{col} = 0 ORDER BY c.created_at",
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_comments_read(self, comment_ids: list, reader: str = "ai"):
        col = "read_by_ai" if reader == "ai" else "read_by_human"
        with self._conn() as conn:
            for cid in comment_ids:
                conn.execute(
                    f"UPDATE comments SET {col} = 1 WHERE comment_id = ?", (cid,)
                )
            conn.commit()

    # ── Wakeup: one-call cold start (design: Veille & 吱吱) ──

    def wakeup(self, n_high_emotion: int = 5, n_random: int = 2,
               high_emotion_days: int = 3, n_recent: int = 5) -> dict:
        """Gather everything needed for cold start in one call.

        Returns pinned + recent + recent high-emotion + random old + unread comments.
        Design principle: rules live here, not in external config.

        Note on recent: pulled by timestamp DESC with NO emotion filter.
        Emotion-sorted "recent" hides calm-but-important events — yesterday's
        quiet decision (emotion 0.4) loses to any intense moment in the same
        window, and "what happened yesterday" becomes invisible at cold start.
        Recency must be its own axis. high_emotion excludes ids already in
        recent, so the two blocks never duplicate.

        Note on random_old: these are surfaced without touch() — they don't
        increment usage_count or update last_used. Any Hebbian edges created
        from co-activation with random memories will be pruned by dream pass
        if not reinforced. This is intentional: temporary connections that
        don't get reinforced fade naturally. Dream pass is the cleanup.
        """
        self._ensure_context_column()
        cutoff = (_utc_now() - timedelta(days=high_emotion_days)).isoformat()

        with self._conn() as conn:
            pinned = conn.execute(
                "SELECT memory_id, text, tag, emotion_score, context FROM memories "
                "WHERE pinned = 1 ORDER BY timestamp"
            ).fetchall()

            recent = conn.execute(
                "SELECT memory_id, text, tag, emotion_score, timestamp, context FROM memories "
                "WHERE pinned = 0 ORDER BY timestamp DESC LIMIT ?",
                (n_recent,),
            ).fetchall() if n_recent > 0 else []
            recent_ids = [r["memory_id"] for r in recent]

            exclude = " AND memory_id NOT IN (%s)" % ",".join("?" * len(recent_ids)) \
                if recent_ids else ""
            high_emotion = conn.execute(
                f"SELECT memory_id, text, tag, emotion_score, timestamp, context FROM memories "
                f"WHERE timestamp >= ? AND pinned = 0{exclude} "
                f"ORDER BY emotion_score DESC LIMIT ?",
                [cutoff] + recent_ids + [n_high_emotion],
            ).fetchall()

            random_old = conn.execute(
                "SELECT memory_id, text, tag, emotion_score, timestamp, context FROM memories "
                "WHERE timestamp < ? AND pinned = 0 "
                "ORDER BY RANDOM() LIMIT ?",
                (cutoff, n_random),
            ).fetchall()

            unread = conn.execute(
                "SELECT c.comment_id, c.memory_id, c.content, c.author, c.created_at, "
                "m.text as memory_text FROM comments c "
                "JOIN memories m ON c.memory_id = m.memory_id "
                "WHERE c.read_by_ai = 0 ORDER BY c.created_at"
            ).fetchall()

        return {
            "pinned": [dict(r) for r in pinned],
            "recent": [dict(r) for r in recent],
            "high_emotion": [dict(r) for r in high_emotion],
            "random_old": [dict(r) for r in random_old],
            "unread_comments": [dict(r) for r in unread],
        }
