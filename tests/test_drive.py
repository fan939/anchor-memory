import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone

from anchor_db import AnchorDB
from anchor_drive import DriveService, compute_effective_strength
# Storage tests do not need optional embedding dependencies.
chromadb = types.ModuleType("chromadb")
chromadb.PersistentClient = object
sentence_transformers = types.ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = object
sys.modules.setdefault("chromadb", chromadb)
sys.modules.setdefault("sentence_transformers", sentence_transformers)
from anchor_memory import AnchorMemory


PROVENANCE = {
    "model": "test", "thread_id": "drive-test", "context_ref": "turn:1",
    "extractor_version": "drive-v1",
}


class EmptyVectors:
    def get(self, ids=None, include=None):
        return {"ids": [], "documents": [], "metadatas": []}


class DriveTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = AnchorDB(os.path.join(self.tempdir.name, "memories.db"))
        self.service = DriveService(self.db)
        self.db.insert("mem-source", "source", memory_layer="event")
        self.db.create_reflection(
            "refl-source", ["mem-source"], "user_invite", "fixture", "old", "new",
            "changed", 0.5, [], [], PROVENANCE, status="tentative",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def create(self, **overrides):
        values = {
            "content": "Keep a stable check-in rhythm.", "reason": "Relationship continuity matters.",
            "strength": 0.8, "priority": 2, "provenance": PROVENANCE,
        }
        values.update(overrides)
        result = self.service.create(**values)
        self.assertTrue(result["created"])
        return result["drive"]

    def test_create_get_and_list_active_drive(self):
        drive = self.create(links=[{
            "target_type": "memory", "target_id": "mem-source", "relation": "motivated_by",
        }])
        fetched = self.service.get(drive["drive_id"])["drive"]
        listed = self.service.list()["drives"]
        self.assertEqual(drive["drive_id"], fetched["drive_id"])
        self.assertEqual("active", listed[0]["status"])
        self.assertEqual("motivated_by", fetched["links"][0]["relation"])

    def test_rejects_invalid_strength_and_status(self):
        with self.assertRaisesRegex(ValueError, "strength"):
            self.create(strength=1.1)
        drive = self.create()
        with self.assertRaisesRegex(ValueError, "status"):
            self.service.update(drive["drive_id"], status="invented")

    def test_terminal_drive_is_preserved_and_requires_explicit_reactivation(self):
        drive = self.create()
        ended = self.service.update(
            drive["drive_id"], status="satisfied", completion_note="Completed deliberately.",
        )["drive"]
        self.assertEqual("satisfied", ended["status"])
        with self.assertRaisesRegex(ValueError, "requires reactivate"):
            self.service.update(drive["drive_id"], status="active")
        before = ended["last_endorsed_at"]
        restored = self.service.update(
            drive["drive_id"], status="active", reactivate=True,
            reactivation_reason="Circumstances changed.",
        )["drive"]
        self.assertEqual("active", restored["status"])
        self.assertGreaterEqual(restored["last_endorsed_at"], before)
        self.assertIsNotNone(self.db.get_drive(drive["drive_id"]))

    def test_links_validate_targets_and_roll_back_creation(self):
        drive = self.create()
        linked = self.service.link(
            drive["drive_id"], target_type="reflection", target_id="refl-source", relation="supported_by",
        )["drive"]
        self.assertEqual("refl-source", linked["links"][0]["target_id"])
        with self.assertRaisesRegex(ValueError, "target not found"):
            self.service.link(
                drive["drive_id"], target_type="memory", target_id="missing", relation="related",
            )
        self.assertEqual(1, len(self.db.get_drive(drive["drive_id"])["links"]))
        before = self.service.list(status=["active"], min_strength=0, min_priority=-1)["drives"]
        with self.assertRaisesRegex(ValueError, "target not found"):
            self.service.create(
                content="Must not be created", reason="invalid link", strength=0.5,
                links=[{"target_type": "memory", "target_id": "missing", "relation": "related"}],
            )
        after = self.service.list(status=["active"], min_strength=0, min_priority=-1)["drives"]
        self.assertEqual([item["drive_id"] for item in before], [item["drive_id"] for item in after])

    def test_duplicate_is_only_reported_not_merged(self):
        drive = self.create()
        duplicate = self.service.create(
            content="  keep a STABLE check-in rhythm. ", reason="same", strength=0.2,
        )
        self.assertFalse(duplicate["created"])
        self.assertEqual("duplicate_suspected", duplicate["status"])
        self.assertEqual([drive["drive_id"]], duplicate["candidate_ids"])

    def test_expiry_is_query_filter_not_state_mutation(self):
        expired = self.create(expires_at="2000-01-01T00:00:00Z")
        self.assertEqual([], self.service.list(min_strength=0, min_priority=-1)["drives"])
        included = self.service.list(min_strength=0, min_priority=-1, include_expired=True)["drives"]
        self.assertTrue(included[0]["is_expired"])
        self.assertEqual("active", self.db.get_drive(expired["drive_id"])["status"])

    def test_review_finds_required_hygiene_cases(self):
        expired = self.create(content="Expired", expires_at="2000-01-01T00:00:00Z")
        high = self.create(content="High without source", strength=0.9)
        conflict = self.create(content="Conflicting")
        self.db.create_drive({
            "drive_id": "drv_review_duplicate", "content": "  conflicting  ", "reason": "fixture",
            "status": "paused", "strength": 0.4, "priority": 0, "expires_at": "",
            "decay_policy": {"type": "none"}, "source_ref": "", "provenance": {},
            "open_questions": [], "completion_note": "",
        }, [])
        self.service.link(conflict["drive_id"], target_type="drive", target_id=high["drive_id"], relation="conflicts_with")
        terminal = self.create(content="Ended without note")
        self.service.update(terminal["drive_id"], status="abandoned")
        with self.db._conn() as conn:
            conn.execute("UPDATE drives SET last_endorsed_at = ? WHERE drive_id = ?", ("2000-01-01T00:00:00", high["drive_id"]))
        review = self.service.review(stale_days=1)
        self.assertIn(expired["drive_id"], [item["drive_id"] for item in review["expired_active"]])
        self.assertIn(high["drive_id"], [item["drive_id"] for item in review["stale"]])
        self.assertIn(conflict["drive_id"], [item["drive_id"] for item in review["conflicts"]])
        self.assertIn(high["drive_id"], [item["drive_id"] for item in review["high_strength_missing_source_links"]])
        self.assertIn(terminal["drive_id"], [item["drive_id"] for item in review["terminal_missing_completion_note"]])
        self.assertTrue(any(
            {conflict["drive_id"], "drv_review_duplicate"}.issubset(set(group))
            for group in review["duplicate_groups"]
        ))

    def test_linear_decay_is_pure_and_correct(self):
        drive = self.create(decay_policy={
            "type": "linear", "grace_days": 3, "decay_per_day": 0.05, "floor": 0.1,
        })
        before = self.db.get_drive(drive["drive_id"])
        now = datetime.fromisoformat(before["last_endorsed_at"]) + timedelta(days=5)
        self.assertAlmostEqual(0.7, compute_effective_strength(before, now))
        self.assertEqual(before["strength"], self.db.get_drive(drive["drive_id"])["strength"])

    def test_reconcile_reports_orphan_drive_link_and_wakeup_ignores_drives(self):
        drive = self.create()
        before = self.db.wakeup(n_high_emotion=0, n_random=0, n_recent=0, n_salient=0, n_unresolved=0, n_identity=0)
        with self.db._conn() as conn:
            conn.execute(
                "INSERT INTO drive_links (drive_id, target_type, target_id, relation, weight, created_at) "
                "VALUES (?, 'memory', 'physically-missing', 'related', 1.0, ?)",
                (drive["drive_id"], "2000-01-01T00:00:00"),
            )
        memory = AnchorMemory.__new__(AnchorMemory)
        memory.db = self.db
        memory._collection = EmptyVectors()
        report = AnchorMemory.reconcile(memory)
        self.assertEqual("physically-missing", report["orphan_drive_links"][0]["target_id"])
        after = self.db.wakeup(n_high_emotion=0, n_random=0, n_recent=0, n_salient=0, n_unresolved=0, n_identity=0)
        self.assertEqual(before, after)
