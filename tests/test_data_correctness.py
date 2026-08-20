import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone


# The production module imports optional heavyweight model packages at import
# time. Storage-coordination tests use controlled fakes and need no model.
chromadb = types.ModuleType("chromadb")
chromadb.PersistentClient = object
sentence_transformers = types.ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = object
sys.modules.setdefault("chromadb", chromadb)
sys.modules.setdefault("sentence_transformers", sentence_transformers)

from anchor_db import AnchorDB
from anchor_memory import AnchorMemory
from anchor_pinned import CONTINUITY_HEADER, LEGACY_CONTINUITY_HEADER, write_session_state
from anchor_proxy import curate_turn


class FakeCollection:
    def __init__(self, ids=()):
        self.ids = set(ids)

    def delete(self, ids):
        self.ids.difference_update(ids)

    def get(self, ids=None, include=None):
        return {"ids": list(self.ids)}


class SnapshotCollection:
    def __init__(self):
        self.records = {
            "same": {
                "embedding": [1.0], "document": "original", "metadata": {"v": 1}
            }
        }

    def get(self, ids=None, include=None):
        found = [i for i in (ids or self.records) if i in self.records]
        return {
            "ids": found,
            "embeddings": [self.records[i]["embedding"] for i in found],
            "documents": [self.records[i]["document"] for i in found],
            "metadatas": [self.records[i]["metadata"] for i in found],
        }

    def upsert(self, ids, embeddings, documents, metadatas):
        for i, embedding, document, metadata in zip(ids, embeddings, documents, metadatas):
            self.records[i] = {
                "embedding": embedding, "document": document, "metadata": metadata
            }

    def delete(self, ids):
        for i in ids:
            self.records.pop(i, None)

    def query(self, **kwargs):
        return {"ids": [[]], "metadatas": [[]]}


class FakeEmbedder:
    def encode(self, text):
        return types.SimpleNamespace(tolist=lambda: [9.0])


class FailingDB:
    def insert(self, *args, **kwargs):
        raise RuntimeError("injected sqlite failure")


class DataCorrectnessTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = AnchorDB(os.path.join(self.tempdir.name, "memories.db"))

    def tearDown(self):
        self.tempdir.cleanup()

    def test_expired_short_memory_is_deleted_from_both_stores(self):
        self.db.insert("expired", "old short memory", tier="short")
        old = (
            datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=30)
        ).isoformat()
        with self.db._conn() as conn:
            conn.execute(
                "UPDATE memories SET timestamp = ? WHERE memory_id = 'expired'",
                (old,),
            )
            conn.commit()

        memory = AnchorMemory.__new__(AnchorMemory)
        memory.db = self.db
        memory._collection = FakeCollection(["expired"])
        memory.split_bundled = lambda **kwargs: 0
        memory.emotion_equalization = False

        result = AnchorMemory.dream_pass(memory, auto_discover=False)

        self.assertEqual(1, result["decayed_memories"])
        self.assertEqual(0, result["decay_failed"])
        self.assertIsNone(self.db.get("expired"))
        self.assertNotIn("expired", memory._collection.ids)

    def test_weak_and_strong_edges_decay_once(self):
        for memory_id in ("a", "b", "c"):
            self.db.insert(memory_id, memory_id)
        self.db.connect("a", "b", weight=1.0)
        self.db.connect("a", "c", weight=2.0)

        stats = self.db.decay_edges_once(
            strong_threshold=1.5,
            weak_decay_factor=0.9,
            strong_decay_factor=0.95,
        )

        self.assertAlmostEqual(0.9, self.db.get_edge_weight("a", "b"))
        self.assertAlmostEqual(1.9, self.db.get_edge_weight("a", "c"))
        self.assertEqual(2, stats["strong_decayed"])  # bidirectional rows

    def test_zero_emotion_is_not_replaced_by_default(self):
        self.db.insert("zero", "neutral", emotion_score=0.0)
        self.db.insert("one", "intense", emotion_score=1.0)
        self.db.connect("zero", "one", weight=1.0)

        self.db.equalize_emotion_scores(nudge=0.1, threshold=0.0)

        self.assertAlmostEqual(0.1, self.db.get_emotion_score("zero"))

    def test_legacy_records_receive_conservative_provenance_defaults(self):
        self.db.insert("legacy", "old imported record")

        row = self.db.get("legacy")

        self.assertEqual("mixed", row["perspective"])
        self.assertEqual("reported", row["epistemic_status"])
        self.assertEqual(0.5, row["confidence"])
        self.assertTrue(row["created_at"])
        self.assertTrue(row["updated_at"])
        self.assertEqual("event", row["memory_layer"])
        self.assertEqual("", row["source_ref"])
        self.assertEqual({}, row["provenance"])

    def test_retract_preserves_record_for_audit(self):
        self.db.insert("old", "superseded belief")

        self.assertTrue(self.db.retract("old", retracted_by="new"))

        row = self.db.get("old")
        self.assertEqual("retracted", row["epistemic_status"])
        self.assertEqual("new", row["retracted_by"])

    def test_reconcile_reports_and_removes_only_chroma_orphans(self):
        self.db.insert("sqlite-only", "needs re-embedding")
        memory = AnchorMemory.__new__(AnchorMemory)
        memory.db = self.db
        memory._collection = FakeCollection(["chroma-only"])

        dry_run = AnchorMemory.reconcile(memory)
        repaired = AnchorMemory.reconcile(memory, repair=True)

        self.assertEqual(["sqlite-only"], dry_run["sqlite_only"])
        self.assertEqual(["chroma-only"], dry_run["chroma_only"])
        self.assertIn("chroma-only", FakeCollection(["chroma-only"]).ids)
        self.assertEqual(1, repaired["removed_chroma_orphans"])
        self.assertNotIn("chroma-only", memory._collection.ids)

    def test_legacy_continuity_header_is_neutralized_without_losing_body(self):
        pinned = os.path.join(self.tempdir.name, "pinned")
        write_session_state(pinned, LEGACY_CONTINUITY_HEADER + "\n\nbody stays")

        with open(os.path.join(pinned, "session_state.md"), encoding="utf-8") as file:
            content = file.read()

        self.assertTrue(content.startswith(CONTINUITY_HEADER))
        self.assertIn("body stays", content)
        self.assertNotIn("same ongoing life", content)

    def test_failed_sqlite_update_restores_existing_vector(self):
        memory = AnchorMemory.__new__(AnchorMemory)
        memory._collection = SnapshotCollection()
        memory._embedder = FakeEmbedder()
        memory.db = FailingDB()
        memory._eager_link = False

        with self.assertRaisesRegex(RuntimeError, "injected sqlite failure"):
            AnchorMemory.store(memory, "same", "replacement")

        self.assertEqual("original", memory._collection.records["same"]["document"])
        self.assertEqual([1.0], memory._collection.records["same"]["embedding"])

    def test_unconfirmed_content_cannot_enter_core_layer(self):
        memory = AnchorMemory.__new__(AnchorMemory)
        with self.assertRaisesRegex(ValueError, "core memories require"):
            AnchorMemory.store(
                memory, "core-candidate", "Unconfirmed interpretation",
                memory_layer="core", epistemic_status="hypothesis",
            )

    def test_database_also_rejects_unconfirmed_core(self):
        with self.assertRaisesRegex(ValueError, "core memories require"):
            self.db.insert(
                "core-direct", "Not confirmed", memory_layer="core",
                epistemic_status="reported",
            )

    def test_dynamic_layer_preserves_changeable_sourced_state(self):
        self.db.insert(
            "dynamic", "Temporary preference", memory_layer="dynamic",
            epistemic_status="reported", source_ref="turn:10",
            provenance={"model": "fixture", "thread_id": "fixture"},
        )
        row = self.db.get("dynamic")
        self.assertEqual("dynamic", row["memory_layer"])
        self.assertEqual("reported", row["epistemic_status"])
        self.assertEqual("turn:10", row["source_ref"])

    def test_only_confirmed_core_can_be_pinned(self):
        self.db.insert("event", "ordinary evidence")
        with self.assertRaisesRegex(ValueError, "only confirmed core"):
            self.db.pin("event")

        self.db.insert(
            "core", "explicitly confirmed", memory_layer="core",
            epistemic_status="confirmed",
        )
        self.db.pin("core")
        self.assertEqual(["core"], [row["memory_id"] for row in self.db.get_pinned()])

    def test_feedback_is_append_only_bounded_and_never_deletes_memory(self):
        self.db.insert("feedback-target", "durable event")
        self.db.record_memory_feedback("feedback-target", False, "not useful")
        self.assertAlmostEqual(0.01, self.db.get_feedback_penalty("feedback-target"))
        self.assertIsNotNone(self.db.get("feedback-target"))
        self.assertEqual(1, len(self.db.get_memory_feedback("feedback-target")))

        self.db.record_memory_feedback("feedback-target", True, "useful later")
        self.assertAlmostEqual(0.0, self.db.get_feedback_penalty("feedback-target"))
        self.assertIsNotNone(self.db.get("feedback-target"))

    def test_passive_consolidation_respects_disabled_auto_hebbian(self):
        memory = AnchorMemory.__new__(AnchorMemory)
        memory.auto_hebbian = False

        result = AnchorMemory.consolidate(memory, "neutral conversation")

        self.assertEqual("disabled", result["status"])

    def test_hard_delete_does_not_remove_vector_when_reflection_references_event(self):
        self.db.insert("source", "event source", memory_layer="event")
        self.db.create_reflection(
            reflection_id="refl-guard",
            source_event_ids=["source"],
            trigger_type="user_invite",
            selection_reason="explicit test",
            previous_interpretation="old",
            current_interpretation="new",
            change_or_tension="no_change",
            confidence=0.5,
            open_questions=[],
            counterevidence=[],
            provenance={"fixture": "test"},
        )
        memory = AnchorMemory.__new__(AnchorMemory)
        memory.db = self.db
        memory._collection = FakeCollection(["source"])

        with self.assertRaisesRegex(ValueError, "Cannot hard-delete"):
            AnchorMemory.delete(memory, "source")

        self.assertIn("source", memory._collection.ids)
        self.assertIsNotNone(self.db.get("source"))

    def test_proxy_curator_is_disabled_by_default(self):
        class Exploding:
            def call(self, **kwargs):
                raise AssertionError("curator should not call the LLM")

            def store(self, **kwargs):
                raise AssertionError("curator should not write memory")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANCHOR_AUTO_CURATE", None)
            curate_turn(Exploding(), Exploding(), self.tempdir.name, "hello", "hi")


if __name__ == "__main__":
    unittest.main()
