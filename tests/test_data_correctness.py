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
from anchor_mcp import format_wakeup_text
from anchor_vector_store import FakeVectorStore


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


class QueryCollection:
    def count(self):
        return 2

    def query(self, **kwargs):
        return {
            "documents": [["low", "high"]],
            "metadatas": [[
                {"memory_id": "low", "timestamp": "", "tag": "general"},
                {"memory_id": "high", "timestamp": "", "tag": "general"},
            ]],
            "distances": [[0.4, 0.4]],
        }


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
        operations = self.db.list_repair_operations(("applied",))
        self.assertEqual("expire", operations[0]["op_type"])
        self.assertEqual("expired", operations[0]["memory_id"])

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

    def test_zero_salience_and_recall_fields_round_trip(self):
        self.db.insert(
            "salient", "recall state", salience=0.0,
            motifs=["choice", "choice", "response"],
            open_questions=["What changes next?"],
        )

        row = self.db.get("salient")

        self.assertEqual(0.0, row["salience"])
        self.assertEqual(["choice", "response"], row["motifs"])
        self.assertTrue(row["unresolved"])

    def test_repair_journal_is_persistent_and_contains_only_recovery_metadata(self):
        started = self.db.start_repair_operation(
            "store-fixture", "store", memory_id="fixture",
            intended_state={"memory_id": "fixture", "text_sha256": "abc123"},
        )
        self.assertEqual("pending", started["status"])
        self.assertEqual("pending", started["sqlite_state"])
        self.assertEqual("abc123", started["intended_state"]["text_sha256"])

        updated = self.db.update_repair_operation(
            "store-fixture", sqlite_state="applied", vector_state="failed",
            status="needs_repair", error="injected vector failure",
        )
        self.assertEqual("needs_repair", updated["status"])
        self.assertEqual("failed", updated["vector_state"])
        self.assertEqual([updated], self.db.list_repair_operations())

    def test_wakeup_excludes_retracted_and_returns_three_recall_hints(self):
        self.db.insert("gone", "retracted item", salience=1.0)
        self.db.retract("gone", retracted_by="replacement")
        self.db.insert(
            "active", "active item", salience=0.9,
            motifs=["motif-a", "motif-b"],
            open_questions=["question-c", "question-d"],
        )

        normal = self.db.wakeup(n_random=0, debug=False)
        audit = self.db.wakeup(n_random=0, debug=True)
        normal_ids = {
            item["memory_id"] for key in ("identity_snapshot", "pinned", "recent", "high_emotion", "salient", "unresolved")
            for item in normal[key]
        }

        self.assertNotIn("gone", normal_ids)
        self.assertEqual(["motif-a", "motif-b", "question-c"], normal["recall_hints"])
        self.assertNotIn("audit", normal)
        self.assertIn("gone", {item["memory_id"] for item in audit["audit"]["filtered_memories"]})

    def test_wakeup_has_stable_identity_dedupes_and_rejects_default_salience(self):
        self.db.insert(
            "identity", "Confirmed identity", memory_layer="core",
            epistemic_status="confirmed", salience=0.5,
        )
        self.db.insert("default", "Neutral default", salience=0.5)
        self.db.insert("important", "Reviewed important", salience=0.8)
        self.db.insert(
            "question", "Explicit question", salience=0.7,
            open_questions=["What changes next?"],
        )

        result = self.db.wakeup(n_random=0, n_identity=5)
        buckets = [
            result[key] for key in (
                "identity_snapshot", "pinned", "unresolved", "salient",
                "recent", "high_emotion", "random_old",
            )
        ]
        ids = [item["memory_id"] for bucket in buckets for item in bucket]

        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(["identity"], [item["memory_id"] for item in result["identity_snapshot"]])
        self.assertNotIn("default", {item["memory_id"] for item in result["salient"]})
        self.assertEqual(2, result["salience_status"]["default_value_memories"])
        self.assertTrue(result["cold_start_contract"]["deduplicated"])

    def test_wakeup_reflections_excludes_drafts_by_default_and_is_compact(self):
        self.db.insert("source", "event source", memory_layer="event")
        common = {
            "source_event_ids": ["source"], "trigger_type": "unresolved",
            "selection_reason": "fixture", "previous_interpretation": "old",
            "current_interpretation": "new", "change_or_tension": "changed",
            "confidence": 0.7, "open_questions": ["Still open?"],
            "counterevidence": [{"content": "one counterexample"}],
            "provenance": {"fixture": "test"},
        }
        self.db.create_reflection(reflection_id="draft-reflection", status="draft", **common)
        self.db.create_reflection(reflection_id="tentative-reflection", status="tentative", **common)

        normal = self.db.wakeup_reflections(limit=5)
        audit = self.db.wakeup_reflections(limit=5, include_drafts=True)

        self.assertEqual(["tentative-reflection"], [item["reflection_id"] for item in normal["items"]])
        self.assertEqual(1, normal["policy"]["excluded_draft_count"])
        self.assertEqual(2, len(audit["items"]))
        self.assertNotIn("sources", normal["items"][0])
        self.assertEqual(1, normal["items"][0]["counterevidence_count"])

    def test_wakeup_reflections_excludes_retracted_sources(self):
        self.db.insert("source", "event source", memory_layer="event")
        self.db.create_reflection(
            reflection_id="retracted-source-reflection", source_event_ids=["source"],
            trigger_type="unresolved", selection_reason="fixture",
            previous_interpretation="old", current_interpretation="new",
            change_or_tension="changed", confidence=0.6,
            open_questions=["Still open?"], counterevidence=[],
            provenance={"fixture": "test"}, status="tentative",
        )
        self.db.retract("source", retracted_by="fixture")

        normal = self.db.wakeup_reflections(limit=5)
        searched = self.db.search_reflections(limit=5)

        self.assertEqual([], normal["items"])
        self.assertEqual(1, normal["policy"]["excluded_retracted_source_count"])
        self.assertEqual([], searched)

    def test_pinned_core_owns_pinned_bucket_not_identity_snapshot(self):
        self.db.insert(
            "pinned-core", "Pinned confirmed identity", memory_layer="core",
            epistemic_status="confirmed",
        )
        self.db.pin("pinned-core")

        result = self.db.wakeup(n_random=0, n_identity=5)

        self.assertEqual([], [item["memory_id"] for item in result["identity_snapshot"]])
        self.assertEqual(["pinned-core"], [item["memory_id"] for item in result["pinned"]])

    def test_wakeup_text_includes_compact_reflection_when_present(self):
        text = format_wakeup_text({
            "recent_reflections": [{
                "reflection_id": "r1", "status": "tentative", "confidence": 0.6,
                "current_interpretation": "A compact interpretation",
            }],
        })
        self.assertIn("Reviewed recent Reflections", text)
        self.assertIn("A compact interpretation", text)

    def test_recall_metadata_reconciliation_is_reviewable_and_reflection_scoped(self):
        self.db.insert("source", "This remains an unresolved question.", memory_layer="event")
        self.db.insert("abandoned-source", "Old event", memory_layer="event")
        self.db.insert(
            "identity-core", "Stable confirmed identity", memory_layer="core",
            epistemic_status="confirmed",
        )
        self.db.create_reflection(
            reflection_id="live-question", source_event_ids=["source"],
            trigger_type="unresolved", selection_reason="fixture",
            previous_interpretation="old", current_interpretation="new",
            change_or_tension="open", confidence=0.5,
            open_questions=["What would settle this?"], counterevidence=[],
            provenance={"fixture": "test"}, status="tentative",
        )
        self.db.create_reflection(
            reflection_id="abandoned-question", source_event_ids=["abandoned-source"],
            trigger_type="unresolved", selection_reason="fixture",
            previous_interpretation="old", current_interpretation="new",
            change_or_tension="open", confidence=0.5,
            open_questions=["This must stay out."], counterevidence=[],
            provenance={"fixture": "test"}, status="abandoned",
        )

        plan = self.db.plan_recall_metadata_reconciliation()

        self.assertEqual(["source"], [item["memory_id"] for item in plan["reflection_question_updates"]])
        self.assertEqual(
            ["What would settle this?"],
            plan["reflection_question_updates"][0]["proposed"]["open_questions"],
        )
        self.assertEqual(["source"], [item["memory_id"] for item in plan["text_review_candidates"]])
        core_candidate = next(
            item for item in plan["salience_review_candidates"]
            if item["memory_id"] == "identity-core"
        )
        self.assertEqual(0.8, core_candidate["suggested_salience"])
        self.assertTrue(core_candidate["manual_review_required"])
        self.assertFalse(self.db.get("source")["unresolved"])
        self.assertFalse(self.db.get("abandoned-source")["unresolved"])

        memory = AnchorMemory.__new__(AnchorMemory)
        memory.db = self.db
        memory._collection = FakeCollection([])
        preview = AnchorMemory.reconcile_recall_metadata(memory, dry_run=True)
        self.assertEqual("preview", preview["status"])
        self.assertFalse(self.db.get("source")["unresolved"])

        applied = AnchorMemory.reconcile_recall_metadata(
            memory, dry_run=False, maintenance_id="recall-metadata-fixture"
        )
        self.assertEqual(["source"], applied["applied_memory_ids"])
        self.assertTrue(self.db.get("source")["unresolved"])
        self.assertEqual(["What would settle this?"], self.db.get("source")["open_questions"])
        repeated = AnchorMemory.reconcile_recall_metadata(
            memory, dry_run=False, maintenance_id="recall-metadata-fixture"
        )
        self.assertEqual("already_recorded", repeated["status"])

    def test_explicit_link_relation_is_readable_in_both_directions(self):
        self.db.insert("a", "a")
        self.db.insert("b", "b")
        self.db.connect("a", "b", weight=1.25, relation="contradicts")

        links = self.db.get_links("a")

        self.assertEqual("contradicts", links["outgoing"][0]["relation"])
        self.assertEqual(1.25, links["outgoing"][0]["weight"])
        self.assertTrue(links["outgoing"][0]["created"])

        self.db.connect("b", "a", weight=0.5, relation="supersedes")
        directional = self.db.get_links("b")
        supersedes = [row for row in directional["outgoing"] if row["relation"] == "supersedes"]
        reverse = [row for row in self.db.get_links("a")["outgoing"] if row["relation"] == "supersedes"]
        self.assertEqual(1, len(supersedes))
        self.assertEqual([], reverse)

    def test_dream_dry_run_is_audited_and_does_not_mutate(self):
        self.db.insert("a", "a")
        self.db.insert("b", "b")
        self.db.connect("a", "b", weight=2.0)
        memory = AnchorMemory.__new__(AnchorMemory)
        memory.db = self.db
        memory._collection = FakeCollection(["a", "b"])
        memory.emotion_equalization = False

        preview = AnchorMemory.dream_pass(
            memory, dry_run=True, maintenance_id="maintenance-test",
            auto_discover=False,
        )
        repeated = AnchorMemory.dream_pass(
            memory, dry_run=False, maintenance_id="maintenance-test",
            auto_discover=False,
        )

        self.assertEqual("preview", preview["status"])
        self.assertEqual(2.0, self.db.get_edge_weight("a", "b"))
        self.assertEqual("already_recorded", repeated["status"])

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

    def test_high_salience_affects_ranking_without_promoting_truth(self):
        self.db.insert("low", "low", salience=0.0)
        self.db.insert("high", "high", salience=1.0)
        memory = AnchorMemory.__new__(AnchorMemory)
        memory.db = self.db
        memory._collection = QueryCollection()
        memory._embedder = FakeEmbedder()
        memory.recency_weight = 0.0
        memory.recency_halflife_days = 30.0
        memory.auto_hebbian = False

        results = AnchorMemory.search(
            memory, "same semantic distance", n_results=2,
            associate=False, debug=True,
        )

        self.assertEqual("high", results[0]["memory_id"])
        self.assertEqual("reported", results[0]["epistemic_status"])
        self.assertEqual("event", results[0]["memory_layer"])
        self.assertEqual(0.15, results[0]["debug"]["salience_boost"])
        self.assertIn("ranking_reason", results[0]["debug"])

    def test_retract_removes_vector_but_keeps_sqlite_audit_row(self):
        self.db.insert("same", "original")
        memory = AnchorMemory.__new__(AnchorMemory)
        memory.db = self.db
        memory._collection = SnapshotCollection()

        self.assertTrue(AnchorMemory.retract(memory, "same", "replacement"))

        self.assertNotIn("same", memory._collection.records)
        self.assertEqual("retracted", self.db.get("same")["epistemic_status"])
        operation = self.db.list_repair_operations(("applied",))[0]
        self.assertEqual("retract", operation["op_type"])

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

    def test_reconcile_reports_pending_repair_operations_without_applying_them(self):
        self.db.start_repair_operation(
            "pending-store", "store", memory_id="missing-vector",
            intended_state={"memory_id": "missing-vector", "text_sha256": "fixture"},
        )
        memory = AnchorMemory.__new__(AnchorMemory)
        memory.db = self.db
        memory._collection = FakeCollection()

        report = AnchorMemory.reconcile(memory)

        self.assertEqual(1, report["repair_pending_count"])
        self.assertEqual("pending-store", report["repair_operations"][0]["op_id"])
        self.assertEqual(
            "pending-store", report["repair_preview"]["review_operations"][0]["op_id"]
        )
        self.assertTrue(report["repair_preview"]["apply_requires_backup"])

    def test_reconcile_preview_distinguishes_rebuild_from_destructive_vector_cleanup(self):
        self.db.insert("sqlite-only", "needs a vector")
        memory = AnchorMemory.__new__(AnchorMemory)
        memory.db = self.db
        memory._collection = FakeCollection(["chroma-only"])

        report = AnchorMemory.reconcile(memory)

        self.assertEqual(
            ["sqlite-only"],
            [item["memory_id"] for item in report["repair_preview"]["rebuild_vectors"]],
        )
        self.assertEqual(
            ["chroma-only"],
            [item["memory_id"] for item in report["repair_preview"]["remove_vectors"]],
        )
        self.assertTrue(report["repair_preview"]["apply_requires_backup"])

    def test_reconcile_reports_vector_document_mismatch_without_exposing_text(self):
        self.db.insert("same", "SQLite authority")
        vector_store = FakeVectorStore()
        vector_store.upsert(
            ids=["same"], embeddings=[[1.0]], documents=["stale vector document"],
            metadatas=[{"memory_id": "same"}],
        )
        memory = AnchorMemory.__new__(AnchorMemory)
        memory.db = self.db
        memory._vector_store = vector_store

        report = AnchorMemory.reconcile(memory)

        self.assertEqual(
            [{"memory_id": "same", "reason": "VECTOR_DOCUMENT_MISMATCH"}],
            report["mismatched_vectors"],
        )
        self.assertNotIn("SQLite authority", str(report))
        self.assertNotIn("stale vector document", str(report))

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

    def test_store_sqlite_failure_is_recorded_as_rolled_back_repair_operation(self):
        memory = AnchorMemory.__new__(AnchorMemory)
        memory._collection = SnapshotCollection()
        memory._embedder = FakeEmbedder()
        memory.db = self.db
        memory._eager_link = False

        with patch.object(self.db, "insert", side_effect=RuntimeError("injected sqlite failure")):
            with self.assertRaisesRegex(RuntimeError, "injected sqlite failure"):
                AnchorMemory.store(memory, "same", "replacement")

        operations = self.db.list_repair_operations(("rolled_back",))
        self.assertEqual(1, len(operations))
        self.assertEqual("store", operations[0]["op_type"])
        self.assertEqual("failed", operations[0]["sqlite_state"])
        self.assertEqual("rolled_back", operations[0]["vector_state"])
        self.assertEqual("original", memory._collection.records["same"]["document"])

    def test_store_records_completed_repair_operation(self):
        memory = AnchorMemory.__new__(AnchorMemory)
        memory._collection = SnapshotCollection()
        memory._embedder = FakeEmbedder()
        memory.db = self.db
        memory._eager_link = False

        self.assertEqual("new", AnchorMemory.store(memory, "new", "durable event"))

        operation = self.db.list_repair_operations(("applied",))[0]
        self.assertEqual("store", operation["op_type"])
        self.assertEqual("new", operation["memory_id"])
        self.assertEqual("applied", operation["sqlite_state"])
        self.assertEqual("applied", operation["vector_state"])
        self.assertNotIn("durable event", str(operation["intended_state"]))

    def test_delete_sqlite_failure_is_recorded_as_rolled_back_repair_operation(self):
        self.db.insert("same", "original")
        memory = AnchorMemory.__new__(AnchorMemory)
        memory._collection = SnapshotCollection()
        memory.db = self.db

        with patch.object(self.db, "delete", side_effect=RuntimeError("injected sqlite delete failure")):
            with self.assertRaisesRegex(RuntimeError, "injected sqlite delete failure"):
                AnchorMemory.delete(memory, "same")

        operations = self.db.list_repair_operations(("rolled_back",))
        self.assertEqual(1, len(operations))
        self.assertEqual("delete", operations[0]["op_type"])
        self.assertEqual("failed", operations[0]["sqlite_state"])
        self.assertEqual("rolled_back", operations[0]["vector_state"])
        self.assertIsNotNone(self.db.get("same"))
        self.assertEqual("original", memory._collection.records["same"]["document"])

    def test_merge_is_atomic_in_sqlite_and_records_completed_dual_store_operation(self):
        for memory_id in ("survivor", "duplicate", "neighbor"):
            self.db.insert(memory_id, memory_id)
        self.db.connect("duplicate", "neighbor", weight=1.25, relation="contradicts")
        memory = AnchorMemory.__new__(AnchorMemory)
        memory.db = self.db
        memory._collection = SnapshotCollection()
        memory._collection.records["survivor"] = {
            "embedding": [2.0], "document": "survivor", "metadata": {"memory_id": "survivor"},
        }
        memory._collection.records["duplicate"] = {
            "embedding": [3.0], "document": "duplicate", "metadata": {"memory_id": "duplicate"},
        }

        result = AnchorMemory.merge_memories(memory, "survivor", "duplicate")

        self.assertTrue(result["duplicate_deleted"])
        self.assertIsNone(self.db.get("duplicate"))
        self.assertNotIn("duplicate", memory._collection.records)
        self.assertAlmostEqual(1.25, self.db.get_edge_weight("survivor", "neighbor"))
        operation = self.db.list_repair_operations(("applied",))[0]
        self.assertEqual("merge", operation["op_type"])
        self.assertEqual(["duplicate"], operation["related_ids"])

    def test_merge_sqlite_failure_restores_duplicate_vector_and_records_rollback(self):
        self.db.insert("survivor", "survivor")
        self.db.insert("duplicate", "duplicate")
        memory = AnchorMemory.__new__(AnchorMemory)
        memory.db = self.db
        memory._collection = SnapshotCollection()
        memory._collection.records["duplicate"] = {
            "embedding": [3.0], "document": "duplicate", "metadata": {"memory_id": "duplicate"},
        }

        with patch.object(self.db, "merge_duplicate_into", side_effect=RuntimeError("injected merge failure")):
            with self.assertRaisesRegex(RuntimeError, "injected merge failure"):
                AnchorMemory.merge_memories(memory, "survivor", "duplicate")

        self.assertIsNotNone(self.db.get("duplicate"))
        self.assertIn("duplicate", memory._collection.records)
        operation = self.db.list_repair_operations(("rolled_back",))[0]
        self.assertEqual("merge", operation["op_type"])
        self.assertEqual("rolled_back", operation["vector_state"])

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
