import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from anchor_db import AnchorDB
from anchor_reflection import ReflectionService
from scripts.migrate_reflection import inspect as inspect_migration


PROVENANCE = {
    "model": "neutral-test-model",
    "thread_id": "thread-fixture",
    "context_ref": "turns:1-2",
    "extractor_version": "reflection-test-v1",
}


class ReflectionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = AnchorDB(os.path.join(self.tempdir.name, "memories.db"))
        self.service = ReflectionService(self.db)
        self.db.insert(
            "event-1", "A neutral project decision was recorded.",
            memory_layer="event", source_ref="thread-fixture:1",
            provenance=PROVENANCE,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def _draft(self, **overrides):
        values = {
            "source_event_ids": ["event-1"],
            "trigger_type": "user_invite",
            "selection_reason": "The event was explicitly selected for review.",
            "previous_interpretation": "The decision looked final.",
            "current_interpretation": "It was a reversible working decision.",
            "change_or_tension": "scope_narrowed",
            "confidence": 0.7,
            "open_questions": ["Was it revisited later?"],
            "counterevidence": [{
                "content": "The word final appeared in one note.",
                "source_ref": "thread-fixture:2",
            }],
            "provenance": PROVENANCE,
            "status": "tentative",
            "supersedes": "",
        }
        values.update(overrides)
        return self.service.draft(**values)

    def test_draft_is_validated_but_not_persisted(self):
        draft = self._draft()

        self.assertFalse(draft["persisted"])
        self.assertEqual([], self.db.search_reflections())
        self.assertEqual(64, len(draft["draft_token"]))

    def test_no_change_is_a_valid_reflection(self):
        draft = self._draft(change_or_tension="no_change")
        saved = self.service.save(draft["draft"], draft["draft_token"])

        self.assertEqual("no_change", saved["change_or_tension"])

    def test_full_reflection_decision_and_counterevidence_chain(self):
        draft = self._draft()
        saved = self.service.save(draft["draft"], draft["draft_token"])
        reflection_id = saved["reflection_id"]

        by_event = self.db.get_reflections_for_event("event-1")
        searched = self.db.search_reflections(
            query="reversible", source_event_id="event-1"
        )
        self.assertEqual(reflection_id, by_event[0]["reflection_id"])
        self.assertEqual(reflection_id, searched[0]["reflection_id"])
        self.assertEqual("intact", searched[0]["source_integrity"])

        effect = self.db.record_reflection_effect(
            decision_id="decision-1",
            reflection_ids=[reflection_id],
            input_summary="Choose whether to reopen the project decision.",
            output_summary="Reopen it for review.",
            effect_note="The Reflection showed the decision was reversible.",
            outcome_status="pending",
            provenance=PROVENANCE,
        )
        self.assertEqual([reflection_id], effect["used_reflection_ids"])

        updated = self.db.append_reflection_evidence(
            reflection_id, "counterevidence",
            "A later record treated the decision as fixed.", "thread-fixture:3",
        )
        self.assertEqual("adopted", updated["status"])
        self.assertEqual(["decision-1"], updated["affected_decision_ids"])
        self.assertEqual(2, len(updated["counterevidence"]))

    def test_retracted_source_is_visible_and_cannot_be_hard_deleted(self):
        draft = self._draft()
        saved = self.service.save(draft["draft"], draft["draft_token"])

        self.db.retract("event-1", retracted_by="event-2")
        reflection = self.db.get_reflection(saved["reflection_id"])

        self.assertEqual("retracted_source", reflection["source_integrity"])
        with self.assertRaises(Exception):
            self.db.delete("event-1")
        self.assertIsNotNone(self.db.get("event-1"))

    def test_save_revalidates_source_ids_and_layer(self):
        draft = self._draft()
        draft["draft"]["source_event_ids"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "Unknown source event ID"):
            self.service.save(draft["draft"])

        self.db.insert(
            "core-1", "Confirmed core state", memory_layer="core",
            epistemic_status="confirmed",
            provenance=PROVENANCE,
        )
        with self.assertRaisesRegex(ValueError, "not an event"):
            self._draft(source_event_ids=["core-1"])

    def test_provenance_is_required_for_reflection(self):
        with self.assertRaisesRegex(ValueError, "provenance is missing"):
            self._draft(provenance={"model": "test"})

    def test_candidate_cooldown_is_bypassed_only_by_evidence_bearing_triggers(self):
        saved = self.service.save(self._draft()["draft"])
        unresolved = self.service.list_candidates("unresolved")
        invited = self.service.list_candidates("user_invite")

        self.assertEqual([], unresolved["candidates"])
        self.assertEqual("event-1", invited["candidates"][0]["memory_id"])
        self.assertTrue(invited["candidates"][0]["cooldown_bypassed"])
        self.assertEqual("UTC", invited["opportunity"]["active_day_timezone"])
        self.assertEqual(saved["reflection_id"], self.db.get_reflections_for_event("event-1")[0]["reflection_id"])

    def test_decision_link_rejects_unknown_reflection_without_partial_write(self):
        with self.assertRaisesRegex(ValueError, "Unknown reflection IDs"):
            self.db.record_reflection_effect(
                "decision-bad", ["missing"], "input", "output", "effect",
                provenance=PROVENANCE,
            )
        with self.db._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS n FROM decisions WHERE decision_id = 'decision-bad'"
            ).fetchone()["n"]
        self.assertEqual(0, count)

    def test_migration_dry_run_does_not_change_legacy_schema(self):
        legacy = os.path.join(self.tempdir.name, "legacy.db")
        conn = sqlite3.connect(legacy)
        conn.execute(
            "CREATE TABLE memories (memory_id TEXT PRIMARY KEY, text TEXT, tier TEXT)"
        )
        conn.commit()
        before = [row[1] for row in conn.execute("PRAGMA table_info(memories)")]
        conn.close()

        report = inspect_migration(legacy)

        verify = sqlite3.connect(legacy)
        after = [row[1] for row in verify.execute("PRAGMA table_info(memories)")]
        verify.close()
        self.assertEqual(before, after)
        self.assertIn("memory_layer", report["missing_memory_columns"])
        self.assertIn("reflections", report["missing_tables"])

    def test_random_non_hot_trigger_is_seeded_probability_and_budget_limited(self):
        self.db.insert("event-2", "A second neutral event.", memory_layer="event")
        with patch.dict(os.environ, {
            "ANCHOR_REFLECTION_RANDOM_PROBABILITY": "1.0",
            "ANCHOR_REFLECTION_RANDOM_BUDGET": "1",
        }):
            service = ReflectionService(self.db)
            first = service.list_candidates("random_non_hot", limit=10, random_seed=42)
            second = service.list_candidates("random_non_hot", limit=10, random_seed=42)

        self.assertEqual(first["candidates"], second["candidates"])
        self.assertEqual(1, len(first["candidates"]))


if __name__ == "__main__":
    unittest.main()
