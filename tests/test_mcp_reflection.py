import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from anchor_db import AnchorDB
from anchor_mcp import TOOL_SCHEMA_META_KEY, TOOL_SCHEMA_VERSION, create_server


PROVENANCE = {
    "model": "neutral-test-model",
    "thread_id": "mcp-test-thread",
    "context_ref": "turns:1-2",
    "extractor_version": "mcp-test-v1",
}


class FakeMemory:
    def __init__(self, db_path):
        os.makedirs(db_path, exist_ok=True)
        self.db = AnchorDB(os.path.join(db_path, "memories.db"))

    def update_recall_state(self, memory_id, **metadata):
        return self.db.update_recall_state(memory_id, **metadata)


class McpReflectionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        fake_module = types.ModuleType("anchor_memory")
        fake_module.AnchorMemory = FakeMemory
        self.module_patch = patch.dict(sys.modules, {"anchor_memory": fake_module})
        self.module_patch.start()
        self.tools, self.handle, self.memory = create_server(self.tempdir.name)
        self.memory.db.insert(
            "event-mcp", "A neutral MCP fixture event.",
            memory_layer="event", provenance=PROVENANCE,
        )

    def tearDown(self):
        self.module_patch.stop()
        self.tempdir.cleanup()

    def test_reflection_tools_are_exposed_with_truthful_annotations(self):
        by_name = {tool["name"]: tool for tool in self.tools}

        self.assertTrue(by_name["draft_reflection"]["annotations"]["readOnlyHint"])
        self.assertTrue(by_name["search_reflections"]["annotations"]["readOnlyHint"])
        self.assertFalse(by_name["save_reflection"]["annotations"]["readOnlyHint"])
        self.assertTrue(by_name["retract_reflection"]["annotations"]["destructiveHint"])

    def test_schema_constraints_and_business_error_shape_are_advertised(self):
        by_name = {tool["name"]: tool for tool in self.tools}
        self.assertEqual(
            {TOOL_SCHEMA_VERSION},
            {tool["_meta"][TOOL_SCHEMA_META_KEY] for tool in self.tools},
        )
        store = by_name["store_memory"]["inputSchema"]

        self.assertEqual(["core", "dynamic", "event"], store["properties"]["memory_layer"]["enum"])
        self.assertEqual(0.0, store["properties"]["salience"]["minimum"])
        self.assertEqual(3, by_name["search_multi"]["inputSchema"]["properties"]["queries"]["maxItems"])
        self.assertIn("get_links", by_name)
        self.assertIn("pin_memory", by_name)
        self.assertIn("unpin_memory", by_name)
        self.assertIn("update_memory_metadata", by_name)
        self.assertNotIn("update_memory_recall_state", by_name)
        draft = by_name["draft_reflection"]["inputSchema"]
        self.assertIn("user_invite", draft["properties"]["trigger_type"]["enum"])
        self.assertEqual(
            ["model", "thread_id", "context_ref", "extractor_version"],
            draft["properties"]["provenance"]["required"],
        )

        failed = self.handle("set_tier", {"memory_id": "missing", "tier": "long"})
        self.assertFalse(failed["ok"])
        self.assertEqual("not_found", failed["error"]["code"])

        updated = self.handle("update_memory_metadata", {
            "memory_id": "event-mcp", "salience": 0.9,
            "motifs": ["repeated-choice"], "state": "active",
            "unresolved": True, "open_questions": ["What changes next?"],
        })
        self.assertEqual("updated", updated["status"])
        self.assertEqual(0.9, updated["memory"]["salience"])
        self.assertEqual(["repeated-choice"], updated["memory"]["motifs"])
        self.assertTrue(updated["memory"]["unresolved"])

    def test_mcp_minimum_reflection_closure(self):
        draft_args = {
            "source_event_ids": ["event-mcp"],
            "trigger_type": "user_invite",
            "selection_reason": "Explicit neutral test review.",
            "previous_interpretation": "The event had one interpretation.",
            "current_interpretation": "The event may support a narrower interpretation.",
            "change_or_tension": "scope_narrowed",
            "confidence": 0.6,
            "open_questions": ["Does later evidence agree?"],
            "counterevidence": [],
            "provenance": PROVENANCE,
            "status": "tentative",
            "supersedes": "",
        }
        drafted = self.handle("draft_reflection", draft_args)
        self.assertFalse(drafted["persisted"])
        self.assertEqual([], self.memory.db.search_reflections())

        saved = self.handle("save_reflection", {
            "draft": drafted["draft"],
            "expected_draft_token": drafted["draft_token"],
        })
        reflection_id = saved["reflection_id"]

        found = self.handle("search_reflections", {
            "source_event_id": "event-mcp"
        })
        self.assertEqual(reflection_id, found["reflections"][0]["reflection_id"])

        effect = self.handle("record_reflection_effect", {
            "decision_id": "decision-mcp",
            "used_reflection_ids": [reflection_id],
            "input_summary": "A neutral decision input.",
            "output_summary": "A neutral decision output.",
            "effect_note": "The narrower interpretation changed the selected option.",
            "outcome_status": "pending",
            "provenance": PROVENANCE,
        })
        self.assertEqual([reflection_id], effect["used_reflection_ids"])

        memory = self.handle("get_memory", {"memory_id": "event-mcp"})
        self.assertEqual(reflection_id, memory["reflections"][0]["reflection_id"])
        self.assertEqual(["decision-mcp"], memory["reflections"][0]["affected_decision_ids"])


if __name__ == "__main__":
    unittest.main()
