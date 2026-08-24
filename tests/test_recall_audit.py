import importlib.util
import io
import json
import os
import sys
import types
import unittest
from contextlib import redirect_stdout


# The audit script imports the production runtime, but these tests use a fake
# memory object and should not require model/vector packages to be installed.
chromadb = types.ModuleType("chromadb")
chromadb.PersistentClient = object
sentence_transformers = types.ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = object
sys.modules.setdefault("chromadb", chromadb)
sys.modules.setdefault("sentence_transformers", sentence_transformers)


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "scripts", "audit_recall_state.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("anchor_recall_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDB:
    def __init__(self):
        self.rows = [
            {"memory_id": "high", "salience": 0.9, "unresolved": False,
             "open_questions": [], "epistemic_status": "reported", "state": "active"},
            {"memory_id": "question", "salience": 0.72, "unresolved": True,
             "open_questions": ["What changes?", "When?"],
             "epistemic_status": "reported", "state": "active"},
            {"memory_id": "old", "salience": 0.8, "unresolved": False,
             "open_questions": [], "epistemic_status": "reported", "state": "superseded"},
            {"memory_id": "retracted", "salience": 1.0, "unresolved": True,
             "open_questions": ["hidden"], "epistemic_status": "retracted", "state": "active"},
        ]

    def list_all(self, limit):
        return list(self.rows)

    def wakeup(self, **kwargs):
        return {
            "salient": [{"memory_id": "high"}],
            "unresolved": [{"memory_id": "question"}],
        }


class FakeMemory:
    def __init__(self, path):
        self.db = FakeDB()

    def reconcile(self, repair=False):
        self.repair = repair
        return {
            "missing_vectors": [], "orphan_vectors": [],
            "audit_only_vectors": [], "mismatched_vectors": [],
        }


class RecallAuditTests(unittest.TestCase):
    def test_report_is_body_free_and_excludes_retracted_or_superseded(self):
        module = load_module()
        report = module.build_report(FakeMemory("fixture"), n_salient=2, n_unresolved=1)

        self.assertEqual("read-only", report["mode"])
        self.assertEqual(2, report["effective_memory_count"])
        self.assertEqual({0.72: 1, 0.9: 1}, report["salience_distribution"])
        self.assertEqual(1, report["unresolved_count"])
        self.assertEqual(2, report["open_question_count"])
        self.assertEqual(["high"], report["wakeup"]["salient_ids"])
        self.assertTrue(report["reconcile"]["clean"])
        self.assertNotIn("What changes?", json.dumps(report, ensure_ascii=False))

    def test_cli_is_read_only_and_returns_nonzero_for_reconcile_drift(self):
        module = load_module()

        class DriftMemory(FakeMemory):
            def reconcile(self, repair=False):
                self.repair = repair
                return {
                    "missing_vectors": ["missing"], "orphan_vectors": [],
                    "audit_only_vectors": [], "mismatched_vectors": [],
                }

        output = io.StringIO()
        with redirect_stdout(output):
            code = module.main(["--db-path", "fixture"], memory_factory=DriftMemory)
        self.assertEqual(1, code)
        self.assertIn('"mode": "read-only"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
