import contextlib
import io
import importlib.util
import os
import sys
import types
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "scripts", "reconcile.py"
)

# CLI parsing tests must not need the heavyweight production vector runtime.
chromadb = types.ModuleType("chromadb")
chromadb.PersistentClient = object
sentence_transformers = types.ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = object
sys.modules.setdefault("chromadb", chromadb)
sys.modules.setdefault("sentence_transformers", sentence_transformers)


def load_module():
    spec = importlib.util.spec_from_file_location("anchor_reconcile_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeMemory:
    def __init__(self, db_path):
        self.db_path = db_path
        self.calls = []

    def reconcile(self, repair=False):
        self.calls.append(repair)
        return {
            "missing_vectors": [],
            "orphan_vectors": [],
            "audit_only_vectors": [],
            "mismatched_vectors": [],
            "repair_preview": {},
        }


class ReconcileCliTests(unittest.TestCase):
    def test_dry_run_is_default_and_does_not_apply(self):
        module = load_module()
        created = []

        def factory(path):
            memory = FakeMemory(path)
            created.append(memory)
            return memory

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(0, module.main(["--db-path", "fixture"], memory_factory=factory))
        self.assertEqual([False], created[0].calls)

    def test_apply_requires_an_existing_backup(self):
        module = load_module()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                module.main(["--db-path", "fixture", "--apply"], memory_factory=FakeMemory)
        self.assertEqual(2, raised.exception.code)

    def test_apply_rechecks_consistency_after_repair(self):
        module = load_module()
        created = []

        def factory(path):
            memory = FakeMemory(path)
            created.append(memory)
            return memory

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                0,
                module.main(
                    ["--db-path", "fixture", "--apply", "--backup", "backup"],
                    memory_factory=factory,
                    path_exists=lambda path: path == "backup",
                ),
            )
        self.assertEqual([False, True, False], created[0].calls)
