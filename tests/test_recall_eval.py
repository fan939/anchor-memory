import os
import sys
import tempfile
import types
import unittest


# Deterministic fakes keep recall regression tests independent of model downloads.
chromadb = types.ModuleType("chromadb")
chromadb.PersistentClient = object
sentence_transformers = types.ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = object
sys.modules.setdefault("chromadb", chromadb)
sys.modules.setdefault("sentence_transformers", sentence_transformers)

from anchor_db import AnchorDB
from anchor_http import INSTRUCTIONS
from anchor_memory import AnchorMemory


class FakeEmbedder:
    def encode(self, text):
        return types.SimpleNamespace(tolist=lambda: [1.0])


class FamilyQueryCollection:
    def count(self):
        return 2

    def query(self, **kwargs):
        return {
            "documents": [["周日回家吃饭与家庭关系背景", "周日超市购物清单"]],
            "metadatas": [[
                {"memory_id": "family", "timestamp": "", "tag": "relationship"},
                {"memory_id": "shopping", "timestamp": "", "tag": "practical"},
            ]],
            "distances": [[0.25, 0.30]],
        }


class RecallFailureEvalTests(unittest.TestCase):
    """Regression boundaries for the four agreed recall failure cases."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = AnchorDB(os.path.join(self.tempdir.name, "memories.db"))

    def tearDown(self):
        self.tempdir.cleanup()

    def test_family_plan_prefers_relevant_durable_context(self):
        self.db.insert(
            "family", "周日回家吃饭与家庭关系背景", tag="relationship",
            salience=0.9, motifs=["family"], unresolved=True,
        )
        self.db.insert(
            "shopping", "周日超市购物清单", tag="practical", salience=0.2,
        )
        memory = AnchorMemory.__new__(AnchorMemory)
        memory.db = self.db
        memory._collection = FamilyQueryCollection()
        memory._embedder = FakeEmbedder()
        memory.recency_weight = 0.0
        memory.recency_halflife_days = 30.0
        memory.auto_hebbian = False

        results = AnchorMemory.search(
            memory, "周日回家吃饭", n_results=2, associate=False, debug=True,
        )

        self.assertEqual("family", results[0]["memory_id"])
        self.assertGreater(results[0]["debug"]["salience_boost"], 0)
        self.assertGreater(results[0]["debug"]["unresolved_boost"], 0)

    def test_ambiguous_short_replies_do_not_authorize_overwrite(self):
        self.assertIn("'算了'", INSTRUCTIONS)
        self.assertIn("'我没事'", INSTRUCTIONS)
        self.assertIn("do not overwrite durable context", INSTRUCTIONS)
        self.assertIn("without confirmation", INSTRUCTIONS)

    def test_explicit_memory_question_requires_search_not_pretence(self):
        self.assertIn("search before claiming memory", INSTRUCTIONS)
        self.assertIn("when no supporting memory is found", INSTRUCTIONS)

    def test_targeted_recall_is_capped_at_three_queries(self):
        self.assertIn("no more than three focused queries", INSTRUCTIONS)


if __name__ == "__main__":
    unittest.main()
