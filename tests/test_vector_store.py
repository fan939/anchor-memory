import unittest

from anchor_vector_store import FakeVectorStore


class VectorStoreTests(unittest.TestCase):
    def test_fake_store_round_trips_update_and_delete(self):
        store = FakeVectorStore()
        store.upsert(
            ids=["one"], embeddings=[[1.0]], documents=["private text"],
            metadatas=[{"memory_id": "one", "salience": 0.5}],
        )
        self.assertEqual(["one"], store.get(ids=["one"])["ids"])
        store.update(ids=["one"], metadatas=[{"memory_id": "one", "salience": 0.9}])
        self.assertEqual(0.9, store.get(ids=["one"])["metadatas"][0]["salience"])
        store.delete(ids=["one"])
        self.assertEqual(0, store.count())
