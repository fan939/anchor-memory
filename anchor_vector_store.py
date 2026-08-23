"""Small adapter boundary around the vector backend.

SQLite remains Anchor's authority for text and metadata. This module keeps
the Chroma-shaped vector index behind a narrow interface so storage recovery
can be tested without downloading an embedding model or starting Chroma.
"""

from __future__ import annotations

from typing import Any, Protocol


class VectorStoreProtocol(Protocol):
    def upsert(self, *, ids: list[str], embeddings: list[list[float]],
               documents: list[str], metadatas: list[dict[str, Any]]) -> None: ...

    def delete(self, *, ids: list[str]) -> None: ...

    def get(self, *, ids: list[str] | None = None,
            include: list[str] | None = None) -> dict[str, Any]: ...

    def update(self, *, ids: list[str], metadatas: list[dict[str, Any]]) -> None: ...

    def query(self, **kwargs: Any) -> dict[str, Any]: ...

    def count(self) -> int: ...


class ChromaVectorStore:
    """Compatibility adapter for the existing Chroma collection."""

    def __init__(self, collection: Any):
        self.collection = collection

    def upsert(self, *, ids, embeddings, documents, metadatas) -> None:
        self.collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def delete(self, *, ids) -> None:
        self.collection.delete(ids=ids)

    def get(self, *, ids=None, include=None) -> dict:
        return self.collection.get(ids=ids, include=include)

    def update(self, *, ids, metadatas) -> None:
        self.collection.update(ids=ids, metadatas=metadatas)

    def query(self, **kwargs) -> dict:
        return self.collection.query(**kwargs)

    def count(self) -> int:
        return self.collection.count()


class FakeVectorStore:
    """In-memory deterministic implementation for storage-coordination tests."""

    def __init__(self):
        self.records: dict[str, dict[str, Any]] = {}

    def upsert(self, *, ids, embeddings, documents, metadatas) -> None:
        for memory_id, embedding, document, metadata in zip(ids, embeddings, documents, metadatas):
            self.records[memory_id] = {
                "embedding": list(embedding),
                "document": document,
                "metadata": dict(metadata or {}),
            }

    def delete(self, *, ids) -> None:
        for memory_id in ids:
            self.records.pop(memory_id, None)

    def get(self, *, ids=None, include=None) -> dict:
        selected = list(ids) if ids is not None else sorted(self.records)
        found = [memory_id for memory_id in selected if memory_id in self.records]
        return {
            "ids": found,
            "embeddings": [self.records[memory_id]["embedding"] for memory_id in found],
            "documents": [self.records[memory_id]["document"] for memory_id in found],
            "metadatas": [self.records[memory_id]["metadata"] for memory_id in found],
        }

    def update(self, *, ids, metadatas) -> None:
        for memory_id, metadata in zip(ids, metadatas):
            if memory_id not in self.records:
                raise KeyError(memory_id)
            self.records[memory_id]["metadata"] = dict(metadata or {})

    def query(self, **kwargs) -> dict:
        raise NotImplementedError("FakeVectorStore.query must be configured by a focused test")

    def count(self) -> int:
        return len(self.records)
