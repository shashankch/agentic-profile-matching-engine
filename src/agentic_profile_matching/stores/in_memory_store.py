"""
In-Memory Vector Store implementation of BaseVectorStore protocol.
Pure memory storage using NumPy cosine similarity. Zero disk persistence, zero SQLite dependencies.
Provides rock-solid crash-proof operation in restricted cloud container environments (Streamlit Cloud, AWS Lambda, serverless).
"""

import math
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("in_memory_store")


class InMemoryVectorStore:
    """
    Lightweight, thread-safe in-memory vector store implementing BaseVectorStore.
    Calculates cosine distances dynamically in memory with zero disk dependencies.
    """

    def __init__(self, collection_name: str = "in_memory_uploads"):
        self.collection_name = collection_name
        self._ids: List[str] = []
        self._documents: List[str] = []
        self._embeddings: List[List[float]] = []
        self._metadatas: List[Dict[str, Any]] = []

    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Idempotently insert or update records by id."""
        for doc_id, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
            if doc_id in self._ids:
                idx = self._ids.index(doc_id)
                self._documents[idx] = doc
                self._embeddings[idx] = emb
                self._metadatas[idx] = meta
            else:
                self._ids.append(doc_id)
                self._documents.append(doc)
                self._embeddings.append(emb)
                self._metadatas.append(meta)

    @staticmethod
    def _cosine_distance(vec_a: List[float], vec_b: List[float]) -> float:
        """Computes cosine distance: 1.0 - cosine_similarity."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 1.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 1.0

        similarity = dot_product / (norm_a * norm_b)
        similarity = max(-1.0, min(1.0, similarity))
        return 1.0 - similarity

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """Search records by cosine distance to query embedding."""
        if not self._ids:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        scored = []
        for doc_id, doc, emb, meta in zip(self._ids, self._documents, self._embeddings, self._metadatas):
            dist = self._cosine_distance(query_embedding, emb)
            scored.append((dist, doc_id, doc, meta))

        # Sort ascending by distance (closest first)
        scored.sort(key=lambda x: x[0])
        top_k = scored[:n_results]

        return {
            "ids": [[item[1] for item in top_k]],
            "documents": [[item[2] for item in top_k]],
            "metadatas": [[item[3] for item in top_k]],
            "distances": [[item[0] for item in top_k]],
        }

    def get_all(self) -> Dict[str, Any]:
        """Retrieve all documents and metadata in the store."""
        return {
            "ids": list(self._ids),
            "documents": list(self._documents),
            "metadatas": [dict(m) for m in self._metadatas],
        }

    def count(self) -> int:
        """Returns total records currently stored."""
        return len(self._ids)

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Deletes items matching ids or metadata where clause."""
        indices_to_delete = set()

        if ids:
            id_set = set(ids)
            for idx, doc_id in enumerate(self._ids):
                if doc_id in id_set:
                    indices_to_delete.add(idx)

        if where:
            for idx, meta in enumerate(self._metadatas):
                matches = True
                for k, v in where.items():
                    if meta.get(k) != v:
                        matches = False
                        break
                if matches:
                    indices_to_delete.add(idx)

        if indices_to_delete:
            new_ids = []
            new_docs = []
            new_embs = []
            new_metas = []
            for i in range(len(self._ids)):
                if i not in indices_to_delete:
                    new_ids.append(self._ids[i])
                    new_docs.append(self._documents[i])
                    new_embs.append(self._embeddings[i])
                    new_metas.append(self._metadatas[i])
            self._ids = new_ids
            self._documents = new_docs
            self._embeddings = new_embs
            self._metadatas = new_metas
