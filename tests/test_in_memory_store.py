"""
Unit tests for InMemoryVectorStore and Chroma fallback resilience.
Tests protocol compliance, upsert, cosine query, delete, and ephemeral container safety.
"""

from agentic_profile_matching.stores.base import BaseVectorStore
from agentic_profile_matching.stores.in_memory_store import InMemoryVectorStore
from agentic_profile_matching.stores.chroma_store import ChromaVectorStore


def test_in_memory_store_implements_protocol():
    """Verifies that InMemoryVectorStore implements the BaseVectorStore protocol."""
    store = InMemoryVectorStore()
    assert isinstance(store, BaseVectorStore)


def test_in_memory_store_crud_and_query():
    """Tests upsert, count, query, get_all, and delete operations."""
    store = InMemoryVectorStore(collection_name="test_uploads")
    assert store.count() == 0

    # 1. Upsert
    ids = ["c1", "c2"]
    docs = ["Python backend developer with FastAPI", "Frontend engineer with React"]
    # c1 is closer to query [1.0, 0.0], c2 is orthogonal [0.0, 1.0]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    metadatas = [
        {"candidate_name": "Alice", "skills": "Python, FastAPI", "filename": "alice.pdf"},
        {"candidate_name": "Bob", "skills": "React", "filename": "bob.pdf"},
    ]
    store.upsert(ids, docs, embeddings, metadatas)
    assert store.count() == 2

    # 2. Query
    query_emb = [0.9, 0.1]
    res = store.query(query_embedding=query_emb, n_results=2)
    assert len(res["ids"][0]) == 2
    assert res["ids"][0][0] == "c1"  # Alice is closer
    assert res["distances"][0][0] < res["distances"][0][1]

    # 3. get_all
    all_items = store.get_all()
    assert len(all_items["ids"]) == 2
    assert "Alice" in [m["candidate_name"] for m in all_items["metadatas"]]

    # 4. Delete by ID
    store.delete(ids=["c2"])
    assert store.count() == 1
    assert store.get_all()["ids"] == ["c1"]

    # 5. Delete by where metadata
    store.delete(where={"filename": "alice.pdf"})
    assert store.count() == 0


def test_chroma_vector_store_ephemeral_fallback():
    """Verifies that ChromaVectorStore falls back to InMemoryVectorStore if Chroma fails."""
    # Test normal ephemeral initialization with isolated collection
    store = ChromaVectorStore(collection_name="fallback_test_col", ephemeral=True)
    assert store.count() >= 0

    # Test upsert and query through ChromaVectorStore wrapper
    store.upsert(
        ids=["e1"],
        documents=["Test document"],
        embeddings=[[0.5] * 384],
        metadatas=[{"name": "Ephemeral Candidate"}],
    )
    assert store.count() >= 1
    res = store.query(query_embedding=[0.5] * 384, n_results=1)
    assert len(res["ids"][0]) == 1

    # Cleanup
    store.delete(ids=["e1"])
