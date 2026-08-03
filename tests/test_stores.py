from agentic_profile_matching.stores import (
    BaseVectorStore,
    ChromaVectorStore,
    QdrantVectorStore,
)
from agentic_profile_matching.resume_rag import ResumeRAGPipeline
from agentic_profile_matching.services.ingestion_service import IngestionService


def test_chroma_store_implements_protocol(tmp_path):
    store = ChromaVectorStore(collection_name="test_resumes", db_path=str(tmp_path / "chroma_db"))
    assert isinstance(store, BaseVectorStore)


def test_qdrant_store_implements_protocol():
    store = QdrantVectorStore(collection_name="test_resumes")
    assert isinstance(store, BaseVectorStore)


def test_chroma_store_operations_and_idempotency(tmp_path):
    store = ChromaVectorStore(collection_name="test_resumes", db_path=str(tmp_path / "chroma_db"))

    # Initial state
    assert store.count() == 0

    ids = ["res_1_exp_0", "res_1_skills_1"]
    docs = ["5 years Python experience", "Skills: Python, Docker, Kubernetes"]
    embeddings = [[0.1] * 384, [0.2] * 384]
    metadatas = [
        {"candidate_name": "John Doe", "section": "EXPERIENCE"},
        {"candidate_name": "John Doe", "section": "SKILLS"},
    ]

    # First upsert
    store.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
    assert store.count() == 2

    # Query
    res = store.query(query_embedding=[0.1] * 384, n_results=1)
    assert "documents" in res
    assert len(res["documents"][0]) == 1

    # Re-upsert identical IDs (idempotency check)
    store.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
    assert store.count() == 2

    # Get all
    all_docs = store.get_all()
    assert "ids" in all_docs
    assert len(all_docs["ids"]) == 2


def test_qdrant_store_operations_and_idempotency():
    store = QdrantVectorStore(collection_name="test_resumes")
    assert store.count() == 0

    ids = ["res_1_exp_0"]
    docs = ["5 years Go experience"]
    embeddings = [[0.1] * 384]
    metadatas = [{"candidate_name": "Jane Smith", "section": "EXPERIENCE"}]

    store.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
    assert store.count() == 1

    # Idempotent upsert
    store.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
    assert store.count() == 1


def test_store_injection_into_pipeline_and_service():
    qdrant_stub = QdrantVectorStore()
    pipeline = ResumeRAGPipeline(store=qdrant_stub)
    assert pipeline.store is qdrant_stub

    service = IngestionService(store=qdrant_stub)
    assert service.pipeline.store is qdrant_stub
