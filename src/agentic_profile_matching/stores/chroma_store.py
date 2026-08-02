import logging
from typing import List, Dict, Any, Optional
import chromadb

from agentic_profile_matching import config
from agentic_profile_matching.stores.exceptions import (
    VectorStoreError,
)

logger = logging.getLogger("chroma_store")


class ChromaVectorStore:
    """
    ChromaDB implementation of BaseVectorStore protocol.
    Wraps chromadb.PersistentClient with domain exception handling and consistent response formatting.
    """

    def __init__(
        self,
        collection_name: str = "resumes",
        db_path: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.db_path = db_path or config.VECTOR_DB_PATH
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_or_create_collection(self.collection_name)
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB persistent client at '{self.db_path}': {e}")
            raise VectorStoreError(f"ChromaDB initialization error: {e}") from e

    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as e:
            logger.error(f"Failed to upsert items to collection '{self.collection_name}': {e}")
            raise VectorStoreError(f"ChromaDB upsert failed: {e}") from e

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
    ) -> Dict[str, Any]:
        try:
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
            )
        except Exception as e:
            logger.error(f"Failed to query collection '{self.collection_name}': {e}")
            raise VectorStoreError(f"ChromaDB query failed: {e}") from e

    def get_all(self) -> Dict[str, Any]:
        try:
            return self.collection.get()
        except Exception as e:
            logger.error(f"Failed to get_all from collection '{self.collection_name}': {e}")
            raise VectorStoreError(f"ChromaDB get_all failed: {e}") from e

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to count items in collection '{self.collection_name}': {e}")
            raise VectorStoreError(f"ChromaDB count failed: {e}") from e
