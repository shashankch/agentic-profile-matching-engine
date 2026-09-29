import logging
import tempfile
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from agentic_profile_matching import config
from agentic_profile_matching.stores.exceptions import (
    VectorStoreError,
)
from agentic_profile_matching.stores.in_memory_store import InMemoryVectorStore

logger = logging.getLogger("chroma_store")


class ChromaVectorStore:
    """
    ChromaDB implementation of BaseVectorStore protocol.
    Wraps chromadb.PersistentClient or chromadb.EphemeralClient with domain exception handling
    and consistent response formatting.
    Falls back gracefully to InMemoryVectorStore if ephemeral client initialization fails.
    """

    def __init__(
        self,
        collection_name: str = "resumes",
        db_path: Optional[str] = None,
        ephemeral: bool = False,
        client: Optional[Any] = None,
    ):
        self.collection_name = collection_name
        self.db_path = db_path or config.VECTOR_DB_PATH
        self.ephemeral = ephemeral
        self._fallback_store: Optional[InMemoryVectorStore] = None

        try:
            if client is not None:
                self.client = client
            elif ephemeral:
                try:
                    # Configure explicit temporary directory and disable telemetry for cloud safety
                    settings = Settings(
                        is_persistent=False,
                        persist_directory=tempfile.gettempdir(),
                        anonymized_telemetry=False,
                    )
                    self.client = chromadb.Client(settings)
                except Exception as eph_err:
                    logger.warning(f"Chroma Client(settings) failed ({eph_err}), trying standard EphemeralClient")
                    self.client = chromadb.EphemeralClient()
            else:
                self.client = chromadb.PersistentClient(path=self.db_path)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=None,
            )
        except Exception as e:
            mode = "ephemeral" if ephemeral else f"persistent at '{self.db_path}'"
            if ephemeral:
                logger.warning(
                    f"ChromaDB initialization failed ({mode}): {e}. Seamlessly falling back to InMemoryVectorStore."
                )
                self._fallback_store = InMemoryVectorStore(collection_name=self.collection_name)
            else:
                logger.error(f"Failed to initialize ChromaDB ({mode}): {e}")
                raise VectorStoreError(f"ChromaDB initialization error: {e}") from e

    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if self._fallback_store is not None:
            self._fallback_store.upsert(ids, documents, embeddings, metadatas)
            return
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
        if self._fallback_store is not None:
            return self._fallback_store.query(query_embedding, n_results)
        try:
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
            )
        except Exception as e:
            logger.error(f"Failed to query collection '{self.collection_name}': {e}")
            raise VectorStoreError(f"ChromaDB query failed: {e}") from e

    def get_all(self) -> Dict[str, Any]:
        if self._fallback_store is not None:
            return self._fallback_store.get_all()
        try:
            return self.collection.get()
        except Exception as e:
            logger.error(f"Failed to get_all from collection '{self.collection_name}': {e}")
            raise VectorStoreError(f"ChromaDB get_all failed: {e}") from e

    def count(self) -> int:
        if self._fallback_store is not None:
            return self._fallback_store.count()
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to count items in collection '{self.collection_name}': {e}")
            raise VectorStoreError(f"ChromaDB count failed: {e}") from e

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._fallback_store is not None:
            self._fallback_store.delete(ids, where)
            return
        try:
            kwargs: Dict[str, Any] = {}
            if ids:
                kwargs["ids"] = ids
            if where:
                kwargs["where"] = where
            if kwargs:
                self.collection.delete(**kwargs)
        except Exception as e:
            logger.error(f"Failed to delete items from collection '{self.collection_name}': {e}")
            raise VectorStoreError(f"ChromaDB delete failed: {e}") from e
