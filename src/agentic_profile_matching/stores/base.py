from typing import Protocol, runtime_checkable, List, Dict, Any


@runtime_checkable
class BaseVectorStore(Protocol):
    """
    Structural typing protocol defining standard vector store contract.
    Decouples storage operations (upsert, query, count) from specific engine implementations.
    """

    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Insert or update vector embeddings and document payloads idempotently."""
        ...

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """Search the vector store using dense query embedding."""
        ...

    def get_all(self) -> Dict[str, Any]:
        """Retrieve all document chunks and associated metadatas."""
        ...

    def count(self) -> int:
        """Return the total document chunk count stored in the vector collection."""
        ...
