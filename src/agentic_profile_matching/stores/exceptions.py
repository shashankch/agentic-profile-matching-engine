class VectorStoreError(Exception):
    """Base domain exception for all vector store operations."""

    pass


class CollectionNotFoundError(VectorStoreError):
    """Raised when a requested collection or index does not exist in the vector store."""

    pass
