class EngineError(Exception):
    """Base domain exception for all agentic profile matching engine operations."""

    pass


class IngestionError(EngineError):
    """Raised when candidate resume document ingestion or chunking fails."""

    pass


class RetrievalError(EngineError):
    """Raised when candidate vector search or hybrid retrieval fails."""

    pass


class LLMParseError(EngineError):
    """Raised when structured LLM output validation or parsing fails."""

    pass
