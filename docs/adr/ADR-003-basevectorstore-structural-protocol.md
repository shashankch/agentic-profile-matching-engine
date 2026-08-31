# ADR-003: `BaseVectorStore` Structural Protocol Over Abstract Base Class (`abc.ABC`)

## Status
Accepted (Implemented in `v0.8.2`)

## Context
To prevent vendor lock-in to ChromaDB and allow switching to other vector engines (e.g. Qdrant), vector storage implementations must expose a unified interface (`add_documents`, `upsert_documents`, `similarity_search`, `delete`). Standard Python inheritance (`abc.ABC`) enforces rigid subclass relationships and requires explicit inheritance declarations.

## Decision
Define `BaseVectorStore` using Python's `typing.Protocol` (structural subtyping / duck typing). `ChromaVectorStore` and `QdrantVectorStore` implement `BaseVectorStore` implicitly without inheriting from a common base class.

## Consequences
- **Positive**: Enforces compile-time and static interface compliance without coupling store classes to a framework base class; enables flexible dependency injection into `IngestionService` and `JobMatcher`.
- **Negative**: Missing interface methods are caught during static analysis rather than at class definition time.
