# ADR-005: Idempotent `upsert()` Ingestion with Section-Scoped Chunk Keys

## Status
Accepted (Implemented in `v0.8.2`)

## Context
Re-running resume directory ingestion using standard vector store `add()` calls generates new random UUIDs for identical document chunks. This duplicates entries in ChromaDB, skewing term frequency calculations and corrupting hybrid vector similarity rankings.

## Decision
Implement `upsert_documents()` in `ChromaVectorStore` using deterministic section-scoped chunk identifiers formatted as `{document_filename}_chunk_{section_index}`. Existing document chunks with matching IDs are updated in-place instead of duplicated.

## Consequences
- **Positive**: Guarantees 100% idempotent document re-ingestion; keeps vector database collection size stable across repeated runs.
- **Negative**: Requires callers to supply deterministic chunk IDs or compute hash keys before storage.
