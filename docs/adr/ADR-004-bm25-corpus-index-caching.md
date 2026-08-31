# ADR-004: BM25 Okapi Corpus Index Caching with Invalidation

## Status
Accepted (Implemented in `v0.8.3`)

## Context
Hybrid candidate search combines dense semantic vector scores from ChromaDB with sparse BM25 keyword matching scores on tokenized candidate text. Rebuilding the BM25 index on every search query parses the entire document corpus ($O(N)$), adding ~200ms latency per query at 1,000 document chunks.

## Decision
Cache the initialized `BM25Okapi` instance on `JobMatcher` construction and compute an MD5 hash of the indexed document contents. Re-tokenize and rebuild the index only when corpus document count or MD5 hash fingerprint changes.

## Consequences
- **Positive**: Reduces BM25 hybrid query latency from ~200ms down to ~0.1ms for cached queries.
- **Negative**: Retains tokenized corpus arrays in process memory.
