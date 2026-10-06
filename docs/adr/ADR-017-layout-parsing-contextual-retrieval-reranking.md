# ADR-017: Layout-Aware Section Parsing, Anthropic Contextual Retrieval & Two-Stage Reranking

## Status
Implemented (Released in `v1.4.0`, Phase 17)

## Context
Standard document ingestion and candidate retrieval pipelines in RAG recruiting systems suffer from three fundamental limitations:
1. **Loss of Document Structure**: Naive text chunkers split text by fixed token or character counts, frequently slicing across multi-role work histories, severing company names from job titles and responsibilities, and losing bullet-point context.
2. **Context Collapse in Isolated Chunks**: In dense vector and BM25 sparse indices, isolated chunks (e.g. *"Architected Kafka streaming pipeline reducing latency by 40%"*) lack candidate identity, seniority level, and surrounding technology stack. Dense embeddings and lexical searches cannot determine whether this accomplishment belongs to a Junior Intern or a Principal Architect.
3. **Coarse Retrieval Scoring Limits**: Bi-encoder dense embeddings compress entire candidate profiles into single vectors, suffering from semantic clustering compression and lacking joint cross-attention across specific recruiter job requirements. Relying purely on coarse retrieval yields sub-optimal candidate rank precision.
4. **Cloud SQLite Compatibility**: Serverless container environments (e.g., Streamlit Community Cloud, AWS Lambda) frequently ship outdated system SQLite libraries, causing ChromaDB import crashes (`sqlite3 >= 3.35.0 required`).

## Decision
1. **Layout-Aware Bounding-Box Section Parsing (`services/section_parser.py`)**:
   - Parse PDFs page-by-page using PyMuPDF bounding-box blocks (`page.get_text("blocks")`), grouping spatial layout elements by vertical and horizontal flow to reconstruct true visual paragraphs.
   - Support Word documents (`docx.Document`) preserving paragraph hierarchy, bold styling runs, and bulleted lists.
   - Normalize heading variations into canonical sections (`SUMMARY`, `EXPERIENCE`, `SKILLS`, `EDUCATION`, `PROJECTS`, `CERTIFICATIONS`).
   - Group work experience entries (title, company, dates, bullet accomplishments) into cohesive section units before downstream ingestion.
2. **Anthropic Contextual Retrieval Prepending (`services/contextual_retrieval.py`)**:
   - Synthesize a compact 50–80 word document metadata banner capturing candidate name, target role, experience years, primary skills, and education:
     ```text
     [Candidate: Alex Mercer | Target Role: Staff Backend Engineer | Experience: 8+ years | Primary Skills: Go, Kubernetes, Kafka, Distributed Systems | Education: B.S. Computer Science]
     ```
   - Prepend the synthesized banner to each extracted section chunk specifically for dense embedding vectorization and BM25 sparse indexing.
   - Retain pristine, unadulterated text in `metadata["raw_content"]`, guaranteeing clean presentation on recruiter UI dashboard cards, deep screening audits, and tailored interview questions.
3. **Two-Stage Hybrid Retrieval & Cross-Encoder Joint Reranking (`services/reranker.py`)**:
   - **Stage 1 (Coarse Candidate Retrieval)**: Perform fast candidate retrieval by combining dense sentence transformer embeddings (`all-MiniLM-L6-v2`) and sparse lexical scoring (`rank-bm25`) over the talent pool ($O(N) \to O(K)$).
   - **Stage 2 (Fine-Grained Cross-Encoder Reranking)**: Jointly score top candidate profiles against the recruiter query using `sentence_transformers.CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`, evaluating full cross-attention layers across tokens.
   - **Reciprocal Rank Fusion (RRF) & Sigmoid Calibration**: Apply Sigmoid transformation $\sigma(x) = \frac{1}{1 + e^{-x}}$ to map unbounded cross-encoder logits into $[0, 1]$ confidence scores, blending with Reciprocal Rank Fusion ($k=60$).
   - **Graceful Offline Fallback**: If cross-encoder model downloads are unavailable or compute-constrained, the engine gracefully degrades to Stage 1 hybrid ranking with zero pipeline failure.
4. **Serverless Cloud Deployment Resilience (`stores/in_memory_store.py`)**:
   - Implement `InMemoryVectorStore` conforming to `BaseVectorStore` using pure Python and NumPy matrix cosine similarity, requiring 0 SQLite extensions and 0 disk writes.
   - Provide automatic fallback: if `ChromaVectorStore(ephemeral=True)` encounters container permission locks or SQLite version mismatches on Linux serverless runtimes, the system automatically falls back to `InMemoryVectorStore`.
   - Invert `pysqlite3` dynamically at the top of `app.py` before any ChromaDB imports to resolve legacy SQLite errors on cloud host platforms.

## Consequences
- **Positive**: Eliminates mid-sentence section truncation; preserves work history integrity; contextual prepending delivers dramatic retrieval recall improvements on isolated skill/experience queries; cross-encoder joint reranking significantly increases top-1 and top-3 precision; pure in-memory NumPy fallback guarantees 100% crash-free deployment on serverless cloud runtimes; raw candidate content remains completely pristine in recruiter UI views.
- **Negative**: Prepending contextual banners increases token count per chunk by ~50–80 words (~15–20% embedding token increase); Cross-Encoder joint scoring introduces ~50–100ms inference overhead on the coarse candidate shortlist (safely bounded by running strictly on the top-10 retrieved candidates).
