"""Services package for agentic profile matching engine."""

from agentic_profile_matching.services.ingestion_service import IngestionService
from agentic_profile_matching.services.section_parser import (
    SectionParser,
    ParsedSectionChunk,
    CANONICAL_SECTION_MAP,
)
from agentic_profile_matching.services.contextual_retrieval import (
    ContextualEnricher,
    ANTHROPIC_CONTEXTUAL_PROMPT_TEMPLATE,
)
from agentic_profile_matching.services.reranker import (
    CrossEncoderReranker,
    compute_rrf_scores,
)

__all__ = [
    "IngestionService",
    "SectionParser",
    "ParsedSectionChunk",
    "CANONICAL_SECTION_MAP",
    "ContextualEnricher",
    "ANTHROPIC_CONTEXTUAL_PROMPT_TEMPLATE",
    "CrossEncoderReranker",
    "compute_rrf_scores",
]
