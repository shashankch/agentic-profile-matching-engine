"""
Unit tests for Anthropic Contextual Retrieval Prepending (Deliverable 17.2).
Tests document context building, situational prepending, and raw content preservation.
"""

from unittest.mock import MagicMock
from agentic_profile_matching.services.section_parser import ParsedSectionChunk
from agentic_profile_matching.services.contextual_retrieval import (
    ContextualEnricher,
    ANTHROPIC_CONTEXTUAL_PROMPT_TEMPLATE,
)


def test_build_document_context():
    """Verifies format and fields of deterministic document context header."""
    metadata = {
        "candidate_name": "Tony Stark",
        "target_role": "Chief Robotics Architect",
        "experience_years": 15,
        "skills": ["Robotics", "AI", "Python", "C++", "Embedded Systems"],
        "education": "Ph.D. in Physics - MIT",
    }

    doc_context = ContextualEnricher.build_document_context(metadata)
    assert "[Candidate: Tony Stark" in doc_context
    assert "Experience: 15 Years" in doc_context
    assert "Robotics" in doc_context
    assert "Ph.D. in Physics" in doc_context


def test_enrich_chunks_prepending():
    """Verifies that contextual enricher situates chunks while preserving raw_content."""
    enricher = ContextualEnricher()
    metadata = {
        "candidate_name": "Diana Prince",
        "title": "Principal Security Engineer",
        "experience_years": 10,
        "skills": ["Cybersecurity", "Zero Trust", "Cloud Architecture", "Go"],
        "education": "M.S. Cybersecurity",
    }

    raw_bullet = "Architected zero-trust authentication gateway handling 200k daily active users."
    chunk = ParsedSectionChunk(
        section_title="WORK EXPERIENCE",
        section_type="experience",
        content=raw_bullet,
        raw_content=raw_bullet,
        role="Lead Security Architect",
        company="Global Defense Tech",
    )

    enriched_chunks = enricher.enrich_chunks([chunk], metadata)
    assert len(enriched_chunks) == 1

    ch = enriched_chunks[0]
    # Situated content should have the document context preamble:
    assert "[Candidate: Diana Prince" in ch.content
    assert "Section: WORK EXPERIENCE | Role: Lead Security Architect" in ch.content
    assert raw_bullet in ch.content

    # raw_content should remain untouched for clean UI display:
    assert ch.raw_content == raw_bullet
    assert ch.metadata["is_contextualized"] is True


def test_anthropic_prompt_template_and_llm_fallback():
    """Tests prompt formatting and opt-in LLM enrichment with graceful fallback."""
    assert "{document_content}" in ANTHROPIC_CONTEXTUAL_PROMPT_TEMPLATE
    assert "{chunk_content}" in ANTHROPIC_CONTEXTUAL_PROMPT_TEMPLATE

    enricher = ContextualEnricher()
    chunk = ParsedSectionChunk(
        section_title="SUMMARY",
        section_type="summary",
        content="Passionate full-stack developer.",
        raw_content="Passionate full-stack developer.",
    )

    # Mock LLM success
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "This chunk describes Diana's career objective in security."
    enriched = enricher.enrich_chunk_with_llm(mock_llm, "Full resume text...", chunk)
    assert "This chunk describes" in enriched.content
    assert enriched.raw_content == "Passionate full-stack developer."

    # Mock LLM failure gracefully returns original chunk
    failing_llm = MagicMock()
    failing_llm.invoke.side_effect = RuntimeError("API Offline")
    fallback_chunk = enricher.enrich_chunk_with_llm(failing_llm, "Full resume text...", chunk)
    assert fallback_chunk.content == chunk.content
