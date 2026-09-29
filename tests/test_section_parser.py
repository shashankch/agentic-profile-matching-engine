"""
Unit tests for layout-aware SectionParser (Deliverable 17.1).
Tests PyMuPDF block parsing, section boundary extraction, and role hierarchy preservation.
"""

from pathlib import Path
from agentic_profile_matching.services.section_parser import (
    SectionParser,
    ParsedSectionChunk,
    CANONICAL_SECTION_MAP,
)


def test_section_parser_text_parsing():
    """Tests parsing raw resume text into canonical section chunks."""
    parser = SectionParser(max_chunk_chars=500)
    sample_resume = """
John Doe
Staff Distributed Systems Engineer

SUMMARY
Experienced software engineer specializing in high-throughput cloud architectures and Kafka streaming.

TECHNICAL SKILLS
Python, Go, Kubernetes, Docker, Apache Kafka, AWS, PostgreSQL, Redis

PROFESSIONAL EXPERIENCE
Staff Systems Engineer at CloudCorp (2021 - Present)
- Designed and operated real-time event streaming pipeline processing 100k events/sec.
- Mentored junior engineers and reduced p99 latency by 35%.

Senior Backend Engineer at TechCorp (2018 - 2021)
- Built microservices in Go and Python using gRPC and Docker.

EDUCATION
B.S. in Computer Science - University of California (2014 - 2018)
    """

    chunks = parser.parse_text(sample_resume)
    assert len(chunks) >= 4

    section_types = [c.section_type for c in chunks]
    assert "summary" in section_types
    assert "skills" in section_types
    assert "experience" in section_types
    assert "education" in section_types

    # Verify experience section has role information
    exp_chunks = [c for c in chunks if c.section_type == "experience"]
    assert len(exp_chunks) >= 1
    assert any("CloudCorp" in c.content or "TechCorp" in c.content for c in exp_chunks)


def test_section_parser_pdf_blocks():
    """Tests layout block parsing on real PDF resume from data/resumes."""
    parser = SectionParser()
    pdf_files = list(Path("data/resumes").glob("*.pdf"))
    if not pdf_files:
        return

    sample_pdf = pdf_files[0]
    chunks = parser.parse_pdf(str(sample_pdf))

    assert len(chunks) > 0
    for ch in chunks:
        assert isinstance(ch, ParsedSectionChunk)
        assert ch.section_title
        assert ch.content
        assert ch.raw_content


def test_canonical_section_map():
    """Verifies that all standard section headers map to normalized canonical types."""
    assert CANONICAL_SECTION_MAP["WORK EXPERIENCE"] == "experience"
    assert CANONICAL_SECTION_MAP["TECHNICAL SKILLS"] == "skills"
    assert CANONICAL_SECTION_MAP["EDUCATION"] == "education"
    assert CANONICAL_SECTION_MAP["SUMMARY"] == "summary"
    assert CANONICAL_SECTION_MAP["PROJECTS"] == "projects"
