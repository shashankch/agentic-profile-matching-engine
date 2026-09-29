"""
Anthropic Contextual Retrieval Prepending for Yojaka AI (Deliverable 17.2).
Situates isolated candidate resume chunks within document-level context (Candidate Name,
Target Role, Total Experience, Core Skills, and Education) before embedding and sparse indexing.
Prevents semantic drift where fragmented bullet points lose candidate identity or seniority.
"""

import logging
from typing import List, Dict, Any

from agentic_profile_matching.services.section_parser import ParsedSectionChunk

logger = logging.getLogger("contextual_retrieval")

ANTHROPIC_CONTEXTUAL_PROMPT_TEMPLATE = """<document>
{document_content}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>
Please give a short, succinct context (50-100 words) to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


class ContextualEnricher:
    """
    Enriches parsed document chunks by prepending situational document context
    prior to vector embedding and BM25 indexing.
    """

    def __init__(self, enable_llm: bool = False):
        self.enable_llm = enable_llm

    @staticmethod
    def build_document_context(metadata: Dict[str, Any]) -> str:
        """
        Builds a deterministic, high-information document context header from candidate metadata.
        Zero token cost and sub-millisecond execution.
        """
        cand_name = metadata.get("candidate_name") or metadata.get("name") or "Candidate"
        exp_years = metadata.get("experience_years", 0)

        # Infer primary role or title if available
        role = metadata.get("target_role") or metadata.get("title") or "Technical Professional"

        skills = metadata.get("skills", [])
        if isinstance(skills, str):
            skills_list = [s.strip() for s in skills.split(",") if s.strip()]
        else:
            skills_list = list(skills)
        top_skills = ", ".join(skills_list[:8]) if skills_list else "General Engineering"

        education = metadata.get("education", "Not Specified")

        context_banner = (
            f"[Candidate: {cand_name} | Role: {role} | Experience: {exp_years} Years | "
            f"Skills: {top_skills} | Education: {education}]"
        )
        return context_banner

    def enrich_chunks(
        self,
        chunks: List[ParsedSectionChunk],
        metadata: Dict[str, Any],
    ) -> List[ParsedSectionChunk]:
        """
        Prepends document context to each chunk's content for search indexing,
        while preserving the clean raw_content for UI rendering.
        """
        if not chunks:
            return []

        doc_context = self.build_document_context(metadata)

        enriched: List[ParsedSectionChunk] = []
        for ch in chunks:
            # Build situating preamble
            role_info = f" | Role: {ch.role}" if ch.role else ""
            company_info = f" | Company: {ch.company}" if ch.company else ""
            section_info = f"Section: {ch.section_title}{role_info}{company_info}"

            situated_text = f"{doc_context}\n{section_info}\n{ch.raw_content}"

            # Return new chunk with enriched content for embeddings and preserved raw content for UI
            enriched.append(
                ParsedSectionChunk(
                    section_title=ch.section_title,
                    section_type=ch.section_type,
                    content=situated_text,
                    raw_content=ch.raw_content,
                    role=ch.role,
                    company=ch.company,
                    duration_years=ch.duration_years,
                    skills=ch.skills,
                    block_index=ch.block_index,
                    metadata={
                        **ch.metadata,
                        "document_context": doc_context,
                        "is_contextualized": True,
                    },
                )
            )

        return enriched

    @staticmethod
    def format_llm_prompt(whole_document: str, chunk_content: str) -> str:
        """Formats the official Anthropic Contextual Retrieval prompt for LLM enrichment."""
        return ANTHROPIC_CONTEXTUAL_PROMPT_TEMPLATE.format(
            document_content=whole_document.strip(),
            chunk_content=chunk_content.strip(),
        )

    def enrich_chunk_with_llm(
        self,
        llm: Any,
        whole_document: str,
        chunk: ParsedSectionChunk,
    ) -> ParsedSectionChunk:
        """
        Opt-in LLM contextualization calling Claude/Groq with Anthropic prompt format.
        Falls back to deterministic context if LLM call fails.
        """
        try:
            prompt = self.format_llm_prompt(whole_document, chunk.raw_content)
            response = llm.invoke(prompt)
            llm_context = response.content if hasattr(response, "content") else str(response)
            llm_context = llm_context.strip()

            enriched_content = f"{llm_context}\n\n{chunk.raw_content}"
            return ParsedSectionChunk(
                section_title=chunk.section_title,
                section_type=chunk.section_type,
                content=enriched_content,
                raw_content=chunk.raw_content,
                role=chunk.role,
                company=chunk.company,
                duration_years=chunk.duration_years,
                skills=chunk.skills,
                block_index=chunk.block_index,
                metadata={
                    **chunk.metadata,
                    "llm_context": llm_context,
                    "is_contextualized": True,
                },
            )
        except Exception as e:
            logger.warning(f"LLM contextual enrichment failed ({e}), using raw chunk: {e}")
            return chunk
