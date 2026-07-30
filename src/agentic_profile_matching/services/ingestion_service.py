import logging
from pathlib import Path
from typing import Dict, Any, Optional

from agentic_profile_matching.resume_rag import (
    ResumeRAGPipeline,
    MetadataExtractor,
    ResumeChunker,
)
from agentic_profile_matching.fs_tools import (
    read_file as direct_read_file,
    list_files as direct_list_files,
)

logger = logging.getLogger("ingestion_service")


class IngestionService:
    """
    Business logic service for candidate resume ingestion into the vector store.
    Decouples protocol handlers (e.g., FastMCP filesystem server) from RAG ingestion mechanics.
    """

    def __init__(self, pipeline: Optional[ResumeRAGPipeline] = None):
        self._pipeline = pipeline

    @property
    def pipeline(self) -> ResumeRAGPipeline:
        """Lazily initialize the underlying vector store RAG pipeline if not injected."""
        if self._pipeline is None:
            self._pipeline = ResumeRAGPipeline()
        return self._pipeline

    def ingest_file(self, filepath: str) -> Dict[str, Any]:
        """
        Ingest a single candidate resume file (PDF, TXT, DOCX) into the vector database.

        Args:
            filepath: Absolute or relative path to the resume file.

        Returns:
            Dict containing ingestion status, metadata, and ingested chunk count.
        """
        path = Path(filepath).resolve()
        if not path.exists() or not path.is_file():
            logger.warning(f"File not found for ingestion: {filepath}")
            return {
                "success": False,
                "filepath": str(path),
                "error": f"File not found: {filepath}",
            }

        data = direct_read_file(str(path))
        if not data.get("success"):
            logger.error(f"Failed to read file for ingestion: {filepath}")
            return {
                "success": False,
                "filepath": str(path),
                "error": data.get("error", "Read failed"),
            }

        text = data.get("content", "")
        if not text or not text.strip():
            logger.warning(f"Empty content in file for ingestion: {filepath}")
            return {"success": False, "filepath": str(path), "error": "Empty content"}

        extractor = MetadataExtractor()
        chunker = ResumeChunker()

        filename = path.name
        meta = extractor.extract(filename, text)
        chunks = chunker.chunk(text)

        logger.info(
            f"Ingesting file '{filename}' - Candidate: {meta['candidate_name']}, "
            f"Exp: {meta['experience_years']} yrs, Skills: {len(meta['skills'])}, Chunks: {len(chunks)}"
        )

        added_chunks = 0
        pipeline = self.pipeline

        for idx, ch in enumerate(chunks):
            emb = pipeline.embedder.encode(ch["content"]).tolist()
            chunk_id = f"{filename}_{idx}"
            chunk_meta = {
                "candidate_name": meta["candidate_name"],
                "skills": ", ".join(meta["skills"]),
                "experience_years": int(meta["experience_years"]),
                "education": meta["education"],
                "resume_path": str(path),
                "filename": filename,
                "section": ch["section"],
            }

            pipeline.collection.upsert(
                ids=[chunk_id],
                documents=[ch["content"]],
                embeddings=[emb],
                metadatas=[chunk_meta],
            )
            added_chunks += 1

        return {
            "success": True,
            "filepath": str(path),
            "filename": filename,
            "candidate_name": meta["candidate_name"],
            "chunks_ingested": added_chunks,
            "metadata": meta,
        }

    def ingest_directory(self, directory: str) -> Dict[str, Any]:
        """
        Ingest all supported resume files from a directory.

        Args:
            directory: Absolute or relative path to directory.

        Returns:
            Dict containing overall summary of directory ingestion results.
        """
        dir_path = Path(directory).resolve()
        if not dir_path.exists() or not dir_path.is_dir():
            logger.warning(f"Directory not found for ingestion: {directory}")
            return {
                "success": False,
                "directory": str(dir_path),
                "error": f"Directory not found: {directory}",
            }

        files = direct_list_files(str(dir_path))
        results = []
        successful = 0
        failed = 0

        for f in files:
            res = self.ingest_file(f["path"])
            results.append(res)
            if res.get("success"):
                successful += 1
            else:
                failed += 1

        return {
            "success": True,
            "directory": str(dir_path),
            "total_files": len(files),
            "successful": successful,
            "failed": failed,
            "results": results,
        }
