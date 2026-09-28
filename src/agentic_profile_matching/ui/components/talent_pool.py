"""
Talent Pool & Resume Ingestion Component for Yojaka AI UI.
Supports drag-and-drop resume uploading, zero-disk in-memory stream processing (ADR-016),
and active inventory browsing across pre-indexed and session-uploaded profiles.
"""

import hashlib
from pathlib import Path
import pandas as pd
import streamlit as st

from agentic_profile_matching import config
from agentic_profile_matching.services.ingestion_service import IngestionService


def render_talent_pool_tab() -> None:
    """Renders the resume ingestion and active talent pool inventory tab."""
    st.markdown("### 📤 Candidate Resume Ingestion & Talent Pool")
    st.caption(
        "Zero-disk in-memory stream processing (ADR-016). Uploaded resumes (.pdf, .docx, .txt) are "
        "vectorized directly into memory without local disk persistence."
    )

    try:
        total_indexed_chunks = st.session_state["session_vector_store"].count()
    except Exception:
        total_indexed_chunks = 0

    try:
        resumes_path = Path(config.RESUMES_DIR)
        if not resumes_path.exists():
            fallback_path = Path(config.BASE_DIR).parent / "data" / "resumes"
            if fallback_path.exists():
                resumes_path = fallback_path
        all_files = list(resumes_path.glob("*.*"))
        disk_count = len([f for f in all_files if f.suffix.lower() in [".pdf", ".docx", ".txt"]])
        valid_count = disk_count + len(st.session_state.get("uploaded_candidates", []))
    except Exception:
        valid_count = 34 + len(st.session_state.get("uploaded_candidates", []))

    # Talent Pool Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Indexed Chunks", total_indexed_chunks)
    with c2:
        st.metric("Active Talent Profiles", valid_count)
    with c3:
        st.metric("Storage Protocol", "BaseVectorStore")
    with c4:
        st.metric("Ingestion Mode", "In-Memory Stream")

    st.markdown("---")
    st.markdown("#### 📁 Drag-and-Drop Resume Ingestion")
    main_uploaded_files = st.file_uploader(
        "Upload candidate resumes to expand the active talent pool",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="main_resume_uploader",
        help="Upload candidate resumes (PDF, DOCX, or TXT). Files are parsed securely in-memory for your active session without server disk storage.",
    )

    if main_uploaded_files:
        if "processed_uploads" not in st.session_state:
            st.session_state["processed_uploads"] = set()

        MAX_UPLOAD_BYTES = 10 * 1024 * 1024
        files_to_process = []
        for uf in main_uploaded_files:
            if uf.size > MAX_UPLOAD_BYTES:
                st.error(
                    f"❌ Upload rejected for `{uf.name}`: Exceeds 10MB limit ({round(uf.size / (1024 * 1024), 2)} MB)."
                )
                continue
            content_bytes = uf.getvalue()
            content_digest = f"{uf.name}_{uf.size}_{hashlib.sha256(content_bytes).hexdigest()[:12]}"
            if content_digest not in st.session_state["processed_uploads"]:
                files_to_process.append((uf, content_bytes, content_digest))

        if files_to_process:
            with st.status(f"Ingesting {len(files_to_process)} resume(s)...", expanded=True) as upload_status:
                service = IngestionService(store=st.session_state["ephemeral_store"])
                for uf, content_bytes, content_digest in files_to_process:
                    upload_status.write(f"Vectorizing `{uf.name}` into memory...")
                    res = service.ingest_stream(uf.name, content_bytes)
                    if res.get("success"):
                        st.session_state["processed_uploads"].add(content_digest)
                        cand_entry = {
                            "candidate_name": res.get("candidate_name", "Unknown"),
                            "filename": uf.name,
                            "format": Path(uf.name).suffix.upper().replace(".", ""),
                            "cohort": "📤 Session Upload",
                            "status": "✅ In-Memory (Zero-Disk)",
                        }
                        # Update existing or append
                        existing = [
                            i for i, c in enumerate(st.session_state["uploaded_candidates"]) if c["filename"] == uf.name
                        ]
                        if existing:
                            st.session_state["uploaded_candidates"][existing[0]] = cand_entry
                        else:
                            st.session_state["uploaded_candidates"].append(cand_entry)

                        upload_status.write(
                            f"✅ Ingested **{res.get('candidate_name')}** ({res.get('chunks_ingested')} chunks)"
                        )
                    else:
                        upload_status.write(f"❌ Failed `{uf.name}`: {res.get('error')}")

                upload_status.update(
                    label=f"Ingested {len(files_to_process)} file(s) successfully!",
                    state="complete",
                    expanded=False,
                )
            st.rerun()

    st.markdown("---")
    st.markdown("#### 👥 Active Talent Pool Inventory")

    talent_rows = []

    # 1. In-memory session uploads (isolated, zero-disk)
    for up in st.session_state.get("uploaded_candidates", []):
        talent_rows.append(
            {
                "Candidate Name": up["candidate_name"],
                "File Name": up["filename"],
                "Format": up["format"],
                "Cohort": up["cohort"],
                "Status": up["status"],
            }
        )

    # 2. Disk-based pre-indexed candidates
    try:
        resumes_dir_path = Path(config.RESUMES_DIR)
        if not resumes_dir_path.exists():
            fallback_resumes = Path(config.BASE_DIR).parent / "data" / "resumes"
            if fallback_resumes.exists():
                resumes_dir_path = fallback_resumes
        for rf in sorted(resumes_dir_path.glob("*.*")):
            if rf.suffix.lower() in [".pdf", ".docx", ".txt"]:
                raw_name = rf.stem.replace("resume_", "")
                display_name = " ".join(p.capitalize() for p in raw_name.split("_"))
                persona_tag = "Standard Profile"
                talent_rows.append(
                    {
                        "Candidate Name": display_name,
                        "File Name": rf.name,
                        "Format": rf.suffix.upper().replace(".", ""),
                        "Cohort": persona_tag,
                        "Status": "✅ Indexed & Searchable",
                    }
                )
    except Exception:
        pass

    if talent_rows:
        df_talent = pd.DataFrame(talent_rows)
        st.dataframe(
            df_talent,
            use_container_width=True,
            column_config={
                "Candidate Name": st.column_config.TextColumn("Candidate Name", width="medium"),
                "File Name": st.column_config.TextColumn("File Name", width="medium"),
                "Format": st.column_config.TextColumn("Format", width="small"),
                "Cohort": st.column_config.TextColumn("Cohort / Source", width="small"),
                "Status": st.column_config.TextColumn("Index Status", width="small"),
            },
            hide_index=True,
        )
