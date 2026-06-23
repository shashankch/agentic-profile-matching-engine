# Agentic Profile Matching Engine: Project Roadmap

This document outlines the phased plan for building the Agentic Profile Matching Engine, along with key upcoming backlog features.

---

## 📍 Core Implementation Phases

1. **Phase 1: Environment & Setup** [Completed]
   - Install dependencies (`langgraph`, `streamlit`, `langchain-groq`, `langchain-google-genai`).
   - Create a local `.env` file for credentials (`GROQ_API_KEY`, `GEMINI_API_KEY`).

2. **Phase 2: Dataset Ingestion & Retrieval** [Completed]
   - Generate synthetic candidate resume files (PDF, DOCX, TXT).
   - Ingest files, generate local embeddings, and populate ChromaDB/BM25 indexes.

3. **Phase 3: LangGraph Agent & State Machine** [Completed]
   - Design the `AgentState` schema and setup conversational graph memory.
   - Implement workflow nodes (parsing, search, deep screening, report generation, and human feedback interrupts).

4. **Phase 4: Assessment & Screening Tools** [Completed]
   - Build custom agent tools for requirements extraction, multi-profile side-by-side comparison matrices, and candidate-specific interview question generation.

5. **Phase 5: Streamlit UI & End-to-End Validation** [Completed]
   - Create the frontend dashboard (`app.py`) for dual-pane chat and shortlist inspection.
   - Test conversational refinement loops against test recruitment scenarios.

---

## 🔮 Future Backlog (Post-Release)

- **Multi-Agent Consensus Loop**: Incorporate independent agent personas (e.g., Technical Screener vs. HR/Sourcing Screener) to run an internal review before final recommendation.
- **Bias & Fairness Auditing**: Add an automated checker to audit Job Descriptions for inclusive language and flag potential constraint biases.
