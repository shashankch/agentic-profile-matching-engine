# ADR-010: Multi-Provider LLM Abstraction with Sarvam AI Indic Model Integration

## Status
Accepted (Implemented in `v1.1.0`)

## Context
Recruitment and candidate sourcing platforms frequently encounter multi-lingual and non-English resumes (such as Indic language profiles across Hindi, Tamil, Telugu, Kannada, etc.). Additionally, enterprise deployments require zero vendor lock-in across diverse inference providers (Groq, Google Gemini, OpenAI, and specialized regional models).

## Decision
Extend the unified LLM provider factory in `src/agentic_profile_matching/config.py` with first-class support for **Sarvam AI** (`sarvam-105b`, `sarvam-2b`):
- Wire Sarvam models via an OpenAI-compatible API bridge targeting `https://api.sarvam.ai/v1`.
- Provide native UI dropdown selection and `.env` pre-population (`SARVAM_API_KEY`).
- Integrate production-grade `invoke_structured` schema enforcement with multi-tier balanced-bracket JSON repair to ensure consistent structured parsing across all providers.

## Consequences
- **Positive**: Unlocks native multi-lingual Indic candidate screening capabilities; allows seamless toggling between ultra-fast Groq models, Google Gemini Pro, OpenAI GPT-4o, and Sarvam 105B with zero code changes.
- **Negative**: Requires managing API keys across multiple third-party providers.
