# Contributing to Agentic Profile Matching Engine

Thank you for your interest in contributing to the **Agentic Profile Matching Engine**! We welcome contributions that improve features, enhance test coverage, optimize RAG retrieval performance, or refine developer tooling.

---

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **Package Manager**: Standard `pip` (or `uv` for ultra-fast dependency resolution)
- **Git**: For version control and submitting Pull Requests
- **API Keys**: Groq API Key and/or Google Gemini API Key (free developer tiers supported)

---

## 🛠️ Local Environment Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/shashankch/agentic-profile-matching-engine.git
   cd agentic-profile-matching-engine
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   # Using Python venv
   python3 -m venv .venv
   source .venv/bin/activate

   # Or using uv (recommended)
   uv venv
   source .venv/bin/activate
   ```

3. **Install Package in Editable Mode**
   ```bash
   pip install -e .
   ```

4. **Configure Environment Secrets**
   Copy the example environment configuration:
   ```bash
   cp .env.example .env  # Or create a .env file
   ```
   Add your API keys to `.env`:
   ```env
   GROQ_API_KEY="your-groq-api-key"
   GEMINI_API_KEY="your-gemini-api-key"
   USE_MCP=False  # Set to True to test Model Context Protocol JSON-RPC mode
   ```

---

## 🧪 Running Tests & Validation

Before submitting any code changes, ensure all unit tests pass and code quality checks pass.

1. **Run Unit Test Suite**
   ```bash
   pytest tests/ -v
   ```

2. **Run Quality Lints & Formatter**
   We enforce code formatting via [Ruff]:
   ```bash
   # Check for lint violations
   ruff check .

   # Automatically format code
   ruff format .
   ```

3. **Install Local Pre-Commit Hooks (Optional but Recommended)**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

---

## 🏃 Running Pipeline & Streamlit App locally

Verify full end-to-end functionality locally:

```bash
# 1. Generate local mock candidate dataset (31 profiles)
python -m agentic_profile_matching.generate_dataset

# 2. Ingest candidate resumes into ChromaDB & BM25 index
python -m agentic_profile_matching.resume_rag

# 3. Launch interactive Streamlit GUI
streamlit run src/agentic_profile_matching/app.py

# 4. Run automated test scenarios suite
python -m agentic_profile_matching.run_scenarios
```

---

## 📐 Development & Architectural Guidelines

Please review our [Engineering Conventions](CONVENTIONS.md) (`CONVENTIONS.md`) before submitting non-trivial PRs.

Key expectations:
- **Clean Separation**: Keep domain logic decoupled from UI/RPC handlers.
- **Explicit Types**: Use type annotations for all function interfaces.
- **Structured Logging**: Use `get_logger()` and `@trace_node` for observability; avoid unformatted `print()` calls.
- **Error Bounds**: Avoid swallowing exceptions without logging or setting state warning flags.
- **Modular Commits**: Use [Conventional Commits](https://www.conventionalcommits.org/) for your commit messages:
  - `feat(component): add new feature`
  - `fix(component): resolve bug in module`
  - `refactor(component): clean up implementation`
  - `docs(component): update documentation`
  - `test(component): add unit test coverage`

---

## 📥 Submitting a Pull Request (PR)

1. **Create a Feature Branch**
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. **Commit Your Changes**
   Follow Conventional Commits style.
3. **Verify Lints & Tests**
   Ensure `pytest tests/ -v` passes and `ruff check src/ tests/` shows zero warnings.
4. **Update Version & Documentation**
   - Increment version in `pyproject.toml` and add release notes to `CHANGELOG.md`.
   - Verify `README.md` features, directory tree, and setup instructions match latest implementation.
5. **Push & Open PR**
   Push to your branch and open a PR against the `main` branch with a clear description of changes.

Thank you for contributing! 🌟
