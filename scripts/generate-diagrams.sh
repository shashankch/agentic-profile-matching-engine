#!/usr/bin/env bash
set -euo pipefail

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="python3"
fi

if [[ "${1:-}" == "--check" ]]; then
    echo "🔍 Checking diagram drift..."
    "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_diagrams.py"
    if ! git diff --quiet "${REPO_ROOT}/docs/assets/diagrams"; then
        echo "❌ Diagram drift detected! Run './scripts/generate-diagrams.sh' to update."
        exit 1
    fi
    echo "✅ No diagram drift detected. All diagrams are up to date."
else
    echo "🎨 Regenerating documentation diagrams with Diagrams..."
    "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_diagrams.py"
    echo "✅ Diagrams regenerated successfully."
fi
