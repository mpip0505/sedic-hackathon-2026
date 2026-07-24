#!/usr/bin/env bash
# setup.sh — create the venv, install deps, and verify the stub predictor runs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"

echo ">> Creating virtual environment (.venv)"
"$PYTHON" -m venv .venv

# shellcheck disable=SC1091
source .venv/bin/activate

echo ">> Upgrading pip"
python -m pip install --upgrade pip

echo ">> Installing requirements (this can take a while — torch is large)"
python -m pip install -r requirements.txt

echo ">> Verifying the stub predictor runs"
python -m src.inference.predict --source none --stub

echo ""
echo ">> Setup complete."
echo "   Activate with:  source .venv/bin/activate"
echo "   Launch GUI:     streamlit run app/app.py"
