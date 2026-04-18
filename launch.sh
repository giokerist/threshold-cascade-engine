#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────
# launch.sh  —  Start the Cascade Propagation Engine Dashboard
# ────────────────────────────────────────────────────────────
# Usage:
#   chmod +x launch.sh
#   ./launch.sh
#
# On Windows:   python -m streamlit run app.py
# ────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  ⚡  Cascade Propagation Engine"
echo "  ─────────────────────────────────────────────"
echo "  Starting dashboard at http://localhost:8501"
echo ""

# Install dependencies if not already present
if ! python -c "import streamlit" 2>/dev/null; then
  echo "  Installing dependencies…"
  pip install -r requirements_gui.txt
fi

python -m streamlit run app.py \
  --server.headless false \
  --theme.base dark \
  --theme.backgroundColor "#0d1117" \
  --theme.secondaryBackgroundColor "#161b22" \
  --theme.textColor "#e6edf3" \
  --theme.primaryColor "#58a6ff"
