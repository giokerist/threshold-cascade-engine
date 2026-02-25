# conftest.py — cascade_engine package root
#
# Ensures the *parent* of cascade_engine is on sys.path so that
# "from cascade_engine.propagation import ..." works in all test files
# when pytest is invoked from within the cascade_engine directory.
#
# Usage (from project root, i.e. parent of cascade_engine/):
#   pytest cascade_engine/tests/ -v
#
# Usage (from inside cascade_engine/):
#   pytest tests/ -v

import sys
from pathlib import Path

# Insert the parent directory of cascade_engine so the package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
