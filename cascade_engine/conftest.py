# conftest.py — project root
#
# Ensures that the cascade_engine source directory is on sys.path when
# pytest is invoked from the project root, so all source imports resolve
# without requiring a package install.
#
# Usage:
#   cd cascade_engine/
#   pytest tests/ -v
#   pytest tests/test_tier2.py -v
#   pytest tests/test_propagation.py -v

import sys
from pathlib import Path

# Insert project root (the directory containing this file) at the front of
# sys.path so that "from propagation import ..." works in all test files.
sys.path.insert(0, str(Path(__file__).parent))
