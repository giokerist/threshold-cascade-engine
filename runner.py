"""Repository-level CLI entrypoint for the cascade engine.

This wrapper preserves the documented invocation style:

    python runner.py <config.json> [--output-dir results/]

It delegates execution to :mod:`cascade_engine.runner`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from cascade_engine.runner import main


def _rewrite_config_path_arg(argv: list[str]) -> list[str]:
    """Rewrite config argument to ``cascade_engine/<name>`` when needed.

    README examples use config paths like ``config_erdos_renyi.json`` from the
    repository root, while config files live under ``cascade_engine/``.
    """
    if len(argv) < 2:
        return argv

    candidate = Path(argv[1])
    if candidate.exists():
        return argv

    alt = Path("cascade_engine") / candidate
    if alt.exists():
        out = list(argv)
        out[1] = str(alt)
        return out

    return argv


if __name__ == "__main__":
    sys.argv = _rewrite_config_path_arg(sys.argv)
from cascade_engine.runner import main


if __name__ == "__main__":
    main()
