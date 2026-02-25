"""Repository-level CLI entrypoint for the cascade engine.

This wrapper preserves the documented invocation style:

    python runner.py <config.json> [--output-dir results/]

It delegates execution to :mod:`cascade_engine.runner`.
"""

from cascade_engine.runner import main


if __name__ == "__main__":
    main()
