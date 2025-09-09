"""CLI entry-point for ``python -m fleetmix``.
Keeps the CLI runnable even without the console-script wrapper.
"""

from fleetmix.app import app

if __name__ == "__main__":
    app()
