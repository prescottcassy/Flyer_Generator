"""Backend package initializer to allow `import backend`.

This file can be empty; presence makes `backend` importable as a package
so uvicorn can load `backend.server:app` when run from the repository root.
"""

__all__ = ["server"]
