"""run_id, run.json schema, paths, registry index (PLAN §3.2).

Notebooks and reports read results only through `gsp.store.load_registry()` / `load_metrics()`.
"""

from .index import build_index, load_metrics, load_registry

__all__ = ["build_index", "load_metrics", "load_registry"]
