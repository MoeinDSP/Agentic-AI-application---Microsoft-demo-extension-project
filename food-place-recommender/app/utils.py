from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


class FileStore:
    """Loads, caches, and persists files on disk."""

    @classmethod
    @lru_cache(maxsize=None)
    def load_yaml(cls, path: Path) -> Any:
        """Return parsed YAML content for the given file path."""
        resolved = path.resolve()
        with resolved.open(encoding="utf-8") as f:
            return yaml.safe_load(f)
