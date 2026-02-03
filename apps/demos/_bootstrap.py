"""Local import bootstrap for demos."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_core_path() -> None:
    root = Path(__file__).resolve().parents[2]
    core = root / "packages" / "core"
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
