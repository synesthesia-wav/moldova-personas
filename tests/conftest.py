"""Test configuration for local imports."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = PROJECT_ROOT / "packages" / "core"

if str(CORE_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_PATH))
