"""
Ethnocultural cross-tab loader.

Loads official 2024 census ethnocultural cross-tabs extracted from the
NBS XLSX (mother tongue by ethnicity, religion by ethnicity).
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Any


logger = logging.getLogger(__name__)

DEFAULT_TABLE_PATH = Path(__file__).parent.parent / "config" / "ethnocultural_tables_2024.json"


def strict_mode_enabled() -> bool:
    """Return True if strict ethnocultural mode is enabled."""
    return os.getenv("MOLDOVA_PERSONAS_STRICT_ETHNOCULTURAL", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


@lru_cache
def load_tables(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load ethnocultural tables from JSON, if present."""
    table_path = Path(path) if path else DEFAULT_TABLE_PATH
    if not table_path.exists():
        return None

    try:
        with open(table_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Failed to load ethnocultural tables: {exc}")
        return None


def get_language_distribution(ethnicity: str) -> Optional[Dict[str, float]]:
    """Return mother tongue distribution for an ethnicity if available."""
    data = load_tables()
    if not data:
        return None
    return data.get("language_by_ethnicity", {}).get(ethnicity)


def get_religion_distribution(ethnicity: str) -> Optional[Dict[str, float]]:
    """Return religion distribution for an ethnicity if available."""
    data = load_tables()
    if not data:
        return None
    return data.get("religion_by_ethnicity", {}).get(ethnicity)


def get_language_distribution_total() -> Optional[Dict[str, float]]:
    """Return national mother tongue distribution if available."""
    data = load_tables()
    if not data:
        return None
    return data.get("language_distribution")


def get_religion_distribution_total() -> Optional[Dict[str, float]]:
    """Return national religion distribution if available."""
    data = load_tables()
    if not data:
        return None
    return data.get("religion_distribution")
