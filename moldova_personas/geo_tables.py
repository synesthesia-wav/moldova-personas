"""
Geography data helpers for strict geo validity.

Loads optional district-level distributions when available and
provides region lookup for districts using the official district list.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from .pxweb_fetcher import NBSDataManager

logger = logging.getLogger(__name__)


def strict_geo_enabled() -> bool:
    """
    Return True if strict geo mode is enabled.

    Default is strict (True) to avoid fabricated locality/district data.
    Set MOLDOVA_PERSONAS_STRICT_GEO=0 to allow legacy locality sampling.
    """
    value = os.getenv("MOLDOVA_PERSONAS_STRICT_GEO")
    if value is None or value.strip() == "":
        return True
    return value.strip().lower() in {"1", "true", "yes"}


def _normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\bmunicipiul\b", "mun", text)
    text = re.sub(r"\bmunicipiu\b", "mun", text)
    text = re.sub(r"\bmun\.\s*", "mun ", text)
    text = re.sub(r"[.,]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _is_total_label(text: str) -> bool:
    norm = _normalize(text)
    return any(
        pattern in norm
        for pattern in (
            "total",
            "total pe tara",
            "total general",
            "total population",
            "populatia totala",
            "populatia total",
            "whole country",
            "country",
        )
    )


@lru_cache
def load_districts_by_region() -> Dict[str, list]:
    """Load district lists by region (official names)."""
    path = Path(__file__).parent.parent / "config" / "districts_by_region.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache
def get_district_region_map() -> Dict[str, Tuple[str, str]]:
    """Return mapping from normalized district name -> (canonical, region)."""
    mapping: Dict[str, Tuple[str, str]] = {}
    districts_by_region = load_districts_by_region()
    for region, districts in districts_by_region.items():
        for district in districts:
            mapping[_normalize(district)] = (district, region)
    return mapping


def derive_region_marginal_from_district(
    district_distribution: Dict[str, float]
) -> Optional[Dict[str, float]]:
    """
    Derive region marginal distribution from a district marginal.

    This ensures strict-geo targets align with district-based sampling.
    """
    if not district_distribution:
        return None
    region_map = get_district_region_map()
    region_totals: Dict[str, float] = {}
    for district, weight in district_distribution.items():
        if weight is None:
            continue
        norm = _normalize(district)
        if norm not in region_map:
            continue
        region = region_map[norm][1]
        region_totals[region] = region_totals.get(region, 0.0) + float(weight)
    total = sum(region_totals.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in region_totals.items()}


def get_region_distribution_from_district(strict: Optional[bool] = None) -> Optional[Dict[str, float]]:
    """Return region marginal derived from district distribution if available."""
    district_distribution = get_district_distribution(strict=strict)
    if not district_distribution:
        return None
    return derive_region_marginal_from_district(district_distribution)


def _extract_counts(payload: Any) -> Dict[str, int]:
    """Extract district counts from various JSON shapes."""
    if isinstance(payload, dict):
        # If values are numeric, treat as mapping already
        if payload and all(isinstance(v, (int, float)) for v in payload.values()):
            return {k: int(v) for k, v in payload.items()}
        for key in ("districts", "by_district", "by_raion"):
            if key in payload:
                return _extract_counts(payload[key])
    if isinstance(payload, list):
        counts: Dict[str, int] = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            name = row.get("district") or row.get("raion") or row.get("name")
            if not name:
                continue
            count = row.get("count") or row.get("population") or row.get("value")
            if count is None:
                continue
            try:
                counts[str(name)] = counts.get(str(name), 0) + int(count)
            except (TypeError, ValueError):
                continue
        return counts
    return {}


def _normalize_distribution(weights: Dict[str, float], strict: bool) -> Optional[Dict[str, float]]:
    if not weights:
        return None
    region_map = get_district_region_map()
    region_labels = {_normalize(region) for region in load_districts_by_region().keys()}
    normalized_weights: Dict[str, float] = {}
    for name, weight in weights.items():
        if weight is None:
            continue
        try:
            value = float(weight)
        except (TypeError, ValueError):
            continue
        if _is_total_label(str(name)):
            continue
        norm = _normalize(name)
        if norm in region_labels:
            continue
        if norm in region_map:
            canonical, _ = region_map[norm]
            normalized_weights[canonical] = normalized_weights.get(canonical, 0.0) + value
        elif strict:
            raise ValueError(f"Unknown district name in distribution: {name}")
        else:
            normalized_weights[name] = normalized_weights.get(name, 0.0) + value

    total = sum(normalized_weights.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in normalized_weights.items()}


@lru_cache
def get_district_distribution(strict: Optional[bool] = None) -> Optional[Dict[str, float]]:
    """
    Return normalized district distribution if available.

    Looks for:
    - nbs_population_by_district_2024.json
    - config/district_distribution_2024.json
    """
    strict = strict_geo_enabled() if strict is None else strict
    candidates = [
        Path(__file__).parent.parent / "nbs_population_by_district_2024.json",
        Path(__file__).parent.parent / "config" / "district_distribution_2024.json",
    ]
    source = next((p for p in candidates if p.exists()), None)
    if source:
        with open(source, "r", encoding="utf-8") as f:
            payload = json.load(f)
        counts = _extract_counts(payload)
        normalized = _normalize_distribution(counts, strict)
        if normalized:
            return normalized

    # Fallback to PxWeb if no local file
    try:
        manager = NBSDataManager()
        dist = manager.get_district_distribution()
    except Exception as exc:
        logger.warning(f"Failed to load district distribution from PxWeb: {exc}")
        return None

    weights = dist.raw_counts if dist.raw_counts else dist.values
    return _normalize_distribution(weights, strict)


def district_distribution_available() -> bool:
    """Return True if a district distribution source is available."""
    return get_district_distribution(strict=False) is not None


def get_region_for_district(district: str, strict: Optional[bool] = None) -> Optional[str]:
    """Lookup region for a district name."""
    strict = strict_geo_enabled() if strict is None else strict
    region_map = get_district_region_map()
    norm = _normalize(district)
    if norm in region_map:
        return region_map[norm][1]
    if strict:
        raise ValueError(f"Cannot map district to region: {district}")
    return None
