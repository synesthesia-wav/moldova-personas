"""
PxWeb Data Fetcher for NBS Moldova statistical data.

Fetches distributions directly from statbank.statistica.md with:
- Full JSON-stat parsing
- Caching with freshness checking
- Provenance tracking
- Fallback to verified hardcoded data
"""

import json
import hashlib
import logging
import os
import re
import unicodedata
import urllib.parse
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict

import numpy as np
import requests

from .ethnocultural_tables import (
    get_language_distribution_total,
    get_religion_distribution_total,
    strict_mode_enabled,
)

logger = logging.getLogger(__name__)

NBS_BASE_URL = "https://statbank.statistica.md/PxWeb/api/v1/en"
NBS_BASE_URL_RO = "https://statbank.statistica.md/PxWeb/api/v1/ro"
CACHE_MAX_AGE_DAYS = 30


class DataProvenance(Enum):
    """Classification of data source types."""
    PXWEB_DIRECT = "Fetched from NBS PxWeb API"
    PXWEB_CACHED = "Cached from previous PxWeb fetch"
    IPF_DERIVED = "Derived via IPF from marginal distributions"
    CENSUS_HARDCODED = "From NBS 2024 final report (static)"
    ESTIMATED = "Demographic estimate (documented assumptions)"


@dataclass
class Distribution:
    """
    A statistical distribution with full provenance tracking.
    
    Attributes:
        values: Dictionary mapping categories to probabilities/weights
        provenance: Source classification
        source_table: PxWeb table code (if applicable)
        last_fetched: Timestamp of last successful fetch
        confidence: Reliability score (0.0-1.0)
        methodology: Description of derivation method
        limitations: Known limitations (for estimates)
        raw_counts: Original counts (if available)
        total: Total population/count
    """
    values: Dict[str, float]
    provenance: DataProvenance
    source_table: Optional[str] = None
    last_fetched: Optional[datetime] = None
    confidence: float = 1.0
    methodology: Optional[str] = None
    limitations: Optional[str] = None
    raw_counts: Optional[Dict[str, int]] = None
    total: Optional[int] = None
    
    def validate(self, tolerance: float = 0.01) -> bool:
        """Validate that probabilities sum to ~1.0."""
        total_prob = sum(self.values.values())
        return abs(total_prob - 1.0) <= tolerance
    
    def normalize(self) -> "Distribution":
        """Normalize values to sum to 1.0."""
        total = sum(self.values.values())
        if total > 0:
            self.values = {k: v / total for k, v in self.values.items()}
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "values": self.values,
            "provenance": self.provenance.value,
            "source_table": self.source_table,
            "last_fetched": self.last_fetched.isoformat() if self.last_fetched else None,
            "confidence": self.confidence,
            "methodology": self.methodology,
            "limitations": self.limitations,
            "raw_counts": self.raw_counts,
            "total": self.total,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Distribution":
        """Create Distribution from dictionary."""
        return cls(
            values=data["values"],
            provenance=DataProvenance(data.get("provenance", "CENSUS_HARDCODED")),
            source_table=data.get("source_table"),
            last_fetched=datetime.fromisoformat(data["last_fetched"]) if data.get("last_fetched") else None,
            confidence=data.get("confidence", 1.0),
            methodology=data.get("methodology"),
            limitations=data.get("limitations"),
            raw_counts=data.get("raw_counts"),
            total=data.get("total"),
        )


class PxWebParser:
    """Parser for PxWeb 2.0 JSON response format."""
    
    @staticmethod
    def extract_distribution(
        data: Dict[str, Any],
        dimension_code: str,
        value_column: Optional[str] = None,
        code_mapping: Optional[Dict[str, str]] = None
    ) -> Tuple[Dict[str, float], Dict[str, int], int]:
        """
        Extract a marginal distribution from PxWeb response.
        
        Args:
            data: PxWeb JSON response
            dimension_code: Code of dimension to extract (e.g., 'Sexe')
            value_column: Name of value column (if None, uses last column)
            code_mapping: Optional mapping from codes to labels (e.g., {'1': 'Masculin'})
            
        Returns:
            Tuple of (probabilities, counts, total)
        """
        columns = data.get('columns', [])
        rows = data.get('data', [])
        
        # Find column indices
        dim_idx = None
        val_idx = None
        
        for i, col in enumerate(columns):
            col_code = col.get('code', '')
            col_type = col.get('type', '')
            
            if col_code == dimension_code:
                dim_idx = i
            elif col_type == 'c' or (value_column and col_code == value_column):
                val_idx = i
        
        # If no value column specified, use last column
        if val_idx is None:
            val_idx = len(columns) - 1
        
        if dim_idx is None:
            raise ValueError(f"Dimension '{dimension_code}' not found in columns: {[c.get('code') for c in columns]}")
        
        # Aggregate counts
        counts = {}
        total = 0
        
        for row in rows:
            key = row.get('key', [])
            values = row.get('values', [])
            
            if dim_idx < len(key):
                dim_value = key[dim_idx]
                # Map code to label if mapping provided
                if code_mapping and dim_value in code_mapping:
                    dim_value = code_mapping[dim_value]
            else:
                continue
            
            # Get value - in PxWeb, values are in 'values' array
            # val_idx is relative to total columns, but values array may be shorter
            val_array_idx = val_idx - len(key)
            if val_array_idx >= 0 and val_array_idx < len(values):
                try:
                    count = int(values[val_array_idx])
                except (ValueError, TypeError):
                    continue
            else:
                continue
            
            counts[dim_value] = counts.get(dim_value, 0) + count
            total += count
        
        # Calculate probabilities
        if total > 0:
            probs = {k: v / total for k, v in counts.items()}
        else:
            probs = {k: 0.0 for k in counts}
        
        return probs, counts, total


class IPFEngine:
    """
    Iterative Proportional Fitting engine for deriving cross-tabulations.
    
    Implements the classic IPF algorithm to find the maximum entropy
    distribution consistent with given marginal distributions.
    """
    
    def fit(
        self,
        row_marginal: Distribution,
        col_marginal: Distribution,
        seed: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> "CrossTabulation":
        """
        Fit a 2-way table using IPF.
        
        Args:
            row_marginal: Target row marginal distribution
            col_marginal: Target column marginal distribution
            seed: Initial guess (default: uniform)
            max_iterations: Maximum IPF iterations
            tolerance: Convergence threshold
            
        Returns:
            Fitted cross-tabulation
        """
        rows = list(row_marginal.values.keys())
        cols = list(col_marginal.values.keys())
        
        n_rows = len(rows)
        n_cols = len(cols)
        
        # Target marginals as arrays
        row_targets = np.array([row_marginal.values[r] for r in rows])
        col_targets = np.array([col_marginal.values[c] for c in cols])
        
        # Initialize seed matrix
        if seed is not None:
            matrix = seed.copy()
        else:
            # Uniform initialization
            matrix = np.ones((n_rows, n_cols)) / (n_rows * n_cols)
        
        # IPF iterations
        for iteration in range(max_iterations):
            matrix_before = matrix.copy()
            
            # Row adjustment
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid div by zero
            row_factors = row_targets / row_sums.flatten()
            matrix = matrix * row_factors[:, np.newaxis]
            
            # Column adjustment
            col_sums = matrix.sum(axis=0, keepdims=True)
            col_sums = np.where(col_sums == 0, 1, col_sums)
            col_factors = col_targets / col_sums.flatten()
            matrix = matrix * col_factors
            
            # Check convergence
            change = np.abs(matrix - matrix_before).max()
            if change < tolerance:
                logger.debug(f"IPF converged in {iteration + 1} iterations")
                break
        else:
            logger.warning(f"IPF did not converge after {max_iterations} iterations")
        
        # Normalize to ensure probabilities sum to 1
        matrix = matrix / matrix.sum()
        
        return CrossTabulation(
            matrix=matrix,
            row_names=rows,
            col_names=cols,
            provenance=DataProvenance.IPF_DERIVED,
            last_computed=datetime.now(),
            confidence=0.85,  # IPF-derived has lower confidence than direct fetch
            methodology=f"IPF from marginals: {row_marginal.provenance.value} x {col_marginal.provenance.value}"
        )


@dataclass
class CrossTabulation:
    """
    A 2-way cross-tabulation with full provenance.
    
    For example: Ethnicity by Region
    """
    matrix: np.ndarray                    # 2D array of probabilities
    row_names: List[str]                  # Row category names
    col_names: List[str]                  # Column category names
    provenance: DataProvenance
    source_table: Optional[str] = None
    last_computed: Optional[datetime] = None
    confidence: float = 1.0
    methodology: Optional[str] = None
    
    def get_row_marginal(self) -> Dict[str, float]:
        """Get row marginal distribution."""
        totals = self.matrix.sum(axis=1)
        return {name: float(totals[i]) for i, name in enumerate(self.row_names)}
    
    def get_col_marginal(self) -> Dict[str, float]:
        """Get column marginal distribution."""
        totals = self.matrix.sum(axis=0)
        return {name: float(totals[i]) for i, name in enumerate(self.col_names)}


class NBSDataManager:
    """
    Manages all NBS census distributions with automatic PxWeb fetching.
    
    This is the primary interface for accessing census data.
    It implements a hierarchy of data sources:
    1. Fetch from PxWeb API (freshest, preferred)
    2. Use cached data (if < 30 days old)
    3. Use verified hardcoded data (fallback)
    """
    
    # Dataset paths in PxWeb (verified correct paths)
    DATASETS = {
        "population_by_region_sex_age": "20 Populatia si procesele demografice/POP010/POPro/POP010200rcl.px",
        "population_by_district": "20 Populatia si procesele demografice/POP010/POPro/POP010400rclreg.px",
        "population_by_ethnicity": None,  # Not yet available in PxWeb
        "population_by_religion": None,   # Not yet available in PxWeb
        "mother_tongue": None,            # Not yet available in PxWeb
        "marital_status": None,           # Not yet available in PxWeb
        "education_by_age_sex": None,     # Not yet available in PxWeb
        "workforce_statistics": None,     # Not yet available in PxWeb
        "employment_by_status": None,     # Not yet available in PxWeb
    }
    
    # Code mappings for PxWeb responses
    CODE_MAPPINGS = {
        "Sexe": {"1": "Masculin", "2": "Feminin", "0": "Total"},
        "Medii": {"1": "Urban", "2": "Rural", "0": "Total"},
    }

    DISTRICT_DIMENSION_PATTERNS = (
        "raion", "raioane", "district", "municip", "mun", "u.a.t", "uat",
        "unitati administrativ", "unitati administrativ teritoriale", "unitati teritoriale",
    )
    YEAR_DIMENSION_PATTERNS = ("ani", "an", "year", "years")
    TOTAL_PATTERNS = (
        "total",
        "total general",
        "total pe tara",
        "total pe țara",
        "total population",
        "populatia totala",
        "populatia total",
        "whole country",
    )
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the data manager.
        
        Args:
            cache_dir: Directory for caching fetched data (default: ~/.moldova_personas/cache)
        """
        env_cache_dir = os.getenv("MOLDOVA_PERSONAS_CACHE_DIR")
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        elif env_cache_dir:
            self.cache_dir = Path(env_cache_dir)
        else:
            self.cache_dir = Path.home() / ".moldova_personas" / "cache"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.parser = PxWebParser()
        self.ipf_engine = IPFEngine()
        
        logger.info(f"NBSDataManager initialized with cache: {self.cache_dir}")

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching across diacritics/case/punctuation."""
        text = (text or "").strip().lower()
        if not text:
            return ""
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _is_total_label(self, text: str) -> bool:
        norm = self._normalize_text(text)
        return any(pattern in norm for pattern in self.TOTAL_PATTERNS)

    def _find_variable(self, variables: List[Dict[str, Any]], patterns: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
        """Find variable matching any pattern; prefer variable with most values."""
        candidates: List[Dict[str, Any]] = []
        for var in variables:
            text = self._normalize_text(var.get("text", ""))
            code = self._normalize_text(var.get("code", ""))
            if any(pat in text or pat in code for pat in patterns):
                candidates.append(var)
        if not candidates:
            return None
        return max(candidates, key=lambda v: len(v.get("values", [])))

    def _select_year_value(self, var: Dict[str, Any]) -> Optional[str]:
        values = [str(v) for v in var.get("values", []) if v is not None]
        if "2024" in values:
            return "2024"
        numeric = [v for v in values if v.isdigit()]
        if numeric:
            return max(numeric, key=int)
        return values[-1] if values else None

    def _select_total_value(self, var: Dict[str, Any]) -> Optional[str]:
        values = [str(v) for v in var.get("values", []) if v is not None]
        texts = [str(v) for v in var.get("valueTexts", []) if v is not None]
        for value, text in zip(values, texts):
            if self._is_total_label(text) or value in {"0", "Total", "TOTAL"}:
                return value
        for value in values:
            if value in {"0", "Total", "TOTAL"}:
                return value
        return None

    def _build_query(
        self,
        variables: List[Dict[str, Any]],
        target_code: str,
        year_code: Optional[str],
        year_value: Optional[str],
    ) -> Dict[str, Any]:
        query: List[Dict[str, Any]] = []
        for var in variables:
            code = var.get("code")
            if not code:
                continue
            if code == target_code:
                selection = {"filter": "all", "values": ["*"]}
            elif year_code and code == year_code and year_value:
                selection = {"filter": "item", "values": [year_value]}
            else:
                total_value = self._select_total_value(var)
                if total_value:
                    selection = {"filter": "item", "values": [total_value]}
                else:
                    selection = {"filter": "all", "values": ["*"]}
            query.append({"code": code, "selection": selection})
        return {"query": query, "response": {"format": "json"}}

    def _district_cache_valid(self, dist: Distribution) -> bool:
        if not dist or not dist.values:
            return False
        for key in dist.values.keys():
            text = str(key)
            if text.startswith(".."):
                return False
            if self._is_total_label(text):
                return False
            norm = self._normalize_text(text)
            if norm in {"nord", "centru", "sud"}:
                return False
        return True

    def _district_dataset_paths(self) -> List[str]:
        raw = os.getenv("MOLDOVA_PERSONAS_DISTRICT_PXWEB_PATH", "").strip()
        paths: List[str] = []
        if raw:
            for part in re.split(r"[;,]", raw):
                part = part.strip()
                if part:
                    paths.append(self._normalize_dataset_path(part))
        for key in ("population_by_district", "population_by_region_sex_age"):
            value = self.DATASETS.get(key)
            if value:
                paths.append(self._normalize_dataset_path(value))
        seen = set()
        unique = []
        for path in paths:
            if path not in seen:
                seen.add(path)
                unique.append(path)
        return unique

    def _normalize_dataset_path(self, raw_path: str) -> str:
        """
        Normalize PxWeb UI/API paths to API dataset path.

        Accepts:
        - Full PxWeb UI URL
        - PxWeb UI path (pxweb/ro/...)
        - API dataset path (already normalized)
        """
        path = (raw_path or "").strip()
        if not path:
            return path

        if "://" in path:
            parsed = urllib.parse.urlparse(path)
            path = parsed.path

        path = urllib.parse.unquote(path)

        for marker in ("/PxWeb/pxweb/", "/pxweb/", "/PxWeb/api/v1/", "/PxWeb/api/v1/"):
            if marker in path:
                path = path.split(marker, 1)[1]
                break

        parts = [p for p in path.split("/") if p]
        if parts and parts[0] in {"ro", "en"}:
            parts = parts[1:]

        if len(parts) >= 2 and "__" in parts[1]:
            prefix = parts[1].split("__", 1)[0]
            if self._normalize_text(prefix) == self._normalize_text(parts[0]):
                parts = parts[1:]

        normalized_parts: List[str] = []
        for part in parts:
            if "__" in part:
                normalized_parts.extend([p for p in part.split("__") if p])
            else:
                normalized_parts.append(part)

        return "/".join(normalized_parts)
    
    def _make_request(
        self,
        dataset_path: str,
        query: Optional[Dict] = None,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Make API request to PxWeb with retry logic.
        
        Args:
            dataset_path: Path to dataset
            query: Optional query parameters
            max_retries: Maximum number of retry attempts
            backoff_factor: Factor for exponential backoff (1.0 = 1s, 2s, 4s...)
            
        Returns:
            JSON response as dictionary
            
        Raises:
            requests.RequestException: If all retries fail
        """
        import time
        
        base = base_url or NBS_BASE_URL
        url = f"{base}/{dataset_path}"
        
        for attempt in range(max_retries):
            try:
                if query:
                    response = requests.post(url, json=query, timeout=60)
                else:
                    response = requests.get(url, timeout=30)
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout as e:
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    sleep_time = backoff_factor * (2 ** attempt)
                    logger.info(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    raise
                    
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else 0
                
                # Don't retry on client errors (4xx) except 429 (rate limit)
                if 400 <= status_code < 500 and status_code != 429:
                    logger.error(f"HTTP {status_code} error (no retry): {e}")
                    raise
                
                logger.warning(f"HTTP {status_code} on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    sleep_time = backoff_factor * (2 ** attempt)
                    logger.info(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    raise
                    
            except requests.RequestException as e:
                logger.warning(f"Request error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    sleep_time = backoff_factor * (2 ** attempt)
                    logger.info(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    raise
        
        # Should not reach here
        raise requests.RequestException(f"All {max_retries} retries failed")
    
    def _load_from_cache(self, name: str) -> Optional[Distribution]:
        """Load distribution from cache if fresh."""
        cache_path = self.cache_dir / f"{name}.json"
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            dist = Distribution.from_dict(data)
            
            # Check freshness
            if dist.last_fetched:
                age = datetime.now() - dist.last_fetched
                if age > timedelta(days=CACHE_MAX_AGE_DAYS):
                    logger.debug(f"Cache for {name} is stale ({age.days} days)")
                    return None
            
            logger.debug(f"Loaded {name} from cache (fresh)")
            return dist
            
        except Exception as e:
            logger.warning(f"Failed to load cache for {name}: {e}")
            return None
    
    def _save_to_cache(self, name: str, distribution: Distribution):
        """Save distribution to cache."""
        cache_path = self.cache_dir / f"{name}.json"
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(distribution.to_dict(), f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved {name} to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache for {name}: {e}")
    
    def _get_with_fallback(
        self,
        name: str,
        fetch_func,
        fallback_dist: Distribution
    ) -> Distribution:
        """
        Get distribution with fallback chain.
        
        Tries: Cache → Fetch → Fallback
        """
        # Try cache
        cached = self._load_from_cache(name)
        if cached:
            return cached
        
        # Try fetch
        try:
            dist = fetch_func()
            self._save_to_cache(name, dist)
            return dist
        except Exception as e:
            logger.warning(f"Failed to fetch {name}: {e}")
        
        # Fallback
        logger.info(f"Using fallback for {name}")
        return fallback_dist
    
    # =========================================================================
    # Distribution Fetchers
    # =========================================================================
    
    def get_sex_distribution(self) -> Distribution:
        """
        Get sex distribution from PxWeb.
        
        Source: POP010200rcl.px (Population by age, sex, areas)
        """
        def fetch():
            dataset_path = self.DATASETS["population_by_region_sex_age"]
            
            # Query for 2024 data with sex breakdown
            query = {
                "query": [
                    {"code": "Ani", "selection": {"filter": "item", "values": ["2024"]}},
                    {"code": "Sexe", "selection": {"filter": "all", "values": ["*"]}},
                    {"code": "Varste", "selection": {"filter": "all", "values": ["*"]}},
                    {"code": "Medii", "selection": {"filter": "all", "values": ["*"]}},
                ],
                "response": {"format": "json"}
            }
            
            data = self._make_request(dataset_path, query)
            probs, counts, total = self.parser.extract_distribution(
                data, "Sexe", code_mapping=self.CODE_MAPPINGS.get("Sexe")
            )
            
            # Filter out 'Total' if present
            if "Total" in probs:
                del probs["Total"]
                del counts["Total"]
            
            # Re-normalize
            total_filtered = sum(counts.values())
            if total_filtered > 0:
                probs = {k: v / total_filtered for k, v in counts.items()}
            
            return Distribution(
                values=probs,
                provenance=DataProvenance.PXWEB_DIRECT,
                source_table="POP010200rcl.px",
                last_fetched=datetime.now(),
                confidence=0.99,
                raw_counts=counts,
                total=total_filtered,
                methodology="Fetched from NBS PxWeb API, 2024 data"
            )
        
        fallback = Distribution(
            values={"Feminin": 0.528, "Masculin": 0.472},
            provenance=DataProvenance.CENSUS_HARDCODED,
            source_table="POP010200rcl.px",
            confidence=0.95,
            methodology="NBS 2024 Census: 52.8% female, 47.2% male",
            limitations="Hardcoded fallback - may not reflect latest NBS data"
        )
        
        return self._get_with_fallback("sex", fetch, fallback)
    
    def get_residence_type_distribution(self) -> Distribution:
        """
        Get urban/rural distribution from PxWeb.
        
        Source: POP010200rcl.px
        """
        def fetch():
            dataset_path = self.DATASETS["population_by_region_sex_age"]
            
            query = {
                "query": [
                    {"code": "Ani", "selection": {"filter": "item", "values": ["2024"]}},
                    {"code": "Medii", "selection": {"filter": "all", "values": ["*"]}},
                    {"code": "Varste", "selection": {"filter": "all", "values": ["*"]}},
                    {"code": "Sexe", "selection": {"filter": "all", "values": ["*"]}},
                ],
                "response": {"format": "json"}
            }
            
            data = self._make_request(dataset_path, query)
            probs, counts, total = self.parser.extract_distribution(
                data, "Medii", code_mapping=self.CODE_MAPPINGS.get("Medii")
            )
            
            # Filter out 'Total' if present
            if "Total" in probs:
                del probs["Total"]
                del counts["Total"]
            
            # Re-normalize
            total_filtered = sum(counts.values())
            if total_filtered > 0:
                probs = {k: v / total_filtered for k, v in counts.items()}
            
            return Distribution(
                values=probs,
                provenance=DataProvenance.PXWEB_DIRECT,
                source_table="POP010200rcl.px",
                last_fetched=datetime.now(),
                confidence=0.99,
                raw_counts=counts,
                total=total_filtered,
                methodology="Fetched from NBS PxWeb API, 2024 data"
            )
        
        fallback = Distribution(
            values={"Urban": 0.464, "Rural": 0.536},
            provenance=DataProvenance.CENSUS_HARDCODED,
            source_table="POP010200rcl.px",
            confidence=0.95,
            methodology="NBS 2024 Census: 46.4% urban, 53.6% rural",
            limitations="Hardcoded fallback - may not reflect latest NBS data"
        )
        
        return self._get_with_fallback("residence_type", fetch, fallback)
    
    def get_age_group_distribution(self) -> Distribution:
        """
        Get age group distribution from PxWeb.
        
        Source: POP010200rcl.px
        """
        def fetch():
            dataset_path = self.DATASETS["population_by_region_sex_age"]
            
            query = {
                "query": [
                    {"code": "Ani", "selection": {"filter": "item", "values": ["2024"]}},
                    {"code": "Varste", "selection": {"filter": "all", "values": ["*"]}},
                    {"code": "Sexe", "selection": {"filter": "all", "values": ["*"]}},
                    {"code": "Medii", "selection": {"filter": "all", "values": ["*"]}},
                ],
                "response": {"format": "json"}
            }
            
            data = self._make_request(dataset_path, query)
            
            # Extract raw age distribution then group
            columns = data.get('columns', [])
            rows = data.get('data', [])
            
            # Find indices
            age_idx = None
            val_idx = None
            for i, col in enumerate(columns):
                if col.get('code') == 'Varste':
                    age_idx = i
                if col.get('type') == 'c':
                    val_idx = i
            
            # Aggregate by age groups
            age_groups = {"0-14": 0, "15-24": 0, "25-34": 0, "35-44": 0, 
                         "45-54": 0, "55-64": 0, "65+": 0}
            
            for row in rows:
                key = row.get('key', [])
                values = row.get('values', [])
                
                if age_idx is None or val_idx is None:
                    continue
                    
                age_str = key[age_idx] if age_idx < len(key) else None
                val_array_idx = val_idx - len(key)
                count = int(values[val_array_idx]) if val_array_idx >= 0 and val_array_idx < len(values) else 0
                
                if age_str is None or count == 0:
                    continue
                
                try:
                    age = int(age_str)
                    if age <= 14:
                        age_groups["0-14"] += count
                    elif age <= 24:
                        age_groups["15-24"] += count
                    elif age <= 34:
                        age_groups["25-34"] += count
                    elif age <= 44:
                        age_groups["35-44"] += count
                    elif age <= 54:
                        age_groups["45-54"] += count
                    elif age <= 64:
                        age_groups["55-64"] += count
                    else:
                        age_groups["65+"] += count
                except ValueError:
                    continue
            
            total = sum(age_groups.values())
            probs = {k: v / total for k, v in age_groups.items()} if total > 0 else age_groups
            
            return Distribution(
                values=probs,
                provenance=DataProvenance.PXWEB_DIRECT,
                source_table="POP010200rcl.px",
                last_fetched=datetime.now(),
                confidence=0.99,
                raw_counts=age_groups,
                total=total,
                methodology="Aggregated from single-year ages via NBS PxWeb API, 2024 data"
            )
        
        fallback = Distribution(
            values={
                "0-14": 0.166, "15-24": 0.098, "25-34": 0.124,
                "35-44": 0.154, "45-54": 0.131, "55-64": 0.144, "65+": 0.183,
            },
            provenance=DataProvenance.CENSUS_HARDCODED,
            source_table="POP010200rcl.px",
            confidence=0.95,
            methodology="NBS 2024 Census final report",
            limitations="Hardcoded fallback - may not reflect latest NBS data"
        )
        
        return self._get_with_fallback("age_group", fetch, fallback)

    def get_district_distribution(self) -> Distribution:
        """
        Get district (raion/municipiu) distribution from PxWeb.

        Requires a PxWeb dataset that includes a district-level dimension.
        Configure via MOLDOVA_PERSONAS_DISTRICT_PXWEB_PATH or rely on
        known datasets for discovery.
        """
        def fetch() -> Distribution:
            dataset_paths = self._district_dataset_paths()
            if not dataset_paths:
                raise ValueError("No PxWeb dataset path configured for district distribution")

            last_error: Optional[Exception] = None
            for dataset_path in dataset_paths:
                for base_url in (NBS_BASE_URL_RO, NBS_BASE_URL):
                    try:
                        meta = self._make_request(dataset_path, base_url=base_url)
                        variables = meta.get("variables", [])
                        if not variables:
                            continue

                        district_var = self._find_variable(variables, self.DISTRICT_DIMENSION_PATTERNS)
                        if not district_var:
                            continue

                        year_var = self._find_variable(variables, self.YEAR_DIMENSION_PATTERNS)
                        year_code = year_var.get("code") if year_var else None
                        year_value = self._select_year_value(year_var) if year_var else None

                        query = self._build_query(variables, district_var["code"], year_code, year_value)
                        data = self._make_request(dataset_path, query, base_url=base_url)

                        code_mapping = None
                        values = district_var.get("values", [])
                        value_texts = district_var.get("valueTexts", [])
                        label_map = {}
                        if values and value_texts and len(values) == len(value_texts):
                            code_mapping = {str(code): str(label) for code, label in zip(values, value_texts)}
                            for label in value_texts:
                                raw_label = str(label)
                                if raw_label.startswith(".."):
                                    canonical = raw_label.lstrip(".").strip()
                                elif raw_label.startswith("Mun.") or raw_label.startswith("U.T.A."):
                                    canonical = raw_label.strip()
                                else:
                                    continue
                                label_map[self._normalize_text(canonical)] = canonical

                        probs, counts, total = self.parser.extract_distribution(
                            data,
                            district_var["code"],
                            code_mapping=code_mapping,
                        )

                        if label_map:
                            filtered_counts = {}
                            for key, count in counts.items():
                                norm = self._normalize_text(str(key))
                                if norm in label_map:
                                    canonical = label_map[norm]
                                    filtered_counts[canonical] = filtered_counts.get(canonical, 0) + count
                        else:
                            filtered_counts = {
                                k: v for k, v in counts.items()
                                if not self._is_total_label(str(k))
                            }
                        if filtered_counts:
                            counts = filtered_counts
                            total = sum(counts.values())
                            probs = {k: v / total for k, v in counts.items()} if total > 0 else {}

                        if not counts:
                            continue

                        return Distribution(
                            values=probs,
                            provenance=DataProvenance.PXWEB_DIRECT,
                            source_table=dataset_path,
                            last_fetched=datetime.now(),
                            confidence=0.99,
                            raw_counts=counts,
                            total=total,
                            methodology="Fetched from NBS PxWeb API (district distribution)",
                        )
                    except Exception as exc:
                        last_error = exc
                        continue

            raise RuntimeError("Failed to fetch district distribution from PxWeb") from last_error

        fallback = Distribution(
            values={},
            provenance=DataProvenance.ESTIMATED,
            source_table=None,
            confidence=0.0,
            methodology="No district distribution available",
            limitations="Provide MOLDOVA_PERSONAS_DISTRICT_PXWEB_PATH or local district file",
        )

        cached = self._load_from_cache("district")
        if cached and self._district_cache_valid(cached):
            return cached

        try:
            dist = fetch()
            self._save_to_cache("district", dist)
            return dist
        except Exception as e:
            logger.warning(f"Failed to fetch district: {e}")

        return fallback
    
    # =========================================================================
    # Fallback distributions (until PxWeb has 2024 census data)
    # =========================================================================
    
    def get_region_distribution(self) -> Distribution:
        """Region distribution (hardcoded until PxWeb has 2024 data)."""
        return Distribution(
            values={
                "Chisinau": 0.299, "Centru": 0.278, "Nord": 0.253,
                "Sud": 0.127, "Gagauzia": 0.043,
            },
            provenance=DataProvenance.CENSUS_HARDCODED,
            source_table="POP010200rcl.px",
            confidence=0.95,
            methodology="NBS 2024 Census final report",
            limitations="Regional distribution not yet available in PxWeb for 2024"
        )
    
    def get_ethnicity_distribution(self) -> Distribution:
        """Ethnicity distribution (hardcoded until PxWeb has 2024 data)."""
        return Distribution(
            values={
                "Moldovean": 0.772, "Român": 0.079, "Ucrainean": 0.050,
                "Găgăuz": 0.043, "Rus": 0.030, "Bulgar": 0.016,
                "Rrom": 0.004, "Altele": 0.006,
            },
            provenance=DataProvenance.CENSUS_HARDCODED,
            source_table="POP010500.px",
            confidence=0.95,
            methodology="NBS 2024 Census final report",
            limitations="Ethnicity data not yet available in PxWeb for 2024"
        )
    
    def get_education_distribution(self) -> Distribution:
        """Education distribution (hardcoded until PxWeb has 2024 data)."""
        return Distribution(
            values={
                "Fără studii": 0.024, "Primar": 0.092, "Gimnazial": 0.227,
                "Liceal": 0.128, "Profesional/Tehnic": 0.336,
                "Superior (Licență/Master)": 0.181, "Doctorat": 0.012,
            },
            provenance=DataProvenance.CENSUS_HARDCODED,
            source_table="EDU010100.px",
            confidence=0.90,
            methodology="NBS 2024 Census education data (population 10+)",
            limitations="Education data not yet available in PxWeb for 2024"
        )
    
    def get_marital_status_distribution(self) -> Distribution:
        """Marital status distribution (hardcoded until PxWeb has 2024 data)."""
        return Distribution(
            values={
                "Necăsătorit": 0.236, "Căsătorit": 0.558, "Divorțat": 0.085,
                "Văduv": 0.100, "Separat": 0.021,
            },
            provenance=DataProvenance.CENSUS_HARDCODED,
            source_table="POP030100.px",
            confidence=0.90,
            methodology="NBS 2024 Census marital status (population 15+)",
            limitations="Marital status data not yet available in PxWeb for 2024"
        )
    
    def get_religion_distribution(self) -> Distribution:
        """Religion distribution (hardcoded until PxWeb has 2024 data)."""
        official = get_religion_distribution_total()
        if official:
            return Distribution(
                values=official,
                provenance=DataProvenance.CENSUS_HARDCODED,
                source_table="Date_Comunicat_Etnoculturale_20_10_25.xlsx (Figura 17)",
                confidence=0.98,
                methodology="NBS 2024 ethnocultural communiqué: religion by ethnicity (total)",
                limitations="Mapped to internal categories; see config/ethnocultural_tables_2024.json",
            )
        if strict_mode_enabled():
            raise ValueError(
                "Strict mode: missing official religion distribution totals. "
                "Provide config/ethnocultural_tables_2024.json or disable strict mode."
            )

        return Distribution(
            values={
                "Ortodox": 0.9646686786009324,
                "Baptist": 0.011139687845779643,
                "Martor al lui Iehova": 0.0070106210590479695,
                "Penticostal": 0.005354491915804803,
                "Adventist": 0.0029656562395802867,
                "Creștină după Evanghelie": 0.0027031561599382744,
                "Staroveri (Ortodoxă Rusă de rit vechi)": 0.0017215417844484338,
                "Islam": 0.0013328887539104852,
                "Catolic": 0.0010984226633564402,
                "Altă religie": 0.002004854977201235,
            },
            provenance=DataProvenance.CENSUS_HARDCODED,
            source_table="POP010600.px",
            confidence=0.95,
            methodology="NBS 2024 Census ethnocultural communiqué totals (fallback)",
            limitations="Religion data not yet available in PxWeb for 2024"
        )
    
    def get_language_distribution(self) -> Distribution:
        """Mother tongue distribution (hardcoded until PxWeb has 2024 data)."""
        official = get_language_distribution_total()
        if official:
            return Distribution(
                values=official,
                provenance=DataProvenance.CENSUS_HARDCODED,
                source_table="Date_Comunicat_Etnoculturale_20_10_25.xlsx (Figura 8)",
                confidence=0.98,
                methodology="NBS 2024 ethnocultural communiqué: mother tongue (total)",
                limitations="Moldovenească mapped to Română; undeclared excluded; see config/ethnocultural_tables_2024.json",
            )
        if strict_mode_enabled():
            raise ValueError(
                "Strict mode: missing official language distribution totals. "
                "Provide config/ethnocultural_tables_2024.json or disable strict mode."
            )

        return Distribution(
            values={
                "Română": 0.800, "Rusă": 0.110, "Găgăuză": 0.038,
                "Ucraineană": 0.029, "Bulgară": 0.015, "Alta": 0.008,
            },
            provenance=DataProvenance.CENSUS_HARDCODED,
            source_table="POP010550.px",
            confidence=0.95,
            methodology="NBS 2024 Census: 49% Moldovan + 31% Romanian = ~80% Romanian",
            limitations="Mother tongue data not yet available in PxWeb for 2024"
        )
    
    def get_employment_status_distribution(self) -> Distribution:
        """Employment status distribution (hardcoded until PxWeb has 2024 data)."""
        return Distribution(
            values={
                "employed": 0.427, "unemployed": 0.018, "student": 0.080,
                "retired": 0.183, "homemaker": 0.060, "other_inactive": 0.232,
            },
            provenance=DataProvenance.CENSUS_HARDCODED,
            source_table="MUN110200.px",
            confidence=0.90,
            methodology="NBS 2024 Labor Force Survey",
            limitations="Employment status not yet available in PxWeb for 2024"
        )
    
    def get_distribution(self, name: str) -> Distribution:
        """Get a distribution by name."""
        method_map = {
            "region": self.get_region_distribution,
            "residence_type": self.get_residence_type_distribution,
            "sex": self.get_sex_distribution,
            "age_group": self.get_age_group_distribution,
            "ethnicity": self.get_ethnicity_distribution,
            "education": self.get_education_distribution,
            "marital_status": self.get_marital_status_distribution,
            "religion": self.get_religion_distribution,
            "language": self.get_language_distribution,
            "employment_status": self.get_employment_status_distribution,
        }
        
        if name in method_map:
            return method_map[name]()
        
        raise ValueError(f"Unknown distribution: {name}. Available: {list(method_map.keys())}")
    
    def get_all_distributions(self) -> Dict[str, Distribution]:
        """Get all available distributions."""
        return {
            "region": self.get_region_distribution(),
            "residence_type": self.get_residence_type_distribution(),
            "sex": self.get_sex_distribution(),
            "age_group": self.get_age_group_distribution(),
            "ethnicity": self.get_ethnicity_distribution(),
            "education": self.get_education_distribution(),
            "marital_status": self.get_marital_status_distribution(),
            "religion": self.get_religion_distribution(),
            "language": self.get_language_distribution(),
            "employment_status": self.get_employment_status_distribution(),
        }
    
    def get_cross_tabulation_ipf(
        self,
        row_dist: Distribution,
        col_dist: Distribution,
        name: str
    ) -> CrossTabulation:
        """
        Derive cross-tabulation using IPF from marginal distributions.
        
        Args:
            row_dist: Row marginal distribution
            col_dist: Column marginal distribution
            name: Name for caching (e.g., "ethnicity_by_region")
            
        Returns:
            CrossTabulation derived via IPF
        """
        def _signature(dist: Distribution) -> str:
            items = sorted((str(k), round(float(v), 12)) for k, v in dist.values.items())
            payload = json.dumps(items, ensure_ascii=False)
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()

        def _cache_signature() -> str:
            return hashlib.sha256(
                json.dumps(
                    {"row": _signature(row_dist), "col": _signature(col_dist)},
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()

        # Check cache first
        cache_path = self.cache_dir / f"crosstab_{name}.json"
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                cached_signature = data.get("signature")
                current_signature = _cache_signature()
                if cached_signature != current_signature:
                    raise ValueError("Cached cross-tab signature mismatch")

                # Deserialize
                matrix = np.array(data['matrix'])
                return CrossTabulation(
                    matrix=matrix,
                    row_names=data['row_names'],
                    col_names=data['col_names'],
                    provenance=DataProvenance.IPF_DERIVED,
                    last_computed=datetime.fromisoformat(data['last_computed']),
                    confidence=data.get('confidence', 0.85),
                    methodology=data.get('methodology', 'IPF from marginals')
                )
            except Exception as e:
                logger.warning(f"Failed to load cached cross-tab {name}: {e}")
        
        # Derive using IPF
        cross_tab = self.ipf_engine.fit(row_dist, col_dist)
        
        # Cache result
        try:
            cache_data = {
                'matrix': cross_tab.matrix.tolist(),
                'row_names': cross_tab.row_names,
                'col_names': cross_tab.col_names,
                'last_computed': cross_tab.last_computed.isoformat(),
                'confidence': cross_tab.confidence,
                'methodology': cross_tab.methodology,
                'signature': _cache_signature(),
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache cross-tab {name}: {e}")
        
        return cross_tab
    
    def get_ethnicity_by_region(self) -> CrossTabulation:
        """
        Get ethnicity by region cross-tabulation via IPF.
        
        Uses IPF to derive cross-tab from ethnicity and region marginals.
        """
        ethnicity_dist = self.get_ethnicity_distribution()
        region_dist = self.get_region_distribution()
        
        return self.get_cross_tabulation_ipf(
            ethnicity_dist,
            region_dist,
            "ethnicity_by_region"
        )
    
    def get_region_urban_cross(self) -> CrossTabulation:
        """
        Get region by urban/rural cross-tabulation via IPF.
        
        Uses IPF to derive cross-tab from region and residence type marginals.
        """
        region_dist = self.get_region_distribution()
        residence_dist = self.get_residence_type_distribution()
        
        return self.get_cross_tabulation_ipf(
            region_dist,
            residence_dist,
            "region_by_urban"
        )
    
    def refresh_cache(self) -> Dict[str, bool]:
        """
        Force refresh all cached distributions from PxWeb.
        
        Returns:
            Dictionary mapping distribution names to success status
        """
        results = {}
        
        # Only these can be fetched from PxWeb currently
        fetchable = {
            "sex": self.get_sex_distribution,
            "residence_type": self.get_residence_type_distribution,
            "age_group": self.get_age_group_distribution,
            "district": self.get_district_distribution,
        }
        
        for name, fetcher in fetchable.items():
            try:
                # Clear cache first
                cache_path = self.cache_dir / f"{name}.json"
                if cache_path.exists():
                    cache_path.unlink()
                
                # Force fetch
                dist = fetcher()
                if dist.provenance == DataProvenance.PXWEB_DIRECT:
                    results[name] = True
                    logger.info(f"✓ Refreshed {name} from PxWeb")
                else:
                    results[name] = False
                    logger.warning(f"✗ {name} used fallback (not from PxWeb)")
            except Exception as e:
                results[name] = False
                logger.error(f"✗ Failed to refresh {name}: {e}")
        
        # Clear cross-tabulation caches
        for cache_file in self.cache_dir.glob("crosstab_*.json"):
            try:
                cache_file.unlink()
                logger.info(f"Cleared cross-tab cache: {cache_file.name}")
            except Exception as e:
                logger.warning(f"Failed to clear cache {cache_file}: {e}")
        
        return results


# Singleton instance for convenience
NBS_MANAGER = NBSDataManager()


def get_distribution(name: str) -> Distribution:
    """Convenience function to get a distribution by name."""
    return NBS_MANAGER.get_distribution(name)
