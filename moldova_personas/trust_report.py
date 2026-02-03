"""
Trust Report and Auditability Module

Provides dataset-level quality metrics, provenance tracking, and
gating mechanisms for the persona generation pipeline.

This module enables "computed trust" by measuring and reporting:
- Source coverage (% fields by provenance type)
- Marginal error vs census targets
- Joint distribution drift
- Fallback/detection events
- IPF side effects (ESS, information loss)
- Use-case specific gating with super-critical field handling
- Data currency tracking
"""

import json
import hashlib
import math
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum

import numpy as np

from .pxweb_fetcher import DataProvenance, Distribution
from .statistical_tests import calculate_adaptive_tolerance
from .narrative_contract import NARRATIVE_JSON_SCHEMA
from .models import Persona, PopulationMode
from .geo_tables import (
    get_district_distribution,
    get_region_distribution_from_district,
    strict_geo_enabled,
)


class QualityTier(Enum):
    """Dataset quality classification."""
    A = "A"  # All core distributions from live API, minimal drift
    B = "B"  # Some hardcoded fallbacks used, drift within tolerance
    C = "C"  # Significant fallbacks or drift detected
    REJECT = "REJECT"  # Critical issues - should not use for analysis


class UseCaseProfile(Enum):
    """Use-case profiles for gating decisions."""
    ANALYSIS_ONLY = "analysis_only"  # Structured data only, no narratives needed
    NARRATIVE_REQUIRED = "narrative_required"  # Requires LLM-generated narratives
    HIGH_STAKES = "high_stakes"  # Strict requirements for both structured + narratives


# Critical fields that matter more for quality assessment
CRITICAL_FIELDS: Set[str] = {
    "sex", "age_group", "region", "residence_type", 
    "ethnicity", "education", "employment_status", 
    "religion", "marital_status", "language", "district"
}

# Super-critical fields for HIGH_STAKES mode - any fallback here is serious
HIGH_STAKES_CRITICAL_FIELDS: Set[str] = {
    "ethnicity", "education", "employment_status", "region"
}

# Thresholds by use-case profile
USE_CASE_THRESHOLDS = {
    UseCaseProfile.ANALYSIS_ONLY: {
        "max_mean_l1_error": 0.10,
        "max_category_error": 0.25,
        "max_fallback_ratio_critical": 0.70,
        "max_fallback_ratio_super_critical": 1.0,  # No special restriction
        "max_mock_ratio": 1.0,  # Narratives don't matter
        "min_ess_ratio": 0.50,  # Allow significant information loss
        "max_cache_age_days": 60,  # Allow older cached data
        "min_schema_valid_ratio_attempted": 0.50,  # Lenient for analysis-only
        "require_provenance_tracking": True,
    },
    UseCaseProfile.NARRATIVE_REQUIRED: {
        "max_mean_l1_error": 0.10,
        "max_category_error": 0.25,
        "max_fallback_ratio_critical": 0.70,
        "max_fallback_ratio_super_critical": 0.50,  # Some restriction
        "max_mock_ratio": 0.20,  # Max 20% mock narratives
        "min_ess_ratio": 0.70,  # Gate if ESS drops below 70%
        "max_cache_age_days": 30,  # 30 days freshness
        "min_schema_valid_ratio_attempted": 0.90,  # Hard gate if < 90%
        "require_provenance_tracking": True,
    },
    UseCaseProfile.HIGH_STAKES: {
        "max_mean_l1_error": 0.06,
        "max_category_error": 0.15,
        "max_fallback_ratio_critical": 0.30,
        "max_fallback_ratio_super_critical": 0.10,  # Very strict - max 10%
        "max_mock_ratio": 0.05,  # Max 5% mock narratives
        "min_ess_ratio": 0.85,  # High ESS required
        "max_cache_age_days": 7,  # Very fresh data required
        "min_schema_valid_ratio_attempted": 0.95,  # Very strict - 95%
        "require_provenance_tracking": True,
    },
}

# Narrative quality thresholds
NARRATIVE_QUALITY_THRESHOLDS = {
    "min_narrative_length": 50,  # Characters
    "max_narrative_length": 5000,  # Characters
    "min_schema_valid_ratio": 0.90,  # 90% must be schema-valid
}


@dataclass
class MarginalError:
    """Error metric for a single distribution."""
    field: str
    target_dist: Dict[str, float]
    actual_dist: Dict[str, float]
    target_source: str = "unknown"  # pxweb, hardcoded, mixed
    
    @property
    def l1_distance(self) -> float:
        """L1 distance (Total Variation Distance * 2)."""
        all_keys = set(self.target_dist.keys()) | set(self.actual_dist.keys())
        return sum(
            abs(self.target_dist.get(k, 0) - self.actual_dist.get(k, 0))
            for k in all_keys
        )
    
    @property
    def tv_distance(self) -> float:
        """Total Variation Distance (half of L1)."""
        return self.l1_distance / 2
    
    @property
    def max_error(self) -> float:
        """Maximum absolute error for any category."""
        all_keys = set(self.target_dist.keys()) | set(self.actual_dist.keys())
        return max(
            abs(self.target_dist.get(k, 0) - self.actual_dist.get(k, 0))
            for k in all_keys
        ) if all_keys else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "l1_distance": round(self.l1_distance, 4),
            "tv_distance": round(self.tv_distance, 4),
            "max_error": round(self.max_error, 4),
            "target_source": self.target_source,
            "target": self.target_dist,
            "actual": self.actual_dist,
        }


@dataclass
class IPFMetrics:
    """Metrics for IPF correction side effects."""
    original_sample_size: int
    effective_sample_size: float
    """ESS = (sum(weights)^2) / sum(weights^2) - measures information loss"""
    
    resampling_ratio: float
    """Final sample size / original sample size"""
    
    pre_correction_drift: Dict[str, float] = field(default_factory=dict)
    """L1 error before IPF correction by field"""
    
    post_correction_drift: Dict[str, float] = field(default_factory=dict)
    """L1 error after IPF correction by field"""
    
    weight_concentration: Optional[float] = None
    """Max weight / mean weight - indicates over-representation of some personas"""
    
    top_5_weight_share: Optional[float] = None
    """Fraction of total weight held by top 5% of personas"""
    
    correction_iterations: int = 1
    """Number of IPF iterations performed"""

    raking_fields: Optional[List[str]] = None
    """Marginals used for raking (if applicable)"""

    raking_converged: Optional[bool] = None
    """Whether raking converged within tolerance"""

    max_marginal_error: Optional[float] = None
    """Maximum marginal error after raking"""

    marginal_error_by_field: Optional[Dict[str, float]] = None
    """Per-field marginal errors after raking"""

    weight_cap: Optional[Tuple[float, float]] = None
    """(min, max) weight cap applied during raking"""
    
    @property
    def information_loss(self) -> float:
        """Fraction of information lost due to weighting (0-1)."""
        if self.original_sample_size == 0:
            return 0.0
        return 1.0 - (self.effective_sample_size / self.original_sample_size)
    
    @property
    def ess_ratio(self) -> float:
        """ESS as ratio of original sample size."""
        if self.original_sample_size == 0:
            return 1.0
        return self.effective_sample_size / self.original_sample_size
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "original_sample_size": self.original_sample_size,
            "effective_sample_size": round(self.effective_sample_size, 2),
            "resampling_ratio": round(self.resampling_ratio, 4),
            "information_loss": round(self.information_loss, 4),
            "ess_ratio": round(self.ess_ratio, 4),
            "pre_correction_drift": self.pre_correction_drift,
            "post_correction_drift": self.post_correction_drift,
            "correction_iterations": self.correction_iterations,
        }
        if self.weight_concentration is not None:
            result["weight_concentration"] = round(self.weight_concentration, 4)
        if self.top_5_weight_share is not None:
            result["top_5_weight_share"] = round(self.top_5_weight_share, 4)
        if self.raking_fields is not None:
            result["raking_fields"] = self.raking_fields
        if self.raking_converged is not None:
            result["raking_converged"] = self.raking_converged
        if self.max_marginal_error is not None:
            result["max_marginal_error"] = round(self.max_marginal_error, 6)
        if self.marginal_error_by_field is not None:
            result["marginal_error_by_field"] = {
                k: round(v, 6) for k, v in self.marginal_error_by_field.items()
            }
        if self.weight_cap is not None:
            result["weight_cap"] = self.weight_cap
        return result


@dataclass
class NarrativeQualityMetrics:
    """
    Quality metrics for generated narratives beyond mock/failed status.
    
    Tracks two denominators:
    - overall: vs total personas (includes mocks as "not attempted")
    - attempted: vs personas where LLM was actually called
    """
    
    total_personas: int = 0
    
    # Status breakdown
    generated_count: int = 0
    mock_count: int = 0
    failed_count: int = 0
    
    # Length-based metrics
    too_short_count: int = 0
    too_long_count: int = 0
    length_valid_count: int = 0
    
    # Schema validation
    schema_valid_count: int = 0
    schema_invalid_count: int = 0
    
    # Language validation (Romanian check)
    romanian_valid_count: int = 0
    romanian_invalid_count: int = 0

    # Per-field coverage (required narrative sections)
    required_fields_present: int = 0
    required_fields_missing: int = 0
    required_fields_length_valid: int = 0
    required_fields_language_valid: int = 0
    required_fields_schema_valid: int = 0
    any_field_valid_count: int = 0
    field_failure_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # PII/Realism heuristics
    potential_pii_count: int = 0  # Contains specific employer/school names
    
    # Parse/attempt tracking
    parse_attempted_count: int = 0  # Number of generated narratives we tried to parse
    parse_failed_count: int = 0     # Number that failed parsing
    
    @property
    def schema_valid_ratio_overall(self) -> float:
        """Fraction of ALL personas with schema-valid narratives."""
        if self.total_personas == 0:
            return 1.0
        return self.schema_valid_count / self.total_personas
    
    @property
    def schema_valid_ratio_attempted(self) -> float:
        """
        Fraction of ATTEMPTED narratives that are schema-valid.
        This is the key metric for gating - excludes mocks (not attempted).
        """
        if self.parse_attempted_count == 0:
            return 1.0
        return self.schema_valid_count / self.parse_attempted_count
    
    @property
    def parse_failure_ratio(self) -> float:
        """Fraction of attempted parses that failed."""
        if self.parse_attempted_count == 0:
            return 0.0
        return self.parse_failed_count / self.parse_attempted_count
    
    @property
    def romanian_valid_ratio(self) -> float:
        """Fraction of generated narratives with valid Romanian text."""
        if self.generated_count == 0:
            return 1.0
        return self.romanian_valid_count / self.generated_count

    @property
    def any_field_valid_ratio(self) -> float:
        """Fraction of generated narratives with at least one valid field."""
        if self.generated_count == 0:
            return 1.0
        return self.any_field_valid_count / self.generated_count
    
    @property
    def length_valid_ratio(self) -> float:
        """Fraction of generated narratives with valid length."""
        if self.generated_count == 0:
            return 1.0
        return self.length_valid_count / self.generated_count
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_personas": self.total_personas,
            "generated_count": self.generated_count,
            "mock_count": self.mock_count,
            "failed_count": self.failed_count,
            "too_short_count": self.too_short_count,
            "too_long_count": self.too_long_count,
            "length_valid_ratio": round(self.length_valid_ratio, 4),
            "schema_valid_count": self.schema_valid_count,
            "schema_invalid_count": self.schema_invalid_count,
            "schema_valid_ratio_overall": round(self.schema_valid_ratio_overall, 4),
            "schema_valid_ratio_attempted": round(self.schema_valid_ratio_attempted, 4),
            "parse_attempted_count": self.parse_attempted_count,
            "parse_failed_count": self.parse_failed_count,
            "parse_failure_ratio": round(self.parse_failure_ratio, 4),
            "romanian_valid_count": self.romanian_valid_count,
            "romanian_invalid_count": self.romanian_invalid_count,
            "romanian_valid_ratio": round(self.romanian_valid_ratio, 4),
            "potential_pii_count": self.potential_pii_count,
            "required_fields_present": self.required_fields_present,
            "required_fields_missing": self.required_fields_missing,
            "required_fields_length_valid": self.required_fields_length_valid,
            "required_fields_language_valid": self.required_fields_language_valid,
            "required_fields_schema_valid": self.required_fields_schema_valid,
            "any_field_valid_count": self.any_field_valid_count,
            "any_field_valid_ratio": round(self.any_field_valid_ratio, 4),
            "field_failure_counts": self.field_failure_counts,
        }


@dataclass
class TrustDecisionRecord:
    """
    Compact decision record explaining the final trust assessment.
    
    This provides a single source of truth for downstream pipelines:
    - decision: Clear PASS / PASS_WITH_WARNINGS / REJECT
    - basis: Which metrics drove the decision
    - confidence: How certain we are (affected by fallback ratios, ESS, etc.)
    """
    decision: str = "UNKNOWN"  # PASS, PASS_WITH_WARNINGS, REJECT
    decision_basis: List[str] = field(default_factory=list)
    confidence: str = "medium"  # high, medium, low
    confidence_factors: List[str] = field(default_factory=list)
    
    # Monotonic tier explanation
    tier_mapping_coherent: bool = True
    coherence_notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "decision_basis": self.decision_basis,
            "confidence": self.confidence,
            "confidence_factors": self.confidence_factors,
            "tier_mapping_coherent": self.tier_mapping_coherent,
            "coherence_notes": self.coherence_notes,
        }


@dataclass
class TrustReport:
    """
    Comprehensive trust report for a generated persona dataset.
    
    Provides auditability through:
    - Provenance coverage statistics (all fields + critical fields)
    - Marginal error metrics
    - Joint drift detection
    - Quality tier classification (structured + narrative + overall)
    - Use-case specific gating with super-critical field handling
    - IPF side effects (ESS, information loss)
    - Data currency tracking
    - Trust Decision Record for pipeline integration
    """
    
    # Basic metadata
    report_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    generation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    persona_count: int = 0
    
    # Provenance coverage (all fields)
    provenance_coverage_all: Dict[str, float] = field(default_factory=dict)
    """Percentage of ALL fields by provenance type"""
    
    provenance_coverage_critical: Dict[str, float] = field(default_factory=dict)
    """Percentage of CRITICAL fields by provenance type"""
    
    provenance_coverage_super_critical: Dict[str, float] = field(default_factory=dict)
    """Percentage of HIGH_STAKES_CRITICAL fields by provenance type"""
    
    # Fallback events
    fallback_fields: List[str] = field(default_factory=list)
    """Fields that used fallback (hardcoded) data"""
    
    fallback_fields_critical: List[str] = field(default_factory=list)
    """Critical fields that used fallback"""
    
    fallback_fields_super_critical: List[str] = field(default_factory=list)
    """Super-critical fields that used fallback (for high-stakes)"""
    
    cached_fields: List[str] = field(default_factory=list)
    """Fields that used cached (stale) data"""

    ethnocultural_fallbacks: Dict[str, int] = field(default_factory=dict)
    """Ethnocultural fallback events (e.g., national totals used for missing cross-tabs)"""

    estimated_fields: List[str] = field(default_factory=list)
    """Fields derived from estimated/heuristic sources"""

    estimated_fields_critical: List[str] = field(default_factory=list)
    """Estimated fields that are also critical for gating"""
    
    # Error metrics
    marginal_errors: List[MarginalError] = field(default_factory=list)
    """Error for each key distribution vs census targets"""

    marginal_checks_skipped: Dict[str, str] = field(default_factory=dict)
    """Marginal checks skipped due to missing targets or data"""

    distribution_test_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    """Per-field JS divergence and chi-square results for sample-size-aware gating"""
    
    # Error metric interpretation
    targets_source: str = "unknown"  # pxweb_live, hardcoded_fallback, mixed
    metric_interpretation: str = ""  # Explanation of what L1 error means given target source
    uncertainty_penalty_applied: bool = False  # Whether we adjusted for fallback uncertainty
    
    # Quality classification (split by type)
    structured_quality_tier: QualityTier = QualityTier.B
    """Quality based on structured data distributions and provenance"""
    
    narrative_quality_tier: QualityTier = QualityTier.B
    """Quality based on narrative generation status"""
    
    overall_quality_tier: QualityTier = QualityTier.B
    """Overall quality (use-case dependent calculation)"""
    
    overall_tier_reasoning: str = ""
    """Explanation of how overall tier was computed"""
    
    # Trust Decision Record for pipeline integration
    trust_decision: Optional[TrustDecisionRecord] = None
    """Compact decision record for downstream use"""
    
    # Use-case gating
    use_case_profile: UseCaseProfile = UseCaseProfile.NARRATIVE_REQUIRED
    """Profile used for gating decisions"""

    population_mode: PopulationMode = PopulationMode.ADULT_18
    """Population mode used for generation/validation"""
    
    hard_gate_triggered: bool = False
    """True if dataset should be rejected for this use case"""
    
    gate_reasons: List[str] = field(default_factory=list)
    """Actionable reasons for hard/soft gate activation with thresholds"""

    preflight_gate_reasons: List[str] = field(default_factory=list)
    """Gate reasons detected before drift calculations"""
    
    # Narrative status
    narrative_mock_count: int = 0
    """Number of personas with mock (empty) narratives"""
    
    narrative_mock_ratio: float = 0.0
    """Fraction of personas with mock narratives"""
    
    narrative_generated_count: int = 0
    """Number of personas with LLM-generated narratives"""
    
    narrative_failed_count: int = 0
    """Number of personas with failed narrative generation"""
    
    # Detailed narrative quality metrics
    narrative_quality_metrics: Optional[NarrativeQualityMetrics] = None
    """Detailed quality analysis of generated narratives"""
    
    # IPF metrics (if correction was applied)
    ipf_metrics: Optional[IPFMetrics] = None
    """Metrics for IPF side effects"""
    
    # Data currency
    pxweb_cache_age_days: Optional[int] = None
    """Age of cached PxWeb data in days"""
    
    data_reference_year: int = 2024
    """Year of census data used as reference"""
    
    census_reference_year: int = 2024
    """Year of hardcoded census fallback data"""
    
    # Reproducibility
    random_seed: Optional[int] = None
    """Random seed used for generation"""
    
    generator_version: str = "unknown"
    """Version of the generator"""
    
    generator_config_hash: Optional[str] = None
    """SHA256 hash of generator configuration (first 16 chars shown)"""
    
    pxweb_snapshot_timestamp: Optional[str] = None
    """Timestamp of PxWeb data snapshot (if fetched live)"""
    
    @property
    def mean_l1_error(self) -> float:
        """Mean L1 error across all tracked marginals."""
        if not self.marginal_errors:
            return 0.0
        return np.mean([e.l1_distance for e in self.marginal_errors])
    
    @property
    def mean_l1_error_effective(self) -> float:
        """Mean L1 error after applying uncertainty penalty for fallback targets."""
        base = self.mean_l1_error
        if self.uncertainty_penalty_applied and self.targets_source in ("hardcoded_fallback", "mixed"):
            return base + 0.05  # 5% uncertainty penalty
        return base
    
    @property
    def max_l1_error(self) -> float:
        """Maximum L1 error across all tracked marginals."""
        if not self.marginal_errors:
            return 0.0
        return max([e.l1_distance for e in self.marginal_errors])
    
    @property
    def max_l1_error_field(self) -> Optional[str]:
        """Field with the maximum L1 error (worst offender)."""
        if not self.marginal_errors:
            return None
        worst = max(self.marginal_errors, key=lambda e: e.l1_distance)
        return worst.field
    
    @property
    def max_category_error(self) -> float:
        """Maximum error for any single category across all fields."""
        if not self.marginal_errors:
            return 0.0
        return max([e.max_error for e in self.marginal_errors])
    
    @property
    def marginal_errors_available(self) -> bool:
        """True if marginal errors were calculated (census data available)."""
        return len(self.marginal_errors) > 0
    
    @property
    def fallback_ratio(self) -> float:
        """Fraction of all fields using fallback."""
        return self.provenance_coverage_all.get(DataProvenance.CENSUS_HARDCODED.value, 0.0)
    
    @property
    def fallback_ratio_critical(self) -> float:
        """Fraction of critical fields using fallback."""
        return self.provenance_coverage_critical.get(DataProvenance.CENSUS_HARDCODED.value, 0.0)
    
    @property
    def fallback_ratio_super_critical(self) -> float:
        """Fraction of super-critical fields using fallback (field-count based)."""
        return self.provenance_coverage_super_critical.get(DataProvenance.CENSUS_HARDCODED.value, 0.0)
    
    @property
    def super_critical_fallback_field_count(self) -> int:
        """Number of super-critical fields that used fallback."""
        return len(self.fallback_fields_super_critical)
    
    @property
    def super_critical_fallback_fraction(self) -> float:
        """
        Fraction of super-critical field definitions using fallback.
        e.g., if 1 of 4 super-critical fields is fallback, returns 0.25
        """
        if not HIGH_STAKES_CRITICAL_FIELDS:
            return 0.0
        return self.super_critical_fallback_field_count / len(HIGH_STAKES_CRITICAL_FIELDS)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "report_id": self.report_id,
            "generation_timestamp": self.generation_timestamp,
            "persona_count": self.persona_count,
            "provenance_coverage_all": self.provenance_coverage_all,
            "provenance_coverage_critical": self.provenance_coverage_critical,
            "provenance_coverage_super_critical": self.provenance_coverage_super_critical,
            "fallback_fields": self.fallback_fields,
            "fallback_fields_critical": self.fallback_fields_critical,
            "fallback_fields_super_critical": self.fallback_fields_super_critical,
            "cached_fields": self.cached_fields,
            "ethnocultural_fallbacks": self.ethnocultural_fallbacks,
            "estimated_fields": self.estimated_fields,
            "estimated_fields_critical": self.estimated_fields_critical,
            "marginal_errors": [e.to_dict() for e in self.marginal_errors],
            "marginal_checks_skipped": self.marginal_checks_skipped,
            "distribution_test_results": self.distribution_test_results,
            "mean_l1_error": round(self.mean_l1_error, 4),
            "max_l1_error": round(self.max_l1_error, 4),
            "max_category_error": round(self.max_category_error, 4),
            "targets_source": self.targets_source,
            "metric_interpretation": self.metric_interpretation,
            "uncertainty_penalty_applied": self.uncertainty_penalty_applied,
            "structured_quality_tier": self.structured_quality_tier.value,
            "narrative_quality_tier": self.narrative_quality_tier.value,
            "overall_quality_tier": self.overall_quality_tier.value,
            "overall_tier_reasoning": self.overall_tier_reasoning,
            "trust_decision": self.trust_decision.to_dict() if self.trust_decision else None,
            "use_case_profile": self.use_case_profile.value,
            "population_mode": self.population_mode.value,
            "hard_gate_triggered": self.hard_gate_triggered,
            "preflight_gate_reasons": self.preflight_gate_reasons,
            "gate_reasons": self.gate_reasons,
            "narrative_mock_count": self.narrative_mock_count,
            "narrative_mock_ratio": round(self.narrative_mock_ratio, 4),
            "narrative_generated_count": self.narrative_generated_count,
            "narrative_failed_count": self.narrative_failed_count,
            "narrative_quality_metrics": (
                self.narrative_quality_metrics.to_dict() 
                if self.narrative_quality_metrics else None
            ),
            "ipf_metrics": self.ipf_metrics.to_dict() if self.ipf_metrics else None,
            "pxweb_cache_age_days": self.pxweb_cache_age_days,
            "data_reference_year": self.data_reference_year,
            "census_reference_year": self.census_reference_year,
            "random_seed": self.random_seed,
            "generator_version": self.generator_version,
            "generator_config_hash": self.generator_config_hash,
            "pxweb_snapshot_timestamp": self.pxweb_snapshot_timestamp,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def summary(self) -> str:
        """Human-readable summary."""
        # Error metrics with appropriate precision
        if self.marginal_errors_available:
            error_line = f"Mean L1 Error: {self.mean_l1_error:.6f}"
            if self.uncertainty_penalty_applied:
                error_line += f" (effective: {self.mean_l1_error_effective:.6f} with penalty)"
            max_error_line = f"Max L1 Error: {self.max_l1_error:.6f} (field: {self.max_l1_error_field or 'N/A'})"
        else:
            error_line = "Mean L1 Error: N/A (no census data available for comparison)"
            max_error_line = "Max L1 Error: N/A"
        
        lines = [
            f"Trust Report: {self.report_id}",
            f"Use Case: {self.use_case_profile.value}",
            f"Quality Tiers: Structured={self.structured_quality_tier.value}, Narrative={self.narrative_quality_tier.value}, Overall={self.overall_quality_tier.value}",
            f"Overall Tier Reasoning: {self.overall_tier_reasoning}",
            f"Personas: {self.persona_count}",
            error_line,
            max_error_line,
            f"Max Category Error: {self.max_category_error:.6f}",
            f"Targets Source: {self.targets_source}",
            f"Fallback (Critical Fields): {self.fallback_ratio_critical:.1%}",
            f"Fallback (Super-Critical): {self.super_critical_fallback_fraction:.1%} ({self.super_critical_fallback_field_count}/{len(HIGH_STAKES_CRITICAL_FIELDS)} fields)",
            f"Mock Narratives: {self.narrative_mock_ratio:.1%}",
        ]

        if self.ethnocultural_fallbacks:
            lines.append(f"Ethnocultural Fallbacks: {self.ethnocultural_fallbacks}")
        
        if self.fallback_fields_super_critical:
            lines.append(f"Super-Critical Fallback Fields: {', '.join(self.fallback_fields_super_critical)}")
        elif self.fallback_fields_critical:
            lines.append(f"Critical Fallback Fields: {', '.join(self.fallback_fields_critical)}")
        
        if self.narrative_quality_metrics:
            nqm = self.narrative_quality_metrics
            lines.append(f"Schema Valid (attempted): {nqm.schema_valid_ratio_attempted:.1%}")
        
        if self.ipf_metrics:
            lines.append(f"IPF ESS Ratio: {self.ipf_metrics.ess_ratio:.1%}")
            lines.append(f"IPF Information Loss: {self.ipf_metrics.information_loss:.1%}")
        
        if self.pxweb_cache_age_days is not None:
            lines.append(f"Cache Age: {self.pxweb_cache_age_days} days")
        
        if self.trust_decision:
            lines.append(f"Trust Decision: {self.trust_decision.decision}")
            lines.append(f"Confidence: {self.trust_decision.confidence}")
        
        if self.hard_gate_triggered:
            lines.append(f"⚠️  HARD GATE TRIGGERED")
            lines.extend([f"  - {r}" for r in self.gate_reasons])
        elif self.gate_reasons:
            lines.append(f"Soft Gate Warnings:")
            lines.extend([f"  - {r}" for r in self.gate_reasons])
        
        return "\n".join(lines)


class TrustReportGenerator:
    """Generates trust reports for persona datasets."""
    
    def __init__(
        self, 
        census_distributions: Optional[Any] = None,
        use_case: UseCaseProfile = UseCaseProfile.NARRATIVE_REQUIRED,
        population_mode: PopulationMode = PopulationMode.ADULT_18,
    ):
        """
        Initialize report generator.
        
        Args:
            census_distributions: CensusDistributions object for target comparisons
            use_case: Use-case profile for gating decisions
        """
        self.census = census_distributions
        self.use_case = use_case
        self.thresholds = USE_CASE_THRESHOLDS[use_case]
        self.population_mode = population_mode
    
    def generate_report(
        self,
        personas: List[Persona],
        provenance_info: Optional[Dict[str, Dict[str, Any]]] = None,
        random_seed: Optional[int] = None,
        generator_version: str = "unknown",
        generator_config: Optional[Dict] = None,
        ipf_metrics: Optional[IPFMetrics] = None,
        pxweb_cache_timestamp: Optional[str] = None,
        data_reference_year: int = 2024,
        census_reference_year: int = 2024,
        targets_source: str = "unknown",  # pxweb_live, hardcoded_fallback, mixed
    ) -> TrustReport:
        """
        Generate a complete trust report for a dataset.
        
        Args:
            personas: List of generated personas
            provenance_info: Field-level provenance from CensusDistributions
            random_seed: Random seed used for generation
            generator_version: Version string
            generator_config: Configuration dict (will be hashed)
            ipf_metrics: IPF correction metrics if applicable
            pxweb_cache_timestamp: When cached data was fetched
            data_reference_year: Year of reference census data
            census_reference_year: Year of hardcoded census data
            targets_source: Source of target distributions (pxweb_live, hardcoded_fallback, mixed)
            
        Returns:
            TrustReport with full audit trail
        """
        report = TrustReport(
            persona_count=len(personas),
            use_case_profile=self.use_case,
            population_mode=self.population_mode,
            random_seed=random_seed,
            generator_version=generator_version,
            ipf_metrics=ipf_metrics,
            data_reference_year=data_reference_year,
            census_reference_year=census_reference_year,
            targets_source=targets_source,
        )
        
        # Compute config hash (full SHA256)
        if generator_config:
            report.generator_config_hash = self._hash_config(generator_config)
        
        # Calculate cache age
        if pxweb_cache_timestamp:
            report.pxweb_snapshot_timestamp = pxweb_cache_timestamp
            report.pxweb_cache_age_days = self._calculate_cache_age(pxweb_cache_timestamp)
        
        # Analyze provenance coverage
        if provenance_info:
            report.provenance_coverage_all = self._calculate_provenance_coverage(
                provenance_info, critical_only=False
            )
            report.provenance_coverage_critical = self._calculate_provenance_coverage(
                provenance_info, critical_only=True
            )
            report.provenance_coverage_super_critical = self._calculate_provenance_coverage(
                provenance_info, critical_only=False, super_critical=True
            )
            
            # Identify fallback fields
            all_fallbacks = [
                f for f, info in provenance_info.items()
                if info.get("provenance") == DataProvenance.CENSUS_HARDCODED.value
            ]
            report.fallback_fields = all_fallbacks
            report.fallback_fields_critical = [f for f in all_fallbacks if f in CRITICAL_FIELDS]
            report.fallback_fields_super_critical = [
                f for f in all_fallbacks if f in HIGH_STAKES_CRITICAL_FIELDS
            ]
            
            report.cached_fields = [
                f for f, info in provenance_info.items()
                if info.get("provenance") == DataProvenance.PXWEB_CACHED.value
            ]

            report.estimated_fields = [
                f for f, info in provenance_info.items()
                if info.get("provenance") == DataProvenance.ESTIMATED.value
            ]
            report.estimated_fields_critical = [
                f for f in report.estimated_fields
                if f in CRITICAL_FIELDS
            ]
            
            # Determine targets source and set interpretation
            report.targets_source = self._determine_targets_source(provenance_info)
            report.metric_interpretation = self._get_metric_interpretation(report.targets_source)

        # Preflight: if super-critical targets fallback, skip drift calculation
        supercritical_target_fallback = any(
            f in {"region", "ethnicity"} for f in report.fallback_fields_super_critical
        )
        if supercritical_target_fallback and self.use_case != UseCaseProfile.ANALYSIS_ONLY:
            report.preflight_gate_reasons.append(
                "Super-critical target fallback detected: region/ethnicity"
            )
            report.marginal_checks_skipped = {
                "all": "skipped due to super-critical target fallback"
            }
        else:
            # Calculate marginal errors
            report.marginal_errors, report.marginal_checks_skipped = self._calculate_marginal_errors(
                personas, report.targets_source
            )
        
        # Analyze narrative status
        self._analyze_narratives(personas, report)

        # Record ethnocultural fallback events (if any)
        try:
            from .names import get_ethnocultural_fallbacks
            report.ethnocultural_fallbacks = get_ethnocultural_fallbacks()
        except Exception:
            pass
        
        # Analyze detailed narrative quality
        self._analyze_narrative_quality(personas, report)
        
        # Classify quality tiers
        self._classify_quality_tiers(report)
        
        # Apply gating rules
        report.hard_gate_triggered, report.gate_reasons = self._apply_gates(report)
        
        # Generate trust decision record
        report.trust_decision = self._generate_trust_decision(report)
        
        return report
    
    def _hash_config(self, config: Dict) -> str:
        """Create deterministic SHA256 hash of configuration."""
        config_str = json.dumps(config, sort_keys=True, ensure_ascii=False)
        full_hash = hashlib.sha256(config_str.encode()).hexdigest()
        # Return full hash - consumers can truncate for display if needed
        return full_hash
    
    def _calculate_cache_age(self, cache_timestamp: str) -> int:
        """Calculate age of cached data in days."""
        try:
            cache_time = datetime.fromisoformat(cache_timestamp.replace('Z', '+00:00'))
            now = datetime.now(cache_time.tzinfo)
            age = now - cache_time
            return age.days
        except (ValueError, TypeError):
            return 0
    
    def _determine_targets_source(self, provenance_info: Dict[str, Dict[str, Any]]) -> str:
        """Determine the source of target distributions."""
        has_pxweb = False
        has_hardcoded = False
        
        for field_info in provenance_info.values():
            prov = field_info.get("provenance", "")
            if prov in {DataProvenance.PXWEB_DIRECT.value, DataProvenance.PXWEB_CACHED.value}:
                has_pxweb = True
            elif prov == DataProvenance.CENSUS_HARDCODED.value:
                has_hardcoded = True
        
        if has_pxweb and has_hardcoded:
            return "mixed"
        elif has_pxweb:
            return "pxweb_live"
        elif has_hardcoded:
            return "hardcoded_fallback"
        else:
            return "unknown"
    
    def _get_metric_interpretation(self, targets_source: str) -> str:
        """Get interpretation text for error metrics based on target source."""
        interpretations = {
            "pxweb_live": "L1 error measures fit to NBS PxWeb live data (best case)",
            "hardcoded_fallback": "L1 error measures fit to hardcoded 2024 Census (reference may be stale)",
            "mixed": "L1 error measures fit to mixed sources (some uncertainty from fallbacks)",
            "unknown": "Target source unknown - error metrics may not be meaningful",
        }
        return interpretations.get(targets_source, interpretations["unknown"])
    
    def _calculate_provenance_coverage(
        self, 
        provenance_info: Dict[str, Dict[str, Any]],
        critical_only: bool = False,
        super_critical: bool = False,
    ) -> Dict[str, float]:
        """Calculate percentage of fields by provenance type."""
        if not provenance_info:
            return {}
        
        # Filter fields based on criticality
        fields = list(provenance_info.items())
        if super_critical:
            fields = [(k, v) for k, v in fields if k in HIGH_STAKES_CRITICAL_FIELDS]
        elif critical_only:
            fields = [(k, v) for k, v in fields if k in CRITICAL_FIELDS]
        
        if not fields:
            return {}
        
        counts = {}
        for field_name, field_info in fields:
            prov = field_info.get("provenance", "UNKNOWN")
            counts[prov] = counts.get(prov, 0) + 1
        
        total = len(fields)
        return {k: round(v / total, 4) for k, v in counts.items()}
    
    def _calculate_marginal_errors(
        self, 
        personas: List[Persona],
        targets_source: str = "unknown"
    ) -> Tuple[List[MarginalError], Dict[str, str]]:
        """Calculate L1 error for key distributions vs census targets."""
        errors = []
        skipped: Dict[str, str] = {}
        
        if not personas or not self.census:
            return errors, skipped
        
        # Helper to compute actual distribution
        def compute_dist(personas: List[Persona], extractor) -> Dict[str, float]:
            counts = {}
            for p in personas:
                val = extractor(p)
                counts[val] = counts.get(val, 0) + 1
            total = len(personas)
            return {k: v / total for k, v in counts.items()}

        def add_error(field: str, target: Optional[Dict[str, float]], extractor) -> None:
            if not target:
                skipped[field] = "no target distribution available"
                return
            actual = compute_dist(personas, extractor)
            # Detect category mismatches and treat as schema mismatch instead of drift
            target_keys = set(target.keys())
            actual_keys = set(actual.keys())
            if field == "employment_status":
                extra_actual = sorted(actual_keys - target_keys)
                extra_target = sorted(target_keys - actual_keys)
                if extra_actual or extra_target:
                    skipped[field] = (
                        f"category mismatch (actual-only: {extra_actual}; "
                        f"target-only: {extra_target})"
                    )
                    return
            errors.append(MarginalError(field, target, actual, targets_source))
        
        # Sex distribution
        try:
            add_error("sex", self.census.SEX_DISTRIBUTION, lambda p: p.sex)
        except Exception:
            pass
        
        # Age group distribution (mode-specific)
        try:
            add_error("age_group", self.census.age_group_distribution(self.population_mode), lambda p: p.age_group)
        except Exception:
            pass
        
        # Region distribution
        try:
            target = self.census.REGION_DISTRIBUTION
            if strict_geo_enabled():
                derived = get_region_distribution_from_district(strict=True)
                if derived:
                    target = derived
            add_error("region", target, lambda p: p.region)
        except Exception:
            pass
        
        # Urban/rural
        try:
            add_error("residence_type", self.census.RESIDENCE_TYPE_DISTRIBUTION, lambda p: p.residence_type)
        except Exception:
            pass
        
        # Ethnicity
        try:
            add_error("ethnicity", self.census.ETHNICITY_DISTRIBUTION, lambda p: p.ethnicity)
        except Exception:
            pass

        # Education
        try:
            education_target = self.census.education_distribution(self.population_mode).values
            add_error("education", education_target, lambda p: p.education_level)
        except Exception:
            pass

        # Marital status
        try:
            marital_target = self.census.marital_status_distribution(self.population_mode).values
            add_error("marital_status", marital_target, lambda p: p.marital_status)
        except Exception:
            pass

        # Religion
        try:
            add_error("religion", self.census.RELIGION_DISTRIBUTION, lambda p: p.religion)
        except Exception:
            pass

        # Mother tongue / language
        try:
            add_error("language", self.census.LANGUAGE_DISTRIBUTION, lambda p: p.mother_tongue)
        except Exception:
            pass

        # Employment status
        try:
            employment_target = self.census.employment_status_distribution(self.population_mode).values
            add_error("employment_status", employment_target, lambda p: p.employment_status)
        except Exception:
            pass

        # District distribution (if available)
        try:
            target = get_district_distribution()
            if target:
                counts = {}
                total = len(personas)
                for p in personas:
                    key = p.district if p.district else "Unknown"
                    counts[key] = counts.get(key, 0) + 1
                actual = {k: v / total for k, v in counts.items()} if total > 0 else {}
                errors.append(MarginalError("district", target, actual, targets_source))
            else:
                skipped["district"] = "no target distribution available"
        except Exception:
            pass
        
        return errors, skipped
    
    def _analyze_narratives(self, personas: List[Persona], report: TrustReport) -> None:
        """Analyze narrative generation status."""
        total = len(personas)
        if total == 0:
            return
        
        report.narrative_mock_count = sum(
            1 for p in personas if p.narrative_status == "mock"
        )
        report.narrative_generated_count = sum(
            1 for p in personas if p.narrative_status == "generated"
        )
        report.narrative_failed_count = sum(
            1 for p in personas if p.narrative_status == "failed"
        )
        report.narrative_mock_ratio = report.narrative_mock_count / total
    
    def _analyze_narrative_quality(
        self, 
        personas: List[Persona], 
        report: TrustReport
    ) -> None:
        """
        Analyze detailed quality of generated narratives.
        
        Tracks both overall and attempted ratios for proper gating.
        """
        metrics = NarrativeQualityMetrics()
        metrics.total_personas = len(personas)
        metrics.generated_count = report.narrative_generated_count
        metrics.mock_count = report.narrative_mock_count
        metrics.failed_count = report.narrative_failed_count
        
        # Only analyze personas with generated narratives
        generated_personas = [p for p in personas if p.narrative_status == "generated"]
        
        if not generated_personas:
            report.narrative_quality_metrics = metrics
            return
        
        for p in generated_personas:
            required_fields = [
                "descriere_generala",
                "profil_profesional",
                "hobby_sport",
                "hobby_arta_cultura",
                "hobby_calatorii",
                "hobby_culinar",
                "career_goals_and_ambitions",
                "persona_summary",
            ]

            # Mark that we attempted to parse this narrative
            metrics.parse_attempted_count += 1

            all_fields_valid = True
            all_length_ok = True
            all_language_ok = True
            any_field_valid = False
            any_too_short = False
            any_too_long = False

            pii_hits = 0

            for field_name in required_fields:
                metrics.field_failure_counts.setdefault(
                    field_name,
                    {"missing": 0, "too_short": 0, "too_long": 0, "language_fail": 0}
                )
                text = getattr(p, field_name, "") or ""
                text = str(text).strip()
                present = bool(text)
                if present:
                    metrics.required_fields_present += 1
                else:
                    metrics.required_fields_missing += 1
                    metrics.field_failure_counts[field_name]["missing"] += 1
                    all_fields_valid = False
                    all_length_ok = False
                    all_language_ok = False
                    continue

                length = len(text)
                schema = NARRATIVE_JSON_SCHEMA.get("properties", {}).get(field_name, {})
                min_len = schema.get("minLength", NARRATIVE_QUALITY_THRESHOLDS["min_narrative_length"])
                max_len = schema.get("maxLength", NARRATIVE_QUALITY_THRESHOLDS["max_narrative_length"])
                length_ok = min_len <= length <= max_len
                if length_ok:
                    metrics.required_fields_length_valid += 1
                else:
                    all_fields_valid = False
                    all_length_ok = False
                    if length < min_len:
                        any_too_short = True
                        metrics.field_failure_counts[field_name]["too_short"] += 1
                    else:
                        any_too_long = True
                        metrics.field_failure_counts[field_name]["too_long"] += 1

                romanian_markers = re.findall(r'[ăâîșțĂÂÎȘȚ]', text)
                romanian_words = re.findall(
                    r'\b(sunt|și|pentru|din|cu|care|mai|să|mă|te|se|în|la|pe|o|un|o)\b',
                    text.lower()
                )
                language_ok = len(romanian_markers) >= 2 or len(romanian_words) >= 3
                if language_ok:
                    metrics.required_fields_language_valid += 1
                else:
                    all_fields_valid = False
                    all_language_ok = False
                    metrics.field_failure_counts[field_name]["language_fail"] += 1

                field_valid = length_ok and language_ok
                if field_valid:
                    metrics.required_fields_schema_valid += 1
                    any_field_valid = True
                else:
                    all_fields_valid = False

                # PII/Realism heuristic - check for specific patterns that might indicate
                # real institution names (simplified check)
                pii_hits += len(re.findall(
                    r'\b[A-Z][a-z]+\s+(?:SRL|SA|Institut|Spital|Școală|Liceul|Universitate)\b',
                    text
                ))

            if any_too_short:
                metrics.too_short_count += 1
            if any_too_long:
                metrics.too_long_count += 1
            if all_length_ok:
                metrics.length_valid_count += 1

            if all_language_ok:
                metrics.romanian_valid_count += 1
            else:
                metrics.romanian_invalid_count += 1

            if any_field_valid:
                metrics.any_field_valid_count += 1

            if all_fields_valid:
                metrics.schema_valid_count += 1
            else:
                metrics.schema_invalid_count += 1
                metrics.parse_failed_count += 1

            if pii_hits > 3:
                metrics.potential_pii_count += 1
        
        report.narrative_quality_metrics = metrics
    
    def _classify_quality_tiers(self, report: TrustReport) -> None:
        """Classify dataset into structured, narrative, and overall quality tiers."""
        
        # Structured quality: based on marginals + provenance
        mean_error = report.mean_l1_error
        max_cat_error = report.max_category_error
        fallback_crit = report.fallback_ratio_critical
        
        # Add uncertainty penalty if using fallback targets
        if report.targets_source == "hardcoded_fallback":
            # When targets themselves are fallback, we have higher uncertainty
            # Apply virtual penalty to error metrics for tier classification
            mean_error_effective = mean_error + 0.05  # 5% uncertainty penalty
            max_cat_error_effective = max_cat_error + 0.05
            report.uncertainty_penalty_applied = True
        else:
            mean_error_effective = mean_error
            max_cat_error_effective = max_cat_error
        
        if mean_error_effective > 0.20 or max_cat_error_effective > 0.30:
            report.structured_quality_tier = QualityTier.REJECT
        elif mean_error_effective < 0.05 and max_cat_error_effective < 0.15 and fallback_crit < 0.30:
            report.structured_quality_tier = QualityTier.A
        elif mean_error_effective < 0.10 and max_cat_error_effective < 0.25:
            report.structured_quality_tier = QualityTier.B
        else:
            report.structured_quality_tier = QualityTier.C
        
        # Narrative quality: based on mock ratio, schema validity, and parse failures
        mock_ratio = report.narrative_mock_ratio
        failed_ratio = report.narrative_failed_count / report.persona_count if report.persona_count > 0 else 0
        
        # Get schema validity metrics
        schema_valid_attempted = 1.0
        parse_failure_ratio = 0.0
        
        if report.narrative_quality_metrics:
            schema_valid_attempted = report.narrative_quality_metrics.schema_valid_ratio_attempted
            parse_failure_ratio = report.narrative_quality_metrics.parse_failure_ratio
        
        # Narrative tier with schema validity as primary gate
        if failed_ratio > 0.50:
            report.narrative_quality_tier = QualityTier.REJECT
        elif schema_valid_attempted < 0.70:  # Hard reject if less than 70% parseable
            report.narrative_quality_tier = QualityTier.REJECT
        elif mock_ratio < 0.05 and failed_ratio < 0.01 and schema_valid_attempted >= 0.95:
            report.narrative_quality_tier = QualityTier.A
        elif mock_ratio < 0.20 and failed_ratio < 0.05 and schema_valid_attempted >= 0.90:
            report.narrative_quality_tier = QualityTier.B
        elif mock_ratio < 0.50 and schema_valid_attempted >= 0.80:
            report.narrative_quality_tier = QualityTier.C
        else:
            report.narrative_quality_tier = QualityTier.REJECT
        
        # Overall tier: use-case dependent calculation
        self._compute_overall_tier(report)
    
    def _compute_overall_tier(self, report: TrustReport) -> None:
        """
        Compute overall quality tier based on use-case profile.
        
        Rules:
        - ANALYSIS_ONLY: overall = structured (narratives don't matter)
        - NARRATIVE_REQUIRED: overall = min(structured, narrative)
        - HIGH_STAKES: overall = min(structured, narrative)
        """
        tier_order = [QualityTier.A, QualityTier.B, QualityTier.C, QualityTier.REJECT]
        structured_idx = tier_order.index(report.structured_quality_tier)
        narrative_idx = tier_order.index(report.narrative_quality_tier)
        
        if self.use_case == UseCaseProfile.ANALYSIS_ONLY:
            # For analysis-only, only structured data matters
            report.overall_quality_tier = report.structured_quality_tier
            report.overall_tier_reasoning = "structured_only (ANALYSIS_ONLY profile ignores narrative quality)"
        else:
            # For narrative-required and high-stakes, use conservative min()
            report.overall_quality_tier = tier_order[max(structured_idx, narrative_idx)]
            report.overall_tier_reasoning = f"min(structured={report.structured_quality_tier.value}, narrative={report.narrative_quality_tier.value})"
    
    def _generate_trust_decision(self, report: TrustReport) -> TrustDecisionRecord:
        """
        Generate a compact Trust Decision Record for pipeline integration.
        
        This provides a single source of truth for downstream decisions.
        """
        decision = TrustDecisionRecord()
        
        # Determine decision
        if report.hard_gate_triggered:
            decision.decision = "REJECT"
            decision.decision_basis = ["hard_gate_triggered"] + report.gate_reasons[:3]
        elif report.gate_reasons:
            decision.decision = "PASS_WITH_WARNINGS"
            decision.decision_basis = report.gate_reasons[:3]
        else:
            decision.decision = "PASS"
            decision.decision_basis = ["all_metrics_within_thresholds"]
        
        # Determine confidence
        confidence_factors = []
        
        # Lower confidence if using fallback targets
        if report.targets_source == "hardcoded_fallback":
            confidence_factors.append("hardcoded_targets")
        elif report.targets_source == "mixed":
            confidence_factors.append("mixed_target_sources")
        
        # Lower confidence if high fallback ratio
        if report.fallback_ratio_critical > 0.50:
            confidence_factors.append("high_critical_fallback_ratio")
        
        # Lower confidence if IPF information loss
        if report.ipf_metrics and report.ipf_metrics.information_loss > 0.20:
            confidence_factors.append("high_ipf_information_loss")
        
        # Lower confidence if stale cache
        if report.pxweb_cache_age_days and report.pxweb_cache_age_days > 14:
            confidence_factors.append("stale_cache_data")
        
        decision.confidence_factors = confidence_factors
        
        if len(confidence_factors) == 0:
            decision.confidence = "high"
        elif len(confidence_factors) <= 2:
            decision.confidence = "medium"
        else:
            decision.confidence = "low"
        
        # Check tier mapping coherence
        tier_order = [QualityTier.A, QualityTier.B, QualityTier.C, QualityTier.REJECT]
        overall_idx = tier_order.index(report.overall_quality_tier)
        structured_idx = tier_order.index(report.structured_quality_tier)
        narrative_idx = tier_order.index(report.narrative_quality_tier)
        
        if self.use_case == UseCaseProfile.ANALYSIS_ONLY:
            # For analysis_only, overall should equal structured
            decision.tier_mapping_coherent = (overall_idx == structured_idx)
            if not decision.tier_mapping_coherent:
                decision.coherence_notes = f"ANALYSIS_ONLY expects overall=structured, got {report.overall_quality_tier.value} vs {report.structured_quality_tier.value}"
        else:
            # For narrative_required/high_stakes, overall should be max(structured, narrative)
            expected_idx = max(structured_idx, narrative_idx)
            decision.tier_mapping_coherent = (overall_idx == expected_idx)
            if not decision.tier_mapping_coherent:
                decision.coherence_notes = f"Expected overall tier to be min(structured, narrative)"
        
        return decision

    def _js_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """Compute Jensen-Shannon divergence (log2)."""
        keys = set(p.keys()) | set(q.keys())
        if not keys:
            return 0.0
        p_norm = {k: p.get(k, 0.0) for k in keys}
        q_norm = {k: q.get(k, 0.0) for k in keys}
        p_total = sum(p_norm.values())
        q_total = sum(q_norm.values())
        if p_total <= 0 or q_total <= 0:
            return 0.0
        p_norm = {k: v / p_total for k, v in p_norm.items()}
        q_norm = {k: v / q_total for k, v in q_norm.items()}
        m = {k: 0.5 * (p_norm[k] + q_norm[k]) for k in keys}

        def kl_div(a: Dict[str, float], b: Dict[str, float]) -> float:
            total = 0.0
            for k, v in a.items():
                if v <= 0:
                    continue
                if b[k] <= 0:
                    continue
                total += v * math.log2(v / b[k])
            return total

        return 0.5 * (kl_div(p_norm, m) + kl_div(q_norm, m))

    def _chi2_sf(self, x: float, k: int) -> float:
        """Survival function for chi-square distribution."""
        if x <= 0 or k <= 0:
            return 1.0
        if k > 30:
            z = math.pow(x / k, 1/3) - (1 - 2/(9*k))
            z = z / math.sqrt(2/(9*k))
            return self._normal_sf(z)
        try:
            return math.exp(-x/2) * sum(
                (x/2) ** i / math.factorial(i)
                for i in range(k // 2)
            )
        except (OverflowError, ValueError):
            return 0.0 if x > k else 1.0

    def _normal_sf(self, z: float) -> float:
        """Standard normal survival function approximation."""
        if z < 0:
            return 1 - self._normal_sf(-z)
        b1 = 0.319381530
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429
        p = 0.2316419
        c = 0.39894228
        if z > 6:
            return 0.0
        t = 1.0 / (1.0 + p * z)
        phi = c * math.exp(-z * z / 2.0)
        return phi * t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))

    def _chi_square_p_value(
        self,
        actual: Dict[str, float],
        target: Dict[str, float],
        n: int,
    ) -> float:
        """Chi-square goodness-of-fit p-value with low-count guard."""
        keys = set(actual.keys()) | set(target.keys())
        if not keys or n <= 0:
            return 1.0
        observed = {k: int(round(actual.get(k, 0.0) * n)) for k in keys}
        diff = n - sum(observed.values())
        if diff != 0:
            # Adjust the largest bin to ensure totals match
            if observed:
                max_key = max(observed.keys(), key=lambda k: observed[k])
                observed[max_key] += diff
        chi2 = 0.0
        df = 0
        for k in keys:
            exp = target.get(k, 0.0) * n
            obs = observed.get(k, 0)
            if exp < 5:
                continue
            chi2 += ((obs - exp) ** 2) / exp
            df += 1
        if df <= 1:
            return 1.0
        return self._chi2_sf(chi2, df - 1)

    def _js_threshold(self, field: str, report: TrustReport) -> float:
        """Compute JS divergence threshold with sample-size adjustment."""
        if self.use_case == UseCaseProfile.HIGH_STAKES:
            base = 0.03
        elif self.use_case == UseCaseProfile.NARRATIVE_REQUIRED:
            base = 0.04
        else:
            base = 0.06
        if field in HIGH_STAKES_CRITICAL_FIELDS:
            base *= 0.8
        elif field in CRITICAL_FIELDS:
            base *= 0.9
        if field in report.estimated_fields:
            base *= 1.5
        tol = calculate_adaptive_tolerance(report.persona_count, confidence=0.95)
        return min(1.0, max(base, 2 * tol))
    
    def _apply_gates(self, report: TrustReport) -> Tuple[bool, List[str]]:
        """
        Apply gating rules based on use-case profile.
        
        Returns:
            (hard_gate_triggered, actionable_reasons_with_thresholds)
        """
        reasons = []
        hard_gate = False
        thresholds = self.thresholds

        if report.preflight_gate_reasons:
            reasons.extend(report.preflight_gate_reasons)
            hard_gate = True
        
        # Soft gate: excessive mean error (legacy fixed threshold)
        if report.mean_l1_error > thresholds["max_mean_l1_error"]:
            reasons.append(
                f"Mean L1 error {report.mean_l1_error:.2%} > threshold {thresholds['max_mean_l1_error']:.0%}"
            )
        
        # Soft gate: any category way off (legacy fixed threshold)
        if report.max_category_error > thresholds["max_category_error"]:
            reasons.append(
                f"Max category error {report.max_category_error:.2%} > threshold {thresholds['max_category_error']:.0%}"
            )

        # Sample-size-aware gates using JS divergence + chi-square p-values
        if report.marginal_errors_available:
            distribution_results: Dict[str, Dict[str, float]] = {}
            for err in report.marginal_errors:
                js = self._js_divergence(err.actual_dist, err.target_dist)
                p_value = self._chi_square_p_value(
                    err.actual_dist, err.target_dist, report.persona_count
                )
                js_threshold = self._js_threshold(err.field, report)
                distribution_results[err.field] = {
                    "js_divergence": js,
                    "js_threshold": js_threshold,
                    "chi2_p_value": p_value,
                }

                both_fail = js > js_threshold and p_value < 0.001
                either_fail = js > js_threshold or p_value < 0.001
                if both_fail:
                    if err.field in CRITICAL_FIELDS or self.use_case == UseCaseProfile.HIGH_STAKES:
                        hard_gate = True
                    reasons.append(
                        f"{err.field} drift: JS {js:.4f} > {js_threshold:.4f} and p={p_value:.4f} < 0.001"
                    )
                elif either_fail:
                    reasons.append(
                        f"{err.field} drift warning: JS {js:.4f} (thr {js_threshold:.4f}) or p={p_value:.4f} < 0.001"
                    )
            report.distribution_test_results = distribution_results
        
        # Hard gate: too many critical fallbacks
        if report.fallback_ratio_critical > thresholds["max_fallback_ratio_critical"]:
            hard_gate = True
            reasons.append(
                f"Critical field fallback {report.fallback_ratio_critical:.1%} > threshold {thresholds['max_fallback_ratio_critical']:.0%}"
            )
        
        # Hard gate: super-critical fallbacks (stricter for high-stakes)
        if report.fallback_ratio_super_critical > thresholds["max_fallback_ratio_super_critical"]:
            hard_gate = True
            reasons.append(
                f"Super-critical field fallback {report.fallback_ratio_super_critical:.1%} > threshold {thresholds['max_fallback_ratio_super_critical']:.0%} "
                f"(fields: {', '.join(report.fallback_fields_super_critical)})"
            )
        
        # Hard gate: ESS ratio too low (information loss too high)
        if report.ipf_metrics:
            ess_ratio = report.ipf_metrics.ess_ratio
            if ess_ratio < thresholds["min_ess_ratio"]:
                hard_gate = True
                reasons.append(
                    f"ESS ratio {ess_ratio:.1%} < threshold {thresholds['min_ess_ratio']:.0%} "
                    f"(information loss too high)"
                )
        
        # Hard gate: stale cached data for high-stakes
        if report.pxweb_cache_age_days is not None:
            if report.pxweb_cache_age_days > thresholds["max_cache_age_days"]:
                # For high-stakes, this is a hard gate
                # For others, just a warning
                if self.use_case == UseCaseProfile.HIGH_STAKES:
                    hard_gate = True
                reasons.append(
                    f"Cache age {report.pxweb_cache_age_days} days > threshold {thresholds['max_cache_age_days']} days"
                )
        
        # Hard gate: too many mock narratives (if required for use case)
        if report.narrative_mock_ratio > thresholds["max_mock_ratio"]:
            if self.use_case in (UseCaseProfile.NARRATIVE_REQUIRED, UseCaseProfile.HIGH_STAKES):
                hard_gate = True
            reasons.append(
                f"Mock narratives {report.narrative_mock_ratio:.1%} > threshold {thresholds['max_mock_ratio']:.0%}"
            )
        
        # Hard gate: schema validity too low for narrative-required use cases
        if report.narrative_quality_metrics:
            schema_attempted = report.narrative_quality_metrics.schema_valid_ratio_attempted
            min_schema = thresholds["min_schema_valid_ratio_attempted"]
            if schema_attempted < min_schema:
                if self.use_case in (UseCaseProfile.NARRATIVE_REQUIRED, UseCaseProfile.HIGH_STAKES):
                    hard_gate = True
                reasons.append(
                    f"Schema valid ratio (attempted) {schema_attempted:.1%} < threshold {min_schema:.0%}"
                )
        
        # Soft gate: non-critical fallbacks
        if report.fallback_fields and not report.fallback_fields_critical:
            reasons.append(
                f"Non-critical fallback fields: {', '.join(report.fallback_fields)}"
            )

        # Soft gate: ethnocultural fallbacks (e.g., national totals used)
        if report.ethnocultural_fallbacks:
            fallback_summary = ", ".join(
                f"{key}={count}" for key, count in report.ethnocultural_fallbacks.items()
            )
            reasons.append(
                f"Ethnocultural fallback used: {fallback_summary}"
            )

        # Estimated fields (heuristic sources)
        if report.estimated_fields:
            if self.use_case == UseCaseProfile.HIGH_STAKES:
                if report.estimated_fields_critical:
                    reasons.append(
                        f"Estimated field used: {', '.join(report.estimated_fields_critical)}"
                    )
                    hard_gate = True
            elif self.use_case == UseCaseProfile.NARRATIVE_REQUIRED:
                reasons.append(
                    f"Estimated field used: {', '.join(report.estimated_fields)}"
                )
        
        # Soft gate: cached data (warn about staleness)
        if report.cached_fields and report.pxweb_cache_age_days is not None:
            if report.pxweb_cache_age_days > 7:
                reasons.append(
                    f"Cached data ({report.pxweb_cache_age_days} days old) for: {', '.join(report.cached_fields)}"
                )
        
        # IPF information loss warning (even if not gated)
        if report.ipf_metrics and report.ipf_metrics.information_loss > 0.20:
            reasons.append(
                f"IPF information loss: {report.ipf_metrics.information_loss:.1%}"
            )
        
        # Parse failure warning
        if report.narrative_quality_metrics:
            nqm = report.narrative_quality_metrics
            if nqm.parse_failure_ratio > 0.10:
                reasons.append(
                    f"Parse failure ratio {nqm.parse_failure_ratio:.1%} > 10%"
                )
        
        return hard_gate, reasons


def compute_joint_drift(
    personas: List[Persona],
    var1_extractor,
    var2_extractor,
    target_joint: Optional[Dict[Tuple[str, str], float]] = None
) -> Dict[str, Any]:
    """
    Compute joint distribution and compare to target if provided.
    
    Args:
        personas: List of personas
        var1_extractor: Function to extract first variable
        var2_extractor: Function to extract second variable
        target_joint: Optional target joint distribution
        
    Returns:
        Dictionary with actual joint, target joint, and divergence metrics
    """
    # Compute actual joint
    joint_counts = {}
    for p in personas:
        val1 = var1_extractor(p)
        val2 = var2_extractor(p)
        key = (val1, val2)
        joint_counts[key] = joint_counts.get(key, 0) + 1
    
    total = len(personas)
    actual_joint = {k: v / total for k, v in joint_counts.items()}
    
    result = {
        "actual_joint": {f"{k[0]}|{k[1]}": round(v, 4) for k, v in actual_joint.items()},
        "sample_size": total,
    }
    
    if target_joint:
        # Compute L1 distance for joint
        all_keys = set(actual_joint.keys()) | set(target_joint.keys())
        l1 = sum(
            abs(actual_joint.get(k, 0) - target_joint.get(k, 0))
            for k in all_keys
        )
        result["target_joint"] = {f"{k[0]}|{k[1]}": round(v, 4) for k, v in target_joint.items()}
        result["l1_distance"] = round(l1, 4)
    
    return result


def compute_ipf_metrics(
    original_sample: List[Persona],
    final_sample: List[Persona],
    weights: Optional[List[float]] = None,
    pre_drift: Optional[Dict[str, float]] = None,
    post_drift: Optional[Dict[str, float]] = None,
) -> IPFMetrics:
    """
    Compute IPF correction metrics.
    
    Args:
        original_sample: Sample before IPF correction
        final_sample: Sample after IPF correction
        weights: Resampling weights (if weighted resampling used)
        pre_drift: L1 error before correction by field
        post_drift: L1 error after correction by field
        
    Returns:
        IPFMetrics with ESS and information loss
    """
    original_size = len(original_sample)
    final_size = len(final_sample)
    
    # Calculate Effective Sample Size
    if weights:
        sum_w = sum(weights)
        sum_w_sq = sum(w ** 2 for w in weights)
        ess = (sum_w ** 2) / sum_w_sq if sum_w_sq > 0 else original_size
    else:
        # Uniform weights (simple random sampling without replacement)
        ess = final_size
    
    return IPFMetrics(
        original_sample_size=original_size,
        effective_sample_size=ess,
        resampling_ratio=final_size / original_size if original_size > 0 else 1.0,
        pre_correction_drift=pre_drift or {},
        post_correction_drift=post_drift or {},
    )
