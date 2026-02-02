"""
Machine-readable gate codes for trust decisions.

These codes provide stable identifiers for gate conditions, enabling
automated routing and escalation in downstream systems (e.g., Gradient).

Codes follow the pattern: CATEGORY_SUBCATEGORY_DETAIL
"""

from enum import Enum
from typing import Set


class GateCode(Enum):
    """
    Machine-readable codes for trust gate conditions.
    
    These codes are stable across versions and should be used for:
    - Automated routing decisions
    - Escalation triggers
    - Metrics and alerting
    - Audit trails
    """
    
    # ========================================================================
    # DATA QUALITY CODES
    # ========================================================================
    
    # Marginal error issues
    MARGINAL_ERROR_HIGH = "MARGINAL_ERROR_HIGH"
    """Mean L1 error exceeds threshold for use case"""
    
    MARGINAL_ERROR_CRITICAL = "MARGINAL_ERROR_CRITICAL"
    """Maximum category error exceeds threshold"""
    
    # Fallback data usage
    FALLBACK_CRITICAL_FIELD = "FALLBACK_CRITICAL_FIELD"
    """Critical field using hardcoded fallback data"""
    
    FALLBACK_SUPERCRITICAL_FIELD = "FALLBACK_SUPERCRITICAL_FIELD"
    """Super-critical field (ethnicity, education, employment, region) using fallback"""

    TARGET_FALLBACK_SUPERCRITICAL = "TARGET_FALLBACK_SUPERCRITICAL"
    """Super-critical target distribution missing; skip drift checks"""

    ESTIMATED_FIELD_USED = "ESTIMATED_FIELD_USED"
    """Estimated/heuristic field used in generation"""
    
    # Data freshness
    PXWEB_CACHE_STALE = "PXWEB_CACHE_STALE"
    """PxWeb cached data exceeds max age for use case"""
    
    PXWEB_FETCH_FAILED = "PXWEB_FETCH_FAILED"
    """Could not fetch from PxWeb, using fallbacks"""

    ETHNOCULTURAL_FALLBACK_USED = "ETHNOCULTURAL_FALLBACK_USED"
    """Ethnocultural cross-tab fallback used (e.g., national totals for missing ethnicity)"""
    
    # ========================================================================
    # IPF / STATISTICAL CORRECTION CODES
    # ========================================================================
    
    ESS_TOO_LOW = "ESS_TOO_LOW"
    """Effective Sample Size ratio below threshold (high information loss)"""
    
    WEIGHT_CONCENTRATION_HIGH = "WEIGHT_CONCENTRATION_HIGH"
    """Weight concentration indicates over-representation of some personas"""
    
    IPF_DIVERGENCE = "IPF_DIVERGENCE"
    """IPF correction did not converge to stable solution"""
    
    # ========================================================================
    # NARRATIVE QUALITY CODES
    # ========================================================================
    
    NARRATIVE_SCHEMA_INVALID = "NARRATIVE_SCHEMA_INVALID"
    """Generated narratives don't match expected schema/sections"""
    
    NARRATIVE_PARSE_FAILURE = "NARRATIVE_PARSE_FAILURE"
    """Failed to parse LLM response into structured sections"""
    
    NARRATIVE_TOO_SHORT = "NARRATIVE_TOO_SHORT"
    """Narrative content below minimum length threshold"""
    
    NARRATIVE_TOO_LONG = "NARRATIVE_TOO_LONG"
    """Narrative content exceeds maximum length threshold"""
    
    NARRATIVE_LANGUAGE_INVALID = "NARRATIVE_LANGUAGE_INVALID"
    """Narrative missing Romanian diacritics or markers"""
    
    NARRATIVE_PRONOUN_MISMATCH = "NARRATIVE_PRONOUN_MISMATCH"
    """Pronouns in narrative don't match persona sex"""
    
    NARRATIVE_MOCK_RATIO_HIGH = "NARRATIVE_MOCK_RATIO_HIGH"
    """Too many personas have mock (empty) narratives"""
    
    NARRATIVE_PII_DETECTED = "NARRATIVE_PII_DETECTED"
    """Potential personally identifiable information detected"""
    
    # ========================================================================
    # CONFIGURATION CODES
    # ========================================================================
    
    CONFIG_INVALID_SEED = "CONFIG_INVALID_SEED"
    """Random seed configuration issue"""
    
    CONFIG_VERSION_MISMATCH = "CONFIG_VERSION_MISMATCH"
    """Generator version incompatible with requested features"""
    
    CONFIG_INCOMPLETE = "CONFIG_INCOMPLETE"
    """Required configuration parameters missing"""
    
    # ========================================================================
    # SYSTEM CODES
    # ========================================================================
    
    SYSTEM_MEMORY_EXCEEDED = "SYSTEM_MEMORY_EXCEEDED"
    """Generation exceeded available memory"""
    
    SYSTEM_TIMEOUT = "SYSTEM_TIMEOUT"
    """Generation exceeded time limits"""
    
    SYSTEM_CHECKPOINT_CORRUPT = "SYSTEM_CHECKPOINT_CORRUPT"
    """Checkpoint file corrupted or unreadable"""
    
    # ========================================================================
    # SUCCESS / WARNING CODES
    # ========================================================================
    
    PASS = "PASS"
    """All quality gates passed"""
    
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    """Quality gates passed but warnings present (review recommended)"""
    
    def is_hard_gate(self) -> bool:
        """
        Determine if this code represents a hard gate (rejection).
        
        Returns:
            True if this code should trigger rejection
        """
        hard_gate_codes: Set[str] = {
            GateCode.MARGINAL_ERROR_CRITICAL.value,
            GateCode.FALLBACK_SUPERCRITICAL_FIELD.value,
            GateCode.TARGET_FALLBACK_SUPERCRITICAL.value,
            GateCode.ESS_TOO_LOW.value,
            GateCode.IPF_DIVERGENCE.value,
            GateCode.NARRATIVE_SCHEMA_INVALID.value,
            GateCode.NARRATIVE_PRONOUN_MISMATCH.value,
            GateCode.NARRATIVE_MOCK_RATIO_HIGH.value,
            GateCode.SYSTEM_MEMORY_EXCEEDED.value,
            GateCode.SYSTEM_TIMEOUT.value,
        }
        return self.value in hard_gate_codes
    
    def is_warning(self) -> bool:
        """
        Determine if this code represents a warning (pass with notes).
        
        Returns:
            True if this code is a warning only
        """
        return not self.is_hard_gate() and self not in {
            GateCode.PASS,
            GateCode.PASS_WITH_WARNINGS,
        }
    
    def category(self) -> str:
        """Get the category prefix of the code."""
        return self.value.split('_')[0]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "code": self.value,
            "category": self.category(),
            "is_hard_gate": self.is_hard_gate(),
            "is_warning": self.is_warning(),
        }


# Mapping from human-readable gate reasons to machine codes
# Used for backward compatibility and migration
GATE_REASON_TO_CODE = {
    # Marginal errors
    "Mean L1 error exceeds threshold": GateCode.MARGINAL_ERROR_HIGH,
    "Maximum category error exceeds threshold": GateCode.MARGINAL_ERROR_CRITICAL,
    
    # Fallbacks
    "Critical field using fallback data": GateCode.FALLBACK_CRITICAL_FIELD,
    "Super-critical field using fallback": GateCode.FALLBACK_SUPERCRITICAL_FIELD,
    "Super-critical target fallback": GateCode.TARGET_FALLBACK_SUPERCRITICAL,
    "Ethnocultural fallback used": GateCode.ETHNOCULTURAL_FALLBACK_USED,
    "Estimated field used": GateCode.ESTIMATED_FIELD_USED,
    
    # Data freshness
    "PxWeb cache is stale": GateCode.PXWEB_CACHE_STALE,
    "Could not fetch from PxWeb": GateCode.PXWEB_FETCH_FAILED,
    
    # IPF issues
    "ESS ratio below threshold": GateCode.ESS_TOO_LOW,
    "Weight concentration too high": GateCode.WEIGHT_CONCENTRATION_HIGH,
    "IPF did not converge": GateCode.IPF_DIVERGENCE,
    
    # Narrative issues
    "Narrative schema validation failed": GateCode.NARRATIVE_SCHEMA_INVALID,
    "Failed to parse narrative response": GateCode.NARRATIVE_PARSE_FAILURE,
    "Narrative too short": GateCode.NARRATIVE_TOO_SHORT,
    "Narrative too long": GateCode.NARRATIVE_TOO_LONG,
    "Narrative missing Romanian diacritics": GateCode.NARRATIVE_LANGUAGE_INVALID,
    "Pronoun mismatch in narrative": GateCode.NARRATIVE_PRONOUN_MISMATCH,
    "Too many mock narratives": GateCode.NARRATIVE_MOCK_RATIO_HIGH,
    "Potential PII detected": GateCode.NARRATIVE_PII_DETECTED,
}


def map_reason_to_code(reason: str) -> GateCode:
    """
    Map a human-readable reason to a machine code.
    
    Args:
        reason: Human-readable gate reason
        
    Returns:
        Corresponding GateCode, or GateCode.PASS if no match
    """
    for pattern, code in GATE_REASON_TO_CODE.items():
        if pattern.lower() in reason.lower():
            return code
    
    # Default: parse from the reason if it looks like a code
    try:
        return GateCode(reason.upper().replace(" ", "_"))
    except ValueError:
        return GateCode.PASS


def get_escalation_priority(codes: list[GateCode]) -> int:
    """
    Get escalation priority for a set of gate codes.
    
    Lower numbers = higher priority (escalate first).
    
    Args:
        codes: List of gate codes
        
    Returns:
        Priority level (1 = highest, 10 = lowest)
    """
    if not codes:
        return 10
    
    # Priority mapping
    priority_map = {
        GateCode.NARRATIVE_PII_DETECTED: 1,
        GateCode.FALLBACK_SUPERCRITICAL_FIELD: 2,
        GateCode.TARGET_FALLBACK_SUPERCRITICAL: 2,
        GateCode.ESS_TOO_LOW: 3,
        GateCode.MARGINAL_ERROR_CRITICAL: 4,
        GateCode.NARRATIVE_PRONOUN_MISMATCH: 5,
        GateCode.PXWEB_FETCH_FAILED: 6,
        GateCode.SYSTEM_MEMORY_EXCEEDED: 7,
        GateCode.SYSTEM_TIMEOUT: 8,
    }
    
    return min(priority_map.get(code, 10) for code in codes)


def sort_gate_codes(codes: list[GateCode]) -> list[GateCode]:
    """
    Sort gate codes for stable, deterministic output.
    
    Ordering:
    1. Hard gates first (alphabetical within)
    2. Warnings second (alphabetical within)
    3. Success codes last
    
    Args:
        codes: List of gate codes (may contain duplicates)
        
    Returns:
        Deduplicated, sorted list of gate codes
    """
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            unique.append(code)
    
    # Sort: hard gates first, then warnings, then success
    # Within each group, alphabetical by value
    def sort_key(code: GateCode) -> tuple:
        if code.is_hard_gate():
            return (0, code.value)
        elif code.is_warning():
            return (1, code.value)
        else:
            # PASS / PASS_WITH_WARNINGS
            return (2, code.value)
    
    return sorted(unique, key=sort_key)
