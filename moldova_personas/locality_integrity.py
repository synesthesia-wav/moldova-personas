"""
Locality Integrity Checks Module

Validates locality configuration to ensure consistency between
residence_type and settlement_type, preventing silent failures
during city selection.

This module provides:
- Startup validation of locality configuration
- Runtime checks for empty candidate sets
- Fail-fast vs fallback strategies based on use-case profile
"""

import json
import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from .trust_report import UseCaseProfile

logger = logging.getLogger(__name__)

# Minimum number of candidates required per region/residence combination
MIN_CANDIDATES_URBAN = 1
MIN_CANDIDATES_RURAL = 2  # Rural needs more options for variety


class LocalityIntegrityError(Exception):
    """Raised when locality configuration has critical integrity issues."""
    pass


@dataclass
class LocalityValidationResult:
    """Result of locality configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    region_coverage: Dict[str, Dict[str, int]]  # region -> {urban: count, rural: count}
    
    def __str__(self) -> str:
        lines = ["Locality Validation Result:"]
        lines.append(f"  Valid: {self.is_valid}")
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def validate_locality_config(
    config_path: Optional[str] = None,
    localities_data: Optional[Dict] = None
) -> LocalityValidationResult:
    """
    Validate locality configuration for integrity issues.
    
    Checks performed:
    1. Every region has at least N urban and N rural candidates
    2. No contradictory metadata (city + rural, village + urban)
    3. All entries have required fields (name, settlement_type, implied_residence)
    4. settlement_type values are valid (city, town, village)
    5. implied_residence values are valid (Urban, Rural)
    6. No duplicate locality names within a region
    
    Args:
        config_path: Path to localities_by_region.json (optional)
        localities_data: Already loaded localities dict (optional)
        
    Returns:
        LocalityValidationResult with detailed findings
        
    Raises:
        LocalityIntegrityError: If config cannot be loaded
    """
    errors = []
    warnings = []
    region_coverage = {}
    
    # Load config if not provided
    if localities_data is None:
        if config_path is None:
            # Default path relative to package
            package_dir = Path(__file__).parent.parent
            config_path = package_dir / "config" / "localities_by_region.json"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                localities_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise LocalityIntegrityError(f"Cannot load locality config: {e}")
    
    valid_settlement_types = {"city", "town", "village", "rural"}
    valid_residence_types = {"Urban", "Rural"}
    
    for region, localities in localities_data.items():
        if isinstance(region, str) and region.startswith("_"):
            continue

        # Support both legacy list format and new dict-with-localities format
        if isinstance(localities, dict):
            localities = localities.get("localities", [])
        elif not isinstance(localities, list):
            errors.append(f"{region}: Invalid localities format (expected list or dict)")
            continue
        urban_count = 0
        rural_count = 0
        seen_names = set()
        
        for i, locality in enumerate(localities):
            # Check required fields
            if "name" not in locality:
                errors.append(f"{region}[{i}]: Missing 'name' field")
                continue
            
            name = locality["name"]
            
            # Check for duplicates
            if name in seen_names:
                errors.append(f"{region}: Duplicate locality name '{name}'")
            seen_names.add(name)
            
            # Validate settlement_type
            settlement_type = locality.get("settlement_type")
            if settlement_type is None:
                errors.append(f"{region}.{name}: Missing 'settlement_type'")
            elif settlement_type not in valid_settlement_types:
                errors.append(
                    f"{region}.{name}: Invalid settlement_type '{settlement_type}'. "
                    f"Must be one of: {valid_settlement_types}"
                )
            
            # Validate implied_residence
            implied_residence = locality.get("implied_residence")
            if implied_residence is None:
                errors.append(f"{region}.{name}: Missing 'implied_residence'")
            elif implied_residence not in valid_residence_types:
                errors.append(
                    f"{region}.{name}: Invalid implied_residence '{implied_residence}'. "
                    f"Must be one of: {valid_residence_types}"
                )
            
            # Check for logical contradictions
            if settlement_type and implied_residence:
                # Cities should be urban
                if settlement_type == "city" and implied_residence != "Urban":
                    errors.append(
                        f"{region}.{name}: Contradiction - city with implied_residence=Rural"
                    )
                # Villages/rural should be rural
                if settlement_type in {"village", "rural"} and implied_residence != "Rural":
                    errors.append(
                        f"{region}.{name}: Contradiction - village with implied_residence=Urban"
                    )
            
            # Count by residence type
            if implied_residence == "Urban":
                urban_count += 1
            elif implied_residence == "Rural":
                rural_count += 1
        
        region_coverage[region] = {"urban": urban_count, "rural": rural_count}
        
        # Check minimum coverage
        if urban_count < MIN_CANDIDATES_URBAN:
            errors.append(
                f"{region}: Only {urban_count} urban candidate(s), "
                f"minimum required: {MIN_CANDIDATES_URBAN}"
            )
        if rural_count < MIN_CANDIDATES_RURAL:
            errors.append(
                f"{region}: Only {rural_count} rural candidate(s), "
                f"minimum required: {MIN_CANDIDATES_RURAL}"
            )
    
    is_valid = len(errors) == 0
    
    return LocalityValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        region_coverage=region_coverage
    )


def check_city_selection_safety(
    region: str,
    residence_type: str,
    candidates: List[Dict],
    use_case: UseCaseProfile = UseCaseProfile.NARRATIVE_REQUIRED
) -> Tuple[bool, Optional[str]]:
    """
    Check if city selection will succeed and return appropriate action.
    
    Args:
        region: Target region
        residence_type: Urban or Rural
        candidates: List of candidate localities
        use_case: Use-case profile for decision strategy
        
    Returns:
        (is_safe, warning_message)
        - is_safe: True if selection should proceed
        - warning_message: Optional message explaining the issue
    """
    if not candidates:
        msg = f"No candidates for {region}/{residence_type} after filtering"
        
        if use_case == UseCaseProfile.HIGH_STAKES:
            # Fail fast in high-stakes mode
            logger.error(f"HIGH_STAKES: {msg}")
            raise LocalityIntegrityError(
                f"{msg}. This is a critical failure in HIGH_STAKES mode."
            )
        else:
            # Log warning for other modes
            logger.warning(msg)
            return False, msg
    
    if len(candidates) < 2 and residence_type == "Rural":
        # Rural should have variety
        msg = f"Only {len(candidates)} rural candidate(s) for {region}"
        logger.warning(msg)
        return True, msg  # Still safe but warned
    
    return True, None


def get_fallback_candidates(
    region: str,
    residence_type: str,
    all_localities: List[Dict],
    use_case: UseCaseProfile = UseCaseProfile.NARRATIVE_REQUIRED
) -> Tuple[List[Dict], str]:
    """
    Get fallback candidates when filtered set is empty.
    
    Strategy depends on use-case:
    - HIGH_STAKES: Never fallback, raise error
    - NARRATIVE_REQUIRED: Use any locality in region with warning
    - ANALYSIS_ONLY: Use any locality with provenance tracking
    
    Args:
        region: Target region
        residence_type: Urban or Rural (original target)
        all_localities: All localities in the region
        use_case: Use-case profile
        
    Returns:
        (candidates, provenance_note)
    """
    if use_case == UseCaseProfile.HIGH_STAKES:
        raise LocalityIntegrityError(
            f"HIGH_STAKES: Empty candidate set for {region}/{residence_type}. "
            f"No fallback allowed."
        )
    
    # For other modes, use all localities with appropriate note
    if use_case == UseCaseProfile.NARRATIVE_REQUIRED:
        logger.warning(
            f"Falling back to unfiltered candidates for {region}/{residence_type}"
        )
        return all_localities, f"FALLBACK_UNFILTERED:residence_mismatch_allowed"
    else:
        # ANALYSIS_ONLY
        return all_localities, "FALLBACK_UNFILTERED:analysis_mode"


def validate_and_log_config(config_path: Optional[str] = None) -> bool:
    """
    Validate locality config and log results at startup.
    
    Args:
        config_path: Path to config file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        result = validate_locality_config(config_path)
        
        if result.is_valid:
            logger.info("Locality configuration validation passed")
            logger.debug(f"Region coverage: {result.region_coverage}")
            return True
        else:
            logger.error("Locality configuration validation FAILED:")
            for error in result.errors:
                logger.error(f"  - {error}")
            return False
            
    except LocalityIntegrityError as e:
        logger.error(f"Locality config integrity error: {e}")
        return False


# Convenience function for generator use
def select_city_with_integrity_check(
    region: str,
    residence_type: str,
    localities_data: Dict,
    use_case: UseCaseProfile,
    random_gen
) -> Tuple[str, str]:
    """
    Select a city with full integrity checking.
    
    Args:
        region: Target region
        residence_type: Urban or Rural
        localities_data: Localities configuration
        use_case: Use-case profile
        random_gen: Random number generator (numpy.random.Generator)
        
    Returns:
        (selected_city, provenance_note)
        
    Raises:
        LocalityIntegrityError: If selection fails in high-stakes mode
    """
    import numpy as np
    
    region_localities = localities_data.get(region, [])
    
    if not region_localities:
        raise LocalityIntegrityError(f"No localities defined for region: {region}")
    
    # Filter by implied_residence matching residence_type
    filtered = [
        loc for loc in region_localities
        if loc.get("implied_residence") == residence_type
    ]
    
    # Check safety
    is_safe, warning = check_city_selection_safety(
        region, residence_type, filtered, use_case
    )
    
    if not is_safe:
        # Need fallback
        filtered, provenance_note = get_fallback_candidates(
            region, residence_type, region_localities, use_case
        )
    else:
        provenance_note = f"RESIDENCE_FILTERED:{residence_type}"
    
    # Apply tiered sampling weights if available
    weights = None
    if residence_type == "Urban":
        # Urban: 55% / 20% / 10% / 15% for population tiers 1-4
        tier_weights = {1: 0.55, 2: 0.20, 3: 0.10, 4: 0.15}
        weights = [
            tier_weights.get(loc.get("population_tier", 3), 0.10)
            for loc in filtered
        ]
    else:
        # Rural: 15% weight for tier 3 (communes), even for others
        weights = [
            0.15 if loc.get("population_tier") == 3 else 0.085
            for loc in filtered
        ]
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        weights = None  # Uniform if no weights
    
    # Select
    selected_idx = random_gen.choice(len(filtered), p=weights)
    selected = filtered[selected_idx]
    
    return selected["name"], provenance_note
