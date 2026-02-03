"""
Census data distributions from Moldova 2024 Census (BNS).

This module provides distributions for persona generation.
Data sources are prioritized as follows:
1. PxWeb API (freshest data from statbank.statistica.md)
2. Verified hardcoded data from NBS 2024 final reports
3. IPF-derived cross-tabulations from marginals

All distributions include provenance tracking for scientific rigor.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field

# Import from new fetcher module
from .pxweb_fetcher import (
    DataProvenance,
    Distribution,
    NBSDataManager,
    NBS_MANAGER,
)
from .models import AgeConstraints, PopulationMode
from .paths import data_path


def _age_to_group(age: int) -> str:
    """Map a single age to the standard age group labels."""
    if age < 15:
        return "0-14"
    if age < 25:
        return "15-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    if age < 65:
        return "55-64"
    return "65+"


@dataclass
class CensusDistributions:
    """
    Container for all census distributions used in persona generation.
    
    This class lazily loads distributions from PxWeb when available,
    falling back to verified hardcoded data from NBS 2024 reports.
    
    Usage:
        census = CensusDistributions()
        region_dist = census.REGION_DISTRIBUTION
        # Returns Distribution object with full provenance
    """
    
    # Data manager for fetching
    _manager: NBSDataManager = field(default_factory=lambda: NBS_MANAGER)
    
    # Cache for loaded distributions
    _cache: Dict[str, Distribution] = field(default_factory=dict)
    _cross_cache: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    _cross_provenance: Dict[str, DataProvenance] = field(default_factory=dict)
    _adult_age_exact: Optional[bool] = None
    
    def _get_dist(self, name: str) -> Distribution:
        """Get distribution from cache or fetch."""
        if name not in self._cache:
            if name == "district":
                self._cache[name] = self._manager.get_district_distribution()
            else:
                self._cache[name] = self._manager.get_distribution(name)
        return self._cache[name]

    def _adult_age_distribution_obj(self) -> Distribution:
        """Return adult-only age distribution as a Distribution object."""
        base = self._get_dist("age_group")
        is_exact = bool(self._adult_age_exact)
        provenance = base.provenance if is_exact else DataProvenance.ESTIMATED
        confidence = base.confidence if is_exact else min(base.confidence, 0.7)
        limitations = base.limitations
        if not is_exact:
            limitations = "Adult-only age distribution approximated from 15-24 group scaling"
        return Distribution(
            values=self.ADULT_AGE_GROUP_DISTRIBUTION,
            provenance=provenance,
            source_table=base.source_table,
            last_fetched=base.last_fetched,
            confidence=confidence,
            methodology="Adult-only derived from age_group distribution",
            limitations=limitations,
        )

    def _derive_adult_marginal(
        self,
        conditional_by_age: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Derive an adult-only marginal distribution from age-conditional data.

        Uses the adult age-group distribution as weights.
        """
        age_weights = self.ADULT_AGE_GROUP_DISTRIBUTION
        totals: Dict[str, float] = {}
        for age_group, weight in age_weights.items():
            if weight is None:
                continue
            conditional = conditional_by_age.get(age_group)
            if not conditional:
                continue
            for category, prob in conditional.items():
                totals[category] = totals.get(category, 0.0) + float(weight) * float(prob)
        total = sum(totals.values())
        if total <= 0:
            return totals
        return {k: v / total for k, v in totals.items()}
    def _age_group_distribution_15_plus(self) -> Dict[str, float]:
        """Return 15+ age group distribution by dropping 0-14 and renormalizing."""
        base = dict(self.AGE_GROUP_DISTRIBUTION)
        base.pop("0-14", None)
        total = sum(base.values())
        return {k: v / total for k, v in base.items()} if total > 0 else base

    def _age_group_distribution_obj(self, population_mode: PopulationMode) -> Distribution:
        """Return age group Distribution object for a given population mode."""
        if population_mode == PopulationMode.ADULT_18:
            return self._adult_age_distribution_obj()
        base = self._get_dist("age_group")
        if population_mode == PopulationMode.AGE_15_PLUS:
            return Distribution(
                values=self._age_group_distribution_15_plus(),
                provenance=base.provenance,
                source_table=base.source_table,
                last_fetched=base.last_fetched,
                confidence=base.confidence,
                methodology="15+ derived by dropping 0-14 and renormalizing",
                limitations=base.limitations,
            )
        return base

    def _derive_conditional_by_age(
        self,
        target_dist: Distribution,
        cache_key: str,
        fallback: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """Derive conditional distributions by age using IPF from marginals."""
        if cache_key in self._cross_cache:
            if cache_key not in self._cross_provenance:
                self._cross_provenance[cache_key] = DataProvenance.ESTIMATED
            return self._cross_cache[cache_key]
        try:
            age_dist = self._adult_age_distribution_obj()
            cross_tab = self._manager.get_cross_tabulation_ipf(
                age_dist, target_dist, cache_key
            )
            result: Dict[str, Dict[str, float]] = {}
            for i, age_group in enumerate(cross_tab.row_names):
                row_total = float(cross_tab.matrix[i, :].sum())
                if row_total <= 0:
                    continue
                result[age_group] = {
                    cross_tab.col_names[j]: float(cross_tab.matrix[i, j]) / row_total
                    for j in range(len(cross_tab.col_names))
                }
            # Ensure any missing age groups fall back to defaults
            for age_group, dist in fallback.items():
                result.setdefault(age_group, dist)
            if result:
                self._cross_cache[cache_key] = result
                self._cross_provenance[cache_key] = DataProvenance.IPF_DERIVED
                return result
        except Exception:
            pass
        self._cross_cache[cache_key] = fallback
        self._cross_provenance[cache_key] = DataProvenance.ESTIMATED
        return fallback

    def _derive_marginal_from_by_age(
        self,
        age_dist: Dict[str, float],
        conditional_by_age: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """Derive a marginal distribution from age-group conditionals."""
        totals: Dict[str, float] = {}
        for age_group, age_prob in age_dist.items():
            conditional = conditional_by_age.get(age_group)
            if not conditional:
                continue
            for category, prob in conditional.items():
                totals[category] = totals.get(category, 0.0) + age_prob * prob
        total = sum(totals.values())
        return {k: v / total for k, v in totals.items()} if total > 0 else totals

    def age_group_distribution(self, population_mode: PopulationMode) -> Dict[str, float]:
        """Return age group distribution for the requested population mode."""
        if population_mode == PopulationMode.ADULT_18:
            return self.ADULT_AGE_GROUP_DISTRIBUTION
        if population_mode == PopulationMode.AGE_15_PLUS:
            return self._age_group_distribution_15_plus()
        return self.AGE_GROUP_DISTRIBUTION

    def education_distribution(self, population_mode: PopulationMode) -> Distribution:
        """Return education distribution aligned to the population mode."""
        if population_mode == PopulationMode.ALL:
            return self._get_dist("education")
        age_dist = self.age_group_distribution(population_mode)
        values = self._derive_marginal_from_by_age(age_dist, self.EDUCATION_BY_AGE_GROUP)
        base = self._get_dist("education")
        is_estimated = (population_mode == PopulationMode.ADULT_18 and not self._adult_age_exact)
        provenance = DataProvenance.ESTIMATED if is_estimated else DataProvenance.IPF_DERIVED
        limitations = None
        if is_estimated:
            limitations = "Adult-only education derived from approximate 18+ age distribution"
        return Distribution(
            values=values,
            provenance=provenance,
            source_table=base.source_table,
            last_fetched=base.last_fetched,
            confidence=0.85 if not is_estimated else 0.65,
            methodology="Derived from education-by-age and age-group distribution",
            limitations=limitations,
        )

    def marital_status_distribution(self, population_mode: PopulationMode) -> Distribution:
        """Return marital status distribution aligned to the population mode."""
        if population_mode == PopulationMode.ALL:
            return self._get_dist("marital_status")
        age_dist = self.age_group_distribution(population_mode)
        values = self._derive_marginal_from_by_age(age_dist, self.MARITAL_STATUS_BY_AGE)
        base = self._get_dist("marital_status")
        is_estimated = (population_mode == PopulationMode.ADULT_18 and not self._adult_age_exact)
        provenance = DataProvenance.ESTIMATED if is_estimated else DataProvenance.IPF_DERIVED
        limitations = None
        if is_estimated:
            limitations = "Adult-only marital status derived from approximate 18+ age distribution"
        return Distribution(
            values=values,
            provenance=provenance,
            source_table=base.source_table,
            last_fetched=base.last_fetched,
            confidence=0.85 if not is_estimated else 0.65,
            methodology="Derived from marital-status-by-age and age-group distribution",
            limitations=limitations,
        )

    def employment_status_distribution(self, population_mode: PopulationMode) -> Distribution:
        """Return employment status distribution aligned to the population mode."""
        if population_mode == PopulationMode.ALL:
            return self._get_dist("employment_status")
        age_dist = self.age_group_distribution(population_mode)
        values = self._derive_marginal_from_by_age(age_dist, self.EMPLOYMENT_STATUS_BY_AGE)
        base = self._get_dist("employment_status")
        is_estimated = (population_mode == PopulationMode.ADULT_18 and not self._adult_age_exact)
        provenance = DataProvenance.ESTIMATED if is_estimated else DataProvenance.IPF_DERIVED
        limitations = None
        if is_estimated:
            limitations = "Adult-only employment status derived from approximate 18+ age distribution"
        return Distribution(
            values=values,
            provenance=provenance,
            source_table=base.source_table,
            last_fetched=base.last_fetched,
            confidence=0.85 if not is_estimated else 0.65,
            methodology="Derived from employment-status-by-age and age-group distribution",
            limitations=limitations,
        )
    
    # =========================================================================
    # REGION DISTRIBUTION
    # Source: BNS PxWeb POP010200rcl.px (or verified hardcoded fallback)
    # =========================================================================
    @property
    def REGION_DISTRIBUTION(self) -> Dict[str, float]:
        """Region distribution from NBS 2024 Census."""
        return self._get_dist("region").values
    
    # =========================================================================
    # URBAN/RURAL DISTRIBUTION
    # Source: BNS PxWeb POP010200rcl.px
    # =========================================================================
    @property
    def RESIDENCE_TYPE_DISTRIBUTION(self) -> Dict[str, float]:
        """Urban/rural distribution from NBS 2024 Census."""
        return self._get_dist("residence_type").values
    
    # =========================================================================
    # REGION × URBAN/RURAL CROSS-TABULATION
    # Note: This is derived via IPF from marginals (not directly in PxWeb)
    # =========================================================================
    @property
    def REGION_URBAN_CROSS(self) -> Dict[str, Dict[str, float]]:
        """
        Region by urban/rural cross-tabulation.
        
        Derived via IPF from region and residence-type marginals.
        This avoids injecting unverified regional urbanization rates.
        """
        cross_tab = self._manager.get_region_urban_cross()
        result: Dict[str, Dict[str, float]] = {}
        for i, region in enumerate(cross_tab.row_names):
            row = {}
            for j, residence in enumerate(cross_tab.col_names):
                row[residence] = float(cross_tab.matrix[i, j])
            result[region] = row
        return result
    
    # =========================================================================
    # SEX DISTRIBUTION
    # Source: BNS PxWeb POP010200rcl.px
    # =========================================================================
    @property
    def SEX_DISTRIBUTION(self) -> Dict[str, float]:
        """Sex distribution from NBS 2024 Census."""
        return self._get_dist("sex").values
    
    # =========================================================================
    # AGE GROUP DISTRIBUTION
    # Source: BNS PxWeb POP010200rcl.px
    # =========================================================================
    @property
    def AGE_GROUP_DISTRIBUTION(self) -> Dict[str, float]:
        """
        Age group distribution from NBS 2024 Census.
        
        Source: POP010200rcl.px (verified 2026-01-29)
        Total population: 2,423,287 as of Jan 1, 2024
        
        NOTE: This is the FULL population including children.
        The generator filters to 18+ only, which renormalizes automatically.
        """
        return self._get_dist("age_group").values

    def _load_single_age_counts(self) -> Optional[Dict[int, int]]:
        """
        Load single-age counts if available locally.

        Uses nbs_population_by_age_2024.json when present to enable
        adult-only age group distributions without approximation.
        """
        age_path = data_path("nbs_population_by_age_2024.json")
        if not age_path.exists():
            return None
        try:
            import json
            with open(age_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            by_single_age = payload.get("by_single_age", [])
        except Exception:
            return None

        counts: Dict[int, int] = {}
        for row in by_single_age:
            try:
                age = int(row.get("age"))
                count = int(row.get("count", 0))
            except (TypeError, ValueError):
                continue
            counts[age] = counts.get(age, 0) + count
        return counts or None

    @property
    def ADULT_AGE_GROUP_DISTRIBUTION(self) -> Dict[str, float]:
        """
        Adult-only (18+) age group distribution.

        If single-age counts are available, derive exact 18+ weights.
        Otherwise, approximate by scaling the 15-24 group by 7/10.
        """
        single_age_counts = self._load_single_age_counts()
        if single_age_counts:
            self._adult_age_exact = True
            adult_counts: Dict[str, int] = {
                "15-24": 0,
                "25-34": 0,
                "35-44": 0,
                "45-54": 0,
                "55-64": 0,
                "65+": 0,
            }
            for age, count in single_age_counts.items():
                if age < AgeConstraints.MIN_PERSONA_AGE:
                    continue
                group = _age_to_group(age)
                if group == "0-14":
                    continue
                adult_counts[group] = adult_counts.get(group, 0) + count

            total = sum(adult_counts.values())
            if total > 0:
                return {k: v / total for k, v in adult_counts.items()}

        # Fallback approximation using existing age-group distribution
        self._adult_age_exact = False
        base = dict(self.AGE_GROUP_DISTRIBUTION)
        base.pop("0-14", None)
        if "15-24" in base:
            base["15-24"] = base["15-24"] * 0.7  # 18-24 is 7/10 of 15-24
        total = sum(base.values())
        return {k: v / total for k, v in base.items()} if total > 0 else base
    
    # =========================================================================
    # ETHNICITY DISTRIBUTION
    # Source: BNS PxWeb POP010500.px
    # =========================================================================
    @property
    def ETHNICITY_DISTRIBUTION(self) -> Dict[str, float]:
        """Ethnicity distribution from NBS 2024 Census."""
        return self._get_dist("ethnicity").values
    
    # =========================================================================
    # ETHNICITY × REGION CROSS-TABULATION
    # Note: This should ideally be derived via IPF, but PxWeb may have it
    # Currently using verified hardcoded values
    # =========================================================================
    @property
    def ETHNICITY_BY_REGION(self) -> Dict[str, Dict[str, float]]:
        """
        Ethnicity by region cross-tabulation.
        
        Derived via IPF from national ethnicity and region marginals.
        This avoids injecting unverified regional ethnicity shares.
        """
        cross_tab = self._manager.get_ethnicity_by_region()
        result: Dict[str, Dict[str, float]] = {}

        # Convert joint distribution to P(ethnicity | region)
        for col_idx, region in enumerate(cross_tab.col_names):
            col_total = float(cross_tab.matrix[:, col_idx].sum())
            if col_total <= 0:
                continue
            conditional = {}
            for row_idx, ethnicity in enumerate(cross_tab.row_names):
                conditional[ethnicity] = float(cross_tab.matrix[row_idx, col_idx]) / col_total
            result[region] = conditional

        return result
    
    # =========================================================================
    # EDUCATION LEVEL DISTRIBUTION
    # Source: BNS PxWeb EDU010100.px
    # =========================================================================
    @property
    def EDUCATION_DISTRIBUTION(self) -> Dict[str, float]:
        """Education level distribution from NBS 2024 Census (population 10+)."""
        return self._get_dist("education").values
    
    # =========================================================================
    # EDUCATION × AGE GROUP CROSS-TABULATION
    # Source: BNS PxWeb EDU010100.px
    # =========================================================================
    @property
    def EDUCATION_BY_AGE_GROUP(self) -> Dict[str, Dict[str, float]]:
        """Education distribution by age group."""
        fallback = {
            "0-14": {"Fără studii": 0.95, "Primar": 0.05, "Gimnazial": 0.00, "Liceal": 0.00, 
                     "Profesional/Tehnic": 0.00, "Superior (Licență/Master)": 0.00, "Doctorat": 0.00},
            "15-24": {"Fără studii": 0.01, "Primar": 0.02, "Gimnazial": 0.25, "Liceal": 0.35,
                      "Profesional/Tehnic": 0.20, "Superior (Licență/Master)": 0.15, "Doctorat": 0.02},
            "25-34": {"Fără studii": 0.01, "Primar": 0.02, "Gimnazial": 0.10, "Liceal": 0.15,
                      "Profesional/Tehnic": 0.28, "Superior (Licență/Master)": 0.42, "Doctorat": 0.02},
            "35-44": {"Fără studii": 0.01, "Primar": 0.04, "Gimnazial": 0.15, "Liceal": 0.18,
                      "Profesional/Tehnic": 0.35, "Superior (Licență/Master)": 0.25, "Doctorat": 0.02},
            "45-54": {"Fără studii": 0.02, "Primar": 0.08, "Gimnazial": 0.22, "Liceal": 0.20,
                      "Profesional/Tehnic": 0.32, "Superior (Licență/Master)": 0.14, "Doctorat": 0.02},
            "55-64": {"Fără studii": 0.03, "Primar": 0.12, "Gimnazial": 0.28, "Liceal": 0.18,
                      "Profesional/Tehnic": 0.25, "Superior (Licență/Master)": 0.12, "Doctorat": 0.02},
            "65+": {"Fără studii": 0.05, "Primar": 0.18, "Gimnazial": 0.30, "Liceal": 0.15,
                    "Profesional/Tehnic": 0.20, "Superior (Licență/Master)": 0.10, "Doctorat": 0.02},
        }
        target = self._get_dist("education")
        return self._derive_conditional_by_age(target, "education_by_age_group", fallback)

    @property
    def ADULT_EDUCATION_DISTRIBUTION(self) -> Dict[str, float]:
        """Adult-only education distribution derived from age-conditional data."""
        return self._derive_adult_marginal(self.EDUCATION_BY_AGE_GROUP)
    
    # =========================================================================
    # MARITAL STATUS DISTRIBUTION
    # Source: BNS PxWeb POP030100.px
    # =========================================================================
    @property
    def MARITAL_STATUS_DISTRIBUTION(self) -> Dict[str, float]:
        """Marital status distribution from NBS 2024 Census (population 15+)."""
        return self._get_dist("marital_status").values
    
    # =========================================================================
    # MARITAL STATUS × AGE CROSS-TABULATION
    # Source: BNS PxWeb POP030100.px
    # =========================================================================
    @property
    def MARITAL_STATUS_BY_AGE(self) -> Dict[str, Dict[str, float]]:
        """Marital status by age group."""
        fallback = {
            "0-14": {"Necăsătorit": 1.00, "Căsătorit": 0.00, "Divorțat": 0.00, "Văduv": 0.00, "Separat": 0.00},
            "15-24": {"Necăsătorit": 0.92, "Căsătorit": 0.07, "Divorțat": 0.01, "Văduv": 0.00, "Separat": 0.00},
            "25-34": {"Necăsătorit": 0.45, "Căsătorit": 0.50, "Divorțat": 0.03, "Văduv": 0.00, "Separat": 0.02},
            "35-44": {"Necăsătorit": 0.15, "Căsătorit": 0.75, "Divorțat": 0.06, "Văduv": 0.01, "Separat": 0.03},
            "45-54": {"Necăsătorit": 0.10, "Căsătorit": 0.72, "Divorțat": 0.10, "Văduv": 0.03, "Separat": 0.05},
            "55-64": {"Necăsătorit": 0.08, "Căsătorit": 0.68, "Divorțat": 0.10, "Văduv": 0.10, "Separat": 0.04},
            "65+": {"Necăsătorit": 0.05, "Căsătorit": 0.45, "Divorțat": 0.08, "Văduv": 0.40, "Separat": 0.02},
        }
        target = self._get_dist("marital_status")
        return self._derive_conditional_by_age(target, "marital_status_by_age", fallback)

    @property
    def ADULT_MARITAL_STATUS_DISTRIBUTION(self) -> Dict[str, float]:
        """Adult-only marital status distribution derived from age-conditional data."""
        return self._derive_adult_marginal(self.MARITAL_STATUS_BY_AGE)
    
    # =========================================================================
    # RELIGION DISTRIBUTION
    # Source: BNS PxWeb POP010600.px
    # =========================================================================
    @property
    def RELIGION_DISTRIBUTION(self) -> Dict[str, float]:
        """Religion distribution from NBS 2024 Census."""
        return self._get_dist("religion").values
    
    # =========================================================================
    # MOTHER TONGUE DISTRIBUTION
    # Source: BNS PxWeb POP010550.px
    # =========================================================================
    @property
    def LANGUAGE_DISTRIBUTION(self) -> Dict[str, float]:
        """Mother tongue distribution from NBS 2024 Census."""
        return self._get_dist("language").values
    
    # =========================================================================
    # OCCUPATION SECTOR
    # Derived from economic activity data (NBS Labor Force Survey)
    # =========================================================================
    @property
    def OCCUPATION_SECTOR(self) -> Dict[str, float]:
        """
        Occupation sector distribution.
        
        Derived from NBS economic activity data:
        - Agriculture ~30% (higher in rural areas)
        - Industry ~20%
        - Services ~50%
        """
        return {
            "Agricultură": 0.30,
            "Industrie": 0.20,
            "Servicii": 0.50,
        }
    
    # =========================================================================
    # EMPLOYMENT STATUS DISTRIBUTION
    # Source: BNS PxWeb MUN110200.px (Labor Force Survey)
    # =========================================================================
    @property
    def EMPLOYMENT_STATUS_DISTRIBUTION(self) -> Dict[str, float]:
        """
        Employment status distribution from NBS Labor Force Survey.
        
        Activity rate: 44.5%, Employment rate: 42.7%, Unemployment: 4.0%
        """
        return self._get_dist("employment_status").values
    
    # =========================================================================
    # EMPLOYMENT STATUS × AGE CROSS-TABULATION
    # Source: NBS Labor Force Survey (MUN110200.px with age breakdowns)
    # =========================================================================
    @property
    def EMPLOYMENT_STATUS_BY_AGE(self) -> Dict[str, Dict[str, float]]:
        """Employment status by age group."""
        fallback = {
            "0-14": {"employed": 0.00, "unemployed": 0.00, "student": 0.95, 
                     "retired": 0.00, "homemaker": 0.00, "other_inactive": 0.05},
            "15-24": {"employed": 0.20, "unemployed": 0.08, "student": 0.60,
                      "retired": 0.00, "homemaker": 0.02, "other_inactive": 0.10},
            "25-34": {"employed": 0.55, "unemployed": 0.03, "student": 0.10,
                      "retired": 0.00, "homemaker": 0.15, "other_inactive": 0.17},
            "35-44": {"employed": 0.65, "unemployed": 0.02, "student": 0.02,
                      "retired": 0.00, "homemaker": 0.12, "other_inactive": 0.19},
            "45-54": {"employed": 0.60, "unemployed": 0.02, "student": 0.01,
                      "retired": 0.05, "homemaker": 0.08, "other_inactive": 0.24},
            "55-64": {"employed": 0.40, "unemployed": 0.02, "student": 0.01,
                      "retired": 0.35, "homemaker": 0.05, "other_inactive": 0.17},
            "65+": {"employed": 0.08, "unemployed": 0.01, "student": 0.01,
                    "retired": 0.75, "homemaker": 0.03, "other_inactive": 0.12},
        }
        target = self._get_dist("employment_status")
        return self._derive_conditional_by_age(target, "employment_status_by_age", fallback)

    @property
    def ADULT_EMPLOYMENT_STATUS_DISTRIBUTION(self) -> Dict[str, float]:
        """Adult-only employment status distribution derived from age-conditional data."""
        return self._derive_adult_marginal(self.EMPLOYMENT_STATUS_BY_AGE)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_distribution_info(
        self,
        name: str,
        population_mode: PopulationMode = PopulationMode.ADULT_18,
    ) -> Optional[Distribution]:
        """
        Get full Distribution object with provenance info.
        
        Args:
            name: Distribution name (e.g., "region", "ethnicity")
            
        Returns:
            Distribution object or None if not found
        """
        try:
            if name == "age_group":
                return self._age_group_distribution_obj(population_mode)
            if name == "education":
                return self.education_distribution(population_mode)
            if name == "marital_status":
                return self.marital_status_distribution(population_mode)
            if name == "employment_status":
                return self.employment_status_distribution(population_mode)
            return self._get_dist(name)
        except ValueError:
            return None
    
    def get_all_provenance(
        self,
        population_mode: PopulationMode = PopulationMode.ADULT_18,
    ) -> Dict[str, Dict[str, any]]:
        """Get provenance information for all distributions."""
        info = {}
        for name in [
            "region", "residence_type", "sex", "age_group",
            "ethnicity", "education", "marital_status",
            "religion", "language", "employment_status", "district"
        ]:
            dist = self.get_distribution_info(name, population_mode)
            if dist:
                info[name] = {
                    "provenance": dist.provenance.value,
                    "source_table": dist.source_table,
                    "confidence": dist.confidence,
                    "last_fetched": dist.last_fetched.isoformat() if dist.last_fetched else None,
                }
        # Estimated/heuristic sources (not in PxWeb/census tables)
        # Force computation of conditional-by-age tables to capture IPF vs fallback provenance.
        _ = self.EDUCATION_BY_AGE_GROUP
        _ = self.MARITAL_STATUS_BY_AGE
        _ = self.EMPLOYMENT_STATUS_BY_AGE
        edu_prov = self._cross_provenance.get("education_by_age_group", DataProvenance.ESTIMATED)
        marital_prov = self._cross_provenance.get("marital_status_by_age", DataProvenance.ESTIMATED)
        emp_prov = self._cross_provenance.get("employment_status_by_age", DataProvenance.ESTIMATED)

        info.update({
            "education_by_age_group": {
                "provenance": edu_prov.value,
                "source_table": "IPF(age_group x education)",
                "confidence": 0.85,
                "last_fetched": None,
            },
            "marital_status_by_age": {
                "provenance": marital_prov.value,
                "source_table": "IPF(age_group x marital_status)",
                "confidence": 0.85,
                "last_fetched": None,
            },
            "employment_status_by_age": {
                "provenance": emp_prov.value,
                "source_table": "IPF(age_group x employment_status)",
                "confidence": 0.85,
                "last_fetched": None,
            },
            "occupation_sector": {
                "provenance": DataProvenance.ESTIMATED.value,
                "source_table": "heuristic: OCCUPATION_SECTOR",
                "confidence": 0.5,
                "last_fetched": None,
            },
            "occupation": {
                "provenance": DataProvenance.ESTIMATED.value,
                "source_table": "heuristic: OCCUPATION_LISTS",
                "confidence": 0.5,
                "last_fetched": None,
            },
        })

        # Override adult-only target provenance for reporting/gating.
        derived_map = {
            "education": ("education_by_age_group", edu_prov),
            "marital_status": ("marital_status_by_age", marital_prov),
            "employment_status": ("employment_status_by_age", emp_prov),
        }
        for field_name, (source_key, prov) in derived_map.items():
            if field_name not in info:
                continue
            info[field_name]["provenance"] = prov.value
            info[field_name]["source_table"] = f"Adult marginal derived from {source_key}"
            if prov == DataProvenance.ESTIMATED:
                info[field_name]["confidence"] = min(info[field_name].get("confidence", 0.0), 0.5)
            else:
                info[field_name]["confidence"] = min(info[field_name].get("confidence", 0.85), 0.85)
            info[field_name]["last_fetched"] = None
        return info

    def get_pxweb_snapshot_timestamp(
        self,
        population_mode: PopulationMode = PopulationMode.ADULT_18,
    ) -> Optional[str]:
        """Return the oldest PxWeb fetch timestamp across available distributions."""
        timestamps = []
        for name in [
            "region", "residence_type", "sex", "age_group",
            "ethnicity", "education", "marital_status",
            "religion", "language", "employment_status", "district"
        ]:
            dist = self.get_distribution_info(name, population_mode)
            if not dist or not dist.last_fetched:
                continue
            if dist.provenance in {DataProvenance.PXWEB_DIRECT, DataProvenance.PXWEB_CACHED}:
                timestamps.append(dist.last_fetched)
        if not timestamps:
            return None
        return min(timestamps).isoformat()
    
    def refresh_from_pxweb(self) -> Dict[str, bool]:
        """
        Force refresh all distributions from PxWeb.
        
        Returns:
            Dictionary mapping distribution names to success status
        """
        return self._manager.refresh_cache()


# Singleton instance for easy access
CENSUS = CensusDistributions()


# Backwards compatibility: direct access to hardcoded values
# These are used when PxWeb is unavailable
HARDCODED_FALLBACKS = {
    "REGION_DISTRIBUTION": {
        "Chisinau": 0.299,
        "Centru": 0.278,
        "Nord": 0.253,
        "Sud": 0.127,
        "Gagauzia": 0.043,
    },
    "RESIDENCE_TYPE_DISTRIBUTION": {"Urban": 0.464, "Rural": 0.536},
    "SEX_DISTRIBUTION": {"Feminin": 0.528, "Masculin": 0.472},
    "AGE_GROUP_DISTRIBUTION": {
        "0-14": 0.166,
        "15-24": 0.098,
        "25-34": 0.124,
        "35-44": 0.154,
        "45-54": 0.131,
        "55-64": 0.144,
        "65+": 0.183,
    },
    "ETHNICITY_DISTRIBUTION": {
        "Moldovean": 0.772,
        "Român": 0.079,
        "Ucrainean": 0.050,
        "Găgăuz": 0.043,
        "Rus": 0.030,
        "Bulgar": 0.016,
        "Rrom": 0.004,
        "Altele": 0.006,
    },
    "EDUCATION_DISTRIBUTION": {
        "Fără studii": 0.024,
        "Primar": 0.092,
        "Gimnazial": 0.227,
        "Liceal": 0.128,
        "Profesional/Tehnic": 0.336,
        "Superior (Licență/Master)": 0.181,
        "Doctorat": 0.012,
    },
    "MARITAL_STATUS_DISTRIBUTION": {
        "Necăsătorit": 0.236,
        "Căsătorit": 0.558,
        "Divorțat": 0.085,
        "Văduv": 0.100,
        "Separat": 0.021,
    },
    "RELIGION_DISTRIBUTION": {
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
    "LANGUAGE_DISTRIBUTION": {
        "Română": 0.800,
        "Rusă": 0.110,
        "Găgăuză": 0.038,
        "Ucraineană": 0.029,
        "Bulgară": 0.015,
        "Alta": 0.008,
    },
}
