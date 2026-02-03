# Scientific and Statistical Robustness Assessment

## Executive Summary

This assessment evaluates the scientific rigor of the Moldova Personas Generator against three key principles:

1. **PxWeb Data First**: All distributions available from NBS PxWeb should be fetched directly
2. **PGM/IPF for Cross-tabs**: Unavailable cross-tabulations should be derived using probabilistic methods
3. **LLM for Narratives Only**: Large Language Models should only generate unstructured narrative content

## Current State Analysis

### ✅ What Works Well

| Component | Status | Notes |
|-----------|--------|-------|
| Multi-marginal Raking | ✅ Working | `_rake_resample()` adjusts key marginal distributions with diagnostics |
| Mock Mode Safety | ✅ Fixed | Returns empty strings, no fabricated content |
| Fake Villages Removed | ✅ Fixed | All geographic data now uses real cities |
| Template Cultural Backgrounds | ✅ Removed | No more Mad Libs-style generation |
| Narrative Content | ✅ Appropriate | LLM generates only hobbies, descriptions, profiles |

### ❌ Critical Issues

| Issue | Impact | Recommendation |
|-------|--------|----------------|
| **12 Distributions Hardcoded** | Data drift risk, manual updates | Fetch from PxWeb automatically |
| **Cross-tabs Estimated** | May not preserve true correlations | Use PGM/IPF from marginals |
| **No Runtime PxWeb Fetch** | Static data, no freshness | Implement live fetch with caching |
| **Occupation Lists Hardcoded** | May not match labor market | Derive from ISCO classifications |

## Detailed Distribution Audit

### Should Fetch from PxWeb (12 distributions)

| Distribution | PxWeb Dataset | Current Status | Priority |
|--------------|---------------|----------------|----------|
| REGION_DISTRIBUTION | POP010200rcl.px | Hardcoded | HIGH |
| RESIDENCE_TYPE_DISTRIBUTION | POP010200rcl.px | Hardcoded | HIGH |
| SEX_DISTRIBUTION | POP010200rcl.px | Hardcoded | HIGH |
| AGE_GROUP_DISTRIBUTION | POP010200rcl.px | Hardcoded (verified) | HIGH |
| ETHNICITY_DISTRIBUTION | POP010500.px | Hardcoded | HIGH |
| EDUCATION_DISTRIBUTION | EDU010100.px | Hardcoded | HIGH |
| EDUCATION_BY_AGE_GROUP | EDU010100.px | Hardcoded | HIGH |
| MARITAL_STATUS_DISTRIBUTION | POP030100.px | Hardcoded | HIGH |
| MARITAL_STATUS_BY_AGE | POP030100.px | Hardcoded | HIGH |
| RELIGION_DISTRIBUTION | POP010600.px | Hardcoded | HIGH |
| LANGUAGE_DISTRIBUTION | POP010550.px | Hardcoded | HIGH |
| EMPLOYMENT_STATUS_DISTRIBUTION | MUN110200.px | Hardcoded | HIGH |

### Should Use PGM/IPF (4 cross-tabulations)

| Cross-tab | Marginals Available | Current | Recommendation |
|-----------|---------------------|---------|----------------|
| REGION_URBAN_CROSS | Region totals, Urban/rural totals | Hardcoded estimate | IPF from marginals |
| ETHNICITY_BY_REGION | Ethnicity totals, Region totals | Hardcoded estimate | IPF from marginals |
| OCCUPATION_SECTOR | Occupation totals | Hardcoded (30/20/50) | Derive from ISCO |
| EMPLOYMENT_STATUS_BY_AGE | Status totals, Age totals | Hardcoded estimate | IPF or fetch |

### Appropriately Hardcoded (Estimated)

| Distribution | Reason | Status |
|--------------|--------|--------|
| Ethnicity-Religion cross-tab | Not in public NBS release | Properly documented as estimated |
| Employment status details | ILO methodology estimates | Properly documented as estimated |

## Architecture Recommendations

### 1. Data Layer Redesign

```python
class DataProvenance(Enum):
    PXWEB_DIRECT = "Fetched from NBS PxWeb API"      # Primary source
    PXWEB_CACHED = "Cached from previous PxWeb fetch" # Validated cache
    IPF_DERIVED = "Derived via IPF from marginals"    # Statistical inference
    CENSUS_HARDCODED = "From NBS 2024 final report"   # Verified static
    ESTIMATED = "Demographic estimate"                # Clearly marked

@dataclass
class Distribution:
    values: Dict[str, float]
    provenance: DataProvenance
    source_table: Optional[str]      # e.g., "POP010200rcl.px"
    last_fetched: Optional[datetime]
    confidence: float                 # 0.0-1.0
```

### 2. Fetching Strategy

```python
class CensusDataManager:
    """
    Manages all census distributions with automatic fetching.
    """
    
    def get_distribution(self, name: str) -> Distribution:
        """
        Get distribution with automatic fetching logic:
        1. Check cache (if < 30 days old, return cached)
        2. Try fetch from PxWeb
        3. If fetch fails, return cached (any age) with warning
        4. If no cache, use hardcoded fallback
        """
        pass
    
    def refresh_all(self) -> RefreshReport:
        """Force refresh all distributions from PxWeb."""
        pass
```

### 3. PGM/IPF Integration

```python
class CrossTabulationEngine:
    """
    Derives cross-tabulations from marginal distributions using IPF.
    """
    
    def derive_cross_tab(
        self,
        row_marginals: Distribution,      # e.g., ethnicity totals
        col_marginals: Distribution,      # e.g., region totals
        seed_matrix: Optional[np.ndarray] = None
    ) -> CrossTabulation:
        """
        Use IPF to find maximum entropy distribution consistent with marginals.
        """
        pass
```

## Implementation Roadmap

### Phase 1: High-Priority PxWeb Integration (Week 1)

1. **Expand NBSDataFetcher**
   - Add fetch methods for all 12 high-priority distributions
   - Implement proper JSON-stat parsing
   - Add caching with timestamps

2. **Create Distribution Registry**
   - Document all distributions with source tables
   - Add data provenance tracking
   - Implement freshness checking

### Phase 2: PGM/IPF Cross-tabulations (Week 2)

1. **Implement IPF Engine**
   - Generic IPF algorithm for 2-way tables
   - Support for seed matrices (prior knowledge)
   - Convergence checking

2. **Derive Cross-tabs**
   - REGION_URBAN_CROSS from region + urban/rural marginals
   - ETHNICITY_BY_REGION from ethnicity + region marginals
   - Document with confidence intervals

### Phase 3: Occupation System (Week 3)

1. **ISCO Integration**
   - Map ISCO-08 codes to Romanian occupations
   - Sample from PxWeb occupation distribution
   - Link to education requirements

2. **Remove Hardcoded Lists**
   - Replace `high_skill_jobs`, `medium_skill_jobs`, etc.
   - Use ISCO classification with education correlation

### Phase 4: Validation & Testing (Week 4)

1. **Statistical Validation**
   - Chi-square tests against NBS marginals
   - Confidence interval reporting
   - Data freshness monitoring

2. **Documentation**
   - Data dictionary with provenance
   - API reference for distributions
   - Scientific methodology paper

## Data Provenance Standards

Every distribution must have:

| Attribute | Required | Description |
|-----------|----------|-------------|
| `source_type` | ✅ | PXWEB, IPF, HARDCODED, ESTIMATED |
| `source_table` | If PXWEB | PxWeb dataset code (e.g., POP010200rcl.px) |
| `last_verified` | If PXWEB | Timestamp of last successful fetch |
| `methodology` | If IPF | Description of derivation method |
| `confidence` | ✅ | 0.0-1.0 reliability score |
| `limitations` | If ESTIMATED | Known limitations and assumptions |

## Example: Refactored Distribution

```python
# BEFORE (hardcoded, no provenance)
REGION_DISTRIBUTION = {
    "Chisinau": 0.299,
    "Centru": 0.278,
    # ...
}

# AFTER (with full provenance)
REGION_DISTRIBUTION = Distribution(
    values={
        "Chisinau": 0.299,
        "Centru": 0.278,
        # ...
    },
    provenance=DataProvenance.PXWEB_CACHED,
    source_table="POP010200rcl.px",
    last_fetched=datetime(2026, 1, 29, 12, 0, 0),
    confidence=0.99
)
```

## Conclusion

The current system has **significant scientific robustness gaps**:

- **12 distributions** should be fetched from PxWeb but are hardcoded
- **4 cross-tabulations** should use PGM/IPF but are estimated
- **No runtime fetching** means data can become stale

**Recommendation**: Implement the 4-phase roadmap to achieve full scientific rigor. This will:
1. Ensure data freshness via automatic PxWeb fetching
2. Improve cross-tabulation accuracy via IPF
3. Provide full provenance tracking for reproducibility
4. Eliminate all hardcoded estimates (except those properly documented)

## Appendix: PxWeb Dataset Reference

| Dataset Code | Description | Priority |
|--------------|-------------|----------|
| POP010200rcl.px | Population by region, sex, age | HIGH |
| POP010500.px | Population by ethnicity | HIGH |
| POP010550.px | Mother tongue | HIGH |
| POP010600.px | Religion | HIGH |
| POP030100.px | Marital status by age/sex | HIGH |
| EDU010100.px | Education by age/sex | HIGH |
| MUN110200.px | Workforce statistics | HIGH |
| MUN120100.px | Employment by status | HIGH |
| MUN120300.px | Occupations (ISCO) | MEDIUM |
| GEN010790mun.px | Gender employment by municipality | MEDIUM |
