# NBS 2024 Implementation Summary

**Date:** 2026-01-29  
**Status:** ✅ COMPLETE

---

## Overview

Successfully verified and corrected the Persona Generator Model (PGM) against real NBS 2024 census data from Moldova's National Bureau of Statistics.

## Data Sources Verified

| Source | Method | Data Retrieved |
|--------|--------|----------------|
| NBS PxWeb API | REST API calls | Population by age (2.4M total), Workforce stats |
| NBS Ethnicity Report | Excel file (Date_Comunicat_Etnoculturale_20_10_25.xlsx) | Full 2024 ethnicity distribution |

---

## Changes Made

### 1. Updated Age Distribution (census_data.py)

**Before (incorrect):**
```python
AGE_GROUP_DISTRIBUTION = {
    "0-14": 0.192,   # Wrong
    "15-24": 0.131,  # Wrong
    "25-34": 0.144,  # Wrong
    "35-44": 0.138,  # Wrong
    "45-54": 0.131,  # OK
    "55-64": 0.109,  # Wrong
    "65+": 0.155,    # Wrong
}
```

**After (NBS 2024 verified):**
```python
AGE_GROUP_DISTRIBUTION = {
    "0-14": 0.166,   # 16.6% (401,522 people)
    "15-24": 0.098,  # 9.8% (237,500 people)
    "25-34": 0.124,  # 12.4% (299,760 people)
    "35-44": 0.154,  # 15.4% (373,682 people)
    "45-54": 0.131,  # 13.1% (318,021 people) - unchanged
    "55-64": 0.144,  # 14.4% (349,370 people)
    "65+": 0.183,    # 18.3% (443,432 people)
}
```

**Source:** NBS PxWeb API - POP010200rcl.px  
**Total Population:** 2,423,287 (as of Jan 1, 2024)

---

### 2. Added Employment Status Field

**New field in Persona model (models.py):**
```python
employment_status: str = Field(
    default="employed",
    description="Employment status (employed, unemployed, student, retired, homemaker, other_inactive)"
)
```

**New distributions in census_data.py:**
```python
EMPLOYMENT_STATUS_DISTRIBUTION = {
    "employed": 0.427,      # 42.7% employment rate
    "unemployed": 0.018,    # ~1.8% of total
    "student": 0.080,       # Students not in labor force
    "retired": 0.183,       # 65+ population
    "homemaker": 0.060,     # Homemakers/caregivers
    "other_inactive": 0.232,# Other inactive
}

EMPLOYMENT_STATUS_BY_AGE = {
    "0-14": {...},
    "15-24": {"employed": 0.20, "student": 0.60, ...},
    "25-34": {"employed": 0.55, "homemaker": 0.15, ...},
    # ... etc
}
```

**Generation method added (generator.py):**
```python
def _generate_employment_status(self, age: int, occupation: str) -> str:
    """Generate employment status based on age and occupation hints."""
```

---

### 3. Ethnicity Verification

**Result:** PGM targets were already accurate!

| Ethnicity | PGM Target | NBS 2024 | Error |
|-----------|-----------|----------|-------|
| Moldovean | 77.2% | 76.73% | +0.47% ✅ |
| Român | 7.9% | 8.02% | -0.12% ✅ |
| Ucrainean | 5.0% | 5.13% | -0.13% ✅ |
| Rus | 3.0% | 3.39% | -0.39% ✅ |
| Găgăuz | 4.3% | 4.03% | +0.27% ✅ |
| Bulgar | 1.6% | 1.59% | +0.01% ✅ |
| Rrom | 0.4% | 0.39% | +0.01% ✅ |

**No changes needed** - ethnicity targets within 0.5% of NBS 2024.

---

### 4. Updated Statistics Export (exporters.py)

Added employment_status_distribution to PersonaStatistics:
```python
employment_dist = dict(Counter(p.employment_status for p in personas))
employment_dist = {k: v/n for k, v in employment_dist.items()}
```

---

## Files Modified

1. **moldova_personas/census_data.py**
   - Updated AGE_GROUP_DISTRIBUTION with NBS 2024 values
   - Added EMPLOYMENT_STATUS_DISTRIBUTION
   - Added EMPLOYMENT_STATUS_BY_AGE

2. **moldova_personas/models.py**
   - Added employment_status field to Persona model
   - Added employment_status_distribution to PersonaStatistics

3. **moldova_personas/generator.py**
   - Added _generate_employment_status() method
   - Updated generate_single() to include employment_status

4. **moldova_personas/exporters.py**
   - Updated StatisticsExporter.generate() to calculate employment distribution

---

## Test Results

```
============================= 101 passed in 6.44s ==============================
```

All existing tests pass with the new changes.

---

## Validation Sample (1000 personas)

### Employment Status Distribution:
| Status | Percentage |
|--------|-----------|
| employed | 35.6% |
| retired | 27.9% |
| other_inactive | 16.2% |
| student | 10.8% |
| homemaker | 7.4% |
| unemployed | 2.1% |

### Sample Output:
```
Persona: Valeriu Sirbu, Age: 61 (55-64)
Employment: employed
Occupation: Muncitor necalificat

Persona: Maria Cercheș, Age: 81 (65+)
Employment: retired
Occupation: Pensionar (fost: inginer)
```

---

## Known Limitations

1. **Age 18+ Filter:** The generator enforces minimum age of 18, which means:
   - 0-14 age group (16.6% of population) is never generated
   - 15-17 portion of 15-24 group is excluded
   - Result: Generated distributions are for **adult population only**

2. **Ethnicity in 18+ Sample:** Due to age demographics:
   - Ukrainians/Russians are overrepresented in older age groups
   - 18+ sample naturally has slightly different ethnicity mix than full population
   - This is demographically accurate, not an error

---

## Recommendations for 100K Generation

1. ✅ **Data quality is now verified** against NBS 2024
2. ✅ **Employment status adds realism** (no longer 100% employed)
3. ⚠️ **Document the 18+ limitation** clearly in dataset metadata
4. ⚠️ **Consider adding children** if full population representation needed

---

## Data Artifacts Created

1. `nbs_population_by_age_2024.json` - NBS age data
2. `nbs_workforce_2024.json` - NBS workforce stats
3. `nbs_ethnicity_2024.json` - NBS ethnicity breakdown
4. `NBS_2024_VERIFICATION_REPORT.md` - Full verification report
5. `NBS_2024_IMPLEMENTATION_SUMMARY.md` - This file

---

## Next Steps

1. Run 100K generation with updated distributions
2. Monitor employment_status distribution in output
3. Verify age distribution matches recalibrated 18+ targets
