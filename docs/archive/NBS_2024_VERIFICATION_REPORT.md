# NBS 2024 Census Data Verification Report

**Date:** 2026-01-29  
**Sources:**
- NBS PxWeb API (statbank.statistica.md)
- Date_Comunicat_Etnoculturale_20_10_25.xlsx (NBS 2024 Census)

---

## Executive Summary

This report verifies the Persona Generator Model (PGM) targets against actual NBS 2024 census data.

### Key Findings:

| Metric | Status | Action Required |
|--------|--------|-----------------|
| **Age Distribution** | ⚠️ INCORRECT | Update targets |
| **Ethnicity** | ✅ ACCURATE | Minor tweaks optional |
| **Sex Ratio** | ✅ ASSUMED OK | Verify with NBS |
| **Region Distribution** | ⚠️ UNVERIFIED | Fetch from NBS |
| **Employment Status** | ⚠️ MISSING FIELD | Add to PGM |

---

## 1. Population Data (NBS 2024)

**Total Population:** 2,423,287 (as of Jan 1, 2024)

Source: NBS PxWeb API - POP010200rcl.px

---

## 2. Age Distribution

### PGM vs NBS 2024 Comparison

| Age Group | PGM Target | NBS 2024 Actual | Error | Population |
|-----------|-----------|-----------------|-------|------------|
| 0-14 | 19.2% | **16.6%** | +2.6% | 401,522 |
| 15-24 | 13.1% | **9.8%** | +3.3% | 237,500 |
| 25-34 | 14.4% | **12.4%** | +2.0% | 299,760 |
| 35-44 | 13.8% | **15.4%** | -1.6% | 373,682 |
| 45-54 | 13.1% | **13.1%** | 0.0% | 318,021 |
| 55-64 | 10.9% | **14.4%** | -3.5% | 349,370 |
| 65+ | 15.5% | **18.3%** | -2.8% | 443,432 |

### Recommended Correction

```python
AGE_GROUP_DISTRIBUTION = {
    "0-14": 0.166,    # Was 0.192
    "15-24": 0.098,   # Was 0.131
    "25-34": 0.124,   # Was 0.144
    "35-44": 0.154,   # Was 0.138
    "45-54": 0.131,   # Unchanged
    "55-64": 0.144,   # Was 0.109
    "65+": 0.183,     # Was 0.155
}
```

---

## 3. Ethnicity Distribution

### PGM vs NBS 2024 Comparison

| Ethnicity | PGM Target | NBS 2024 Actual | Error | Status |
|-----------|-----------|-----------------|-------|--------|
| Moldovean | 77.2% | **76.73%** | +0.47% | ✅ OK |
| Român | 7.9% | **8.02%** | -0.12% | ✅ OK |
| Ucrainean | 5.0% | **5.13%** | -0.13% | ✅ OK |
| Rus | 3.0% | **3.39%** | -0.39% | ✅ OK |
| Găgăuz | 4.3% | **4.03%** | +0.27% | ✅ OK |
| Bulgar | 1.6% | **1.59%** | +0.01% | ✅ OK |
| Rrom | 0.4% | **0.39%** | +0.01% | ✅ OK |
| Altele | 0.6% | **0.72%** | -0.12% | ✅ OK |

### Conclusion

**PGM ethnicity targets are well-calibrated!** All ethnicities are within 0.5% of NBS 2024 actuals.

The previous claim of "Moldovans underrepresented by 6.4%" was **incorrect**.

---

## 4. Ethnicity by Age Group (NBS 2024)

| Age Group | Moldoveni | Români | Ucraineni | Ruși | Găgăuzi | Bulgari |
|-----------|-----------|--------|-----------|------|---------|---------|
| 0-9 ani | 13.3% | 12.7% | 6.2% | 6.5% | 12.3% | 8.4% |
| 10-19 ani | 13.2% | 12.7% | 8.0% | 8.8% | 12.4% | 10.5% |
| 20-29 ani | 9.5% | 8.2% | 6.7% | 5.8% | 8.7% | 7.4% |
| 30-39 ani | 14.4% | 13.0% | 11.0% | 10.5% | 13.7% | 12.5% |
| 40-49 ani | 13.5% | 13.4% | 13.1% | 13.7% | 12.8% | 15.0% |
| 50-59 ani | 12.3% | 12.6% | 15.2% | 14.7% | 11.9% | 13.3% |
| 60-69 ani | 14.0% | 16.1% | 20.1% | 19.1% | 17.5% | 17.8% |
| 70-79 ani | 7.4% | 9.2% | 13.7% | 15.0% | 8.5% | 11.7% |
| 80+ ani | 2.3% | 2.0% | 6.1% | 5.8% | 2.2% | 3.6% |

**Key Insight:** Ukrainian and Russian populations are significantly older on average.

---

## 5. Ethnicity by Urban/Rural (NBS 2024)

| Ethnicity | Urban | Rural | Difference |
|-----------|-------|-------|------------|
| Moldoveni | 70.45% | 82.19% | +11.7% rural |
| Români | 9.81% | 6.46% | +3.4% urban |
| Ucraineni | 6.27% | 4.14% | +2.1% urban |
| Ruși | 5.90% | 1.21% | +4.7% urban |
| Găgăuzi | 3.70% | 4.33% | +0.6% rural |
| Bulgari | 2.03% | 1.20% | +0.8% urban |

---

## 6. Workforce Statistics (NBS 2024)

Source: NBS PxWeb API - MUN110200.px

| Indicator | Rate |
|-----------|------|
| Activity Rate | 44.5% |
| Employment Rate | 42.7% |
| Unemployment Rate | 4.0% |

**Note:** These are for population aged 15+ (working age definition).

---

## 7. Employment Status (Missing Field)

The PGM currently has no `employment_status` field. Based on NBS data:

**Recommendation for new field:**

```python
EMPLOYMENT_STATUS_DISTRIBUTION = {
    "employed": 0.427,      # 42.7% employment rate (15+)
    "unemployed": 0.018,    # ~1.8% of total population
    "student": 0.08,        # Estimated
    "retired": 0.20,        # Estimated (65+ is 18.3%)
    "homemaker": 0.06,      # Estimated
    "other": 0.15,          # Other inactive
}
```

---

## 8. Action Items

### High Priority
1. ✅ **Verify NBS data access** - COMPLETED
2. ⚠️ **Update AGE_GROUP_DISTRIBUTION** in census_data.py
3. ⚠️ **Add employment_status field** to persona model

### Medium Priority
4. ⚠️ **Fetch region distribution** from NBS (if available)
5. ⚠️ **Verify sex ratio** with NBS data
6. ⚠️ **Update ethnicity by region** with 2024 data (if available)

### Low Priority
7. Optional: Fine-tune ethnicity targets to match NBS exactly
8. Optional: Add ethnicity × age interaction in PGM

---

## 9. Files Created

1. `nbs_population_by_age_2024.json` - Single-year age data
2. `nbs_workforce_2024.json` - Workforce statistics
3. `nbs_ethnicity_2024.json` - Comprehensive ethnicity data
4. `nbs_corrections_2024.json` - Correction recommendations
5. `NBS_2024_VERIFICATION_REPORT.md` - This report

---

## 10. Data Sources

### NBS PxWeb API
- **Base URL:** https://statbank.statistica.md/PxWeb/api/v1/en
- **Population:** 20 Populatia si procesele demografice / POP010 / POPro
- **Workforce:** 30 Statistica sociala / 03 FM / 03 MUN / MUN010

### Excel File
- **File:** Date_Comunicat_Etnoculturale_20_10_25.xlsx
- **Source:** NBS Census 2024 (published Oct 20, 2025)
- **Contains:** Ethnicity, language, age, sex distributions

---

## Conclusion

The PGM is in good shape overall. The main issue is the **age distribution targets** which are significantly off from NBS 2024 actuals. Ethnicity targets are remarkably accurate.

**Immediate action:** Update `AGE_GROUP_DISTRIBUTION` in `census_data.py` with corrected values.
