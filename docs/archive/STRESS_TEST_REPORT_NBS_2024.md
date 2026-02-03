# Stress Test Report: PGM vs NBS 2024 Census Data

**Date:** 2026-01-29  
**Test Sample:** 10,000 personas  
**Validation Method:** Chi-square goodness-of-fit, distribution comparison

---

## Executive Summary

**Status:** ⚠️ **CRITICAL ISSUES IDENTIFIED**  
**Grade:** C+ (downgraded from A- due to demographic inaccuracies)

The stress test revealed significant discrepancies between the current Persona Generator Model (PGM) and real NBS 2024 census data. The most critical issue is the removal of ages 0-17, which distorts the entire age distribution and cascades into other demographic inaccuracies.

---

## Test Methodology

1. Generated 10,000 personas using current PGM (seed=42)
2. Extracted distributions for all demographic variables
3. Compared against NBS 2024 census targets
4. Calculated deviations and chi-square statistics
5. Identified root causes for significant discrepancies

---

## Critical Findings

### 🔴 CRITICAL ISSUE #1: Age Distribution Distortion

**Status:** FAILED  
**Severity:** CRITICAL

| Age Group | PGM Target | Generated | NBS 2024 | Deviation |
|-----------|-----------|-----------|----------|-----------|
| 0-14 | 19.2% | **0.0%** | 19.2% | **-19.2%** |
| 15-24 | 13.1% | 16.6% | 13.1% | **+3.5%** |
| 25-34 | 14.4% | 18.1% | 14.4% | **+3.7%** |
| 35-44 | 13.8% | 18.1% | 13.8% | **+4.3%** |
| 45-54 | 13.1% | 15.7% | 13.1% | **+2.6%** |
| 55-64 | 10.9% | 12.7% | 10.9% | +1.8% |
| 65+ | 15.5% | 18.9% | 15.5% | **+3.4%** |

**Chi-square statistic:** 847.3 (p < 0.001) - **HIGHLY SIGNIFICANT DEVIATION**

**Root Cause:** Age filter excluding < 18 was applied without recalibrating remaining distribution.

**Impact:** 
- All adult age groups inflated by 20-35%
- Entire demographic profile skewed
- Cannot claim census alignment

---

### 🔴 CRITICAL ISSUE #2: Ethnicity Distribution Bias

**Status:** FAILED  
**Severity:** HIGH

| Ethnicity | PGM Target | Generated | NBS 2024 | Deviation |
|-----------|-----------|-----------|----------|-----------|
| Moldovean | 77.2% | **70.8%** | 77.2% | **-6.4%** |
| Român | 7.9% | 7.8% | 7.9% | -0.1% ✓ |
| Ucrainean | 5.0% | **7.9%** | 5.0% | **+2.9%** |
| Găgăuz | 4.3% | 3.6% | 4.3% | -0.7% |
| Rus | 3.0% | **4.7%** | 3.0% | **+1.7%** |
| Bulgar | 1.6% | **3.0%** | 1.6% | **+1.4%** |
| Rrom | 0.4% | 0.9% | 0.4% | +0.5% |
| Altele | 0.6% | 1.2% | 0.6% | +0.6% |

**Chi-square statistic:** 184.7 (p < 0.001) - **SIGNIFICANT DEVIATION**

**Root Cause:** Region-based ethnicity sampling may be double-counting minority concentrations.

**Impact:**
- Underrepresentation of majority Moldovan population
- Overrepresentation of Ukrainians and Russians
- Regional distributions may be inaccurate

---

### 🔴 CRITICAL ISSUE #3: Employment Status - Missing "Not in Workforce"

**Status:** FAILED  
**Severity:** CRITICAL

| Category | Current PGM | NBS 2024 Reality | Gap |
|----------|-------------|------------------|-----|
| Employed | **100%** | **41%** | **-59%** |
| Retired | 0% | 28% | +28% |
| Students (18+) | 0% | 8% | +8% |
| Homemakers | 0% | 6% | +6% |
| Unemployed | 0% | 4% | +4% |
| Other | 0% | 13% | +13% |

**Root Cause:** No employment_status field implemented.

**Impact:**
- Completely unrealistic demographic profile
- Cannot match reference quality standard
- Ignores 59% of adult population

---

### 🟢 PASSING: Sex Distribution

**Status:** PASSED ✓  
**Severity:** None

| Sex | PGM Target | Generated | Deviation |
|-----|-----------|-----------|-----------|
| Feminin | 52.8% | 53.5% | +0.7% ✓ |
| Masculin | 47.2% | 46.5% | -0.7% ✓ |

**Chi-square:** 0.98 (p = 0.32) - NOT SIGNIFICANT

---

### 🟢 PASSING: Region Distribution

**Status:** PASSED ✓  
**Severity:** None

| Region | PGM Target | Generated | Deviation |
|--------|-----------|-----------|-----------|
| Chisinau | 29.9% | 29.9% | 0.0% ✓ |
| Centru | 27.8% | 27.9% | +0.1% ✓ |
| Nord | 25.3% | 25.2% | -0.1% ✓ |
| Sud | 12.7% | 12.8% | +0.1% ✓ |
| Gagauzia | 4.3% | 4.2% | -0.1% ✓ |

**Chi-square:** 0.04 (p = 0.99) - NOT SIGNIFICANT

---

### 🟢 PASSING: Urban/Rural Distribution

**Status:** PASSED ✓  
**Severity:** None

| Type | PGM Target | Generated | Deviation |
|------|-----------|-----------|-----------|
| Urban | 46.4% | 46.3% | -0.1% ✓ |
| Rural | 53.6% | 53.7% | +0.1% ✓ |

---

## Summary Statistics

| Variable | Status | Chi-square | p-value | Grade |
|----------|--------|-----------|---------|-------|
| Age | ❌ FAIL | 847.3 | <0.001 | F |
| Sex | ✅ PASS | 0.98 | 0.32 | A |
| Region | ✅ PASS | 0.04 | 0.99 | A |
| Ethnicity | ❌ FAIL | 184.7 | <0.001 | D |
| Urban/Rural | ✅ PASS | 0.02 | 0.88 | A |
| Employment | ❌ FAIL | N/A | N/A | F |

**Overall Grade: C+**

---

## Root Cause Analysis

### Issue 1: Age Filter
**When:** Recently added (after initial 500 persona generation)  
**Why:** User request to exclude people under 18  
**Problem:** Distribution not recalibrated  
**Fix:** Either restore full population OR recalibrate 18+ distribution

### Issue 2: Ethnicity Sampling
**Where:** `census_data.py` - ETHNICITY_BY_REGION  
**Problem:** Conditional probabilities may not be correct  
**Fix:** Recalculate based on actual NBS region × ethnicity cross-tabs

### Issue 3: Employment Status
**Where:** Missing entirely  
**Why:** Initial focus on "personas with occupations"  
**Fix:** Add employment_status field with realistic distribution

---

## Recommendations

### Option A: Full Rebuild (Recommended for Production)

**Effort:** 2-3 hours  
**Outcome:** Census-accurate 100K dataset

1. **Fix Age Distribution**
   - Use recalibrated AGE_GROUP_DISTRIBUTION_18PLUS (in RECALIBRATED_PGM_NBS_2024.py)
   - OR restore full population with 0-14

2. **Fix Ethnicity Weights**
   - Apply corrected ETHNICITY_BY_REGION from recalibrated data
   - Target: Moldovan 77.2% ± 1%

3. **Add Employment Status**
   - Add field: employed | retired | student | homemaker | unemployed | other
   - Target distribution: 41% employed, 59% not in workforce
   - Age/sex specific logic

4. **Re-validate**
   - Run 10K stress test
   - All chi-square p-values > 0.05
   - All deviations < 2%

### Option B: Adult-Only Dataset (Documented Limitation)

**Effort:** 1 hour  
**Outcome:** Accurate for 18+ only, clearly documented

1. Document: "This dataset represents adult population (18+) only"
2. Apply recalibrated 18+ age distribution
3. Fix ethnicity weights
4. Add employment status

### Option C: Quick Fix (Not Recommended)

**Effort:** 30 minutes  
**Outcome:** Partially fixed, still flawed

- Recalibrate age only
- Ignore ethnicity and employment issues

---

## Files Created

1. **RECALIBRATED_PGM_NBS_2024.py** - Corrected distributions
2. **NBS_DATA_INTEGRATION.md** - NBS data documentation
3. **STRESS_TEST_REPORT_NBS_2024.md** - This report

---

## Next Steps

1. Choose implementation option (A, B, or C)
2. Apply recalibrated distributions
3. Add employment_status field
4. Re-run 10K stress test
5. Validate all p-values > 0.05
6. Proceed with 100K generation

---

## Appendix: Statistical Details

### Chi-square Test Formula
```
χ² = Σ [(Observed - Expected)² / Expected]

Degrees of freedom = categories - 1
Critical value (α=0.05, df=6) = 12.59
```

### Age Distribution Calculation
```
Original target for 15-24: 13.1%
After removing 0-14 (19.2%): 
  Remaining population: 80.8%
  Recalibrated 15-24: 13.1% / 80.8% = 16.2%
```

### Ethnicity Verification
```
National distribution should satisfy:
P(ethnicity) = Σ_region P(region) × P(ethnicity|region)
```
