# Nemotron Personas vs Moldova Personas - Comparison Analysis

**Date:** 2026-01-29  
**Nemotron Version:** 1.1 (USA)  
**Moldova Version:** 0.3.0

---

## Executive Summary

NVIDIA's Nemotron-Personas-USA is a gold-standard synthetic persona dataset with 1M records. This document compares our Moldova personas against Nemotron to identify improvement opportunities.

| Metric | Nemotron USA | Moldova | Gap |
|--------|-------------|---------|-----|
| **Scale** | 1,000,000 | 100,000 (target) | 10x smaller |
| **Occupations** | 559 unique | ~30 unique | 18.6x fewer |
| **Age Range** | 0-105 | 18-90 | Adults only ✅ |
| **Narrative Length** | ~247 chars (combined) | ~400 chars (6 sections) | More detailed |
| **Career Goals** | ✅ Yes | ❌ No | **MISSING** |
| **PGM Grounding** | ✅ Yes | ✅ Yes | Same approach |

---

## Schema Comparison

### Nemotron Fields (23 total)

**Persona Fields (6 narrative):**
1. `professional_persona` - Career/professional life
2. `sports_persona` - Sports & physical activities  
3. `arts_persona` - Arts & cultural interests
4. `travel_persona` - Travel preferences
5. `culinary_persona` - Cooking & food habits
6. `persona` - **Combined/summary persona**

**Contextual Fields (17 structured):**
- `uuid` - Unique identifier
- `cultural_background` - Ethnic/cultural context
- `skills_and_expertise` (text) + `_list` (structured)
- `hobbies_and_interests` (text) + `_list` (structured)
- `career_goals_and_ambitions` - **MISSING in Moldova**
- `sex`, `age`, `marital_status`
- `education_level`, `bachelors_field`
- `occupation` - 559 categories
- `city`, `state`, `zipcode`, `country`

### Moldova Fields (25 total)

**Persona Fields (6 narrative):**
1. `descriere_generala` - General personality
2. `profil_profesional` - Professional life
3. `hobby_sport` - Sports
4. `hobby_arta_cultura` - Arts & culture
5. `hobby_calatorii` - Travel
6. `hobby_culinar` - Culinary

**Contextual Fields (19 structured):**
- `uuid`, `name` (+ first/last derived)
- `sex`, `age`, `age_group`
- `ethnicity`, `mother_tongue`, `religion`
- `marital_status`
- `education_level`, `field_of_study`
- `occupation`, `occupation_sector` - ~30 categories
- `city`, `district`, `region`, `residence_type`
- `country` (default: Moldova)
- `skills_and_expertise_list`
- `hobbies_and_interests_list`

---

## Key Gaps Identified

### 1. Career Goals and Ambitions ❌ MISSING

**Nemotron has:** `career_goals_and_ambitions`  
**Example:** "She aims to advance from crew member to shift supervisor within the next year..."

**Our status:** No equivalent field

**Impact:** HIGH - Adds professional development dimension  
**Recommendation:** Add `career_goals_and_ambitions` field

---

### 2. Combined Persona Summary ❌ MISSING

**Nemotron has:** `persona` field (247 chars avg)  
**Example:** "Mary Alberti is a routine-obsessed, bullet-journal aficionado who balances disciplined work ambition..."

**Our status:** Only have 6 separate sections

**Impact:** MEDIUM - Useful for quick persona overview  
**Recommendation:** Add combined `persona_summary` field

---

### 3. Occupation Diversity ⚠️ LIMITED

**Nemotron:** 559 unique occupations  
**Top categories:**
- not_in_workforce: 43.5%
- manager: 1.2%
- customer_service_representative: 1.1%
- driver: 1.1%
- retail_salesperson: 1.1%
- software_developer: 1.0%

**Moldova:** ~30 occupations  
**Categories:** Profesor, Medic, Inginer, Șofer, etc.

**Impact:** MEDIUM - Limits persona diversity  
**Recommendation:** Expand to ~100+ occupations based on Moldovan labor statistics

---

### 4. Not in Workforce Category ⚠️ MISSING

**Nemotron:** 43.5% "not_in_workforce"  
**Subcategories:** unemployed, retired, student, homemaker, disabled

**Moldova:** No explicit category  
**Current:** Everyone has an occupation assigned

**Impact:** MEDIUM - Affects demographic realism  
**Recommendation:** Add "not_in_workforce" category with realistic distribution

---

### 5. Skills/Hobbies List Length ⚠️ SHORTER

**Nemotron:** 
- Skills: 10.7 items avg
- Hobbies: 9.1 items avg

**Moldova:**
- Skills: ~3-5 items (extracted)
- Hobbies: ~3-5 items (extracted)

**Impact:** LOW - Our extraction is conservative  
**Recommendation:** Improve extraction or generate more explicitly

---

## What We Do Better ✅

### 1. Ethnicity & Cultural Context
**Moldova:** 
- 8 ethnicities (Moldovean, Român, Ucrainean, Găgăuz, Rus, Bulgar, Rrom, Altele)
- Language mapping by ethnicity
- Religion correlated with ethnicity

**Nemotron:** 
- Names infused with ethnic distribution but not explicit ethnicity field
- Cultural background as narrative only

**Advantage:** Moldova ✅

---

### 2. Geographic Granularity
**Moldova:**
- 5 development regions
- 40+ cities
- Urban/rural classification
- District (raion) level

**Nemotron:**
- 50 states + territories
- 15,200+ cities
- ZIP code level (29,000+ ZCTAs)

**Advantage:** Nemotron for granularity, but Moldova appropriate for country size

---

### 3. Narrative Detail
**Moldova:** 6 separate sections, ~400 chars total
**Nemotron:** 6 sections + combined, ~247 chars for combined

**Advantage:** Moldova for detail per section

---

### 4. Statistical Validation
**Moldova:**
- Chi-square goodness-of-fit tests
- Joint distribution validation
- Adaptive tolerance based on sample size

**Nemotron:**
- Grounded in US Census distributions
- Visual validation (heatmaps, choropleths)

**Advantage:** Comparable - both use rigorous statistical grounding

---

## Recommendations Priority Matrix

| Feature | Priority | Effort | Impact | Implementation |
|---------|----------|--------|--------|----------------|
| Career goals field | HIGH | Low | High | Add to model + prompt |
| Combined persona | MEDIUM | Low | Medium | Add summary prompt |
| Expand occupations | MEDIUM | Medium | Medium | Research Moldovan labor stats |
| Not in workforce | MEDIUM | Low | Medium | Add occupation category |
| Longer skills lists | LOW | Low | Low | Tune extraction |

---

## Suggested Schema Additions

```python
# Add to Persona model:
career_goals_and_ambitions: str = Field(default="", description="Professional aspirations and career trajectory")
persona_summary: str = Field(default="", description="Combined summary of all persona aspects")

# Modify occupation to include:
# - "not_in_workforce" category (unemployed, retired, student, homemaker)
# - Expand to 100+ occupation codes based on Moldovan statistics
```

---

## Implementation Plan

### Phase 1: High Priority (Next)
1. Add `career_goals_and_ambitions` field to model
2. Add generation prompt for career goals
3. Add combined `persona_summary` field
4. Update exporters

### Phase 2: Medium Priority (Later)
5. Expand occupation categories to 100+
6. Add "not_in_workforce" category with distribution
7. Improve skills/hobbies extraction

---

## Conclusion

**Overall Assessment:** Our Moldova personas are comparable to Nemotron in quality and approach, with some specific gaps that can be addressed.

**Strengths:**
- ✅ Same PGM methodology
- ✅ Ethnic/cultural context
- ✅ Statistical validation
- ✅ Checkpointing/parallel processing (Nemotron doesn't mention this)

**Gaps:**
- ⚠️ Career goals field
- ⚠️ Fewer occupation categories
- ⚠️ No "not in workforce" category

**Recommendation:** Implement Phase 1 improvements before 100K generation to match Nemotron quality standards.
