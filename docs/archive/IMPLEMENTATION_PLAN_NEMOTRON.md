# Implementation Plan: Nemotron Alignment

**Objective:** Bring Moldova Personas to Nemotron-quality standards  
**Target Version:** 0.4.0  
**Estimated Total Effort:** 2-3 hours  
**Dependencies:** None (all can be done independently)

---

## Phase 1: Core Schema Additions (30 min)

### 1.1 Add Career Goals Field
**Priority:** HIGH  
**Effort:** 10 min  
**Files:** `models.py`, `prompts.py`

```python
# models.py - Add to Persona class
career_goals_and_ambitions: str = Field(
    default="",
    description="Professional aspirations and career trajectory"
)

# prompts.py - Add section to adult/elderly prompts
"""
**7. Obiective profesionale** - Aspirații de carieră, planuri de dezvoltare 
profesională, dorințe de avansare. 2-3 propoziții.
"""
```

**Validation:** Check that age-appropriate (not for retirees)  
**Tests:** Add to test_prompts.py

---

### 1.2 Add Combined Persona Summary
**Priority:** HIGH  
**Effort:** 10 min  
**Files:** `models.py`, `prompts.py`

```python
# models.py - Add to Persona class
persona_summary: str = Field(
    default="",
    description="Combined summary of all persona aspects (~250 chars)"
)

# prompts.py - Add as section 8 or generate post-hoc
"""
**8. Rezumat personal** - O sinteză concisă (2-3 propoziții) care capturează 
esența acestei persoane: personalitate, ocupație, valori principale.
"""
```

**Alternative:** Generate summary from existing sections using LLM post-processing

---

### 1.3 Update Exporters
**Priority:** MEDIUM  
**Effort:** 10 min  
**Files:** `exporters.py`

Ensure new fields are included in Parquet/JSON/CSV exports.

---

## Phase 2: Occupation Expansion (45 min)

### 2.1 Research Moldovan Occupation Categories
**Priority:** MEDIUM  
**Effort:** 15 min  
**Source:** BNS Statistical Yearbook, Labor Force Survey

Target: Expand from ~30 to ~100+ occupations

**Categories to add:**
- Administrative: manager, secretar, funcționar
- Technical: tehnician, operator, mecanic
- Services: ospătar, bucătar, coafez, șofer
- Sales: vânzător, agent vânzări, casier
- Healthcare: asistent medical, infirmier, farmacist
- Education: învățător, educator, bibliotecar
- Agriculture: agronom, zootehnist, tractorist
- IT: programator, administrator rețea, designer web
- Finance: contabil, economist, bancher
- Construction: zidar, tâmplar, electrician, instalator
- Arts/Media: jurnalist, designer, fotograf, cameraman
- Legal: avocat, notar, executor judecătoresc
- etc.

---

### 2.2 Implement Occupation Distribution
**Priority:** MEDIUM  
**Effort:** 20 min  
**Files:** `census_data.py`, `generator.py`

```python
# census_data.py - Add detailed occupation distribution
OCCUPATION_DETAILED: Dict[str, float] = {
    # Services (~25%)
    "Vânzător": 0.08,
    "Șofer": 0.06,
    "Ospătar": 0.04,
    "Bucătar": 0.03,
    "Casier": 0.02,
    "Coafez": 0.02,
    
    # Industry (~20%)
    "Muncitor industrial": 0.08,
    "Mecanic": 0.04,
    "Electrician": 0.03,
    "Instalator": 0.02,
    "Zidar": 0.02,
    "Tâmplar": 0.01,
    
    # Agriculture (~15%)
    "Agricultor": 0.10,
    "Zootehnist": 0.03,
    "Tractorist": 0.02,
    
    # Education/Health (~12%)
    "Profesor": 0.04,
    "Învățător": 0.03,
    "Asistent medical": 0.03,
    "Medic": 0.01,
    "Farmacist": 0.01,
    
    # Administration (~10%)
    "Manager": 0.03,
    "Funcționar public": 0.03,
    "Secretar": 0.02,
    "Contabil": 0.02,
    
    # IT/Finance (~8%)
    "Programator": 0.03,
    "Economist": 0.02,
    "Administrator rețea": 0.02,
    "Bancher": 0.01,
    
    # Other (~10%)
    "Inginer": 0.03,
    "Avocat": 0.01,
    "Jurnalist": 0.01,
    "Designer": 0.01,
    "Agricultor": 0.02,
    "Muncitor necalificat": 0.02,
}
```

---

### 2.3 Add Occupation by Education Mapping
**Priority:** MEDIUM  
**Effort:** 10 min  
**Files:** `generator.py`

Ensure high-skill jobs require appropriate education level.

---

## Phase 3: "Not in Workforce" Category (30 min)

### 3.1 Add Employment Status Field
**Priority:** MEDIUM  
**Effort:** 15 min  
**Files:** `models.py`, `census_data.py`, `generator.py`

```python
# models.py
employment_status: str = Field(
    default="employed",
    pattern="^(employed|unemployed|retired|student|homemaker|unable_to_work)$"
)

# census_data.py - Distribution for 18+ population
EMPLOYMENT_STATUS_DISTRIBUTION: Dict[str, float] = {
    "employed": 0.52,        # ~52% employed
    "unemployed": 0.05,      # ~5% unemployed
    "retired": 0.27,         # ~27% retirees (age 60+)
    "student": 0.08,         # ~8% students (18-25)
    "homemaker": 0.06,       # ~6% homemakers
    "unable_to_work": 0.02,  # ~2% unable to work
}

EMPLOYMENT_BY_AGE: Dict[str, Dict[str, float]] = {
    "15-24": {"employed": 0.25, "student": 0.60, "unemployed": 0.10, ...},
    "25-34": {"employed": 0.70, "unemployed": 0.08, "homemaker": 0.05, ...},
    "35-44": {"employed": 0.75, "unemployed": 0.06, ...},
    "45-54": {"employed": 0.72, "unemployed": 0.07, ...},
    "55-64": {"employed": 0.55, "retired": 0.30, ...},
    "65+":   {"retired": 0.85, "employed": 0.05, ...},
}
```

---

### 3.2 Modify Occupation Assignment
**Priority:** MEDIUM  
**Effort:** 15 min  
**Files:** `generator.py`

```python
def _generate_occupation(self, age, education, employment_status):
    if employment_status != "employed":
        return employment_status, None  # e.g., "Pensionar", "Șomer", "Student"
    
    # Existing occupation logic for employed persons
    ...
```

**Occupation values for non-employed:**
- retired: "Pensionar" + former profession
- unemployed: "Șomer în căutarea unui loc de muncă"
- student: "Student" (already exists)
- homemaker: "Gospodină" / "Responsabil de gospodărie"
- unable_to_work: "Concediu medical / Incapacitate"

---

## Phase 4: Improved Skills/Hobbies Extraction (20 min)

### 4.1 Enhance Extraction Logic
**Priority:** LOW  
**Effort:** 20 min  
**Files:** `prompts.py`

Current: Extracts 3-5 items from narrative  
Target: Generate 8-12 items explicitly

```python
# Option 1: Improve extraction
SKILL_KEYWORDS = [
    "competențe", "abilități", "cunoștințe", "experiență în",
    "specializat", "calificat", "priceput", "aptitudini",
    "expert în", "cunoaște", "stăpânește", "lucrează cu",
    # Add more...
]

# Option 2: Generate explicitly via LLM
"""
Listează exact 5-7 competențe profesionale specifice:
- 
- 
- 
"""
```

---

### 4.2 Add Hobbies Categories
**Priority:** LOW  
**Effort:** Included above  
**Files:** `prompts.py`

Expand hobby categories:
- Sports: fitness, yoga, swimming, hiking, cycling
- Arts: music, painting, photography, theater, reading
- Social: dancing, cooking, gardening, traveling
- Intellectual: chess, puzzles, learning languages
- etc.

---

## Phase 5: Testing & Validation (30 min)

### 5.1 Update Test Suite
**Priority:** HIGH  
**Effort:** 20 min  
**Files:** `tests/`

- Add tests for new fields (career goals, summary)
- Add tests for employment status
- Add tests for expanded occupations
- Verify all 101+ tests pass

---

### 5.2 Update Documentation
**Priority:** MEDIUM  
**Effort:** 10 min  
**Files:** `CHANGELOG.md`, `README.md`

Document schema changes and Nemotron alignment.

---

## Implementation Order

### Option A: Sequential (Safest)
1. Phase 1 (30 min)
2. Phase 2 (45 min)
3. Phase 3 (30 min)
4. Phase 4 (20 min)
5. Phase 5 (30 min)
**Total:** ~3 hours

### Option B: Parallel (Fastest)
- Task 1: Schema additions (Phase 1) - 30 min
- Task 2: Occupation expansion (Phase 2) - 45 min
- Task 3: Employment status (Phase 3) - 30 min
- Task 4: Skills extraction (Phase 4) - 20 min
- Task 5: Testing (Phase 5) - 30 min
**Total:** ~45 min (with parallel work)

### Option C: Phased Delivery (Recommended)
**Phase 1:** Core additions (career goals + summary) - 30 min
→ Can proceed to 100K generation

**Phase 2:** Occupation expansion + employment status - 1 hour
→ Improved demographic realism

**Phase 3:** Skills/hobbies enhancement - 20 min
→ Polish and completeness

---

## Acceptance Criteria

- [ ] `career_goals_and_ambitions` field exists and populated
- [ ] `persona_summary` field exists and populated
- [ ] `employment_status` field exists with 6 categories
- [ ] 100+ unique occupations available
- [ ] "Not in workforce" personas generated with realistic distribution
- [ ] All 101+ tests pass
- [ ] Schema documented
- [ ] CHANGELOG.md updated

---

## Rollback Plan

If issues arise:
1. Revert to v0.3.0 tag
2. Check out checkpoint from before changes
3. All changes are additive (no breaking changes)

---

**Ready for implementation when you give the go-ahead.**
