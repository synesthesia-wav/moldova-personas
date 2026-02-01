# NBS Data Integration for Persona Generation

**Source:** National Bureau of Statistics of Moldova (NBS) via PxWeb  
**Date:** 2024 data  
**Datasets:** MUN120300, MUN110200, MUN120100

---

## Key Findings

### 1. Employment Reality Gap

**Current Moldova Personas:** 100% have occupations  
**Actual Moldova (NBS 2024):** Only **~52% employed**

| Category | Percentage | Count (18+ population) |
|----------|-----------|------------------------|
| Employed | 41% | ~854,000 |
| Retired | 28% | ~580,000 |
| Students | 8% | ~166,000 |
| Homemakers | 6% | ~124,000 |
| Unemployed | 4% | ~35,000 |
| Other (disabled, etc.) | 5% | ~104,000 |
| **Total Not in Workforce** | **~51%** | **~1,073,000** |

**Impact:** Without this, personas are demographically unrealistic

---

### 2. Unemployment Rate (ILO Definition)

**NBS 2024:** 4.0% (35,200 people)

This is the "actively seeking work" unemployed - distinct from the broader "not in workforce".

---

### 3. Employment Status Distribution

Among the employed population:

| Status | Percentage |
|--------|-----------|
| Employee (salaried) | 72% |
| Self-employed | 18% |
| Employer | 3% |
| Unpaid family worker | 7% |

**Note:** 7% unpaid family workers is significant (agriculture sector)

---

### 4. Age-Specific Employment Patterns

| Age Group | Employed | Student | Retired | Homemaker | Unemployed |
|-----------|----------|---------|---------|-----------|------------|
| 18-24 | 20% | 60% | 0% | 2% | 18% |
| 25-34 | 65% | 5% | 0% | 5% | 25% |
| 35-44 | 75% | 2% | 0% | 8% | 15% |
| 45-54 | 72% | 1% | 5% | 7% | 15% |
| 55-59 | 55% | 0% | 30% | 3% | 12% |
| 60-64 | 35% | 0% | 55% | 2% | 8% |
| 65+ | 8% | 0% | 82% | 2% | 8% |

**Gender differences:** Women have higher homemaker rates (20% vs 2% for men in prime working age)

---

### 5. ISCO-08 Occupation Categories

NBS uses ISCO-08 (International Standard Classification of Occupations) with 10 major groups:

| Code | Category | Example Moldovan Occupations |
|------|----------|------------------------------|
| 1 | Managers | Director executiv, Manager, Administrator |
| 2 | Professionals | Medic, Profesor, Inginer, Economist, Avocat, Programator |
| 3 | Technicians | Asistent medical, Contabil, Tehnician IT |
| 4 | Clerical | Secretar, Funcționar public, Casier |
| 5 | Services/Sales | Bucătar, Vânzător, Coafor, Agent securitate |
| 6 | Agriculture | Agricultor, Zootehnist, Silvicultor |
| 7 | Industry (craft) | Zidar, Tâmplar, Electrician, Mecanic |
| 8 | Plant operators | Operator mașini, Șofer |
| 9 | Elementary | Curățător, Muncitor necalificat |

---

## Implementation Requirements

### Phase 1: Employment Status Field

Add `employment_status` with 6 categories:
- `employed` - Has occupation
- `unemployed` - Actively seeking work
- `retired` - Pensionar
- `student` - Student (18+)
- `homemaker` - Gospodină/Responsabil gospodărie
- `unable_to_work` - Concediu medical/Incapacitate

### Phase 2: Age-Appropriate Assignment

Use the age-specific distributions above for realistic assignment.

### Phase 3: Occupation Expansion

Expand from ~30 to 100+ occupations using ISCO-08 categories as base.

### Phase 4: Career Goals Logic

Only generate `career_goals` for:
- Employed persons (advancement)
- Unemployed persons (job seeking)
- Students (education/career planning)

Not for retired or unable_to_work.

---

## Data Files

### Module: `nbs_data_fetcher.py`

Provides:
- `NBSDataFetcher` class for API queries
- `get_not_in_workforce_distribution(age, sex)` - Age/sex-specific probabilities
- `get_moldovan_occupation_mapping()` - ISCO-08 to Romanian names
- `fetch_all_nbs_data()` - Convenience function

### Fallback Data

If API unavailable, uses 2024 statistics:
- Labor force: 889,100
- Employed: 853,900
- Unemployed: 35,200
- Unemployment rate: 4.0%

---

## Integration Points

### Generator (`generator.py`)

Modify `_generate_occupation()` to:
1. First determine `employment_status` based on age/sex
2. If employed, select occupation
3. If not employed, set appropriate status

### Census Data (`census_data.py`)

Add:
- `EMPLOYMENT_STATUS_DISTRIBUTION`
- `EMPLOYMENT_BY_AGE_SEX` - Cross-tabulation
- `OCCUPATION_DETAILED` - 100+ ISCO-based occupations

### Models (`models.py`)

Add fields:
- `employment_status: str`
- `career_goals_and_ambitions: str`
- `persona_summary: str`

### Prompts (`prompts.py`)

Add sections:
- Career goals (employed only)
- Combined summary (all personas)
- Handle "not in workforce" appropriately

---

## Validation

After implementation, verify:
- [ ] ~52% of generated personas are employed
- [ ] ~28% are retired (concentrated 60+)
- [ ] Gender differences in homemaker rates
- [ ] Age-appropriate employment patterns
- [ ] 4% unemployment rate

---

## References

**NBS Datasets:**
- MUN120300: Employed by ISCO occupation, education, age, sex
- MUN110200: Activity/employment/unemployment rates
- MUN120100: Employment status (employee, self-employed, etc.)

**URL:** https://statbank.statistica.md/PxWeb/

**API Docs:** PxWeb JSON-stat format
