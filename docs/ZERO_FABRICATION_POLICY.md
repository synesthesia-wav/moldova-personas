# Zero Fabricated Data Policy

**Version:** 1.0  
**Effective Date:** 2026-01-30  
**Applies to:** Moldova Personas Generator (all versions)

---

## 1. Policy Statement

This project maintains a strict **Zero Fabricated Data Policy**. We generate synthetic personas based on statistically rigorous methods grounded in official census data. We do not invent facts, create fake entities, or generate content that could be mistaken for real personally identifiable information (PII).

---

## 2. Definitions

### 2.1 What Counts as "Fabrication"

Fabrication includes:

- **Specific Real-world Entities**: Named companies, schools, hospitals, government offices
- **Specific Addresses**: Street addresses, building numbers, postal codes tied to real locations
- **Real Person Names**: Names of actual living or deceased individuals
- **Contact Information**: Phone numbers, email addresses, social media handles
- **Specific Identifiers**: ID numbers, passport numbers, employee IDs
- **Fabricated Geographic Data**: Village/city names that don't exist
- **Template-generated Cultural Content**: Mad Libs-style fill-in-the-blank backgrounds

### 2.2 What Does NOT Count as "Fabrication"

These are acceptable:

- **Statistically Representative Names**: Generated from weighted distributions of real name frequencies
- **Generic Occupations**: "Economist", "Teacher", "Driver" (not "Chief Economist at BNM")
- **Generic Locations**: "Chișinău" (real city), "a town in northern Moldova" (vague but accurate)
- **Cultural Context**: "Găgăuz traditions", "Orthodox Christianity" (demographic patterns)
- **Abstract Narrative Content**: Personality descriptions, hobby preferences, lifestyle patterns

---

## 3. Implementation Boundaries

### 3.1 Structured Data Fields

| Field | Policy | Implementation |
|-------|--------|----------------|
| `name` | ✅ Statistical sampling | Ethnicity/sex-weighted name pools |
| `city` | ✅ Real places only | NBS-verified cities and villages |
| `occupation` | ✅ Generic only | No specific employer names |
| `education_level` | ✅ ISCED categories | No specific university names |
| `field_of_study` | ✅ Broad categories | No specific degree programs |

### 3.2 Narrative Fields

| Field | Allowed | Forbidden |
|-------|---------|-----------|
| `descriere_generala` | Personality traits, general background | Specific childhood stories, named family members |
| `profil_profesional` | Generic responsibilities, skills | Named employers, specific projects, real clients |
| `hobby_sport` | Activity types, frequency | Named sports clubs, specific competitions |
| `hobby_arta_cultura` | Genre preferences, general interests | Named artists (except public figures), specific venues |
| `hobby_calatorii` | Travel styles, preferred regions | Specific hotels, real itineraries |
| `hobby_culinar` | Cuisine types, cooking habits | Named restaurants, specific family recipes |

### 3.3 Explicitly Empty (No Fallback)

These fields are left empty when LLM generation is unavailable:

- `cultural_background` - No template fallbacks
- `career_goals_and_ambitions` - Optional, may be empty
- `persona_summary` - Optional, may be empty

---

## 4. LLM Content Guidelines

### 4.1 Prompt Engineering Safeguards

Our prompts to LLMs include explicit constraints:

```
RESTRICȚII:
- NU inventa nume de companii, școli, sau instituții reale
- NU folosi adrese specifice sau numere de telefon
- NU menționa persoane reale (politicieni, celebrități)
- Fii general: „o companie locală", „o universitate din Chișinău"
```

### 4.2 Content Validation

Post-generation validation includes:

1. **PII Detection**: Heuristics for phone numbers, emails, IDs
2. **Named Entity Recognition**: Check for unexpected proper nouns
3. **Romanian Language Validation**: Ensure diacritics present (reduces hallucination)
4. **Length Validation**: Too short = likely failed, too long = possible drift

### 4.3 Acceptable Generic Placeholders

When specific details would normally be provided, use these patterns:

| Instead of... | Use... |
|---------------|--------|
| "Works at Moldova Agroindbank" | "Works at a financial institution" |
| "Studied at ASEM" | "Studied at a university in Chișinău" |
| "Lives on București Street 55" | "Lives in a central neighborhood" |
| "Manager at Kaufland" | "Works in retail management" |

---

## 5. Data Provenance Levels

We categorize data by fabrication risk:

| Level | Description | Examples |
|-------|-------------|----------|
| **Measured** | Direct from NBS PxWeb | Sex distribution, age groups |
| **Derived** | IPF from marginals | Ethnicity×Region cross-tab |
| **Sampled** | Statistical distributions | Names, occupations |
| **Generated** | LLM narrative content | Hobby descriptions |
| **Empty** | Intentionally blank | When generation fails |

**Policy**: Measured > Derived > Sampled > Generated > Empty (preferred over fabrication)

---

## 6. Compliance Verification

### 6.1 Automated Checks

- `NarrativeQualityMetrics.potential_pii_count` tracks PII-like patterns
- `NarrativeContractValidator` enforces format and content rules
- `TrustReport` flags mock/empty content (preferred over bad generation)

### 6.2 Manual Review

Quarterly audit of:
- Random sample of 100 generated narratives
- All high-stakes persona outputs
- Any content flagged by automated systems

### 6.3 User Reporting

Users can report fabricated content via:
- GitHub issues
- Direct contact (for production deployments)
- Automated feedback pipeline (if integrated)

---

## 7. Exceptions

**No exceptions** are granted for:
- Adding real PII for "realism"
- Hardcoding specific institutions for "accuracy"
- Using actual addresses for "geographic precision"

**The only acceptable output for unavailable data is empty string or mock status.**

---

## 8. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-30 | Initial policy |

---

## 9. Related Documents

- `SCIENTIFIC_ROBUSTNESS_ASSESSMENT.md` - Data provenance methodology
- `TRUST_REPORT.md` - Quality metrics and gating
- `NARRATIVE_CONTRACT.md` - LLM response format specifications
