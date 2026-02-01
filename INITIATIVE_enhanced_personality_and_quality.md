# Initiative: Enhanced Personality Modeling & Quality Scoring

**Project**: Moldova Synthetic Personas Generator  
**Date**: 2026-01-28  
**Status**: Proposed  
**Priority**: High  
**Estimated Effort**: 1-2 weeks

---

## Executive Summary

This initiative proposes two major enhancements to the Moldova personas pipeline, inspired by NVIDIA's Nemotron methodology:

1. **OCEAN Personality Model Integration**: Add Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) to each persona for richer psychological dimensionality.

2. **Multi-Attribute Quality Scoring**: Replace binary pass/fail validation with continuous 5-dimensional quality scoring, enabling threshold-based filtering similar to Nemotron's HelpSteer2 reward model approach.

---

## 1. OCEAN Personality Model Integration

### 1.1 Background

Current personas lack psychological depth. While Nemotron's marketing claims include "personality modeling (OCEAN)", our research confirmed this is **not actually implemented** in their published datasets. This presents an opportunity for differentiation.

### 1.2 Proposed Implementation

#### Option A: LLM-Generated OCEAN Scores (Recommended)

Add OCEAN scoring via existing LLM integration:

```python
# models.py - Add to Persona class
class Persona(BaseModel):
    # ... existing fields ...
    
    # OCEAN Big Five traits (0-100 scale)
    openness: int = Field(..., ge=0, le=100, description="Openness to Experience")
    conscientiousness: int = Field(..., ge=0, le=100, description="Conscientiousness")
    extraversion: int = Field(..., ge=0, le=100, description="Extraversion")
    agreeableness: int = Field(..., ge=0, le=100, description="Agreeableness")
    neuroticism: int = Field(..., ge=0, le=100, description="Neuroticism (Emotional Stability)")
```

**Generation Logic**:
- Sample base scores from population-normed distributions
- Apply conditional adjustments based on demographics:
  - Age correlates with reduced Neuroticism, increased Conscientiousness
  - Education correlates with increased Openness
  - Occupation provides trait constraints (e.g., accountants higher Conscientiousness)
- Use LLM to generate consistency narrative explaining trait combinations

**Validation Sources**:
- Cross-cultural Big Five research (Schmitt et al., 2007)
- Romanian population norms where available
- Moldova-specific cultural adjustments

#### Option B: Rule-Based Correlation Matrix

Define explicit correlations between demographic attributes and OCEAN traits:

```python
# ocean_model.py
OCEAN_CORRELATIONS = {
    'occupation': {
        'profesor': {'openness': +15, 'extraversion': +10},
        'inginer': {'conscientiousness': +20, 'openness': +10},
        'agricultor': {'conscientiousness': +15, 'extraversion': -10},
        # ... etc
    },
    'education_level': {
        'superioare': {'openness': +20, 'neuroticism': -10},
        'med_ii': {'conscientiousness': +10},
        # ... etc
    },
    'age': {
        '18-25': {'neuroticism': +15, 'conscientiousness': -10},
        '60+': {'neuroticism': -20, 'agreeableness': +15},
        # ... etc
    }
}
```

### 1.3 Usage Scenarios

- **Survey Simulation**: Personas with specific trait profiles can simulate how different personality types respond to questions
- **Behavioral Prediction**: Traits inform likelihood of certain behaviors (e.g., High Conscientiousness → retirement planning)
- **Marketing Segmentation**: Psychological profiles enable psychographic targeting

---

## 2. Multi-Attribute Quality Scoring

### 2.1 Current State

Existing validation (`validators.py`) uses binary pass/fail:
```python
if not self._validate_age_education(persona):
    errors.append(ValidationError(...))
```

**Problems**:
- No granularity: "barely passes" vs "excellent" are treated equally
- No quality ranking for downstream filtering
- Cannot filter top-X% of generated personas

### 2.2 Proposed Solution: QualityScorer Class

```python
# validators.py - New class
@dataclass
class QualityScore:
    """Multi-dimensional quality score for a persona."""
    demographic_consistency: float  # 0-1
    narrative_coherence: float      # 0-1
    statistical_alignment: float    # 0-1
    completeness: float             # 0-1
    age_education_alignment: float  # 0-1
    
    @property
    def overall(self) -> float:
        return sum([
            self.demographic_consistency,
            self.narrative_coherence,
            self.statistical_alignment,
            self.completeness,
            self.age_education_alignment
        ]) / 5
    
    def is_high_quality(self, threshold: float = 0.8) -> bool:
        return self.overall >= threshold


class QualityScorer:
    """
    Nemotron-style multi-attribute quality scoring.
    Replaces/supplements binary ValidationPipeline.
    """
    
    def score_persona(self, persona: Persona) -> QualityScore:
        return QualityScore(
            demographic_consistency=self._score_demographics(persona),
            narrative_coherence=self._score_narrative(persona),
            statistical_alignment=self._score_census_alignment(persona),
            completeness=self._score_field_coverage(persona),
            age_education_alignment=self._score_age_education(persona),
        )
    
    def _score_demographics(self, p: Persona) -> float:
        """Check internal consistency of demographics."""
        score = 1.0
        
        # Language-ethnicity mismatch penalty
        if p.ethnicity == 'moldovean' and p.mother_tongue == 'rusa':
            score -= 0.3  # Possible but less common
        
        # Region-city consistency
        if p.city not in CITIES_BY_REGION.get(p.region, []):
            score -= 0.5
        
        return max(0.0, score)
    
    def _score_narrative(self, p: Persona) -> float:
        """Score LLM-generated narrative quality (if present)."""
        if not p.narrative or p.narrative == "":
            return 1.0  # No narrative = no penalty
        
        # Check for required elements in narrative
        required = [p.name.split()[0], str(p.age), p.occupation]
        present = sum(1 for r in required if r.lower() in p.narrative.lower())
        return present / len(required)
    
    def _score_census_alignment(self, p: Persona) -> float:
        """How well does persona match target census distributions?"""
        # Compare persona attributes to expected distributions
        # Return 1.0 if perfectly representative, lower if outlier
        pass  # Implementation TBD
    
    def _score_field_coverage(self, p: Persona) -> float:
        """Completeness of populated fields."""
        required_fields = ['name', 'age', 'sex', 'ethnicity', 'education_level', 'occupation']
        populated = sum(1 for f in required_fields if getattr(p, f))
        return populated / len(required_fields)
    
    def _score_age_education(self, p: Persona) -> float:
        """Graduated scoring instead of binary pass/fail."""
        typical_age_at_education = {
            'primare': (6, 12),
            'gimnaziale': (12, 16),
            'liceale': (16, 19),
            'postliceale': (19, 22),
            'superioare': (22, 26),
            'masterat': (24, 28),
            'doctorat': (26, 35)
        }
        
        if p.education_level not in typical_age_at_education:
            return 1.0
        
        min_age, max_age = typical_age_at_education[p.education_level]
        
        # Graduated penalty instead of binary
        if min_age <= p.age <= max_age + 5:
            return 1.0
        elif min_age <= p.age <= max_age + 15:
            return 0.7  # Unusual but possible
        else:
            return 0.3  # Very unusual
```

### 2.3 Integration Points

#### In `generator.py`:
```python
def generate(self, count: int, quality_threshold: float = 0.0) -> List[Persona]:
    """
    Generate personas with optional quality filtering.
    
    Args:
        count: Target number of personas
        quality_threshold: Minimum overall quality score (0-1)
                          0.0 = no filtering (current behavior)
                          0.8 = Nemotron-style high quality only
    """
    scorer = QualityScorer()
    personas = []
    
    while len(personas) < count:
        candidate = self._generate_single()
        score = scorer.score_persona(candidate)
        
        if quality_threshold == 0.0 or score.overall >= quality_threshold:
            candidate.quality_score = score  # Store for export/analysis
            personas.append(candidate)
    
    return personas
```

#### In `cli.py`:
```python
gen_parser.add_argument('--quality-threshold', type=float, default=0.0,
                      help='Minimum quality score (0-1). 0.8 = high quality only')
```

### 2.4 Comparison to Nemotron

| Aspect | Nemotron | Our Implementation |
|--------|----------|-------------------|
| Scoring Model | Learned reward model (HelpSteer2) | Rule-based heuristic |
| Dimensions | 5 (Helpfulness, Correctness, Coherence, Complexity, Verbosity) | 5 (Demographic, Narrative, Statistical, Completeness, Age-Edu) |
| Training Data | 10k human preferences | None (hand-crafted rules) |
| Inference Cost | GPU required for 340B model | CPU-only, O(n) |
| Customizability | Fixed model weights | Fully customizable rules |

**Trade-off**: We sacrifice learned human preference alignment for simplicity, interpretability, and zero inference cost. This is appropriate for our use case.

---

## 3. Implementation Plan

### Phase 1: OCEAN Model (3-4 days)
- [ ] Research Romanian Big Five population norms
- [ ] Add OCEAN fields to `models.py`
- [ ] Implement `ocean_generator.py` with demographic correlations
- [ ] Add OCEAN narrative generation to LLM prompts
- [ ] Update exporters to include OCEAN fields

### Phase 2: Quality Scorer (2-3 days)
- [ ] Implement `QualityScore` dataclass
- [ ] Implement `QualityScorer` class with 5 dimensions
- [ ] Integrate into `PersonaGenerator.generate()`
- [ ] Add `--quality-threshold` CLI flag
- [ ] Add quality distribution to `stats` command

### Phase 3: Validation & Testing (2-3 days)
- [ ] Generate sample personas with OCEAN + quality scores
- [ ] Validate OCEAN distributions against research literature
- [ ] Benchmark generation rate with/without quality filtering
- [ ] Document correlation matrix methodology

---

## 4. Acceptance Criteria

- [ ] 100K personas can be generated with `--quality-threshold 0.8` in <10 minutes
- [ ] OCEAN scores show realistic population distributions (bell curves, appropriate means)
- [ ] Quality scores correlate with validation error rates (higher score = fewer errors)
- [ ] Export formats include OCEAN fields and quality scores
- [ ] Documentation includes rationale for scoring weights

---

## 5. Open Questions

1. **OCEAN Data Source**: Do we have access to Romanian/Moldovan Big Five normative data?
2. **Quality Weight Tuning**: How do we validate the 5-dimensional scoring weights?
3. **Narrative Integration**: Should OCEAN traits be mentioned explicitly in generated narratives?
4. **Privacy**: Do OCEAN scores (even synthetic) raise any privacy concerns for downstream users?

---

## References

- Nemotron-4 340B Technical Report (NVIDIA, 2024)
- HelpSteer2: Multi-attribute Reward Model Training (NVIDIA, 2024)
- Schmitt et al. (2007): "The Geographic Distribution of Big Five Personality Traits"
- Costa & McCrae (1992): "Revised NEO Personality Inventory"

---

**Next Steps**: Review this initiative, prioritize phases, assign implementation resources.

**Document Version**: 1.0  
**Last Updated**: 2026-01-28
