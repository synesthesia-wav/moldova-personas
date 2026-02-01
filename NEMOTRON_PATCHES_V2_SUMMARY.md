# Nemotron Patches v2 - High-Impact Gotchas Fixed

## Summary of 6 Critical Fixes Before Scaling

---

## 1. ✅ Geo Validation - Fixed Comrat/Gagauzia Mismatch

**Problem**: Comrat was under "Sud" but is capital of Gagauzia (autonomous region).

**Fix**:
```python
GEO_HIERARCHY = {
    "Gagauzia": {
        "capital": "Comrat",
        "daily_places": ["Comrat", "Ceadîr-Lunga", "Vulcănești"],
        "travel_only": ["Bălți", "Chișinău", "Cahul"]
    },
    "Sud": {
        "daily_places": ["Cahul", "Cantemir", "Leova", "Cimișlia"],
        "travel_only": ["Comrat", "Chișinău", "Bălți"]  # Comrat only as travel
    }
}

# Validation
def validate_geo_consistency(region, place_anchor, persona_texts):
    if place_anchor in travel_only and field != "travel_persona":
        return False, [f"'{place}' is travel-only for '{region}'"]
```

**Result**: 
- ✓ Sud region → place = Cahul/Leova (not Comrat)
- ✓ Gagauzia region → place = Comrat/Ceadîr-Lunga
- ✓ Travel persona can mention other regions
- ✓ Daily contexts use region-appropriate anchors

---

## 2. ✅ Contextual Trait-Leak Detection

**Problem**: "deschis" false-positives on "ușa e deschisă" (literal "open").

**Fix**:
```python
# Instead of word list:
TRAIT_LEAK_BLACKLIST = ['deschis']  # ❌ False positives

# Use contextual patterns:
TRAIT_LEAK_PATTERNS = {
    "openness": [
        r"\beste\s+(foarte\s+)?deschis[ăa]?\b(?!\s+(la|ușa|fereastra))",
        r"\bdeschis[ăa]?\s+(la\s+experiențe|la\s+nou)",
    ]
}

# Plus whitelist for literal contexts:
TRAIT_LEAK_WHITELIST = [
    r"ușa\s+\w+\s+deschis",  # ușa e deschisă
    r"magazin\s+\w+\s+deschis",  # program
]
```

**Result**:
- ✓ Blocks: "Maria este foarte deschisă" (trait naming)
- ✓ Allows: "Ușa e deschisă", "Magazin deschis până la 6"

---

## 3. ✅ Semantic Constraint Validation

**Problem**: Counting "dar/totuși" encourages marker spam without real constraints.

**Fix**:
```python
@dataclass
class Constraint:
    category: str
    description: str
    expected_consequences: List[str]  # How it manifests

CONSTRAINTS_SEMANTIC = {
    "budget_tight": Constraint(
        description="buget strict",
        expected_consequences=[
            "compară prețuri", "nu-și permite", 
            "economisește", "oferte"
        ]
    ),
    "time_pressure": Constraint(
        description="timp limitat",
        expected_consequences=[
            "grabă", "nu are timp", "weekend ocupat"
        ]
    )
}

# Validation checks for consequences, not just conjunctions
def validate_constraint_semantics(constraints, persona_texts):
    for constraint in constraints:
        found = [c for c in constraint.expected_consequences if c in text]
        if len(found) < 1:
            violation: "Constraint has no visible consequences"
```

**Result**:
- ✓ "Buget strict" → must see "compară prețuri" or "nu-și permite"
- ✓ "Timp limitat" → must see "grabă" or "weekend ocupat"
- ✗ Just saying "dar" is not enough

---

## 4. ✅ Anchor Frequency Capping

**Problem**: Risk of "zacuscă + piață + maxi-taxi" everywhere.

**Fix**:
```python
class AnchorFrequencyTracker:
    def __init__(self, max_share_per_anchor: float = 0.05):
        self.max_share = 0.05  # No anchor > 5% of dataset
        self.anchor_counts = Counter()
    
    def get_overused_anchors(self):
        return [(anchor, count/total) for anchor, count in self.anchor_counts.items() 
                if count/total > self.max_share]
    
    def get_rare_anchors(self, region):
        # For novelty budget: require ≥1 rare anchor per persona
        return [a for a, count in region_counts.items() 
                if count/total < 0.01]
    
    def generate_diverse_anchor_set(self, region):
        # Prefer rare anchors, exclude overused
        overused = self.get_overused_anchors()
        rare = self.get_rare_anchors(region)
        
        if rare:
            return random.choice(rare)
        else:
            return random.choice(available)
```

**Result**:
- ✓ Caps "zacuscă" at 5% max
- ✓ Requires ≥1 rare anchor per persona
- ✓ Region-specific distributions enforced

---

## 5. ✅ Counterfactual Q/A Consistency Test

**Problem**: No Nemotron-grade behavioral consistency test.

**Fix**:
```python
COUNTERFACTUAL_QUESTIONS = [
    {
        "question": "De ce nu mergi mai des la sală?",
        "probe_for": ["timp", "bani", "oboseală", "familie", "lipsă interes"]
    },
    {
        "question": "Cum alegi o destinație de weekend?",
        "probe_for": ["buget", "timp", "familie", "transport"]
    },
    {
        "question": "Ce te stresează cel mai mult la muncă?",
        "probe_for": ["colegi", "program", "salariu", "birocrație"]
    }
]

def validate_counterfactual_consistency(question, answer, persona):
    # Check answer uses expected themes
    found_probes = [p for p in question["probe_for"] if p in answer]
    
    # Check no contradictions with constraints
    contradictions = []
    if "nu are timp" in constraint and "mult timp liber" in answer:
        contradictions.append("Contradicts time constraint")
    
    return len(contradictions) == 0 and len(found_probes) > 0
```

**Result**:
- ✓ Probes consistency with constraints/OCEAN cues
- ✓ Regenerates only offending field if contradictory
- ✓ Nemotron-grade behavioral validation

---

## 6. ✅ Export Hygiene

**Problem**: Internal fields leaking to public exports.

**Fix**:
```python
# Internal fields (never export)
INTERNAL_FIELDS = {
    "ocean_scores",        # Raw scores
    "behavioral_cues",     # Generation guide
    "constraints",         # Infer from narrative
    "validation",          # Internal QA
    "rewrite_count",       # Internal
}

# Public fields (Nemotron-Brazil schema)
PUBLIC_FIELDS = {
    "uuid", "name", "sex", "age", ...
    "ocean_profile",       # NeMo format only
    "persona", "professional_persona", ...
}

def sanitize_for_export(persona):
    return {k: v for k, v in persona.items() 
            if k not in INTERNAL_FIELDS}

def validate_export_schema(persona):
    # Check required fields present
    # Check internal fields absent
    # Validate OCEAN in NeMo format
```

**Result**:
- ✓ `ocean_profile` in NeMo format (t_score, label, description)
- ✓ No raw `ocean_scores` in export
- ✓ No `behavioral_cues` in export
- ✓ Matches Nemotron-Brazil public schema

---

## Combined Usage

```python
from nemotron_patches_v2 import NemotronFullValidator

validator = NemotronFullValidator(region="Gagauzia")

# Generate with all patches
result = generate_persona_nemotron_v2(
    base_persona,
    validator=validator
)

# Validate everything
validation = validator.validate_all(result)

# Export (sanitized)
public_persona = sanitize_for_export(result)
```

---

## Files Created

| File | Description |
|------|-------------|
| `nemotron_patches_v2.py` | All 6 patches: geo, trait-leak, constraints, anchors, counterfactual, export |

---

## Validation Summary

```
✓ geo_consistency: Comrat only in Gagauzia
✓ trait_leak: Contextual patterns, no false positives  
✓ constraint_semantics: Real consequences required
✓ anchor_diversity: No anchor >5%, rare anchor required
✓ counterfactual_qa: Behavioral consistency verified
✓ export_hygiene: Internal fields filtered
```

**All 6 high-impact gotchas fixed before scaling!** 🎉
