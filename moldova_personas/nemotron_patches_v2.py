"""
Nemotron patches v2 - addressing high-impact gotchas before scaling.

1. Geo validation (Comrat ∈ Gagauzia only)
2. Contextual trait-leak detection (not just word list)
3. Semantic constraint validation (not just conjunction frequency)
4. Anchor frequency capping (prevent zacuscă-everywhere)
5. Counterfactual Q/A consistency test
6. Export hygiene (internal fields filtering)
"""

import random
import re
import json
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict


# ============================================================================
# 1. GEO VALIDATION - Fix Comrat/Gagauzia mismatch
# ============================================================================

# Administrative unit mapping
GEO_HIERARCHY = {
    "Gagauzia": {
        "capital": "Comrat",
        "places": ["Comrat", "Ceadîr-Lunga", "Vulcănești", "Bălți (doar călătorie)"],
        "daily_places": ["Comrat", "Ceadîr-Lunga", "Vulcănești"],  # Can be routine anchors
        "travel_only": ["Bălți", "Chișinău", "Cahul"]  # Only in travel_persona
    },
    "Chisinau": {
        "places": ["Botanica", "Râșcani", "Ciocana", "Buiucani", "Centru", 
                   "Valea Morilor", "piața Centrală", "scara blocului"],
        "daily_places": ["Botanica", "Râșcani", "Ciocana", "Buiucani", "Centru"],
        "travel_only": ["Comrat", "Bălți", "Cahul"]
    },
    "Nord": {
        "places": ["Bălți", "Bălți (piața)", "Soroca", "Drochia", "Fălești", 
                   "Glodeni", "Râbnița", "Nisporeni"],
        "daily_places": ["Bălți", "Soroca", "Drochia", "Fălești", "Glodeni"],
        "travel_only": ["Comrat", "Chișinău", "Cahul"]
    },
    "Centru": {
        "places": ["Ungheni", "Strășeni", "Hâncești", "Ialoveni", "Călărași",
                   "Orhei", "Criuleni", "Nistru", "piața din Ungheni"],
        "daily_places": ["Ungheni", "Strășeni", "Hâncești", "Ialoveni", "Călărași"],
        "travel_only": ["Comrat", "Bălți", "Chișinău"]
    },
    "Sud": {
        "places": ["Cahul", "Cahul (piața)", "Cantemir", "Leova", "Cimișlia",
                   "Basarabeasca", "Vulcănești (Sud)", "lacul de acumulare"],
        "daily_places": ["Cahul", "Cantemir", "Leova", "Cimișlia", "Basarabeasca"],
        "travel_only": ["Comrat", "Chișinău", "Bălți"]
    }
}


def get_valid_anchors_for_region(region: str) -> Dict[str, List[str]]:
    """Get anchors valid for daily routine in this region."""
    region_key = region
    if region in ["Chisinau", "Chişinău"]:
        region_key = "Chisinau"
    elif "Gagauzia" in region or "Găgăuzia" in region or "UTA" in region:
        region_key = "Gagauzia"
    
    return GEO_HIERARCHY.get(region_key, GEO_HIERARCHY["Centru"])


def validate_geo_consistency(
    region: str,
    place_anchor: str,
    persona_texts: Dict[str, str]
) -> Tuple[bool, List[str]]:
    """
    Validate that anchors match region.
    
    Returns:
        (valid, list of violations)
    """
    violations = []
    geo_data = get_valid_anchors_for_region(region)
    
    # Check daily anchor
    daily_places = geo_data.get("daily_places", [])
    travel_only = geo_data.get("travel_only", [])
    
    # If place_anchor is in travel_only list, it's a violation for daily context
    if place_anchor in travel_only:
        violations.append(
            f"Place '{place_anchor}' is travel-only for region '{region}', "
            f"not valid as daily routine anchor"
        )
    
    # Check persona texts for misplaced anchors
    for field_name, text in persona_texts.items():
        if field_name == "travel_persona":
            continue  # Travel persona can mention anywhere
        
        # Check if text mentions places from other regions as daily routine
        for other_region, other_geo in GEO_HIERARCHY.items():
            if other_region == region or "Gagauzia" in region and "Gagauzia" in other_region:
                continue
            
            other_capital = other_geo.get("capital", "")
            if other_capital and other_capital in text:
                # Allow if explicitly as travel
                if field_name != "travel_persona":
                    violations.append(
                        f"Field '{field_name}' mentions '{other_capital}' as daily context, "
                        f"but it's not in {region}"
                    )
    
    return len(violations) == 0, violations


# ============================================================================
# 2. CONTEXTUAL TRAIT-LEAK DETECTION
# ============================================================================

# Patterns that indicate personality trait naming (not just word presence)
TRAIT_LEAK_PATTERNS = {
    "openness": [
        r"\beste\s+(mai\s+)?(foarte\s+)?deschis[ăa]?\b(?!\s+(la|spre|ușa|fereastra))",
        r"\bdeschis[ăa]?\s+(la\s+experiențe|la\s+nou|la\s+idei)",
        r"\bscor\s+(de\s+)?\bdeschidere",
        r"\bnivel\s+(de\s+)?\bopenness",
    ],
    "conscientiousness": [
        r"\beste\s+(mai\s+)?(foarte\s+)?conștiincios[ăa]?\b",
        r"\bconștiincios[ăa]?\s+(în|la)\s+munc",
        r"\bscor\s+(de\s+)?\bconștiinciozitate",
        r"\bconscientiousness\b",
    ],
    "extraversion": [
        r"\beste\s+(mai\s+)?(foarte\s+)?extrovertit[ăa]?\b",
        r"\bextrovertit[ăa]?\s+și",
        r"\bscor\s+(de\s+)?\bextraversiune",
        r"\bextraversion\b",
    ],
    "agreeableness": [
        r"\beste\s+(mai\s+)?(foarte\s+)?agreabil[ăa]?\b",
        r"\bagreabil[ăa]?\s+(cu|pentru)",
        r"\bscor\s+(de\s+)?\bagreabilitate",
        r"\bagreeableness\b",
    ],
    "neuroticism": [
        r"\beste\s+(mai\s+)?(foarte\s+)?nevrotic[ăa]?\b",
        r"\bscor\s+(de\s+)?\bneuroticism",
        r"\bneuroticism\s+(ridicat|scăzut)",
    ],
    "framework": [
        r"\b(big\s+five|ocean|oceam)\b",
        r"\bt_score\b",
        r"\bprocentil\b",
        r"\bpersonalitate\s+oceam\b",
        r"\btrăsătura\s+de\s+personalitate\b",
    ]
}

# Whitelist: contexts where banned words are allowed
TRAIT_LEAK_WHITELIST = [
    r"ușa\s+\w+\s+deschis",  # ușa e deschisă
    r"fereastra\s+\w+\s+deschis",
    r"magazin\s+\w+\s+deschis",
    r"deschis\s+la\s+ora",   # program deschis la ora X
    r"deschis\s+pentru",     # deschis pentru public
]


def validate_trait_leak_contextual(text: str) -> Tuple[bool, List[Dict]]:
    """
    Contextual trait-leak detection with whitelist support.
    
    Returns:
        (passed, list of violations with context)
    """
    violations = []
    text_lower = text.lower()
    
    # First check whitelist
    for pattern in TRAIT_LEAK_WHITELIST:
        if re.search(pattern, text_lower):
            # This is an allowed usage, skip further checks
            return True, []
    
    # Check each trait pattern
    for trait, patterns in TRAIT_LEAK_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                # Get context (50 chars before and after)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                violations.append({
                    "trait": trait,
                    "pattern": pattern,
                    "matched_text": match.group(),
                    "context": context,
                    "position": match.start()
                })
    
    return len(violations) == 0, violations


# Optional: LLM judge for ambiguous cases
TRAIT_LEAK_LLM_PROMPT = """Evaluează dacă textul NUMEȘTE EXPLICIT trăsături de personalitate.

TEXT: {text}

Întrebări:
1. Textul folosește cuvinte precum "deschis", "conștiincios", "extrovertit", 
   "agreabil", "nevrotic" pentru a DESCRIE PERSONALITATEA unei persoane?
2. Textul menționează "scor", "t_score", "Big Five", "OCEAN"?
3. Textul etichetează pe cineva cu o trăsătură de personalitate?

Răspunde JSON:
{{
  "names_personality": true/false,
  "confidence": 0-1,
  "explanation": "..."
}}

IMPORTANT: Distinge între:
- "Maria este deschisă" (NUMEȘTE trăsătură) → true
- "Ușa e deschisă" (literal) → false
- "Magazin deschis până la 6" (program) → false"""


# ============================================================================
# 3. SEMANTIC CONSTRAINT VALIDATION
# ============================================================================

@dataclass
class Constraint:
    """A realistic constraint with semantic validation."""
    category: str  # budget, time_pressure, health, family, etc.
    description: str
    expected_consequences: List[str]  # How this should manifest in personas


CONSTRAINTS_SEMANTIC = {
    "budget_tight": Constraint(
        category="budget",
        description="buget strict, compară prețuri",
        expected_consequences=[
            "evită cheltuieli",
            "compară prețuri",
            "nu-și permite",
            "economisește",
            "oferte",
            "ieftin"
        ]
    ),
    "time_pressure": Constraint(
        category="time",
        description="timp limitat între job și familie",
        expected_consequences=[
            "grabă",
            "nu are timp",
            "seara târziu",
            "weekend ocupat",
            "eficiență",
            "multitasking"
        ]
    ),
    "family_duty": Constraint(
        category="family",
        description="responsabil de nepoți frecvent",
        expected_consequences=[
            "nepoți",
            "bunici",
            "familie",
            "copii",
            "responsabilitate",
            "grijă"
        ]
    ),
    "commute_burden": Constraint(
        category="commute",
        description="navetează 1+ ore zilnic",
        expected_consequences=[
            "maxi-taxi",
            "autobuz",
            "drum lung",
            "oboseală",
            "timp pierdut",
            "trafic"
        ]
    ),
    "health_nuisance": Constraint(
        category="health",
        description="dureri de spate ocazionale",
        expected_consequences=[
            "spate",
            "durere",
            "masaj",
            "repaus",
            "evită efort",
            "poziție"
        ]
    ),
}


def validate_constraint_semantics(
    constraints: List[str],
    persona_texts: Dict[str, str]
) -> Tuple[bool, List[Dict]]:
    """
    Validate that constraints have actual consequences in personas.
    
    Returns:
        (valid, list of missing consequences)
    """
    violations = []
    all_text = " ".join(persona_texts.values()).lower()
    
    for constraint_desc in constraints:
        # Find matching constraint definition
        matched_constraint = None
        for key, constraint in CONSTRAINTS_SEMANTIC.items():
            if any(word in constraint_desc.lower() for word in constraint.description.lower().split()[:3]):
                matched_constraint = constraint
                break
        
        if not matched_constraint:
            continue
        
        # Check if any expected consequence appears
        found_consequences = [
            cons for cons in matched_constraint.expected_consequences
            if cons in all_text
        ]
        
        if len(found_consequences) < 1:
            violations.append({
                "constraint": constraint_desc,
                "category": matched_constraint.category,
                "expected_consequences": matched_constraint.expected_consequences,
                "found": found_consequences,
                "issue": f"Constraint '{constraint_desc}' has no visible consequences in persona texts"
            })
    
    return len(violations) == 0, violations


CONSTRAINT_SEMANTICS_LLM_PROMPT = """Evaluează dacă constrângerile se manifestă realist în descrieri.

CONSTRÂNGERI:
{constraints}

DESCRIERI PERSONA:
{personas}

Întrebări (scor 0-5 pentru fiecare):
1. Constrângerile sunt vizibile în comportament? (nu doar menționate)
2. Constrângerile schimbă alegerile/activitățile?
3. Persoana pare să accepte/negocieze cu limitările?

Răspunde JSON:
{{
  "constraint_visibility": 0-5,
  "behavioral_impact": 0-5,
  "acceptance_realism": 0-5,
  "overall": 0-5,
  "feedback": "..."
}}"""


# ============================================================================
# 4. ANCHOR FREQUENCY CAPPING
# ============================================================================

class AnchorFrequencyTracker:
    """Track anchor usage to prevent collapse into zacuscă-everywhere."""
    
    def __init__(self, max_share_per_anchor: float = 0.05):
        self.max_share = max_share_per_anchor
        self.anchor_counts = Counter()
        self.total_records = 0
        self.region_anchor_counts = defaultdict(Counter)
    
    def record_usage(self, anchors: Dict[str, str], region: str):
        """Record anchor usage for a persona."""
        self.total_records += 1
        
        # Count each anchor type
        for anchor_type, anchor_value in anchors.items():
            self.anchor_counts[anchor_value] += 1
            self.region_anchor_counts[region][anchor_value] += 1
    
    def get_overused_anchors(self) -> List[Tuple[str, float]]:
        """Get anchors that exceed max_share threshold."""
        overused = []
        for anchor, count in self.anchor_counts.most_common():
            share = count / max(self.total_records, 1)
            if share > self.max_share:
                overused.append((anchor, share))
        return overused
    
    def get_rare_anchors(self, region: str, min_rarity: float = 0.01) -> List[str]:
        """Get anchors that are underused (good for novelty budget)."""
        rare = []
        region_counts = self.region_anchor_counts[region]
        
        for anchor, count in region_counts.items():
            share = count / max(self.total_records, 1)
            if share < min_rarity:
                rare.append(anchor)
        
        return rare
    
    def generate_diverse_anchor_set(
        self,
        region: str,
        require_rare: bool = True
    ) -> Dict[str, str]:
        """Generate anchors ensuring diversity."""
        from .nemotron_feel_patch import get_anchors_for_region
        
        anchors = get_anchors_for_region(region, "")
        all_food = anchors.get("food", []) + anchors.get("routine", [])
        all_places = anchors.get("places", [])
        
        # Filter out overused
        overused = {a for a, _ in self.get_overused_anchors()}
        available_food = [a for a in all_food if a not in overused] or all_food
        available_places = [a for a in all_places if a not in overused] or all_places
        
        # Prefer rare anchors
        if require_rare:
            rare = self.get_rare_anchors(region)
            rare_food = [a for a in rare if a in available_food]
            rare_places = [a for a in rare if a in available_places]
            
            if rare_food:
                food = random.choice(rare_food)
            else:
                food = random.choice(available_food)
            
            if rare_places:
                place = random.choice(rare_places)
            else:
                place = random.choice(available_places)
        else:
            food = random.choice(available_food)
            place = random.choice(available_places)
        
        return {"place": place, "routine": food}


# ============================================================================
# 5. COUNTERFACTUAL Q/A CONSISTENCY TEST
# ============================================================================

COUNTERFACTUAL_QUESTIONS = [
    {
        "question": "De ce nu mergi mai des la sală/sport?",
        "probe_for": ["timp", "bani", "oboseală", "sănătate", "familie", "lipsă interes"]
    },
    {
        "question": "Cum alegi o destinație de weekend?",
        "probe_for": ["buget", "timp", "familie", "transport", "interese", "companions"]
    },
    {
        "question": "Ce te stresează cel mai mult la muncă?",
        "probe_for": ["colegi", "program", "salariu", "trafic", "responsabilități", "birocrație"]
    },
    {
        "question": "De ce nu socializezi mai mult cu vecinii?",
        "probe_for": ["timp", "personalitate", "diferențe", "preferință", "familie"]
    },
    {
        "question": "Ce faci când ai o zi proastă?",
        "probe_for": ["singurătate", "familie", "activități", "mâncare", "somn", "plâns"]
    }
]


def generate_counterfactual_prompt(
    question: str,
    persona_data: Dict
) -> str:
    """Generate prompt for counterfactual Q/A."""
    return f"""Pe baza descrierii de mai jos, răspunde la întrebare ÎN PERSONA personajului.

DESPRE {persona_data['name']}:
{persona_data['context']['cultural_background'][:300]}

Personalitate (comportament observabil):
{persona_data.get('behavioral_cues', '')}

Constrângeri:
{chr(10).join('- ' + c for c in persona_data.get('constraints', []))}

ÎNTREBARE: {question}

RĂSPUNS SCURT (1-2 propoziții) în persoana lui {persona_data['name']}:
"""


def validate_counterfactual_consistency(
    question_data: Dict,
    answer: str,
    persona_data: Dict
) -> Tuple[bool, float, str]:
    """
    Validate counterfactual answer is consistent with persona.
    
    Returns:
        (consistent, confidence, explanation)
    """
    # Check if answer mentions expected themes
    answer_lower = answer.lower()
    
    # Score based on expected probes
    found_probes = [
        probe for probe in question_data["probe_for"]
        if probe in answer_lower
    ]
    
    # Also check for direct contradictions with constraints
    constraints = persona_data.get("constraints", [])
    contradictions = []
    
    for constraint in constraints:
        # If answer says opposite of constraint, that's a problem
        if "nu are timp" in constraint.lower() and "mult timp liber" in answer_lower:
            contradictions.append(f"Answer contradicts constraint: {constraint}")
        if "buget strict" in constraint.lower() and "cheltuiește mult" in answer_lower:
            contradictions.append(f"Answer contradicts constraint: {constraint}")
    
    if contradictions:
        return False, 0.0, "; ".join(contradictions)
    
    # Score based on probe coverage
    probe_coverage = len(found_probes) / len(question_data["probe_for"])
    
    return probe_coverage > 0.2, probe_coverage, f"Found {len(found_probes)} relevant themes"


# ============================================================================
# 6. EXPORT HYGIENE - Filter internal fields
# ============================================================================

# Fields to exclude from public exports
INTERNAL_FIELDS = {
    "ocean_scores",           # Raw scores
    "ocean_raw_scores",       # Duplicate
    "behavioral_cues",        # Internal generation guide
    "constraints",            # Can be inferred from narrative
    "validation",             # Internal QA
    "passed_validation",      # Internal QA
    "rewrite_count",          # Internal
    "ocean_deviation_score",  # Internal
    "repair_history",         # Internal
}

# Fields to include in public exports (Nemotron-Brazil schema)
PUBLIC_FIELDS = {
    "uuid",
    "name",
    "sex",
    "age",
    "age_group",
    "marital_status",
    "education_level",
    "occupation",
    "municipality",
    "state",
    "locality",
    "country",
    "ethnicity",
    "religion",
    "ocean_profile",          # NeMo format (t_score, label, description)
    "persona",
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "cultural_background",
    "skills_and_expertise",
    "skills_and_expertise_list",
    "hobbies_and_interests",
    "hobbies_and_interests_list",
    "career_goals_and_ambitions",
}


def sanitize_for_export(persona: Dict) -> Dict:
    """Remove internal fields from persona for public export."""
    return {
        k: v for k, v in persona.items()
        if k not in INTERNAL_FIELDS and not k.startswith("_")
    }


def validate_export_schema(persona: Dict) -> Tuple[bool, List[str]]:
    """
    Validate that exported persona matches Nemotron-Brazil schema.
    
    Returns:
        (valid, list of issues)
    """
    issues = []
    
    # Check required fields
    for field in PUBLIC_FIELDS:
        if field not in persona:
            issues.append(f"Missing required field: {field}")
    
    # Check for internal field leakage
    for field in INTERNAL_FIELDS:
        if field in persona:
            issues.append(f"Internal field leaked to export: {field}")
    
    # Validate OCEAN profile format
    if "ocean_profile" in persona:
        ocean = persona["ocean_profile"]
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            if trait not in ocean:
                issues.append(f"OCEAN profile missing trait: {trait}")
            elif not isinstance(ocean[trait], dict):
                issues.append(f"OCEAN trait {trait} not in NeMo format (should be dict)")
            elif "t_score" not in ocean[trait]:
                issues.append(f"OCEAN trait {trait} missing t_score")
    
    return len(issues) == 0, issues


# ============================================================================
# COMBINED VALIDATOR
# ============================================================================

class NemotronFullValidator:
    """Combined validator with all v2 patches."""
    
    def __init__(self, region: str = "Centru"):
        self.region = region
        self.geo_data = get_valid_anchors_for_region(region)
        self.anchor_tracker = AnchorFrequencyTracker()
    
    def validate_all(
        self,
        persona: Dict,
        require_counterfactual: bool = False
    ) -> Dict:
        """Run full validation suite."""
        results = {
            "passed": True,
            "checks": {},
            "repairs": []
        }
        
        # 1. Geo validation
        geo_ok, geo_violations = validate_geo_consistency(
            self.region,
            persona.get("anchors", {}).get("place", ""),
            persona.get("personas", {})
        )
        results["checks"]["geo_consistency"] = {
            "passed": geo_ok,
            "violations": geo_violations
        }
        if not geo_ok:
            results["passed"] = False
        
        # 2. Contextual trait leak
        all_text = " ".join(str(v) for v in persona.get("personas", {}).values())
        trait_ok, trait_violations = validate_trait_leak_contextual(all_text)
        results["checks"]["trait_leak"] = {
            "passed": trait_ok,
            "violations": trait_violations
        }
        if not trait_ok:
            results["passed"] = False
        
        # 3. Constraint semantics
        constraint_ok, constraint_violations = validate_constraint_semantics(
            persona.get("constraints", []),
            persona.get("personas", {})
        )
        results["checks"]["constraint_semantics"] = {
            "passed": constraint_ok,
            "violations": constraint_violations
        }
        if not constraint_ok:
            results["passed"] = False
        
        # 4. Export hygiene
        public_persona = sanitize_for_export(persona)
        export_ok, export_issues = validate_export_schema(public_persona)
        results["checks"]["export_hygiene"] = {
            "passed": export_ok,
            "issues": export_issues
        }
        if not export_ok:
            results["passed"] = False
        
        # Track anchor usage
        if "anchors" in persona:
            self.anchor_tracker.record_usage(
                persona["anchors"],
                self.region
            )
        
        # Check for anchor overuse
        overused = self.anchor_tracker.get_overused_anchors()
        results["checks"]["anchor_diversity"] = {
            "passed": len(overused) == 0,
            "overused_anchors": overused
        }
        
        return results
