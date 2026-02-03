"""
Persona "feel" patches - prompt + validator improvements.

Addresses 4 deltas:
1. Make OCEAN implicit (ban trait words; require behavioral evidence)
2. Kill repetitive openings with constrained variation
3. De-sanitize: enforce friction, constraints, and mild negatives
4. Strengthen Moldova anchors without becoming "tourism brochure"
"""

import random
import re
import json
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


# ============================================================================
# 1. TRAIT-LEAK BLACKLIST VALIDATOR
# ============================================================================

TRAIT_LEAK_BLACKLIST = {
    # Direct trait words
    'deschis', 'deschisa', 'deschisă',
    'conștiincios', 'conștiincioasă', 'constiincios',
    'extrovertit', 'extrovertită', 'extravert',
    'agreabil', 'agreabilă',
    'nevrotic', 'nevrotică',
    'trăsătură', 'trăsături',
    'personalitate', 'tipar',
    # Framework references
    'big five', 'ocean', 'oceam',
    'scor', 't_score', 't-score', 'tscore',
    'procentil', 'percentil',
    # Direct adjective forms
    'neurotic', 'neurotica', 'neurotică',
    'conscientios', 'constiincioasa',
}


def validate_trait_leak(text: str) -> Tuple[bool, List[str]]:
    """
    Check if text contains banned trait words.
    
    Returns:
        (passed, found_words)
    """
    text_lower = text.lower()
    found = []
    
    for word in TRAIT_LEAK_BLACKLIST:
        if word in text_lower:
            found.append(word)
    
    return len(found) == 0, found


# ============================================================================
# 2. BEHAVIORAL CUES GENERATOR (Stage B hidden field)
# ============================================================================

@dataclass
class BehavioralCues:
    """Behavioral cues derived from OCEAN (not exported, guides generation)."""
    planning_horizon: str  # short/medium/long-term
    routine_tolerance: str  # rigid/flexible/chaotic
    social_initiation: str  # proactive/selective/avoidant
    conflict_style: str  # direct/accommodating/avoidant/compromising
    novelty_budget: str  # explorer/curious/balanced/traditional
    stress_signals: List[str]  # how stress manifests
    decision_cues: List[str]  # how they make decisions
    social_energy: str  # gregarious/outgoing/selective/reserved
    
    def to_prompt(self) -> str:
        """Convert to prompt-ready format (no trait words)."""
        return f"""COMPORTAMENT OBSERVABIL (fără a menționa trăsături de personalitate):

- Planificare: {self.planning_horizon}
- Rutină: {self.routine_tolerance}
- Inițiere socială: {self.social_initiation}
- Stil conflict: {self.conflict_style}
- Noutate: {self.novelty_budget}
- Semnale stres: {', '.join(self.stress_signals)}
- Decizii: {', '.join(self.decision_cues)}
- Energie socială: {self.social_energy}
"""


def generate_behavioral_cues(ocean_scores: Dict[str, int]) -> BehavioralCues:
    """Generate behavioral cues from OCEAN scores (no trait words)."""
    O = ocean_scores.get('openness', 50)
    C = ocean_scores.get('conscientiousness', 50)
    E = ocean_scores.get('extraversion', 50)
    A = ocean_scores.get('agreeableness', 50)
    N = ocean_scores.get('neuroticism', 50)
    
    # Planning horizon (C-based)
    if C >= 65:
        planning = "planifică cu săptămâni/luni în avans, liste și reminder-e"
    elif C >= 45:
        planning = "planifică câteva zile înainte, dar adaptabil"
    else:
        planning = "decide spontan, preferă flexibilitate"
    
    # Routine tolerance (C-based)
    if C >= 65:
        routine = "rutină strictă, aceleași ore pentru masă/somn"
    elif C >= 45:
        routine = "rutină generală cu excepții ocazionale"
    else:
        routine = "fără rutină fixă, variază de la zi la zi"
    
    # Social initiation (E-based)
    if E >= 65:
        social_init = "contactează primul, propune ieșiri, inițiază conversații"
    elif E >= 45:
        social_init = "răspunde invitațiilor, mai rar inițiază"
    else:
        social_init = "așteaptă să fie invitat, preferă să fie abordat"
    
    # Conflict style (A + N based)
    if A >= 65 and N <= 50:
        conflict = "caută compromis, evită confruntarea directă"
    elif A <= 40 and E >= 60:
        conflict = "spune direct ce gândește, fără ocolișuri"
    elif N >= 60:
        conflict = "retrage-se din conflict, ruminează după"
    else:
        conflict = "negociază pragmatic, dă și ia"
    
    # Novelty budget (O + E based)
    score = O + E - C
    if score >= 80:
        novelty = "încearcă constant lucruri noi, se plictisește rapid de rutină"
    elif score >= 40:
        novelty = "deschis la noutate, dar cu măsură"
    elif score >= 0:
        novelty = "echilibrat între vechi și nou"
    else:
        novelty = "preferă ce-i familiar, schimbare doar când e necesar"
    
    # Stress signals (N-based)
    stress = []
    if N >= 60:
        stress.extend(["insomnie ocazională", "ruminație", "întreabă pe alții dacă totul e ok"])
    elif N >= 40:
        stress.extend(["se plânge de oboseală", "caută activități relaxante"])
    else:
        stress.extend(["păstrează calm aparent", "rezolvă pas cu pas"])
    
    if C >= 60 and N >= 50:
        stress.append("micromanagement când e stresat")
    
    # Decision cues (C + O based)
    decisions = []
    if C >= 60:
        decisions.append("cântărește pro/contra")
    else:
        decisions.append("decide intuitiv")
    
    if O >= 60:
        decisions.append("întreabă părerea altora")
    else:
        decisions.append("se bazează pe experiență proprie")
    
    # Social energy (E-based)
    if E >= 70:
        energy = "câștigă energie din interacțiuni, petrece mult timp cu alții"
    elif E >= 50:
        energy = "echilibrat, socializează dar are nevoie și de timp singur"
    elif E >= 30:
        energy = "socializare în doze mici, apoi are nevoie de liniște"
    else:
        energy = "interacțiunile sociale îl/o obosesc, preferă solitudinea"
    
    return BehavioralCues(
        planning_horizon=planning,
        routine_tolerance=routine,
        social_initiation=social_init,
        conflict_style=conflict,
        novelty_budget=novelty,
        stress_signals=stress[:3],
        decision_cues=decisions,
        social_energy=energy
    )


# ============================================================================
# 3. CONSTRAINED OPENING VARIATION
# ============================================================================

OPENING_MOVES = [
    "situational_hook",      # "După turele lungi..."
    "preference_statement",  # "Preferă..." / "Îi place..."
    "micro_story",          # "Sâmbăta trecută..."
    "contrast",             # "Deși..."
    "value",                # "Ține mult la..."
    "sensory_anchor",       # "Mirosul de..."
]


def validate_opening_variation(personas: List[str]) -> Tuple[bool, List[int]]:
    """
    Check that 6 persona fields don't share opening n-grams.
    
    Returns:
        (passed, list of failing indices)
    """
    # Extract first 6-10 tokens from each
    prefixes = []
    for i, text in enumerate(personas):
        words = text.split()[:8]  # First ~8 words
        prefix = ' '.join(words).lower()
        prefixes.append((i, prefix))
    
    # Check for duplicates
    seen = {}
    failures = []
    
    for idx, prefix in prefixes:
        # Normalize: remove name
        normalized = re.sub(r'^[a-zăâîșț]+\s+', '', prefix)
        
        if normalized in seen:
            failures.append(idx)
            failures.append(seen[normalized])
        else:
            seen[normalized] = idx
    
    return len(failures) == 0, list(set(failures))


def get_opening_move_instruction(assigned_move: str) -> str:
    """Get prompt instruction for specific opening move."""
    instructions = {
        "situational_hook": "Începe cu un context situațional ('După turele lungi...', 'În weekend...')",
        "preference_statement": "Începe cu o preferință explicită ('Preferă...', 'Îi place...', 'Evită...')",
        "micro_story": "Începe cu o mică poveste ('Sâmbăta trecută...', 'Recent...')",
        "contrast": "Începe cu un contrast ('Deși...', 'Chiar dacă...')",
        "value": "Începe cu o valoare ('Ține mult la...', 'Pretuiește...')",
        "sensory_anchor": "Începe cu un detaliu senzorial ('Mirosul de...', 'Sunetul...')"
    }
    return instructions.get(assigned_move, "Variază începutul")


# ============================================================================
# 4. REALISM CONSTRAINTS GENERATOR
# ============================================================================

CONSTRAINT_CATEGORIES = {
    "budget": [
        "buget strict, compară prețuri",
        "economii mici, grijă la cheltuieli",
        "trăiește de la salariu la salariu",
        "nu-și permite vacanțe scumpe",
    ],
    "time_pressure": [
        "timp limitat între job și familie",
        "program supraîncărcat, rareori timp liber",
        "weekend-ul ocupat cu treburi gospodărești",
    ],
    "health_nuisance": [
        "dureri de spate ocazionale",
        "insomnie când e stresat",
        "alergie sezonieră la praf/polen",
        "probleme stomacale la nervi",
    ],
    "bureaucratic_fatigue": [
        "obosită de hârtii la primărie/spital",
        "experiențe negative cu instituții",
        "evită birocrația cât poate",
    ],
    "family_duty": [
        "grijă de părinți bolnavi",
        "responsabil de nepoți frecvent",
        "trebuie să ajute la gospodărie rurală",
    ],
    "commute_burden": [
        "navetează 1+ ore zilnic",
        "depinde de maxi-taxi cu orar neregulat",
        "drum greu iarna",
    ],
}


def generate_realism_constraints(count: int = 2) -> List[str]:
    """Generate 2-3 realistic constraints per persona."""
    categories = list(CONSTRAINT_CATEGORIES.keys())
    selected_categories = random.sample(categories, min(count, len(categories)))
    
    constraints = []
    for cat in selected_categories:
        constraints.append(random.choice(CONSTRAINT_CATEGORIES[cat]))
    
    return constraints


def validate_not_pollyanna(texts: List[str]) -> Tuple[bool, float]:
    """
    Check that texts aren't uniformly positive.
    
    Returns:
        (passed, positivity_ratio)
    """
    # Markers of realism (friction/negativity)
    friction_markers = [
        'dar', 'totuși', 'deși', 'însă',
        'uneori', 'din când în când',
        'îl irită', 'o irită', 'îl enervează',
        'îi e greu', 'îi e dificil',
        'nu suportă', 'evită',
        'obosit', 'stresat',
        'probleme', 'dificultăți',
        'limitat', 'restrictiv',
        'nu-și permite', 'prea scump',
        'bătăi de cap', 'complicații',
    ]
    
    # Positive-only markers (suspicious)
    pure_positive = [
        'mereu', 'tot timpul', 'întotdeauna',
        'adoră', 'iubește', 'perfect',
        'minunat', 'excelent', 'extraordinar',
        'totul e bine', 'fără probleme',
    ]
    
    total_friction = 0
    total_positive = 0
    
    for text in texts:
        text_lower = text.lower()
        total_friction += sum(1 for m in friction_markers if m in text_lower)
        total_positive += sum(1 for m in pure_positive if m in text_lower)
    
    # Need some friction markers, not pure positive
    total_words = sum(len(t.split()) for t in texts)
    friction_ratio = total_friction / max(total_words * 0.01, 1)
    
    # Pass if we have at least 1 friction marker per 100 words
    passed = total_friction >= len(texts) and friction_ratio > 0.5
    
    return passed, friction_ratio


# ============================================================================
# 5. MOLDOVA ANCHOR BANK
# ============================================================================

MOLDOVA_ANCHORS = {
    "Chisinau": {
        "places": ["parcul Valea Morilor", "piața Centrală", "Botanica", "Râșcani", "Ciocana", "Buiucani", "troleibuzul 30", "maxi-taxi Bălți-Chișinău", "la castel", "scara blocului"],
        "food": ["plăcinte cu brânză", "zeamă de găină", "mămăligă cu smântână", "sarmale în foi de viță", "cozonac de casă", "compot de cireșe", "pește de Dunăre", "pâine de casă"],
        "routine": ["vin de casă de la țară", "piață de la 6 dimineața", "troleibuz aglomerat", "scara blocului", "balconul de la bucătărie", "beciul cu murături", "aprovizionare de la Metro"],
    },
    "Nord": {
        "places": ["piața din Bălți", "parcul Pușkin", "Râbnița", "Soroca și dealul", "Nisporeni", "Drochia", "Fălești", "Glodeni", "maxi-taxi spre Chișinău", "la gară"],
        "food": ["plăcinte tărcate", "borș de burechi", "mămăligă cu jumări", "cârnați de casă", "dulceață de trandafir", "compot de mere", "zacuscă", "ghiveci"],
        "routine": ["gradina de lângă casă", "piața săptămânală", "autobuzul interurban", "beciul cu cartofi", "cazanul pentru vin", "animale la țară", "podgoria de lângă sat"],
    },
    "Centru": {
        "places": ["piața din Ungheni", "Nistru la răsărit", "Călărași", "Strășeni", "Hâncești", "Ialoveni", "Criuleni", "Orhei și împrejurimile", "maxi-taxi", "la pod"],
        "food": ["plăcinte în tigaie", "sarmale în foi de varză", "mămăligă cu tochitură", "pâine de casă cu chisăliță", "dulceață de caise", "compot de prune", "pește din Nistru", "ciuperci de pădure"],
        "routine": ["apa din fântână", "gospodăria din spatele casei", "pomii fructiferi", "vin de casă", "trenul de dimineață", "drumul național", "câmpul de lângă sat"],
    },
    "Sud": {
        "places": ["Cahul și piața", "Comrat", "Cantemir", "Leova", "Cimișlia", "Basarabeasca", "Vulcănești", "lacul de acumulare", "maxi-taxi", "la frontieră"],
        "food": ["plăcinte cu varză", "pește prăjit", "mămăligă cu brânză de oaie", "sarmale mici", "pepeni de Dănceni", "struguri de masă", "compot de piersici", "zacuscă de gogosari"],
        "routine": ["via de lângă casă", "câmpul de viță", "piața de legume", "soarele puternic", "căldura verii", "drumul spre Giurgiulești", "cazanele de tuică", "gradina de zarzavat"],
    },
    "Gagauzia": {
        "places": ["Comrat", "Ceadîr-Lunga", "Vulcănești", "bulevardul principal", "piața din centru", "la câmp", "maxi-taxi", "autogara", "la fabrica de conserve"],
        "food": ["plăcinte gagauze", "shorpa", "mămăligă cu brânză", "pește din lac", "ciorbă de pește", "compot de gutui", "dovleci copți", "pâine de casă"],
        "routine": ["viața în sat", "câmpurile de viță", "munca la câmp", "autobuzul spre Comrat", "piața de vineri", "vecinii de la țară", "limba gagauză între prieteni"],
    },
}

# Generic fallbacks
GENERIC_ANCHORS = {
    "places": ["piața din weekend", "parcul din centru", "statia de maxi-taxi", "magazinul din colț", "scara blocului", "gradina din spate"],
    "food": ["plăcinte", "mămăligă", "sarmale", "ciorbă", "compot", "pâine de casă"],
    "routine": ["vin de casă", "piața locală", "vecinii de la etajul 2", "troleibuzul", "gradina de legume"],
}


def get_anchors_for_region(region: str, district: str) -> Dict[str, List[str]]:
    """Get anchor bank for a specific region/district."""
    # Map region names
    region_key = region
    if region in ["Chisinau", "Chişinău"]:
        region_key = "Chisinau"
    elif region in ["Nord", "Bălți"]:
        region_key = "Nord"
    elif region in ["Centru", "Ungheni"]:
        region_key = "Centru"
    elif region in ["Sud", "Cahul"]:
        region_key = "Sud"
    elif "Gagauzia" in region or "Găgăuzia" in region:
        region_key = "Gagauzia"
    
    return MOLDOVA_ANCHORS.get(region_key, GENERIC_ANCHORS)


def select_anchors(region: str, district: str) -> Tuple[str, str]:
    """Select 2 anchors: 1 place + 1 food/routine."""
    anchors = get_anchors_for_region(region, district)
    
    place = random.choice(anchors.get("places", GENERIC_ANCHORS["places"]))
    
    # Mix food and routine
    food_routine = (anchors.get("food", []) + anchors.get("routine", [])) or GENERIC_ANCHORS["food"]
    routine = random.choice(food_routine)
    
    return place, routine


def validate_anchor_specificity(text: str, region: str) -> Tuple[bool, int]:
    """
    Check if text uses region-specific anchors.
    
    Returns:
        (has_anchor, anchor_count)
    """
    anchors = get_anchors_for_region(region, "")
    all_anchors = (
        anchors.get("places", []) + 
        anchors.get("food", []) + 
        anchors.get("routine", [])
    )
    
    text_lower = text.lower()
    found = sum(1 for anchor in all_anchors if anchor.lower() in text_lower)
    
    return found > 0, found


# ============================================================================
# 6. NEMOTRON-FEEL PROMPT PACK
# ============================================================================

def generate_stage_b_prompt_feel(
    name: str,
    demographics: Dict[str, any],
    ocean_scores: Dict[str, int]
) -> str:
    """
    Stage B prompt with feel patches.
    
    Generates: behavioral cues + constraints + anchor slots
    """
    # Generate components
    behavioral_cues = generate_behavioral_cues(ocean_scores)
    constraints = generate_realism_constraints(2)
    place_anchor, routine_anchor = select_anchors(
        demographics.get('region', 'Centru'),
        demographics.get('district', '')
    )
    
    prompt = f"""Generează CONTEXT pentru o persoană din Moldova (FĂRĂ a menționa trăsături de personalitate directe).

DATE DE BAZĂ:
- Nume: {name}
- Vârstă: {demographics.get('age')} ani
- Sex: {demographics.get('sex')}
- Ocupație: {demographics.get('occupation')}
- Educație: {demographics.get('education_level')}
- Localitate: {demographics.get('locality')}, {demographics.get('district')}
- Regiune: {demographics.get('region')}

{behavioral_cues.to_prompt()}

CONSTRÂNGERI REALISTE (sunt parte din viață, nu tragice):
- {constraints[0]}
- {constraints[1]}

ANCORE LOCALE (folosește natural, în context):
- Loc: {place_anchor}
- Rutină/food: {routine_anchor}

REGULI STRICTE:
1. INTERZIS: cuvinte precum "deschis", "conștiincios", "extrovertit", "agreabil", "nevrotic", "personalitate", "Big Five", "scor", "t_score"
2. Arată comportamentul, nu spune trăsătura
3. Include frecțiune realistă (nu totul e perfect)
4. Folosește ancorele local într-o propoziție, nu ca listă turistică
5. Exprimă constrângerile natural, nu ca plângere

GENEREAZĂ:
1. CULTURAL_BACKGROUND (400-1500 caractere): Context de viață cu ancore locale integrate
2. SKILLS_AND_EXPERTISE (200-800 caractere): Abilități vizibile în acțiune
3. SKILLS_LIST: 5-8 abilități concrete
4. HOBBIES_AND_INTERESTS (200-800 caractere): Activități cu frecțiune realistă
5. HOBBIES_LIST: 4-6 hobby-uri
6. CAREER_GOALS (150-600 caractere): Obiective realiste, nu utopice

RĂSPUNS JSON:
{{
  "cultural_background": "...",
  "skills_and_expertise": "...",
  "skills_list": ["..."],
  "hobbies_and_interests": "...",
  "hobbies_list": ["..."],
  "career_goals": "...",
  "behavioral_cues_summary": "..."
}}"""
    
    return prompt


def generate_stage_c_prompt_feel(
    name: str,
    demographics: Dict[str, any],
    context: Dict[str, str],
    behavioral_cues: BehavioralCues,
    opening_moves: List[str]
) -> str:
    """
    Stage C prompt with feel patches.
    
    Generates: 6 personas with controlled, varied openings
    """
    first_name = name.split()[0]
    
    # Build opening instructions
    opening_instructions = []
    persona_fields = ["persona", "professional_persona", "sports_persona", "arts_persona", "travel_persona", "culinary_persona"]
    
    for i, field in enumerate(persona_fields):
        move = opening_moves[i % len(opening_moves)]
        instruction = get_opening_move_instruction(move)
        opening_instructions.append(f"{field}: {instruction}")
    
    background = context.get('cultural_background', '')[:400]
    opening_instructions_text = '\n'.join(opening_instructions)
    behavioral_cues_text = behavioral_cues.to_prompt()
    
    prompt = f"""Generează 6 variante scurte de persona pentru {name}.

DATE:
- {demographics.get('age')} ani, {demographics.get('occupation')}
- {demographics.get('locality')}, {demographics.get('district')}

{behavioral_cues_text}

CONTEXT:
{background}

CELE 6 PERSONA (1-3 propoziții fiecare):

{opening_instructions_text}

REGULI STRICTE:
1. INTERZIS: "deschis", "conștiincios", "extrovertit", "agreabil", "nevrotic", "personalitate", "trăsătură", "scor"
2. Fiecare începe DIFERIT (vezi instrucțiunile de mai sus)
3. Toate 6 arată ACELAȘI comportament, nu ACEEAȘI trăsătură
4. Include elemente de frecțiune: "dar", "totuși", "uneori", "îi e greu", "nu poate"
5. Fii specific la loc, nu generic
6. Nu idealiza - persoanele reale au limitări

RĂSPUNS JSON:
{{
  "persona": "{first_name}...",
  "professional_persona": "{first_name}...",
  "sports_persona": "{first_name}...",
  "arts_persona": "{first_name}...",
  "travel_persona": "{first_name}...",
  "culinary_persona": "{first_name}..."
}}"""
    
    return prompt


def generate_repair_prompt(
    field_name: str,
    current_text: str,
    failure_type: str,
    behavioral_cues: BehavioralCues
) -> str:
    """
    Generate repair prompt for specific validator failure.
    """
    repairs = {
        "trait_leak": {
            "instruction": "Rescrie eliminând cuvintele despre trăsături de personalitate. Arată comportamentul, nu numi trăsătura.",
            "banned": "Evită: deschis, conștiincios, extrovertit, agreabil, nevrotic, personalitate, scor"
        },
        "opening_repetition": {
            "instruction": "Rescrie PRIMUL PARAGRAF cu un început diferit. Păstrează restul intact.",
            "banned": "Nu începe la fel ca celelalte variante"
        },
        "too_positive": {
            "instruction": "Adaugă frecțiune realistă: limitări, mici frustrări, compromisuri.",
            "banned": "Evită: mereu, tot timpul, perfect, minunat, totul e bine"
        },
        "weak_anchors": {
            "instruction": "Adaugă detalii specifice locului (piață, transport, mâncare locală).",
            "banned": "Evită: descrieri generice care ar putea fi oriunde"
        }
    }
    
    repair = repairs.get(failure_type, repairs["too_positive"])
    
    prompt = f"""REPARĂ câmpul "{field_name}" conform regulii de mai jos.

PROBLEMA: {repair['instruction']}

TEXT ACTUAL:
{current_text}

{behavioral_cues.to_prompt()}

REGULI REPARAȚIE:
- {repair['banned']}
- Păstrează aceeași voce și personă
- Fii specific și concret

RĂSPUNS: textul reparat, doar acest câmp"""
    
    return prompt


# ============================================================================
# VALIDATOR SUITE
# ============================================================================

class PersonaFeelValidator:
    """Combined validator for feel patches."""
    
    def __init__(self, region: str = "Centru"):
        self.region = region
        self.used_openings = set()
    
    def validate_full(
        self,
        texts: Dict[str, any],
        ocean_scores: Dict[str, int]
    ) -> Dict[str, any]:
        """
        Run full validation suite.
        
        Returns:
            Dict with validation results and repair instructions
        """
        results = {
            "passed": True,
            "checks": {},
            "repairs": []
        }
        
        # 1. Trait leak check
        # Convert all values to strings (lists become joined strings)
        text_parts = []
        for v in texts.values():
            if isinstance(v, list):
                text_parts.append(" ".join(str(x) for x in v))
            elif isinstance(v, str):
                text_parts.append(v)
        all_text = " ".join(text_parts)
        trait_ok, trait_words = validate_trait_leak(all_text)
        results["checks"]["trait_leak"] = {
            "passed": trait_ok,
            "found_words": trait_words
        }
        if not trait_ok:
            results["passed"] = False
            results["repairs"].append({
                "type": "trait_leak",
                "fields": list(texts.keys()),
                "details": trait_words
            })
        
        # 2. Opening variation check
        persona_texts = [
            texts.get("persona", ""),
            texts.get("professional_persona", ""),
            texts.get("sports_persona", ""),
            texts.get("arts_persona", ""),
            texts.get("travel_persona", ""),
            texts.get("culinary_persona", "")
        ]
        opening_ok, failing_indices = validate_opening_variation(persona_texts)
        results["checks"]["opening_variation"] = {
            "passed": opening_ok,
            "failing_indices": failing_indices
        }
        if not opening_ok:
            results["passed"] = False
            field_names = ["persona", "professional_persona", "sports_persona", 
                          "arts_persona", "travel_persona", "culinary_persona"]
            for idx in failing_indices:
                results["repairs"].append({
                    "type": "opening_repetition",
                    "field": field_names[idx],
                    "details": "Duplicate opening"
                })
        
        # 3. Pollyanna check
        polly_ok, friction_ratio = validate_not_pollyanna(persona_texts)
        results["checks"]["pollyanna"] = {
            "passed": polly_ok,
            "friction_ratio": friction_ratio
        }
        if not polly_ok:
            results["passed"] = False
            results["repairs"].append({
                "type": "too_positive",
                "fields": list(texts.keys()),
                "details": f"Friction ratio {friction_ratio:.2f} too low"
            })
        
        # 4. Anchor specificity check
        anchor_ok, anchor_count = validate_anchor_specificity(
            all_text, self.region
        )
        results["checks"]["anchor_specificity"] = {
            "passed": anchor_ok,
            "anchor_count": anchor_count
        }
        if not anchor_ok:
            results["passed"] = False
            results["repairs"].append({
                "type": "weak_anchors",
                "fields": ["cultural_background", "persona"],
                "details": f"Only {anchor_count} anchors found"
            })
        
        return results
