"""Prompt templates for generating narrative content in Romanian.

Prompt Version: 1.4.0
Changes from 1.3.0:
- Switched Stage B output to JSON-only with strict key contract
- Added explicit JSON schema guidance and minimum lengths

Follows the 8-field JSON structure:
1. descriere_generala - General personality
2. profil_profesional - Professional life
3. hobby_sport - Sports and activities
4. hobby_arta_cultura - Cultural interests
5. hobby_calatorii - Travel preferences
6. hobby_culinar - Culinary habits
7. career_goals_and_ambitions - Career goals
8. persona_summary - One-line summary
"""

import json
from typing import Dict, List, Optional, Tuple
from .models import Persona
from .geo_tables import strict_geo_enabled

# Prompt versioning for reproducibility
PROMPT_VERSION = "1.4.0"


def get_prompt_version() -> str:
    """Return current prompt version for reproducibility tracking."""
    return PROMPT_VERSION


def get_base_context(persona: Persona) -> str:
    """
    Generate base context string from persona data.
    
    Args:
        persona: Persona object with structured data
        
    Returns:
        Context string for prompts
    """
    lines = [
        f"Nume: {persona.name}",
        f"Sex: {persona.sex}",
        f"Vârstă: {persona.age} ani",
        f"Etnie: {persona.ethnicity}",
        f"Limba maternă: {persona.mother_tongue}",
        f"Religie: {persona.religion}",
        f"Stare civilă: {persona.marital_status}",
        f"Educație: {persona.education_level}",
    ]
    
    if persona.field_of_study:
        lines.append(f"Domeniul de studiu: {persona.field_of_study}")
    
    lines.append(f"Ocupație: {persona.occupation}")

    if strict_geo_enabled() and not (persona.city and persona.city.strip()):
        if persona.district:
            lines.append(f"Raion: {persona.district}")
        lines.append(f"Regiune: {persona.region} ({persona.residence_type})")
        lines.append("Instrucțiune: Nu inventa localitatea; folosește doar raionul și regiunea.")
    else:
        lines.append(f"Localitate: {persona.city}, {persona.region} ({persona.residence_type})")
    
    return "\n".join(lines)


def _get_sex_pronouns(sex: str) -> Dict[str, str]:
    """Get appropriate pronouns based on sex."""
    if sex == "Feminin":
        return {
            "subject": "ea",
            "object": "o",
            "possessive": "ei",
            "reflexive": "își",
            "adjective_f": "o femeie",
            "adjective_m": "o persoană",
        }
    else:
        return {
            "subject": "el",
            "object": "îl",
            "possessive": "lui",
            "reflexive": "își",
            "adjective_f": "un bărbat",
            "adjective_m": "o persoană",
        }


def _locality_phrase(persona: Persona) -> str:
    """Return a locality phrase that respects strict-geo mode."""
    if strict_geo_enabled() and not (persona.city and persona.city.strip()):
        if persona.district:
            return f"din raionul {persona.district}"
        return f"din regiunea {persona.region}"
    if persona.city:
        return f"din {persona.city}"
    return f"din regiunea {persona.region}"


def _community_phrase(persona: Persona) -> str:
    """Return a community phrase that respects strict-geo mode."""
    if strict_geo_enabled() and not (persona.city and persona.city.strip()):
        if persona.district:
            return f"în comunitatea raionului {persona.district}"
        return f"în comunitatea regiunii {persona.region}"
    if persona.city:
        return f"în comunitatea {persona.city}"
    return f"în comunitatea regiunii {persona.region}"


def _get_default_required_keys_and_lengths() -> Tuple[List[str], Dict[str, int]]:
    """Load required keys and minimum lengths from schema, with safe fallback."""
    fallback_required = [
        "descriere_generala",
        "profil_profesional",
        "hobby_sport",
        "hobby_arta_cultura",
        "hobby_calatorii",
        "hobby_culinar",
        "career_goals_and_ambitions",
        "persona_summary",
    ]
    fallback_min_lengths = {
        "descriere_generala": 20,
        "profil_profesional": 20,
        "hobby_sport": 10,
        "hobby_arta_cultura": 10,
        "hobby_calatorii": 10,
        "hobby_culinar": 10,
        "career_goals_and_ambitions": 10,
        "persona_summary": 10,
    }

    try:
        from .narrative_contract import NARRATIVE_JSON_SCHEMA
    except Exception:
        return fallback_required, fallback_min_lengths

    required = NARRATIVE_JSON_SCHEMA.get("required", fallback_required)
    properties = NARRATIVE_JSON_SCHEMA.get("properties", {})
    min_lengths = {
        key: properties.get(key, {}).get("minLength", fallback_min_lengths.get(key, 1))
        for key in required
    }
    return required, min_lengths


def _format_json_template(required_keys: List[str]) -> str:
    """Return a JSON skeleton with required keys."""
    lines = ["{"]
    for i, key in enumerate(required_keys):
        comma = "," if i < len(required_keys) - 1 else ""
        lines.append(f'  "{key}": "..."' + comma)
    lines.append("}")
    return "\n".join(lines)


def _format_min_lengths(required_keys: List[str], min_lengths: Dict[str, int]) -> str:
    """Format minimum length constraints for prompt guidance."""
    if not min_lengths:
        return ""
    parts = []
    for key in required_keys:
        min_len = min_lengths.get(key)
        if min_len:
            parts.append(f"{key}>={min_len}")
    if not parts:
        return ""
    return "LUNGIMI MINIME (caractere): " + ", ".join(parts)


def generate_full_prompt(
    persona: Persona,
    required_keys: Optional[List[str]] = None,
    min_lengths: Optional[Dict[str, int]] = None,
) -> str:
    """
    Generate a single comprehensive prompt for all narrative fields (JSON-only).
    
    Args:
        persona: Persona object
        required_keys: JSON keys to require
        min_lengths: Minimum lengths per key
        
    Returns:
        Complete prompt string
    """
    if required_keys is None or min_lengths is None:
        required_keys, min_lengths = _get_default_required_keys_and_lengths()
    context = get_base_context(persona)
    pronouns = _get_sex_pronouns(persona.sex)
    first_name = persona.name.split()[0]
    
    # Age-appropriate adaptations
    if persona.age < 18:
        return _generate_child_prompt(persona, context, pronouns, required_keys, min_lengths)
    elif persona.age >= 65:
        return _generate_elderly_prompt(persona, context, pronouns, required_keys, min_lengths)
    else:
        return _generate_adult_prompt(persona, context, pronouns, required_keys, min_lengths)


def _generate_adult_prompt(
    persona: Persona, 
    context: str, 
    pronouns: Dict[str, str],
    required_keys: List[str],
    min_lengths: Dict[str, int],
) -> str:
    """Generate prompt for adult personas (18-64)."""
    
    first_name = persona.name.split()[0]
    sex_adj = pronouns["adjective_f"] if persona.sex == "Feminin" else "un bărbat"
    locality = _locality_phrase(persona)
    json_template = _format_json_template(required_keys)
    min_lengths_text = _format_min_lengths(required_keys, min_lengths)

    guidance = {
        "descriere_generala": (
            f"{first_name} este {sex_adj} de {persona.age} ani {locality}. "
            "Descrie personalitatea în 2-3 propoziții, menționând 2-3 trăsături esențiale."
        ),
        "profil_profesional": (
            f"Lucrează ca {persona.occupation}. Detaliază responsabilitățile, abilitățile "
            "și etica în muncă în 3-4 propoziții."
        ),
        "hobby_sport": (
            f"Sporturi sau activități fizice practicate în {persona.region}. "
            "Menționează activități populare în regiune. 2-3 propoziții."
        ),
        "hobby_arta_cultura": (
            f"Gusturi în muzică, artă, literatură specifice contextului {persona.ethnicity} "
            f"și regiunii {persona.region}. Include nume de artiști sau tradiții. 2-3 propoziții."
        ),
        "hobby_calatorii": (
            "Preferințe de călătorie: stil (economic/confortabil/aventură) și destinații preferate. "
            "2 propoziții."
        ),
        "hobby_culinar": (
            f"Relația cu gastronomia: gătește? Preparate tradiționale din {persona.region}? "
            f"Influențe {persona.ethnicity} în bucătărie? 2-3 propoziții."
        ),
        "career_goals_and_ambitions": (
            "Obiective pe termen scurt și mediu, realiste pentru vârsta și ocupația sa. "
            "2-3 propoziții."
        ),
        "persona_summary": (
            "O singură propoziție (15-25 cuvinte) care include nume, vârstă, ocupație, regiune "
            "+ un element distinctiv."
        ),
    }
    section_lines = [
        f"- {key}: {guidance.get(key, 'Completează cu text relevant.')}"
        for key in required_keys
    ]
    
    return f"""GENEREAZĂ PROFIL NARATIV - Versiunea {PROMPT_VERSION}

DATE STRUCTURATE (trebuie respectate):
{context}

INSTRUCȚIUNI OBLIGATORII:
1. Folosește EXCLUSIV pronumele "{pronouns['subject']}" ({persona.sex.lower()}) pentru persoană
2. Vârsta menționată în text TREBUIE să fie {persona.age} ani
3. Ocupația menționată TREBUIE să fie: {persona.occupation}
4. Include obligatoriu cel puțin o referință specifică la regiunea {persona.region}
5. Include obligatoriu cel puțin un marker cultural pentru etnia {persona.ethnicity}
6. Folosește diacritice corecte în limba română

STRUCTURA CONȚINUTULUI (chei JSON):
{chr(10).join(section_lines)}

RESTRICȚII:
- NU folosi pronumele "el" dacă sexul este Feminin
- NU folosi pronumele "ea" dacă sexul este Masculin  
- NU inventa alte date demografice decât cele de mai sus
- NU folosi stereotipuri negative
- Total: 300-400 cuvinte

FORMAT DE RĂSPUNS (JSON ONLY):
{json_template}

REGULI JSON:
- Return JSON only. No markdown. No explanations.
- All keys must be present. Values must be non-empty strings.
- Use exact keys as listed; no extra keys.
- If you are unsure, still produce plausible content; do not omit keys.
{min_lengths_text}
"""


def _generate_child_prompt(
    persona: Persona, 
    context: str, 
    pronouns: Dict[str, str],
    required_keys: List[str],
    min_lengths: Dict[str, int],
) -> str:
    """Generate age-appropriate prompt for children (< 18)."""
    
    first_name = persona.name.split()[0]
    json_template = _format_json_template(required_keys)
    min_lengths_text = _format_min_lengths(required_keys, min_lengths)
    guidance = {
        "descriere_generala": (
            f"Cum este {first_name} ca copil la {persona.age} ani? Trăsături de caracter și comportament. "
            "2-3 propoziții."
        ),
        "profil_profesional": (
            "Viața școlară: materii preferate, relația cu colegii și profesorii. 2-3 propoziții."
        ),
        "hobby_sport": (
            f"Activități și jocuri potrivite vârstei de {persona.age} ani în {persona.region}. 2-3 propoziții."
        ),
        "hobby_arta_cultura": (
            "Interese și pasiuni: cărți, desen, muzică, natură. 2 propoziții."
        ),
        "hobby_calatorii": (
            "Preferințe de călătorie împreună cu familia sau școala, adaptate vârstei. 1-2 propoziții."
        ),
        "hobby_culinar": (
            "Relația cu mâncarea și tradițiile culinare din familie. 1-2 propoziții."
        ),
        "career_goals_and_ambitions": (
            "Aspirații educaționale sau personale potrivite vârstei. 1-2 propoziții."
        ),
        "persona_summary": (
            "O singură propoziție (15-25 cuvinte) care include nume, vârstă, localitate + un element distinctiv."
        ),
    }
    section_lines = [
        f"- {key}: {guidance.get(key, 'Completează cu text relevant.')}"
        for key in required_keys
    ]
    
    return f"""GENEREAZĂ PROFIL NARATIV PENTRU COPIL - Versiunea {PROMPT_VERSION}

DATE STRUCTURATE (trebuie respectate):
{context}

INSTRUCȚIUNI OBLIGATORII:
1. Vârsta este {persona.age} ANI - folosește limbaj adecvat acestei vârste
2. NU menționa carieră profesională sau experiență de muncă
3. Folosește pronumele "{pronouns['subject']}" corespunzător sexului {persona.sex}
4. Include context familial și comunitar din {persona.region}
5. Include aspirații educaționale potrivite vârstei

STRUCTURA CONȚINUTULUI (chei JSON):
{chr(10).join(section_lines)}

RESTRICȚII:
- Fără referințe la job sau carieră
- Ton adecvat vârstei de {persona.age} ani
- Diacritice corecte

FORMAT DE RĂSPUNS (JSON ONLY):
{json_template}

REGULI JSON:
- Return JSON only. No markdown. No explanations.
- All keys must be present. Values must be non-empty strings.
- Use exact keys as listed; no extra keys.
- If you are unsure, still produce plausible content; do not omit keys.
{min_lengths_text}
"""


def _generate_elderly_prompt(
    persona: Persona, 
    context: str, 
    pronouns: Dict[str, str],
    required_keys: List[str],
    min_lengths: Dict[str, int],
) -> str:
    """Generate age-appropriate prompt for elderly (65+)."""
    
    first_name = persona.name.split()[0]
    sex_adj = pronouns["adjective_f"] if persona.sex == "Feminin" else "un bărbat"
    community = _community_phrase(persona)
    json_template = _format_json_template(required_keys)
    min_lengths_text = _format_min_lengths(required_keys, min_lengths)
    guidance = {
        "descriere_generala": (
            f"{first_name} este {sex_adj} de {persona.age} ani. Experiența de viață acumulată și personalitatea. "
            "2-3 propoziții."
        ),
        "profil_profesional": (
            f"A lucrat ca {persona.occupation.lower()}. Realizări profesionale și amintiri din perioada activă. "
            "2-3 propoziții."
        ),
        "hobby_sport": (
            f"Viața la pensie și activități ușoare în {community}. 2-3 propoziții."
        ),
        "hobby_arta_cultura": (
            f"Interese culturale și artistice, incluzând tradiții din cultura {persona.ethnicity}. 2-3 propoziții."
        ),
        "hobby_calatorii": (
            "Preferințe de călătorie la vârsta respectivă, cu accent pe familie sau comunitate. 1-2 propoziții."
        ),
        "hobby_culinar": (
            f"Tradiții culinare păstrate din regiunea {persona.region}. 2-3 propoziții."
        ),
        "career_goals_and_ambitions": (
            "Obiective personale sau comunitare potrivite vârstei (voluntariat, familie, sănătate). 1-2 propoziții."
        ),
        "persona_summary": (
            "O singură propoziție (15-25 cuvinte) care include nume, vârstă, ocupație anterioară, regiune + un element distinctiv."
        ),
    }
    section_lines = [
        f"- {key}: {guidance.get(key, 'Completează cu text relevant.')}"
        for key in required_keys
    ]
    
    return f"""GENEREAZĂ PROFIL NARATIV PENTRU PERSOANĂ ÎN VÂRSTĂ - Versiunea {PROMPT_VERSION}

DATE STRUCTURATE (trebuie respectate):
{context}

INSTRUCȚIUNI OBLIGATORII:
1. Vârsta este {persona.age} ANI - abordează cu respect experiența de viață
2. Cariera anterioară: {persona.occupation} - folosește TIMPUL TRECUT
3. Stare curentă: LA PENSIE
4. Folosește pronumele "{pronouns['subject']}" corespunzător sexului {persona.sex}
5. Include tradiții și valori din cultura {persona.ethnicity}

STRUCTURA CONȚINUTULUI (chei JSON):
{chr(10).join(section_lines)}

RESTRICȚII:
- Folosește timpul trecut pentru carieră ("a lucrat", "a fost")
- Menționează pensia ca stare curentă
- Ton respectuos și călduros
- Diacritice corecte

FORMAT DE RĂSPUNS (JSON ONLY):
{json_template}

REGULI JSON:
- Return JSON only. No markdown. No explanations.
- All keys must be present. Values must be non-empty strings.
- Use exact keys as listed; no extra keys.
- If you are unsure, still produce plausible content; do not omit keys.
{min_lengths_text}
"""


def generate_career_goals_prompt(persona: Persona) -> str:
    """
    Generate prompt for career goals and ambitions section.
    
    Inspired by Nemotron Personas - adds future-oriented dimension
    to professional profile.
    
    Args:
        persona: Persona object
        
    Returns:
        Prompt string for career goals generation
    """
    first_name = persona.name.split()[0]
    pronouns = _get_sex_pronouns(persona.sex)
    
    # Age-appropriate career stage
    if persona.age < 25:
        stage_context = "la început de carieră"
        focus = "formare profesională, primii pași în domeniu"
    elif persona.age < 45:
        stage_context = "în plină dezvoltare profesională"
        focus = "avansare, specializare, noi oportunități"
    elif persona.age < 65:
        stage_context = "cu experiență solidă"
        focus = "consolidare, mentoring, stabilitate"
    else:
        stage_context = "la pensie"
        focus = "tranziție, voluntariat, transmiterea cunoștințelor"
    
    return f"""GENEREAZĂ OBIECTIVE ȘI ASPIRAȚII PROFESIONALE - Versiunea {PROMPT_VERSION}

DATE:
- Nume: {persona.name}
- Vârstă: {persona.age} ani ({stage_context})
- Ocupație actuală: {persona.occupation}
- Educație: {persona.education_level}
- Regiune: {persona.region}, {persona.residence_type}

INSTRUCȚIUNI:
1. Scrie la persoana a III-a ("{pronouns['subject']}")
2. FOCUSEAZĂ pe: {focus}
3. Include atât obiective PE TERMEN SCURT (1-2 ani) cât și MEDIU (3-5 ani)
4. Fii SPECIFIC și REALIST pentru contextul moldovenesc
5. Conectează obiectivele cu ocupația actuală ({persona.occupation})
6. Menționează eventuale cursuri, certificări, sau schimbări de direcție
7. 80-120 cuvinte, 4-6 propoziții

CONȚINUT SUGERAT:
- Ce își dorește să realizeze în carieră?
- Vrea să avanseze pe verticală (promovare) sau orizontală (schimbare domeniu)?
- Are intenția de a începe o afacere proprie?
- Cum vede evoluția profesională în contextul {persona.region}?
- Există planuri de formare continuă?

TON: Realist, ambițios dar nu exagerat, adaptat vârstei și contextului.

RĂSPUNDE STRICT ÎN FORMATUL:

[OBIECTIVE ȘI ASPIRAȚII]
(textul tău aici)
"""


def generate_persona_summary_prompt(persona: Persona) -> str:
    """
    Generate prompt for one-liner persona summary.
    
    Useful for quick dataset exploration and filtering.
    
    Args:
        persona: Persona object
        
    Returns:
        Prompt string for summary generation
    """
    first_name = persona.name.split()[0]
    sex_adj = "o femeie" if persona.sex == "Feminin" else "un bărbat"
    
    # Key traits based on age
    if persona.age < 30:
        life_stage = "tânăr"
    elif persona.age < 50:
        life_stage = "matur"
    else:
        life_stage = "experimentat"
    
    return f"""GENEREAZĂ REZUMAT PROFIL - Versiunea {PROMPT_VERSION}

DATE:
- Nume: {persona.name}
- Vârstă: {persona.age} ani
- Sex: {persona.sex}
- Ocupație: {persona.occupation}
- Regiune: {persona.region}
- Etnie: {persona.ethnicity}
- Educație: {persona.education_level}

INSTRUCȚIUNI:
1. Creează O SINGURĂ propoziție de 15-25 cuvinte
2. Include: nume, vârstă, ocupație, regiune + un element distinctiv
3. Folosește al treilea person (el/ea)
4. Fii concis dar informativ
5. Captură esența persoanei într-o frază

EXEMPLE BUNE:
- "Maria Popescu, o învățătoare de 34 de ani din nordul Moldovei, combină pasiunea pentru educație cu tradițiile ucrainene ale comunității sale."
- "Ion Ciobanu, un mecanic de 52 de ani din Chișinău, este cunoscut pentru priceperea sa în repararea mașinilor sovietice."

RĂSPUNDE STRICT ÎN FORMATUL:

[REZUMAT]
(textul tău aici)
"""


def parse_json_narrative_response_strict(
    response: str,
    allowed_keys: List[str],
    require_all: bool = True,
) -> Dict[str, str]:
    """
    Strict JSON-only parser for narrative responses.
    
    Enforces: response starts with '{', ends with '}', no extra keys,
    and all values are strings. Missing required keys raise if require_all.
    """
    if response is None or not response.startswith("{") or not response.endswith("}"):
        raise ValueError("Response is not JSON-only or has trailing/leading content")
    
    try:
        data = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON response") from exc
    
    if not isinstance(data, dict):
        raise ValueError("JSON response must be an object")
    
    allowed = set(allowed_keys)
    for key, value in data.items():
        if key not in allowed:
            raise ValueError(f"Unexpected key in JSON: {key}")
        if not isinstance(value, str):
            raise ValueError(f"Non-string value for key: {key}")
    
    if require_all:
        missing = [key for key in allowed_keys if key not in data]
        if missing:
            raise ValueError(f"Missing required keys: {missing}")
    
    return {key: data.get(key, "") for key in allowed_keys}


def parse_narrative_response(response: str) -> Dict[str, str]:
    """
    Parse LLM response into structured narrative fields.
    
    Uses multiple parsing strategies for robustness:
    1. Look for explicit section markers [SECTION]
    2. Fallback to header detection
    3. Final fallback: put all in descriere_generala
    
    Args:
        response: Raw LLM response text
        
    Returns:
        Dict mapping field names to text content
    """
    sections = {
        "descriere_generala": "",
        "profil_profesional": "",
        "hobby_sport": "",
        "hobby_arta_cultura": "",
        "hobby_calatorii": "",
        "hobby_culinar": "",
        "career_goals_and_ambitions": "",
        "persona_summary": "",
    }
    
    # Strategy 1: Look for explicit [SECTION] markers (most robust)
    section_markers = {
        "[DESCRIERE GENERALA]": "descriere_generala",
        "[PROFIL PROFESIONAL]": "profil_profesional",
        "[HOBBY SPORT]": "hobby_sport",
        "[HOBBY ARTA CULTURA]": "hobby_arta_cultura",
        "[HOBBY CALATORII]": "hobby_calatorii",
        "[HOBBY CULINAR]": "hobby_culinar",
        "[OBIECTIVE SI ASPIRATII]": "career_goals_and_ambitions",
        "[OBIECTIVE ȘI ASPIRAȚII]": "career_goals_and_ambitions",
        "[REZUMAT]": "persona_summary",
    }
    
    # Try to parse by markers
    parsed_by_markers = _parse_by_markers(response, section_markers)
    if parsed_by_markers and any(parsed_by_markers.values()):
        return parsed_by_markers
    
    # Strategy 2: Parse by headers
    parsed_by_headers = _parse_by_headers(response)
    if parsed_by_headers and any(parsed_by_headers.values()):
        return parsed_by_headers
    
    # Strategy 3: Everything in descriere_generala
    sections["descriere_generala"] = response.strip()
    return sections


def generate_repair_prompt(
    persona: Persona,
    missing_fields: List[str],
    existing_sections: Optional[Dict[str, str]] = None,
    min_lengths: Optional[Dict[str, int]] = None,
    force_fill_all: bool = False,
) -> str:
    """
    Generate a targeted repair prompt for missing narrative sections.
    
    Args:
        persona: Persona object
        missing_fields: Fields to regenerate
        existing_sections: Existing sections to keep unchanged
        
    Returns:
        Repair prompt string
    """
    context = get_base_context(persona)
    pronouns = _get_sex_pronouns(persona.sex)

    existing_sections = existing_sections or {}
    json_template = _format_json_template(missing_fields)
    min_lengths = min_lengths or {}
    min_lengths_text = _format_min_lengths(missing_fields, min_lengths)

    frozen_sections = []
    for key, value in existing_sections.items():
        if key in missing_fields or not value:
            continue
        frozen_sections.append(f'- {key}: "{value.strip()}"')
    frozen_text = "\n".join(frozen_sections) if frozen_sections else "(nu există)"
    force_text = ""
    if force_fill_all:
        force_text = (
            "Trebuie să completezi TOATE cheile. Nicio omisiune. "
            "Dacă e nevoie, scrie un text scurt dar valid (respectă lungimile minime)."
        )

    return f"""REPARĂ SECTIUNI LIPSĂ - Versiunea {PROMPT_VERSION}

DATE STRUCTURATE (trebuie respectate):
{context}

INSTRUCȚIUNI:
1. Return JSON only. No markdown. No explanations.
2. Completează DOAR cheile lipsă listate mai jos; NU adăuga alte chei.
3. NU modifica secțiunile deja existente.
4. Respectă vârsta {persona.age}, ocupația {persona.occupation}, regiunea {persona.region}.
5. Folosește pronumele "{pronouns['subject']}" ({persona.sex.lower()}).
6. Folosește diacritice corecte în limba română.
{force_text}

SECTIUNI EXISTENTE (NU LE MODIFICA):
{frozen_text}

FORMAT DE RĂSPUNS (JSON ONLY):
{json_template}

REGULI JSON:
- All keys must be present. Values must be non-empty strings.
- Use exact keys as listed; no extra keys.
- If you are unsure, still produce plausible content; do not omit keys.
{min_lengths_text}
"""


def _parse_by_markers(response: str, markers: Dict[str, str]) -> Dict[str, str]:
    """Parse response using explicit markers like [SECTION]."""
    sections = {
        "descriere_generala": "",
        "profil_profesional": "",
        "hobby_sport": "",
        "hobby_arta_cultura": "",
        "hobby_calatorii": "",
        "hobby_culinar": "",
        "career_goals_and_ambitions": "",
        "persona_summary": "",
    }
    
    # Find all marker positions
    marker_positions = []
    for marker, field in markers.items():
        pos = response.find(marker)
        if pos != -1:
            marker_positions.append((pos, marker, field))
    
    # Sort by position
    marker_positions.sort()
    
    # Extract content between markers
    for i, (pos, marker, field) in enumerate(marker_positions):
        start = pos + len(marker)
        if i + 1 < len(marker_positions):
            end = marker_positions[i + 1][0]
        else:
            end = len(response)
        
        content = response[start:end].strip()
        # Remove leading/trailing newlines and dashes
        content = content.strip('-').strip()
        sections[field] = content
    
    return sections


def _parse_by_headers(response: str) -> Dict[str, str]:
    """Parse response by detecting section headers."""
    sections = {
        "descriere_generala": "",
        "profil_profesional": "",
        "hobby_sport": "",
        "hobby_arta_cultura": "",
        "hobby_calatorii": "",
        "hobby_culinar": "",
        "career_goals_and_ambitions": "",
        "persona_summary": "",
    }
    
    # Map possible headers to section names
    header_map = {
        "descriere generala": "descriere_generala",
        "descriere": "descriere_generala",
        "profil profesional": "profil_profesional",
        "profil": "profil_profesional",
        "hobby sport": "hobby_sport",
        "sport": "hobby_sport",
        "hobby arta cultura": "hobby_arta_cultura",
        "arta": "hobby_arta_cultura",
        "hobby calatorii": "hobby_calatorii",
        "calatorii": "hobby_calatorii",
        "hobby culinar": "hobby_culinar",
        "culinar": "hobby_culinar",
        "obietive si aspiratii": "career_goals_and_ambitions",
        "obiective": "career_goals_and_ambitions",
        "aspiratii": "career_goals_and_ambitions",
        "rezumat": "persona_summary",
    }
    
    lines = response.split('\n')
    current_section = None
    buffer = []
    
    for line in lines:
        # Check if this line is a header
        line_clean = line.strip().lower().strip('*:1234567890. ')
        matched_section = None
        
        for header, section in header_map.items():
            if header in line_clean and len(line_clean) < 50:
                # Additional check: likely a header if short or has formatting
                if len(line) < 60 or '**' in line or line.strip().endswith(':'):
                    matched_section = section
                    break
        
        if matched_section:
            # Save previous section
            if current_section and buffer:
                sections[current_section] = '\n'.join(buffer).strip()
            current_section = matched_section
            buffer = []
        elif current_section:
            buffer.append(line)
    
    # Save last section
    if current_section and buffer:
        sections[current_section] = '\n'.join(buffer).strip()
    
    return sections


def validate_narrative_against_persona(
    narrative: Dict[str, str], 
    persona: Persona
) -> List[str]:
    """
    Validate that narrative content matches structured persona data.
    
    Args:
        narrative: Parsed narrative sections
        persona: Original persona data
        
    Returns:
        List of validation warnings/errors
    """
    issues = []
    
    combined_text = ' '.join(narrative.values()).lower()
    
    # Check 1: Age consistency
    age_mentions = [f"{persona.age} ani", f"de {persona.age} ani"]
    if not any(mention in combined_text for mention in age_mentions):
        issues.append(f"Age {persona.age} not mentioned in narrative")
    
    # Check 2: Occupation mentioned (for adults)
    if persona.age >= 18 and persona.age < 65:
        occ_words = persona.occupation.lower().split()
        if not any(word in combined_text for word in occ_words if len(word) > 3):
            issues.append(f"Occupation '{persona.occupation}' not mentioned")
    
    # Check 3: Region mentioned
    region_lower = persona.region.lower()
    if region_lower not in combined_text:
        issues.append(f"Region '{persona.region}' not mentioned")
    
    # Check 4: Name appears
    first_name = persona.name.split()[0].lower()
    if first_name not in combined_text:
        issues.append(f"First name '{first_name}' not found in narrative")
    
    # Check 5: Pronoun consistency (case-insensitive)
    if persona.sex == "Feminin":
        male_pronouns = [" el ", " lui ", "bărbat", "un băiat", " el\n", " el."]
        for pronoun in male_pronouns:
            if pronoun.lower() in combined_text:
                issues.append(f"Found male pronoun in female persona")
                break
        # Additional check: sentence starting with "El "
        if combined_text.startswith("el "):
            issues.append(f"Found male pronoun in female persona")
    else:
        female_pronouns = [" ea ", " ei ", "femeie", "o fată", " ea\n", " ea."]
        for pronoun in female_pronouns:
            if pronoun.lower() in combined_text:
                issues.append(f"Found female pronoun in male persona")
                break
        # Additional check: sentence starting with "Ea "
        if combined_text.startswith("ea "):
            issues.append(f"Found female pronoun in male persona")
    
    return issues


def extract_skills_from_text(text: str) -> List[str]:
    """
    Extract professional skills from narrative text.
    
    Args:
        text: Professional narrative text
        
    Returns:
        List of extracted skills
    """
    skill_indicators = [
        "competențe", "abilități", "cunoștințe", "experiență în",
        "specializat", "calificat", "priceput", "aptitudini",
        "expert în", "cunoaște", "stăpânește", "lucrează cu"
    ]
    
    skills = []
    text_lower = text.lower()
    
    for indicator in skill_indicators:
        if indicator in text_lower:
            idx = text_lower.find(indicator)
            start = max(0, idx - 30)
            end = min(len(text), idx + 100)
            context = text[start:end].strip()
            if len(context) > 10:
                skills.append(context)
    
    return skills[:5]


def extract_hobbies_from_text(text: str) -> List[str]:
    """
    Extract hobbies from narrative text.
    
    Args:
        text: Hobby-related narrative text
        
    Returns:
        List of extracted hobbies
    """
    hobby_keywords = [
        "fotbal", "volei", "bashet", "tenis", "înot", "alergare", "ciclism",
        "drumeții", "muzică", "cântat", "dans", "pictură", "lectură", "cărți",
        "filme", "teatru", "grădinărit", "gătit", "călătorii", "fotografie",
        "sport", "handbal", "fitness", "yoga", "pilates", "alpinism",
        "pescuit", "vânătoare", "șah", "table", "colecționare"
    ]
    
    found = []
    text_lower = text.lower()
    
    for hobby in hobby_keywords:
        if hobby in text_lower:
            found.append(hobby)
    
    return found[:8]
