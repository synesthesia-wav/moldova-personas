"""Name generation for different ethnic groups in Moldova.

Names are generated based on ethnicity to ensure realistic representation.

NOTE: The name lists here are not sourced from an official NBS dataset.
They should be treated as synthetic placeholders unless replaced with a
verifiable Moldova name frequency source.
"""

import logging
import random
from typing import List, Dict, Tuple, Union, Optional

from .ethnocultural_tables import (
    get_language_distribution,
    get_religion_distribution,
    load_tables,
    strict_mode_enabled,
)


logger = logging.getLogger(__name__)

_ETHNOCULTURAL_FALLBACKS: Dict[str, int] = {}


def record_ethnocultural_fallback(kind: str, detail: str) -> bool:
    """Record an ethnocultural fallback event for audit reporting."""
    key = f"{kind}:{detail}"
    first = key not in _ETHNOCULTURAL_FALLBACKS
    _ETHNOCULTURAL_FALLBACKS[key] = _ETHNOCULTURAL_FALLBACKS.get(key, 0) + 1
    return first


def get_ethnocultural_fallbacks() -> Dict[str, int]:
    """Return aggregated ethnocultural fallback events."""
    return dict(_ETHNOCULTURAL_FALLBACKS)


def reset_ethnocultural_fallbacks() -> None:
    """Clear ethnocultural fallback event counters."""
    _ETHNOCULTURAL_FALLBACKS.clear()


# Weighted first names: (name, weight) where weights sum to ~1.0 within each category
# Higher weight = more common name
FIRST_NAMES_WEIGHTED: Dict[str, Dict[str, List[Tuple[str, float]]]] = {
    "Moldovean": {
        "Feminin": [
            # Top names from Moldovan census data (~70% of population)
            ("Maria", 0.12), ("Elena", 0.10), ("Ana", 0.08), ("Natalia", 0.07),
            ("Irina", 0.06), ("Galina", 0.05), ("Tatiana", 0.05), ("Svetlana", 0.04),
            ("Viorica", 0.04), ("Ludmila", 0.04), ("Lidia", 0.03), ("Ecaterina", 0.03),
            ("Valentina", 0.03), ("Liudmila", 0.03), ("Rodica", 0.03), ("Veronica", 0.03),
            ("Doina", 0.03), ("Silvia", 0.02), ("Alina", 0.02), ("Cristina", 0.02),
            ("Daniela", 0.02), ("Mariana", 0.02), ("Olga", 0.02), ("Tamara", 0.02),
            ("Raisa", 0.01), ("Zinaida", 0.01), ("Antonina", 0.01), ("Angela", 0.01),
            ("Iuliana", 0.01), ("Liliana", 0.01), ("Mihaela", 0.01), ("Simona", 0.01),
        ],
        "Masculin": [
            ("Ion", 0.10), ("Vasile", 0.08), ("Andrei", 0.07), ("Alexandru", 0.07),
            ("Sergiu", 0.06), ("Mihai", 0.06), ("Victor", 0.05), ("Vladimir", 0.05),
            ("Nicolae", 0.05), ("Dumitru", 0.04), ("Gheorghe", 0.04), ("Valeriu", 0.04),
            ("Petru", 0.03), ("Constantin", 0.03), ("Anatolie", 0.03), ("Grigore", 0.03),
            ("Iurie", 0.03), ("Vitalie", 0.03), ("Oleg", 0.02), ("Maxim", 0.02),
            ("Adrian", 0.02), ("Boris", 0.02), ("Pavel", 0.02), ("Serghei", 0.02),
            ("Alexei", 0.02), ("Dmitrii", 0.02), ("Radu", 0.02), ("Ghenadie", 0.02),
            ("Igor", 0.02), ("Stefan", 0.02), ("Marcel", 0.01), ("Gabriel", 0.01),
        ],
    },
    "Român": {
        "Feminin": [
            ("Maria", 0.10), ("Elena", 0.08), ("Ioana", 0.07), ("Andreea", 0.06),
            ("Alexandra", 0.06), ("Ana", 0.05), ("Raluca", 0.05), ("Cristina", 0.04),
            ("Diana", 0.04), ("Mihaela", 0.04), ("Gabriela", 0.04), ("Simona", 0.03),
            ("Laura", 0.03), ("Alina", 0.03), ("Carmen", 0.03), ("Mariana", 0.03),
            ("Nicoleta", 0.03), ("Oana", 0.03), ("Roxana", 0.02), ("Bianca", 0.02),
            ("Claudia", 0.02), ("Daniela", 0.02), ("Georgiana", 0.02), ("Iulia", 0.02),
            ("Larisa", 0.02), ("Madalina", 0.02), ("Monica", 0.02), ("Patricia", 0.02),
            ("Sofia", 0.02), ("Ema", 0.01),
        ],
        "Masculin": [
            ("Andrei", 0.09), ("Alexandru", 0.08), ("Mihai", 0.07), ("Ionut", 0.06),
            ("Gabriel", 0.06), ("Stefan", 0.05), ("Cristian", 0.05), ("Daniel", 0.04),
            ("Marian", 0.04), ("Florin", 0.04), ("Adrian", 0.04), ("Bogdan", 0.04),
            ("Razvan", 0.03), ("Cosmin", 0.03), ("Dragos", 0.03), ("Lucian", 0.03),
            ("Catalin", 0.03), ("George", 0.03), ("Vlad", 0.03), ("Robert", 0.02),
            ("Darius", 0.02), ("Matei", 0.02), ("David", 0.02), ("Nicolas", 0.02),
            ("Sebastian", 0.02), ("Paul", 0.02), ("Radu", 0.02), ("Eduard", 0.02),
        ],
    },
    "Ucrainean": {
        "Feminin": [
            ("Olena", 0.08), ("Tetiana", 0.07), ("Natalia", 0.07), ("Iryna", 0.06),
            ("Olha", 0.06), ("Svitlana", 0.06), ("Liudmyla", 0.05), ("Halyna", 0.05),
            ("Kateryna", 0.05), ("Viktoriia", 0.04), ("Anastasiia", 0.04), ("Marharyta", 0.04),
            ("Yuliia", 0.04), ("Anna", 0.03), ("Sofia", 0.03), ("Maria", 0.03),
            ("Oksana", 0.03), ("Larysa", 0.03), ("Nadiia", 0.03), ("Zoriana", 0.02),
        ],
        "Masculin": [
            ("Oleksandr", 0.08), ("Serhii", 0.07), ("Andrii", 0.07), ("Viktor", 0.06),
            ("Ivan", 0.06), ("Bohdan", 0.05), ("Maksym", 0.05), ("Dmytro", 0.05),
            ("Artem", 0.05), ("Volodymyr", 0.05), ("Mykola", 0.04), ("Yurii", 0.04),
            ("Roman", 0.04), ("Pavlo", 0.04), ("Oleh", 0.03), ("Taras", 0.03),
            ("Hryhorii", 0.03), ("Vitalii", 0.03), ("Denys", 0.03), ("Mykhailo", 0.03),
        ],
    },
    "Găgăuz": {
        "Feminin": [
            ("Maria", 0.10), ("Anna", 0.08), ("Elena", 0.07), ("Tatiana", 0.06),
            ("Galina", 0.06), ("Irina", 0.05), ("Natalia", 0.05), ("Liubov", 0.04),
            ("Olga", 0.04), ("Tamara", 0.04), ("Valentina", 0.04), ("Ludmila", 0.04),
            ("Svetlana", 0.03), ("Nadejda", 0.03), ("Vera", 0.03), ("Lilia", 0.03),
            ("Inna", 0.03), ("Alla", 0.03), ("Zinaida", 0.02), ("Raisa", 0.02),
        ],
        "Masculin": [
            ("Ivan", 0.08), ("Petr", 0.07), ("Vasili", 0.07), ("Dmitrii", 0.06),
            ("Nikolai", 0.06), ("Andrei", 0.06), ("Sergei", 0.05), ("Mihail", 0.05),
            ("Alexei", 0.05), ("Viktor", 0.05), ("Georgi", 0.04), ("Grigori", 0.04),
            ("Anatolii", 0.04), ("Valerii", 0.04), ("Iurii", 0.04), ("Vladimir", 0.04),
            ("Konstantin", 0.03), ("Oleg", 0.03), ("Semion", 0.03), ("Stepan", 0.03),
        ],
    },
    "Rus": {
        "Feminin": [
            ("Anastasia", 0.07), ("Maria", 0.07), ("Daria", 0.06), ("Anna", 0.06),
            ("Victoria", 0.05), ("Polina", 0.05), ("Elizaveta", 0.05), ("Sofia", 0.05),
            ("Ekaterina", 0.05), ("Arina", 0.04), ("Alina", 0.04), ("Olga", 0.04),
            ("Tatiana", 0.04), ("Natalia", 0.04), ("Irina", 0.03), ("Elena", 0.03),
            ("Svetlana", 0.03), ("Marina", 0.03), ("Galina", 0.03), ("Ludmila", 0.02),
        ],
        "Masculin": [
            ("Aleksandr", 0.08), ("Dmitrii", 0.07), ("Maksim", 0.06), ("Sergei", 0.06),
            ("Andrei", 0.06), ("Aleksei", 0.05), ("Artem", 0.05), ("Ivan", 0.05),
            ("Roman", 0.04), ("Kirill", 0.04), ("Mikhail", 0.04), ("Nikita", 0.04),
            ("Matvei", 0.03), ("Egor", 0.03), ("Denis", 0.03), ("Ilia", 0.03),
            ("Vladimir", 0.03), ("Pavel", 0.03), ("Evgenii", 0.03), ("Konstantin", 0.03),
        ],
    },
    "Bulgar": {
        "Feminin": [
            ("Maria", 0.10), ("Ivana", 0.08), ("Elena", 0.07), ("Anna", 0.06),
            ("Teodora", 0.05), ("Desislava", 0.05), ("Petia", 0.05), ("Nina", 0.05),
            ("Penka", 0.04), ("Stanka", 0.04), ("Gina", 0.04), ("Liliana", 0.04),
            ("Rosa", 0.04), ("Sonia", 0.04), ("Violeta", 0.04), ("Zlatka", 0.04),
            ("Bojidara", 0.03), ("Donka", 0.03), ("Elica", 0.03), ("Gergana", 0.03),
        ],
        "Masculin": [
            ("Georgi", 0.09), ("Ivan", 0.08), ("Dimitar", 0.07), ("Petar", 0.07),
            ("Hristo", 0.06), ("Nikolai", 0.06), ("Stefan", 0.06), ("Aleksandar", 0.05),
            ("Bojidar", 0.05), ("Vasil", 0.05), ("Atanas", 0.04), ("Todor", 0.04),
            ("Plamen", 0.04), ("Rumen", 0.04), ("Krasimir", 0.04), ("Liuben", 0.04),
            ("Mihail", 0.03), ("Pavel", 0.03), ("Valentin", 0.03), ("Zdravko", 0.03),
        ],
    },
    "Rrom": {
        # Moldovan-Roma names (not Hungarian)
        "Feminin": [
            ("Marioara", 0.10), ("Floare", 0.09), ("Vasilica", 0.08), ("Stela", 0.07),
            ("Mănăilă", 0.07), ("Saveta", 0.06), ("Doina", 0.06), ("Luminița", 0.06),
            ("Rodica", 0.05), ("Marinela", 0.05), ("Costela", 0.05), ("Iuliana", 0.05),
            ("Maria", 0.04), ("Elena", 0.04), ("Diana", 0.04), ("Mirela", 0.04),
            ("Roxana", 0.03), ("Alina", 0.03), ("Cristina", 0.03), ("Daniela", 0.03),
        ],
        "Masculin": [
            ("Cioabă", 0.09), ("Marian", 0.08), ("Marius", 0.07), ("Gheorghe", 0.07),
            ("Nicolae", 0.06), ("Ion", 0.06), ("Costel", 0.06), ("Florin", 0.06),
            ("Sorin", 0.05), ("Valentin", 0.05), ("Adrian", 0.05), ("Ciprian", 0.05),
            ("Daniel", 0.04), ("Emil", 0.04), ("Liviu", 0.04), ("Răzvan", 0.04),
            ("Sebastian", 0.04), ("Viorel", 0.04), ("Alin", 0.04), ("Cosmin", 0.03),
        ],
    },
    "Altele": {
        # Actual minorities in Moldova: Armenian, Jewish, Polish, etc.
        "Feminin": [
            ("Ana", 0.10), ("Sofia", 0.09), ("Magdalena", 0.08), ("Lilit", 0.07),
            ("Bella", 0.07), ("Rachel", 0.06), ("Sarah", 0.06), ("Estera", 0.06),
            ("Agnieszka", 0.05), ("Katarzyna", 0.05), ("Małgorzata", 0.05), ("Anna", 0.05),
            ("Hasmik", 0.04), ("Siranush", 0.04), ("Marine", 0.04), ("Narine", 0.04),
            ("Ruth", 0.03), ("Deborah", 0.03), ("Miriam", 0.03), ("Leah", 0.03),
        ],
        "Masculin": [
            ("Artur", 0.09), ("Samuil", 0.08), ("David", 0.07), ("Mark", 0.07),
            ("Armen", 0.06), ("Ashot", 0.06), ("Gagik", 0.06), ("Hayk", 0.06),
            ("Piotr", 0.05), ("Krzysztof", 0.05), ("Andrzej", 0.05), ("Tomasz", 0.05),
            ("Moshe", 0.04), ("Abraham", 0.04), ("Isaac", 0.04), ("Jacob", 0.04),
            ("Efraim", 0.04), ("Shlomo", 0.04), ("Aaron", 0.04), ("Benjamin", 0.04),
        ],
    },
}


# Last names by ethnicity with frequency weights
LAST_NAMES_WEIGHTED: Dict[str, List[Tuple[str, float]]] = {
    "Moldovean": [
        ("Popa", 0.08), ("Rusu", 0.07), ("Ciobanu", 0.06), ("Sandu", 0.05),
        ("Cojocaru", 0.05), ("Munteanu", 0.04), ("Rotaru", 0.04), ("Gheorghe", 0.04),
        ("Melnic", 0.03), ("Cebotari", 0.03), ("Sirbu", 0.03), ("Balan", 0.03),
        ("Ursu", 0.03), ("Cojocari", 0.03), ("Lungu", 0.02), ("Bologa", 0.02),
        ("Bostan", 0.02), ("Buza", 0.02), ("Cara", 0.02), ("Cazacu", 0.02),
        ("Cazac", 0.02), ("Cercheș", 0.02), ("Chirila", 0.02), ("Ciorba", 0.02),
        ("Cojuhari", 0.01), ("Cosovan", 0.01), ("Curcă", 0.01), ("Donciu", 0.01),
        ("Furtuna", 0.01), ("Grosu", 0.01), ("Guțu", 0.01), ("Istrati", 0.01),
    ],
    "Român": [
        ("Popescu", 0.09), ("Ionescu", 0.08), ("Pop", 0.06), ("Radu", 0.05),
        ("Dumitrescu", 0.05), ("Stoica", 0.04), ("Gheorghe", 0.04), ("Matei", 0.04),
        ("Stan", 0.03), ("Dinu", 0.03), ("Tudor", 0.03), ("Barbu", 0.03),
        ("Vasile", 0.03), ("Marin", 0.03), ("Neagu", 0.02), ("Florea", 0.02),
        ("Mihai", 0.02), ("Ene", 0.02), ("Lazar", 0.02), ("Voicu", 0.02),
        ("Cristea", 0.02), ("Dragomir", 0.02), ("Nistor", 0.02), ("Ilie", 0.02),
    ],
    "Ucrainean": [
        ("Shevchenko", 0.06), ("Kovalenko", 0.05), ("Boyko", 0.05), ("Koval", 0.05),
        ("Moroz", 0.04), ("Tkachenko", 0.04), ("Bondarenko", 0.04), ("Oliynyk", 0.04),
        ("Kravchenko", 0.04), ("Ivanov", 0.03), ("Kovalchuk", 0.03), ("Polishchuk", 0.03),
        ("Lysenko", 0.03), ("Petrenko", 0.03), ("Kuzmenko", 0.03), ("Marchenko", 0.03),
        ("Yatsenko", 0.02), ("Ponomarenko", 0.02), ("Savchuk", 0.02), ("Vasilenko", 0.02),
    ],
    "Găgăuz": [
        ("Topal", 0.06), ("Ciocan", 0.05), ("Jardan", 0.05), ("Esir", 0.04),
        ("Ceban", 0.04), ("Chiosa", 0.04), ("Cristea", 0.04), ("Cvasniuc", 0.03),
        ("Dogaru", 0.03), ("Gaidarji", 0.03), ("Garstea", 0.03), ("Geanga", 0.03),
        ("Harbuli", 0.03), ("Ivanov", 0.03), ("Jelezoglo", 0.02), ("Karaman", 0.02),
        ("Kazak", 0.02), ("Kiriak", 0.02), ("Kolev", 0.02), ("Kolcu", 0.02),
    ],
    "Rus": [
        ("Ivanov", 0.06), ("Smirnov", 0.05), ("Kuznetsov", 0.05), ("Popov", 0.05),
        ("Vasiliev", 0.04), ("Petrov", 0.04), ("Sokolov", 0.04), ("Mikhailov", 0.04),
        ("Fedorov", 0.03), ("Morozov", 0.03), ("Volkov", 0.03), ("Alexeev", 0.03),
        ("Lebedev", 0.03), ("Semyonov", 0.03), ("Egorov", 0.02), ("Pavlov", 0.02),
        ("Kozlov", 0.02), ("Stepanov", 0.02), ("Novikov", 0.02), ("Alexandrov", 0.02),
    ],
    "Bulgar": [
        ("Ivanov", 0.07), ("Petrov", 0.06), ("Georgiev", 0.06), ("Dimitrov", 0.05),
        ("Stoyanov", 0.05), ("Iliev", 0.04), ("Todorov", 0.04), ("Angelov", 0.04),
        ("Marinov", 0.04), ("Hristov", 0.04), ("Yordanov", 0.03), ("Atanasov", 0.03),
        ("Vasilev", 0.03), ("Nikolov", 0.03), ("Popov", 0.03), ("Kolev", 0.03),
        ("Mihaylov", 0.02), ("Stefanov", 0.02), ("Alexandrov", 0.02), ("Manolov", 0.02),
    ],
    "Rrom": [
        # Moldovan-Roma surnames
        ("Mănăilă", 0.08), ("Cioabă", 0.07), ("Lăcătuș", 0.06), ("Stancu", 0.06),
        ("Marian", 0.05), ("Dinu", 0.05), ("Radu", 0.05), ("Ivan", 0.04),
        ("Costea", 0.04), ("Florea", 0.04), ("Mihai", 0.04), ("Gheorghe", 0.04),
        ("Nicolae", 0.03), ("Ion", 0.03), ("Sandu", 0.03), ("Popa", 0.03),
        ("Rusu", 0.03), ("Ciobanu", 0.03), ("Lupu", 0.03), ("Balan", 0.03),
    ],
    "Altele": [
        # Armenian, Jewish, Polish surnames in Moldova
        ("Avagian", 0.07), ("Movsesian", 0.06), ("Karapetian", 0.05), ("Hakobian", 0.05),
        ("Cohen", 0.06), ("Levi", 0.05), ("Moscovici", 0.04), ("Rosenberg", 0.04),
        ("Abramovici", 0.04), ("Goldenberg", 0.04), ("Kogan", 0.03), ("Schwartz", 0.03),
        ("Nowak", 0.05), ("Kowalski", 0.04), ("Wiśniewski", 0.04), ("Wójcik", 0.04),
        ("Kowalczyk", 0.03), ("Lewandowski", 0.03), ("Szymański", 0.03), ("Woźniak", 0.03),
    ],
}


def weighted_choice(choices: List[Tuple[str, float]]) -> str:
    """
    Make a weighted random choice from a list of (item, weight) tuples.
    
    Args:
        choices: List of (item, weight) tuples
        
    Returns:
        Selected item
    """
    names, weights = zip(*choices)
    return random.choices(names, weights=weights, k=1)[0]


def generate_name(ethnicity: str, sex: str) -> str:
    """
    Generate a realistic name based on ethnicity and sex with frequency weighting.
    
    Args:
        ethnicity: One of the ethnicity keys (Moldovean, Român, etc.)
        sex: "Masculin" or "Feminin"
        
    Returns:
        Full name as "First Last"
    """
    # Fallback to "Altele" if ethnicity not found
    if ethnicity not in FIRST_NAMES_WEIGHTED:
        ethnicity = "Altele"
    
    # Get weighted first name
    first_names = FIRST_NAMES_WEIGHTED[ethnicity].get(
        sex, 
        FIRST_NAMES_WEIGHTED[ethnicity]["Masculin"]
    )
    first_name = weighted_choice(first_names)
    
    # Get weighted last name
    last_names = LAST_NAMES_WEIGHTED.get(ethnicity, LAST_NAMES_WEIGHTED["Altele"])
    last_name = weighted_choice(last_names)
    
    return f"{first_name} {last_name}"


def get_language_by_ethnicity(
    ethnicity: str,
    region: Optional[str] = None,
    residence_type: Optional[str] = None,
) -> str:
    """
    Determine mother tongue based on ethnicity with regional variation.
    
    Instead of deterministic mapping, uses probabilistic conditional
    distribution to reflect bilingual realities in Moldova.
    
    Args:
        ethnicity: Ethnic group
        region: Optional region for regional adjustments (Chisinau, Nord, etc.)
        residence_type: Optional "Urban" or "Rural" for residence-based adjustments
        
    Returns:
        Language name in Romanian
        
    Note:
        These probabilities are estimates based on linguistic patterns in Moldova.
        Urban areas and Chisinau show higher Russian bilingualism.
        Rural areas tend toward ethnic language preservation.
    """
    # Prefer official ethnocultural cross-tabs if present
    tables = load_tables()
    strict = strict_mode_enabled()
    if tables:
        official = get_language_distribution(ethnicity)
        if official:
            languages = list(official.keys())
            weights = list(official.values())
            return random.choices(languages, weights=weights, k=1)[0]

        # If tables exist but this ethnicity is missing, use national distribution (official totals)
        national = tables.get("language_distribution")
        if national and not strict:
            if record_ethnocultural_fallback("language", f"national_totals:{ethnicity}"):
                logger.warning(
                    "No official language distribution for '%s'. Using national totals.",
                    ethnicity,
                )
            languages = list(national.keys())
            weights = list(national.values())
            return random.choices(languages, weights=weights, k=1)[0]

        # If national totals missing, use "Altele" proxy as last resort
        proxy = tables.get("language_by_ethnicity", {}).get("Altele")
        if proxy and not strict:
            if record_ethnocultural_fallback("language", f"altele_proxy:{ethnicity}"):
                logger.warning(
                    "No official language distribution for '%s'. Using 'Altele' proxy.",
                    ethnicity,
                )
            languages = list(proxy.keys())
            weights = list(proxy.values())
            return random.choices(languages, weights=weights, k=1)[0]

        if strict:
            raise ValueError(
                f"Strict mode: missing official mother tongue distribution for '{ethnicity}'."
            )

    if strict:
        raise ValueError(
            "Strict mode: ethnocultural language tables not available. "
            "Set MOLDOVA_PERSONAS_STRICT_ETHNOCULTURAL=0 to allow estimates."
        )

    logger.warning(
        "Ethnocultural language tables not available. "
        "Falling back to estimated language probabilities.",
    )

    # Base probabilities by ethnicity (primary language most likely)
    base_probs = {
        "Moldovean": {
            "Română": 0.92,
            "Rusă": 0.07,
            "Alta": 0.01,
        },
        "Român": {
            "Română": 0.98,
            "Rusă": 0.01,
            "Alta": 0.01,
        },
        "Ucrainean": {
            "Ucraineană": 0.78,
            "Română": 0.15,
            "Rusă": 0.06,
            "Alta": 0.01,
        },
        "Găgăuz": {
            "Găgăuză": 0.82,
            "Română": 0.10,
            "Rusă": 0.07,
            "Alta": 0.01,
        },
        "Rus": {
            "Rusă": 0.89,
            "Română": 0.10,
            "Alta": 0.01,
        },
        "Bulgar": {
            "Bulgară": 0.75,
            "Română": 0.15,
            "Rusă": 0.09,
            "Alta": 0.01,
        },
        "Rrom": {
            "Română": 0.88,
            "Rromani": 0.08,
            "Rusă": 0.03,
            "Alta": 0.01,
        },
        "Altele": {
            "Română": 0.45,
            "Rusă": 0.25,
            "Alta": 0.30,  # Armenian, Polish, etc.
        },
    }
    
    probs = base_probs.get(ethnicity, base_probs["Altele"]).copy()
    
    # Regional adjustments
    if region == "Chisinau":
        # Capital has more Russian bilingualism across all groups
        if "Rusă" in probs:
            probs["Rusă"] += 0.05
            probs["Română"] = probs.get("Română", 0) - 0.03
            # Re-normalize by reducing other languages proportionally
    elif region == "Gagauzia":
        # Gagauzia preserves Turkic language more
        if ethnicity == "Găgăuz":
            probs["Găgăuză"] += 0.08
            probs["Română"] = probs.get("Română", 0) - 0.05
    
    # Urban/rural adjustments
    if residence_type == "Urban":
        # Urban areas: slightly more Russian, slightly less ethnic language preservation
        if "Rusă" in probs:
            probs["Rusă"] += 0.03
        if ethnicity in ["Ucrainean", "Bulgar"]:
            ethnic_lang = {"Ucrainean": "Ucraineană", "Bulgar": "Bulgară"}[ethnicity]
            probs[ethnic_lang] -= 0.02
    elif residence_type == "Rural":
        # Rural areas: better preservation of ethnic languages
        if ethnicity in ["Găgăuz", "Ucrainean", "Bulgar"]:
            ethnic_lang = {"Găgăuz": "Găgăuză", "Ucrainean": "Ucraineană", "Bulgar": "Bulgară"}[ethnicity]
            if ethnic_lang in probs:
                probs[ethnic_lang] += 0.05
                probs["Română"] = probs.get("Română", 0) - 0.03
    
    # Normalize probabilities
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    
    # Sample from distribution
    languages = list(probs.keys())
    weights = list(probs.values())
    
    return random.choices(languages, weights=weights, k=1)[0]


def get_religion_by_ethnicity(ethnicity: str, region: Optional[str] = None) -> str:
    """
    Determine likely religion based on ethnicity and region with proper correlations.
    
    Args:
        ethnicity: Ethnic group
        region: Optional region for region-specific adjustments
        
    Returns:
        Religion name
    
    Note:
        These ethnicity-religion cross-tabulations are ESTIMATED based on
        demographic knowledge and regional patterns. The NBS 2024 ethnocultural
        communiqué provides national religion totals (≈96.5% Orthodox) and
        ethnicity-level cross-tabs. These probabilities are only used when
        official tables are unavailable.
        
        Sources consulted:
        - NBS 2024 Census: Religion distribution (national level)
        - Historical ethnic-religious patterns in Moldova
        - Regional demographic studies
    """
    # Prefer official ethnocultural cross-tabs if present
    tables = load_tables()
    strict = strict_mode_enabled()
    if tables:
        official = get_religion_distribution(ethnicity)
        if official:
            religions = list(official.keys())
            weights = list(official.values())
            return random.choices(religions, weights=weights, k=1)[0]

        # If tables exist but this ethnicity is missing, use national distribution (official totals)
        national = tables.get("religion_distribution")
        if national and not strict:
            if record_ethnocultural_fallback("religion", f"national_totals:{ethnicity}"):
                logger.warning(
                    "No official religion distribution for '%s'. Using national totals.",
                    ethnicity,
                )
            religions = list(national.keys())
            weights = list(national.values())
            return random.choices(religions, weights=weights, k=1)[0]

        # If national totals missing, use "Altele" proxy as last resort
        proxy = tables.get("religion_by_ethnicity", {}).get("Altele")
        if proxy and not strict:
            if record_ethnocultural_fallback("religion", f"altele_proxy:{ethnicity}"):
                logger.warning(
                    "No official religion distribution for '%s'. Using 'Altele' proxy.",
                    ethnicity,
                )
            religions = list(proxy.keys())
            weights = list(proxy.values())
            return random.choices(religions, weights=weights, k=1)[0]

        if strict:
            raise ValueError(
                f"Strict mode: missing official religion distribution for '{ethnicity}'."
            )

    if strict:
        raise ValueError(
            "Strict mode: ethnocultural religion tables not available. "
            "Set MOLDOVA_PERSONAS_STRICT_ETHNOCULTURAL=0 to allow estimates."
        )

    logger.warning(
        "Ethnocultural religion tables not available. "
        "Falling back to estimated religion probabilities.",
    )

    # Estimated ethnicity-religion probabilities (see note above)
    # Aligned to official category labels from the NBS ethnocultural tables.
    base_probs = {
        "Moldovean": {
            "Ortodox": 0.95, "Baptist": 0.01, "Penticostal": 0.01,
            "Martor al lui Iehova": 0.005, "Catolic": 0.005,
            "Altă religie": 0.02
        },
        "Român": {
            "Ortodox": 0.93, "Catolic": 0.02, "Baptist": 0.01,
            "Penticostal": 0.01, "Martor al lui Iehova": 0.005,
            "Altă religie": 0.025
        },
        "Ucrainean": {
            "Ortodox": 0.87, "Catolic": 0.08, "Baptist": 0.01,
            "Altă religie": 0.04
        },
        "Găgăuz": {
            "Ortodox": 0.35, "Islam": 0.60, "Altă religie": 0.05
        },
        "Rus": {
            "Ortodox": 0.75, "Baptist": 0.05, "Martor al lui Iehova": 0.05,
            "Penticostal": 0.03, "Altă religie": 0.12
        },
        "Bulgar": {
            "Ortodox": 0.70, "Catolic": 0.15, "Baptist": 0.05,
            "Altă religie": 0.10
        },
        "Rrom": {
            "Ortodox": 0.60, "Penticostal": 0.20, "Baptist": 0.08,
            "Martor al lui Iehova": 0.05, "Altă religie": 0.07
        },
        "Altele": {
            "Ortodox": 0.30, "Catolic": 0.15, "Baptist": 0.05,
            "Altă religie": 0.50
        },
    }
    
    probs = base_probs.get(ethnicity, base_probs["Moldovean"])
    
    religions = list(probs.keys())
    weights = list(probs.values())
    
    return random.choices(religions, weights=weights, k=1)[0]


# Backwards compatibility: export unweighted versions
FIRST_NAMES = {
    ethnicity: {
        sex: [name for name, _ in names]
        for sex, names in sexes.items()
    }
    for ethnicity, sexes in FIRST_NAMES_WEIGHTED.items()
}

LAST_NAMES = {
    ethnicity: [name for name, _ in names]
    for ethnicity, names in LAST_NAMES_WEIGHTED.items()
}
