"""
Core persona generation engine using PGM (Probabilistic Graphical Models)
and IPF (Iterative Proportional Fitting).

Implements the dependency graph from the technical plan:
    Region → Ethnicity → Language → Name
    Age → Education → Occupation
    Location → Occupation
"""

import random
import uuid
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np

from .census_data import CENSUS, CensusDistributions
from .names import (
    generate_name,
    get_language_by_ethnicity,
    get_religion_by_ethnicity,
    reset_ethnocultural_fallbacks,
)
from .ocean_framework import OCEANSampler, OCEANBehaviorMapper
from .ocean_nemo_schema import convert_to_nemo_schema
from .geo_tables import (
    strict_geo_enabled,
    get_district_distribution,
    get_region_for_district,
)
from .models import Persona, get_age_group, AgeConstraints, PopulationMode


class PersonaGenerator:
    """
    Generator for synthetic Moldovan personas.
    
    Uses conditional probability sampling to ensure realistic correlations
    between demographic variables.
    """
    
    def __init__(
        self,
        census_data: Optional[CensusDistributions] = None,
        seed: Optional[int] = None,
        population_mode: PopulationMode = PopulationMode.ADULT_18,
        include_ocean: bool = True,
    ):
        """
        Initialize the generator.
        
        Args:
            census_data: Census distributions to use (defaults to CENSUS singleton)
            seed: Random seed for reproducibility
        """
        self.census = census_data or CENSUS
        self.population_mode = population_mode
        self.include_ocean = include_ocean
        if self.population_mode != PopulationMode.ADULT_18:
            raise ValueError("PopulationMode is adult-only; non-adult modes are not supported yet.")
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.ocean_sampler = OCEANSampler(seed=seed) if self.include_ocean else None
        
        # Pre-compute city/district data for realistic location generation
        self._init_location_data()
    
    def _init_location_data(self) -> None:
        """Initialize realistic locality data by region from config files."""
        import json
        from pathlib import Path
        
        config_dir = Path(__file__).parent.parent / "config"
        
        # Load localities with settlement type metadata
        localities_path = config_dir / "localities_by_region.json"
        if localities_path.exists():
            with open(localities_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Parse localities with metadata
                self.localities_by_region: Dict[str, List[Dict]] = {
                    region: info["localities"] 
                    for region, info in data.items() 
                    if not region.startswith("_")
                }
        else:
            # Fallback to legacy cities config
            cities_path = config_dir / "cities_by_region.json"
            if cities_path.exists():
                with open(cities_path, 'r', encoding='utf-8') as f:
                    cities = json.load(f)
                # Convert to new format without implied_residence
                self.localities_by_region = {
                    region: [{"name": city, "settlement_type": "unknown", "implied_residence": None}
                            for city in cities_list]
                    for region, cities_list in cities.items()
                }
            else:
                # Ultimate fallback
                self.localities_by_region = {
                    "Chisinau": [{"name": "Chișinău", "settlement_type": "city", "implied_residence": "Urban"}],
                    "Centru": [{"name": "Ungheni", "settlement_type": "city", "implied_residence": "Urban"}],
                    "Nord": [{"name": "Bălți", "settlement_type": "city", "implied_residence": "Urban"}],
                    "Sud": [{"name": "Cahul", "settlement_type": "city", "implied_residence": "Urban"}],
                    "Gagauzia": [{"name": "Comrat", "settlement_type": "city", "implied_residence": "Urban"}],
                }
        
        # Also load legacy cities for backwards compatibility
        cities_path = config_dir / "cities_by_region.json"
        if cities_path.exists():
            with open(cities_path, 'r', encoding='utf-8') as f:
                self.cities_by_region: Dict[str, List[str]] = json.load(f)
        else:
            self.cities_by_region = {
                region: [loc["name"] for loc in localities]
                for region, localities in self.localities_by_region.items()
            }
        
        # Load districts by region
        districts_path = config_dir / "districts_by_region.json"
        if districts_path.exists():
            with open(districts_path, 'r', encoding='utf-8') as f:
                self.districts_by_region: Dict[str, List[str]] = json.load(f)
        else:
            # Fallback to default data
            self.districts_by_region = {
                "Chisinau": ["Mun. Chișinău"],
                "Centru": ["Ungheni", "Călărași", "Nisporeni", "Strășeni", "Criuleni",
                          "Dubăsari", "Orhei", "Rezina", "Telenești"],
                "Nord": ["Mun. Bălți", "Soroca", "Edineț", "Ocnița", "Dondușeni", "Drochia",
                        "Fălești", "Florești", "Glodeni", "Rîșcani", "Sîngerei"],
                "Sud": ["Cahul", "Cantemir", "Cimișlia", "Leova", "Taraclia", "Basarabeasca"],
                "Gagauzia": ["UTA Găgăuzia"],
            }
    
    def _sample_from_dict(self, distribution: Dict[str, float]) -> str:
        """
        Sample a key from a probability distribution.
        
        Args:
            distribution: Dict mapping outcomes to probabilities
        
        Returns:
            Sampled key
        """
        keys = list(distribution.keys())
        probabilities = list(distribution.values())
        return random.choices(keys, weights=probabilities, k=1)[0]
    
    def _min_persona_age(self) -> int:
        """Minimum age based on population mode."""
        if self.population_mode == PopulationMode.ADULT_18:
            return AgeConstraints.MIN_PERSONA_AGE
        if self.population_mode == PopulationMode.AGE_15_PLUS:
            return 15
        return 0

    def _sample_age_from_group(self, age_group: str) -> int:
        """
        Sample a specific age within an age group using realistic distribution.
        
        Uses a slightly bell-shaped distribution within each group rather than
        uniform random. Minimum age is enforced by population mode.
        
        Args:
            age_group: Age group string (e.g., "25-34")
        
        Returns:
            Specific age in years (minimum 18)
        """
        ranges = {
            "0-14": (0, 14),
            "15-24": (15, 24),
            "25-34": (25, 34),
            "35-44": (35, 44),
            "45-54": (45, 54),
            "55-64": (55, 64),
            "65+": (65, AgeConstraints.MAX_REALISTIC_AGE),
        }
        min_age, max_age = ranges.get(age_group, (25, 65))
        
        # Enforce minimum age for personas
        min_age = max(min_age, self._min_persona_age())

        # Prefer single-age sampling when available
        try:
            single_age_counts = self.census._load_single_age_counts()
        except Exception:
            single_age_counts = None
        if single_age_counts:
            ages = [age for age in range(min_age, max_age + 1) if age in single_age_counts]
            if ages:
                weights = [single_age_counts[age] for age in ages]
                if sum(weights) > 0:
                    return random.choices(ages, weights=weights, k=1)[0]
        
        # Use triangular distribution for more realistic age distribution within group
        mode = (min_age + max_age) // 2
        return int(random.triangular(min_age, max_age, mode))
    
    def _generate_occupation(self, age: int, education: str, 
                            residence_type: str, region: str) -> Tuple[str, Optional[str]]:
        """
        Generate realistic occupation based on age, education, and location.
        
        Args:
            age: Age in years
            education: Education level
            residence_type: Urban or Rural
            region: Development region
        
        Returns:
            Tuple of (occupation, sector)
        """
        # Handle special cases by age using census-based probabilities
        # Source: NBS 2024 Census - Employment Status by Age Group
        if age < 22 and education in ["Liceal", "Superior (Licență/Master)", "Doctorat"]:
            # For 15-24 age group: 60% students (from EMPLOYMENT_STATUS_BY_AGE)
            # Younger ages (< 22) with higher education have higher student probability
            if random.random() < 0.75:  # Estimated: 75% for this specific subgroup
                return "Student", "Educație"
        elif age >= 63:
            # For 65+ age group: 75% retired (from EMPLOYMENT_STATUS_BY_AGE census data)
            # Age 63-64 has lower retirement rate, age 65+ uses census rate
            retirement_rate = 0.55 if age == 63 else (0.65 if age == 64 else 0.75)
            if random.random() < retirement_rate:
                # Sometimes include former profession
                former_jobs = ["profesor", "inginer", "medic", "contabil", "muncitor"]
                return f"Pensionar (fost: {random.choice(former_jobs)})", None
        
        # Education-based occupation selection
        high_skill_jobs = [
            "Profesor", "Medic", "Inginer", "Avocat", "Contabil", "Manager",
            "Programator", "Farmacist", "Arhitect", "Economist", "Psiholog"
        ]
        
        medium_skill_jobs = [
            "Învățător", "Asistent medical", "Tehnician", "Secretar", "Vânzător",
            "Șofer", "Bucătar", "Croitoreasă", "Electrician", "Instalator",
            "Mecanic", "Coafor", "Asistent social"
        ]
        
        low_skill_jobs = [
            "Muncitor necalificat", "Agricultor", "Personal de curățenie", "Paznic",
            "Îngrijitor", "Lăcătuș", "Tâmplar", "Zidar", "Ospătar"
        ]
        
        rural_jobs = [
            "Agricultor", "Fermier", "Meșteșugar", "Învățător", "Asistent medical",
            "Comerciant mic", "Preot", "Îngrijitor", "Muncitor agricol"
        ]
        
        urban_jobs = [
            "Funcționar public", "Vânzător", "Șofer", "Bucătar", "Contabil",
            "Manager", "Programator", "Consultant", "Agent de vânzări"
        ]
        
        # Apply location bias
        if residence_type == "Rural":
            if random.random() < 0.4:  # 40% agricultural in rural
                sector = "Agricultură"
                job = random.choice(rural_jobs)
            else:
                sector = "Servicii"
                job = random.choice(low_skill_jobs + medium_skill_jobs[:5])
        else:  # Urban
            if education in ["Superior (Licență/Master)", "Doctorat"]:
                sector = "Servicii"
                job = random.choice(high_skill_jobs)
            elif education == "Profesional/Tehnic":
                sector = random.choice(["Industrie", "Servicii"])
                job = random.choice(medium_skill_jobs)
            else:
                sector = "Servicii"
                job = random.choice(low_skill_jobs + medium_skill_jobs[:8])
        
        # Regional adjustments
        if region == "Chisinau" and education in ["Superior (Licență/Master)", "Doctorat"]:
            if random.random() < 0.3:  # Higher chance of IT/finance in capital
                job = random.choice(["Programator", "Analist financiar", "Consultant IT"])
                sector = "Servicii"
        
        return job, sector
    
    def _generate_employment_status(self, age: int, occupation: str) -> str:
        """
        Generate employment status based on age and current occupation.
        
        Uses the EMPLOYMENT_STATUS_BY_AGE distribution from census data.
        Also considers occupation hints (e.g., "Student", "Pensionar").
        
        Args:
            age: Age in years
            occupation: Current occupation (may hint at status)
            
        Returns:
            Employment status string
        """
        # First check occupation hints
        if "Student" in occupation:
            return "student"
        if "Pensionar" in occupation:
            return "retired"
        
        # Get age group and corresponding distribution
        age_group = self._get_age_group(age)
        status_dist = self.census.EMPLOYMENT_STATUS_BY_AGE.get(
            age_group, 
            {"employed": 0.427, "unemployed": 0.018, "student": 0.08,
             "retired": 0.183, "homemaker": 0.06, "other_inactive": 0.232}
        )
        
        return self._sample_from_dict(status_dist)
    
    def _generate_realistic_education(self, age: int) -> str:
        """
        Generate education level that is realistic for the given age.
        
        Uses AgeConstraints for consistent minimum ages.
        
        Args:
            age: Age in years
        
        Returns:
            Education level string
        """
        ac = AgeConstraints  # Shorthand for readability
        
        # Age-based constraints using centralized constants
        if age < ac.MIN_PRIMARY_SCHOOL:
            # Before school age
            return "Fără studii"
        
        elif age < ac.MIN_GYMNASIUM:
            # Early primary (age 6-7)
            return random.choices(
                ["Fără studii", "Primar"],
                weights=[0.2, 0.8]
            )[0]
        
        elif age < 11:
            # Primary school age
            return "Primar"
        
        elif age < ac.MIN_HIGH_SCHOOL:
            # Gymnasium age
            return random.choices(
                ["Primar", "Gimnazial"],
                weights=[0.1, 0.9]
            )[0]
        
        elif age < 16:
            # Can have started high school or vocational
            return random.choices(
                ["Gimnazial", "Liceal", "Profesional/Tehnic"],
                weights=[0.3, 0.4, 0.3]
            )[0]
        
        elif age < ac.MIN_HIGHER_EDUCATION:
            # High school/vocational age (16-18), NO university yet
            return random.choices(
                ["Gimnazial", "Liceal", "Profesional/Tehnic"],
                weights=[0.15, 0.45, 0.40]
            )[0]
        
        elif age < ac.MIN_DOCTORATE:
            # University age or early career - can have superior, no doctorate yet
            return random.choices(
                ["Gimnazial", "Liceal", "Profesional/Tehnic", "Superior (Licență/Master)"],
                weights=[0.05, 0.25, 0.25, 0.45]
            )[0]
        
        else:
            # Use the census distribution for the age group
            # But filter out education levels that are too advanced for the age
            age_group = self._get_age_group(age)
            dist = dict(self.census.EDUCATION_BY_AGE_GROUP[age_group])
            
            # Filter: no Superior for ages < MIN_HIGHER_EDUCATION
            if age < ac.MIN_HIGHER_EDUCATION:
                dist.pop("Superior (Licență/Master)", None)
                dist.pop("Doctorat", None)
            # Filter: no Doctorat for ages < MIN_DOCTORATE
            elif age < ac.MIN_DOCTORATE:
                dist.pop("Doctorat", None)
            
            # Renormalize if we removed anything
            total = sum(dist.values())
            if total > 0:
                dist = {k: v / total for k, v in dist.items()}
            
            return self._sample_from_dict(dist)
    
    def _get_age_group(self, age: int) -> str:
        """Map age to age group."""
        if age < 15:
            return "0-14"
        elif age < 25:
            return "15-24"
        elif age < 35:
            return "25-34"
        elif age < 45:
            return "35-44"
        elif age < 55:
            return "45-54"
        elif age < 65:
            return "55-64"
        else:
            return "65+"

    def _generate_field_of_study(self, education: str) -> Optional[str]:
        """
        Generate field of study for higher education.
        
        Args:
            education: Education level
        
        Returns:
            Field name or None if not applicable
        """
        if education not in ["Superior (Licență/Master)", "Doctorat"]:
            return None
        
        fields = [
            ("Educație/Pedagogie", 0.20),
            ("Medicină/Sănătate", 0.15),
            ("Economie/Finanțe", 0.15),
            ("Inginerie", 0.14),
            ("Drept", 0.08),
            ("IT/Informatică", 0.10),
            ("Științe Sociale", 0.08),
            ("Agricultură", 0.05),
            ("Arte/Umanistice", 0.05),
        ]
        
        names, weights = zip(*fields)
        return random.choices(names, weights=weights, k=1)[0]
    
    def generate_single(self) -> Persona:
        """
        Generate a single persona with all structured fields.
        
        Follows the dependency graph:
            Region → Urban/Rural
            Region → Ethnicity → Language, Religion
            Sex + Ethnicity → Name
            Age → Education → Occupation
            Location → Occupation
        
        Returns:
            Persona with structured fields populated
        """
        # Step 1: Sample Region (top of hierarchy) or District if available
        strict_geo = strict_geo_enabled()
        district_distribution = get_district_distribution() if strict_geo else None
        district = ""

        if strict_geo and district_distribution:
            # Sample district from official distribution, derive region
            district = self._sample_from_dict(district_distribution)
            region = get_region_for_district(district, strict=True)
        else:
            region = self._sample_from_dict(self.census.REGION_DISTRIBUTION)
        
        # Step 2: Sample Urban/Rural (conditional on region)
        urban_rural_dist = self.census.REGION_URBAN_CROSS[region]
        residence_type = self._sample_from_dict(urban_rural_dist)
        
        # Step 3: Sample Ethnicity (conditional on region)
        ethnicity_dist = self.census.ETHNICITY_BY_REGION[region]
        ethnicity = self._sample_from_dict(ethnicity_dist)
        
        # Step 4: Derive Language and Religion from ethnicity
        mother_tongue = get_language_by_ethnicity(ethnicity, region, residence_type)
        religion = get_religion_by_ethnicity(ethnicity, region)
        
        # Step 5: Sample Sex (BEFORE name - name depends on sex)
        sex = self._sample_from_dict(self.census.SEX_DISTRIBUTION)
        
        # Step 6: Generate Name (based on ethnicity and sex - now sex is known)
        name = generate_name(ethnicity, sex)
        
        # Step 7: Sample Age Group, then specific age (mode-specific distribution)
        age_group_dist = self.census.age_group_distribution(self.population_mode)
        age_group = self._sample_from_dict(age_group_dist)
        age = self._sample_age_from_group(age_group)
        
        # Step 8: Sample Education (conditional on age group with realistic constraints)
        education = self._generate_realistic_education(age)
        
        # Step 9: Sample Marital Status (conditional on age)
        marital_dist = self.census.MARITAL_STATUS_BY_AGE[age_group]
        marital_status = self._sample_from_dict(marital_dist)
        
        # Step 10: Generate Occupation (conditional on age, education, location)
        occupation, sector = self._generate_occupation(
            age, education, residence_type, region
        )
        
        # Step 11: Generate Employment Status (conditional on age)
        employment_status = self._generate_employment_status(age, occupation)
        
        # Step 12: Generate Field of Study (if applicable)
        field_of_study = self._generate_field_of_study(education)

        # Step 12b: Sample OCEAN personality and behavioral contract (optional)
        ocean_scores = None
        ocean_profile = None
        behavioral_contract = None
        ocean_source = None
        ocean_confidence = None
        if self.include_ocean and self.ocean_sampler:
            ocean = self.ocean_sampler.sample(
                age=age,
                sex=sex,
                education_level=education,
                occupation=occupation,
            )
            ocean_scores = {
                "openness": ocean.openness,
                "conscientiousness": ocean.conscientiousness,
                "extraversion": ocean.extraversion,
                "agreeableness": ocean.agreeableness,
                "neuroticism": ocean.neuroticism,
            }
            ocean_profile = convert_to_nemo_schema(ocean_scores)
            behavioral_contract = OCEANBehaviorMapper.generate_behavioral_contract(ocean)
            if "ocean_profile" in behavioral_contract:
                behavioral_contract = {
                    k: v for k, v in behavioral_contract.items() if k != "ocean_profile"
                }
            ocean_source = ocean.source
            ocean_confidence = ocean.confidence
        
        # Step 13: Generate Location details (strict geo avoids fabricated localities)
        if strict_geo:
            city = ""
            if not district:
                district = ""
        else:
            # Use weighted city selection based on residence type
            city = self._select_city(region, residence_type)
            district = random.choice(self.districts_by_region[region])
        
        # Create and return persona
        return Persona(
            uuid=str(uuid.uuid4()),
            name=name,
            sex=sex,
            age=age,
            age_group=age_group,
            ethnicity=ethnicity,
            mother_tongue=mother_tongue,
            religion=religion,
            marital_status=marital_status,
            education_level=education,
            field_of_study=field_of_study,
            occupation=occupation,
            occupation_sector=sector,
            employment_status=employment_status,
            city=city,
            district=district,
            region=region,
            residence_type=residence_type,
            country="Moldova",
            ocean_openness=ocean_scores["openness"] if ocean_scores else None,
            ocean_conscientiousness=ocean_scores["conscientiousness"] if ocean_scores else None,
            ocean_extraversion=ocean_scores["extraversion"] if ocean_scores else None,
            ocean_agreeableness=ocean_scores["agreeableness"] if ocean_scores else None,
            ocean_neuroticism=ocean_scores["neuroticism"] if ocean_scores else None,
            ocean_source=ocean_source,
            ocean_confidence=ocean_confidence,
            ocean_profile=ocean_profile,
            behavioral_contract=behavioral_contract,
        )
    
    def _select_city(self, region: str, residence_type: str) -> str:
        """
        Select a city/locality consistent with residence_type.
        
        Uses locality metadata to ensure consistency:
        - Urban personas get cities and urban towns
        - Rural personas get rural towns and villages
        
        Args:
            region: Development region
            residence_type: "Urban" or "Rural"
            
        Returns:
            Locality name
            
        Raises:
            ValueError: If no suitable localities found for residence_type
        """
        localities = self.localities_by_region.get(region, [])
        
        if not localities:
            return "Chișinău"
        
        # Filter localities by implied_residence if available
        # implied_residence in config should match residence_type
        matching_localities = []
        fallback_localities = []
        
        for loc in localities:
            implied = loc.get("implied_residence")
            if implied == residence_type:
                matching_localities.append(loc)
            elif implied is None:
                # No metadata - use as fallback
                fallback_localities.append(loc)
            # Mismatched implied_residence is excluded (e.g., Rural + Chișinău)
        
        # If we have matching localities with metadata, use those
        if matching_localities:
            candidates = matching_localities
        elif fallback_localities:
            # No metadata available - use fallback with legacy weights
            candidates = fallback_localities
        else:
            # No matching localities found - this is a config error
            # Fall back to legacy cities but log a warning
            cities = self.cities_by_region.get(region, ["Chișinău"])
            # For rural, exclude first city (usually capital)
            if residence_type == "Rural" and len(cities) > 1:
                return random.choice(cities[1:])
            return cities[0] if cities else "Chișinău"
        
        # Apply weighted sampling based on settlement_type
        # Cities get higher weight for urban, lower for rural
        weights = []
        for loc in candidates:
            settlement_type = loc.get("settlement_type", "unknown")
            if residence_type == "Urban":
                if settlement_type == "city":
                    weights.append(0.7)
                elif settlement_type == "town":
                    weights.append(0.25)
                else:  # rural or unknown
                    weights.append(0.05)
            else:  # Rural
                if settlement_type == "rural":
                    weights.append(0.6)
                elif settlement_type == "town":
                    weights.append(0.35)
                else:  # city or unknown (avoid cities for rural)
                    weights.append(0.05)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = None  # uniform
        
        names = [loc["name"] for loc in candidates]
        return random.choices(names, weights=weights, k=1)[0]
    
    def generate(self, n: int, show_progress: bool = True, 
                 use_ethnicity_correction: bool = True) -> List[Persona]:
        """
        Generate multiple personas.
        
        Args:
            n: Number of personas to generate
            show_progress: Whether to show progress bar
            use_ethnicity_correction: Whether to apply multi-marginal raking correction
        
        Returns:
            List of Persona objects
        """
        reset_ethnocultural_fallbacks()

        if use_ethnicity_correction:
            return self.generate_with_ethnicity_correction(n, oversample_factor=3)
        
        iterator = range(n)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating personas", total=n)
        
        personas = []
        for _ in iterator:
            personas.append(self.generate_single())
        
        return personas
    
    def generate_with_ipf_adjustment(
        self,
        n: int,
        max_iterations: int = 100,
        return_metrics: bool = False,
    ) -> List[Persona] | Tuple[List[Persona], "IPFMetrics"]:
        """
        Generate personas with multi-marginal raking adjustment.
        
        This ensures the final sample more closely matches target distributions
        by iteratively adjusting sampling weights across multiple marginals.
        
        Args:
            n: Number of personas to generate
            max_iterations: Maximum raking iterations
        
        Returns:
            List of Persona objects adjusted to match targets, or tuple with metrics if return_metrics=True
        """
        from .trust_report import IPFMetrics

        # Generate initial oversample
        oversample_factor = 2
        # Avoid double-correction: generate without raking
        initial_sample = self.generate(
            n * oversample_factor,
            show_progress=True,
            use_ethnicity_correction=False,
        )
        
        # Apply raking adjustment through resampling
        adjusted, metrics = self._rake_resample(initial_sample, n, max_iterations=max_iterations)
        
        if return_metrics:
            return adjusted, metrics
        return adjusted
    
    def _get_raking_targets(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, callable]]:
        """Return target distributions and extractors for raking."""
        targets: Dict[str, Dict[str, float]] = {
            "sex": self.census.SEX_DISTRIBUTION,
            "age_group": self.census.age_group_distribution(self.population_mode),
            "residence_type": self.census.RESIDENCE_TYPE_DISTRIBUTION,
            "ethnicity": self.census.ETHNICITY_DISTRIBUTION,
        }
        extractors: Dict[str, callable] = {
            "sex": lambda p: p.sex,
            "age_group": lambda p: p.age_group,
            "residence_type": lambda p: p.residence_type,
            "ethnicity": lambda p: p.ethnicity,
        }
        # Prefer district targets in strict geo mode when available
        if strict_geo_enabled():
            district_target = get_district_distribution()
            if district_target:
                targets["district"] = district_target
                extractors["district"] = lambda p: p.district if p.district else "Unknown"
            else:
                targets["region"] = self.census.REGION_DISTRIBUTION
                extractors["region"] = lambda p: p.region
        else:
            targets["region"] = self.census.REGION_DISTRIBUTION
            extractors["region"] = lambda p: p.region
        return targets, extractors

    def _compute_weighted_dist(
        self,
        sample: List[Persona],
        weights: List[float],
        extractor,
    ) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        total_w = sum(weights)
        if total_w <= 0:
            return {}
        for p, w in zip(sample, weights):
            key = extractor(p)
            totals[key] = totals.get(key, 0.0) + w
        return {k: v / total_w for k, v in totals.items()}

    def _compute_unweighted_dist(
        self,
        sample: List[Persona],
        extractor,
    ) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for p in sample:
            key = extractor(p)
            totals[key] = totals.get(key, 0) + 1
        total = len(sample)
        return {k: v / total for k, v in totals.items()} if total > 0 else {}

    def _rake_weights(
        self,
        sample: List[Persona],
        targets: Dict[str, Dict[str, float]],
        extractors: Dict[str, callable],
        max_iterations: int = 50,
        tol: float = 1e-6,
        weight_cap: Optional[Tuple[float, float]] = (0.2, 5.0),
    ) -> Tuple[List[float], Dict[str, float], int, bool, float]:
        """Iteratively rake weights to match target marginals."""
        weights = [1.0 for _ in sample]
        converged = False
        field_errors: Dict[str, float] = {}

        for iteration in range(1, max_iterations + 1):
            for field, target in targets.items():
                extractor = extractors[field]
                current = self._compute_weighted_dist(sample, weights, extractor)
                for i, persona in enumerate(sample):
                    key = extractor(persona)
                    cur = current.get(key, 0.0)
                    if cur <= 0:
                        continue
                    adj = target.get(key, 0.0) / cur
                    weights[i] *= adj
                if weight_cap:
                    min_w, max_w = weight_cap
                    weights = [min(max(w, min_w), max_w) for w in weights]

            # Evaluate convergence after full pass
            max_err = 0.0
            field_errors = {}
            for field, target in targets.items():
                extractor = extractors[field]
                current = self._compute_weighted_dist(sample, weights, extractor)
                keys = set(current.keys()) | set(target.keys())
                err = max((abs(current.get(k, 0.0) - target.get(k, 0.0)) for k in keys), default=0.0)
                field_errors[field] = err
                max_err = max(max_err, err)

            if max_err < tol:
                converged = True
                break

        return weights, field_errors, iteration, converged, max_err

    def _weighted_resample(
        self,
        sample: List[Persona],
        weights: List[float],
        target_n: int,
    ) -> List[Persona]:
        """Weighted resampling with replacement."""
        if not sample:
            return []
        return random.choices(sample, weights=weights, k=target_n)

    def _rake_resample(
        self,
        sample: List[Persona],
        target_n: int,
        max_iterations: int = 50,
    ) -> Tuple[List[Persona], "IPFMetrics"]:
        """
        Resample from initial sample to better match multiple target distributions.
        
        Args:
            sample: Initial oversampled personas
            target_n: Target number of personas
            max_iterations: Maximum raking iterations
        
        Returns:
            Tuple of (resampled personas, IPFMetrics with ESS and weight concentration)
        """
        from .trust_report import IPFMetrics
        
        targets, extractors = self._get_raking_targets()
        weights, field_errors, iterations, converged, max_err = self._rake_weights(
            sample,
            targets,
            extractors,
            max_iterations=max_iterations,
        )

        # Compute ESS and weight concentration metrics
        weights_array = np.array(weights, dtype=float)
        sum_weights = float(np.sum(weights_array))
        sum_weights_sq = float(np.sum(weights_array ** 2))
        effective_sample_size = (sum_weights ** 2) / sum_weights_sq if sum_weights_sq > 0 else 0.0
        mean_weight = float(np.mean(weights_array)) if len(weights_array) > 0 else 0.0
        max_weight = float(np.max(weights_array)) if len(weights_array) > 0 else 0.0
        weight_concentration = max_weight / mean_weight if mean_weight > 0 else 1.0

        # Top 5% weight share
        top_k = max(1, int(0.05 * len(weights_array)))
        top_5_share = float(np.sum(np.sort(weights_array)[-top_k:]) / sum_weights) if sum_weights > 0 else 0.0

        # Pre-correction drift per field
        pre_correction_drift = {}
        for field, target in targets.items():
            current = self._compute_unweighted_dist(sample, extractors[field])
            pre_correction_drift[field] = sum(
                abs(current.get(k, 0.0) - target.get(k, 0.0))
                for k in set(current) | set(target)
            )

        # Resample
        result = self._weighted_resample(sample, weights, target_n)

        # Post-correction drift per field
        post_correction_drift = {}
        for field, target in targets.items():
            current = self._compute_unweighted_dist(result, extractors[field])
            post_correction_drift[field] = sum(
                abs(current.get(k, 0.0) - target.get(k, 0.0))
                for k in set(current) | set(target)
            )
        
        # Create metrics object
        metrics = IPFMetrics(
            original_sample_size=len(sample),
            effective_sample_size=float(effective_sample_size),
            resampling_ratio=target_n / len(sample) if len(sample) > 0 else 0,
            pre_correction_drift=pre_correction_drift,
            post_correction_drift=post_correction_drift,
            correction_iterations=iterations,
        )
        
        # Add raking diagnostics
        metrics.weight_concentration = weight_concentration
        metrics.top_5_weight_share = top_5_share
        metrics.raking_fields = list(targets.keys())
        metrics.raking_converged = converged
        metrics.max_marginal_error = max_err
        metrics.marginal_error_by_field = field_errors
        metrics.weight_cap = (0.2, 5.0)
        
        return result[:target_n], metrics
    
    def generate_with_ethnicity_correction(
        self, 
        n: int, 
        oversample_factor: int = 3,
        return_metrics: bool = False
    ) -> List[Persona] | Tuple[List[Persona], "IPFMetrics"]:
        """
        Generate personas with multi-marginal raking correction.
        
        Uses raking to ensure the final sample matches multiple target
        distributions while preserving correlations.
        
        Args:
            n: Number of personas to generate
            oversample_factor: How many times to oversample for raking
            return_metrics: If True, return (personas, metrics) tuple
        
        Returns:
            List of Persona objects, or tuple with metrics if return_metrics=True
        """
        from .trust_report import IPFMetrics

        reset_ethnocultural_fallbacks()
        
        # Generate initial oversample (without correction to avoid recursion)
        initial_sample = []
        target_n = n * oversample_factor
        for _ in tqdm(range(target_n), desc="Generating personas", total=target_n):
            initial_sample.append(self.generate_single())
        
        # Apply raking correction
        corrected, metrics = self._rake_resample(initial_sample, n)
        
        if return_metrics:
            return corrected, metrics
        return corrected
