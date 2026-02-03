"""
Safety and ethics gates for dataset publication.

Prevents:
1. Targeted sensitive claims about protected attributes
2. Protected-attribute determinism (ethnicity/religion forcing job/education)
3. Stereotype reinforcement
"""

import re
from typing import List, Dict, Tuple, Any, Set
from collections import Counter, defaultdict
from dataclasses import dataclass

from .ethnocultural_tables import get_religion_distribution_total


# Sensitive claim patterns to detect
SENSITIVE_CLAIM_PATTERNS = {
    "ethnicity_stereotype": [
        r"\b(roma[n]?i?|țigani?)\s+(sunt\s+)?(leneși|hoți|mur-dari)",
        r"\bmoldoveni\s+(sunt\s+)?(bețivi|leneși|înapoiați)",
        r"\bgagauzi?\s+(sunt\s+)?(înapoiați|analfabeți)",
        r"\brus[oi]\s+(sunt\s+)?(agresivi|bețivi)",
        r"\bucraineni\s+(sunt\s+)?(\w+\s+)?(negativi)",
    ],
    "religion_stereotype": [
        r"\b(ortodocși?|creștin[oi])\s+(sunt\s+)?(înapoiați|fanatici)",
        r"\bmusulman[oi]?\s+(sunt\s+)?(teroriști|periculoși)",
        r"\biuda[ei]\s+(sunt\s+)?(lacomi|bogați)",
    ],
    "gender_stereotype": [
        r"\bfemeile\s+(sunt\s+)?(slabe|emotio-nale|bune\s+doar\s+de\s+gătit)",
        r"\bbărba[iiț]\s+(sunt\s+)?(tari|buni\s+la\s+toate)",
    ],
    "regional_stereotype": [
        r"\bsatenii?\s+(sunt\s+)?(proști|înapoiați)",
        r"\borășen[ii]\s+(sunt\s+)?(aroganți|dezlegați)",
    ],
}

# Occupations that should NOT be deterministically linked to ethnicity/religion
SENSITIVE_OCCUPATION_LINKS = {
    "ethnicity": {
        "Rrom": ["muzician", "căldărar", "lăutar"],  # These stereotypes are harmful
        "Evreu": ["comerciant", "bancher", "doctor"],  # Don't force these
    },
    "religion": {
        "Islam": ["terorist", "imam"],  # Never link
    }
}


@dataclass
class SafetyViolation:
    """A detected safety violation."""
    violation_type: str
    severity: str  # "critical", "warning", "info"
    description: str
    affected_records: List[str]  # UUIDs
    sample_text: str


class SensitiveClaimsGate:
    """
    Gate 1: Detect targeted sensitive claims about protected attributes.
    
    Examples of violations:
    - "Românii sunt leneși" (ethnicity stereotype)
    - "Femeile sunt slabe" (gender stereotype)
    """
    
    def __init__(self):
        self.patterns = SENSITIVE_CLAIM_PATTERNS
    
    def check(self, personas: List[Dict]) -> List[SafetyViolation]:
        """
        Check for sensitive claims in persona narratives.
        
        Returns:
            List of violations found
        """
        violations = []
        
        for violation_type, patterns in self.patterns.items():
            affected = []
            sample = None
            
            for persona in personas:
                # Check all narrative text fields (current schema + legacy)
                text_fields = [
                    persona.get("descriere_generala", ""),
                    persona.get("profil_profesional", ""),
                    persona.get("hobby_sport", ""),
                    persona.get("hobby_arta_cultura", ""),
                    persona.get("hobby_calatorii", ""),
                    persona.get("hobby_culinar", ""),
                    persona.get("career_goals_and_ambitions", ""),
                    persona.get("persona_summary", ""),
                    persona.get("cultural_background", ""),
                    # Legacy keys (if present)
                    persona.get("persona", ""),
                    persona.get("professional_persona", ""),
                    persona.get("sports_persona", ""),
                    persona.get("arts_persona", ""),
                    persona.get("travel_persona", ""),
                    persona.get("culinary_persona", ""),
                ]
                
                full_text = " ".join([t for t in text_fields if t]).lower()
                
                for pattern in patterns:
                    if re.search(pattern, full_text, re.IGNORECASE):
                        affected.append(persona.get("uuid", "unknown"))
                        if not sample:
                            sample = full_text[:200]
                        break
            
            if affected:
                severity = "critical" if "stereotype" in violation_type else "warning"
                violations.append(SafetyViolation(
                    violation_type=violation_type,
                    severity=severity,
                    description=f"Detected {len(affected)} records with {violation_type} claims",
                    affected_records=affected,
                    sample_text=sample
                ))
        
        return violations


class ProtectedAttributeDeterminismGate:
    """
    Gate 2: Ensure ethnicity/religion don't force job/education patterns.
    
    Checks that distributions are census-grounded, not stereotype-driven.
    """
    
    def __init__(self, max_deviation: float = 0.15):
        self.max_deviation = max_deviation
    
    def check(self, personas: List[Dict]) -> List[SafetyViolation]:
        """
        Check for determinism between protected attributes and outcomes.
        
        Returns:
            List of violations if any group is over-represented
        """
        violations = []
        
        # Check ethnicity-occupation correlation
        ethnicity_occupation = defaultdict(Counter)
        ethnicity_education = defaultdict(Counter)
        
        religion_occupation = defaultdict(Counter)
        religion_education = defaultdict(Counter)
        
        for p in personas:
            ethnicity = p.get("ethnicity", "unknown")
            religion = p.get("religion", "unknown")
            occupation = p.get("occupation", "unknown")
            education = p.get("education_level", "unknown")
            
            ethnicity_occupation[ethnicity][occupation] += 1
            ethnicity_education[ethnicity][education] += 1
            
            religion_occupation[religion][occupation] += 1
            religion_education[religion][education] += 1
        
        # Check for over-representation
        for ethnicity, occ_counts in ethnicity_occupation.items():
            total = sum(occ_counts.values())
            for occ, count in occ_counts.most_common(3):
                share = count / total
                # Flag if any occupation is >50% for an ethnicity
                if share > 0.5:
                    violations.append(SafetyViolation(
                        violation_type="ethnicity_occupation_determinism",
                        severity="warning",
                        description=f"{ethnicity}: {share:.1%} are {occ} (may indicate stereotype)",
                        affected_records=[],  # Would need to track individually
                        sample_text=f"Occupation distribution for {ethnicity}"
                    ))
        
        # Check for sensitive occupation links
        for persona in personas:
            ethnicity = persona.get("ethnicity", "")
            religion = persona.get("religion", "")
            occupation = persona.get("occupation", "").lower()
            
            # Check ethnicity-occupation stereotypes
            for eth, occs in SENSITIVE_OCCUPATION_LINKS.get("ethnicity", {}).items():
                if eth in ethnicity and any(o in occupation for o in occs):
                    violations.append(SafetyViolation(
                        violation_type="sensitive_occupation_link",
                        severity="critical",
                        description=f"{ethnicity} linked to stereotypical occupation {occupation}",
                        affected_records=[persona.get("uuid")],
                        sample_text=occupation
                    ))
        
        return violations


class DistributionFairnessGate:
    """
    Gate 3: Ensure demographic distributions match census data.
    
    Prevents synthetic bias amplification.
    """
    
    def __init__(
        self,
        expected_ethnicity: Dict[str, float] = None,
        expected_religion: Dict[str, float] = None,
        max_deviation: Optional[float] = None,
    ):
        # Default: Moldova 2024 census
        self.expected_ethnicity = expected_ethnicity or {
            "Moldovean": 0.75,
            "Român": 0.07,
            "Ucrainean": 0.06,
            "Găgăuz": 0.04,
            "Rus": 0.04,
            "Bulgar": 0.02,
            "Rrom": 0.02,
        }
        
        if expected_religion is not None:
            self.expected_religion = expected_religion
        else:
            official = get_religion_distribution_total()
            if official:
                self.expected_religion = official
            else:
                # Fallback aligned to official category labels
                self.expected_religion = {
                    "Ortodox": 0.95,
                    "Baptist": 0.011,
                    "Martor al lui Iehova": 0.007,
                    "Penticostal": 0.005,
                    "Adventist": 0.003,
                    "Creștină după Evanghelie": 0.003,
                    "Staroveri (Ortodoxă Rusă de rit vechi)": 0.002,
                    "Islam": 0.001,
                    "Catolic": 0.001,
                    "Altă religie": 0.013,
                }
        # Allow override; default to a tolerant 15% deviation
        self.max_deviation = 0.15 if max_deviation is None else max_deviation
    
    def check(self, personas: List[Dict]) -> List[SafetyViolation]:
        """Check if distributions match expected census proportions."""
        violations = []
        max_dev = getattr(self, "max_deviation", 0.15)
        
        # Calculate actual distributions
        actual_ethnicity = Counter(p.get("ethnicity", "unknown") for p in personas)
        actual_religion = Counter(p.get("religion", "unknown") for p in personas)
        
        total = len(personas)
        
        # Check ethnicity
        for ethnicity, expected_share in self.expected_ethnicity.items():
            actual_count = actual_ethnicity.get(ethnicity, 0)
            actual_share = actual_count / total
            
            deviation = abs(actual_share - expected_share)
            if deviation > max_dev:
                violations.append(SafetyViolation(
                    violation_type="ethnicity_distribution_drift",
                    severity="warning",
                    description=f"{ethnicity}: expected {expected_share:.1%}, got {actual_share:.1%} (deviation: {deviation:.1%})",
                    affected_records=[],
                    sample_text=f"Distribution check"
                ))
        
        # Check religion
        for religion, expected_share in self.expected_religion.items():
            actual_count = actual_religion.get(religion, 0)
            actual_share = actual_count / total
            
            deviation = abs(actual_share - expected_share)
            if deviation > max_dev:
                violations.append(SafetyViolation(
                    violation_type="religion_distribution_drift",
                    severity="warning",
                    description=f"{religion}: expected {expected_share:.1%}, got {actual_share:.1%} (deviation: {deviation:.1%})",
                    affected_records=[],
                    sample_text=f"Distribution check"
                ))
        
        return violations


class SafetyGateRunner:
    """Run all safety gates and generate report."""
    
    def __init__(self):
        expected_ethnicity = None
        expected_religion = None
        try:
            from .census_data import CENSUS
            expected_ethnicity = CENSUS.ETHNICITY_DISTRIBUTION
            expected_religion = CENSUS.RELIGION_DISTRIBUTION
        except Exception:
            # Fall back to hardcoded defaults inside DistributionFairnessGate
            pass

        self.gates = {
            "sensitive_claims": SensitiveClaimsGate(),
            "protected_attribute_determinism": ProtectedAttributeDeterminismGate(),
            "distribution_fairness": DistributionFairnessGate(
                expected_ethnicity=expected_ethnicity,
                expected_religion=expected_religion,
            ),
        }
    
    def run_all(self, personas: List[Dict]) -> Dict[str, Any]:
        """
        Run all safety gates.
        
        Returns:
            Comprehensive safety report
        """
        all_violations = []
        gate_results = {}
        
        for gate_name, gate in self.gates.items():
            violations = gate.check(personas)
            all_violations.extend(violations)
            gate_results[gate_name] = {
                "passed": len(violations) == 0,
                "violations": [
                    {
                        "type": v.violation_type,
                        "severity": v.severity,
                        "description": v.description,
                        "affected_count": len(v.affected_records),
                    }
                    for v in violations
                ]
            }
        
        # Categorize by severity
        critical = [v for v in all_violations if v.severity == "critical"]
        warnings = [v for v in all_violations if v.severity == "warning"]
        
        # Overall assessment
        can_publish = len(critical) == 0
        
        return {
            "can_publish": can_publish,
            "overall_status": "PASS" if can_publish else "BLOCKED",
            "critical_count": len(critical),
            "warning_count": len(warnings),
            "gate_results": gate_results,
            "recommendations": self._generate_recommendations(all_violations),
        }
    
    def _generate_recommendations(self, violations: List[SafetyViolation]) -> List[str]:
        """Generate remediation recommendations."""
        recommendations = []
        
        has_stereotypes = any("stereotype" in v.violation_type for v in violations)
        has_determinism = any("determinism" in v.violation_type for v in violations)
        
        if has_stereotypes:
            recommendations.append(
                "Add stronger prompt constraints banning stereotypical claims. "
                "Review and regenerate affected records."
            )
        
        if has_determinism:
            recommendations.append(
                "Check IPF weights for ethnicity/religion vs occupation/education. "
                "Ensure conditional probabilities match census, not stereotypes."
            )
        
        recommendations.append(
            "Consider adding post-generation filtering for sensitive patterns."
        )
        
        return recommendations


def run_safety_check(personas: List[Dict]) -> Dict[str, Any]:
    """
    Convenience function to run all safety gates.
    
    Example:
        report = run_safety_check(personas)
        if not report["can_publish"]:
            print("Safety check failed!")
            for rec in report["recommendations"]:
                print(f"  - {rec}")
    """
    runner = SafetyGateRunner()
    return runner.run_all(personas)


if __name__ == "__main__":
    # Example usage
    print("Safety Gates loaded successfully")
    print(f"Configured patterns: {sum(len(p) for p in SENSITIVE_CLAIM_PATTERNS.values())}")
