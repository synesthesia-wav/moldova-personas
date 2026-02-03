"""Statistical validation tests for persona distributions.

Implements formal statistical tests to validate generated personas
against census distributions:
- Chi-square goodness-of-fit tests
- Joint distribution validation
- Confidence interval-based tolerance
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import math

from .models import Persona, AgeConstraints
from .geo_tables import (
    get_district_distribution,
    get_region_distribution_from_district,
    strict_geo_enabled,
)
from .census_data import CENSUS, CensusDistributions


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: int
    passed: bool
    details: str
    alpha: float = 0.05
    
    def summary(self) -> str:
        """Human-readable summary of test result."""
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return (
            f"{self.test_name}: {status}\n"
            f"  Statistic: {self.statistic:.4f}\n"
            f"  p-value: {self.p_value:.4f}\n"
            f"  df: {self.degrees_of_freedom}\n"
            f"  {self.details}"
        )


class StatisticalValidator:
    """Formal statistical validation for persona distributions."""
    
    def __init__(self, census_data: Optional[CensusDistributions] = None, alpha: float = 0.05):
        """
        Initialize validator.
        
        Args:
            census_data: Census distributions to test against
            alpha: Significance level for tests (default 0.05)
        """
        self.census = census_data or CENSUS
        self.alpha = alpha
    
    def chi_square_goodness_of_fit(
        self, 
        observed: Dict[str, int], 
        expected_probs: Dict[str, float],
        n: int
    ) -> StatisticalTestResult:
        """
        Perform chi-square goodness-of-fit test.
        
        H0: Observed distribution matches expected distribution
        H1: Observed distribution differs from expected
        
        Args:
            observed: Observed counts by category
            expected_probs: Expected probabilities by category
            n: Total sample size
            
        Returns:
            StatisticalTestResult with test outcome
        """
        # Ensure all categories present
        all_categories = set(observed.keys()) | set(expected_probs.keys())
        
        chi2_stat = 0.0
        df = 0
        details_parts = []
        
        for category in all_categories:
            obs = observed.get(category, 0)
            exp_prob = expected_probs.get(category, 0)
            exp_count = n * exp_prob
            
            # Skip if expected count < 5 (chi-square assumption violation)
            if exp_count < 5:
                if obs > 0:
                    details_parts.append(f"{category}: exp={exp_count:.1f}<5, skipped")
                continue
            
            if exp_count > 0:
                contribution = ((obs - exp_count) ** 2) / exp_count
                chi2_stat += contribution
                df += 1
                
                # Flag large contributions
                if contribution > 3.84:  # Significant at p<0.05 for 1 df
                    details_parts.append(
                        f"{category}: obs={obs}, exp={exp_count:.1f}, χ²={contribution:.2f}"
                    )
        
        # Calculate p-value using chi-square approximation
        if df > 0:
            from math import gamma
            # Use incomplete gamma function for p-value
            p_value = self._chi2_sf(chi2_stat, df)
        else:
            p_value = 1.0
            
        passed = p_value >= self.alpha
        
        details = "; ".join(details_parts) if details_parts else "All categories within expected ranges"
        
        return StatisticalTestResult(
            test_name="Chi-Square Goodness-of-Fit",
            statistic=chi2_stat,
            p_value=p_value,
            degrees_of_freedom=df - 1 if df > 0 else 0,  # df = categories - 1
            passed=passed,
            details=details,
            alpha=self.alpha
        )
    
    def _chi2_sf(self, x: float, k: int) -> float:
        """
        Survival function (1 - CDF) for chi-square distribution.
        
        Uses approximation for computational efficiency.
        """
        if x <= 0 or k <= 0:
            return 1.0
        
        # Wilson-Hilferty approximation for large k
        if k > 30:
            z = math.pow(x / k, 1/3) - (1 - 2/(9*k))
            z = z / math.sqrt(2/(9*k))
            return self._normal_sf(z)
        
        # Direct approximation for small k
        try:
            # Use regularized gamma approximation
            return math.exp(-x/2) * sum(
                (x/2) ** i / math.factorial(i) 
                for i in range(k // 2)
            )
        except (OverflowError, ValueError):
            return 0.0 if x > k else 1.0
    
    def _normal_sf(self, z: float) -> float:
        """Standard normal survival function approximation."""
        # Abramowitz and Stegun approximation
        if z < 0:
            return 1 - self._normal_sf(-z)
        
        b1 = 0.319381530
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429
        p = 0.2316419
        c = 0.39894228
        
        if z > 6:
            return 0.0
        
        t = 1.0 / (1.0 + p * z)
        phi = c * math.exp(-z * z / 2.0)
        
        return phi * t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
    
    def test_sex_distribution(self, personas: List[Persona]) -> StatisticalTestResult:
        """Test sex distribution against census."""
        n = len(personas)
        observed = Counter(p.sex for p in personas)
        expected = self.census.SEX_DISTRIBUTION
        
        return self.chi_square_goodness_of_fit(
            dict(observed), expected, n
        )
    
    def test_region_distribution(self, personas: List[Persona]) -> StatisticalTestResult:
        """Test region distribution against census."""
        n = len(personas)
        observed = Counter(p.region for p in personas)
        expected = self.census.REGION_DISTRIBUTION
        if strict_geo_enabled():
            derived = get_region_distribution_from_district(strict=True)
            if derived:
                expected = derived
        
        return self.chi_square_goodness_of_fit(
            dict(observed), expected, n
        )
    
    def test_ethnicity_distribution(self, personas: List[Persona]) -> StatisticalTestResult:
        """Test ethnicity distribution against census."""
        n = len(personas)
        observed = Counter(p.ethnicity for p in personas)
        expected = self.census.ETHNICITY_DISTRIBUTION
        
        return self.chi_square_goodness_of_fit(
            dict(observed), expected, n
        )
    
    def test_residence_distribution(self, personas: List[Persona]) -> StatisticalTestResult:
        """Test urban/rural distribution against census."""
        n = len(personas)
        observed = Counter(p.residence_type for p in personas)
        expected = self.census.RESIDENCE_TYPE_DISTRIBUTION
        
        return self.chi_square_goodness_of_fit(
            dict(observed), expected, n
        )

    def test_district_distribution(self, personas: List[Persona]) -> StatisticalTestResult:
        """Test district distribution when official targets are available."""
        target = get_district_distribution()
        if not target:
            return StatisticalTestResult(
                test_name="Chi-Square Goodness-of-Fit (District)",
                statistic=0.0,
                p_value=1.0,
                degrees_of_freedom=0,
                passed=True,
                details="No official district distribution available"
            )

        missing = [p for p in personas if not p.district]
        if missing:
            return StatisticalTestResult(
                test_name="Chi-Square Goodness-of-Fit (District)",
                statistic=0.0,
                p_value=0.0,
                degrees_of_freedom=0,
                passed=False,
                details=f"Missing district for {len(missing)} personas in strict geo mode"
            )

        n = len(personas)
        observed = Counter(p.district for p in personas)
        return self.chi_square_goodness_of_fit(dict(observed), target, n)
    
    def test_joint_distribution(
        self, 
        personas: List[Persona], 
        var1: str, 
        var2: str
    ) -> StatisticalTestResult:
        """
        Test joint distribution of two variables.
        
        Currently supports:
        - ("region", "ethnicity")
        - ("age_group", "education")
        
        Args:
            personas: List of personas
            var1: First variable name
            var2: Second variable name
            
        Returns:
            StatisticalTestResult
        """
        n = len(personas)
        
        if (var1, var2) == ("region", "ethnicity"):
            return self._test_region_ethnicity_joint(personas, n)
        elif (var1, var2) == ("age_group", "education"):
            return self._test_age_education_joint(personas, n)
        else:
            return StatisticalTestResult(
                test_name=f"Joint Distribution ({var1} × {var2})",
                statistic=0.0,
                p_value=1.0,
                degrees_of_freedom=0,
                passed=True,
                details="Joint distribution test not implemented for these variables"
            )
    
    def _test_region_ethnicity_joint(
        self, 
        personas: List[Persona], 
        n: int
    ) -> StatisticalTestResult:
        """Test Region × Ethnicity joint distribution."""
        # Observed joint counts
        observed_joint = Counter((p.region, p.ethnicity) for p in personas)
        
        # Expected under independence: P(region) × P(ethnicity|region)
        chi2_stat = 0.0
        df = 0
        details_parts = []
        
        for region, region_prob in self.census.REGION_DISTRIBUTION.items():
            for ethnicity in self.census.ETHNICITY_BY_REGION[region].keys():
                # Expected: n × P(region) × P(ethnicity|region)
                eth_given_region = self.census.ETHNICITY_BY_REGION[region].get(ethnicity, 0)
                expected = n * region_prob * eth_given_region
                observed = observed_joint.get((region, ethnicity), 0)
                
                if expected >= 5:  # Chi-square assumption
                    contribution = ((observed - expected) ** 2) / expected
                    chi2_stat += contribution
                    df += 1
                    
                    if contribution > 3.84:
                        details_parts.append(
                            f"({region}, {ethnicity}): obs={observed}, exp={expected:.1f}, χ²={contribution:.2f}"
                        )
        
        p_value = self._chi2_sf(chi2_stat, df - 1) if df > 1 else 1.0
        passed = p_value >= self.alpha
        
        details = "; ".join(details_parts) if details_parts else "Joint distribution within expected ranges"
        
        return StatisticalTestResult(
            test_name="Joint Distribution (Region × Ethnicity)",
            statistic=chi2_stat,
            p_value=p_value,
            degrees_of_freedom=df - 1 if df > 0 else 0,
            passed=passed,
            details=details,
            alpha=self.alpha
        )
    
    def _test_age_education_joint(
        self, 
        personas: List[Persona], 
        n: int
    ) -> StatisticalTestResult:
        """Test Age Group × Education joint distribution."""
        # Detect adult-only datasets to use the correct age marginal
        min_age = min((p.age for p in personas), default=AgeConstraints.MIN_PERSONA_AGE)
        if min_age >= AgeConstraints.MIN_PERSONA_AGE:
            age_group_dist = self.census.ADULT_AGE_GROUP_DISTRIBUTION
        else:
            age_group_dist = self.census.AGE_GROUP_DISTRIBUTION

        # Observed joint counts
        observed_joint = Counter((p.age_group, p.education_level) for p in personas)
        
        chi2_stat = 0.0
        df = 0
        details_parts = []
        
        for age_group in self.census.EDUCATION_BY_AGE_GROUP.keys():
            for education in self.census.EDUCATION_BY_AGE_GROUP[age_group].keys():
                # Expected: n × P(age_group) × P(education|age_group)
                age_prob = age_group_dist.get(age_group, 0)
                ed_given_age = self.census.EDUCATION_BY_AGE_GROUP[age_group].get(education, 0)
                expected = n * age_prob * ed_given_age
                observed = observed_joint.get((age_group, education), 0)
                
                if expected >= 5:  # Chi-square assumption
                    contribution = ((observed - expected) ** 2) / expected
                    chi2_stat += contribution
                    df += 1
                    
                    if contribution > 3.84:
                        details_parts.append(
                            f"({age_group}, {education}): obs={observed}, exp={expected:.1f}"
                        )
        
        p_value = self._chi2_sf(chi2_stat, df - 1) if df > 1 else 1.0
        passed = p_value >= self.alpha
        
        details = "; ".join(details_parts) if details_parts else "Joint distribution within expected ranges"
        
        return StatisticalTestResult(
            test_name="Joint Distribution (Age × Education)",
            statistic=chi2_stat,
            p_value=p_value,
            degrees_of_freedom=df - 1 if df > 0 else 0,
            passed=passed,
            details=details,
            alpha=self.alpha
        )
    
    def run_all_tests(self, personas: List[Persona]) -> List[StatisticalTestResult]:
        """Run all statistical validation tests."""
        tests = [
            self.test_sex_distribution(personas),
            self.test_region_distribution(personas),
            self.test_ethnicity_distribution(personas),
            self.test_residence_distribution(personas),
            self.test_joint_distribution(personas, "region", "ethnicity"),
            self.test_joint_distribution(personas, "age_group", "education"),
            self.test_district_distribution(personas),
        ]
        return tests
    
    def summary_report(self, personas: List[Persona]) -> str:
        """Generate comprehensive statistical validation report."""
        tests = self.run_all_tests(personas)
        
        lines = [
            "Statistical Validation Report",
            "=" * 50,
            f"Sample size: {len(personas)}",
            f"Significance level (α): {self.alpha}",
            "",
        ]
        
        passed = sum(1 for t in tests if t.passed)
        failed = len(tests) - passed
        
        lines.append(f"Results: {passed}/{len(tests)} tests passed")
        if failed > 0:
            lines.append(f"         {failed} test(s) failed")
        lines.append("")
        
        for test in tests:
            lines.append(test.summary())
            lines.append("")
        
        return "\n".join(lines)


def calculate_adaptive_tolerance(n: int, confidence: float = 0.95) -> float:
    """
    Calculate adaptive tolerance based on sample size.
    
    Uses the margin of error formula for proportions:
    ME = z × sqrt(p(1-p)/n)
    
    For 95% confidence with p=0.5 (worst case):
    ME ≈ 1.96 × 0.5/sqrt(n) = 0.98/sqrt(n)
    
    Args:
        n: Sample size
        confidence: Confidence level (default 0.95)
        
    Returns:
        Recommended tolerance
    """
    # Z-scores for common confidence levels
    z_scores = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }
    z = z_scores.get(confidence, 1.96)
    
    # Worst-case margin of error (p=0.5)
    margin_of_error = z * 0.5 / math.sqrt(n)
    
    return margin_of_error
