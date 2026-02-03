"""Tests for statistical validation functions."""

import pytest
import math

from moldova_personas import PersonaGenerator
from moldova_personas.statistical_tests import (
    StatisticalValidator,
    StatisticalTestResult,
    calculate_adaptive_tolerance,
)


class TestStatisticalValidator:
    """Tests for StatisticalValidator class."""
    
    def test_chi_square_perfect_match(self):
        """Test chi-square with perfect distribution match."""
        validator = StatisticalValidator(alpha=0.05)
        
        # Create observed that exactly matches expected
        n = 1000
        expected_probs = {"A": 0.5, "B": 0.3, "C": 0.2}
        observed = {"A": 500, "B": 300, "C": 200}
        
        result = validator.chi_square_goodness_of_fit(
            observed, expected_probs, n
        )
        
        # Perfect match should have chi2 = 0, p-value = 1
        assert result.statistic == 0.0
        assert result.p_value == 1.0
        assert result.passed is True
    
    def test_chi_square_known_deviation(self):
        """Test chi-square with known significant deviation."""
        validator = StatisticalValidator(alpha=0.05)
        
        n = 100
        expected_probs = {"A": 0.5, "B": 0.5}
        # Large deviation: 70/30 instead of 50/50
        observed = {"A": 70, "B": 30}
        
        result = validator.chi_square_goodness_of_fit(
            observed, expected_probs, n
        )
        
        # Should detect significant deviation
        assert result.statistic > 0
        assert result.p_value < 0.05
        assert result.passed is False

    def test_chi_square_p_value_df2(self):
        """Test chi-square p-value for df=2 against a known range."""
        validator = StatisticalValidator(alpha=0.05)

        n = 200
        expected_probs = {"A": 0.5, "B": 0.3, "C": 0.2}
        observed = {"A": 120, "B": 50, "C": 30}

        result = validator.chi_square_goodness_of_fit(
            observed, expected_probs, n
        )

        assert result.degrees_of_freedom == 2
        assert 0.01 < result.p_value < 0.03
    
    def test_sex_distribution_test(self):
        """Test sex distribution validation."""
        gen = PersonaGenerator(seed=42)
        personas = gen.generate(500)
        
        validator = StatisticalValidator(alpha=0.05)
        result = validator.test_sex_distribution(personas)
        
        assert result.test_name == "Chi-Square Goodness-of-Fit"
        assert result.degrees_of_freedom == 1  # 2 categories - 1
        # With 500 samples, should pass (distribution is calibrated)
        assert result.passed is True
    
    def test_region_distribution_test(self):
        """Test region distribution validation."""
        gen = PersonaGenerator(seed=42)
        personas = gen.generate(500)
        
        validator = StatisticalValidator(alpha=0.05)
        result = validator.test_region_distribution(personas)
        
        assert result.degrees_of_freedom == 4  # 5 regions - 1
        assert result.passed is True
    
    def test_ethnicity_distribution_test(self):
        """Test ethnicity distribution validation."""
        gen = PersonaGenerator(seed=42)
        personas = gen.generate(500)
        
        validator = StatisticalValidator(alpha=0.05)
        result = validator.test_ethnicity_distribution(personas)
        
        # Note: Some ethnicities have very low proportions
        # which may result in expected counts < 5
        assert result.statistic >= 0
    
    def test_joint_distribution_region_ethnicity(self):
        """Test joint distribution validation."""
        gen = PersonaGenerator(seed=42)
        personas = gen.generate(500)
        
        validator = StatisticalValidator(alpha=0.05)
        result = validator.test_joint_distribution(
            personas, "region", "ethnicity"
        )
        
        assert result.test_name == "Joint Distribution (Region × Ethnicity)"
        assert result.statistic >= 0
    
    def test_joint_distribution_age_education(self):
        """Test age-education joint distribution."""
        gen = PersonaGenerator(seed=42)
        personas = gen.generate(500)
        
        validator = StatisticalValidator(alpha=0.05)
        result = validator.test_joint_distribution(
            personas, "age_group", "education"
        )
        
        assert result.test_name == "Joint Distribution (Age × Education)"
        assert result.statistic >= 0
    
    def test_run_all_tests(self):
        """Test running all statistical tests."""
        gen = PersonaGenerator(seed=42)
        personas = gen.generate(200)
        
        validator = StatisticalValidator(alpha=0.05)
        results = validator.run_all_tests(personas)
        
        assert len(results) == 7
        assert all(isinstance(r, StatisticalTestResult) for r in results)
        
        # At least some tests should pass with proper calibration
        passed = sum(1 for r in results if r.passed)
        assert passed >= 3  # Most should pass
    
    def test_summary_report_format(self):
        """Test that summary report is properly formatted."""
        gen = PersonaGenerator(seed=42)
        personas = gen.generate(100)
        
        validator = StatisticalValidator(alpha=0.05)
        report = validator.summary_report(personas)
        
        assert "Statistical Validation Report" in report
        assert "Sample size: 100" in report
        assert "Significance level" in report
        assert "tests passed" in report


class TestAdaptiveTolerance:
    """Tests for adaptive tolerance calculation."""
    
    def test_adaptive_tolerance_decreases_with_n(self):
        """Test that tolerance decreases as sample size increases."""
        tol_100 = calculate_adaptive_tolerance(100)
        tol_1000 = calculate_adaptive_tolerance(1000)
        tol_10000 = calculate_adaptive_tolerance(10000)
        
        # Tolerance should decrease with larger n
        assert tol_100 > tol_1000 > tol_10000
    
    def test_adaptive_tolerance_95_percent(self):
        """Test 95% confidence tolerance."""
        # For n=100 at 95% confidence with p=0.5
        # ME = 1.96 * 0.5 / sqrt(100) = 0.098
        tol = calculate_adaptive_tolerance(100, confidence=0.95)
        assert abs(tol - 0.098) < 0.01
    
    def test_adaptive_tolerance_99_percent(self):
        """Test 99% confidence tolerance."""
        # For n=100 at 99% confidence
        # ME = 2.576 * 0.5 / sqrt(100) = 0.1288
        tol = calculate_adaptive_tolerance(100, confidence=0.99)
        assert abs(tol - 0.1288) < 0.01
    
    def test_adaptive_tolerance_large_n(self):
        """Test with large sample size."""
        tol = calculate_adaptive_tolerance(100000)
        # With 100K samples, tolerance should be very small (~0.003)
        assert tol < 0.01
    
    def test_default_confidence(self):
        """Test default confidence level is 95%."""
        tol_default = calculate_adaptive_tolerance(100)
        tol_95 = calculate_adaptive_tolerance(100, confidence=0.95)
        
        assert abs(tol_default - tol_95) < 0.0001


class TestStatisticalTestResult:
    """Tests for StatisticalTestResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a test result."""
        result = StatisticalTestResult(
            test_name="Test",
            statistic=5.0,
            p_value=0.025,
            degrees_of_freedom=2,
            passed=False,
            details="Some details"
        )
        
        assert result.test_name == "Test"
        assert result.p_value == 0.025
        assert result.passed is False
    
    def test_summary_format_passed(self):
        """Test summary format for passed test."""
        result = StatisticalTestResult(
            test_name="Good Test",
            statistic=1.0,
            p_value=0.6,
            degrees_of_freedom=3,
            passed=True,
            details="All good"
        )
        
        summary = result.summary()
        assert "✓ PASS" in summary
        assert "Good Test" in summary
        assert "p-value: 0.6000" in summary
    
    def test_summary_format_failed(self):
        """Test summary format for failed test."""
        result = StatisticalTestResult(
            test_name="Bad Test",
            statistic=15.0,
            p_value=0.001,
            degrees_of_freedom=1,
            passed=False,
            details="Significant deviation"
        )
        
        summary = result.summary()
        assert "✗ FAIL" in summary
        assert "Bad Test" in summary
