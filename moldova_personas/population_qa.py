"""
Population-level QA for Nemotron-style personas.

Implements stratified distribution checks and divergence metrics
for ensuring demographic and personality distributions match targets.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import math


@dataclass
class StratifiedDistribution:
    """Distribution for a stratified demographic slice."""
    slice_key: str  # e.g., "Feminin_25-34_Chisinau"
    count: int
    target_proportion: float
    observed_proportion: float
    absolute_error: float


@dataclass
class DivergenceMetrics:
    """Distribution divergence metrics."""
    kl_divergence: float  # Kullback-Leibler
    js_divergence: float  # Jensen-Shannon
    max_absolute_error: float
    mean_absolute_error: float
    
    def is_acceptable(self, kl_threshold: float = 0.1, max_error_threshold: float = 0.05) -> bool:
        """Check if distributions are acceptably close."""
        return (
            self.kl_divergence <= kl_threshold and
            self.max_absolute_error <= max_error_threshold
        )


class PopulationQA:
    """
    Population-level quality assurance for persona datasets.
    
    Validates that generated distributions match target distributions
    across demographic strata and personality traits.
    """
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate Kullback-Leibler divergence D_KL(P||Q).
        
        Args:
            p: Target distribution (should sum to 1)
            q: Observed distribution (should sum to 1)
        
        Returns:
            KL divergence value
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = np.array(p) + epsilon
        q = np.array(q) + epsilon
        
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        
        return np.sum(p * np.log(p / q))
    
    def calculate_js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate Jensen-Shannon divergence.
        
        JS divergence is symmetric and bounded, unlike KL.
        """
        m = 0.5 * (p + q)
        return 0.5 * self.calculate_kl_divergence(p, m) + 0.5 * self.calculate_kl_divergence(q, m)
    
    def check_demographic_strata(
        self,
        personas: List[Dict],
        ipf_targets: Dict[str, Dict[str, float]],
        strata_vars: List[str] = ["sex", "age_group", "region"]
    ) -> Tuple[DivergenceMetrics, List[StratifiedDistribution]]:
        """
        Check demographic distributions against IPF targets.
        
        Args:
            personas: List of persona dictionaries
            ipf_targets: Nested dict of target proportions from IPF
                         e.g., {"Feminin": {"25-34": {"Chisinau": 0.05}}}
            strata_vars: Variables to stratify by
        
        Returns:
            (divergence_metrics, stratified_distributions)
        """
        # Build observed counts
        observed_counts = defaultdict(int)
        total = len(personas)
        
        for p in personas:
            # Build slice key
            key_parts = []
            for var in strata_vars:
                if var == "age_group":
                    value = self._get_age_group(p.get("age", 30))
                else:
                    value = p.get(var, "unknown")
                key_parts.append(str(value))
            
            slice_key = "_".join(key_parts)
            observed_counts[slice_key] += 1
        
        # Build target and observed distributions
        all_slices = set(observed_counts.keys())
        
        # Add any missing slices from targets
        def extract_slices(target_dict, prefix=""):
            slices = set()
            for key, value in target_dict.items():
                new_prefix = f"{prefix}_{key}" if prefix else key
                if isinstance(value, dict):
                    slices.update(extract_slices(value, new_prefix))
                else:
                    slices.add(new_prefix)
            return slices
        
        all_slices.update(extract_slices(ipf_targets))
        
        # Calculate proportions and errors
        target_props = []
        observed_props = []
        stratified_results = []
        
        for slice_key in sorted(all_slices):
            observed_count = observed_counts.get(slice_key, 0)
            observed_prop = observed_count / total if total > 0 else 0
            
            # Get target proportion
            target_prop = self._get_nested_target(ipf_targets, slice_key)
            
            target_props.append(target_prop)
            observed_props.append(observed_prop)
            
            stratified_results.append(StratifiedDistribution(
                slice_key=slice_key,
                count=observed_count,
                target_proportion=target_prop,
                observed_proportion=observed_prop,
                absolute_error=abs(observed_prop - target_prop)
            ))
        
        # Calculate divergence metrics
        target_arr = np.array(target_props)
        observed_arr = np.array(observed_props)
        
        # Normalize to sum to 1
        target_arr = target_arr / target_arr.sum() if target_arr.sum() > 0 else target_arr
        observed_arr = observed_arr / observed_arr.sum() if observed_arr.sum() > 0 else observed_arr
        
        metrics = DivergenceMetrics(
            kl_divergence=self.calculate_kl_divergence(target_arr, observed_arr),
            js_divergence=self.calculate_js_divergence(target_arr, observed_arr),
            max_absolute_error=max(sd.absolute_error for sd in stratified_results),
            mean_absolute_error=np.mean([sd.absolute_error for sd in stratified_results])
        )
        
        return metrics, stratified_results
    
    def check_ocean_distributions(
        self,
        personas: List[Dict],
        expected_mean: float = 50.0,
        expected_std: float = 15.0,
        min_variance: float = 100.0  # Minimum variance to avoid collapse
    ) -> Dict[str, DivergenceMetrics]:
        """
        Check that OCEAN trait distributions don't collapse.
        
        Args:
            personas: List of persona dictionaries with ocean_* fields
            expected_mean: Expected population mean (typically 50)
            expected_std: Expected population std (typically 15)
            min_variance: Minimum acceptable variance
        
        Returns:
            Dict mapping trait name to divergence metrics
        """
        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        results = {}
        
        for trait in traits:
            # Extract scores
            scores = [p.get(f"ocean_{trait}", 50) for p in personas if f"ocean_{trait}" in p]
            
            if not scores:
                continue
            
            observed_mean = np.mean(scores)
            observed_std = np.std(scores)
            observed_var = observed_std ** 2
            
            # Create histogram distributions
            bins = np.arange(0, 101, 10)  # 0-10, 10-20, ..., 90-100
            observed_hist, _ = np.histogram(scores, bins=bins, density=True)
            
            # Expected normal distribution
            expected_hist = self._normal_histogram(expected_mean, expected_std, bins)
            
            # Normalize
            observed_hist = observed_hist / observed_hist.sum() if observed_hist.sum() > 0 else observed_hist
            expected_hist = expected_hist / expected_hist.sum() if expected_hist.sum() > 0 else expected_hist
            
            # Calculate metrics
            max_error = max(abs(observed_hist - expected_hist))
            mean_error = np.mean(abs(observed_hist - expected_hist))
            
            metrics = DivergenceMetrics(
                kl_divergence=self.calculate_kl_divergence(expected_hist, observed_hist),
                js_divergence=self.calculate_js_divergence(expected_hist, observed_hist),
                max_absolute_error=max_error,
                mean_absolute_error=mean_error
            )
            
            results[trait] = {
                "metrics": metrics,
                "observed_mean": observed_mean,
                "observed_std": observed_std,
                "observed_variance": observed_var,
                "variance_ok": observed_var >= min_variance,
                "collapse_warning": observed_var < min_variance
            }
        
        return results
    
    def check_ocean_stability(
        self,
        personas: List[Dict],
        tolerance: float = 5.0
    ) -> Dict[str, any]:
        """
        Check that OCEAN traits don't drift across generation batches.
        
        Compares current batch statistics against population norms.
        
        Args:
            personas: Current batch of personas
            tolerance: Acceptable deviation from expected mean
        
        Returns:
            Stability report
        """
        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        
        drift_detected = []
        stable_traits = []
        
        for trait in traits:
            scores = [p.get(f"ocean_{trait}", 50) for p in personas if f"ocean_{trait}" in p]
            
            if len(scores) < 10:
                continue
            
            mean = np.mean(scores)
            deviation = abs(mean - 50)  # Expected mean is 50
            
            if deviation > tolerance:
                drift_detected.append({
                    "trait": trait,
                    "observed_mean": mean,
                    "expected_mean": 50,
                    "deviation": deviation
                })
            else:
                stable_traits.append(trait)
        
        return {
            "drift_detected": len(drift_detected) > 0,
            "drifted_traits": drift_detected,
            "stable_traits": stable_traits,
            "overall_stable": len(drift_detected) == 0
        }
    
    def generate_report(
        self,
        personas: List[Dict],
        ipf_targets: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Generate comprehensive population QA report.
        
        Args:
            personas: List of persona dictionaries
            ipf_targets: IPF target distributions (optional)
        
        Returns:
            Complete QA report
        """
        report = {
            "total_personas": len(personas),
            "checks": {}
        }
        
        # OCEAN distribution check
        ocean_check = self.check_ocean_distributions(personas)
        report["checks"]["ocean_distributions"] = {
            trait: {
                "kl_divergence": data["metrics"].kl_divergence,
                "js_divergence": data["metrics"].js_divergence,
                "max_absolute_error": data["metrics"].max_absolute_error,
                "observed_mean": data["observed_mean"],
                "observed_std": data["observed_std"],
                "collapse_warning": data["collapse_warning"]
            }
            for trait, data in ocean_check.items()
        }
        
        # OCEAN stability check
        stability = self.check_ocean_stability(personas)
        report["checks"]["ocean_stability"] = stability
        
        # Demographic strata check (if targets provided)
        if ipf_targets:
            metrics, strata = self.check_demographic_strata(personas, ipf_targets)
            report["checks"]["demographic_strata"] = {
                "kl_divergence": metrics.kl_divergence,
                "js_divergence": metrics.js_divergence,
                "max_absolute_error": metrics.max_absolute_error,
                "mean_absolute_error": metrics.mean_absolute_error,
                "acceptable": metrics.is_acceptable(),
                "worst_strata": [
                    {"slice": s.slice_key, "error": s.absolute_error}
                    for s in sorted(strata, key=lambda x: x.absolute_error, reverse=True)[:5]
                ]
            }
        
        # Overall assessment
        all_acceptable = all(
            not data.get("collapse_warning", False)
            for data in ocean_check.values()
        )
        all_acceptable = all_acceptable and stability["overall_stable"]
        
        if ipf_targets and "demographic_strata" in report["checks"]:
            all_acceptable = all_acceptable and report["checks"]["demographic_strata"]["acceptable"]
        
        report["overall_acceptable"] = all_acceptable
        
        return report
    
    def _get_age_group(self, age: int) -> str:
        """Map age to age group."""
        if age < 25:
            return "18-24"
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
    
    def _get_nested_target(self, target_dict: Dict, slice_key: str) -> float:
        """Extract target proportion from nested dict using slice key."""
        parts = slice_key.split("_")
        
        current = target_dict
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return 0.0
        
        return float(current) if isinstance(current, (int, float)) else 0.0
    
    def _normal_histogram(self, mean: float, std: float, bins: np.ndarray) -> np.ndarray:
        """Generate histogram from normal distribution."""
        from scipy import stats
        
        # Calculate probability for each bin
        probs = []
        for i in range(len(bins) - 1):
            prob = stats.norm.cdf(bins[i+1], mean, std) - stats.norm.cdf(bins[i], mean, std)
            probs.append(prob)
        
        return np.array(probs)


def quick_population_qa(personas: List[Dict]) -> Dict[str, any]:
    """
    Quick population QA check.
    
    Returns summary of distribution health.
    """
    qa = PopulationQA()
    
    # OCEAN check
    ocean_results = qa.check_ocean_distributions(personas)
    
    # Stability check
    stability = qa.check_ocean_stability(personas)
    
    return {
        "total": len(personas),
        "ocean_traits": {
            trait: {
                "mean": round(data["observed_mean"], 1),
                "std": round(data["observed_std"], 1),
                "collapse_risk": data["collapse_warning"]
            }
            for trait, data in ocean_results.items()
        },
        "stability": "OK" if stability["overall_stable"] else f"Drift: {[t['trait'] for t in stability['drifted_traits']]}",
        "healthy": all(not d["collapse_warning"] for d in ocean_results.values()) and stability["overall_stable"]
    }
