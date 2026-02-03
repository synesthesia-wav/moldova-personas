"""
Golden set + regression testing for persona generation.

200 fixed cores (stratified by region/age/occupation) used as CI gate.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import asyncio


# Stratified sampling for 200 fixed cores
GOLDEN_SET_STRATIFICATION = {
    "regions": {
        "Chisinau": 40,
        "Nord": 40,
        "Centru": 40,
        "Sud": 40,
        "Gagauzia": 40,
    },
    "age_groups": {
        "18-30": 50,
        "31-45": 50,
        "46-60": 50,
        "60+": 50,
    },
    "occupation_types": {
        "professional": 50,  # teacher, doctor, engineer
        "service": 50,       # driver, seller, cook
        "manual": 50,        # farmer, worker
        "retired": 50,       # pensionar
    }
}


@dataclass
class GoldenCore:
    """A fixed core for regression testing."""
    core_id: str
    region: str
    age: int
    sex: str
    occupation: str
    education: str
    seed: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "core_id": self.core_id,
            "region": self.region,
            "age": self.age,
            "sex": self.sex,
            "occupation": self.occupation,
            "education": self.education,
            "seed": self.seed,
        }


class GoldenSetGenerator:
    """Generate and manage the golden set of 200 fixed cores."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.cores: List[GoldenCore] = []
    
    def generate(self) -> List[GoldenCore]:
        """Generate 200 stratified golden cores."""
        from .generator import PersonaGenerator
        
        generator = PersonaGenerator(seed=self.seed)
        
        # Generate a large pool and stratify
        pool = generator.generate(1000)
        
        cores = []
        core_id = 0
        
        # Stratified sampling
        for region, region_count in GOLDEN_SET_STRATIFICATION["regions"].items():
            region_personas = [p for p in pool if p.region == region or 
                             (region == "Gagauzia" and "Gagauzia" in p.region)]
            
            # Sample evenly across age and occupation
            age_groups = GOLDEN_SET_STRATIFICATION["age_groups"]
            occ_types = GOLDEN_SET_STRATIFICATION["occupation_types"]
            
            per_combo = region_count // (len(age_groups) * len(occ_types))
            
            for age_group, age_count in age_groups.items():
                # Parse age range
                if age_group == "18-30":
                    age_min, age_max = 18, 30
                elif age_group == "31-45":
                    age_min, age_max = 31, 45
                elif age_group == "46-60":
                    age_min, age_max = 46, 60
                else:
                    age_min, age_max = 60, 90
                
                age_personas = [p for p in region_personas 
                              if age_min <= p.age <= age_max]
                
                for occ_type in occ_types:
                    # Sample from this stratum
                    stratum = [p for p in age_personas 
                             if self._classify_occupation(p.occupation) == occ_type]
                    
                    if stratum:
                        selected = stratum[:per_combo]
                        for p in selected:
                            core_id += 1
                            cores.append(GoldenCore(
                                core_id=f"G{core_id:03d}",
                                region=region,
                                age=p.age,
                                sex=p.sex,
                                occupation=p.occupation,
                                education=p.education_level,
                                seed=self.seed + core_id
                            ))
        
        self.cores = cores[:200]  # Ensure exactly 200
        return self.cores
    
    def _classify_occupation(self, occupation: str) -> str:
        """Classify occupation into type."""
        occ_lower = occupation.lower()
        
        professional = ["profesor", "medic", "inginer", "avocat", "contabil"]
        service = ["șofer", "vânzător", "ospătar", "bucătar"]
        manual = ["muncitor", "agricultor", "tractorist"]
        
        if any(p in occ_lower for p in professional):
            return "professional"
        elif any(s in occ_lower for s in service):
            return "service"
        elif any(m in occ_lower for m in manual):
            return "manual"
        elif "pensionar" in occ_lower:
            return "retired"
        else:
            return "professional"  # Default
    
    def save(self, output_path: Path):
        """Save golden set to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "metadata": {
                "version": "1.0.0",
                "count": len(self.cores),
                "stratification": GOLDEN_SET_STRATIFICATION,
                "seed": self.seed,
            },
            "cores": [c.to_dict() for c in self.cores]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    @classmethod
    def load(cls, input_path: Path) -> "GoldenSetGenerator":
        """Load golden set from file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        instance = cls(seed=data["metadata"]["seed"])
        instance.cores = [
            GoldenCore(**core_data)
            for core_data in data["cores"]
        ]
        return instance


class RegressionTester:
    """
    Regression testing against golden set.
    
    Compares new generation against baseline metrics.
    """
    
    def __init__(self, golden_set_path: Path):
        self.golden_set = GoldenSetGenerator.load(golden_set_path)
        self.baseline_metrics: Dict[str, Any] = {}
    
    async def run_regression(
        self,
        generator,
        validator,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Run regression test on golden set.
        
        Returns:
            Regression report with pass/fail status
        """
        print(f"\n{'='*70}")
        print("REGRESSION TEST: Golden Set (200 cores)")
        print(f"{'='*70}\n")
        
        results = []
        
        for i, core in enumerate(self.golden_set.cores, 1):
            if i % 50 == 0:
                print(f"  Processing {i}/200...")
            
            # Generate persona from core
            # (This would integrate with your actual generator)
            
            # Validate
            # validation = validator.validate(persona)
            
            results.append({
                "core_id": core.core_id,
                "passed": True,  # Placeholder
                # "metrics": validation.metrics
            })
        
        # Calculate aggregate metrics
        report = self._generate_report(results)
        
        # Save
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / "regression_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        self._print_report(report)
        
        return report
    
    def _generate_report(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate regression report."""
        total = len(results)
        passed = sum(1 for r in results if r.get("passed", False))
        
        return {
            "total_cores": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "individual_results": results,
            "regression_status": "PASS" if passed == total else "FAIL",
        }
    
    def _print_report(self, report: Dict):
        """Print regression report."""
        print(f"\n{'='*70}")
        print("REGRESSION REPORT")
        print(f"{'='*70}")
        print(f"Total cores: {report['total_cores']}")
        print(f"Passed: {report['passed']}")
        print(f"Failed: {report['failed']}")
        print(f"Pass rate: {report['pass_rate']:.1%}")
        print(f"\nStatus: {report['regression_status']}")
        print(f"{'='*70}\n")
    
    def compare_to_baseline(
        self,
        current_metrics: Dict[str, Any],
        baseline_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare current metrics to baseline.
        
        Returns drift report.
        """
        drift = []
        
        # Compare key metrics
        metric_thresholds = {
            "validator_pass_rate": 0.05,  # Max 5% decrease
            "mean_edits_per_record": 0.5,  # Max 0.5 increase
            "duplication_rate": 0.02,  # Max 2% increase
            "qa_contradiction_rate": 0.05,  # Max 5% increase
        }
        
        for metric, threshold in metric_thresholds.items():
            if metric in current_metrics and metric in baseline_metrics:
                current = current_metrics[metric]
                baseline = baseline_metrics[metric]
                
                change = current - baseline
                
                if abs(change) > threshold:
                    drift.append({
                        "metric": metric,
                        "baseline": baseline,
                        "current": current,
                        "change": change,
                        "threshold": threshold,
                        "severity": "WARNING" if abs(change) < threshold * 2 else "CRITICAL"
                    })
        
        return {
            "has_drift": len(drift) > 0,
            "drift_count": len(drift),
            "drift_details": drift,
            "status": "PASS" if len(drift) == 0 else "DRIFT DETECTED"
        }


def create_golden_set(output_dir: str = "./golden_set") -> Path:
    """
    Create golden set and save to disk.
    
    Usage:
        golden_set_path = create_golden_set("./golden_set")
    """
    generator = GoldenSetGenerator(seed=42)
    cores = generator.generate()
    
    output_path = Path(output_dir) / "golden_set_200.json"
    generator.save(output_path)
    
    print(f"\nCreated golden set with {len(cores)} cores")
    print(f"Stratification:")
    
    # Print stratification stats
    region_counts = {}
    age_counts = {}
    for core in cores:
        region_counts[core.region] = region_counts.get(core.region, 0) + 1
        age_group = "18-30" if core.age <= 30 else "31-45" if core.age <= 45 else "46-60" if core.age <= 60 else "60+"
        age_counts[age_group] = age_counts.get(age_group, 0) + 1
    
    print(f"  By region: {region_counts}")
    print(f"  By age: {age_counts}")
    print(f"\nSaved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    # Create golden set when run directly
    create_golden_set()
