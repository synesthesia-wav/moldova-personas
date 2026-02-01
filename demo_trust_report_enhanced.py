#!/usr/bin/env python3
"""
Demonstration of Enhanced Trust Report Features

Shows all six improvements:
1. Use-case dependent overall tier calculation
2. ESS gate for IPF sanity
3. Field-specific gating for super-criticals
4. Locality integrity checks
5. Narrative quality metrics beyond mock/failed
6. Data currency reporting
"""

from moldova_personas import (
    PersonaGenerator,
    TrustReportGenerator,
    UseCaseProfile,
    QualityTier,
    IPFMetrics,
    NarrativeQualityMetrics,
    validate_locality_config,
    LocalityIntegrityError,
    CRITICAL_FIELDS,
    HIGH_STAKES_CRITICAL_FIELDS,
)
from moldova_personas.models import Persona
import json


def demo_use_case_tiers():
    """Demo: Overall tier depends on use-case profile."""
    print("=" * 70)
    print("1. USE-CASE DEPENDENT OVERALL TIER CALCULATION")
    print("=" * 70)
    
    # Generate personas using proper generator (ensures realistic distributions)
    gen = PersonaGenerator(seed=42)
    personas = gen.generate(100)  # Generate realistic personas
    
    # Force all narratives to mock for this demo
    for p in personas:
        p.narrative_status = "mock"
    
    for use_case in UseCaseProfile:
        print(f"\n--- {use_case.value.upper()} ---")
        # Pass census distributions for proper L1 error calculation
        report_gen = TrustReportGenerator(
            census_distributions=gen.census,
            use_case=use_case
        )
        report = report_gen.generate_report(
            personas,
            provenance_info={
                "sex": {"provenance": "PXWEB_DIRECT"},
                "region": {"provenance": "PXWEB_DIRECT"},
                "ethnicity": {"provenance": "PXWEB_DIRECT"},
                "education": {"provenance": "PXWEB_DIRECT"},
                "employment_status": {"provenance": "PXWEB_DIRECT"},
            },
            generator_version="0.3.0",
            targets_source="pxweb_live",
        )
        
        print(f"  Structured: {report.structured_quality_tier.value}")
        print(f"  Narrative:  {report.narrative_quality_tier.value}")
        print(f"  Overall:    {report.overall_quality_tier.value}")
        print(f"  Reasoning:  {report.overall_tier_reasoning}")
        print(f"  Hard Gate:  {report.hard_gate_triggered}")


def demo_ess_gate():
    """Demo: ESS ratio gating for IPF sanity."""
    print("\n" + "=" * 70)
    print("2. ESS GATE (IPF INFORMATION LOSS)")
    print("=" * 70)
    
    personas = []
    for i in range(100):
        p = Persona(
            id=f"test_{i}",
            name="Test Person",
            age=30,
            sex="Feminin" if i % 2 == 0 else "Masculin",
            age_group="25-34",
            region="Chisinau",
            residence_type="Urban",
            city="Chișinău",
            district="Chișinău",
            ethnicity="Moldovean",
            mother_tongue="Română",
            religion="Ortodox",
            marital_status="Căsătorit",
            education_level="Licență",
            occupation="Profesor",
            employment_status="employed",
            narrative_status="generated",
            field_provenance={}
        )
        personas.append(p)
    
    # Test different ESS ratios
    for ess_ratio, label in [(0.90, "Good (90%)"), (0.60, "Marginal (60%)"), (0.40, "Poor (40%)")]:
        print(f"\n--- ESS Ratio: {label} ---")
        
        # Create IPF metrics
        original_size = 1000
        ess = original_size * ess_ratio
        ipf = IPFMetrics(
            original_sample_size=original_size,
            effective_sample_size=ess,
            resampling_ratio=0.1,
            pre_correction_drift={"ethnicity": 0.15},
            post_correction_drift={"ethnicity": 0.05},
        )
        
        gen = TrustReportGenerator(use_case=UseCaseProfile.NARRATIVE_REQUIRED)
        report = gen.generate_report(
            personas,
            ipf_metrics=ipf,
            generator_version="0.3.0",
        )
        
        print(f"  Information Loss: {ipf.information_loss:.1%}")
        print(f"  ESS Ratio: {ipf.ess_ratio:.1%}")
        print(f"  Hard Gate Triggered: {report.hard_gate_triggered}")
        if report.gate_reasons:
            print(f"  Gate Reasons: {report.gate_reasons}")


def demo_super_critical_fields():
    """Demo: Field-specific gating for super-critical fields."""
    print("\n" + "=" * 70)
    print("3. SUPER-CRITICAL FIELD GATING")
    print("=" * 70)
    print(f"\nCritical Fields: {CRITICAL_FIELDS}")
    print(f"Super-Critical Fields: {HIGH_STAKES_CRITICAL_FIELDS}")
    
    personas = []
    for i in range(100):
        p = Persona(
            id=f"test_{i}",
            name="Test Person",
            age=30,
            sex="Feminin" if i % 2 == 0 else "Masculin",
            age_group="25-34",
            region="Chisinau",
            residence_type="Urban",
            city="Chișinău",
            district="Chișinău",
            ethnicity="Moldovean",
            mother_tongue="Română",
            religion="Ortodox",
            marital_status="Căsătorit",
            education_level="Licență",
            occupation="Profesor",
            employment_status="employed",
            narrative_status="generated",
            field_provenance={}
        )
        personas.append(p)
    
    # Test different fallback scenarios
    scenarios = [
        ("All Live Data", {"ethnicity": "PXWEB_DIRECT", "education": "PXWEB_DIRECT", "employment_status": "PXWEB_DIRECT", "region": "PXWEB_DIRECT"}),
        ("Ethnicity Fallback", {"ethnicity": "CENSUS_HARDCODED", "education": "PXWEB_DIRECT", "employment_status": "PXWEB_DIRECT", "region": "PXWEB_DIRECT"}),
        ("Education Fallback", {"ethnicity": "PXWEB_DIRECT", "education": "CENSUS_HARDCODED", "employment_status": "PXWEB_DIRECT", "region": "PXWEB_DIRECT"}),
    ]
    
    for name, prov in scenarios:
        print(f"\n--- Scenario: {name} ---")
        
        # Build proper provenance info structure with nested dicts
        base_prov = {
            "sex": {"provenance": "PXWEB_DIRECT"},
            "age_group": {"provenance": "PXWEB_DIRECT"},
            "residence_type": {"provenance": "PXWEB_DIRECT"},
            "religion": {"provenance": "PXWEB_DIRECT"},
            "marital_status": {"provenance": "PXWEB_DIRECT"},
            "language": {"provenance": "PXWEB_DIRECT"},
        }
        # Add scenario-specific provenances
        for field, prov_type in prov.items():
            base_prov[field] = {"provenance": prov_type}
        full_prov = base_prov
        
        for use_case in [UseCaseProfile.NARRATIVE_REQUIRED, UseCaseProfile.HIGH_STAKES]:
            gen = TrustReportGenerator(use_case=use_case)
            report = gen.generate_report(personas, provenance_info=full_prov)
            
            fallback_sc = report.fallback_fields_super_critical
            print(f"  {use_case.value}:")
            print(f"    Super-Critical Fallbacks: {fallback_sc if fallback_sc else 'None'}")
            print(f"    Fallback Ratio (SC): {report.fallback_ratio_super_critical:.1%}")
            print(f"    Hard Gate: {report.hard_gate_triggered}")


def demo_locality_integrity():
    """Demo: Locality configuration validation."""
    print("\n" + "=" * 70)
    print("4. LOCALITY INTEGRITY CHECKS")
    print("=" * 70)
    
    # Valid config
    valid_config = {
        "Chisinau": [
            {"name": "Chișinău", "settlement_type": "city", "implied_residence": "Urban", "population_tier": 1},
            {"name": "Bubuieci", "settlement_type": "village", "implied_residence": "Rural", "population_tier": 3},
            {"name": "Stăuceni", "settlement_type": "village", "implied_residence": "Rural", "population_tier": 3},
        ]
    }
    
    # Invalid config - contradictions
    invalid_config = {
        "Chisinau": [
            {"name": "Chișinău", "settlement_type": "city", "implied_residence": "Rural"},  # Contradiction!
            {"name": "Bubuieci", "settlement_type": "village", "implied_residence": "Urban"},  # Contradiction!
        ],
        "EmptyRegion": []  # No localities at all
    }
    
    print("\n--- Valid Config ---")
    result = validate_locality_config(localities_data=valid_config)
    print(f"Valid: {result.is_valid}")
    print(f"Coverage: {result.region_coverage}")
    
    print("\n--- Invalid Config ---")
    result = validate_locality_config(localities_data=invalid_config)
    print(f"Valid: {result.is_valid}")
    print(f"Errors ({len(result.errors)}):")
    for e in result.errors:
        print(f"  - {e}")


def demo_narrative_quality():
    """Demo: Narrative quality metrics beyond mock/failed."""
    print("\n" + "=" * 70)
    print("5. NARRATIVE QUALITY METRICS")
    print("=" * 70)
    
    # Create personas with different narrative qualities
    narratives = [
        ("generated", "Mă numesc Maria și sunt profesoară. Locuiesc în Chișinău și iubesc orașul meu."),  # Good
        ("generated", "Lucrez la o companie mare. Studiez medicina."),  # Short, no diacritics
        ("generated", "x" * 6000),  # Too long
        ("mock", ""),  # Mock
        ("failed", ""),  # Failed
    ]
    
    personas = []
    for i, (status, narrative) in enumerate(narratives):
        p = Persona(
            id=f"test_{i}",
            name="Maria" if i == 0 else "Test",
            age=30,
            sex="Feminin",
            age_group="25-34",
            region="Chisinau",
            residence_type="Urban",
            city="Chișinău",
            district="Chișinău",
            ethnicity="Moldovean",
            mother_tongue="Română",
            religion="Ortodox",
            marital_status="Căsătorit",
            education_level="Licență",
            occupation="Profesor",
            employment_status="employed",
            narrative_status=status,
            descriere_generala=narrative if status == "generated" else "",
            field_provenance={}
        )
        personas.append(p)
    
    gen = TrustReportGenerator(use_case=UseCaseProfile.NARRATIVE_REQUIRED)
    report = gen.generate_report(personas, generator_version="0.3.0")
    
    print("\nNarrative Status Counts:")
    print(f"  Generated: {report.narrative_generated_count}")
    print(f"  Mock: {report.narrative_mock_count}")
    print(f"  Failed: {report.narrative_failed_count}")
    
    if report.narrative_quality_metrics:
        nqm = report.narrative_quality_metrics
        print("\nDetailed Quality Metrics:")
        print(f"  Total Personas: {nqm.total_personas}")
        print(f"  Generated: {nqm.generated_count}, Mock: {nqm.mock_count}, Failed: {nqm.failed_count}")
        print(f"  Parse Attempted: {nqm.parse_attempted_count}, Failed: {nqm.parse_failed_count}")
        print(f"  Too Short: {nqm.too_short_count}, Too Long: {nqm.too_long_count}")
        print(f"  Length Valid Ratio: {nqm.length_valid_ratio:.1%}")
        print(f"  Schema Valid (overall): {nqm.schema_valid_ratio_overall:.1%}")
        print(f"  Schema Valid (attempted): {nqm.schema_valid_ratio_attempted:.1%}")
        print(f"  Romanian Valid Ratio: {nqm.romanian_valid_ratio:.1%}")


def demo_data_currency():
    """Demo: Data currency reporting and gating."""
    print("\n" + "=" * 70)
    print("6. DATA CURRENCY REPORTING")
    print("=" * 70)
    
    personas = []
    for i in range(100):
        p = Persona(
            id=f"test_{i}",
            name="Test Person",
            age=30,
            sex="Feminin" if i % 2 == 0 else "Masculin",
            age_group="25-34",
            region="Chisinau",
            residence_type="Urban",
            city="Chișinău",
            district="Chișinău",
            ethnicity="Moldovean",
            mother_tongue="Română",
            religion="Ortodox",
            marital_status="Căsătorit",
            education_level="Licență",
            occupation="Profesor",
            employment_status="employed",
            narrative_status="generated",
            field_provenance={}
        )
        personas.append(p)
    
    # Different cache ages
    from datetime import datetime, timedelta
    now = datetime.now()
    
    scenarios = [
        ("Fresh (1 day)", (now - timedelta(days=1)).isoformat(), False),
        ("Stale (15 days)", (now - timedelta(days=15)).isoformat(), False),
        ("Very Stale (45 days)", (now - timedelta(days=45)).isoformat(), True),
    ]
    
    for name, timestamp, expect_gate in scenarios:
        print(f"\n--- Cache Age: {name} ---")
        
        for use_case in [UseCaseProfile.NARRATIVE_REQUIRED, UseCaseProfile.HIGH_STAKES]:
            gen = TrustReportGenerator(use_case=use_case)
            report = gen.generate_report(
                personas,
                pxweb_cache_timestamp=timestamp,
                generator_version="0.3.0",
            )
            
            print(f"  {use_case.value}:")
            print(f"    Cache Age: {report.pxweb_cache_age_days} days")
            print(f"    Hard Gate: {report.hard_gate_triggered}")
            
            # Check for cache age in gate reasons
            cache_warnings = [r for r in report.gate_reasons if "Cache age" in r or "cache" in r.lower()]
            if cache_warnings:
                print(f"    Warning: {cache_warnings[0]}")


def demo_full_report_summary():
    """Demo: Complete trust report summary."""
    print("\n" + "=" * 70)
    print("FULL TRUST REPORT SUMMARY")
    print("=" * 70)
    
    # Use PersonaGenerator to get realistic personas AND census data
    persona_gen = PersonaGenerator(seed=42)
    personas = persona_gen.generate(1000)
    
    # Set narrative status for demo
    for i, p in enumerate(personas):
        p.narrative_status = "generated" if i < 950 else "mock"
    
    # Create IPF metrics showing good correction
    ipf = IPFMetrics(
        original_sample_size=5000,
        effective_sample_size=4800,
        resampling_ratio=0.2,
        pre_correction_drift={"ethnicity": 0.12, "region": 0.08},
        post_correction_drift={"ethnicity": 0.03, "region": 0.02},
    )
    
    # Pass census distributions for proper L1 error calculation
    report_gen = TrustReportGenerator(
        census_distributions=persona_gen.census,
        use_case=UseCaseProfile.NARRATIVE_REQUIRED
    )
    
    from datetime import datetime, timedelta
    report = report_gen.generate_report(
        personas,
        provenance_info={
            "sex": {"provenance": "PXWEB_DIRECT"},
            "region": {"provenance": "PXWEB_DIRECT"},
            "ethnicity": {"provenance": "CENSUS_HARDCODED"},
            "education": {"provenance": "PXWEB_DIRECT"},
            "employment_status": {"provenance": "PXWEB_DIRECT"},
            "residence_type": {"provenance": "PXWEB_DIRECT"},
            "age_group": {"provenance": "PXWEB_DIRECT"},
            "religion": {"provenance": "CENSUS_HARDCODED"},
            "marital_status": {"provenance": "PXWEB_DIRECT"},
            "language": {"provenance": "PXWEB_DIRECT"},
        },
        ipf_metrics=ipf,
        pxweb_cache_timestamp=(datetime.now() - timedelta(days=3)).isoformat(),
        data_reference_year=2024,
        census_reference_year=2024,
        random_seed=42,
        generator_version="0.3.0",
        generator_config={"seed": 42, "use_ipf": True, "adult_only": True},
        targets_source="mixed",
    )
    
    print(report.summary())
    
    print("\n" + "=" * 70)
    print("JSON EXPORT (truncated)")
    print("=" * 70)
    
    report_dict = report.to_dict()
    # Show key fields
    preview = {
        "report_id": report_dict["report_id"],
        "quality_tiers": {
            "structured": report_dict["structured_quality_tier"],
            "narrative": report_dict["narrative_quality_tier"],
            "overall": report_dict["overall_quality_tier"],
            "reasoning": report_dict["overall_tier_reasoning"],
        },
        "persona_count": report_dict["persona_count"],
        "provenance": {
            "critical_fallback_ratio": report_dict["provenance_coverage_critical"].get("CENSUS_HARDCODED", 0),
        },
        "ipf": report_dict["ipf_metrics"],
        "cache_age_days": report_dict["pxweb_cache_age_days"],
        "config_hash_prefix": report_dict["generator_config_hash"][:16] + "..." if report_dict["generator_config_hash"] else None,
    }
    print(json.dumps(preview, indent=2))


if __name__ == "__main__":
    demo_use_case_tiers()
    demo_ess_gate()
    demo_super_critical_fields()
    demo_locality_integrity()
    demo_narrative_quality()
    demo_data_currency()
    demo_full_report_summary()
    
    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETED SUCCESSFULLY")
    print("=" * 70)
