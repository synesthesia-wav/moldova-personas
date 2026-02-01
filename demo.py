#!/usr/bin/env python3
"""
Demo script for Moldova Synthetic Personas Generator.

This script demonstrates the complete pipeline:
1. Generate sample personas
2. Validate them
3. Export to multiple formats
4. Show statistics and comparison to census
"""

import sys
from pathlib import Path

# Add the module to path
sys.path.insert(0, str(Path(__file__).parent))

from moldova_personas import PersonaGenerator, ValidationPipeline
from moldova_personas.exporters import export_all_formats, StatisticsExporter
from moldova_personas.census_data import CENSUS


def main():
    print("=" * 70)
    print("Moldova Synthetic Personas Generator - Demo")
    print("=" * 70)
    print()
    
    # Configuration
    SAMPLE_SIZE = 1000
    OUTPUT_DIR = "./demo_output"
    SEED = 42
    
    print(f"Configuration:")
    print(f"  Sample size: {SAMPLE_SIZE:,}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Random seed: {SEED}")
    print()
    
    # Step 1: Generate personas
    print("Step 1: Generating personas...")
    print("-" * 70)
    
    generator = PersonaGenerator(seed=SEED)
    personas = generator.generate(SAMPLE_SIZE, show_progress=True)
    
    print(f"✓ Generated {len(personas)} personas")
    print()
    
    # Step 2: Show example personas
    print("Step 2: Example personas (first 3)")
    print("-" * 70)
    
    for i, p in enumerate(personas[:3], 1):
        print(f"\nPersona #{i}:")
        print(f"  Name:       {p.name}")
        print(f"  Demographics: {p.sex}, {p.age} years, {p.ethnicity}")
        print(f"  Location:   {p.city}, {p.region} ({p.residence_type})")
        print(f"  Education:  {p.education_level}")
        if p.field_of_study:
            print(f"  Field:      {p.field_of_study}")
        print(f"  Occupation: {p.occupation}")
        print(f"  Marital:    {p.marital_status}")
    
    print()
    
    # Step 3: Validate
    print("Step 3: Validating personas...")
    print("-" * 70)
    
    validator = ValidationPipeline()
    report = validator.validate(personas, check_statistical=True, tolerance=0.03)
    
    print(report.summary())
    print()
    
    # Step 4: Generate and show statistics
    print("Step 4: Dataset statistics")
    print("-" * 70)
    
    stats = StatisticsExporter().generate(personas)
    
    print(f"\nTotal personas: {stats.total_count}")
    
    print("\nSex distribution:")
    for sex, prop in sorted(stats.sex_distribution.items()):
        target = CENSUS.SEX_DISTRIBUTION.get(sex, 0)
        diff = prop - target
        print(f"  {sex:10}: {prop:.3f} (target: {target:.3f}, diff: {diff:+.3f})")
    
    print("\nRegion distribution:")
    for region, prop in sorted(stats.region_distribution.items()):
        target = CENSUS.REGION_DISTRIBUTION.get(region, 0)
        diff = prop - target
        print(f"  {region:10}: {prop:.3f} (target: {target:.3f}, diff: {diff:+.3f})")
    
    print("\nEthnicity distribution (top 5):")
    for eth, prop in sorted(stats.ethnicity_distribution.items(), 
                           key=lambda x: x[1], reverse=True)[:5]:
        target = CENSUS.ETHNICITY_DISTRIBUTION.get(eth, 0)
        diff = prop - target
        print(f"  {eth:12}: {prop:.3f} (target: {target:.3f}, diff: {diff:+.3f})")
    
    print("\nUrban/Rural distribution:")
    for res, prop in sorted(stats.urban_rural_distribution.items()):
        target = CENSUS.RESIDENCE_TYPE_DISTRIBUTION.get(res, 0)
        diff = prop - target
        print(f"  {res:10}: {prop:.3f} (target: {target:.3f}, diff: {diff:+.3f})")
    
    print()
    
    # Step 5: Export
    print("Step 5: Exporting to multiple formats...")
    print("-" * 70)
    
    results = export_all_formats(personas, OUTPUT_DIR, basename="demo_personas")
    
    print("Exported files:")
    for fmt, path in results.items():
        size = Path(path).stat().st_size / 1024  # KB
        print(f"  {fmt:18}: {path} ({size:8.1f} KB)")
    
    print()
    print("=" * 70)
    print("Demo complete! Check the output files in:", OUTPUT_DIR)
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
