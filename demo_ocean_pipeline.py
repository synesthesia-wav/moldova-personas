#!/usr/bin/env python3
"""
Demo script for Nemotron-style pipeline with OCEAN personality.

Shows the two-layer architecture:
  Layer 1: Structure (demographics + OCEAN + behavioral contract)
  Layer 2: Narrative (consistent with structure)

With score-and-rewrite loop for OCEAN validation.

Usage:
  python demo_ocean_pipeline.py --count 3 --provider dashscope --api-key <key>
"""

import asyncio
import argparse
import json
import os
from pathlib import Path
from typing import List

from moldova_personas.generator import PersonaGenerator
from moldova_personas.names import generate_name
from moldova_personas.nemotron_pipeline_v2 import generate_nemotron_v2_personas


async def main():
    parser = argparse.ArgumentParser(
        description="Generate Nemotron-style personas with OCEAN personality"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=3,
        help="Number of personas to generate (default: 3)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="ocean_personas.json",
        help="Output JSON file (default: ocean_personas.json)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        choices=["mock", "dashscope", "openai"],
        help="LLM provider (default: mock)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (or set DASHSCOPE_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen-turbo",
        help="Model name (default: qwen-turbo)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Max concurrent LLM calls (default: 3)"
    )
    parser.add_argument(
        "--ocean-tolerance",
        type=int,
        default=15,
        help="Max OCEAN deviation per trait (default: 15)"
    )
    parser.add_argument(
        "--max-rewrites",
        type=int,
        default=3,
        help="Max rewrite attempts (default: 3)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock generation (no LLM API calls)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("NEMOTRON-STYLE PIPELINE WITH OCEAN PERSONALITY")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Count: {args.count}")
    print(f"  Provider: {'mock' if args.mock else args.provider}")
    print(f"  OCEAN tolerance: ±{args.ocean_tolerance} points")
    print(f"  Max rewrites: {args.max_rewrites}")
    print(f"  Output: {args.output}")
    print()
    
    # Step 1: Generate structured personas
    print("Stage A: Generating structured persona bases...")
    generator = PersonaGenerator()
    base_personas = generator.generate(args.count)
    
    names = []
    for p in base_personas:
        name = generate_name(getattr(p, 'ethnicity', 'Moldovean'), p.sex)
        names.append(name)
    
    print(f"  Generated {len(base_personas)} base personas")
    for i, (p, name) in enumerate(zip(base_personas[:3], names[:3]), 1):
        print(f"    {i}. {name}, {p.age} ani, {p.occupation}, {p.district}")
    if len(base_personas) > 3:
        print(f"    ... and {len(base_personas) - 3} more")
    print()
    
    # Step 2: Run OCEAN pipeline
    print("=" * 70)
    print("TWO-PASS GENERATION WITH OCEAN")
    print("=" * 70)
    print()
    print("Pass 1: Structure (demographics → OCEAN → behavioral contract)")
    print("Pass 2: Narrative (OCEAN-guided generation)")
    print("Validation: Score-and-rewrite loop for OCEAN consistency")
    print()
    
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key and args.provider == "dashscope" and not args.mock:
        print("Error: DASHSCOPE_API_KEY not set")
        return 1
    
    try:
        ocean_personas = await generate_nemotron_v2_personas(
            personas=base_personas,
            names=names,
            provider="mock" if args.mock else args.provider,
            api_key=api_key,
            model=args.model,
            max_concurrent=args.max_concurrent,
            ocean_tolerance=args.ocean_tolerance,
            max_rewrites=args.max_rewrites
        )
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\nGenerated {len(ocean_personas)} personas with OCEAN")
    print()
    
    # Display sample
    if ocean_personas:
        sample = ocean_personas[0]
        print("-" * 70)
        print("SAMPLE OUTPUT (first persona):")
        print("-" * 70)
        
        # Layer 1: Structure
        print("\n📊 LAYER 1: STRUCTURED DATA")
        print(f"  UUID: {sample.uuid}")
        print(f"\n  Demographics:")
        print(f"    Name: {names[0]}")
        print(f"    Sex: {sample.sex}, Age: {sample.age}")
        print(f"    Education: {sample.education_level}")
        print(f"    Occupation: {sample.occupation}")
        print(f"    Location: {sample.locality}, {sample.municipality}")
        
        print(f"\n  🧠 OCEAN Profile:")
        print(f"    Openness (O):          {sample.ocean_openness}/100")
        print(f"    Conscientiousness (C): {sample.ocean_conscientiousness}/100")
        print(f"    Extraversion (E):      {sample.ocean_extraversion}/100")
        print(f"    Agreeableness (A):     {sample.ocean_agreeableness}/100")
        print(f"    Neuroticism (N):       {sample.ocean_neuroticism}/100")
        print(f"    Source: {sample.ocean_source}, Confidence: {sample.ocean_confidence:.2f}")
        
        print(f"\n  📋 Behavioral Contract:")
        print(f"    Risk tolerance: {sample.behavior_risk_tolerance}")
        print(f"    Decision style: {sample.behavior_decision_style}")
        print(f"    Conflict style: {sample.behavior_conflict_style}")
        print(f"    Social pattern: {sample.behavior_social_pattern}")
        print(f"    Novelty seeking: {sample.behavior_novelty_seeking}")
        print(f"    Will say 'no' probability: {sample.behavior_dissent_probability:.0%}")
        print(f"    Complaint likelihood: {sample.behavior_complaint_likelihood}")
        print(f"    Planning horizon: {sample.behavior_planning_horizon}")
        
        # Layer 2: Narrative
        print(f"\n📝 LAYER 2: NARRATIVE")
        print(f"  General persona:")
        print(f"    {sample.persona[:150]}...")
        print(f"\n  Professional:")
        print(f"    {sample.professional_persona[:150]}...")
        
        # Validation metadata
        print(f"\n✅ Validation:")
        print(f"    OCEAN deviation score: {sample.ocean_deviation_score}")
        print(f"    Rewrite count: {sample.rewrite_count}")
        
        print()
    
    # Save output
    output_data = {
        "metadata": {
            "format": "Nemotron-Moldova-v2-OCEAN",
            "count": len(ocean_personas),
            "provider": args.provider if not args.mock else "mock",
            "model": args.model if not args.mock else "n/a",
            "ocean_tolerance": args.ocean_tolerance,
            "max_rewrites": args.max_rewrites,
            "fields": [
                # Structure
                "uuid", "sex", "age", "marital_status", "education_level",
                "occupation", "municipality", "state", "locality", "country",
                "ethnicity", "religion",
                # OCEAN
                "ocean_openness", "ocean_conscientiousness", "ocean_extraversion",
                "ocean_agreeableness", "ocean_neuroticism", "ocean_source", "ocean_confidence",
                # Behavioral contract
                "behavior_risk_tolerance", "behavior_decision_style", "behavior_conflict_style",
                "behavior_social_pattern", "behavior_novelty_seeking",
                "behavior_dissent_probability", "behavior_complaint_likelihood", "behavior_planning_horizon",
                # Narrative
                "persona", "professional_persona", "sports_persona", "arts_persona",
                "travel_persona", "culinary_persona",
                "cultural_background", "skills_and_expertise", "skills_and_expertise_list",
                "hobbies_and_interests", "hobbies_and_interests_list", "career_goals_and_ambitions",
                # Validation
                "ocean_deviation_score", "rewrite_count"
            ]
        },
        "personas": [p.to_dict() for p in ocean_personas]
    }
    
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("=" * 70)
    print(f"✅ Saved {len(ocean_personas)} personas to {output_path}")
    print("=" * 70)
    print(f"\nFile size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"Records: {len(ocean_personas)}")
    print(f"Fields per record: 35 (24 base + 11 OCEAN/behavioral)")
    print()
    print("To generate with real LLM:")
    print(f"  python demo_ocean_pipeline.py --count 5 --provider dashscope --api-key <key>")
    print()
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
