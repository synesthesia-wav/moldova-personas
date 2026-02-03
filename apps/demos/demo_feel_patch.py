#!/usr/bin/env python3
"""
Demo of feel patches applied to persona generation.

Shows improvements:
1. Implicit OCEAN (no trait words, only behavioral evidence)
2. Varied openings (no repetitive "În timpul liber...")
3. Realistic constraints (friction, not just positivity)
4. Strong Moldova anchors (specific, not generic)

Usage:
  python apps/demos/demo_feel_patch.py --count 3 --provider dashscope --api-key <key>
"""

from _bootstrap import ensure_core_path

ensure_core_path()


import asyncio
import argparse
import json
import os
from pathlib import Path

from moldova_personas.generator import PersonaGenerator
from moldova_personas.names import generate_name
from moldova_personas.ocean_framework import OCEANSampler
from moldova_personas.persona_feel_patch import (
    generate_behavioral_cues,
    generate_realism_constraints,
    select_anchors,
    generate_stage_b_prompt_feel,
    generate_stage_c_prompt_feel,
    PersonaFeelValidator,
    OPENING_MOVES,
    validate_trait_leak,
    validate_not_pollyanna,
)
from moldova_personas.llm_client import create_llm_client, GenerationConfig


async def generate_one_persona_feel(
    base_persona,
    name: str,
    llm_client,
    validator: PersonaFeelValidator
) -> dict:
    """Generate one persona with feel patches."""
    
    # Extract OCEAN scores
    sampler = OCEANSampler()
    ocean_profile = sampler.sample(
        age=base_persona.age,
        sex=base_persona.sex,
        education_level=base_persona.education_level,
        occupation=base_persona.occupation
    )
    
    ocean_scores = {
        'openness': ocean_profile.openness,
        'conscientiousness': ocean_profile.conscientiousness,
        'extraversion': ocean_profile.extraversion,
        'agreeableness': ocean_profile.agreeableness,
        'neuroticism': ocean_profile.neuroticism
    }
    
    # Generate behavioral cues (hidden, guides generation)
    behavioral_cues = generate_behavioral_cues(ocean_scores)
    
    # Generate constraints
    constraints = generate_realism_constraints(2)
    
    # Select anchors
    place_anchor, routine_anchor = select_anchors(
        base_persona.region,
        base_persona.district
    )
    
    # Build demographics
    demographics = {
        'age': base_persona.age,
        'sex': base_persona.sex,
        'occupation': base_persona.occupation,
        'education_level': base_persona.education_level,
        'locality': base_persona.city,
        'district': base_persona.district,
        'region': base_persona.region
    }
    
    # === STAGE B: Generate Context ===
    stage_b_prompt = generate_stage_b_prompt_feel(
        name, demographics, ocean_scores
    )
    
    response_b = await asyncio.to_thread(
        llm_client.generate,
        stage_b_prompt,
        GenerationConfig(max_tokens=1500, temperature=0.7)
    )
    
    # Parse context
    try:
        import re
        json_match = re.search(r'\{.*\}', response_b, re.DOTALL)
        if json_match:
            context = json.loads(json_match.group())
        else:
            context = {}
    except:
        context = {}
    
    # === STAGE C: Generate 6 Personas ===
    # Assign varied opening moves
    import random
    opening_moves = OPENING_MOVES.copy()
    random.shuffle(opening_moves)
    
    stage_c_prompt = generate_stage_c_prompt_feel(
        name, demographics, context, behavioral_cues, opening_moves
    )
    
    response_c = await asyncio.to_thread(
        llm_client.generate,
        stage_c_prompt,
        GenerationConfig(max_tokens=1200, temperature=0.7)
    )
    
    # Parse personas
    try:
        json_match = re.search(r'\{.*\}', response_c, re.DOTALL)
        if json_match:
            personas = json.loads(json_match.group())
        else:
            personas = {}
    except:
        personas = {}
    
    # === VALIDATION ===
    all_texts = {
        **context,
        **personas
    }
    
    validation = validator.validate_full(all_texts, ocean_scores)
    
    return {
        "uuid": base_persona.uuid,
        "name": name,
        "demographics": demographics,
        "ocean_scores": ocean_scores,
        "behavioral_cues": behavioral_cues.to_prompt(),
        "constraints": constraints,
        "anchors": {
            "place": place_anchor,
            "routine": routine_anchor
        },
        "context": context,
        "personas": personas,
        "validation": validation,
        "passed_validation": validation["passed"]
    }


async def main():
    parser = argparse.ArgumentParser(description="Feel patch demo")
    parser.add_argument("--count", "-n", type=int, default=3)
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", "-o", default="feel_patch_output.json")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("NEMOTRON-FEEL PATCHES DEMO")
    print("=" * 70)
    print("\nImprovements:")
    print("  ✓ Implicit OCEAN (no trait words)")
    print("  ✓ Varied openings (6 different styles)")
    print("  ✓ Realistic constraints (friction, not just positivity)")
    print("  ✓ Strong Moldova anchors (specific places, food, routines)")
    print()
    
    # Setup
    generator = PersonaGenerator()
    base_personas = generator.generate(args.count)
    
    names = [generate_name(getattr(p, 'ethnicity', 'Moldovean'), p.sex) 
             for p in base_personas]
    
    # LLM client
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")
    if args.provider == "dashscope" and not api_key:
        print("Error: DASHSCOPE_API_KEY not set")
        return 1
    
    if args.provider == "mock":
        llm_client = create_llm_client("mock")
    else:
        llm_client = create_llm_client(args.provider, api_key=api_key)
    
    # Validator (use first persona's region for anchor validation)
    validator = PersonaFeelValidator(
        region=base_personas[0].region if base_personas else "Centru"
    )
    
    # Generate
    print(f"Generating {args.count} personas with feel patches...\n")
    
    results = []
    for i, (base, name) in enumerate(zip(base_personas, names), 1):
        print(f"  Persona {i}/{args.count}: {name}...")
        
        if args.provider == "mock":
            # Create mock output for demonstration
            ocean_scores = {
                'openness': 55, 'conscientiousness': 45,
                'extraversion': 60, 'agreeableness': 50,
                'neuroticism': 40
            }
            behavioral_cues = generate_behavioral_cues(ocean_scores)
            constraints = generate_realism_constraints(2)
            place, routine = select_anchors(base.region, base.district)
            
            result = {
                "uuid": base.uuid,
                "name": name,
                "demographics": {
                    'age': base.age, 'sex': base.sex,
                    'occupation': base.occupation,
                    'region': base.region
                },
                "ocean_scores": ocean_scores,
                "behavioral_cues": behavioral_cues.to_prompt(),
                "constraints": constraints,
                "anchors": {"place": place, "routine": routine},
                "context": {
                    "cultural_background": f"[MOCK] Context pentru {name} din {base.city}...",
                    "career_goals": "[MOCK] Obiective realiste..."
                },
                "personas": {
                    "persona": f"[MOCK] {name.split()[0]} este...",
                    "professional_persona": f"[MOCK] La muncă...",
                    "sports_persona": f"[MOCK] Sport...",
                    "arts_persona": f"[MOCK] Artă...",
                    "travel_persona": f"[MOCK] Călătorii...",
                    "culinary_persona": f"[MOCK] Gatit..."
                },
                "validation": {"passed": True, "checks": {}},
                "passed_validation": True
            }
        else:
            result = await generate_one_persona_feel(
                base, name, llm_client, validator
            )
        
        results.append(result)
    
    # Display sample
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT")
    print("=" * 70)
    
    sample = results[0]
    
    print(f"\n📊 STRUCTURE")
    print(f"  Name: {sample['name']}")
    print(f"  {sample['demographics']['sex']}, {sample['demographics']['age']} ani")
    print(f"  {sample['demographics']['occupation']}")
    print(f"  {sample['demographics']['region']}")
    
    print(f"\n  🧠 OCEAN (implicit - no trait words in output):")
    for trait, score in sample['ocean_scores'].items():
        print(f"    {trait}: {score}")
    
    print(f"\n  📝 Behavioral Cues (hidden, guides generation):")
    for line in sample['behavioral_cues'].strip().split('\n')[1:]:  # Skip header
        if line.strip():
            print(f"    {line}")
    
    print(f"\n  ⚠️  Constraints (realistic friction):")
    for c in sample['constraints']:
        print(f"    - {c}")
    
    print(f"\n  📍 Anchors (specific Moldova):")
    print(f"    Place: {sample['anchors']['place']}")
    print(f"    Routine: {sample['anchors']['routine']}")
    
    if not args.provider == "mock":
        print(f"\n  📝 Sample Narrative:")
        if 'personas' in sample and 'persona' in sample['personas']:
            print(f"    {sample['personas']['persona'][:200]}...")
        
        print(f"\n  ✅ Validation:")
        val = sample['validation']
        print(f"    Passed: {val['passed']}")
        for check_name, check_result in val.get('checks', {}).items():
            status = "✓" if check_result.get('passed') else "✗"
            print(f"    {status} {check_name}")
    
    # Save
    output = {
        "metadata": {
            "format": "Moldova-Personas-Feel-Patched",
            "count": len(results),
            "features": [
                "implicit_ocean",
                "varied_openings", 
                "realism_constraints",
                "moldova_anchors"
            ]
        },
        "personas": results
    }
    
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print(f"✅ Saved {len(results)} personas to {output_path}")
    print("=" * 70)
    
    # Summary
    passed = sum(1 for r in results if r.get('passed_validation', True))
    print(f"\nValidation: {passed}/{len(results)} passed")
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
