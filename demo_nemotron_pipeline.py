#!/usr/bin/env python3
"""
Demo script for Nemotron-style pipeline.

Generates Moldova personas using the 3-stage compound approach:
  Stage A: Structured core (census-grounded)
  Stage B: Context fields (long narratives)
  Stage C: Short personas (6 variants)

Usage:
  python demo_nemotron_pipeline.py --count 3 --output sample_nemotron.json
  
With LLM:
  python demo_nemotron_pipeline.py --count 3 --provider dashscope --api-key <key>
"""

import asyncio
import argparse
import json
import os
from pathlib import Path
from typing import List

from moldova_personas.generator import PersonaGenerator
from moldova_personas.names import generate_name
from moldova_personas.nemotron_pipeline import (
    generate_nemotron_personas,
    NemotronPersona,
    NemotronPipeline,
    PersonaCore,
    ContextFields,
    ShortPersonas
)
from moldova_personas.llm_client import create_llm_client


# Mock pipeline for testing without API
def create_mock_nemotron_persona(core: PersonaCore) -> NemotronPersona:
    """Create a mock Nemotron-style persona for testing."""
    context = ContextFields(
        cultural_background=(
            f"{core.first_name if hasattr(core, 'first_name') else 'Persoana'} locuiește în {core.locality}, "
            f"un sat din raionul {core.raion}, regiunea {core.region} a Moldovei. "
            f"Zona este cunoscută pentru tradițiile sale agricole și comunitatea strâns unită. "
            f"Localnicii prețuiesc valorile familiale și petrec timpul împreună la sărbători."
        ),
        skills_and_expertise=(
            f"Expert în {core.occupation.lower()}, cu abilități practice dobândite prin experiență. "
            f"Cunoaște bine domeniul și lucrează cu atenție la detalii. "
            f"Comunică eficient cu colegii și clienții."
        ),
        skills_and_expertise_list=[
            f"Expertiză în {core.occupation.lower()}",
            "Lucru în echipă",
            "Comunicare",
            "Rezolvare probleme",
            "Organizare"
        ],
        hobbies_and_interests=(
            f"În timpul liber, îi place să citească și să petreacă timp în natură. "
            f"Participă la evenimente culturale locale și vizitează familia. "
            f"Îi place să gătească bucate tradiționale moldovenești."
        ),
        hobbies_and_interests_list=[
            "Citit",
            "Plimbări în natură",
            "Evenimente culturale",
            "Gătit tradițional",
            "Vizite la familie"
        ],
        career_goals_and_ambitions=(
            f"Își dorește să-și dezvolte cariera în domeniul {core.occupation.lower()}, "
            f"să învețe noi tehnici și să-și îmbunătățească abilitățile profesionale."
        )
    )
    
    personas = ShortPersonas(
        persona=(
            f"{core.first_name if hasattr(core, 'first_name') else 'Persoana'}, {core.age} ani, "
            f"este o persoană dedicată familiei și muncii, cu rădăcini adânci în tradițiile moldovenești."
        ),
        professional_persona=(
            f"{core.first_name if hasattr(core, 'first_name') else 'Persoana'} lucrează ca {core.occupation.lower()} "
            f"și se mândrește cu profesionalismul și dedicarea sa. Abordează fiecare sarcină cu seriozitate."
        ),
        sports_persona=(
            f"În timpul liber, {core.first_name if hasattr(core, 'first_name') else 'persoana'} "
            f"îi place să se plimbe prin parc și să practice activități ușoare în aer liber."
        ),
        arts_persona=(
            f"{core.first_name if hasattr(core, 'first_name') else 'Persoana'} apreciază muzica populară moldovenească "
            f"și participă la evenimente culturale în comunitatea locală."
        ),
        travel_persona=(
            f"Își dorește să viziteze mănăstirile din Moldova și să călătorească în România, "
            f"țara de origine a limbii și culturii noastre."
        ),
        culinary_persona=(
            f"La masă, preferă bucatele tradiționale moldovenești: mămăligă cu brânză și smântână, "
            f"sarmale și plăcinte. Gătește cu plăcere pentru familie."
        )
    )
    
    return NemotronPersona.from_stages(core, context, personas)


async def main():
    parser = argparse.ArgumentParser(
        description="Generate Nemotron-style Moldova personas"
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
        default="nemotron_personas.json",
        help="Output JSON file (default: nemotron_personas.json)"
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
        "--mock",
        action="store_true",
        help="Use mock generation (no LLM API calls)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NEMOTRON-STYLE PIPELINE FOR MOLDOVA PERSONAS")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Count: {args.count}")
    print(f"  Provider: {'mock' if args.mock else args.provider}")
    print(f"  Output: {args.output}")
    print()
    
    # Step 1: Generate structured personas
    print("Stage A: Generating structured persona cores...")
    generator = PersonaGenerator()
    structured_personas = generator.generate(args.count)
    
    # Get names
    names = []
    for p in structured_personas:
        # Generate appropriate name based on sex and ethnicity
        name = generate_name(getattr(p, 'ethnicity', 'Moldovean'), p.sex)
        names.append(name)
    
    print(f"  Generated {len(structured_personas)} structured cores")
    for i, (p, name) in enumerate(zip(structured_personas[:3], names[:3]), 1):
        print(f"    {i}. {name}, {p.age} ani, {p.occupation}, {p.district}")
    if len(structured_personas) > 3:
        print(f"    ... and {len(structured_personas) - 3} more")
    print()
    
    # Step 2 & 3: Generate Nemotron-style personas
    if args.mock or args.provider == "mock":
        print("Stage B+C: Using MOCK generation (no LLM calls)...")
        print("  (Use --provider dashscope --api-key <key> for real LLM generation)")
        print()
        
        nemotron_personas = []
        for p, name in zip(structured_personas, names):
            core = PersonaCore(
                uuid=p.uuid,
                sex=p.sex,
                age=p.age,
                marital_status=p.marital_status,
                education_level=p.education_level,
                occupation=p.occupation,
                region=p.region,
                raion=p.district,
                locality=getattr(p, 'locality', p.city),
                country="Moldova",
                ethnicity=getattr(p, 'ethnicity', 'Moldovean'),
                religion=getattr(p, 'religion', 'Ortodox'),
                employment_status=getattr(p, 'employment_status', 'Employat'),
                migration_background=getattr(p, 'migration_background', 'Rezident')
            )
            # Add first_name attribute for mock generation
            core.first_name = name.split()[0]
            nemotron_personas.append(create_mock_nemotron_persona(core))
    else:
        print("Stage B: Generating context fields (long narratives)...")
        print("Stage C: Generating short persona variants...")
        print(f"  Provider: {args.provider}")
        print(f"  Model: {args.model}")
        print(f"  Max concurrent: {args.max_concurrent}")
        print()
        
        api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key and args.provider == "dashscope":
            print("Error: DASHSCOPE_API_KEY not set. Use --api-key or env var.")
            return 1
        
        try:
            nemotron_personas = await generate_nemotron_personas(
                personas=structured_personas,
                names=names,
                provider=args.provider,
                api_key=api_key,
                model=args.model,
                max_concurrent=args.max_concurrent
            )
        except Exception as e:
            print(f"Error during generation: {e}")
            print("\nFalling back to mock generation...")
            nemotron_personas = []
            for p, name in zip(structured_personas, names):
                core = PersonaCore(
                    uuid=p.uuid,
                    sex=p.sex,
                    age=p.age,
                    marital_status=p.marital_status,
                    education_level=p.education_level,
                    occupation=p.occupation,
                    region=p.region,
                    raion=p.raion,
                    locality=getattr(p, 'locality', p.raion),
                    country="Moldova"
                )
                core.first_name = name.split()[0]
                nemotron_personas.append(create_mock_nemotron_persona(core))
    
    print(f"Generated {len(nemotron_personas)} Nemotron-style personas")
    print()
    
    # Display sample
    if nemotron_personas:
        sample = nemotron_personas[0]
        print("-" * 60)
        print("SAMPLE OUTPUT (first persona):")
        print("-" * 60)
        print(f"\nUUID: {sample.uuid}")
        print(f"\n📋 DEMOGRAPHICS:")
        print(f"  Sex: {sample.sex}")
        print(f"  Age: {sample.age}")
        print(f"  Education: {sample.education_level}")
        print(f"  Occupation: {sample.occupation}")
        print(f"  Location: {sample.locality}, {sample.municipality}, {sample.state}")
        print(f"\n📝 GENERAL PERSONA (essence):")
        print(f"  {sample.persona}")
        print(f"\n💼 PROFESSIONAL:")
        print(f"  {sample.professional_persona}")
        print(f"\n⚽ SPORTS:")
        print(f"  {sample.sports_persona}")
        print(f"\n🎨 ARTS:")
        print(f"  {sample.arts_persona}")
        print(f"\n✈️ TRAVEL:")
        print(f"  {sample.travel_persona}")
        print(f"\n🍽️ CULINARY:")
        print(f"  {sample.culinary_persona}")
        print(f"\n🌍 CULTURAL BACKGROUND:")
        print(f"  {sample.cultural_background[:200]}...")
        print(f"\n🎯 SKILLS: {sample.skills_and_expertise_list}")
        print(f"\n❤️ HOBBIES: {sample.hobbies_and_interests_list}")
        print()
    
    # Save to file
    output_data = {
        "metadata": {
            "format": "Nemotron-style",
            "count": len(nemotron_personas),
            "provider": args.provider if not args.mock else "mock",
            "model": args.model if not args.mock else "n/a",
            "country": "Moldova",
            "fields": [
                "uuid", "persona", "professional_persona", "sports_persona",
                "arts_persona", "travel_persona", "culinary_persona",
                "cultural_background", "skills_and_expertise", "skills_and_expertise_list",
                "hobbies_and_interests", "hobbies_and_interests_list",
                "career_goals_and_ambitions", "sex", "age", "marital_status",
                "education_level", "occupation", "municipality", "state", "country",
                "locality", "ethnicity", "religion"
            ]
        },
        "personas": [p.to_dict() for p in nemotron_personas]
    }
    
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("=" * 60)
    print(f"✅ Saved {len(nemotron_personas)} personas to {output_path}")
    print("=" * 60)
    print(f"\nFile size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"Records: {len(nemotron_personas)}")
    print(f"Fields per record: 24")
    print()
    print("To generate with LLM:")
    print(f"  python demo_nemotron_pipeline.py --count 5 --provider dashscope --api-key <key>")
    print()
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
