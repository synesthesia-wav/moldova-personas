#!/usr/bin/env python3
"""
Demo script for Moldova Synthetic Personas with LLM narratives.

This demonstrates:
1. Generating structured personas
2. Adding narrative content with LLM
3. Viewing complete profiles

Usage:
    # Mock mode (no API calls)
    python demo_narrative.py --mode mock
    
    # With OpenAI (requires API key)
    export OPENAI_API_KEY="your-key"
    python demo_narrative.py --mode openai --count 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from moldova_personas import PersonaGenerator
from moldova_personas.narrative_generator import NarrativeGenerator


def main():
    parser = argparse.ArgumentParser(description="Demo narrative generation")
    parser.add_argument('--mode', type=str, default='mock', 
                       choices=['mock', 'openai', 'gemini', 'kimi', 'qwen', 'qwen-local', 'dashscope'],
                       help='LLM provider mode')
    parser.add_argument('--count', type=int, default=3,
                       help='Number of personas to generate')
    parser.add_argument('--api-key', type=str, default=None,
                       help='API key (OpenAI or DashScope)')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name (e.g., gpt-3.5-turbo, qwen-turbo, qwen2.5-7b)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Moldova Synthetic Personas - Narrative Generation Demo")
    print("=" * 80)
    print(f"\nMode: {args.mode}")
    print(f"Count: {args.count}")
    
    # Step 1: Generate structured personas
    print("\n" + "-" * 80)
    print("Step 1: Generating structured personas...")
    print("-" * 80)
    
    generator = PersonaGenerator(seed=42)
    personas = generator.generate(args.count, show_progress=False)
    
    print(f"✓ Generated {len(personas)} structured personas")
    
    # Step 2: Add narratives
    print("\n" + "-" * 80)
    print("Step 2: Generating narrative content...")
    print("-" * 80)
    
    kwargs = {}
    if args.api_key:
        kwargs['api_key'] = args.api_key
    if args.model:
        # Map model names appropriately
        if args.mode in ('qwen', 'dashscope'):
            kwargs['model'] = args.model
        elif args.mode == 'openai':
            kwargs['model'] = args.model
        elif args.mode == 'qwen-local':
            kwargs['model_name'] = args.model
    
    nar_gen = NarrativeGenerator(provider=args.mode, **kwargs)
    
    if args.mode == "mock":
        print("(Using mock mode - no actual LLM calls)")
    else:
        print("(This will make API calls - may take a minute...)")
    
    personas = nar_gen.generate_batch(personas, show_progress=True)
    
    print(f"✓ Generated narratives")
    
    # Step 3: Display complete profiles
    print("\n" + "=" * 80)
    print("Step 3: Complete Persona Profiles")
    print("=" * 80)
    
    for i, p in enumerate(personas, 1):
        print(f"\n{'#'*80}")
        print(f"PERSONA #{i}: {p.name}")
        print(f"{'#'*80}")
        
        print(f"\n📋 STRUCTURED DATA:")
        print(f"   UUID:         {p.uuid}")
        print(f"   Demographics: {p.sex}, {p.age} years, {p.ethnicity}")
        print(f"   Location:     {p.city}, {p.region} ({p.residence_type})")
        print(f"   Education:    {p.education_level}")
        print(f"   Occupation:   {p.occupation}")
        print(f"   Marital:      {p.marital_status}")
        
        print(f"\n📝 NARRATIVE CONTENT:")
        
        if p.cultural_background:
            print(f"\n   [Cultural Background]")
            print(f"   {p.cultural_background}")
        
        if p.descriere_generala:
            print(f"\n   [Descriere Generală]")
            print(f"   {p.descriere_generala[:300]}..." if len(p.descriere_generala) > 300 else f"   {p.descriere_generala}")
        
        if p.profil_profesional:
            print(f"\n   [Profil Profesional]")
            print(f"   {p.profil_profesional[:300]}..." if len(p.profil_profesional) > 300 else f"   {p.profil_profesional}")
        
        if p.hobby_sport:
            print(f"\n   [Hobby Sport]")
            print(f"   {p.hobby_sport[:200]}..." if len(p.hobby_sport) > 200 else f"   {p.hobby_sport}")
        
        if p.skills_and_expertise_list:
            print(f"\n   [Skills]: {', '.join(p.skills_and_expertise_list[:5])}")
        
        if p.hobbies_and_interests_list:
            print(f"   [Hobbies]: {', '.join(p.hobbies_and_interests_list[:5])}")
    
    # Step 4: Export sample
    print("\n" + "=" * 80)
    print("Step 4: Exporting sample...")
    print("=" * 80)
    
    output_dir = Path("./demo_output_narrative")
    output_dir.mkdir(exist_ok=True)
    
    # Export first persona as JSON for inspection
    import json
    sample_file = output_dir / "sample_persona.json"
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(personas[0].model_dump(), f, indent=2, ensure_ascii=False)
    
    print(f"✓ Sample exported to: {sample_file}")
    
    # Cost estimate for full 100k
    if args.mode in ("openai", "qwen", "dashscope"):
        print("\n" + "=" * 80)
        print("💰 COST ESTIMATE FOR FULL DATASET (100k personas)")
        print("=" * 80)
        
        avg_tokens = 500  # per persona
        total_tokens = 100_000 * avg_tokens
        
        if args.mode == "openai":
            # GPT-3.5-turbo pricing (as of 2024)
            input_price = 0.0015  # per 1K tokens
            output_price = 0.002   # per 1K tokens
            
            input_tokens = total_tokens * 0.6
            output_tokens = total_tokens * 0.4
            
            cost = (input_tokens / 1000 * input_price) + (output_tokens / 1000 * output_price)
            
            print(f"Estimated tokens: {total_tokens:,}")
            print(f"Estimated cost with GPT-3.5-turbo: ${cost:.2f}")
            print(f"Estimated cost with GPT-4: ~${cost * 10:.2f}")
        
        else:  # Qwen/DashScope
            # DashScope pricing (approximate, as of 2024)
            # qwen-turbo: ~$0.0005/1K tokens
            # qwen-plus: ~$0.002/1K tokens
            # qwen-max: ~$0.01/1K tokens
            
            model = args.model or "qwen-turbo"
            if "turbo" in model:
                price_per_1k = 0.0005
            elif "plus" in model:
                price_per_1k = 0.002
            elif "max" in model:
                price_per_1k = 0.01
            else:
                price_per_1k = 0.0005  # default to turbo
            
            cost = total_tokens / 1000 * price_per_1k
            
            print(f"Estimated tokens: {total_tokens:,}")
            print(f"Estimated cost with {model}: ${cost:.2f}")
            print(f"\nQwen pricing (via DashScope):")
            print(f"  qwen-turbo: ~$0.05/1k personas (fastest)")
            print(f"  qwen-plus:  ~$0.20/1k personas")
            print(f"  qwen-max:   ~$1.00/1k personas (highest quality)")
        
        print(f"\nTip: Use batch mode for 50% discount!")
    
    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
