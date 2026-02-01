#!/usr/bin/env python3
"""
Demo: Generate and Display 5 Personas

Simple demonstration of the Moldova Personas Generator with
trust validation and quality gates.
"""

import json
from moldova_personas import PersonaGenerator, UseCaseProfile
from moldova_personas.gradient_integration import generate_dataset


def main():
    print('=' * 60)
    print('🇲🇩 Moldova Personas Generator - Demo (5 personas)')
    print('=' * 60)

    # Generate with trust validation
    bundle = generate_dataset(
        n=5,
        profile=UseCaseProfile.HIGH_STAKES,
        seed=42,
        use_ipf=True,
        strict=True,
    )

    print(f'\n📊 Generation Summary:')
    print(f'  Decision: {bundle.decision}')
    print(f'  Quality Tier: {bundle.quality_tier.value}')
    print(f'  Personas: {len(bundle.personas)}')
    print(f'  Time: {bundle.generation_time_seconds:.3f}s')
    print(f'  Gate Codes: {[c.value for c in bundle.gate_codes]}')

    if bundle.ipf_metrics:
        print(f'  ESS Ratio: {bundle.ipf_metrics.ess_ratio:.2%}')
        print(f'  Information Loss: {bundle.ipf_metrics.information_loss:.2%}')

    print('\n' + '=' * 60)
    print('Generated Personas:')
    print('=' * 60)

    for i, p in enumerate(bundle.personas, 1):
        print(f'\n👤 Persona {i}:')
        print(f'  UUID: {p.uuid}')
        print(f'  Name: {p.name}')
        print(f'  Age: {p.age} ({p.age_group})')
        print(f'  Sex: {p.sex}')
        print(f'  Education: {p.education_level}')
        if p.field_of_study:
            print(f'  Field: {p.field_of_study}')
        print(f'  Occupation: {p.occupation}')
        if p.occupation_sector:
            print(f'  Sector: {p.occupation_sector}')
        print(f'  Employment: {p.employment_status}')
        print(f'  Region: {p.region} ({p.residence_type})')
        print(f'  City: {p.city}, {p.district}')
        print(f'  Ethnicity: {p.ethnicity}')
        print(f'  Religion: {p.religion}')
        print(f'  Mother Tongue: {p.mother_tongue}')
        print(f'  Marital Status: {p.marital_status}')
        
        # Verify age-education consistency
        if p.education_level == 'Superior (Licență/Master)' and p.age < 19:
            print('  ⚠️  AGE-EDUCATION VIOLATION!')
        elif p.education_level == 'Doctorat' and p.age < 27:
            print('  ⚠️  AGE-EDUCATION VIOLATION!')
        else:
            print('  ✓ Age-education consistent')

    print('\n' + '=' * 60)
    print('Gradient Trust Payload:')
    print('=' * 60)
    payload = bundle.to_gradient_trust_payload()
    
    # Display key fields
    display_payload = {
        'payload_version': payload['payload_version'],
        'generator_version': payload['generator_version'],
        'schema_hash_short': payload['config_hash'][:16] + '...',
        'decision': payload['decision'],
        'confidence': payload['confidence'],
        'quality_tier': payload['quality_tier'],
        'gate_codes': payload['gate_codes'],
        'gate_code_details': payload['gate_code_details'],
        'hard_gates_triggered': payload['hard_gates_triggered'],
        'escalation_priority': payload['escalation_priority'],
        'profile': payload['profile'],
        'strict_mode': payload['strict_mode'],
        'signals': payload['signals'],
    }
    print(json.dumps(display_payload, indent=2, default=str))

    print('\n' + '=' * 60)
    print('Run Manifest Summary:')
    print('=' * 60)
    manifest = bundle.run_manifest
    print(f'  Run ID: {manifest.run_id}')
    print(f'  Config Hash: {manifest.config_hash_short}')
    print(f'  Generator Version: {manifest.generator_version}')
    print(f'  Target Count: {manifest.target_persona_count}')
    print(f'  IPF Applied: {manifest.ipf_applied}')
    print(f'  Quality Tier: {manifest.quality_tier_achieved}')
    print(f'  Hard Gate Triggered: {manifest.hard_gate_triggered}')

    print('\n✅ Demo complete!')
    print('\nTo save outputs:')
    print(f'  bundle.save("./output")')


if __name__ == "__main__":
    main()
