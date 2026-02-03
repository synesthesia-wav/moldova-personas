"""Tests for OCEAN profiling helpers."""

import random

from moldova_personas.generator import PersonaGenerator
from moldova_personas.ocean_framework import OCEANSampler, OCEANTextAnalyzer
from moldova_personas.ocean_nemo_schema import convert_to_nemo_schema


def test_ocean_sampler_does_not_affect_global_rng():
    random.seed(123)
    _ = random.random()

    sampler = OCEANSampler(seed=999)
    _ = sampler.sample(
        age=30,
        sex="Feminin",
        education_level="Liceal",
        occupation="Profesor",
    )

    second = random.random()

    random.seed(123)
    _ = random.random()
    expected_second = random.random()

    assert second == expected_second


def test_generator_includes_ocean_by_default():
    gen = PersonaGenerator(seed=42)
    persona = gen.generate_single()

    assert persona.ocean_openness is not None
    assert persona.ocean_profile is not None


def test_generator_can_disable_ocean():
    gen = PersonaGenerator(seed=42, include_ocean=False)
    persona = gen.generate_single()

    assert persona.ocean_openness is None
    assert persona.ocean_profile is None


def test_ocean_text_analyzer_empty_text():
    analyzer = OCEANTextAnalyzer()
    inferred = analyzer.analyze("")

    assert inferred.confidence == 0.0
    assert inferred.openness == 50
    assert inferred.conscientiousness == 50


def test_nemo_t_score_conversion():
    profile = convert_to_nemo_schema(
        {
            "openness": 50,
            "conscientiousness": 65,
            "extraversion": 35,
            "agreeableness": 50,
            "neuroticism": 50,
        }
    )

    assert profile["openness"]["t_score"] == 50
    assert profile["conscientiousness"]["t_score"] == 60
