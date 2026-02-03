"""Tests for quality dashboard."""

from moldova_personas.quality_dashboard import QualityDashboard


def test_quality_dashboard_handles_current_narrative_fields():
    dashboard = QualityDashboard()
    personas = [
        {
            "descriere_generala": "Prefera sa lucreze dimineata si este organizat.",
            "profil_profesional": "Lucreaza ca profesor si preda matematica.",
            "hobby_sport": "Joaca fotbal in weekend.",
            "hobby_arta_cultura": "Asculta muzica clasica.",
            "hobby_calatorii": "Calatoreste in Romania.",
            "hobby_culinar": "Gateste ciorba.",
        }
    ]

    metrics = dashboard.analyze_dataset(personas)

    assert metrics.total_records == 1
    assert metrics.lexical_diversity_per_field["persona"] > 0
    assert sum(metrics.opening_move_distribution.values()) > 0
