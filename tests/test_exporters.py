"""Tests for exporters."""

from moldova_personas.exporters import StatisticsExporter


def test_statistics_exporter_empty_dataset():
    exporter = StatisticsExporter()
    stats = exporter.generate([])

    assert stats.total_count == 0
    assert stats.sex_distribution == {}
    assert stats.age_distribution == {}
    assert stats.region_distribution == {}
    assert stats.ethnicity_distribution == {}
    assert stats.education_distribution == {}
    assert stats.marital_status_distribution == {}
    assert stats.urban_rural_distribution == {}
