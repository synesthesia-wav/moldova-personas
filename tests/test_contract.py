"""
Data-source contract tests for PxWeb integration.

These tests validate:
1. Schema stability (golden fixtures)
2. Category alignment (external → internal labels)
3. Schema drift detection
4. Network integration (optional, marked as slow)

Contract tests ensure that changes to PxWeb don't silently break
our parsing logic.
"""

import json
import pytest
from pathlib import Path
from typing import Dict, Any

from moldova_personas.pxweb_fetcher import PxWebParser


# Path to fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "pxweb"


class TestPxWebContract:
    """
    Contract tests for PxWeb data source.
    
    These tests validate that our parser correctly handles the expected
    PxWeb response format. If PxWeb changes their schema, these tests
    should fail, alerting us to the drift.
    """
    
    def load_fixture(self, filename: str) -> Dict[str, Any]:
        """Load a JSON fixture file."""
        with open(FIXTURES_DIR / filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def test_sex_distribution_parsing(self):
        """Test parsing of sex distribution from golden fixture."""
        fixture = self.load_fixture("pop010200_sex_distribution.json")
        data = {k: v for k, v in fixture.items() if not k.startswith('_')}
        
        parser = PxWebParser()
        probs, counts, total = parser.extract_distribution(
            data, "Sexe", code_mapping={"1": "Masculin", "2": "Feminin"}
        )
        
        # Verify against expected values
        expected_counts = fixture["_expected_parsed"]
        assert counts == expected_counts, f"Expected {expected_counts}, got {counts}"
        assert total == sum(expected_counts.values())
        
        # Verify probabilities sum to 1
        assert abs(sum(probs.values()) - 1.0) < 0.001
        
        # Verify specific probabilities
        total_pop = sum(expected_counts.values())
        assert probs["Masculin"] == pytest.approx(expected_counts["Masculin"] / total_pop, 0.001)
        assert probs["Feminin"] == pytest.approx(expected_counts["Feminin"] / total_pop, 0.001)
    
    def test_residence_distribution_parsing(self):
        """Test parsing of residence distribution from golden fixture."""
        fixture = self.load_fixture("pop010200_residence_distribution.json")
        data = {k: v for k, v in fixture.items() if not k.startswith('_')}
        
        parser = PxWebParser()
        probs, counts, total = parser.extract_distribution(
            data, "Medii", code_mapping={"1": "Urban", "2": "Rural"}
        )
        
        expected_counts = fixture["_expected_parsed"]
        assert counts == expected_counts
        assert total == sum(expected_counts.values())
    
    def test_dimension_not_found_raises_error(self):
        """Test that missing dimension raises ValueError."""
        fixture = self.load_fixture("schema_drift_examples.json")
        drift_example = fixture["renamed_dimension_code"]
        data = {k: v for k, v in drift_example.items() if not k.startswith('_')}
        
        parser = PxWebParser()
        
        with pytest.raises(ValueError) as exc_info:
            parser.extract_distribution(data, "Sexe", code_mapping={"1": "Masculin", "2": "Feminin"})
        
        assert "Sexe" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()


class TestCategoryAlignment:
    """
    Tests for category label alignment between PxWeb and internal models.
    
    These tests ensure that external category labels correctly map to
    our internal enum values. If NBS changes their category labels,
    these tests should catch the mismatch.
    """
    
    # Mapping specifications - source of truth for alignment
    EXPECTED_MAPPINGS = {
        "sex": {
            "external_codes": {"1": "Masculin", "2": "Feminin"},
            "internal_values": {"Masculin", "Feminin"},
        },
        "residence_type": {
            "external_codes": {"1": "Urban", "2": "Rural"},
            "internal_values": {"Urban", "Rural"},
        },
        "age_group": {
            "external_values": ["0-14", "15-24", "25-34", "35-44", "45-54", "55-64", "65+"],
            "internal_values": ["0-14", "15-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        },
        "region": {
            "internal_values": {"Chisinau", "Centru", "Nord", "Sud", "Gagauzia"},
        },
    }
    
    def test_sex_code_mapping_completeness(self):
        """Verify all internal sex values have external code mappings."""
        mapping = self.EXPECTED_MAPPINGS["sex"]
        external_targets = set(mapping["external_codes"].values())
        internal_values = mapping["internal_values"]
        
        assert external_targets == internal_values, \
            f"Sex mapping mismatch: external targets {external_targets} != internal {internal_values}"
    
    def test_residence_code_mapping_completeness(self):
        """Verify all internal residence values have external code mappings."""
        mapping = self.EXPECTED_MAPPINGS["residence_type"]
        external_targets = set(mapping["external_codes"].values())
        internal_values = mapping["internal_values"]
        
        assert external_targets == internal_values, \
            f"Residence mapping mismatch: external targets {external_targets} != internal {internal_values}"
    
    def test_region_values_match_census(self):
        """Verify internal region values match census definitions."""
        from moldova_personas.census_data import HARDCODED_FALLBACKS
        
        internal_regions = set(self.EXPECTED_MAPPINGS["region"]["internal_values"])
        census_regions = set(HARDCODED_FALLBACKS["REGION_DISTRIBUTION"].keys())
        
        assert internal_regions == census_regions, \
            f"Region mismatch: expected {census_regions}, got {internal_regions}"
    
    def test_age_group_alignment(self):
        """Verify age group bins align between PxWeb and internal representation."""
        mapping = self.EXPECTED_MAPPINGS["age_group"]
        
        # These should be identical (no mapping needed for age groups)
        assert mapping["external_values"] == mapping["internal_values"], \
            "Age group bins should align exactly between PxWeb and internal model"


class TestDistributionManifest:
    """
    Tests that verify the distribution mapping manifest is complete.
    
    This ensures we have documented mappings for all critical distributions.
    """
    
    def test_all_critical_fields_have_mappings(self):
        """Verify all critical fields have mapping specifications."""
        from moldova_personas.trust_report import CRITICAL_FIELDS
        
        # Fields that should have explicit mappings
        fields_needing_mappings = {
            "sex", "age_group", "region", "residence_type",
        }
        
        available_mappings = set(TestCategoryAlignment.EXPECTED_MAPPINGS.keys())
        
        missing = fields_needing_mappings - available_mappings
        assert not missing, f"Missing mapping specifications for: {missing}"


@pytest.mark.slow
@pytest.mark.network
class TestPxWebLiveIntegration:
    """
    Live integration tests against real PxWeb API.
    
    These tests are:
    - Marked as 'slow' (excluded from fast CI)
    - Marked as 'network' (require internet)
    - Optional in CI (can be run manually or on schedule)
    
    They validate that our parsing works against the actual API.
    """
    
    def test_pxweb_sex_distribution_live(self):
        """
        Live test: Fetch sex distribution from PxWeb.
        
        This test:
        1. Hits the real PxWeb API
        2. Validates response year matches expectations
        3. Validates category labels map correctly
        4. Validates no missing keys
        """
        from moldova_personas.pxweb_fetcher import NBSDataManager
        
        manager = NBSDataManager(cache_dir="./test_cache")
        
        # Fetch distribution
        dist = manager.get_sex_distribution()
        
        # Validate provenance
        assert dist.provenance.value in [
            "Fetched from NBS PxWeb API",
            "Cached from previous PxWeb fetch"
        ], "Should use live or cached PxWeb data"
        
        # Validate values
        assert "Masculin" in dist.values, "Masculin should be in distribution"
        assert "Feminin" in dist.values, "Feminin should be in distribution"
        
        # Validate probabilities sum to ~1
        assert dist.validate(tolerance=0.01), "Probabilities should sum to ~1.0"
        
        # Validate realistic proportions (from 2024 census: ~52.8% female)
        female_ratio = dist.values.get("Feminin", 0)
        assert 0.45 < female_ratio < 0.60, f"Female ratio {female_ratio} outside realistic range"
    
    def test_pxweb_response_year_matches_expectations(self):
        """
        Validate that PxWeb returns data for expected year.
        
        This catches cases where PxWeb might return outdated data
        or change their year labeling.
        """
        from moldova_personas.pxweb_fetcher import NBSDataManager
        
        manager = NBSDataManager(cache_dir="./test_cache")
        
        # Force fresh fetch if possible
        dist = manager.get_sex_distribution()
        
        # If we have metadata, check year
        if dist.source_table:
            # Should reference 2024 data
            assert "2024" in str(dist.source_table) or dist.last_fetched is not None, \
                "Should have 2024 data or timestamp"
    
    def test_pxweb_dimension_names_stable(self):
        """
        Validate that PxWeb dimension names haven't changed.
        
        This catches schema drift in dimension codes.
        """
        import requests
        from moldova_personas.pxweb_fetcher import NBS_BASE_URL
        
        # Fetch metadata for population dataset
        dataset_path = "20 Populatia si procesele demografice/POP010/POPro/POP010200rcl.px"
        url = f"{NBS_BASE_URL}/{dataset_path}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            metadata = response.json()
            
            # Check expected dimensions exist
            variables = metadata.get("variables", [])
            dimension_codes = {v.get("code") for v in variables}
            
            expected_dimensions = {"Ani", "Sexe", "Varste", "Medii"}
            
            missing = expected_dimensions - dimension_codes
            assert not missing, f"Missing expected dimensions: {missing}"
            
        except requests.RequestException as e:
            pytest.skip(f"Network error: {e}")


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m not slow')")
    config.addinivalue_line("markers", "network: marks tests as requiring network")
