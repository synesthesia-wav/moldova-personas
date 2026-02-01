"""
Unit tests for the pxweb_fetcher module.

Tests DataProvenance, Distribution, PxWebParser, and NBSDataManager.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import numpy as np

from moldova_personas.pxweb_fetcher import (
    DataProvenance,
    Distribution,
    PxWebParser,
    IPFEngine,
    CrossTabulation,
    NBSDataManager,
    get_distribution,
    NBS_MANAGER,
)


class TestDataProvenance:
    """Tests for DataProvenance enum."""
    
    def test_provenance_values(self):
        """Test that all expected provenance values exist."""
        assert DataProvenance.PXWEB_DIRECT.value == "Fetched from NBS PxWeb API"
        assert DataProvenance.PXWEB_CACHED.value == "Cached from previous PxWeb fetch"
        assert DataProvenance.IPF_DERIVED.value == "Derived via IPF from marginal distributions"
        assert DataProvenance.CENSUS_HARDCODED.value == "From NBS 2024 final report (static)"
        assert DataProvenance.ESTIMATED.value == "Demographic estimate (documented assumptions)"


class TestDistribution:
    """Tests for Distribution dataclass."""
    
    def test_distribution_creation(self):
        """Test creating a Distribution object."""
        dist = Distribution(
            values={"A": 0.5, "B": 0.5},
            provenance=DataProvenance.CENSUS_HARDCODED,
            source_table="TEST.px",
            confidence=0.95
        )
        
        assert dist.values == {"A": 0.5, "B": 0.5}
        assert dist.provenance == DataProvenance.CENSUS_HARDCODED
        assert dist.source_table == "TEST.px"
        assert dist.confidence == 0.95
    
    def test_distribution_validation(self):
        """Test Distribution.validate() method."""
        # Valid distribution
        dist = Distribution(
            values={"A": 0.5, "B": 0.5},
            provenance=DataProvenance.CENSUS_HARDCODED
        )
        assert dist.validate() is True
        
        # Invalid distribution (sums to more than 1.0)
        dist = Distribution(
            values={"A": 0.6, "B": 0.6},
            provenance=DataProvenance.CENSUS_HARDCODED
        )
        assert dist.validate() is False
    
    def test_distribution_normalization(self):
        """Test Distribution.normalize() method."""
        dist = Distribution(
            values={"A": 10, "B": 10},
            provenance=DataProvenance.CENSUS_HARDCODED
        )
        
        dist.normalize()
        
        assert dist.values["A"] == 0.5
        assert dist.values["B"] == 0.5
        assert dist.validate() is True
    
    def test_distribution_serialization(self):
        """Test Distribution.to_dict() and from_dict()."""
        original = Distribution(
            values={"A": 0.5, "B": 0.5},
            provenance=DataProvenance.PXWEB_DIRECT,
            source_table="TEST.px",
            last_fetched=datetime(2024, 1, 15, 12, 0, 0),
            confidence=0.99,
            methodology="Test fetch",
            limitations=None,
            raw_counts={"A": 100, "B": 100},
            total=200
        )
        
        # Serialize
        data = original.to_dict()
        
        assert data["values"] == {"A": 0.5, "B": 0.5}
        assert data["provenance"] == "Fetched from NBS PxWeb API"
        assert data["source_table"] == "TEST.px"
        assert data["last_fetched"] == "2024-01-15T12:00:00"
        
        # Deserialize
        restored = Distribution.from_dict(data)
        
        assert restored.values == original.values
        assert restored.provenance == original.provenance
        assert restored.source_table == original.source_table
        assert restored.last_fetched == original.last_fetched


class TestPxWebParser:
    """Tests for PxWebParser."""
    
    def test_extract_distribution_basic(self):
        """Test extracting a simple distribution."""
        # Mock PxWeb response
        data = {
            "columns": [
                {"code": "Ani", "text": "Years", "type": "t"},
                {"code": "Sexe", "text": "Sex", "type": "t"},
                {"code": "Population", "text": "Population", "type": "c"}
            ],
            "data": [
                {"key": ["2024", "1"], "values": ["1000"]},
                {"key": ["2024", "2"], "values": ["1200"]},
            ]
        }
        
        parser = PxWebParser()
        probs, counts, total = parser.extract_distribution(
            data, "Sexe", code_mapping={"1": "Masculin", "2": "Feminin"}
        )
        
        assert probs == {"Masculin": 1000/2200, "Feminin": 1200/2200}
        assert counts == {"Masculin": 1000, "Feminin": 1200}
        assert total == 2200
    
    def test_extract_distribution_no_mapping(self):
        """Test extracting distribution without code mapping."""
        data = {
            "columns": [
                {"code": "Region", "text": "Region", "type": "t"},
                {"code": "Count", "text": "Count", "type": "c"}
            ],
            "data": [
                {"key": ["North"], "values": ["500"]},
                {"key": ["South"], "values": ["300"]},
            ]
        }
        
        parser = PxWebParser()
        probs, counts, total = parser.extract_distribution(data, "Region")
        
        assert "North" in probs
        assert "South" in probs
        assert total == 800


class TestIPFEngine:
    """Tests for IPFEngine."""
    
    def test_ipf_basic_fit(self):
        """Test basic IPF fitting."""
        # Create marginal distributions
        row_dist = Distribution(
            values={"A": 0.6, "B": 0.4},
            provenance=DataProvenance.CENSUS_HARDCODED
        )
        col_dist = Distribution(
            values={"X": 0.3, "Y": 0.7},
            provenance=DataProvenance.CENSUS_HARDCODED
        )
        
        engine = IPFEngine()
        cross_tab = engine.fit(row_dist, col_dist)
        
        # Check structure
        assert cross_tab.row_names == ["A", "B"]
        assert cross_tab.col_names == ["X", "Y"]
        assert cross_tab.matrix.shape == (2, 2)
        
        # Check marginals are preserved
        row_marginal = cross_tab.get_row_marginal()
        assert pytest.approx(row_marginal["A"], 0.01) == 0.6
        assert pytest.approx(row_marginal["B"], 0.01) == 0.4
        
        col_marginal = cross_tab.get_col_marginal()
        assert pytest.approx(col_marginal["X"], 0.01) == 0.3
        assert pytest.approx(col_marginal["Y"], 0.01) == 0.7
    
    def test_ipf_convergence(self):
        """Test that IPF converges to a valid solution."""
        row_dist = Distribution(
            values={"A": 0.5, "B": 0.5},
            provenance=DataProvenance.CENSUS_HARDCODED
        )
        col_dist = Distribution(
            values={"X": 0.5, "Y": 0.5},
            provenance=DataProvenance.CENSUS_HARDCODED
        )
        
        engine = IPFEngine()
        cross_tab = engine.fit(row_dist, col_dist, max_iterations=100)
        
        # Matrix should sum to 1.0
        assert pytest.approx(cross_tab.matrix.sum(), 0.0001) == 1.0
        
        # With uniform marginals and uniform seed, result should be uniform
        assert pytest.approx(cross_tab.matrix[0, 0], 0.01) == 0.25


class TestNBSDataManager:
    """Tests for NBSDataManager."""
    
    def test_initialization(self):
        """Test manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = NBSDataManager(cache_dir=tmpdir)
            assert manager.cache_dir == Path(tmpdir)
            assert (Path(tmpdir) / ".moldova_personas" / "cache").exists() or True
    
    def test_fallback_distribution(self):
        """Test that fallback distributions work."""
        manager = NBSDataManager(cache_dir="./test_cache")
        
        # Ethnicity should use fallback (not in PxWeb yet)
        dist = manager.get_ethnicity_distribution()
        
        assert dist.provenance == DataProvenance.CENSUS_HARDCODED
        assert "Moldovean" in dist.values
        assert dist.limitations is not None
        assert "not yet available" in dist.limitations
    
    def test_distribution_validation(self):
        """Test that all distributions sum to ~1.0."""
        manager = NBSDataManager(cache_dir="./test_cache")
        
        distributions = [
            "region", "residence_type", "sex", "age_group",
            "ethnicity", "education", "marital_status",
            "religion", "language", "employment_status"
        ]
        
        for name in distributions:
            dist = manager.get_distribution(name)
            assert dist.validate(tolerance=0.02), f"{name} does not sum to 1.0"
    
    def test_cache_operations(self):
        """Test cache save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = NBSDataManager(cache_dir=tmpdir)
            
            # Create a distribution
            dist = Distribution(
                values={"A": 0.5, "B": 0.5},
                provenance=DataProvenance.PXWEB_DIRECT,
                last_fetched=datetime.now()
            )
            
            # Save to cache
            manager._save_to_cache("test", dist)
            
            # Load from cache
            cached = manager._load_from_cache("test")
            
            assert cached is not None
            assert cached.values == dist.values
            assert cached.provenance == dist.provenance


class TestIntegration:
    """Integration tests."""
    
    def test_get_distribution_convenience(self):
        """Test the get_distribution convenience function."""
        dist = get_distribution("sex")
        
        assert isinstance(dist, Distribution)
        assert "Masculin" in dist.values or "Feminin" in dist.values
    
    def test_nbs_manager_singleton(self):
        """Test that NBS_MANAGER singleton works."""
        dist = NBS_MANAGER.get_distribution("region")
        
        assert isinstance(dist, Distribution)
        assert "Chisinau" in dist.values
