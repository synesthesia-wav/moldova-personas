"""Tests for name generation with frequency weighting."""

import pytest
from collections import Counter

from moldova_personas.ethnocultural_tables import get_language_distribution

from moldova_personas.names import (
    generate_name,
    get_language_by_ethnicity,
    get_religion_by_ethnicity,
    weighted_choice,
    FIRST_NAMES_WEIGHTED,
    LAST_NAMES_WEIGHTED,
)


class TestWeightedChoice:
    """Tests for weighted choice function."""
    
    def test_weighted_choice_basic(self):
        """Test weighted choice returns valid item."""
        choices = [("A", 0.7), ("B", 0.2), ("C", 0.1)]
        result = weighted_choice(choices)
        assert result in ["A", "B", "C"]
    
    def test_weighted_choice_distribution(self):
        """Test that weights affect distribution."""
        choices = [("Common", 0.9), ("Rare", 0.1)]
        results = [weighted_choice(choices) for _ in range(1000)]
        counts = Counter(results)
        
        # Common should appear ~9x more often
        assert counts["Common"] > counts["Rare"] * 5
    
    def test_weighted_choice_single_option(self):
        """Test with only one option."""
        choices = [("Only", 1.0)]
        result = weighted_choice(choices)
        assert result == "Only"


class TestNameGeneration:
    """Tests for name generation."""
    
    def test_generate_name_basic(self):
        """Test basic name generation."""
        name = generate_name("Moldovean", "Masculin")
        assert " " in name
        parts = name.split()
        assert len(parts) >= 2
    
    def test_generate_name_different_ethnicities(self):
        """Test name generation for different ethnicities."""
        ethnicities = ["Moldovean", "Român", "Ucrainean", "Găgăuz", "Rus", "Bulgar", "Rrom", "Altele"]
        
        for ethnicity in ethnicities:
            name = generate_name(ethnicity, "Feminin")
            assert " " in name
            assert len(name) > 3
    
    def test_generate_name_frequency_distribution(self):
        """Test that name frequency weighting works."""
        # Generate many Moldovan male names
        names = [generate_name("Moldovean", "Masculin") for _ in range(500)]
        first_names = [n.split()[0] for n in names]
        counts = Counter(first_names)
        
        # Most common names should be Ion, Vasile, Andrei
        top_names = [name for name, _ in counts.most_common(5)]
        assert "Ion" in top_names or "Vasile" in top_names or "Andrei" in top_names
    
    def test_generate_name_no_western_names_in_altele(self):
        """Test that 'Altele' ethnicity doesn't have generic Western names."""
        names = [generate_name("Altele", "Masculin") for _ in range(50)]
        first_names = [n.split()[0] for n in names]
        
        # Should NOT have Smith, Johnson, Williams, etc.
        western_names = ["Smith", "Johnson", "Williams", "Brown", "Jones"]
        for name in first_names:
            assert name not in western_names, f"Found Western name {name} in Altele"
    
    def test_generate_name_roma_not_hungarian(self):
        """Test that Roma names are Moldovan-Roma, not Hungarian."""
        names = [generate_name("Rrom", "Masculin") for _ in range(50)]
        first_names = [n.split()[0] for n in names]
        
        # Should NOT have Hungarian-Roma names
        hungarian_names = ["Lakatos", "Kovacs", "Horvath", "Szabo"]
        for name in first_names:
            assert name not in hungarian_names, f"Found Hungarian name {name} in Roma"
        
        # Should have Moldovan-Roma names
        moldovan_roma_names = ["Cioabă", "Marian", "Mănăilă"]
        found = [n for n in first_names if n in moldovan_roma_names]
        assert len(found) > 0, "No Moldovan-Roma names found"


class TestLanguageByEthnicity:
    """Tests for language mapping."""

    def _assert_primary_language(
        self,
        ethnicity: str,
        fallback_language: str,
        fallback_min_count: int,
        sample_size: int = 200,
        tolerance: float = 0.15,
    ):
        """Assert primary language share using official tables when available."""
        official = get_language_distribution(ethnicity)
        sample = [get_language_by_ethnicity(ethnicity) for _ in range(sample_size)]
        if official:
            primary_lang, primary_share = max(official.items(), key=lambda item: item[1])
            min_share = max(0.0, primary_share - tolerance)
            count = sample.count(primary_lang)
            assert count >= int(sample_size * min_share), (
                f"Only {count}/{sample_size} {ethnicity} speak {primary_lang} "
                f"(expected ≥ {min_share:.0%} from official tables)"
            )
        else:
            count = sample.count(fallback_language)
            assert count >= fallback_min_count, (
                f"Only {count}/{sample_size} {ethnicity} speak {fallback_language}"
            )
    
    def test_moldovan_romanian_language(self):
        """Test Moldovans and Romanians predominantly speak Romanian."""
        self._assert_primary_language("Moldovean", "Română", 85, sample_size=200)
        self._assert_primary_language("Român", "Română", 190, sample_size=200, tolerance=0.10)
    
    def test_specific_languages(self):
        """Test specific ethnicities have correct primary languages."""
        self._assert_primary_language("Ucrainean", "Ucraineană", 70, sample_size=200)
        self._assert_primary_language("Găgăuz", "Găgăuză", 75, sample_size=200)
        self._assert_primary_language("Rus", "Rusă", 170, sample_size=200, tolerance=0.10)
        self._assert_primary_language("Bulgar", "Bulgară", 70, sample_size=200)


class TestReligionByEthnicity:
    """Tests for religion mapping with proper correlations."""
    
    def test_moldovan_mostly_orthodox(self):
        """Test Moldovans are mostly Orthodox."""
        religions = [get_religion_by_ethnicity("Moldovean") for _ in range(100)]
        orthodox_count = religions.count("Ortodox")
        # Should be ~96% Orthodox
        assert orthodox_count >= 90, f"Only {orthodox_count}% Orthodox, expected >90%"
    
    def test_gagauz_mostly_orthodox(self):
        """Test Gagauz are mostly Orthodox in official tables."""
        religions = [get_religion_by_ethnicity("Găgăuz") for _ in range(100)]
        orthodox_count = religions.count("Ortodox")
        # Should be strongly Orthodox
        assert orthodox_count > 85, f"Only {orthodox_count}% Orthodox, expected dominant"
    
    def test_ukrainian_has_non_orthodox_minorities(self):
        """Test Ukrainians include some non-Orthodox minorities."""
        religions = [get_religion_by_ethnicity("Ucrainean") for _ in range(100)]
        non_orthodox = [r for r in religions if r != "Ortodox"]
        
        # Should have some non-Orthodox
        assert len(non_orthodox) > 0, "No non-Orthodox Ukrainians found"
    
    def test_altele_diverse_religions(self):
        """Test 'Altele' has diverse religions including 'Altă religie'."""
        religions = [get_religion_by_ethnicity("Altele") for _ in range(100)]
        
        has_other = "Altă religie" in religions
        has_mainstream = any(r in religions for r in ["Ortodox", "Catolic", "Baptist"])
        
        assert has_other, "No 'Altă religie' found in Altele"
        assert has_mainstream, "No mainstream religions found in Altele"


class TestNameDataStructures:
    """Tests for name data structure validity."""
    
    def test_all_ethnicities_have_names(self):
        """Test all ethnicities have first and last names."""
        ethnicities = FIRST_NAMES_WEIGHTED.keys()
        
        for ethnicity in ethnicities:
            assert ethnicity in LAST_NAMES_WEIGHTED, f"{ethnicity} missing last names"
            assert "Feminin" in FIRST_NAMES_WEIGHTED[ethnicity]
            assert "Masculin" in FIRST_NAMES_WEIGHTED[ethnicity]
    
    def test_weights_sum_to_approx_one(self):
        """Test that weights roughly sum to 1.0."""
        for ethnicity, sexes in FIRST_NAMES_WEIGHTED.items():
            for sex, names in sexes.items():
                total_weight = sum(w for _, w in names)
                # Allow some tolerance (0.8 - 1.2)
                assert 0.8 <= total_weight <= 1.2, \
                    f"{ethnicity} {sex} weights sum to {total_weight}"
    
    def test_no_duplicate_names(self):
        """Test that there are no duplicate names within each category."""
        for ethnicity, sexes in FIRST_NAMES_WEIGHTED.items():
            for sex, names in sexes.items():
                name_list = [n for n, _ in names]
                assert len(name_list) == len(set(name_list)), \
                    f"Duplicates found in {ethnicity} {sex}"
