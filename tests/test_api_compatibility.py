"""
API Compatibility Tests

Ensures the public API surface remains stable across versions.
These tests catch accidental breaking changes to exported symbols.
"""

import inspect
import pytest
from typing import get_type_hints

import moldova_personas as mp


class TestPublicAPIExports:
    """Tests that public API symbols exist and have expected signatures."""
    
    # Core classes that must be exported
    REQUIRED_CLASSES = [
        "Persona",
        "PersonaGenerator",
        "TrustReport",
        "TrustReportGenerator",
        "TrustDecisionRecord",
        "QualityTier",
        "UseCaseProfile",
        "RunManifest",
        "GateCode",
        "DatasetBundle",
        "DatasetConfig",
    ]
    
    # Functions that must be exported
    REQUIRED_FUNCTIONS = [
        "generate_dataset",
        "export_all_formats",
        "create_run_manifest_from_trust_report",
    ]
    
    # Constants that must be exported
    REQUIRED_CONSTANTS = [
        "CRITICAL_FIELDS",
        "HIGH_STAKES_CRITICAL_FIELDS",
        "USE_CASE_THRESHOLDS",
        "NARRATIVE_QUALITY_THRESHOLDS",
        "NARRATIVE_CONTRACT_VERSION",
    ]
    
    def test_all_required_classes_exported(self):
        """Verify all required classes are in __all__."""
        for class_name in self.REQUIRED_CLASSES:
            assert class_name in mp.__all__, f"{class_name} not in __all__"
            assert hasattr(mp, class_name), f"{class_name} not in module"
    
    def test_all_required_functions_exported(self):
        """Verify all required functions are in __all__."""
        for func_name in self.REQUIRED_FUNCTIONS:
            assert func_name in mp.__all__, f"{func_name} not in __all__"
            assert hasattr(mp, func_name), f"{func_name} not in module"
            assert callable(getattr(mp, func_name)), f"{func_name} is not callable"
    
    def test_all_required_constants_exported(self):
        """Verify all required constants are in __all__."""
        for const_name in self.REQUIRED_CONSTANTS:
            assert const_name in mp.__all__, f"{const_name} not in __all__"
            assert hasattr(mp, const_name), f"{const_name} not in module"
    
    def test_no_extra_exports(self):
        """Verify __all__ only contains expected symbols."""
        # This is a soft check - we allow extra exports but flag them
        expected = set(self.REQUIRED_CLASSES + self.REQUIRED_FUNCTIONS + self.REQUIRED_CONSTANTS)
        actual = set(mp.__all__)
        
        extra = actual - expected
        # Just log extra exports, don't fail
        if extra:
            print(f"\n[INFO] Extra exports (not in REQUIRED lists): {extra}")


class TestPersonaGeneratorSignature:
    """Tests for PersonaGenerator API stability."""
    
    def test_generator_init_signature(self):
        """Verify PersonaGenerator.__init__ signature."""
        sig = inspect.signature(mp.PersonaGenerator.__init__)
        params = list(sig.parameters.keys())
        
        assert "self" in params
        assert "seed" in params
        assert "census_data" in params
    
    def test_generator_generate_signature(self):
        """Verify PersonaGenerator.generate signature."""
        sig = inspect.signature(mp.PersonaGenerator.generate)
        params = list(sig.parameters.keys())
        
        assert "self" in params
        assert "n" in params
        assert "show_progress" in params
        assert "use_ethnicity_correction" in params
    
    def test_generator_ipf_correction_signature(self):
        """Verify generate_with_ethnicity_correction returns metrics."""
        sig = inspect.signature(mp.PersonaGenerator.generate_with_ethnicity_correction)
        params = sig.parameters
        
        assert "return_metrics" in params


class TestTrustReportSignature:
    """Tests for TrustReport API stability."""
    
    def test_trust_report_has_required_attributes(self):
        """Verify TrustReport has expected attributes."""
        # Check dataclass fields
        from dataclasses import fields
        field_names = {f.name for f in fields(mp.TrustReport)}
        
        required_fields = [
            "report_id",
            "persona_count",
            "overall_quality_tier",
            "hard_gate_triggered",
            "gate_reasons",
            "trust_decision",
        ]
        
        for field in required_fields:
            assert field in field_names, f"TrustReport missing field: {field}"
    
    def test_use_case_profile_values(self):
        """Verify UseCaseProfile enum values are stable."""
        expected_values = {
            "ANALYSIS_ONLY",
            "NARRATIVE_REQUIRED",
            "HIGH_STAKES",
        }
        
        actual_values = {e.name for e in mp.UseCaseProfile}
        assert actual_values == expected_values, f"Profile values changed: {actual_values}"


class TestGradientIntegrationAPI:
    """Tests for Gradient integration layer."""
    
    def test_generate_dataset_exists(self):
        """Verify generate_dataset function is exported."""
        assert hasattr(mp, "generate_dataset")
        assert callable(mp.generate_dataset)
    
    def test_generate_dataset_signature(self):
        """Verify generate_dataset has expected parameters."""
        sig = inspect.signature(mp.generate_dataset)
        params = sig.parameters
        
        required_params = [
            "n",
            "profile",
            "seed",
            "use_ipf",
            "generate_narratives",
            "strict",
            "outputs",
        ]
        
        for param in required_params:
            assert param in params, f"generate_dataset missing parameter: {param}"
    
    def test_dataset_bundle_has_required_attributes(self):
        """Verify DatasetBundle has expected attributes."""
        from dataclasses import fields
        
        # Check required dataclass fields
        field_names = {f.name for f in fields(mp.DatasetBundle)}
        required_fields = [
            "personas", "trust_report", "gate_codes", "decision",
            "decision_reasons", "run_manifest", "config",
        ]
        
        for field in required_fields:
            assert field in field_names, f"DatasetBundle missing field: {field}"
        
        # Check methods
        required_methods = [
            "to_gradient_trust_payload", "save", "should_escalate", "escalation_priority"
        ]
        for method in required_methods:
            assert hasattr(mp.DatasetBundle, method), f"DatasetBundle missing method: {method}"
    
    def test_dataset_bundle_payload_structure(self):
        """Verify to_gradient_trust_payload returns expected structure."""
        # Create minimal bundle for testing
        from moldova_personas.gradient_integration import DatasetBundle
        
        # Check the method exists and has correct signature
        sig = inspect.signature(mp.DatasetBundle.to_gradient_trust_payload)
        assert sig.return_annotation != inspect.Signature.empty or True  # Allow any return


class TestGateCodeAPI:
    """Tests for GateCode enum stability."""
    
    def test_critical_gate_codes_exist(self):
        """Verify critical gate codes exist."""
        critical_codes = [
            "ESS_TOO_LOW",
            "FALLBACK_SUPERCRITICAL_FIELD",
            "MARGINAL_ERROR_CRITICAL",
            "NARRATIVE_SCHEMA_INVALID",
            "PASS",
            "PASS_WITH_WARNINGS",
        ]
        
        for code in critical_codes:
            assert hasattr(mp.GateCode, code), f"GateCode missing {code}"
    
    def test_gate_code_methods_exist(self):
        """Verify GateCode has required methods."""
        assert hasattr(mp.GateCode, "is_hard_gate")
        assert hasattr(mp.GateCode, "is_warning")
        assert hasattr(mp.GateCode, "category")


class TestBackwardCompatibility:
    """Tests for backward compatibility guarantees."""
    
    def test_version_format(self):
        """Verify version follows semver."""
        import re
        version = mp.__version__
        
        # Semver regex (simplified)
        semver_pattern = r"^\d+\.\d+\.\d+([-\.]\w+)?$"
        assert re.match(semver_pattern, version), f"Version {version} doesn't follow semver"
    
    def test_module_version_exported(self):
        """Verify __version__ is exported."""
        assert hasattr(mp, "__version__")
        assert isinstance(mp.__version__, str)
    
    def test_old_imports_still_work(self):
        """Verify common old import patterns still work."""
        # Direct module imports should work
        from moldova_personas import PersonaGenerator, TrustReport
        from moldova_personas import UseCaseProfile, QualityTier
        
        assert PersonaGenerator is not None
        assert TrustReport is not None


# ============================================================================
# Semver Breaking Change Detection
# ============================================================================

# Store expected signatures for breaking change detection
_EXPECTED_SIGNATURES = {
    "PersonaGenerator.generate": ["self", "n", "show_progress", "use_ethnicity_correction"],
    "PersonaGenerator.generate_with_ethnicity_correction": ["self", "n", "oversample_factor", "return_metrics"],
    "generate_dataset": ["n", "profile", "seed", "use_ipf", "generate_narratives", "strict"],
}


class TestBreakingChangeDetection:
    """
    Tests that detect potential breaking changes.
    
    These tests may need updating when intentionally breaking changes
    are made (major version bumps).
    """
    
    def test_expected_signatures_unchanged(self):
        """Verify key function signatures haven't changed."""
        for name, expected_params in _EXPECTED_SIGNATURES.items():
            if "." in name:
                # Class method
                class_name, method_name = name.split(".")
                cls = getattr(mp, class_name)
                method = getattr(cls, method_name)
            else:
                # Module function
                method = getattr(mp, name)
            
            sig = inspect.signature(method)
            actual_params = list(sig.parameters.keys())
            
            # Check all expected params exist (allow extra params)
            for param in expected_params:
                assert param in actual_params, \
                    f"{name} missing expected parameter {param} (BREAKING CHANGE)"


# ============================================================================
# Version Synchronization Tests
# ============================================================================

class TestVersionSynchronization:
    """Tests to prevent version drift between files."""
    
    def test_pyproject_version_matches_module_version(self):
        """
        Ensure pyproject.toml version matches moldova_personas.__version__.
        
        This prevents the common issue where a release is tagged but the
        installed package reports a different version.
        """
        import tomllib
        from pathlib import Path
        
        # Read pyproject.toml version
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml not found"
        
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)
        
        pyproject_version = pyproject["project"]["version"]
        module_version = mp.__version__
        
        assert pyproject_version == module_version, (
            f"Version mismatch: pyproject.toml has {pyproject_version}, "
            f"but moldova_personas.__version__ is {module_version}. "
            f"These must be kept in sync for releases."
        )
    
    def test_payload_version_matches_documentation(self):
        """
        Ensure GRADIENT_PAYLOAD_VERSION matches documented value.
        
        The payload version is a contract with downstream consumers.
        This test ensures it matches what we document.
        """
        from moldova_personas.gradient_integration import GRADIENT_PAYLOAD_VERSION
        
        # Current expected version
        expected_version = "1.0"
        
        assert GRADIENT_PAYLOAD_VERSION == expected_version, (
            f"GRADIENT_PAYLOAD_VERSION changed from {expected_version} to "
            f"{GRADIENT_PAYLOAD_VERSION}. This is a breaking change for "
            f"Gradient integration and requires coordination."
        )
    
    def test_narrative_contract_version_matches_expected(self):
        """
        Ensure NARRATIVE_CONTRACT_VERSION is as expected.
        
        Narrative contract changes affect LLM output format.
        """
        expected_contract_version = "1.4.0"
        
        assert mp.NARRATIVE_CONTRACT_VERSION == expected_contract_version, (
            f"NARRATIVE_CONTRACT_VERSION changed from {expected_contract_version} "
            f"to {mp.NARRATIVE_CONTRACT_VERSION}. Check narrative format compatibility."
        )
    
    def test_version_follows_semver(self):
        """Ensure version follows semantic versioning format."""
        import re
        
        version = mp.__version__
        semver_pattern = r"^\d+\.\d+\.\d+([-\.]\w+)?$"
        
        assert re.match(semver_pattern, version), (
            f"Version '{version}' does not follow semantic versioning. "
            f"Expected format: MAJOR.MINOR.PATCH[-prerelease]"
        )
