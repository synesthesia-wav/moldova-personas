"""
Gradient Integration Layer

Unified entrypoint for embedding Moldova Personas Generator into Gradient.
Provides a single function that orchestrates the entire pipeline and returns
a complete artifact bundle.

Example:
    from moldova_personas.gradient_integration import generate_dataset, UseCaseProfile
    
    bundle = generate_dataset(
        n=10000,
        profile=UseCaseProfile.HIGH_STAKES,
        seed=42,
        outputs=["parquet", "trust_report", "run_manifest"]
    )
    
    if bundle.decision == "PASS":
        bundle.save("./output")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Literal

from .generator import PersonaGenerator
from .trust_report import (
    TrustReport,
    TrustReportGenerator,
    TrustDecisionRecord,
    UseCaseProfile,
    QualityTier,
    IPFMetrics,
    USE_CASE_THRESHOLDS,
)
from .run_manifest import RunManifest, create_run_manifest_from_trust_report
from .gate_codes import GateCode, map_reason_to_code, sort_gate_codes
from .models import Persona, PopulationMode
from .exporters import export_all_formats
from .census_data import CENSUS
from .pxweb_fetcher import NBS_BASE_URL


logger = logging.getLogger(__name__)


# Fields excluded from HIGH_STAKES outputs due to estimated provenance.
HIGH_STAKES_OUTPUT_DROP_FIELDS = ["occupation", "occupation_sector"]


# Payload version for API compatibility
GRADIENT_PAYLOAD_VERSION = "1.0"


# Recommended actions for each gate code
# These provide actionable guidance for both automated and human consumers
GATE_CODE_RECOMMENDATIONS: Dict[GateCode, str] = {
    # Data Quality
    GateCode.MARGINAL_ERROR_HIGH: "Review census alignment; consider regenerating with fresh data",
    GateCode.MARGINAL_ERROR_CRITICAL: "Regenerate with updated census data or reduce sample size",
    GateCode.FALLBACK_CRITICAL_FIELD: "Refresh PxWeb cache to get live data for critical fields",
    GateCode.FALLBACK_SUPERCRITICAL_FIELD: "CRITICAL: Refresh PxWeb cache immediately; super-critical fields must use live data",
    GateCode.TARGET_FALLBACK_SUPERCRITICAL: "CRITICAL: Super-critical targets missing; refresh PxWeb data or block run",
    GateCode.PXWEB_CACHE_STALE: "Refresh PxWeb cache and re-run generation",
    GateCode.PXWEB_FETCH_FAILED: "Check network connectivity; verify PxWeb API availability",
    GateCode.ETHNOCULTURAL_FALLBACK_USED: "Review ethnocultural tables; missing ethnicity rows defaulted to national totals",
    GateCode.ESTIMATED_FIELD_USED: "Estimated/heuristic fields present; disable strict mode or supply official distributions",
    
    # IPF / Statistical
    GateCode.ESS_TOO_LOW: "Increase oversample factor (e.g., 3→5) or reduce IPF constraints",
    GateCode.WEIGHT_CONCENTRATION_HIGH: "Review ethnicity distribution; consider stratified sampling",
    GateCode.IPF_DIVERGENCE: "Simplify correction targets or increase max_iterations",
    
    # Narrative Quality
    GateCode.NARRATIVE_SCHEMA_INVALID: "Check LLM provider status; verify prompt version compatibility",
    GateCode.NARRATIVE_PARSE_FAILURE: "Switch LLM provider or regenerate without narratives",
    GateCode.NARRATIVE_TOO_SHORT: "Increase max_tokens or switch to more capable model",
    GateCode.NARRATIVE_TOO_LONG: "Decrease max_tokens or add length constraints to prompt",
    GateCode.NARRATIVE_LANGUAGE_INVALID: "Verify LLM supports Romanian; check temperature setting",
    GateCode.NARRATIVE_PRONOUN_MISMATCH: "Regenerate narratives or use structured output mode",
    GateCode.NARRATIVE_MOCK_RATIO_HIGH: "Check LLM API rate limits; verify API key validity",
    GateCode.NARRATIVE_PII_DETECTED: "ROUTE TO HUMAN REVIEW: Do not export narratives; check LLM temperature",
    
    # Configuration
    GateCode.CONFIG_INVALID_SEED: "Provide valid integer seed or omit for random",
    GateCode.CONFIG_VERSION_MISMATCH: "Update generator or downgrade feature requirements",
    GateCode.CONFIG_INCOMPLETE: "Fill required configuration parameters",
    
    # System
    GateCode.SYSTEM_MEMORY_EXCEEDED: "Reduce batch size or increase available memory",
    GateCode.SYSTEM_TIMEOUT: "Reduce persona count or optimize generation parameters",
    GateCode.SYSTEM_CHECKPOINT_CORRUPT: "Remove corrupted checkpoint and restart from scratch",
    
    # Success
    GateCode.PASS: "No action required; proceed with export",
    GateCode.PASS_WITH_WARNINGS: "Review warnings; consider re-running with stricter settings",
}


def get_recommendation(code: GateCode) -> str:
    """Get recommended action for a gate code."""
    return GATE_CODE_RECOMMENDATIONS.get(code, "Review gate code documentation for guidance")


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    n: int
    profile: UseCaseProfile
    seed: Optional[int] = None
    use_ipf: bool = True
    oversample_factor: int = 3
    population_mode: PopulationMode = PopulationMode.ADULT_18
    generate_narratives: bool = False
    llm_provider: str = "mock"
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    allow_mock_narratives: Optional[bool] = None
    llm_rate_limit: Optional[float] = None
    strict: bool = False
    cache_dir: Optional[str] = None
    raise_on_reject: bool = False
    output_drop_fields: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive fields)."""
        return {
            "n": self.n,
            "profile": self.profile.value,
            "seed": self.seed,
            "use_ipf": self.use_ipf,
            "oversample_factor": self.oversample_factor,
            "population_mode": self.population_mode.value,
            "generate_narratives": self.generate_narratives,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "allow_mock_narratives": self.allow_mock_narratives,
            "llm_rate_limit": self.llm_rate_limit,
            "strict": self.strict,
            "raise_on_reject": self.raise_on_reject,
            "output_drop_fields": list(self.output_drop_fields),
        }


@dataclass
class DatasetBundle:
    """
    Complete artifact bundle from a generation run.
    
    This is the single return type from `generate_dataset()`, containing
    everything needed for downstream processing in Gradient.
    """
    
    # Core data (required)
    personas: List[Persona]
    """Generated persona objects"""
    
    trust_report: TrustReport
    """Complete trust report with metrics and gating"""
    
    gate_codes: List[GateCode]
    """Machine-readable gate condition codes"""
    
    decision: Literal["PASS", "PASS_WITH_WARNINGS", "REJECT"]
    """Final gate decision"""
    
    decision_reasons: List[str]
    """Human-readable reasons for decision"""
    
    run_manifest: RunManifest
    """Complete run manifest for reproducibility"""
    
    config: DatasetConfig
    """Configuration used for this run (excluded from repr for security)"""
    
    # Optional fields with defaults
    ipf_metrics: Optional[IPFMetrics] = None
    """IPF correction metrics"""
    
    generation_time_seconds: float = 0.0
    """Total generation time"""
    
    @property
    def quality_tier(self) -> QualityTier:
        """Get the overall quality tier."""
        return self.trust_report.overall_quality_tier
    
    @property
    def should_escalate(self) -> bool:
        """
        Determine if this bundle requires human escalation.
        
        Returns:
            True if any hard gates triggered or decision is REJECT
        """
        return self.decision == "REJECT" or any(
            code.is_hard_gate() for code in self.gate_codes
        )
    
    @property
    def escalation_priority(self) -> int:
        """
        Get escalation priority (1 = highest, 10 = lowest).
        
        Returns:
            Priority level for routing
        """
        from .gate_codes import get_escalation_priority
        return get_escalation_priority(self.gate_codes)
    
    def to_gradient_trust_payload(self) -> Dict[str, Any]:
        """
        Convert to Gradient 'computed trust' payload format.
        
        This provides the signal mapping for Gradient's trust system:
        - decision: pass/warn/reject (gating)
        - confidence: high/medium/low (explainability)
        - signals: provenance, ESS, cache age, etc. (computed trust inputs)
        
        Versioning:
            payload_version: Schema version of this payload (1.0)
            generator_version: Version of moldova-personas package
        
        Monotonicity Guarantee:
            - If decision == "REJECT", trust score must be capped at "low"
            - If targets_source == "hardcoded_fallback" and profile is HIGH_STAKES,
              confidence is capped at "medium" regardless of other signals
        """
        # Get base confidence from trust report
        base_confidence = self.trust_report.trust_decision.confidence if self.trust_report.trust_decision else "unknown"
        
        # Apply monotonicity constraints for computed trust
        # If rejected, cap confidence at low
        if self.decision == "REJECT":
            effective_confidence = "low"
        # If using fallback in HIGH_STAKES mode, cap at medium
        elif (self.config.profile == UseCaseProfile.HIGH_STAKES and 
              self.trust_report.targets_source == "hardcoded_fallback"):
            effective_confidence = "medium"
        else:
            effective_confidence = base_confidence
        
        # Sort gate codes deterministically
        sorted_codes = sort_gate_codes(self.gate_codes)
        
        # Build payload with versioning
        return {
            # Versioning (for migration/compatibility)
            "payload_version": GRADIENT_PAYLOAD_VERSION,
            "generator_version": "1.0.0",
            "schema_hash": self.run_manifest.config_hash,
            
            # Core decision
            "decision": self.decision,
            "quality_tier": self.quality_tier.value,
            
            # Confidence with monotonicity applied
            "confidence": effective_confidence,
            "base_confidence": base_confidence,  # Original before capping
            "confidence_factors": self.trust_report.trust_decision.confidence_factors if self.trust_report.trust_decision else [],
            "monotonicity_applied": effective_confidence != base_confidence,
            
            # Gate codes (deterministically sorted)
            "gate_codes": [code.value for code in sorted_codes],
            "gate_code_details": [
                {
                    "code": code.value,
                    "category": code.category(),
                    "is_hard_gate": code.is_hard_gate(),
                    "recommendation": get_recommendation(code),
                }
                for code in sorted_codes
            ],
            "hard_gates_triggered": self.should_escalate,
            "escalation_priority": self.escalation_priority,
            
            # Configuration (for debugging)
            "profile": self.config.profile.value,
            "strict_mode": self.config.strict,
            
            # Computed trust signals
            "signals": {
                "provenance_coverage_critical": self.trust_report.provenance_coverage_critical,
                "targets_source": self.trust_report.targets_source,
                "pxweb_cache_age_days": self.trust_report.pxweb_cache_age_days,
                "ess_ratio": self.ipf_metrics.ess_ratio if self.ipf_metrics else None,
                "information_loss": self.ipf_metrics.information_loss if self.ipf_metrics else None,
                "mean_l1_error": self.trust_report.mean_l1_error,
                "max_l1_error": self.trust_report.max_l1_error,
                "fallback_ratio_super_critical": self.trust_report.fallback_ratio_super_critical,
                "narrative_mock_ratio": self.trust_report.narrative_mock_ratio,
            },
            
            # Metadata for debugging
            "run_id": self.run_manifest.run_id,
            "config_hash": self.run_manifest.config_hash_short,
            "generation_time_seconds": self.generation_time_seconds,
            "timestamp": datetime.now().isoformat(),
        }
    
    def save(self, output_dir: str, formats: List[str] = None) -> Dict[str, str]:
        """
        Save bundle to output directory.
        
        Args:
            output_dir: Directory to save files
            formats: Output formats (default: ["parquet", "trust_report", "run_manifest"])
            
        Returns:
            Dictionary mapping format to file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        formats = formats or ["parquet", "trust_report", "run_manifest"]
        saved = {}
        
        if "parquet" in formats or "all" in formats:
            from .exporters import ParquetExporter
            parquet_path = output_path / f"personas_{self.run_manifest.run_id}.parquet"
            ParquetExporter().export(
                self.personas,
                str(parquet_path),
                drop_fields=self.config.output_drop_fields,
            )
            saved["parquet"] = str(parquet_path)
        
        if "json" in formats or "all" in formats:
            from .exporters import JSONExporter
            json_path = output_path / f"personas_{self.run_manifest.run_id}.json"
            JSONExporter().export_json(
                self.personas,
                str(json_path),
                drop_fields=self.config.output_drop_fields,
            )
            saved["json"] = str(json_path)
        
        if "trust_report" in formats or "all" in formats:
            trust_path = output_path / f"trust_report_{self.run_manifest.run_id}.json"
            with open(trust_path, 'w', encoding='utf-8') as f:
                f.write(self.trust_report.to_json())
            saved["trust_report"] = str(trust_path)
        
        if "run_manifest" in formats or "all" in formats:
            manifest_path = output_path / f"run_manifest_{self.run_manifest.run_id}.json"
            self.run_manifest.save(str(manifest_path))
            saved["run_manifest"] = str(manifest_path)
        
        if "gradient_payload" in formats:
            payload_path = output_path / f"gradient_trust_{self.run_manifest.run_id}.json"
            with open(payload_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(self.to_gradient_trust_payload(), f, indent=2)
            saved["gradient_payload"] = str(payload_path)
        
        return saved
    
    def summary(self) -> str:
        """Human-readable summary of the bundle."""
        lines = [
            f"Dataset Bundle: {self.run_manifest.run_id}",
            f"Personas: {len(self.personas):,}",
            f"Decision: {self.decision}",
            f"Quality Tier: {self.quality_tier.value}",
        ]
        
        if self.ipf_metrics:
            lines.append(f"ESS Ratio: {self.ipf_metrics.ess_ratio:.1%}")
        
        if self.gate_codes:
            lines.append(f"Gate Codes: {', '.join(c.value for c in self.gate_codes)}")
        
        if self.should_escalate:
            lines.append(f"⚠️  ESCALATION REQUIRED (Priority: {self.escalation_priority})")
        
        return "\n".join(lines)


def generate_dataset(
    n: int,
    profile: UseCaseProfile,
    seed: Optional[int] = None,
    use_ipf: bool = True,
    population_mode: PopulationMode = PopulationMode.ADULT_18,
    generate_narratives: bool = False,
    llm_provider: str = "mock",
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    allow_mock_narratives: Optional[bool] = None,
    llm_rate_limit: Optional[float] = None,
    strict: Optional[bool] = None,
    raise_on_reject: bool = False,
    outputs: List[str] = None,
    config_override: Dict[str, Any] = None,
) -> DatasetBundle:
    """
    Generate a complete persona dataset with trust validation.
    
    This is the single entrypoint for Gradient integration. It orchestrates:
    1. Structured generation (PGM)
    2. IPF correction (if enabled)
    3. Narrative enrichment (if enabled)
    4. Trust validation and gating
    5. Run manifest creation
    
    Args:
        n: Number of personas to generate
        profile: Use case profile (ANALYSIS_ONLY, NARRATIVE_REQUIRED, HIGH_STAKES)
        seed: Random seed for reproducibility
        use_ipf: Whether to apply IPF correction
        generate_narratives: Whether to generate LLM narratives
        llm_provider: LLM provider ("mock", "openai", "dashscope")
        llm_api_key: API key for LLM provider
        llm_model: Model name for LLM
        strict: Override strict mode (default: inferred from profile)
        outputs: Output formats to generate
        config_override: Additional configuration overrides
        
    Returns:
        DatasetBundle with personas, trust report, and metadata
        
    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If generation fails (strict mode)
    """
    import time
    
    start_time = time.perf_counter()
    
    # Determine strict mode from profile if not specified
    if strict is None:
        strict = profile == UseCaseProfile.HIGH_STAKES
    if allow_mock_narratives is None:
        allow_mock_narratives = not (generate_narratives and profile != UseCaseProfile.ANALYSIS_ONLY)
    
    # Create configuration
    config = DatasetConfig(
        n=n,
        profile=profile,
        seed=seed,
        use_ipf=use_ipf,
        generate_narratives=generate_narratives,
        population_mode=population_mode,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        allow_mock_narratives=allow_mock_narratives,
        llm_rate_limit=llm_rate_limit,
        strict=strict,
        raise_on_reject=raise_on_reject,
    )

    # High-stakes outputs exclude estimated fields by policy.
    if profile == UseCaseProfile.HIGH_STAKES:
        config.output_drop_fields = list(HIGH_STAKES_OUTPUT_DROP_FIELDS)
    
    if config_override:
        for key, value in config_override.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    logger.info(f"Generating dataset: n={n}, profile={profile.value}, strict={strict}")
    
    # Stage 1: Structured generation
    logger.info("Stage 1: Structured generation...")
    generator = PersonaGenerator(seed=seed, population_mode=population_mode)
    
    if use_ipf:
        personas, ipf_metrics = generator.generate_with_ethnicity_correction(
            n,
            oversample_factor=config.oversample_factor,
            return_metrics=True
        )
    else:
        personas = generator.generate(n, show_progress=False, use_ethnicity_correction=False)
        ipf_metrics = None
    
    logger.info(f"Generated {len(personas)} personas")
    
    # Stage 2: Narrative enrichment (if enabled)
    if generate_narratives and profile != UseCaseProfile.ANALYSIS_ONLY:
        logger.info("Stage 2: Narrative enrichment...")
        try:
            from .narrative_generator import NarrativeGenerator
            
            nar_gen = NarrativeGenerator(
                provider=llm_provider,
                api_key=llm_api_key,
                model=llm_model,
            )
            delay = 0.0
            if config.llm_rate_limit and config.llm_rate_limit > 0:
                delay = max(delay, 1.0 / config.llm_rate_limit)
            personas = nar_gen.generate_batch(personas, show_progress=True, delay=delay)
            
        except Exception as e:
            logger.error(f"Narrative generation failed: {e}")
            if strict or not config.allow_mock_narratives:
                raise RuntimeError(f"Narrative generation failed in strict mode: {e}")
            # In non-strict mode, continue with mock narratives
    
    # Stage 3: Trust validation
    logger.info("Stage 3: Trust validation...")
    report_gen = TrustReportGenerator(
        census_distributions=CENSUS,
        use_case=profile,
        population_mode=population_mode,
    )
    
    # Get provenance info
    provenance_info = CENSUS.get_all_provenance(population_mode)
    if config.output_drop_fields:
        provenance_info = {
            k: v for k, v in provenance_info.items()
            if k not in set(config.output_drop_fields)
        }
    
    trust_report = report_gen.generate_report(
        personas,
        provenance_info=provenance_info,
        random_seed=seed,
        generator_version="1.0.0",
        generator_config=config.to_dict(),
        ipf_metrics=ipf_metrics,
        pxweb_cache_timestamp=CENSUS.get_pxweb_snapshot_timestamp(population_mode),
    )
    
    # Determine decision and gate codes
    if trust_report.hard_gate_triggered:
        decision = "REJECT"
    elif trust_report.gate_reasons:
        decision = "PASS_WITH_WARNINGS"
    else:
        decision = "PASS"
    
    # Map reasons to machine codes
    gate_codes = []
    for reason in trust_report.gate_reasons:
        code = map_reason_to_code(reason)
        if code != GateCode.PASS:
            gate_codes.append(code)
    
    # Add IPF-related codes
    if ipf_metrics and ipf_metrics.ess_ratio < USE_CASE_THRESHOLDS[profile]["min_ess_ratio"]:
        gate_codes.append(GateCode.ESS_TOO_LOW)
    
    # In strict mode, any warning becomes a hard gate
    if strict and decision == "PASS_WITH_WARNINGS":
        decision = "REJECT"
        gate_codes.append(GateCode.CONFIG_INCOMPLETE)  # Marker for strict escalation
    
    # Sort gate codes deterministically
    gate_codes = sort_gate_codes(gate_codes)
    
    # Stage 4: Create run manifest
    logger.info("Stage 4: Creating run manifest...")
    run_manifest = create_run_manifest_from_trust_report(
        trust_report,
        config=config.to_dict(),
        generator_version="1.0.0",
        provenance_info=provenance_info,
        pxweb_base_url=NBS_BASE_URL,
    )
    
    # Add strict mode to run manifest for debugging
    run_manifest.config_summary["strict_mode"] = strict
    run_manifest.config_summary["profile"] = profile.value
    
    elapsed = time.perf_counter() - start_time
    run_manifest.generation_end_time = datetime.now().isoformat()
    run_manifest.duration_seconds = elapsed
    
    # Create bundle
    bundle = DatasetBundle(
        personas=personas,
        trust_report=trust_report,
        gate_codes=gate_codes,
        decision=decision,
        decision_reasons=trust_report.gate_reasons,
        run_manifest=run_manifest,
        ipf_metrics=ipf_metrics,
        config=config,
        generation_time_seconds=elapsed,
    )
    
    logger.info(f"Generation complete: {bundle.summary()}")
    
    # In strict mode, raise on rejection
    if strict and decision == "REJECT" and config.raise_on_reject:
        raise RuntimeError(
            f"Dataset generation failed quality gates in strict mode. "
            f"Codes: {[c.value for c in gate_codes]}"
        )
    
    return bundle


# Convenience exports for Gradient
__all__ = [
    "generate_dataset",
    "DatasetBundle",
    "DatasetConfig",
    "GateCode",
]
