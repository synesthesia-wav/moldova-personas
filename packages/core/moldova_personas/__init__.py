"""
Moldova Synthetic Personas Generator

A prototype pipeline for generating 100,000 synthetic personas representative 
of Moldova's population based on 2024 Census data.

Basic usage:
    from moldova_personas import PersonaGenerator, ValidationPipeline
    from moldova_personas.narrative_generator import NarrativeGenerator
    
    # Generate structured data
    generator = PersonaGenerator(seed=42)
    personas = generator.generate(1000)
    
    # Add narratives (requires LLM)
    nar_gen = NarrativeGenerator(provider="openai")
    personas = nar_gen.generate_batch(personas)

CLI usage:
    python -m moldova_personas generate --count 1000 --output ./output
    python -m moldova_personas generate --count 100 --llm-provider openai
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .generator import PersonaGenerator
from .validators import ValidationPipeline
from .exporters import ParquetExporter, export_all_formats
from .models import Persona, AgeConstraints
from .statistical_tests import StatisticalValidator, calculate_adaptive_tolerance
from .async_narrative_generator import (
    AsyncNarrativeGenerator,
    enrich_personas_parallel
)
from .names import (
    generate_name,
    get_language_by_ethnicity,
    get_religion_by_ethnicity,
)
from .prompts import (
    generate_full_prompt,
    parse_narrative_response,
    validate_narrative_against_persona,
    get_prompt_version,
)
from .checkpoint import (
    Checkpoint,
    CheckpointManager,
    CheckpointStreamingGenerator,
    get_checkpoint_summary,
)
from .trust_report import (
    TrustReport,
    TrustReportGenerator,
    TrustDecisionRecord,
    QualityTier,
    UseCaseProfile,
    MarginalError,
    IPFMetrics,
    NarrativeQualityMetrics,
    compute_joint_drift,
    compute_ipf_metrics,
    CRITICAL_FIELDS,
    HIGH_STAKES_CRITICAL_FIELDS,
    USE_CASE_THRESHOLDS,
    NARRATIVE_QUALITY_THRESHOLDS,
)
from .locality_integrity import (
    validate_locality_config,
    check_city_selection_safety,
    get_fallback_candidates,
    validate_and_log_config,
    LocalityIntegrityError,
    LocalityValidationResult,
)
from .run_manifest import (
    RunManifest,
    FieldProvenance,
    DistributionMapping,
    DataSourceType,
    create_run_manifest_from_trust_report,
)
from .narrative_contract import (
    NarrativeContract,
    NarrativeContractValidator,
    ContractValidationResult,
    GoldenNarrativeFixture,
    NarrativeRegressionTest,
    get_ci_regression_fixture,
    NARRATIVE_CONTRACT_VERSION,
    NARRATIVE_JSON_SCHEMA,
)
from .gate_codes import (
    GateCode,
)
from .gradient_integration import (
    generate_dataset,
    DatasetBundle,
    DatasetConfig,
    GRADIENT_PAYLOAD_VERSION,
    GATE_CODE_RECOMMENDATIONS,
    get_recommendation,
)
from .gate_codes import (
    sort_gate_codes,
)
from .data_freshness import (
    DataFreshnessMonitor,
    FreshnessReport,
    FreshnessStatus,
    check_data_freshness,
)
from .metrics import (
    MetricsCollector,
    get_collector,
    record_persona_generation,
    timed_generation,
)
from .streaming import (
    StreamingGenerator,
    StreamingParquetExporter,
    StreamingJSONLExporter,
    StreamingCSVExporter,
    generate_and_export_streaming,
    generate_million_personas,
)

__all__ = [
    "PersonaGenerator",
    "ValidationPipeline", 
    "ParquetExporter",
    "export_all_formats",
    "Persona",
    "AgeConstraints",
    "StatisticalValidator",
    "calculate_adaptive_tolerance",
    "AsyncNarrativeGenerator",
    "enrich_personas_parallel",
    "generate_name",
    "get_language_by_ethnicity",
    "get_religion_by_ethnicity",
    "generate_full_prompt",
    "parse_narrative_response",
    "validate_narrative_against_persona",
    "get_prompt_version",
    "Checkpoint",
    "CheckpointManager",
    "CheckpointStreamingGenerator",
    "get_checkpoint_summary",
    "TrustReport",
    "TrustReportGenerator",
    "TrustDecisionRecord",
    "QualityTier",
    "UseCaseProfile",
    "MarginalError",
    "IPFMetrics",
    "NarrativeQualityMetrics",
    "compute_joint_drift",
    "compute_ipf_metrics",
    "CRITICAL_FIELDS",
    "HIGH_STAKES_CRITICAL_FIELDS",
    "USE_CASE_THRESHOLDS",
    "NARRATIVE_QUALITY_THRESHOLDS",
    "validate_locality_config",
    "check_city_selection_safety",
    "get_fallback_candidates",
    "validate_and_log_config",
    "LocalityIntegrityError",
    "LocalityValidationResult",
    "RunManifest",
    "FieldProvenance",
    "DistributionMapping",
    "DataSourceType",
    "create_run_manifest_from_trust_report",
    "NarrativeContract",
    "NarrativeContractValidator",
    "ContractValidationResult",
    "GoldenNarrativeFixture",
    "NarrativeRegressionTest",
    "get_ci_regression_fixture",
    "NARRATIVE_CONTRACT_VERSION",
    "NARRATIVE_JSON_SCHEMA",
    "GateCode",
    "sort_gate_codes",
    "generate_dataset",
    "DatasetBundle",
    "DatasetConfig",
    "GRADIENT_PAYLOAD_VERSION",
    "GATE_CODE_RECOMMENDATIONS",
    "get_recommendation",
    # Data freshness
    "DataFreshnessMonitor",
    "FreshnessReport",
    "FreshnessStatus",
    "check_data_freshness",
    # Metrics
    "MetricsCollector",
    "get_collector",
    "record_persona_generation",
    "timed_generation",
    # Streaming
    "StreamingGenerator",
    "StreamingParquetExporter",
    "StreamingJSONLExporter",
    "StreamingCSVExporter",
    "generate_and_export_streaming",
    "generate_million_personas",
]
