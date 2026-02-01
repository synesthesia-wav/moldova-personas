"""
Run Manifest for reproducible persona generation.

Captures complete provenance of a generation run including:
- Configuration (hashed)
- Data sources (with timestamps)
- Software versions
- Random seeds
- Target distributions by field
- IPF metrics (ESS, information loss)
- Quality gates applied

This creates a single source of truth for auditability and reproducibility.
"""

import json
import hashlib
import platform
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class DataSourceType(Enum):
    """Classification of data sources."""
    PXWEB_LIVE = "pxweb_live"
    PXWEB_CACHED = "pxweb_cached"
    IPF_DERIVED = "ipf_derived"
    CENSUS_HARDCODED = "census_hardcoded"
    ESTIMATED = "estimated"


@dataclass
class FieldProvenance:
    """Provenance for a single field's distribution."""
    field_name: str
    source_type: str
    source_table: Optional[str] = None
    source_timestamp: Optional[str] = None  # ISO format
    cache_age_days: Optional[int] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "source_type": self.source_type,
            "source_table": self.source_table,
            "source_timestamp": self.source_timestamp,
            "cache_age_days": self.cache_age_days,
            "confidence": self.confidence,
        }


@dataclass
class DistributionMapping:
    """Mapping between external and internal category labels."""
    external_label: str
    internal_label: str
    mapping_rule: str  # e.g., "exact", "aggregated", "estimated"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "external_label": self.external_label,
            "internal_label": self.internal_label,
            "mapping_rule": self.mapping_rule,
        }


@dataclass
class RunManifest:
    """
    Complete manifest for a persona generation run.
    
    This is the single artifact that enables full reproducibility
    and auditability of a generation run.
    """
    
    # =========================================================================
    # IDENTIFICATION
    # =========================================================================
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    manifest_version: str = "1.0.0"
    
    # =========================================================================
    # TIMING
    # =========================================================================
    generation_start_time: Optional[str] = None  # ISO format
    generation_end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    # =========================================================================
    # SOFTWARE ENVIRONMENT
    # =========================================================================
    generator_version: str = "unknown"
    python_version: str = field(default_factory=lambda: platform.python_version())
    platform: str = field(default_factory=lambda: platform.platform())
    
    # =========================================================================
    # CONFIGURATION (HASHED FOR SECURITY)
    # =========================================================================
    config_hash: Optional[str] = None  # SHA256 of full config
    config_hash_short: Optional[str] = None  # First 16 chars for display
    config_summary: Dict[str, Any] = field(default_factory=dict)  # Non-sensitive subset
    
    # =========================================================================
    # RANDOMIZATION
    # =========================================================================
    random_seed: Optional[int] = None
    np_random_seed: Optional[int] = None
    
    # =========================================================================
    # TARGET SPECIFICATION
    # =========================================================================
    target_persona_count: int = 0
    
    # =========================================================================
    # DATA PROVENANCE BY FIELD
    # =========================================================================
    field_provenance: List[FieldProvenance] = field(default_factory=list)
    
    # =========================================================================
    # DISTRIBUTION MAPPINGS (CATEGORY ALIGNMENT)
    # =========================================================================
    # Tracks how external category labels map to internal ones
    distribution_mappings: Dict[str, List[DistributionMapping]] = field(default_factory=dict)
    
    # =========================================================================
    # PXWEB SNAPSHOT METADATA
    # =========================================================================
    pxweb_snapshot_timestamp: Optional[str] = None  # When data was fetched
    pxweb_base_url: str = "https://statbank.statistica.md/PxWeb/api/v1/en"
    
    # =========================================================================
    # IPF METRICS
    # =========================================================================
    ipf_applied: bool = False
    ipf_original_sample_size: Optional[int] = None
    ipf_effective_sample_size: Optional[float] = None
    ipf_ess_ratio: Optional[float] = None
    ipf_information_loss: Optional[float] = None
    ipf_weight_concentration: Optional[float] = None  # max_weight / mean_weight
    
    # =========================================================================
    # QUALITY GATES
    # =========================================================================
    use_case_profile: str = "narrative_required"
    quality_tier_achieved: str = "B"
    hard_gate_triggered: bool = False
    gate_reasons: List[str] = field(default_factory=list)
    
    # =========================================================================
    # OUTPUT FILES
    # =========================================================================
    output_files: Dict[str, str] = field(default_factory=dict)
    
    # =========================================================================
    # CENSUS REFERENCE
    # =========================================================================
    census_reference_year: int = 2024
    data_reference_year: int = 2024
    
    def compute_config_hash(self, config: Dict[str, Any]) -> None:
        """Compute SHA256 hash of configuration."""
        config_str = json.dumps(config, sort_keys=True, ensure_ascii=False)
        full_hash = hashlib.sha256(config_str.encode()).hexdigest()
        self.config_hash = full_hash
        self.config_hash_short = full_hash[:16]
    
    def add_field_provenance(self, field_name: str, source_type: str, 
                            source_table: Optional[str] = None,
                            source_timestamp: Optional[str] = None,
                            cache_age_days: Optional[int] = None,
                            confidence: float = 1.0) -> None:
        """Add provenance for a field."""
        self.field_provenance.append(FieldProvenance(
            field_name=field_name,
            source_type=source_type,
            source_table=source_table,
            source_timestamp=source_timestamp,
            cache_age_days=cache_age_days,
            confidence=confidence,
        ))
    
    def add_distribution_mapping(self, field: str, external: str, 
                                 internal: str, rule: str = "exact") -> None:
        """Record how external categories map to internal ones."""
        if field not in self.distribution_mappings:
            self.distribution_mappings[field] = []
        self.distribution_mappings[field].append(DistributionMapping(
            external_label=external,
            internal_label=internal,
            mapping_rule=rule,
        ))
    
    def get_fields_by_source(self, source_type: str) -> List[str]:
        """Get all fields with a specific source type."""
        return [fp.field_name for fp in self.field_provenance 
                if fp.source_type == source_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "manifest_version": self.manifest_version,
            "generation_start_time": self.generation_start_time,
            "generation_end_time": self.generation_end_time,
            "duration_seconds": self.duration_seconds,
            "software": {
                "generator_version": self.generator_version,
                "python_version": self.python_version,
                "platform": self.platform,
            },
            "configuration": {
                "config_hash": self.config_hash,
                "config_hash_short": self.config_hash_short,
                "config_summary": self.config_summary,
            },
            "randomization": {
                "random_seed": self.random_seed,
                "np_random_seed": self.np_random_seed,
            },
            "targets": {
                "target_persona_count": self.target_persona_count,
                "census_reference_year": self.census_reference_year,
                "data_reference_year": self.data_reference_year,
            },
            "data_provenance": {
                "pxweb_snapshot_timestamp": self.pxweb_snapshot_timestamp,
                "pxweb_base_url": self.pxweb_base_url,
                "fields": [fp.to_dict() for fp in self.field_provenance],
            },
            "distribution_mappings": {
                field: [m.to_dict() for m in mappings]
                for field, mappings in self.distribution_mappings.items()
            },
            "ipf_metrics": {
                "applied": self.ipf_applied,
                "original_sample_size": self.ipf_original_sample_size,
                "effective_sample_size": self.ipf_effective_sample_size,
                "ess_ratio": self.ipf_ess_ratio,
                "information_loss": self.ipf_information_loss,
                "weight_concentration": self.ipf_weight_concentration,
            },
            "quality_gates": {
                "use_case_profile": self.use_case_profile,
                "quality_tier_achieved": self.quality_tier_achieved,
                "hard_gate_triggered": self.hard_gate_triggered,
                "gate_reasons": self.gate_reasons,
            },
            "output_files": self.output_files,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save(self, filepath: str) -> None:
        """Save manifest to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, filepath: str) -> "RunManifest":
        """Load manifest from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        manifest = cls()
        manifest.run_id = data.get("run_id", "")
        manifest.manifest_version = data.get("manifest_version", "1.0.0")
        manifest.generation_start_time = data.get("generation_start_time")
        manifest.generation_end_time = data.get("generation_end_time")
        manifest.duration_seconds = data.get("duration_seconds")
        
        software = data.get("software", {})
        manifest.generator_version = software.get("generator_version", "unknown")
        manifest.python_version = software.get("python_version", "")
        manifest.platform = software.get("platform", "")
        
        config = data.get("configuration", {})
        manifest.config_hash = config.get("config_hash")
        manifest.config_hash_short = config.get("config_hash_short")
        manifest.config_summary = config.get("config_summary", {})
        
        randomization = data.get("randomization", {})
        manifest.random_seed = randomization.get("random_seed")
        manifest.np_random_seed = randomization.get("np_random_seed")
        
        targets = data.get("targets", {})
        manifest.target_persona_count = targets.get("target_persona_count", 0)
        manifest.census_reference_year = targets.get("census_reference_year", 2024)
        manifest.data_reference_year = targets.get("data_reference_year", 2024)
        
        provenance = data.get("data_provenance", {})
        manifest.pxweb_snapshot_timestamp = provenance.get("pxweb_snapshot_timestamp")
        manifest.pxweb_base_url = provenance.get("pxweb_base_url", "")
        manifest.field_provenance = [
            FieldProvenance(**fp) for fp in provenance.get("fields", [])
        ]
        
        mappings = data.get("distribution_mappings", {})
        manifest.distribution_mappings = {
            field: [DistributionMapping(**m) for m in ms]
            for field, ms in mappings.items()
        }
        
        ipf = data.get("ipf_metrics", {})
        manifest.ipf_applied = ipf.get("applied", False)
        manifest.ipf_original_sample_size = ipf.get("original_sample_size")
        manifest.ipf_effective_sample_size = ipf.get("effective_sample_size")
        manifest.ipf_ess_ratio = ipf.get("ess_ratio")
        manifest.ipf_information_loss = ipf.get("information_loss")
        manifest.ipf_weight_concentration = ipf.get("weight_concentration")
        
        gates = data.get("quality_gates", {})
        manifest.use_case_profile = gates.get("use_case_profile", "narrative_required")
        manifest.quality_tier_achieved = gates.get("quality_tier_achieved", "B")
        manifest.hard_gate_triggered = gates.get("hard_gate_triggered", False)
        manifest.gate_reasons = gates.get("gate_reasons", [])
        
        manifest.output_files = data.get("output_files", {})
        
        return manifest
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Run Manifest: {self.run_id}",
            f"Generator Version: {self.generator_version}",
            f"Config Hash: {self.config_hash_short or 'N/A'}",
            f"Target Count: {self.target_persona_count}",
            f"Seed: {self.random_seed}",
        ]
        
        if self.duration_seconds:
            lines.append(f"Duration: {self.duration_seconds:.2f}s")
        
        # Data source summary
        pxweb_fields = self.get_fields_by_source("PXWEB_DIRECT")
        if pxweb_fields:
            lines.append(f"PxWeb Live Fields: {', '.join(pxweb_fields)}")
        
        fallback_fields = self.get_fields_by_source("CENSUS_HARDCODED")
        if fallback_fields:
            lines.append(f"Fallback Fields: {', '.join(fallback_fields)}")
        
        if self.ipf_applied:
            lines.append(f"IPF ESS Ratio: {self.ipf_ess_ratio:.1%}")
            lines.append(f"IPF Information Loss: {self.ipf_information_loss:.1%}")
        
        lines.append(f"Quality Tier: {self.quality_tier_achieved}")
        
        if self.hard_gate_triggered:
            lines.append("⚠️  HARD GATE TRIGGERED")
            for reason in self.gate_reasons:
                lines.append(f"  - {reason}")
        
        return "\n".join(lines)


def create_run_manifest_from_trust_report(
    trust_report: Any,  # TrustReport
    config: Dict[str, Any],
    generator_version: str = "0.3.0",
    provenance_info: Optional[Dict[str, Dict[str, Any]]] = None,
    pxweb_base_url: Optional[str] = None,
) -> RunManifest:
    """
    Create a RunManifest from a TrustReport.
    
    This bridges the gap between the quality assessment (TrustReport)
    and the reproducibility artifact (RunManifest).
    """
    from datetime import datetime
    from .pxweb_fetcher import DataProvenance, NBS_BASE_URL
    
    manifest = RunManifest()
    manifest.run_id = trust_report.report_id
    manifest.generation_start_time = trust_report.generation_timestamp
    manifest.generator_version = generator_version
    manifest.compute_config_hash(config)
    manifest.config_summary = config
    manifest.random_seed = trust_report.random_seed
    manifest.target_persona_count = trust_report.persona_count
    manifest.census_reference_year = trust_report.census_reference_year
    manifest.data_reference_year = trust_report.data_reference_year
    
    # IPF metrics
    if trust_report.ipf_metrics:
        manifest.ipf_applied = True
        manifest.ipf_original_sample_size = trust_report.ipf_metrics.original_sample_size
        manifest.ipf_effective_sample_size = trust_report.ipf_metrics.effective_sample_size
        manifest.ipf_ess_ratio = trust_report.ipf_metrics.ess_ratio
        manifest.ipf_information_loss = trust_report.ipf_metrics.information_loss
        manifest.ipf_weight_concentration = trust_report.ipf_metrics.weight_concentration
    
    # Quality gates
    manifest.use_case_profile = trust_report.use_case_profile.value
    manifest.quality_tier_achieved = trust_report.overall_quality_tier.value
    manifest.hard_gate_triggered = trust_report.hard_gate_triggered
    manifest.gate_reasons = trust_report.gate_reasons
    
    # PxWeb snapshot
    manifest.pxweb_snapshot_timestamp = trust_report.pxweb_snapshot_timestamp
    manifest.pxweb_base_url = pxweb_base_url or NBS_BASE_URL

    # Field-level provenance (optional)
    if provenance_info:
        now = datetime.now()
        for field_name, info in provenance_info.items():
            prov = info.get("provenance", "")
            source_type = prov
            for enum_val in DataProvenance:
                if prov == enum_val.value or prov == enum_val.name:
                    source_type = enum_val.name
                    break
            source_ts = info.get("last_fetched")
            cache_age_days = None
            if source_ts:
                try:
                    ts = datetime.fromisoformat(str(source_ts).replace("Z", "+00:00"))
                    cache_age_days = (now - ts).days
                except Exception:
                    cache_age_days = None
            manifest.add_field_provenance(
                field_name,
                source_type,
                source_table=info.get("source_table"),
                source_timestamp=source_ts,
                cache_age_days=cache_age_days,
                confidence=info.get("confidence", 1.0),
            )
    
    return manifest
