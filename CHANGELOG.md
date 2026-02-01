# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Payload Version Policy

- **Additive changes only** in `payload_version: 1.0` (new optional fields allowed)
- **Any rename/removal** → bump to `payload_version: 2.0`
- Consumers should ignore unknown fields for forward compatibility

---

## [1.0.0] - 2026-01-30

### Added
- **Unified Gradient entrypoint**: `generate_dataset()` function for single-call dataset generation
- **Versioned trust payload**: `payload_version: "1.0"` with `generator_version` and `schema_hash`
- **Machine-readable gate codes**: `GateCode` enum with 25 stable condition codes
- **Deterministic gate ordering**: Hard gates first, then warnings, alphabetical within groups
- **Gate code recommendations**: Human-readable actionable guidance for each condition
- **Monotonicity guarantee**: Confidence capped at "low" for REJECT, "medium" for HIGH_STAKES with fallback
- **Strict mode capture**: Profile and strict_mode in all artifacts (payload, trust report, run manifest)
- **Canary tests**: 31 tests validating payload shape, types, ranges, and backward compatibility
- **API compatibility tests**: 19 tests ensuring public API stability
- **Complete documentation**: PRODUCTION_READY.md, GRADIENT_INTEGRATION.md, ZERO_FABRICATION_POLICY.md

### Changed
- **Bumped version**: 0.3.0 → 1.0.0 (production ready)
- **Enhanced payload**: Added `base_confidence`, `monotonicity_applied`, `gate_code_details` with recommendations

### Security
- Zero-fabrication policy enforced: no real PII, addresses, or institution names
- Monotonic trust caps prevent "high trust" scores for rejected datasets

---

## [0.3.0] - 2026-01-29

### Added
- IPF correction with ESS metrics and weight concentration tracking
- Trust report with quality tiers (A/B/C/REJECT) and use-case profiles
- Run manifest for complete reproducibility
- Narrative contract for LLM format drift protection
- Data-source contract tests with golden fixtures
- Docker packaging and CI/CD workflow

---

## [0.2.0] - 2026-01-28

### Added
- PxWeb API integration with caching
- IPF engine for cross-tabulation derivation
- Provenance tracking (PXWEB_DIRECT, IPF_DERIVED, CENSUS_HARDCODED)
- Statistical validation with chi-square tests

---

## [0.1.0] - 2026-01-27

### Added
- Initial PGM-based persona generation
- Census data distributions (NBS 2024)
- Ethnicity-weighted name generation
- Basic validation pipeline
- Parquet/JSON/CSV export
