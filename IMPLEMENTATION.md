# Implementation Guide

## Overview

This document consolidates the implementation plans and summaries for the Moldova Personas Generator project.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate 500 personas
python -m moldova_personas generate --count 500 --output ./output

# Generate with narratives (requires API key)
python -m moldova_personas generate --count 100 --narratives --llm-provider dashscope
```

## Implementation Status

### Core Features (Completed ✅)
- [x] PGM-based persona generation with IPF correction
- [x] NBS 2024 Census data integration
- [x] Scientific-grade data provenance tracking
- [x] Live PxWeb API fetching (3 distributions)
- [x] IPF-derived cross-tabulations
- [x] Comprehensive test suite (181 tests)


### 100K Generation (Pending)
See [TODO.md](TODO.md) for generation roadmap.

## Architecture

```
packages/core/moldova_personas/
├── pxweb_fetcher.py      # NBS PxWeb data fetching
├── census_data.py        # Census distributions
├── generator.py          # Core PGM generator
├── models.py             # Persona data models
├── prompts.py            # LLM prompt templates
└── ...
```

## Data Sources

| Distribution | Source | Status |
|--------------|--------|--------|
| Sex | PxWeb API | ✅ Live |
| Residence | PxWeb API | ✅ Live |
| Age Group | PxWeb API | ✅ Live |
| Ethnicity | Hardcoded | ⏳ Awaiting NBS 2024 |
| Region | Hardcoded | ⏳ Awaiting NBS 2024 |

## API Keys

Set environment variables:
```bash
export DASHSCOPE_API_KEY="your-key-here"
```

## Documentation

- [SCIENTIFIC_ROBUSTNESS_ASSESSMENT.md](SCIENTIFIC_ROBUSTNESS_ASSESSMENT.md) - Scientific methodology
- [NBS_DATA_INTEGRATION.md](NBS_DATA_INTEGRATION.md) - Data integration details
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
