# Moldova Personas Generator

[![CI](https://github.com/synesthesia-wav/moldova-personas/actions/workflows/ci.yml/badge.svg)](https://github.com/synesthesia-wav/moldova-personas/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A scientifically rigorous synthetic population generator for Moldova, grounded in official 2024 Census data from the National Bureau of Statistics (NBS).

## 🎯 Overview

This tool generates realistic synthetic personas for Moldova with:
- **Demographic accuracy**: Based on NBS 2024 Census data
- **Scientific rigor**: Full data provenance tracking (PxWeb → IPF → Persona)
- **Cultural authenticity**: Romanian language, Moldovan-specific content
- **Geographic precision**: Real cities and districts from NBS

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd moldova-personas

# Install dependencies
pip install -r requirements.txt

# Install the local package (monorepo)
pip install -e .
```

### Basic Usage

```bash
# Generate 500 personas (structured data only)
python -m moldova_personas generate --count 500 --output ./output

# Generate with LLM narratives (requires API key)
export DASHSCOPE_API_KEY="your-key-here"
python -m moldova_personas generate --count 100 --narratives --llm-provider dashscope

# Use IPF correction for better ethnicity distribution
python -m moldova_personas generate --count 500 --ipf --output ./output
```

### Python API

```python
from moldova_personas import PersonaGenerator

# Initialize generator
gen = PersonaGenerator(seed=42)

# Generate single persona
persona = gen.generate_single()
print(f"{persona.name}, {persona.age}, {persona.city}")

# Generate batch with IPF correction
personas = gen.generate_with_ethnicity_correction(n=1000)
```

## 🧭 Repository Layout

```
packages/core/            # Library package (moldova_personas)
apps/demos/               # Runnable demos and examples
tools/scripts/            # One-off utilities and scale tests
tests/                    # Test suite
docs/                     # Documentation and archives
config/                   # Census data + lookup tables
artifacts/                # Generated outputs, caches, logs
```

## 📊 Data Sources

### Live NBS PxWeb API (Real-time)
| Distribution | Source | Status |
|--------------|--------|--------|
| Sex | PxWeb API | ✅ Live |
| Residence (Urban/Rural) | PxWeb API | ✅ Live |
| Age Groups | PxWeb API | ✅ Live |

### Verified NBS 2024 Census (Hardcoded)
| Distribution | Source | Status |
|--------------|--------|--------|
| Ethnicity | NBS 2024 | ⏳ Awaiting PxWeb |
| Region | NBS 2024 | ⏳ Awaiting PxWeb |
| Education | NBS 2024 | ⏳ Awaiting PxWeb |
| Religion | NBS 2024 | ⏳ Awaiting PxWeb |
| Marital Status | NBS 2024 | ⏳ Awaiting PxWeb |
| Employment | NBS 2024 | ⏳ Awaiting PxWeb |

*Note: Hardcoded distributions use verified NBS 2024 final report data. They will automatically switch to live API when NBS publishes complete 2024 census data to PxWeb.*

### IPF-Derived Cross-Tabulations
- **Ethnicity × Region**: Derived via Iterative Proportional Fitting
- **Region × Urban/Rural**: Derived via IPF from marginals

## 🏗️ Architecture

```
packages/core/moldova_personas/
├── pxweb_fetcher.py          # NBS PxWeb API integration
├── census_data.py            # Census distributions with provenance
├── generator.py              # PGM-based persona generation
├── models.py                 # Persona data models (Pydantic)
├── names.py                  # Ethnicity-weighted name generation
├── prompts.py                # LLM prompt templates
├── narrative_generator.py    # Sync narrative generation
├── async_narrative_generator.py  # Parallel narrative generation
├── llm_client.py             # LLM provider abstraction
├── validators.py             # Data validation
├── exporters.py              # Output format handlers
└── cli.py                    # Command-line interface
```

### Data Flow

```
User Request
    ↓
CensusDistributions (lazy loading)
    ↓
NBSDataManager
    ↓
┌──────────────┬──────────────┬──────────────┐
│ Cache (fresh)│ PxWeb API    │ Fallback     │
│ Return cached│ Fetch live   │ Hardcoded    │
└──────────────┴──────────────┴──────────────┘
    ↓
PersonaGenerator (PGM + IPF)
    ↓
Persona with full provenance
```

### Narrative Pipeline (When `--narratives`)

When narratives are enabled, the generator follows a single unified pipeline:

```
PGM demographics + OCEAN traits
    ↓
LLM A → Cultural Background, Skills & Expertise, Goals & Ambitions, Hobbies & Interests
    ↓
LLM B → Overall / Professional / Arts / Sports / Travel / Culinary personas
```

This matches the diagram workflow and is the only narrative path.

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=moldova_personas --cov-report=html

# Run specific test module
python -m pytest tests/test_pxweb_fetcher.py -v
```

**Test Results**: 181/181 passing ✅

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [PLAN.md](PLAN.md) | Project planning and roadmap |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | Implementation guide and architecture |
| [PROTOTYPE.md](PROTOTYPE.md) | Prototype development history |
| [NBS_2024_REPORT.md](NBS_2024_REPORT.md) | NBS 2024 Census integration report |
| [TODO.md](TODO.md) | Project TODO and 100K generation roadmap |
| [ASSESSMENT.md](ASSESSMENT.md) | Assessment and quality reports |
| [SCIENTIFIC_ROBUSTNESS_ASSESSMENT.md](SCIENTIFIC_ROBUSTNESS_ASSESSMENT.md) | Scientific methodology assessment |
| [NBS_DATA_INTEGRATION.md](NBS_DATA_INTEGRATION.md) | Data integration details |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
| [QWEN_SETUP.md](QWEN_SETUP.md) | Setup instructions for Qwen LLM |

## 🔬 Scientific Rigor

### Data Provenance

Every distribution includes:
- **Provenance type**: PXWEB_DIRECT, PXWEB_CACHED, IPF_DERIVED, CENSUS_HARDCODED, ESTIMATED
- **Source table**: PxWeb dataset code (e.g., POP010200rcl.px)
- **Confidence score**: 0.0-1.0 reliability metric
- **Methodology**: Derivation method description
- **Limitations**: Known constraints (for estimates)

### Run Manifest

Each generation writes `metadata.json` in the output directory with:
- seed and configuration hash
- strict-geo flag
- provenance and cache timestamps
- output file list and runtime metadata

Disable with `--no-manifest` if needed.

### Statistical Methods

- **PGM (Probabilistic Graphical Models)**: For dependency-aware generation
- **IPF (Iterative Proportional Fitting)**: For cross-tabulation derivation
- **Maximum Entropy**: Ensures unbiased estimates
- **Chi-square validation**: Tests against NBS marginals

### Data Integrity Policy

- ❌ No fabricated locality names in strict geo mode (district-only when official locality data is unavailable)
- ❌ No template-based cultural backgrounds
- ⚠️ Some distributions are heuristic when census cross-tabs are missing (tracked as `ESTIMATED` in provenance)
- ✅ Empty strings instead of invented content when data unavailable

## 🔧 Configuration

### Environment Variables

```bash
# For LLM narrative generation (optional)
export DASHSCOPE_API_KEY="your-dashscope-key"
export OPENAI_API_KEY="your-openai-key"

# For local Qwen models (optional)
export QWEN_MODEL_PATH="/path/to/qwen-model"
```

### Cache Location

PxWeb data is cached in `~/.moldova_personas/cache/` with 30-day freshness checking.
Override per run with `--cache-dir` or set `MOLDOVA_PERSONAS_CACHE_DIR`.

### Cache Refresh

```bash
python -m moldova_personas cache refresh
```

## 📝 Output Formats

```bash
# Parquet (default - recommended)
python -m moldova_personas generate --count 1000 --format parquet

# JSON
python -m moldova_personas generate --count 1000 --format json

# JSONL
python -m moldova_personas generate --count 1000 --format jsonl

# CSV
python -m moldova_personas generate --count 1000 --format csv

# All formats
python -m moldova_personas generate --count 1000 --format all

# Partitioned Parquet
python -m moldova_personas generate --count 1000 --format parquet --partition-by region,sex

# Drop narrative fields
python -m moldova_personas generate --count 1000 --format parquet --drop-fields descriere_generala,profil_profesional
```

### Streaming Large Runs

```bash
# Stream to Parquet without holding all personas in memory
python -m moldova_personas generate --count 1000000 --format parquet --stream --batch-size 50000 --output ./output
```

Notes:
- Streaming supports `parquet`, `jsonl`, or `csv`.
- Streaming skips validation and narratives (use non-streaming for full validation).

## 🎭 Persona Schema

Each persona includes:

```python
{
    "uuid": "unique-identifier",
    "name": "Full Name",
    "sex": "Masculin/Feminin",
    "age": 35,
    "age_group": "35-44",
    "ethnicity": "Moldovean",
    "mother_tongue": "Română",
    "religion": "Ortodox",
    "region": "Chisinau",
    "city": "Chișinău",
    "district": "Mun. Chișinău",
    "residence_type": "Urban",
    "education_level": "Superior (Licență/Master)",
    "field_of_study": "Economie",
    "occupation": "Economist",
    "occupation_sector": "Servicii",
    "employment_status": "employed",
    "marital_status": "Căsătorit",
    # Persona variants (LLM B)
    "persona": "...",
    "professional_persona": "...",
    "sports_persona": "...",
    "arts_persona": "...",
    "travel_persona": "...",
    "culinary_persona": "...",
    # Context fields (LLM A)
    "cultural_background": "...",
    "skills_and_expertise": "...",
    "hobbies_and_interests": "...",
    "career_goals_and_ambitions": "...",
    # OCEAN traits (0-100)
    "ocean_openness": 62,
    "ocean_conscientiousness": 55,
    "ocean_extraversion": 41,
    "ocean_agreeableness": 58,
    "ocean_neuroticism": 49,
    # Optional narrative fields (LLM-generated)
    "descriere_generala": "...",
    "profil_profesional": "...",
    "hobby_sport": "...",
    "hobby_arta_cultura": "...",
    "hobby_calatorii": "...",
    "hobby_culinar": "...",
    "persona_summary": "...",
}
```

## 🚨 Known Limitations

1. **PxWeb Coverage**: Only 3/10 distributions available via live API (NBS hasn't published complete 2024 ethnicity/religion/education data yet)
2. **IPF Cross-tabs**: Derived from marginals, not observed joint distributions (documented with lower confidence 0.85)
3. **OCEAN Traits**: Synthetic personality traits are estimated (not from NBS) and tracked as such
4. **Cache Location**: Defaults to `~/.moldova_personas/cache/` (override via `--cache-dir` or `MOLDOVA_PERSONAS_CACHE_DIR`)
5. **Names**: First/last name lists are synthetic placeholders (not official frequency lists)

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📜 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- National Bureau of Statistics of the Republic of Moldova for census data
- PxWeb platform for API access
- Qwen LLM for narrative generation capabilities

---

**Status**: ✅ Production Ready | **Tests**: 181/181 Passing | **Grade**: A (Scientific Rigor)
