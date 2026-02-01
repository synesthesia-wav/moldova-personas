# PR Checklist: Moldova Personas Pipeline Improvements

**Project**: `/Users/victorvanica/Coding Projects/moldova-personas/moldova_personas`  
**Created**: 2026-01-28  
**Target**: Production-grade pipeline (Grade B → Grade A)

---

## Phase 1: Testing Infrastructure (Priority: Critical)

- [ ] Add `pytest` to `requirements.txt` (or dev-requirements.txt)
- [ ] Create `tests/` directory structure:
  ```
  tests/
  ├── __init__.py
  ├── test_generator.py
  ├── test_validators.py
  ├── test_exporters.py
  ├── test_llm_client.py
  └── test_cli.py
  ```
- [ ] Write unit tests for `PersonaGenerator` (at least 5 test cases)
- [ ] Write unit tests for `ValidationPipeline` (test each validation layer)
- [ ] Write unit tests for exporters (verify output formats)
- [ ] Add `pytest-cov` for coverage reporting (target: 80%)
- [ ] Create `pytest.ini` with sensible defaults
- [ ] Add CI workflow `.github/workflows/ci.yml`:
  ```yaml
  name: CI
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: '3.11'
        - run: pip install -r requirements.txt
        - run: pip install pytest pytest-cov mypy
        - run: pytest --cov=. --cov-report=xml
        - run: mypy --strict .
  ```

---

## Phase 2: Error Handling & Logging (Priority: High)

- [ ] Create `exceptions.py` with custom exception classes:
  ```python
  class LLMGenerationError(Exception): pass
  class ValidationError(Exception): pass
  class ExportError(Exception): pass
  ```
- [ ] Replace bare `except Exception` in `llm_client.py` with specific handling
- [ ] Add retry logic with exponential backoff for LLM API calls
- [ ] Replace all `print()` statements with `logging` module
- [ ] Add structured logging (JSON format) for production use
- [ ] Implement proper error propagation in CLI commands

---

## Phase 3: Type Safety (Priority: High)

- [ ] Create `pyproject.toml` with mypy configuration:
  ```toml
  [tool.mypy]
  python_version = "3.11"
  strict = true
  warn_return_any = true
  warn_unused_ignores = true
  ```
- [ ] Add `py.typed` marker file
- [ ] Fix all mypy errors (run `mypy --strict .`)
- [ ] Add type stubs for optional dependencies (`types-openai`, etc.)

---

## Phase 4: Dependency Management (Priority: Medium)

- [ ] Pin all dependencies with upper bounds in `requirements.txt`
- [ ] OR migrate to `pyproject.toml` with Poetry:
  ```toml
  [tool.poetry.dependencies]
  python = "^3.11"
  numpy = ">=1.24.0,<2.0.0"
  pandas = ">=2.0.0,<3.0.0"
  ```
- [ ] Create `requirements-dev.txt` with test/development dependencies
- [ ] Add `Makefile` or `justfile` for common tasks:
  ```makefile
  test:
      pytest --cov=. --cov-report=term-missing
  lint:
      mypy --strict .
      ruff check .
  format:
      ruff format .
  ```

---

## Phase 5: Configuration Management (Priority: Medium)

- [ ] Create `.env.example` file with all environment variables
- [ ] Add `pydantic-settings` for type-safe configuration
- [ ] Create `config/` directory with YAML/JSON data files:
  ```
  config/
  ├── cities_by_region.json
  ├── districts_by_region.json
  └── census_distributions.json
  ```
- [ ] Move hardcoded data from `generator.py` to config files
- [ ] Add config validation on startup

---

## Phase 6: Observability (Priority: Medium)

- [ ] Add `prometheus-client` for metrics
- [ ] Track key metrics:
  - Generation rate (personas/second)
  - LLM latency (p50, p95, p99)
  - Validation error rate
  - Export duration by format
- [ ] Add OpenTelemetry tracing for LLM calls
- [ ] Create `docker-compose.yml` with Prometheus + Grafana for local monitoring

---

## Phase 7: Performance & Benchmarks (Priority: Low)

- [ ] Add `pytest-benchmark` to dev dependencies
- [ ] Create benchmark tests for:
  - 1K persona generation
  - 10K persona generation
  - 100K persona generation
- [ ] Profile memory usage with `memory_profiler`
- [ ] Document performance characteristics in README

---

## Phase 8: Documentation (Priority: Low)

- [ ] Add API documentation (docstrings → Sphinx)
- [ ] Create `CONTRIBUTING.md` with development setup
- [ ] Add architecture decision records (ADRs) for key design choices
- [ ] Create example notebooks showing usage patterns

---

## Definition of Done

A PR can be merged when:
- [ ] All new code has tests with >80% coverage
- [ ] CI passes (lint, type-check, test)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Version bump in `pyproject.toml` (if using) or `__init__.py`

---

## Estimated Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1 (Testing) | 3-4 days | Week 1 |
| Phase 2 (Error Handling) | 2-3 days | Week 1-2 |
| Phase 3 (Type Safety) | 2 days | Week 2 |
| Phase 4 (Dependencies) | 1 day | Week 2 |
| Phase 5 (Configuration) | 2 days | Week 2-3 |
| Phase 6 (Observability) | 3 days | Week 3 |
| Phase 7-8 (Nice-to-have) | Ongoing | Week 4+ |

**Total: ~2 weeks for core improvements (Grade B → Grade A)**

---

## Related Documents

- [Rigor Assessment](./moldova-personas-rigor-assessment.md) — Full assessment details
