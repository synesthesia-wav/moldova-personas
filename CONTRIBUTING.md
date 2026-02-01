# Contributing to Moldova Personas Generator

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- pip or conda for package management

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/moldova-personas.git
cd moldova-personas
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. Copy environment template and configure:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Running Tests

Run the full test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=moldova_personas --cov-report=term-missing
```

Run specific test file:
```bash
pytest tests/test_generator.py -v
```

## Code Quality

### Type Checking

We use mypy for static type checking:
```bash
mypy moldova_personas/ --ignore-missing-imports
```

### Linting

We use ruff for linting and formatting:
```bash
# Check code
ruff check moldova_personas/

# Auto-fix issues
ruff check --fix moldova_personas/

# Format code
ruff format moldova_personas/
```

## Project Structure

```
moldova-personas/
├── moldova_personas/     # Main package
│   ├── __init__.py
│   ├── cli.py           # Command-line interface
│   ├── generator.py     # Core generation engine
│   ├── validators.py    # Validation pipeline
│   ├── exporters.py     # Export utilities
│   ├── llm_client.py    # LLM integrations
│   ├── models.py        # Pydantic models
│   ├── census_data.py   # Census distributions
│   ├── names.py         # Name generation
│   ├── prompts.py       # LLM prompts
│   ├── narrative_generator.py
│   └── exceptions.py    # Custom exceptions
├── tests/               # Test suite
├── config/              # Configuration JSON files
├── docs/                # Documentation
└── output/              # Generated outputs (gitignored)
```

## Adding New Features

### Adding a New LLM Provider

1. Add client class in `llm_client.py` inheriting from `LLMClient`
2. Implement `generate()` method with retry logic
3. Add to `create_llm_client()` factory function
4. Add tests in `tests/test_llm_client.py`
5. Update CLI help text

### Adding New Validations

1. Add validation method in `ValidationPipeline` class
2. Return `ValidationError` objects for issues found
3. Add test cases in `tests/test_validators.py`
4. Document in validation report

### Adding New Export Formats

1. Create exporter class in `exporters.py`
2. Implement export method
3. Add to `export_all_formats()` function
4. Add test cases

## Commit Guidelines

- Use clear, descriptive commit messages
- Reference issue numbers when applicable
- Keep commits focused on single changes

Example:
```
feat: Add retry logic for LLM API calls

- Implements exponential backoff (2x multiplier)
- Retries up to 3 times on rate limits
- Adds logging for retry attempts

Fixes #123
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure they pass
5. Run type checking and linting
6. Update CHANGELOG.md
7. Commit your changes
8. Push to your fork
9. Open a Pull Request

### PR Checklist

- [ ] Tests pass (`pytest`)
- [ ] Type checking passes (`mypy`)
- [ ] Linting passes (`ruff`)
- [ ] New code has test coverage
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

## Code Style

- Follow PEP 8 guidelines
- Use type annotations for function signatures
- Document classes and public methods with docstrings
- Keep functions focused and under 50 lines when possible
- Use meaningful variable names

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages or stack traces

## Questions?

Feel free to open an issue for questions or join discussions.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
