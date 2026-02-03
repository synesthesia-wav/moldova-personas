"""Path helpers for monorepo layouts."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root by walking upwards for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback to a best-effort guess (packages/core/moldova_personas)
    return current.parents[3]


def config_dir() -> Path:
    """Return the config directory path."""
    return repo_root() / "config"


def data_path(filename: str) -> Path:
    """Return an absolute path to a repo data file."""
    return repo_root() / filename
