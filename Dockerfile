# Moldova Personas Generator
# Multi-stage Dockerfile for production and development

# =============================================================================
# Stage 1: Base builder
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY packages/core/moldova_personas/ ./packages/core/moldova_personas/

# Create virtual environment and install
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel && \
    pip install .[all]

# =============================================================================
# Stage 2: Runtime (production)
# =============================================================================
FROM python:3.12-slim AS runtime

LABEL maintainer="Moldova Personas Team <team@example.com>"
LABEL description="Scientifically rigorous synthetic population generator for Moldova"
LABEL version="1.0.0"

# Security: Run as non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create cache directory with proper permissions
RUN mkdir -p /home/appuser/.moldova_personas/cache && \
    chown -R appuser:appgroup /home/appuser && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Set environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MOLDOVA_PERSONAS_CACHE_DIR=/home/appuser/.moldova_personas/cache

# Default command shows help
CMD ["moldova-personas", "--help"]

# =============================================================================
# Stage 3: Development
# =============================================================================
FROM builder AS development

WORKDIR /app

# Install dev dependencies
RUN pip install .[all,dev,test]

# Copy test files
COPY tests/ ./tests/
COPY benchmarks/ ./benchmarks/
COPY config/ ./config/

# Install pre-commit if available
RUN pip install pre-commit || true

# Default to bash for development
CMD ["/bin/bash"]

# =============================================================================
# Stage 4: CI/Test
# =============================================================================
FROM builder AS ci

WORKDIR /app

# Install test dependencies
RUN pip install .[all,test]

# Copy everything needed for tests
COPY tests/ ./tests/
COPY benchmarks/ ./benchmarks/
COPY config/ ./config/
COPY pytest.ini ./

# Run tests by default (fast only)
CMD ["pytest", "tests/", "-m", "not slow and not network", "-v"]
