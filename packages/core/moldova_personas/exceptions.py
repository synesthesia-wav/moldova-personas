"""Custom exceptions for the Moldova personas pipeline."""

from typing import Optional


class MoldovaPersonasError(Exception):
    """Base exception for all pipeline errors."""
    pass


class LLMGenerationError(MoldovaPersonasError):
    """Raised when LLM generation fails."""
    
    def __init__(self, message: str, provider: Optional[str] = None, retryable: bool = True):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class ValidationError(MoldovaPersonasError):
    """Raised when persona validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, persona_id: Optional[str] = None):
        super().__init__(message)
        self.field = field
        self.persona_id = persona_id


class ExportError(MoldovaPersonasError):
    """Raised when exporting personas fails."""
    
    def __init__(self, message: str, format: Optional[str] = None, output_path: Optional[str] = None):
        super().__init__(message)
        self.format = format
        self.output_path = output_path


class ConfigurationError(MoldovaPersonasError):
    """Raised when configuration is invalid."""
    pass
