"""Custom exceptions for Query2Label system."""


class Query2LabelError(Exception):
    """Base exception for all Query2Label errors."""
    pass


class QueryTranslationError(Query2LabelError):
    """Raised when query translation fails."""
    
    def __init__(self, message: str, original_query: str = None, details: dict = None):
        super().__init__(message)
        self.original_query = original_query
        self.details = details or {}


class LabelValidationError(Query2LabelError):
    """Raised when label validation fails."""
    
    def __init__(self, message: str, invalid_labels: list = None, available_labels: list = None):
        super().__init__(message)
        self.invalid_labels = invalid_labels or []
        self.available_labels = available_labels or []


class DataLoadingError(Query2LabelError):
    """Raised when data loading operations fail."""
    
    def __init__(self, message: str, file_path: str = None, details: dict = None):
        super().__init__(message)
        self.file_path = file_path
        self.details = details or {}


class ConfigurationError(Query2LabelError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_key: str = None, expected_type: type = None):
        super().__init__(message)
        self.config_key = config_key
        self.expected_type = expected_type


class DSPyModuleError(Query2LabelError):
    """Raised when DSPy module operations fail."""
    
    def __init__(self, message: str, module_name: str = None, details: dict = None):
        super().__init__(message)
        self.module_name = module_name
        self.details = details or {}


class PaperFilterError(Query2LabelError):
    """Raised when paper filtering operations fail."""
    
    def __init__(self, message: str, query: str = None, paper_count: int = None):
        super().__init__(message)
        self.query = query
        self.paper_count = paper_count