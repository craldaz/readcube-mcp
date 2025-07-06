"""Configuration management for Query2Label system.

This module provides a centralized configuration system with support for
environment variables, configuration files, and runtime settings.
"""

import os
import json
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field, asdict
from ..core.exceptions import ConfigurationError


@dataclass
class DSPyConfig:
    """DSPy module configuration."""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 16384
    max_retries: int = 3
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format_type: str = "simple"
    output_file: Optional[str] = None
    console_output: bool = True
    component_levels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DataConfig:
    """Data processing configuration."""
    label_delimiters: List[str] = field(default_factory=lambda: [';', ',', '|'])
    normalize_case: bool = True
    remove_special_chars: bool = True
    min_label_length: int = 2
    max_label_length: int = 100
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PerformanceConfig:
    """Performance and monitoring configuration."""
    enable_metrics: bool = True
    enable_timing: bool = True
    cache_size: int = 1000
    batch_size: int = 100
    max_workers: int = 4
    timeout_seconds: int = 300
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MCPConfig:
    """MCP server configuration."""
    server_name: str = "readcube-mcp"
    server_version: str = "0.1.0"
    host: str = "localhost"
    port: int = 8000
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_request_size: int = 10485760  # 10MB
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Query2LabelConfig:
    """Main configuration for Query2Label system."""
    dspy: DSPyConfig = field(default_factory=DSPyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Query2LabelConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        if 'dspy' in config_dict:
            config.dspy = DSPyConfig(**config_dict['dspy'])
        
        if 'logging' in config_dict:
            config.logging = LoggingConfig(**config_dict['logging'])
        
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        
        if 'performance' in config_dict:
            config.performance = PerformanceConfig(**config_dict['performance'])
        
        if 'mcp' in config_dict:
            config.mcp = MCPConfig(**config_dict['mcp'])
        
        return config


class ConfigManager:
    """Configuration manager for Query2Label system."""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config_file = Path(config_file) if config_file else None
        self._config = Query2LabelConfig()
        self._loaded = False
    
    def load(self, reload: bool = False) -> Query2LabelConfig:
        """Load configuration from various sources.
        
        Args:
            reload: Whether to reload configuration if already loaded
            
        Returns:
            Loaded configuration
        """
        if self._loaded and not reload:
            return self._config
        
        # Start with default configuration
        self._config = Query2LabelConfig()
        
        # Load from file if specified
        if self.config_file and self.config_file.exists():
            self._load_from_file()
        
        # Override with environment variables
        self._load_from_env()
        
        # Validate configuration
        self._validate_config()
        
        self._loaded = True
        return self._config
    
    def _load_from_file(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                if self.config_file.suffix.lower() == '.json':
                    config_dict = json.load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported config file format: {self.config_file.suffix}",
                        config_key=str(self.config_file),
                        expected_type=str
                    )
            
            self._config = Query2LabelConfig.from_dict(config_dict)
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration from {self.config_file}: {e}",
                config_key=str(self.config_file),
                expected_type=dict
            )
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # DSPy configuration
        if model := os.getenv('QUERY2LABEL_DSPY_MODEL'):
            self._config.dspy.model = model
        
        if temp := os.getenv('QUERY2LABEL_DSPY_TEMPERATURE'):
            try:
                self._config.dspy.temperature = float(temp)
            except ValueError:
                pass
        
        if tokens := os.getenv('QUERY2LABEL_DSPY_MAX_TOKENS'):
            try:
                self._config.dspy.max_tokens = int(tokens)
            except ValueError:
                pass
        
        if retries := os.getenv('QUERY2LABEL_DSPY_MAX_RETRIES'):
            try:
                self._config.dspy.max_retries = int(retries)
            except ValueError:
                pass
        
        if api_key := os.getenv('OPENAI_API_KEY'):
            self._config.dspy.api_key = api_key
        
        if api_base := os.getenv('OPENAI_API_BASE'):
            self._config.dspy.api_base = api_base
        
        # Logging configuration
        if log_level := os.getenv('QUERY2LABEL_LOG_LEVEL'):
            self._config.logging.level = log_level
        
        if log_format := os.getenv('QUERY2LABEL_LOG_FORMAT'):
            self._config.logging.format_type = log_format
        
        if log_file := os.getenv('QUERY2LABEL_LOG_FILE'):
            self._config.logging.output_file = log_file
        
        if console := os.getenv('QUERY2LABEL_LOG_CONSOLE'):
            self._config.logging.console_output = console.lower() == 'true'
        
        # Data configuration
        if delimiters := os.getenv('QUERY2LABEL_LABEL_DELIMITERS'):
            self._config.data.label_delimiters = delimiters.split(',')
        
        if normalize := os.getenv('QUERY2LABEL_NORMALIZE_CASE'):
            self._config.data.normalize_case = normalize.lower() == 'true'
        
        if cache_enabled := os.getenv('QUERY2LABEL_CACHE_ENABLED'):
            self._config.data.cache_enabled = cache_enabled.lower() == 'true'
        
        if cache_ttl := os.getenv('QUERY2LABEL_CACHE_TTL'):
            try:
                self._config.data.cache_ttl_seconds = int(cache_ttl)
            except ValueError:
                pass
        
        # Performance configuration
        if metrics := os.getenv('QUERY2LABEL_ENABLE_METRICS'):
            self._config.performance.enable_metrics = metrics.lower() == 'true'
        
        if timing := os.getenv('QUERY2LABEL_ENABLE_TIMING'):
            self._config.performance.enable_timing = timing.lower() == 'true'
        
        if cache_size := os.getenv('QUERY2LABEL_CACHE_SIZE'):
            try:
                self._config.performance.cache_size = int(cache_size)
            except ValueError:
                pass
        
        if batch_size := os.getenv('QUERY2LABEL_BATCH_SIZE'):
            try:
                self._config.performance.batch_size = int(batch_size)
            except ValueError:
                pass
        
        # MCP configuration
        if server_name := os.getenv('QUERY2LABEL_MCP_SERVER_NAME'):
            self._config.mcp.server_name = server_name
        
        if host := os.getenv('QUERY2LABEL_MCP_HOST'):
            self._config.mcp.host = host
        
        if port := os.getenv('QUERY2LABEL_MCP_PORT'):
            try:
                self._config.mcp.port = int(port)
            except ValueError:
                pass
        
        if cors := os.getenv('QUERY2LABEL_MCP_ENABLE_CORS'):
            self._config.mcp.enable_cors = cors.lower() == 'true'
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Validate DSPy config
        if self._config.dspy.temperature < 0 or self._config.dspy.temperature > 2:
            raise ConfigurationError(
                "DSPy temperature must be between 0 and 2",
                config_key="dspy.temperature",
                expected_type=float
            )
        
        if self._config.dspy.max_tokens <= 0:
            raise ConfigurationError(
                "DSPy max_tokens must be positive",
                config_key="dspy.max_tokens",
                expected_type=int
            )
        
        if self._config.dspy.max_retries < 0:
            raise ConfigurationError(
                "DSPy max_retries must be non-negative",
                config_key="dspy.max_retries",
                expected_type=int
            )
        
        # Validate logging config
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self._config.logging.level.upper() not in valid_log_levels:
            raise ConfigurationError(
                f"Invalid log level: {self._config.logging.level}",
                config_key="logging.level",
                expected_type=str
            )
        
        valid_formats = ['simple', 'detailed', 'structured']
        if self._config.logging.format_type not in valid_formats:
            raise ConfigurationError(
                f"Invalid log format: {self._config.logging.format_type}",
                config_key="logging.format_type",
                expected_type=str
            )
        
        # Validate data config
        if self._config.data.min_label_length < 1:
            raise ConfigurationError(
                "Minimum label length must be at least 1",
                config_key="data.min_label_length",
                expected_type=int
            )
        
        if self._config.data.max_label_length <= self._config.data.min_label_length:
            raise ConfigurationError(
                "Maximum label length must be greater than minimum",
                config_key="data.max_label_length",
                expected_type=int
            )
        
        # Validate performance config
        if self._config.performance.cache_size <= 0:
            raise ConfigurationError(
                "Cache size must be positive",
                config_key="performance.cache_size",
                expected_type=int
            )
        
        if self._config.performance.batch_size <= 0:
            raise ConfigurationError(
                "Batch size must be positive",
                config_key="performance.batch_size",
                expected_type=int
            )
        
        # Validate MCP config
        if not (1 <= self._config.mcp.port <= 65535):
            raise ConfigurationError(
                "MCP port must be between 1 and 65535",
                config_key="mcp.port",
                expected_type=int
            )
    
    def save(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file.
        
        Args:
            file_path: Optional path to save to (defaults to config_file)
        """
        if not file_path:
            file_path = self.config_file
        
        if not file_path:
            raise ConfigurationError(
                "No file path specified for saving configuration",
                config_key="file_path",
                expected_type=str
            )
        
        file_path = Path(file_path)
        
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self._config.to_dict(), f, indent=2)
                
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration to {file_path}: {e}",
                config_key=str(file_path),
                expected_type=dict
            )
    
    def get_config(self) -> Query2LabelConfig:
        """Get current configuration.
        
        Returns:
            Current configuration
        """
        if not self._loaded:
            self.load()
        
        return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        if not self._loaded:
            self.load()
        
        # Apply updates
        config_dict = self._config.to_dict()
        self._deep_update(config_dict, updates)
        
        # Recreate config and validate
        self._config = Query2LabelConfig.from_dict(config_dict)
        self._validate_config()
    
    def _deep_update(self, base_dict: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Deep update dictionary with nested values.
        
        Args:
            base_dict: Base dictionary to update
            updates: Updates to apply
        """
        for key, value in updates.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


# Global configuration manager instance
_config_manager = None


def get_config_manager(config_file: Optional[Union[str, Path]] = None) -> ConfigManager:
    """Get global configuration manager instance.
    
    Args:
        config_file: Optional configuration file path
        
    Returns:
        Configuration manager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    
    return _config_manager


def get_config() -> Query2LabelConfig:
    """Get current configuration.
    
    Returns:
        Current configuration
    """
    return get_config_manager().get_config()


def load_config(config_file: Optional[Union[str, Path]] = None, reload: bool = False) -> Query2LabelConfig:
    """Load configuration from file and environment.
    
    Args:
        config_file: Optional configuration file path
        reload: Whether to reload if already loaded
        
    Returns:
        Loaded configuration
    """
    global _config_manager
    
    if _config_manager is None or config_file:
        _config_manager = ConfigManager(config_file)
    
    return _config_manager.load(reload=reload)


def save_config(file_path: Union[str, Path]) -> None:
    """Save current configuration to file.
    
    Args:
        file_path: Path to save configuration
    """
    get_config_manager().save(file_path)


def update_config(updates: Dict[str, Any]) -> None:
    """Update current configuration.
    
    Args:
        updates: Configuration updates
    """
    get_config_manager().update_config(updates)


# Auto-load configuration on import
def _auto_load_config():
    """Auto-load configuration from default locations."""
    # Look for config file in common locations
    config_paths = [
        Path.cwd() / "query2label.json",
        Path.cwd() / "config" / "query2label.json",
        Path.home() / ".query2label.json",
        Path("/etc/query2label/config.json")
    ]
    
    config_file = None
    for path in config_paths:
        if path.exists():
            config_file = path
            break
    
    # Also check environment variable
    if env_config := os.getenv('QUERY2LABEL_CONFIG_FILE'):
        env_path = Path(env_config)
        if env_path.exists():
            config_file = env_path
    
    try:
        load_config(config_file)
    except Exception:
        # Fall back to default configuration
        load_config()


# Auto-load on import if not in test environment
import sys
if 'pytest' not in sys.modules:
    _auto_load_config()