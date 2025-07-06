"""Logging configuration for Query2Label system.

This module provides structured logging configuration with support for different
log levels, formats, and output destinations. It includes specialized loggers
for different components of the system.
"""

import logging
import logging.config
import sys
import json
from typing import Optional, Dict, Any, Union
from pathlib import Path
import os
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""
    
    def __init__(self, include_extra: bool = True):
        """Initialize the structured formatter.
        
        Args:
            include_extra: Whether to include extra fields in log records
        """
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON formatted log string
        """
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if requested
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                              'pathname', 'filename', 'module', 'exc_info', 
                              'exc_text', 'stack_info', 'lineno', 'funcName', 
                              'created', 'msecs', 'relativeCreated', 'thread', 
                              'threadName', 'processName', 'process', 'getMessage']:
                    log_data[key] = value
        
        return json.dumps(log_data)


class Query2LabelLogger:
    """Centralized logging configuration for Query2Label system."""
    
    def __init__(self):
        """Initialize the logging system."""
        self._configured = False
        self._loggers = {}
    
    def configure(self, 
                  level: Union[str, int] = logging.INFO,
                  format_type: str = "structured",
                  output_file: Optional[str] = None,
                  console_output: bool = True,
                  component_levels: Optional[Dict[str, Union[str, int]]] = None) -> None:
        """Configure the logging system.
        
        Args:
            level: Default logging level
            format_type: Log format type ("structured", "simple", "detailed")
            output_file: Optional file path for log output
            console_output: Whether to output logs to console
            component_levels: Optional component-specific log levels
        """
        if self._configured:
            return
        
        # Convert string level to logging constant
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        # Create root logger configuration
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': self._get_formatters(format_type),
            'handlers': self._get_handlers(output_file, console_output),
            'loggers': self._get_logger_config(level, component_levels),
            'root': {
                'level': level,
                'handlers': ['console'] if console_output else []
            }
        }
        
        # Add file handler to root if specified
        if output_file:
            logging_config['root']['handlers'].append('file')
        
        logging.config.dictConfig(logging_config)
        self._configured = True
    
    def _get_formatters(self, format_type: str) -> Dict[str, Dict[str, Any]]:
        """Get formatter configurations.
        
        Args:
            format_type: Type of format to use
            
        Returns:
            Dictionary of formatter configurations
        """
        formatters = {
            'structured': {
                '()': StructuredFormatter,
                'include_extra': True
            },
            'simple': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
            }
        }
        
        return {format_type: formatters.get(format_type, formatters['simple'])}
    
    def _get_handlers(self, output_file: Optional[str], console_output: bool) -> Dict[str, Dict[str, Any]]:
        """Get handler configurations.
        
        Args:
            output_file: Optional file path for log output
            console_output: Whether to include console output
            
        Returns:
            Dictionary of handler configurations
        """
        handlers = {}
        
        if console_output:
            handlers['console'] = {
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',
                'formatter': list(self._get_formatters('simple').keys())[0]
            }
        
        if output_file:
            # Ensure log directory exists
            log_path = Path(output_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            handlers['file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': output_file,
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'formatter': list(self._get_formatters('structured').keys())[0]
            }
        
        return handlers
    
    def _get_logger_config(self, 
                          default_level: int, 
                          component_levels: Optional[Dict[str, Union[str, int]]]) -> Dict[str, Dict[str, Any]]:
        """Get logger configurations for different components.
        
        Args:
            default_level: Default logging level
            component_levels: Optional component-specific levels
            
        Returns:
            Dictionary of logger configurations
        """
        component_levels = component_levels or {}
        
        # Convert string levels to logging constants
        for component, level in component_levels.items():
            if isinstance(level, str):
                component_levels[component] = getattr(logging, level.upper())
        
        loggers = {}
        
        # Query2Label component loggers
        components = [
            'query2label.core',
            'query2label.dspy_modules', 
            'query2label.data',
            'query2label.utils',
            'query2label.translators',
            'query2label.filters',
            'query2label.loaders',
            'query2label.processors'
        ]
        
        for component in components:
            loggers[component] = {
                'level': component_levels.get(component, default_level),
                'handlers': ['console'],
                'propagate': False
            }
        
        return loggers
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance for a specific component.
        
        Args:
            name: Logger name (typically module name)
            
        Returns:
            Logger instance
        """
        if not self._configured:
            self.configure()
        
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        
        return self._loggers[name]
    
    def log_performance(self, 
                       logger: logging.Logger, 
                       operation: str, 
                       duration: float, 
                       **kwargs) -> None:
        """Log performance metrics.
        
        Args:
            logger: Logger instance to use
            operation: Name of the operation
            duration: Duration in seconds
            **kwargs: Additional context data
        """
        logger.info(
            f"Performance: {operation} completed",
            extra={
                'operation': operation,
                'duration_seconds': duration,
                'performance_metric': True,
                **kwargs
            }
        )
    
    def log_query_translation(self, 
                             logger: logging.Logger, 
                             query: str, 
                             result: Any, 
                             duration: float,
                             **kwargs) -> None:
        """Log query translation events.
        
        Args:
            logger: Logger instance to use
            query: Original query string
            result: Translation result
            duration: Processing duration
            **kwargs: Additional context data
        """
        logger.info(
            f"Query translated: {query[:100]}...",
            extra={
                'query_text': query,
                'result_type': type(result).__name__,
                'duration_seconds': duration,
                'query_translation': True,
                **kwargs
            }
        )
    
    def log_data_processing(self, 
                           logger: logging.Logger, 
                           operation: str, 
                           input_count: int, 
                           output_count: int,
                           **kwargs) -> None:
        """Log data processing events.
        
        Args:
            logger: Logger instance to use
            operation: Type of data processing
            input_count: Number of input items
            output_count: Number of output items
            **kwargs: Additional context data
        """
        logger.info(
            f"Data processing: {operation}",
            extra={
                'operation': operation,
                'input_count': input_count,
                'output_count': output_count,
                'data_processing': True,
                **kwargs
            }
        )


# Global logger instance
_logger_instance = Query2LabelLogger()


def configure_logging(**kwargs) -> None:
    """Configure the global logging system.
    
    Args:
        **kwargs: Configuration parameters for Query2LabelLogger.configure()
    """
    _logger_instance.configure(**kwargs)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return _logger_instance.get_logger(name)


def log_performance(operation: str, duration: float, logger_name: str = "query2label", **kwargs) -> None:
    """Log performance metrics using the global logger.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        logger_name: Logger name to use
        **kwargs: Additional context data
    """
    logger = get_logger(logger_name)
    _logger_instance.log_performance(logger, operation, duration, **kwargs)


def log_query_translation(query: str, result: Any, duration: float, logger_name: str = "query2label.translators", **kwargs) -> None:
    """Log query translation using the global logger.
    
    Args:
        query: Original query string
        result: Translation result
        duration: Processing duration
        logger_name: Logger name to use
        **kwargs: Additional context data
    """
    logger = get_logger(logger_name)
    _logger_instance.log_query_translation(logger, query, result, duration, **kwargs)


def log_data_processing(operation: str, input_count: int, output_count: int, logger_name: str = "query2label.data", **kwargs) -> None:
    """Log data processing using the global logger.
    
    Args:
        operation: Type of data processing
        input_count: Number of input items
        output_count: Number of output items
        logger_name: Logger name to use
        **kwargs: Additional context data
    """
    logger = get_logger(logger_name)
    _logger_instance.log_data_processing(logger, operation, input_count, output_count, **kwargs)


# Configure logging from environment variables on import
def _configure_from_env():
    """Configure logging from environment variables."""
    log_level = os.getenv('QUERY2LABEL_LOG_LEVEL', 'INFO')
    log_format = os.getenv('QUERY2LABEL_LOG_FORMAT', 'simple')
    log_file = os.getenv('QUERY2LABEL_LOG_FILE')
    console_output = os.getenv('QUERY2LABEL_LOG_CONSOLE', 'true').lower() == 'true'
    
    configure_logging(
        level=log_level,
        format_type=log_format,
        output_file=log_file,
        console_output=console_output
    )


# Auto-configure on import if not in test environment
if 'pytest' not in sys.modules:
    _configure_from_env()