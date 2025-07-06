"""Tests for configuration management."""

import pytest
import tempfile
import json
import os
from pathlib import Path

from readcube_mcp.query2label.utils.config import (
    ConfigManager,
    Query2LabelConfig,
    DSPyConfig,
    LoggingConfig,
    DataConfig,
    PerformanceConfig,
    MCPConfig,
    get_config,
    load_config
)
from readcube_mcp.query2label.core.exceptions import ConfigurationError


class TestConfigDataClasses:
    """Test configuration data classes."""
    
    def test_dspy_config_defaults(self):
        """Test DSPy config defaults."""
        config = DSPyConfig()
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.0
        assert config.max_tokens == 16384
        assert config.max_retries == 3
        assert config.api_key is None
    
    def test_logging_config_defaults(self):
        """Test logging config defaults."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format_type == "simple"
        assert config.console_output is True
        assert config.output_file is None
    
    def test_data_config_defaults(self):
        """Test data config defaults."""
        config = DataConfig()
        assert config.label_delimiters == [';', ',', '|']
        assert config.normalize_case is True
        assert config.min_label_length == 2
        assert config.max_label_length == 100
    
    def test_query2label_config_to_dict(self):
        """Test converting config to dictionary."""
        config = Query2LabelConfig()
        config_dict = config.to_dict()
        
        assert 'dspy' in config_dict
        assert 'logging' in config_dict
        assert 'data' in config_dict
        assert 'performance' in config_dict
        assert 'mcp' in config_dict
    
    def test_query2label_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'dspy': {'model': 'gpt-4', 'temperature': 0.5},
            'logging': {'level': 'DEBUG'},
            'data': {'normalize_case': False}
        }
        
        config = Query2LabelConfig.from_dict(config_dict)
        assert config.dspy.model == 'gpt-4'
        assert config.dspy.temperature == 0.5
        assert config.logging.level == 'DEBUG'
        assert config.data.normalize_case is False


class TestConfigManager:
    """Test configuration manager."""
    
    def test_init_default(self):
        """Test initialization with defaults."""
        manager = ConfigManager()
        assert manager.config_file is None
        assert not manager._loaded
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        manager = ConfigManager()
        config = manager.load()
        
        assert isinstance(config, Query2LabelConfig)
        assert config.dspy.model == "gpt-4o-mini"
        assert manager._loaded is True
    
    def test_load_from_json_file(self):
        """Test loading configuration from JSON file."""
        config_data = {
            'dspy': {
                'model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 8192
            },
            'logging': {
                'level': 'DEBUG',
                'format_type': 'structured'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            
            manager = ConfigManager(f.name)
            config = manager.load()
        
        assert config.dspy.model == 'gpt-4'
        assert config.dspy.temperature == 0.7
        assert config.dspy.max_tokens == 8192
        assert config.logging.level == 'DEBUG'
        assert config.logging.format_type == 'structured'
    
    def test_load_from_env_variables(self):
        """Test loading configuration from environment variables."""
        # Set environment variables
        env_vars = {
            'QUERY2LABEL_DSPY_MODEL': 'gpt-3.5-turbo',
            'QUERY2LABEL_DSPY_TEMPERATURE': '0.8',
            'QUERY2LABEL_LOG_LEVEL': 'WARNING',
            'QUERY2LABEL_CACHE_ENABLED': 'false'
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        try:
            manager = ConfigManager()
            config = manager.load()
            
            assert config.dspy.model == 'gpt-3.5-turbo'
            assert config.dspy.temperature == 0.8
            assert config.logging.level == 'WARNING'
            assert config.data.cache_enabled is False
            
        finally:
            # Clean up environment variables
            for key in env_vars:
                os.environ.pop(key, None)
    
    def test_validation_errors(self):
        """Test configuration validation."""
        manager = ConfigManager()
        
        # Test invalid temperature
        with pytest.raises(ConfigurationError):
            manager.update_config({'dspy': {'temperature': 3.0}})
        
        # Test invalid log level
        with pytest.raises(ConfigurationError):
            manager.update_config({'logging': {'level': 'INVALID'}})
        
        # Test invalid port
        with pytest.raises(ConfigurationError):
            manager.update_config({'mcp': {'port': 70000}})
    
    def test_save_config(self):
        """Test saving configuration to file."""
        manager = ConfigManager()
        config = manager.load()
        
        # Modify config
        config.dspy.model = 'test-model'
        config.logging.level = 'ERROR'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            manager.save(f.name)
            
            # Load and verify
            with open(f.name, 'r') as read_f:
                saved_data = json.load(read_f)
        
        assert saved_data['dspy']['model'] == 'test-model'
        assert saved_data['logging']['level'] == 'ERROR'
    
    def test_update_config(self):
        """Test updating configuration."""
        manager = ConfigManager()
        config = manager.load()
        
        original_model = config.dspy.model
        
        # Update config
        manager.update_config({
            'dspy': {'model': 'updated-model'},
            'data': {'min_label_length': 5}
        })
        
        updated_config = manager.get_config()
        assert updated_config.dspy.model == 'updated-model'
        assert updated_config.data.min_label_length == 5
        
        # Other values should remain unchanged
        assert updated_config.dspy.temperature == 0.0


class TestGlobalFunctions:
    """Test global configuration functions."""
    
    def test_get_config(self):
        """Test getting global configuration."""
        config = get_config()
        assert isinstance(config, Query2LabelConfig)
    
    def test_load_config_with_env(self):
        """Test loading config with environment variable."""
        os.environ['QUERY2LABEL_DSPY_MODEL'] = 'env-test-model'
        
        try:
            config = load_config(reload=True)
            assert config.dspy.model == 'env-test-model'
        finally:
            os.environ.pop('QUERY2LABEL_DSPY_MODEL', None)


class TestConfigIntegration:
    """Integration tests for configuration."""
    
    def test_real_world_config(self):
        """Test a realistic configuration setup."""
        config_data = {
            'dspy': {
                'model': 'gpt-4o-mini',
                'temperature': 0.0,
                'max_tokens': 16384,
                'max_retries': 3
            },
            'logging': {
                'level': 'INFO',
                'format_type': 'simple',
                'console_output': True
            },
            'data': {
                'label_delimiters': [';', ',', '|'],
                'normalize_case': True,
                'cache_enabled': True
            },
            'performance': {
                'enable_metrics': True,
                'cache_size': 1000,
                'batch_size': 100
            },
            'mcp': {
                'server_name': 'readcube-mcp',
                'host': 'localhost',
                'port': 8000
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            
            config = load_config(f.name, reload=True)
        
        # Verify all sections loaded correctly
        assert config.dspy.model == 'gpt-4o-mini'
        assert config.logging.level == 'INFO'
        assert config.data.normalize_case is True
        assert config.performance.enable_metrics is True
        assert config.mcp.server_name == 'readcube-mcp'
    
    def test_partial_config_file(self):
        """Test loading partial configuration file."""
        # Only specify some config sections
        partial_config = {
            'dspy': {'model': 'custom-model'},
            'logging': {'level': 'DEBUG'}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(partial_config, f)
            f.flush()
            
            config = load_config(f.name, reload=True)
        
        # Specified values should be set
        assert config.dspy.model == 'custom-model'
        assert config.logging.level == 'DEBUG'
        
        # Unspecified values should use defaults
        assert config.dspy.temperature == 0.0  # default
        assert config.data.normalize_case is True  # default