import pytest
import os
import yaml
from local_agent import LlamaAgent

@pytest.fixture
def test_agent(test_dir):
    """Create a test agent with a temporary directory structure"""
    return LlamaAgent(str(test_dir))

def test_default_config_creation(test_agent):
    """Test that default configuration is created correctly"""
    config = test_agent.config
    
    assert 'model' in config
    assert config['model']['name'] == 'llama3.2'
    assert config['model']['temperature'] == 0.7
    
    assert 'paths' in config
    assert '.obsidian/*' in config['paths']['exclude']
    assert '.md' in config['paths']['watch_extensions']

def test_config_file_creation(tmp_path):
    """Test that config file is created on disk"""
    (tmp_path / "docs").mkdir()
    (tmp_path / ".llama").mkdir()
    
    # Create agent to trigger config creation
    agent = LlamaAgent(str(tmp_path))
    
    # Check config file exists
    config_path = tmp_path / ".llama" / "config.yaml"
    assert config_path.exists()
    
    # Verify config content
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        assert config['model']['name'] == 'llama3.2'

def test_custom_config_loading(tmp_path):
    """Test loading custom configuration"""
    # Create custom config
    config_dir = tmp_path / ".llama"
    config_dir.mkdir()
    (tmp_path / "docs").mkdir()
    
    custom_config = {
        'model': {
            'name': 'custom-model',
            'temperature': 0.5
        },
        'paths': {
            'exclude': ['custom/*'],
            'watch_extensions': ['.md']
        },
        'editing': {
            'backup': False
        }
    }
    
    with open(config_dir / "config.yaml", 'w') as f:
        yaml.dump(custom_config, f)
    
    # Create agent and verify config
    agent = LlamaAgent(str(tmp_path))
    assert agent.config['model']['name'] == 'custom-model'
    assert agent.config['model']['temperature'] == 0.5
    assert agent.config['paths']['exclude'] == ['custom/*']