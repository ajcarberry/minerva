from dataclasses import dataclass
from typing import List, Dict
import os
import yaml

# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class AgentModelConfig:
    """Configuration for the LLM model"""
    name: str
    context_window: int
    temperature: float
    host: str = "http://localhost:11435"

@dataclass
class EmbeddingModelConfig:
    """Configuration for the LLM model"""
    name: str
    context_window: int
    temperature: float
    host: str = "http://localhost:11435"

@dataclass
class PathConfig:
    """Configuration for file paths and extensions"""
    exclude: List[str]
    watch_extensions: List[str]

@dataclass
class EditingConfig:
    """Configuration for document editing behavior"""
    backup: bool
    backup_dir: str
    require_approval: bool

class ConfigManager:
    """Handles configuration loading and management"""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.config_path = os.path.join(project_path, '.llama', 'config.yaml')
        self.load_config()

    def load_config(self) -> None:
        """Load configuration from file or create default"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        except FileNotFoundError:
            config_data = self._create_default_config()
            
        self.agent_config = AgentModelConfig(**config_data['agent_model'])
        self.embedding_config = EmbeddingModelConfig(**config_data['embedding_model'])
        self.path_config = PathConfig(**config_data['paths'])
        self.editing_config = EditingConfig(**config_data['editing'])

    def _create_default_config(self) -> Dict:
        """Create and save default configuration"""
        config = {
            'agent_model': {
                'name': 'llama3.2',
                'context_window': 4096,
                'temperature': 0.7
            },
            'embedding_model': {
                'name': 'nomic-embed-text',
                'context_window': 4096,
                'temperature': 0.7
            },
            'paths': {
                'exclude': ['.obsidian/*', '.git/*'],
                'watch_extensions': ['.md']
            },
            'editing': {
                'backup': True,
                'backup_dir': '.llama/backups',
                'require_approval': True
            }
        }
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return config