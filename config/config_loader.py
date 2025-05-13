"""Configuration loader module."""

import yaml
import os
import torch


class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_environment()
        self._validate_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_environment(self):
        """Set up environment variables from config."""
        os.environ["HF_TOKEN"] = self.config['api_keys']['huggingface_token']
        os.environ["ROBOFLOW_API_KEY"] = self.config['api_keys']['roboflow_api_key']
        
        # Configure CUDA if available
        if self.config['performance']['use_gpu'] and torch.cuda.is_available():
            os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"
        else:
            self.config['performance']['device'] = 'cpu'
            print("CUDA not available, falling back to CPU")
    
    def _validate_config(self):
        """Validate required configuration fields."""
        required_fields = [
            'api_keys.huggingface_token',
            'api_keys.roboflow_api_key',
            'models.player_detection_model_id',
            'models.field_detection_model_id',
            'video.input_path'
        ]
        
        for field in required_fields:
            keys = field.split('.')
            value = self.config
            for key in keys:
                if key not in value:
                    raise ValueError(f"Missing required config field: {field}")
                value = value[key]
    
    def get(self, key, default=None):
        """Get configuration value by dot notation key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
