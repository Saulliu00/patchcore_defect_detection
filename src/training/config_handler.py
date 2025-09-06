# src/training/config_handler.py
import yaml
from pathlib import Path
from omegaconf import OmegaConf

class ConfigHandler:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
    def load_config(self):
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return OmegaConf.create(config_dict)
        
    def get_config(self):
        """Get the loaded configuration"""
        return self.config
        
    def update_config(self, updates):
        """Update configuration with new values"""
        self.config = OmegaConf.merge(self.config, updates)
        
    def save_config(self, path=None):
        """Save configuration to file"""
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(OmegaConf.to_yaml(self.config), f, default_flow_style=False)

