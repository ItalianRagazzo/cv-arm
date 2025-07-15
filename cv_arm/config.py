from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
import yaml
import logging

@dataclass
class Config:
    """Centralized configuration for CV to arm control parameters."""
    smoothing_alpha: float = 0.3
    confidence_threshold: float = 0.9
    max_deg_per_sec: float = 360.0
    upper_ratio: float = 0.65
    fore_ratio: float = 0.55
    neutral_angles: List[int] = (90, 90, 90, 90, 90, 90)
    # Rate limiting config
    hand_openness_window: float = 0.2  # range over which confidence blends openness
    confidence_blend_window: float = 0.2  # range over which joint confidence blends angles

def load_config(config_path: str = None) -> Config:
    """Load configuration from YAML file if it exists, else use defaults."""
    # Determine config file location
    base = Path(__file__).parent
    path = Path(config_path) if config_path else base / 'config.yaml'
    if path.is_file():
        try:
            data = yaml.safe_load(path.read_text()) or {}
            return Config(**data)
        except Exception as e:
            logging.warning("Failed to load config from %s: %s", path, e)
            return Config()
    else:
        return Config()
