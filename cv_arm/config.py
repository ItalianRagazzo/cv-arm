from dataclasses import dataclass
from typing import List, Tuple

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
