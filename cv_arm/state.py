from dataclasses import dataclass
from typing import List

@dataclass
class RobotState:
    """State of robot control for smoothing and rate limiting."""
    prev_t: float
    filtered_angles: List[float]
    filtered_torso_len: float
    filtered_upper_len: float
    filtered_fore_len: float
    last_sent_angles: List[int]
