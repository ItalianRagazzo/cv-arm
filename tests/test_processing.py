import pytest
import numpy as np
import cv2
from cv_arm.processing import (
    compute_pose_angles,
    compute_hand_metrics,
    blend_and_rate_limit,
    render_overlay,
)
from cv_arm.config import Config
from cv_arm.state import RobotState

class DummyLandmark:
    def __init__(self, x=0, y=0, z=0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

def make_dummy_pose_landmarks():
    # Create 33 landmarks with default values
    lm = [DummyLandmark() for _ in range(33)]
    # Set shoulder (12), elbow (14), wrist (16) in a straight horizontal line
    lm[12] = DummyLandmark(0.0, 0.0, 0.0, visibility=1.0)
    lm[14] = DummyLandmark(1.0, 0.0, 0.0, visibility=1.0)
    lm[16] = DummyLandmark(2.0, 0.0, 0.0, visibility=1.0)
    return lm

def make_dummy_hand_landmarks():
    # Single hand with at least indices 0,5,17
    hlm = [DummyLandmark() for _ in range(21)]
    hlm[0] = DummyLandmark(0.0, 0.0, 0.0)
    hlm[5] = DummyLandmark(0.0, 1.0, 0.0)
    hlm[17] = DummyLandmark(1.0, 1.0, 0.0)
    return [hlm]

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def state():
    return RobotState(
        prev_t=0.0,
        filtered_angles=[90.0]*6,
        filtered_torso_len=0.5,
        filtered_upper_len=0.4,
        filtered_fore_len=0.4,
        last_sent_angles=[90]*6,
    )


def test_compute_pose_angles_basic(config):
    lm = make_dummy_pose_landmarks()
    angles, confs = compute_pose_angles(lm, config)
    assert isinstance(angles, list) and len(angles) == 4
    assert isinstance(confs, list) and len(confs) == 4
    # Angles should be numeric
    for a in angles:
        assert isinstance(a, float)
    for c in confs:
        assert 0.0 <= c <= 1.0


def test_compute_hand_metrics_empty(config):
    angles, confs = compute_hand_metrics([], config)
    assert angles == [90.0, 0.0]
    assert confs == [0.0, 0.0]


def test_blend_and_rate_limit_defaults(state, config):
    raw_angles = [90.0]*6
    confs = [1.0]*6
    fps = 30
    result = blend_and_rate_limit(raw_angles, confs, state, config, fps)
    # Since raw == filtered, rate limiting should yield last_sent_angles
    assert result == state.last_sent_angles


def test_render_overlay_no_hand():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    # Dummy pose landmark with minimal attributes
    class PLP:
        x, y = 0.5, 0.5
    # Monkey-patch draw_pose_landmarks to avoid import issues
    from cv_arm.utils import draw_pose_landmarks as dpl
    # Call render_overlay with no exception
    render_overlay(frame, PLP, None)
    # Frame should remain same shape
    assert frame.shape == (10, 10, 3)
