import math
import time
import numpy as np
import cv2
from typing import List, Tuple, Optional
from cv_arm.config import Config
from cv_arm.state import RobotState
from cv_arm.utils import angle_between, align_vector_to_z, draw_pose_landmarks
from mediapipe.tasks.python import vision as mp_vision


def compute_pose_angles(lm, config: Config) -> Tuple[List[float], List[float]]:
    """Compute raw joint angles 0-3 and their confidences from a single pose landmark."""
    v = [lm[i].visibility for i in range(33)]
    sh = np.array([lm[12].x, lm[12].y, lm[12].z])
    el = np.array([lm[14].x, lm[14].y, lm[14].z])
    wr = np.array([lm[16].x, lm[16].y, lm[16].z])
    # Compute shoulder-elbow-wrist vectors
    sw = wr - sh
    se = el - sh
    ew = wr - el

    # Joint 0: shoulder flexion/extension
    j0 = 90 - math.degrees(math.atan2(-sw[0], sw[2]))
    # Joint 1: shoulder abduction/adduction
    sw_yz = min(math.degrees(math.atan2(sw[1], -sw[2])), 0.0)
    sw_se = angle_between(sw, se)
    # ensure j1 is always a float
    j1 = 0.0 if sw_yz == 0.0 else 180.0 + sw_yz + sw_se
    # Joint 2: elbow flexion
    R = align_vector_to_z(sw)
    el_rot = R @ (el - sh)
    j2 = 90 - math.degrees(math.atan2(el_rot[1], el_rot[0]))

    # Joint 3: wrist flexion
    ew_rot = R @ ew
    fore_conf = float(min(v[14], v[16]))
    if np.linalg.norm(ew_rot[:2]) > 1e-2 and fore_conf > config.confidence_threshold:
        angle_xy = math.degrees(
            math.acos(np.clip(np.dot([1, 0], ew_rot[:2] / np.linalg.norm(ew_rot[:2])), -1.0, 1.0))
        )
        if ew_rot[1] < 0:
            angle_xy = -angle_xy
        j3 = 90 - angle_xy - 40
    else:
        j3 = 90.0
    # Return raw angles and confidences
    angles = [float(j0), float(j1), float(j2), float(j3)]
    confidences = [
        float(min(v[12], v[16])),
        float(min(v[12], v[16])),
        float(min(v[12], v[14])),
        fore_conf,
    ]
    return angles, confidences


def compute_hand_metrics(hand_landmarks, config: Config) -> Tuple[List[float], List[float]]:
    """Compute raw wrist roll and hand openness angles and confidences."""
    if hand_landmarks:
        hl = hand_landmarks[0]
        # Wrist roll raw
        v1h = np.array([hl[5].x - hl[0].x, hl[5].y - hl[0].y])
        v2h = np.array([hl[17].x - hl[0].x, hl[17].y - hl[0].y])
        raw_roll = math.degrees(math.atan2(v2h[1], v2h[0]) - math.atan2(v1h[1], v1h[0]))
        j4 = 90 + raw_roll
        hand_conf = 1.0
        # Hand openness raw
        tips = [4, 8, 12, 16, 20]
        mcps = [2, 5, 9, 13, 17]
        openness = sum(
            np.linalg.norm(
                np.array([hl[tip].x, hl[tip].y]) - np.array([hl[mcp].x, hl[mcp].y])
            )
            for tip, mcp in zip(tips, mcps)
        ) / 5.0
        palm_width = np.linalg.norm(
            np.array([hl[5].x, hl[5].y]) - np.array([hl[17].x, hl[17].y])
        )
        if palm_width > 1e-6:
            openness_norm = openness / palm_width
            min_open, max_open = 0.5, 1.5
            openness_norm = np.clip(openness_norm, min_open, max_open)
            hand_openness = (openness_norm - min_open) / (max_open - min_open) * 180.0
            j5 = float(np.clip(hand_openness, 0, 180))
        else:
            j5 = 0.0
        openness_conf = 1.0
    else:
        j4 = 90.0
        j5 = 0.0
        hand_conf = 0.0
        openness_conf = 0.0
    angles = [j4, j5]
    confidences = [hand_conf, openness_conf]
    return angles, confidences


def blend_and_rate_limit(
    raw_angles: List[float],
    confidences: List[float],
    state: RobotState,
    config: Config,
    fps: int,
) -> List[int]:
    """Apply exponential smoothing and rate limiting, update state, and return sendable integer angles."""
    # Exponential smoothing
    for i in range(len(raw_angles)):
        alpha = max(0.1, config.smoothing_alpha * confidences[i])
        state.filtered_angles[i] = alpha * raw_angles[i] + (1 - alpha) * state.filtered_angles[i]
    # Rate limiting per frame
    max_per_frame = config.max_deg_per_sec / fps
    limited = []
    for i, fa in enumerate(state.filtered_angles):
        d = fa - state.last_sent_angles[i]
        d = max(-max_per_frame, min(max_per_frame, d))
        la = state.last_sent_angles[i] + d
        limited.append(int(round(la)))
    state.last_sent_angles = limited
    return limited


def render_overlay(frame, pose_landmark, hand_landmarks) -> None:
    """Draw pose and hand landmarks on the frame."""
    # Allow single pose_landmark or iterable of landmarks
    try:
        landmarks = list(pose_landmark)
    except TypeError:
        landmarks = [pose_landmark]
    draw_pose_landmarks(frame, landmarks)
    if hand_landmarks:
        for hl in hand_landmarks:
            for lm in hl:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)
