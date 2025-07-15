"""
Utility functions for cv-arm package.
"""

import math
import numpy as np
import cv2


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the angle in degrees between two 3D vectors.
    """
    v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    return math.degrees(math.acos(np.clip(v1 @ v2, -1.0, 1.0)))


def align_vector_to_z(v: np.ndarray) -> np.ndarray:
    """
    Compute a rotation matrix that aligns vector v to the Z axis.
    """
    v = v / (np.linalg.norm(v) + 1e-8)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(v, z)
    angle = math.acos(np.clip(np.dot(v, z), -1.0, 1.0))
    if np.linalg.norm(axis) < 1e-8:
        return np.eye(3)
    axis = axis / np.linalg.norm(axis)
    # Rodrigues' rotation formula
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    return R


def draw_pose_landmarks(image: np.ndarray, landmarks) -> None:
    """
    Draw pose landmarks and connections on the image using OpenCV.
    """
    h, w = image.shape[:2]
    # Draw landmarks as small circles
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
    # Key connections
    connections = [
        (11, 12), (12, 14), (14, 16),
        (11, 13), (13, 15), (15, 17),
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
        (25, 27), (26, 28)
    ]
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            s = landmarks[start_idx]
            e = landmarks[end_idx]
            pt1 = (int(s.x * w), int(s.y * h))
            pt2 = (int(e.x * w), int(e.y * h))
            cv2.line(image, pt1, pt2, (255, 255, 255), 2)
