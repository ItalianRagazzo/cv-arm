#!/usr/bin/env python3
"""
CV Arm Control (RTMPose edition)
Webcam → RTMPose WholeBody → body-relative 2D angles → DOFBOT

Key improvements over cv_arm.py:
- RTMPose (GPU) for better joint accuracy
- Body-relative coordinate frame: invariant to camera distance and position in frame
- Z/depth entirely ignored — 2D only
- One Euro Filter: adaptive smoothing (low lag on fast motion, stable at rest)
- Holds last position on low confidence instead of snapping to neutral
"""

import time
import struct
import math
import sys
import serial
import cv2
import numpy as np
from typing import Optional, List

# ─────────────────────────── Configuration ───────────────────────────────────

DEFAULT_COM  = 'COM5'
DEFAULT_BAUD = 2_000_000
DEFAULT_FPS  = 30

CONFIDENCE_THRESHOLD = 0.5   # min RTMPose score for any body keypoint
MIN_SHOULDER_WIDTH   = 0.05  # normalised image units; reject if person too small/far

MAX_DEG_PER_SEC = 180.0      # safety rate limit

# One Euro Filter — see: Casiez et al., CHI 2012
OEF_MIN_CUTOFF = 1.0   # Hz: lower = smoother at rest (raise if too laggy at rest)
OEF_BETA       = 0.05  # speed coeff: raise if fast motions still feel laggy
OEF_D_CUTOFF   = 1.0   # Hz: derivative filter cutoff (rarely needs tuning)

# Body-relative mapping scales.  Wrist ±1.5× shoulder-widths maps to ±RANGE degrees.
# Increase a range if the robot arm doesn't reach the extremes you move to.
J0_RANGE_DEG = 90.0   # base pan (left/right)
J1_RANGE_DEG = 90.0   # shoulder elevation (up/down)
J3_RANGE_DEG = 60.0   # wrist/forearm flexion

# RTMPose keypoint indices (COCO-WholeBody)
IDX_L_SHOULDER = 5
IDX_R_SHOULDER = 6
IDX_L_ELBOW    = 7
IDX_R_ELBOW    = 8
IDX_L_WRIST    = 9
IDX_R_WRIST    = 10
# Hand keypoints (for future J4/J5): left=92-112, right=113-133 (21 pts each)


# ─────────────────────────── One Euro Filter ──────────────────────────────────

class OneEuroFilter:
    """Adaptive low-pass filter for a single scalar signal.

    At slow speeds, uses a low cutoff frequency to remove jitter.
    At fast speeds, raises the cutoff so the filtered signal tracks quickly.
    """

    def __init__(self, min_cutoff: float = OEF_MIN_CUTOFF,
                 beta: float = OEF_BETA, d_cutoff: float = OEF_D_CUTOFF):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return dt / (dt + tau)

    def filter(self, x: float, dt: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx_raw        = (x - self.x_prev) / dt
        alpha_d       = self._alpha(self.d_cutoff, dt)
        self.dx_prev  = alpha_d * dx_raw + (1.0 - alpha_d) * self.dx_prev
        cutoff        = self.min_cutoff + self.beta * abs(self.dx_prev)
        alpha         = self._alpha(cutoff, dt)
        self.x_prev   = alpha * x + (1.0 - alpha) * self.x_prev
        return self.x_prev

    def reset(self, x: Optional[float] = None) -> None:
        self.x_prev  = x
        self.dx_prev = 0.0


# ─────────────────────────── Arm Controller ───────────────────────────────────

class ArmController:
    def __init__(self, com_port: Optional[str] = None, baud: int = DEFAULT_BAUD):
        self.ser  = None
        self.pack = struct.Struct(">H6B")

        self.oef_filters = [OneEuroFilter() for _ in range(6)]
        for f in self.oef_filters:
            f.reset(90.0)

        self.last_good_angles: List[float] = [90.0] * 6
        self.last_sent_angles: List[int]   = [90] * 6
        self.prev_t = time.time()

        if com_port:
            try:
                self.ser = serial.Serial(com_port, baud, timeout=0)
                print(f"Connected to {com_port}")
            except Exception as e:
                print(f"Serial error: {e}")

    def send_angles(self, angles: List[int], duration_ms: int = 33) -> None:
        if self.ser:
            try:
                self.ser.write(self.pack.pack(duration_ms, *angles))
            except Exception as e:
                print(f"Serial write failed: {e}")

    def close(self) -> None:
        if self.ser:
            self.ser.close()


# ─────────────────────────── Geometry helpers ─────────────────────────────────

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two vectors (2D or 3D)."""
    v1n = v1 / (np.linalg.norm(v1) + 1e-6)
    v2n = v2 / (np.linalg.norm(v2) + 1e-6)
    return math.degrees(math.acos(float(np.clip(v1n @ v2n, -1.0, 1.0))))


# ─────────────────────────── Angle computation ────────────────────────────────

def compute_arm_angles(
    keypoints: np.ndarray,
    scores: np.ndarray,
    frame_shape: tuple,
) -> Optional[List[float]]:
    """Convert RTMPose keypoints to DOFBOT joint angles.

    Args:
        keypoints: (N, 133, 2) pixel coordinates from RTMPose, first person used.
        scores:    (N, 133) confidence scores.
        frame_shape: (H, W, ...) — used to normalise pixel coords to [0,1].

    Returns:
        List of 6 float angles [0,180], or None if confidence too low.
    """
    if keypoints.shape[0] == 0:
        return None

    kp = keypoints[0]    # (133, 2)  pixel coords
    sc = scores[0]       # (133,)

    # Confidence gate on the four body joints we actually use
    body_indices = [IDX_L_SHOULDER, IDX_R_SHOULDER, IDX_R_ELBOW, IDX_R_WRIST]
    if min(sc[i] for i in body_indices) < CONFIDENCE_THRESHOLD:
        return None

    H, W = frame_shape[:2]

    # Normalise to [0,1] image space
    def norm(idx: int) -> np.ndarray:
        return np.array([kp[idx, 0] / W, kp[idx, 1] / H])

    ls = norm(IDX_L_SHOULDER)
    rs = norm(IDX_R_SHOULDER)
    el = norm(IDX_R_ELBOW)
    wr = norm(IDX_R_WRIST)

    # ── Body-relative coordinate frame ────────────────────────────────────────
    # x-axis: right-shoulder → left-shoulder (positive = toward person's left)
    # y-axis: 90° CCW from x-axis → points "up" in image space (decreasing y)
    shoulder_vec = ls - rs
    W_body = float(np.linalg.norm(shoulder_vec))
    if W_body < MIN_SHOULDER_WIDTH:
        return None                       # person too far / partially visible

    body_x = shoulder_vec / W_body
    body_y = np.array([-body_x[1], body_x[0]])   # 90° CCW rotation

    # ── Arm vectors ───────────────────────────────────────────────────────────
    sw = wr - rs   # shoulder → wrist
    se = el - rs   # shoulder → elbow
    ew = wr - el   # elbow → wrist

    # ── J0: base pan (left / right) ───────────────────────────────────────────
    # Wrist position along the across-body axis, normalised by shoulder width.
    # positive sw_bx → wrist toward left shoulder → arm swings across body
    sw_bx = float(np.dot(sw, body_x)) / W_body
    j0 = 90.0 + float(np.clip(sw_bx, -1.5, 1.5)) * (J0_RANGE_DEG / 1.5)

    # ── J1: shoulder elevation (up / down) ────────────────────────────────────
    # Wrist position along the vertical axis, normalised by shoulder width.
    # positive sw_by → wrist above shoulder → arm raised
    sw_by = float(np.dot(sw, body_y)) / W_body
    j1 = 90.0 - float(np.clip(sw_by, -1.5, 1.5)) * (J1_RANGE_DEG / 1.5)

    # ── J2: elbow flexion ─────────────────────────────────────────────────────
    # Interior angle at the elbow in 2D — exact geometry, most reliable measurement.
    # 180° = fully straight, ~30° = fully flexed.
    v_upper = rs - el    # elbow → shoulder
    v_lower = wr - el    # elbow → wrist
    if np.linalg.norm(v_upper) < 1e-3 or np.linalg.norm(v_lower) < 1e-3:
        j2 = 90.0
    else:
        j2 = angle_between(v_upper, v_lower)

    # ── J3: wrist / forearm flexion ───────────────────────────────────────────
    # Relative angle: how much the forearm rotates beyond the upper-arm direction.
    # 0° relative → arm straight → j3=90 (neutral).
    if np.linalg.norm(se) < 1e-3 or np.linalg.norm(ew) < 1e-3:
        j3 = 90.0
    else:
        se_bx = float(np.dot(se, body_x))
        se_by = float(np.dot(se, body_y))
        ew_bx = float(np.dot(ew, body_x))
        ew_by = float(np.dot(ew, body_y))
        upper_ang = math.atan2(se_by, se_bx)
        lower_ang = math.atan2(ew_by, ew_bx)
        j3_rel = math.degrees(lower_ang - upper_ang)
        j3_rel = (j3_rel + 180.0) % 360.0 - 180.0     # normalise to [-180, 180]
        j3 = 90.0 + float(np.clip(j3_rel, -J3_RANGE_DEG, J3_RANGE_DEG)) * (90.0 / J3_RANGE_DEG)

    return [
        float(np.clip(j0, 0, 180)),
        float(np.clip(j1, 0, 180)),
        float(np.clip(j2, 0, 180)),
        float(np.clip(j3, 0, 180)),
        90.0,   # J4: wrist rotation — not measurable in 2D, add hand tracking later
        90.0,   # J5: gripper — not controlled by pose
    ]


# ─────────────────────────── Smoothing ────────────────────────────────────────

def smooth_angles(
    raw_angles: Optional[List[float]],
    arm: ArmController,
    fps: int,
) -> List[int]:
    """Apply One Euro Filter + rate limit.

    If raw_angles is None (low confidence), holds last known good position
    instead of snapping to neutral.
    """
    if raw_angles is None:
        target = arm.last_good_angles
    else:
        target = raw_angles
        arm.last_good_angles[:] = raw_angles

    now = time.time()
    dt  = max(now - arm.prev_t, 1e-4)
    arm.prev_t = now

    filtered = [f.filter(v, dt) for f, v in zip(arm.oef_filters, target)]

    max_step = MAX_DEG_PER_SEC / fps
    limited  = []
    for sent, fa in zip(arm.last_sent_angles, filtered):
        d = max(-max_step, min(max_step, fa - sent))
        limited.append(int(round(float(np.clip(sent + d, 0, 180)))))
    arm.last_sent_angles[:] = limited
    return limited


# ─────────────────────────── Visualisation ────────────────────────────────────

def draw_overlay(frame: np.ndarray, keypoints: np.ndarray, angles: List[int],
                 tracking: bool) -> None:
    """Draw skeleton and angle readout on frame."""
    if keypoints.shape[0] == 0:
        return

    H, W = frame.shape[:2]
    kp   = keypoints[0]

    # Skeleton connections for the right arm + shoulder bar
    connections = [
        (IDX_L_SHOULDER, IDX_R_SHOULDER),
        (IDX_R_SHOULDER, IDX_R_ELBOW),
        (IDX_R_ELBOW,    IDX_R_WRIST),
    ]
    for a, b in connections:
        pt1 = (int(kp[a, 0]), int(kp[a, 1]))
        pt2 = (int(kp[b, 0]), int(kp[b, 1]))
        cv2.line(frame, pt1, pt2, (255, 255, 255), 2)

    # Joint dots
    for idx in [IDX_L_SHOULDER, IDX_R_SHOULDER, IDX_R_ELBOW, IDX_R_WRIST]:
        cv2.circle(frame, (int(kp[idx, 0]), int(kp[idx, 1])), 5, (0, 255, 0), -1)

    # Text overlays
    status_col = (0, 255, 0) if tracking else (0, 165, 255)
    status_txt = "TRACKING" if tracking else "HOLD"
    cv2.putText(frame, status_txt, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_col, 2)
    cv2.putText(frame, f"J0={angles[0]:3d}  J1={angles[1]:3d}  J2={angles[2]:3d}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(frame, f"J3={angles[3]:3d}  J4={angles[4]:3d}  J5={angles[5]:3d}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)


# ─────────────────────────── Main ─────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='CV Arm Control (RTMPose)')
    parser.add_argument('--com',       default=DEFAULT_COM,  help='COM port (e.g. COM5)')
    parser.add_argument('--baud',      type=int, default=DEFAULT_BAUD, help='Baud rate')
    parser.add_argument('--fps',       type=int, default=DEFAULT_FPS,  help='Target FPS')
    parser.add_argument('--display',   action='store_true', help='Show video with overlay')
    parser.add_argument('--no-serial', action='store_true', help='Disable serial output')
    parser.add_argument('--cpu',       action='store_true', help='Force CPU inference')
    args = parser.parse_args()

    device = 'cpu' if args.cpu else 'cuda'

    # ── Detector init ─────────────────────────────────────────────────────────
    try:
        from rtmlib import Wholebody
    except ImportError:
        print("rtmlib not found. Run: uv sync")
        sys.exit(1)

    print(f"Loading RTMPose WholeBody ({device})…")
    try:
        detector = Wholebody(
            mode='performance',
            backend='onnxruntime',
            device=device,
        )
    except Exception as e:
        if device == 'cuda':
            print(f"CUDA init failed ({e}), falling back to CPU.")
            detector = Wholebody(
                mode='performance',
                backend='onnxruntime',
                device='cpu',
            )
        else:
            raise

    # ── Camera init ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        sys.exit(1)

    # ── Arm controller ────────────────────────────────────────────────────────
    arm = ArmController(None if args.no_serial else args.com, args.baud)
    arm.send_angles([90, 90, 90, 90, 90, 90], 1000)
    time.sleep(1)

    print("CV Arm Control (RTMPose) running. Press ESC or Ctrl+C to exit.")
    print(f"  Serial:  {'enabled' if arm.ser else 'disabled'}")
    print(f"  Display: {'enabled' if args.display else 'disabled'}")
    print(f"  Device:  {device}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ── Pose detection ─────────────────────────────────────────────
            keypoints, scores = detector(frame)   # (N,133,2), (N,133)

            raw_angles = None
            if keypoints is not None and keypoints.shape[0] > 0:
                raw_angles = compute_arm_angles(keypoints, scores, frame.shape)

            send_angles = smooth_angles(raw_angles, arm, args.fps)
            arm.send_angles(send_angles, int(1000 / args.fps))

            # ── Display ────────────────────────────────────────────────────
            if args.display:
                if keypoints is not None and keypoints.shape[0] > 0:
                    draw_overlay(frame, keypoints, send_angles,
                                 tracking=(raw_angles is not None))
                cv2.imshow('CV Arm Control (RTMPose)', frame)
                if cv2.waitKey(1) & 0xFF == 27:   # ESC
                    break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Returning to neutral…")
        arm.send_angles([90, 90, 90, 90, 90, 90], 1000)
        time.sleep(1)
        cap.release()
        arm.close()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
