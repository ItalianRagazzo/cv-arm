"""
cv_arm.pose_to_arm - CLI entry point converting webcam → MediaPipe Pose → DOFBOT angles
"""
import time
import struct
import math
import sys
import serial
import cv2
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, HandLandmarker
import mediapipe as mediapipe
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple
import typer

from cv_arm.utils import angle_between, align_vector_to_z
from cv_arm.processing import compute_pose_angles, compute_hand_metrics, blend_and_rate_limit, render_overlay
from cv_arm.config import Config


@dataclass
class RobotState:
    prev_t: float
    filtered_angles: List[float]
    filtered_torso_len: float
    filtered_upper_len: float
    filtered_fore_len: float
    last_sent_angles: List[int]


def initialize_serial(com: Optional[str], baud: int) -> Optional[serial.Serial]:
    """Open and return serial connection, or None."""
    if not com:
        return None
    try:
        return serial.Serial(com, baud, timeout=0)
    except serial.SerialException as e:
        logging.warning("Serial connection failed: %s", e)
        return None

def load_models(variant: str) -> Tuple[any, any]:
    pose_path = f"checkpoints/mediapipe/pose_landmarker_{variant}.task"
    logging.info("Loading pose model %s", pose_path)
    pose = mp_vision.PoseLandmarker.create_from_model_path(pose_path)
    hand_path = "checkpoints/mediapipe/hand_landmarker.task"
    logging.info("Loading hand model %s", hand_path)
    hand = mp_vision.HandLandmarker.create_from_model_path(hand_path)
    return pose, hand


def initialize_camera() -> cv2.VideoCapture:
    """Open webcam capture."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logging.error("Webcam not found.")
        sys.exit(1)
    return cap


def reset_robot(ser: Optional[serial.Serial], pack: struct.Struct,
                neutral_angles: List[int], init_ms: int) -> None:
    """Send robot to neutral pose."""
    if not ser:
        return
    logging.info("Resetting robot to neutral pose")
    try:
        ser.write(pack.pack(init_ms, *neutral_angles))
        time.sleep(init_ms / 1000)
    except Exception as e:
        logging.warning("Failed to send init pose: %s", e)


def run_loop(
    cap: cv2.VideoCapture,
    landmarker,
    hand_landmarker,
    ser: Optional[serial.Serial],
    pack: struct.Struct,
    durms: int,
    fps: int,
    display: bool,
    state: RobotState,
    config: Config,
) -> None:
    # Initialize timing
    prev_t = state.prev_t

    while cv2.waitKey(1) != 27:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)
        hand_result = hand_landmarker.detect(mp_img)
        if not result.pose_landmarks:
            cv2.imshow("pose", frame)
            continue

        if display:
            render_overlay(frame, result.pose_landmarks[0], hand_result.hand_landmarks)

        # Compute pose angles and confidences
        pose_lm = result.pose_landmarks[0]
        raw_pose_angles, pose_confs = compute_pose_angles(pose_lm, config)
        # Compute hand metrics
        raw_hand_angles, hand_confs = compute_hand_metrics(hand_result.hand_landmarks, config)
        # Combine raw angles & confidences
        raw_angles = raw_pose_angles + raw_hand_angles
        confs = pose_confs + hand_confs
        # Smooth and limit
        send_angles = blend_and_rate_limit(raw_angles, confs, state, config, fps)
        # Send over serial
        if ser and time.time() - prev_t >= 1 / fps:
            ser.write(pack.pack(durms, *send_angles))
            prev_t = time.time()
        # Show output angle window
        if display:
            render_overlay(frame, pose_lm, hand_result.hand_landmarks)
        cv2.imshow("pose", frame)

    cap.release()
    if ser:
        ser.close()
    cv2.destroyAllWindows()


app = typer.Typer()

@app.callback(invoke_without_command=True)
def main(
    com: Optional[str] = typer.Option(None, "--com", help="e.g. COM3 (leave empty to disable serial output)"),
    baud: int = typer.Option(2_000_000, "--baud", help="Baud rate (default: 2M)"),
    fps: int = typer.Option(60, "--fps", help="Target FPS for robot updates"),
    variant: str = typer.Option("lite", "--variant", help="Model size to fetch (lite, full, heavy)"),
    display: bool = typer.Option(False, "--display", help="Overlay pose landmarks on video"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose output"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Enable quiet output"),
) -> None:
    """Webcam → MediaPipe Pose → DOFBOT angles"""
    if verbose and quiet:
        typer.echo("Cannot use both --verbose and --quiet", err=True)
        raise typer.Exit(code=1)
    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.ERROR
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")

    ser = initialize_serial(com, baud)
    pack = struct.Struct(">H6B")
    durms = int(10000 / fps)
    landmarker, hand_landmarker = load_models(variant)
    cap = initialize_camera()
    state = RobotState(
        prev_t=time.time(),
        filtered_angles=[90.0]*6,
        filtered_torso_len=0.5,
        filtered_upper_len=0.4,
        filtered_fore_len=0.4,
        last_sent_angles=[90]*6,
    )
    reset_robot(ser, pack, [90]*6, init_ms=1000)
    # Centralized configuration
    config = Config()
    run_loop(cap, landmarker, hand_landmarker, ser, pack, durms, fps, display, state, config)
     
if __name__ == "__main__":
    app()
