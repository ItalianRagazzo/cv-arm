#!/usr/bin/env python3
"""
Simple CV Arm Control - Webcam → MediaPipe Pose → DOFBOT angles
Simplified single-file version with minimal dependencies
"""

import time
import struct
import math
import sys
import serial
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, List

# Configuration
DEFAULT_COM = 'COM5'
DEFAULT_BAUD = 2_000_000
DEFAULT_FPS = 30
SMOOTHING_ALPHA = 0.3
CONFIDENCE_THRESHOLD = 0.5
MAX_DEG_PER_SEC = 180.0

class ArmController:
    def __init__(self, com_port: Optional[str] = None, baud: int = DEFAULT_BAUD):
        self.ser = None
        self.pack = struct.Struct(">H6B")
        self.filtered_angles = [90.0] * 6
        self.last_sent_angles = [90] * 6
        self.prev_t = time.time()
        
        if com_port:
            try:
                self.ser = serial.Serial(com_port, baud, timeout=0)
                print(f"✓ Connected to {com_port}")
            except Exception as e:
                print(f"⚠️ Serial error: {e}")
    
    def send_angles(self, angles: List[int], duration_ms: int = 33):
        """Send angles to robot"""
        if self.ser:
            try:
                self.ser.write(self.pack.pack(duration_ms, *angles))
            except Exception as e:
                print(f"⚠️ Serial write failed: {e}")
    
    def close(self):
        if self.ser:
            self.ser.close()

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate angle between two vectors in degrees"""
    v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    return math.degrees(math.acos(np.clip(v1 @ v2, -1.0, 1.0)))

def align_vector_to_z(v: np.ndarray) -> np.ndarray:
    """Create rotation matrix to align vector v with z-axis"""
    v = v / (np.linalg.norm(v) + 1e-8)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(v, z)
    angle = math.acos(np.clip(np.dot(v, z), -1.0, 1.0))
    if np.linalg.norm(axis) < 1e-8:
        return np.eye(3)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)

def compute_arm_angles(landmarks) -> List[float]:
    """Compute DOFBOT joint angles from MediaPipe pose landmarks"""
    # Get right arm landmarks (shoulder, elbow, wrist)
    sh = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
    el = np.array([landmarks[14].x, landmarks[14].y, landmarks[14].z])
    wr = np.array([landmarks[16].x, landmarks[16].y, landmarks[16].z])
    
    # Check confidence
    confidences = [landmarks[i].visibility for i in [12, 14, 16]]
    if min(confidences) < CONFIDENCE_THRESHOLD:
        return [90.0, 90.0, 90.0, 90.0, 90.0, 90.0]  # Return neutral
    
    # Compute vectors
    sw = wr - sh  # shoulder to wrist
    se = el - sh  # shoulder to elbow
    ew = wr - el  # elbow to wrist
    
    # Joint 0: shoulder rotation
    j0 = 90 - math.degrees(math.atan2(-sw[0], sw[2]))
    
    # Joint 1: shoulder elevation
    sw_yz = min(math.degrees(math.atan2(sw[1], -sw[2])), 0.0)
    sw_se = angle_between(sw, se)
    j1 = 0.0 if sw_yz == 0.0 else 180.0 + sw_yz + sw_se
    
    # Joint 2: elbow flexion
    R = align_vector_to_z(sw)
    el_rot = R @ (el - sh)
    j2 = 90 - math.degrees(math.atan2(el_rot[1], el_rot[0]))
    
    # Joint 3: wrist flexion
    ew_rot = R @ ew
    if np.linalg.norm(ew_rot[:2]) > 1e-2:
        angle_xy = math.degrees(
            math.acos(np.clip(np.dot([1, 0], ew_rot[:2] / np.linalg.norm(ew_rot[:2])), -1.0, 1.0))
        )
        if ew_rot[1] < 0:
            angle_xy = -angle_xy
        j3 = 90 - angle_xy - 40
    else:
        j3 = 90.0
    
    # Clamp angles to valid range
    angles = [
        np.clip(j0, 0, 180),
        np.clip(j1, 0, 180), 
        np.clip(j2, 0, 180),
        np.clip(j3, 0, 180),
        90.0,  # Wrist rotation (neutral)
        90.0   # Gripper (neutral)
    ]
    
    return angles

def smooth_angles(raw_angles: List[float], filtered_angles: List[float], 
                 last_sent: List[int], fps: int) -> List[int]:
    """Apply smoothing and rate limiting to angles"""
    # Exponential smoothing
    for i in range(len(raw_angles)):
        filtered_angles[i] = (SMOOTHING_ALPHA * raw_angles[i] + 
                            (1 - SMOOTHING_ALPHA) * filtered_angles[i])
    
    # Rate limiting
    max_per_frame = MAX_DEG_PER_SEC / fps
    limited = []
    for i, fa in enumerate(filtered_angles):
        d = fa - last_sent[i]
        d = max(-max_per_frame, min(max_per_frame, d))
        la = last_sent[i] + d
        limited.append(int(round(np.clip(la, 0, 180))))
    
    # Update last sent
    for i, angle in enumerate(limited):
        last_sent[i] = angle
    
    return limited

def draw_landmarks(frame, landmarks):
    """Draw pose landmarks on frame"""
    h, w = frame.shape[:2]
    
    # Draw key points
    key_points = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
    for idx in key_points:
        if idx < len(landmarks):
            x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    # Draw connections
    connections = [(11, 12), (12, 14), (14, 16), (11, 13), (13, 15)]
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            pt1 = (int(start.x * w), int(start.y * h))
            pt2 = (int(end.x * w), int(end.y * h))
            cv2.line(frame, pt1, pt2, (255, 255, 255), 2)

def main():
    """Main application loop"""
    import argparse
    parser = argparse.ArgumentParser(description='CV Arm Control')
    parser.add_argument('--com', default=DEFAULT_COM, help='COM port (e.g., COM5)')
    parser.add_argument('--baud', type=int, default=DEFAULT_BAUD, help='Baud rate')
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS, help='Target FPS')
    parser.add_argument('--display', action='store_true', help='Show video with landmarks')
    parser.add_argument('--no-serial', action='store_true', help='Disable serial output')
    args = parser.parse_args()
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    # Initialize arm controller
    arm = ArmController(None if args.no_serial else args.com, args.baud)
    
    # Reset to neutral position
    arm.send_angles([90, 90, 90, 90, 90, 90], 1000)
    time.sleep(1)
    
    print("🎯 CV Arm Control started. Press ESC to exit.")
    print(f"📡 Serial: {'Enabled' if arm.ser else 'Disabled'}")
    print(f"📹 Display: {'Enabled' if args.display else 'Disabled'}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Compute angles
                raw_angles = compute_arm_angles(results.pose_landmarks.landmark)
                
                # Smooth and rate limit
                send_angles = smooth_angles(raw_angles, arm.filtered_angles, 
                                          arm.last_sent_angles, args.fps)
                
                # Send to robot
                now = time.time()
                if now - arm.prev_t >= 1 / args.fps:
                    arm.send_angles(send_angles, int(1000 / args.fps))
                    arm.prev_t = now
                
                # Draw landmarks if display enabled
                if args.display:
                    draw_landmarks(frame, results.pose_landmarks.landmark)
                    
                    # Show angles
                    cv2.putText(frame, f"Angles: {send_angles}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Show confidence
                    conf = min([results.pose_landmarks.landmark[i].visibility for i in [12, 14, 16]])
                    cv2.putText(frame, f"Confidence: {conf:.2f}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show frame if display enabled
            if args.display:
                cv2.imshow('CV Arm Control', frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    finally:
        # Cleanup
        print("🔄 Returning to neutral position...")
        arm.send_angles([90, 90, 90, 90, 90, 90], 1000)
        time.sleep(1)
        
        cap.release()
        arm.close()
        cv2.destroyAllWindows()
        print("✓ Cleanup complete")

if __name__ == "__main__":
    main()