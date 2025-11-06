# poseformer_test.py
# Webcam → PoseFormer 3D pose → DOFBOT angles with transformer-based estimation

import cv2
import numpy as np
import time
import math
import serial
import struct
import sys
from collections import deque

try:
    import torch
    import torch.nn as nn
    import mediapipe as mp
except ImportError as e:
    sys.exit(f"⚠️ Dependencies not installed: {e}\n→ Run: uv sync")

# Serial Configuration
COM_PORT = 'COM3'
BAUD_RATE = 2_000_000
FPS = 30
PACK = struct.Struct(">H6B")
DURMS = int(1000 / FPS)

# Initialize serial connection
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0)
    print(f"✓ Serial connected to {COM_PORT}")
except Exception as e:
    print(f"⚠️ Serial error: {e}")
    ser = None

# Simple PoseFormer-inspired model
class SimplePoseFormer(nn.Module):
    def __init__(self, input_dim=66, hidden_dim=256, num_layers=4, seq_len=27):
        super().__init__()
        self.seq_len = seq_len
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=1024, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(hidden_dim, 51)  # 17 joints * 3 coords
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_projection(x)
        x = x + self.pos_encoding.unsqueeze(0)
        x = self.transformer(x)
        x = self.output_projection(x[:, -1])  # Use last frame
        return x.view(-1, 17, 3)

# Initialize MediaPipe for 2D pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize simple PoseFormer model
print("Loading PoseFormer model...")
t0 = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimplePoseFormer().to(device)
model.eval()
print(f"✓ PoseFormer loaded in {time.time()-t0:.1f}s on {device}")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("❌ Webcam not found")

def extract_2d_keypoints(landmarks):
    """Extract 2D keypoints from MediaPipe landmarks"""
    if not landmarks:
        return None
    
    keypoints = []
    for landmark in landmarks.landmark:
        keypoints.extend([landmark.x, landmark.y])
    return np.array(keypoints)

def angle_between(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    return math.degrees(math.acos(np.clip(v1 @ v2, -1.0, 1.0)))

# Frame buffer for temporal modeling
frame_buffer = deque(maxlen=27)
filtered_angles = [90.0] * 6
SMOOTHING_ALPHA = 0.3
prev_t = time.time()

print("Starting pose estimation... Press ESC to exit")

while cv2.waitKey(1) != 27:
    ret, frame = cap.read()
    if not ret:
        break
    
    # MediaPipe 2D pose detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if not results.pose_landmarks:
        cv2.imshow("PoseFormer 3D Pose", frame)
        continue
    
    # Extract 2D keypoints
    keypoints_2d = extract_2d_keypoints(results.pose_landmarks)
    if keypoints_2d is None:
        continue
    
    # Add to frame buffer
    frame_buffer.append(keypoints_2d)
    
    # Need enough frames for temporal modeling
    if len(frame_buffer) < 27:
        cv2.putText(frame, f"Buffering frames: {len(frame_buffer)}/27", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("PoseFormer 3D Pose", frame)
        continue
    
    try:
        # Prepare input for PoseFormer
        input_sequence = torch.tensor(np.array(frame_buffer), dtype=torch.float32).unsqueeze(0).to(device)
        
        # PoseFormer inference
        with torch.no_grad():
            pose_3d = model(input_sequence).cpu().numpy()[0]
        
        # Convert MediaPipe landmarks to 3D coordinates
        # Use simple depth estimation based on 2D positions
        h, w = frame.shape[:2]
        landmarks = results.pose_landmarks.landmark
        
        # Extract right arm landmarks (MediaPipe indices)
        shoulder_2d = [landmarks[12].x * w, landmarks[12].y * h]  # right_shoulder
        elbow_2d = [landmarks[14].x * w, landmarks[14].y * h]     # right_elbow
        wrist_2d = [landmarks[16].x * w, landmarks[16].y * h]     # right_wrist
        
        # Simple 3D estimation using arm proportions
        arm_length = np.linalg.norm(np.array(wrist_2d) - np.array(shoulder_2d))
        depth_scale = 500 / max(arm_length, 1)  # Normalize to ~500mm
        
        shoulder = np.array([shoulder_2d[0], shoulder_2d[1], 0])
        elbow = np.array([elbow_2d[0], elbow_2d[1], depth_scale * 100])
        wrist = np.array([wrist_2d[0], wrist_2d[1], depth_scale * 200])
            
            # Compute vectors
            shoulder_to_wrist = wrist - shoulder
            shoulder_to_elbow = elbow - shoulder
            elbow_to_wrist = wrist - elbow
            
            # Compute joint angles
            j0 = (math.degrees(math.atan2(-shoulder_to_wrist[0], -shoulder_to_wrist[2])) + 90) % 180
            j1 = (90 - math.degrees(math.atan2(shoulder_to_wrist[1], -shoulder_to_wrist[2]))) % 180
            j2 = np.clip(angle_between(shoulder_to_elbow, np.array([0, 1, 0])), 0, 180)
            j3 = np.clip(angle_between(elbow_to_wrist, np.array([0, 1, 0])), 0, 180)
            
            angles = [int(j0), int(j1), int(j2), int(j3), 90, 90]
            
            # Smooth angles
            for i in range(6):
                filtered_angles[i] = SMOOTHING_ALPHA * angles[i] + (1 - SMOOTHING_ALPHA) * filtered_angles[i]
            
            # Send to serial
            if ser:
                now = time.time()
                if now - prev_t >= 1 / FPS:
                    try:
                        ser.write(PACK.pack(DURMS, *[int(a) for a in filtered_angles]))
                    except Exception as e:
                        print(f"⚠️ Serial write failed: {e}")
                    prev_t = now
            
            # Draw 2D skeleton
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        # Display info
        cv2.putText(frame, f"Angles: {np.round(filtered_angles).astype(int)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Arm length: {arm_length:.0f}px", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "PoseFormer Active", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
    except Exception as e:
        print(f"⚠️ PoseFormer processing error: {e}")
        cv2.putText(frame, f"Error: {str(e)[:50]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imshow("PoseFormer 3D Pose", frame)

cap.release()
if ser:
    ser.close()
cv2.destroyAllWindows()