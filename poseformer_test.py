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
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
t0 = time.time()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

def align_vector_to_z(v):
    """Create rotation matrix to align vector v with z-axis"""
    v = v / (np.linalg.norm(v) + 1e-6)
    z = np.array([0, 0, 1])
    if np.allclose(v, z):
        return np.eye(3)
    if np.allclose(v, -z):
        return np.diag([1, -1, -1])
    
    axis = np.cross(v, z)
    axis = axis / (np.linalg.norm(axis) + 1e-6)
    angle = math.acos(np.clip(np.dot(v, z), -1, 1))
    
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * np.dot(K, K)

def compute_dofbot_angles(landmarks):
    """Compute DOFBOT angles using the same method as pose_to_arm.py"""
    # Extract right arm landmarks (MediaPipe uses normalized coordinates)
    sh = np.array([landmarks.landmark[12].x, landmarks.landmark[12].y, landmarks.landmark[12].z])  # right_shoulder
    el = np.array([landmarks.landmark[14].x, landmarks.landmark[14].y, landmarks.landmark[14].z])  # right_elbow
    wr = np.array([landmarks.landmark[16].x, landmarks.landmark[16].y, landmarks.landmark[16].z])  # right_wrist
    
    # Get visibility scores for confidence
    v = [landmarks.landmark[i].visibility for i in [12, 14, 16]]
    min_confidence = 0.5
    
    if min(v) < min_confidence:
        return [90, 90, 90, 90, 90, 90]  # Return neutral if low confidence
    
    # Compute vectors
    sw = wr - sh  # shoulder to wrist
    se = el - sh  # shoulder to elbow
    ew = wr - el  # elbow to wrist
    
    # Joint 0: shoulder flexion/extension
    j0 = 90 - math.degrees(math.atan2(-sw[0], sw[2]))
    
    # Joint 1: shoulder abduction/adduction
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
    angles = [np.clip(j0, 0, 180), np.clip(j1, 0, 180), np.clip(j2, 0, 180), 
              np.clip(j3, 0, 180), 90, 90]  # j4, j5 neutral
    
    return [int(a) for a in angles]

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
        input_sequence = torch.tensor(np.array(frame_buffer), dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
        
        # PoseFormer inference
        with torch.no_grad():
            pose_3d = model(input_sequence).cpu().numpy()[0]
        
        # Use proper DOFBOT angle calculation
        angles = compute_dofbot_angles(results.pose_landmarks)
        
        # Get arm length for display (using normalized coordinates)
        landmarks = results.pose_landmarks.landmark
        shoulder_2d = [landmarks[12].x, landmarks[12].y]
        wrist_2d = [landmarks[16].x, landmarks[16].y]
        arm_length = np.linalg.norm(np.array(wrist_2d) - np.array(shoulder_2d)) * frame.shape[1]  # Convert to pixels for display
        
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
        cv2.putText(frame, f"Raw: {angles}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Smooth: {np.round(filtered_angles).astype(int)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, f"Arm: {arm_length:.0f}px | PoseFormer ({device})", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Show confidence
        conf = min([results.pose_landmarks.landmark[i].visibility for i in [12, 14, 16]])
        cv2.putText(frame, f"Confidence: {conf:.2f}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
    except Exception as e:
        print(f"⚠️ PoseFormer processing error: {e}")
        cv2.putText(frame, f"Error: {str(e)[:50]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imshow("PoseFormer 3D Pose", frame)

cap.release()
if ser:
    ser.close()
cv2.destroyAllWindows()