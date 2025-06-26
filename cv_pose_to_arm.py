"""
cv_pose_to_arm.py – webcam → MediaPipe Pose → DOFBOT angles

First run downloads the chosen model (lite/full/heavy) into
%USERPROFILE%\\.mediapipe\\models\\.
"""

# Import required libraries
import argparse, time, struct, math, sys          # CLI args, timing, binary packing, math, system
import serial, cv2, numpy as np                   # Serial comm, OpenCV, NumPy arrays
from mediapipe.tasks import python as mp          # MediaPipe tasks framework
from mediapipe.tasks.python import vision as mp_vision  # MediaPipe vision tasks
import mediapipe as mediapipe                      # Main MediaPipe module

# -------- CLI ---------------------------------------------------------------
parser = argparse.ArgumentParser()                # Create argument parser
parser.add_argument("--com",   required=True, help="e.g. COM3")         # Serial port (required)
parser.add_argument("--baud",  type=int, default=2_000_000)             # Baud rate (default 2M)
parser.add_argument("--fps",   type=int, default=60)                    # Target FPS for robot updates
parser.add_argument("--variant", choices=("lite", "full", "heavy"),     # Model complexity choice
                    default="lite", help="Model size to fetch (default: lite)")
args = parser.parse_args()                        # Parse command line arguments

# -------- Serial ------------------------------------------------------------
ser   = serial.Serial(args.com, args.baud, timeout=0)  # Open serial connection to robot
PACK  = struct.Struct(">H6B")                     # Binary format: big-endian, 1 short + 6 bytes
DURMS = int(10000 / args.fps)                     # Duration in milliseconds between robot commands

# -------- MediaPipe Pose – let it auto-download -----------------------------
model_file = f"pose_landmarker_{args.variant}.task"     # Construct model filename
print(f"Loading {model_file} (auto-download if missing)…")  # User feedback
landmarker = mp_vision.PoseLandmarker.create_from_model_path(model_file)  # Load pose detection model
print("Pose model ready.")                        # Confirm model loaded

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)          # Open webcam (index 0) with DirectShow backend
if not cap.isOpened():                            # Check if camera opened successfully
    sys.exit("❌ Webcam not found.")              # Exit with error if no camera

# -------- Helpers -----------------------------------------------------------
def angle_between(v1, v2):                        # Calculate angle between two 3D vectors
    v1 = v1 / (np.linalg.norm(v1) + 1e-6)        # Normalize first vector (avoid division by zero)
    v2 = v2 / (np.linalg.norm(v2) + 1e-6)        # Normalize second vector (avoid division by zero)
    return math.degrees(math.acos(np.clip(v1 @ v2, -1.0, 1.0)))  # Dot product → angle in degrees

# -------- Main loop ---------------------------------------------------------
prev_t = time.time()                              # Store time of last robot command
while cv2.waitKey(1) != 27:                      # Continue until Esc key pressed
    ok, frame = cap.read()                        # Capture frame from webcam
    if not ok:                                    # If frame capture failed
        break                                     # Exit loop

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB (MediaPipe)
    mp_img = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb)  # Create MediaPipe image
    result = landmarker.detect(mp_img)            # Run pose detection on image
    if not result.pose_landmarks:                 # If no pose detected
        cv2.imshow("pose", frame)                 # Show original frame
        continue                                  # Skip to next frame


    lm = result.pose_landmarks[0]                 # Get first person's landmarks
    # Extract 3D coordinates of key body points (x, y, z normalized 0-1)
    sh = np.array([lm[12].x, lm[12].y, lm[12].z])    # RIGHT_SHOULDER = index 12
    el = np.array([lm[14].x, lm[14].y, lm[14].z])    # RIGHT_ELBOW = index 14
    wr = np.array([lm[16].x, lm[16].y, lm[16].z])    # RIGHT_WRIST = index 16
    hip = np.array([lm[24].x, lm[24].y, lm[24].z])   # RIGHT_HIP = index 24

    # Calculate 3D vectors between body points
    upper, torso, fore = el - sh, hip - sh, wr - el  # Upper arm, torso, forearm vectors

    # Convert body pose to robot joint angles
    j2 = angle_between(upper, torso)                 # Shoulder pitch: upper arm vs torso
    j3 = 180 - angle_between(upper, fore)            # Elbow flex: supplement of upper arm vs forearm
    j1 = (math.degrees(math.atan2(-upper[0], -upper[2])) + 360) % 360  # Shoulder yaw: upper arm rotation
    roll = angle_between(np.cross(upper, fore), np.array([0, 0, -1]))  # Wrist roll: hand orientation
    
    # Pack angles into robot command format with range limits
    angles = [
        int(np.clip(j1, 0, 180)),                    # joint-1: base rotation (0-180°)
        int(np.clip(j2, 0, 180)),                    # joint-2: shoulder pitch (0-180°)
        int(np.clip(j3, 0, 180)),                    # joint-3: elbow flex (0-180°)
        int(np.clip(j3, 0, 180)),                    # joint-4: duplicate elbow (robot specific)
        int(np.clip(roll, 0, 270)),                  # joint-5: wrist roll (0-270°)
        90                                           # joint-6: gripper fixed at 90°
    ]

    # Send robot commands at specified FPS rate
    now = time.time()                                # Get current time
    if now - prev_t >= 1/args.fps:                  # If enough time passed since last command
        ser.write(PACK.pack(DURMS, *angles))         # Send binary packet: duration + 6 angles
        prev_t = now                                 # Update last command time

    # Display angle values on video feed
    cv2.putText(frame, f"{angles}", (10, 30),        # Draw text at position (10, 30)
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)  # Green text, size 0.6, thickness 2
    cv2.imshow("pose", frame)                        # Show frame with overlay

# Cleanup when exiting
cap.release()                                        # Release webcam
ser.close()                                          # Close serial connection
cv2.destroyAllWindows()                              # Close all OpenCV windows
