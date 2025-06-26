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
from mediapipe import solutions                    # MediaPipe drawing utilities

# -------- CLI ---------------------------------------------------------------
parser = argparse.ArgumentParser()                # Create argument parser
parser.add_argument("--com", help="e.g. COM3 (leave empty to disable serial output)")
parser.add_argument("--baud",  type=int, default=2_000_000)             # Baud rate (default 2M)
parser.add_argument("--fps",   type=int, default=60)                    # Target FPS for robot updates
parser.add_argument("--variant", choices=("lite", "full", "heavy"),     # Model complexity choice
                    default="lite", help="Model size to fetch (default: lite)")
parser.add_argument("--display", action="store_true", help="Overlay pose landmarks on video")  # Display pose option
args = parser.parse_args()                        # Parse command line arguments

# -------- Serial ------------------------------------------------------------
ser = None # Open serial connection to robot
if args.com:
    try:
        ser = serial.Serial(args.com, args.baud, timeout=0)
    except serial.SerialException as e:
        print(f"⚠️ Serial connection failed: {e}")
        ser = None
 
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

def draw_pose_landmarks(image, landmarks):
    """Draw pose landmarks and connections on the image"""
    h, w = image.shape[:2]
    
    # Draw landmarks as circles
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
    
    # Define key connections to draw
    connections = [
        (11, 12), (12, 14), (14, 16),  # Left arm
        (11, 13), (13, 15), (15, 17),  # Right arm  
        (11, 23), (12, 24),           # Torso
        (23, 24), (23, 25), (24, 26), # Hips and legs
        (25, 27), (26, 28)            # Lower legs
    ]
    
    # Draw connections as lines
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            cv2.line(image, start_point, end_point, (255, 255, 255), 2)

# Initialize pose drawing utilities if display is enabled
if args.display:
    print("Pose display mode enabled.")

# Initialize filtered joint angle buffer
filtered_angles = [90] * 6  # Start at 90° neutral
SMOOTHING_ALPHA = 0.3       # Smoothing factor (adjust 0.1–0.5 to taste)

# For rate limiting
MAX_DEG_PER_SEC = 45
MAX_DEG_PER_FRAME = MAX_DEG_PER_SEC / args.fps

# Track previous frame's angles for rate limiting
last_sent_angles = [90] * 6

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

    # Draw pose landmarks if display option is enabled
    if args.display and result.pose_landmarks:
        draw_pose_landmarks(frame, result.pose_landmarks[0])

    lm = result.pose_landmarks[0]                 # Get first person's landmarks
    # Extract 3D coordinates of key body points (x, y, z normalized 0-1)
    # Get 3D joint positions
    sh = np.array([lm[12].x, lm[12].y, lm[12].z])  # Shoulder
    el = np.array([lm[14].x, lm[14].y, lm[14].z])  # Elbow
    wr = np.array([lm[16].x, lm[16].y, lm[16].z])  # Wrist

    # Vectors
    sw = wr - sh  # Shoulder to wrist
    se = el - sh  # Shoulder to elbow
    ew = wr - el  # Elbow to wrist

    # Compute angles in degrees
    # 1. Horizontal angle of shoulder→wrist (XZ plane)
    j0 = math.degrees(math.atan2(-sw[0], -sw[2]))  # side-to-side
    j0 = 90 + j0  # 0 when pointing Z+, make it 90

    # 2. Vertical angle of shoulder→wrist (YZ plane)
    j1 = math.degrees(math.atan2(sw[1], -sw[2]))
    j1 = 90 - j1  # 0 when Z+, make it 90

    # Angle between shoulder→elbow vector and global Y-axis (vertical)
    up = np.array([0, 1, 0])  # unit Y
    se_norm = se / (np.linalg.norm(se) + 1e-6)
    j2 = angle_between(se_norm, up)  # 0° when pointing up, 90° when horizontal
    #j2 = 90 - j2

    # 4. Vertical angle of elbow→wrist
    ew_norm = ew / (np.linalg.norm(ew) + 1e-6)
    j3 = angle_between(ew_norm, up)
    #j3 = 90 - j3

    # Clamp and package
    angles = [
        int(np.clip(j0, 0, 180)),
        int(np.clip(j1, 0, 180)),
        int(np.clip(j2, 0, 180)),
        int(np.clip(j3, 0, 180)),
        90,
        90
    ]

    # Determine effective visibility
    v = [lm[i].visibility for i in range(33)]  # All visibilities
    confidences = [
        min(v[12], v[16]),  # joint 0: shoulder-wrist horiz
        min(v[12], v[16]),  # joint 1: shoulder-wrist vertical
        min(v[12], v[14]),  # joint 2: shoulder-elbow vertical
        min(v[14], v[16]),  # joint 3: elbow-wrist vertical
        1.0,                # joint 4: fixed
        1.0                 # joint 5: fixed
    ]

    # Smooth joint angles using exponential moving average
    for i in range(6):
        alpha = max(0.1, SMOOTHING_ALPHA * confidences[i])
        filtered_angles[i] = alpha * angles[i] + (1 - alpha) * filtered_angles[i]
        if confidences[i] < 0.3:
            filtered_angles[i] = filtered_angles[i] # don't change
        elif confidences[i] >= 0.3:
            alpha = max(0.1, SMOOTHING_ALPHA * confidences[i])
            filtered_angles[i] = alpha * angles[i] + (1 - alpha) * filtered_angles[i]
    

    # Round for robot compatibility (convert to ints)
    #send_angles = [int(round(a)) for a in filtered_angles]

    # Rate limit each joint
    limited_angles = []
    for i in range(6):
        diff = filtered_angles[i] - last_sent_angles[i]
        # Clamp to ±MAX_DEG_PER_FRAME
        if diff > MAX_DEG_PER_FRAME:
            diff = MAX_DEG_PER_FRAME
        elif diff < -MAX_DEG_PER_FRAME:
            diff = -MAX_DEG_PER_FRAME
        # Apply clamped delta
        limited_angle = last_sent_angles[i] + diff
        limited_angles.append(int(round(limited_angle)))

    # Update for next frame
    last_sent_angles = limited_angles.copy()

    # Use for sending and displaying
    send_angles = limited_angles

    # Send robot commands at specified FPS rate
    if ser:
        now = time.time()                                # Get current time
        if now - prev_t >= 1/args.fps:                  # If enough time passed since last command
            ser.write(PACK.pack(DURMS, *send_angles))         # Send binary packet: duration + 6 angles
            prev_t = now                                 # Update last command time

    # Display angle values on video feed
    cv2.putText(frame, f"{send_angles}", (10, 30),        # Draw text at position (10, 30)
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)  # Green text, size 0.6, thickness 2
    cv2.putText(frame, f"Conf: {[round(c, 2) for c in confidences]}", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.imshow("pose", frame)                        # Show frame with overlay

# Cleanup when exiting
cap.release()                                        # Release webcam
if ser:
    ser.close()                                          # Close serial connection
cv2.destroyAllWindows()                              # Close all OpenCV windows
