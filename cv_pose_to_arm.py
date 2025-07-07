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

def align_vector_to_z(v):
    v = v / (np.linalg.norm(v) + 1e-8)
    z = np.array([0, 0, 1])

    axis = np.cross(v, z)
    angle = np.arccos(np.clip(np.dot(v, z), -1.0, 1.0))

    if np.linalg.norm(axis) < 1e-8:
        return np.eye(3)  # Already aligned

    axis = axis / np.linalg.norm(axis)

    # Rodrigues' rotation formula
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R

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
filtered_torso_len = 0.5  # Initial torso length (normalized)
filtered_upper_len = 0.4  # Initial upper arm length (normalized)
filtered_fore_len = 0.4   # Initial forearm length (normalized)
SMOOTHING_ALPHA = 0.3       # Smoothing factor (adjust 0.1–0.5 to taste)
CONFIDENCE_THRESHOLD = 0.9

# For rate limiting
MAX_DEG_PER_SEC = 360
MAX_DEG_PER_FRAME = MAX_DEG_PER_SEC / args.fps

# Track previous frame's angles for rate limiting
last_sent_angles = [90] * 6

# Anthropometric ratios relative to shoulder–hip length
upper_ratio = 0.65     # upper arm ≈ 65% of shoulder–hip
fore_ratio  = 0.55     # forearm ≈ 55% of shoulder–hip

torso_len = filtered_torso_len  # Use filtered torso length
upper_len = torso_len * upper_ratio
fore_len  = torso_len * fore_ratio

# -------- Initial Robot Reset (move to neutral pose) --------
INIT_DURATION_MS = 1000  # 1 second
NEUTRAL_ANGLES = [90, 90, 90, 90, 90, 90]

if ser:  # Only send if serial is active
    print("Sending robot to neutral pose...")
    try:
        ser.write(PACK.pack(INIT_DURATION_MS, *NEUTRAL_ANGLES))
        time.sleep(INIT_DURATION_MS / 1000)  # Wait for motion to complete
    except Exception as e:
        print(f"⚠️ Failed to send init pose: {e}")

# -------- Main loop ---------------------------------------------------------
prev_t = time.time()                              # Store time of last robot command
while cv2.waitKey(1) != 27:                      # Continue until Esc key pressed
    ok, frame = cap.read()                        # Capture frame from webcam
    if not ok:                                    # If frame capture failed
        break                                     # Exit loop

    # Do NOT flip here; process frame as-is
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB (MediaPipe)
    mp_img = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb)  # Create MediaPipe image
    result = landmarker.detect(mp_img)            # Run pose detection on image
    if not result.pose_landmarks:                 # If no pose detected
        cv2.imshow("pose", frame)   # display
        continue                                  # Skip to next frame

    # Draw pose landmarks if display option is enabled
    if args.display and result.pose_landmarks:
        draw_pose_landmarks(frame, result.pose_landmarks[0])

    # Extract landmarks and visibility
    lm = result.pose_landmarks[0]                 # Get first person's landmarks
    v = [lm[i].visibility for i in range(33)]     # All visibilities

    # Extract 3D coordinates of key body points (x, y, z normalized 0-1)
    # Get 3D joint positions
    sh = np.array([lm[12].x, lm[12].y, lm[12].z])  # Shoulder
    el = np.array([lm[14].x, lm[14].y, lm[14].z])  # Elbow
    wr = np.array([lm[16].x, lm[16].y, lm[16].z])  # Wrist
    hip = np.array([lm[24].x, lm[24].y, lm[24].z])
    left_wrist = np.array([lm[15].x, lm[15].y, lm[15].z])  # Left wrist
    left_shoulder = np.array([lm[11].x, lm[11].y, lm[11].z])  # Left shoulder

    # Vectors
    sw = wr - sh  # Shoulder to wrist
    se = el - sh  # Shoulder to elbow
    ew = wr - el  # Elbow to wrist

    # Calibrate arm model based on vision data
    if left_wrist[1] < left_shoulder[1] and v[12] > CONFIDENCE_THRESHOLD and v[24] > CONFIDENCE_THRESHOLD:
        # Shoulder to hip distance (torso length)
        #torso_len = np.linalg.norm(sh - hip)  # Shoulder to hip distance
        #torso_confidence = min(v[12], v[24])  # Confidence of shoulder and hip visibility
        #alpha = max(0.1, SMOOTHING_ALPHA * torso_confidence)
        #filtered_torso_len = alpha * torso_len + (1 - alpha) * filtered_torso_len

        # Update arm lengths based on XY data
        upper_len = np.linalg.norm(se[:2])  # Shoulder to elbow length in XY
        upper_confidence = min(v[12], v[14])  # Confidence of shoulder and elbow visibility
        alpha = max(0.1, SMOOTHING_ALPHA * upper_confidence)
        filtered_upper_len = alpha * upper_len + (1 - alpha) * filtered_upper_len

        fore_len = np.linalg.norm(ew[:2])   # Elbow to wrist length
        fore_confidence = min(v[14], v[16])  # Confidence of elbow and wrist visibility
        alpha = max(0.1, SMOOTHING_ALPHA * fore_confidence)
        filtered_fore_len = alpha * fore_len + (1 - alpha) * filtered_fore_len

    # Geometric calculations for simple arm model
    filtered_upper_len = filtered_upper_len if filtered_upper_len > 1e-8 else 1e-8  # Avoid zero division
    filtered_fore_len = filtered_fore_len if filtered_fore_len > 1e-8 else 1e-8      # Avoid zero division
    cos_phi_upper = np.clip((se[0]**2 + se[1]**2)**0.5 / filtered_upper_len, -1.0, 1.0)  # Cosine of angle between shoulder-elbow and horizontal
    cos_phi_fore  = np.clip((ew[0]**2 + ew[1]**2)**0.5 / filtered_fore_len, -1.0, 1.0)   # Cosine of angle between elbow-wrist and horizontal

    P_el_z = upper_len * (max(0.0, 1 - cos_phi_upper**2))**0.5  # Elbow Z position based on shoulder-elbow angle
    P_wr_z = P_el_z + fore_len * (max(0.0, 1 - cos_phi_fore**2))**0.5  # Wrist Z position based on elbow-wrist angle

    # Adjust Z based on geometric calcs
    sh[2] = 0  # Shoulder Z is always 0 in this model
    el[2] = P_el_z  # Elbow Z from geometric calculation
    wr[2] = P_wr_z  # Wrist Z from geometric calculation
    #el[2] = 0  # Elbow Z from geometric calculation
    #wr[2] = 0  # Wrist Z from geometric calculation

    # Reset Vectors based on new update
    sw = wr - sh  # Shoulder to wrist
    se = el - sh  # Shoulder to elbow
    ew = wr - el  # Elbow to wrist

    # Compute robot angles in degrees ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

    # 1. Horizontal angle of shoulder→wrist (XZ plane)
    # For straight up, sw[0] == 0, sw[2] == negative (since Z is forward), so atan2(0, -sw[2]) = 0, j0 = 90-0 = 90
    j0 = math.degrees(math.atan2(-sw[0], sw[2]))
    j0 = 90 - j0  # 0 when pointing Z+, make it 90

    # 2. Vertical angle of shoulder→wrist (YZ plane)
    # For straight up, sw[1] > 0, sw[2] == negative, atan2(sw[1], -sw[2]) = 90, so j1 = 90+angle_between(sw, se)
    sw_yz = min(math.degrees(math.atan2(sw[1], -sw[2])), 0)
    sw_se = angle_between(sw, se)
    if sw_yz == 0:
        j1 = 0
    else:
        j1 = 180 + sw_yz + sw_se
    # To force 90 when straight up, subtract (j1-90) from j1
    j1 = j1  # This will make j1 = 90 when arm is straight up

    # 3. Elbow roll (should be 90 when straight up)
    R = align_vector_to_z(sw)
    el_local = el - sh
    el_rotated = R @ el_local
    el_xy = el_rotated[:2]
    elbow_angle_roll = math.degrees(math.atan2(el_xy[1], el_xy[0]))
    j2 = 90 - elbow_angle_roll  # 90 when straight up

    # 4. Elbow-wrist vector in XY plane (relative to X axis)
    #ew_xy = ew[:2]
    #j3 = math.degrees(math.atan2(ew_xy[1], ew_xy[0]))
    j3 = math.degrees(math.atan2(ew[1], ew[0]))

    # 4. Elbow-wrist vector in XY plane (relative to X axis)
    ew_rotated = R @ ew
    #ew_xy = ew[:2]
    ew_xy = ew_rotated[:2]  # Use rotated vector in XY plane
    ew_xy_norm = np.linalg.norm(ew_xy)
    fore_conf = min(v[14], v[16])

    if ew_xy_norm > 1e-2 and fore_conf > CONFIDENCE_THRESHOLD:
        x_unit = np.array([1.0, 0.0])
        ew_xy_unit = ew_xy / ew_xy_norm
        dot = np.clip(np.dot(x_unit, ew_xy_unit), -1.0, 1.0)
        angle_xy = math.degrees(math.acos(dot))
        # Use sign from cross product to determine direction
        if ew_xy[1] < 0:
            angle_xy = -angle_xy
        j3_raw = 90 - angle_xy  # 90 when pointing along +X
        # Optionally, you can clamp j3_raw to [0, 135] here if needed
        j3 = j3_raw - 40 # intuitive correction
    else:
        j3 = 90  # Default to 90 if unreliable

    # Optionally, blend toward 90 if confidence is low or forearm is short
    blend_weight = min(1.0, max(0.0, (fore_conf - CONFIDENCE_THRESHOLD) / 0.2)) if ew_xy_norm > 1e-2 else 0.0
    j3 = blend_weight * j3 + (1 - blend_weight) * 90


    j2 = (j1 + j3)/2  # Average of j1 and j3 for elbow angle

    # Elbow angle calculation
    elbow_angle = angle_between(-se, ew)  # Angle between shoulder-elbow and elbow

    # Clamp and package
    angles = [
        int(np.clip(j0, 0, 180)),
        int(np.clip(j1, 0, 180)),
        int(np.clip(j2, 0, 135)),
        int(np.clip(j3, 0, 135)),
        90,
        90
    ]

    angles = [
        int((j0)),
        int((j1)),
        int((j2)),
        int((j3)),
        90,
        90
    ]

    # Determine effective visibility
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
        if confidences[i] < CONFIDENCE_THRESHOLD:
            filtered_angles[i] = filtered_angles[i] # don't change
        elif confidences[i] >= CONFIDENCE_THRESHOLD:
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
    yposition_display_line = 30
    cv2.putText(frame, f"{send_angles}", (10, yposition_display_line),        # Draw text at position (10, 30)
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)  # Green text, size 0.6, thickness 2
    yposition_display_line = yposition_display_line + 20
    cv2.putText(frame, f"Conf: {[round(c, 2) for c in confidences]}", (10, yposition_display_line),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    yposition_display_line = yposition_display_line + 20
    cv2.putText(frame, f"Elbow angle: {round(elbow_angle)}", (10, yposition_display_line),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    yposition_display_line = yposition_display_line + 20
    cv2.putText(frame, 
            f"Shoulder: ({sh[0]:.2f}, {sh[1]:.2f}, {sh[2]:.2f})", 
            (10, yposition_display_line),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 255), 
            1)
    yposition_display_line = yposition_display_line + 20
    cv2.putText(frame, 
            f"Elbow: ({el[0]:.2f}, {el[1]:.2f}, {el[2]:.2f})", 
            (10, yposition_display_line),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 255), 
            1)
    yposition_display_line = yposition_display_line + 20
    cv2.putText(frame, 
            f"Wrist: ({wr[0]:.2f}, {wr[1]:.2f}, {wr[2]:.2f})", 
            (10, yposition_display_line),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 255), 
            1)
    yposition_display_line = yposition_display_line + 20
    cv2.putText(frame, 
            f"Upper len: {filtered_upper_len:.3f}  Fore len: {filtered_fore_len:.3f}", 
            (10, yposition_display_line),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 255), 
            1)
    yposition_display_line = yposition_display_line + 20
    # Show calibration status with more clarity
    if left_wrist[1] < left_shoulder[1] and v[12] > CONFIDENCE_THRESHOLD and v[24] > CONFIDENCE_THRESHOLD:
        cv2.putText(frame, "Calibration: ACTIVE", (10, yposition_display_line),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Calibration: INACTIVE", (10, yposition_display_line),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)



    cv2.imshow("pose", frame)                        # Show frame with overlay (mirror effect)

# Cleanup when exiting
cap.release()                                        # Release webcam
if ser:
    ser.close()                                          # Close serial connection
cv2.destroyAllWindows()                              # Close all OpenCV windows
