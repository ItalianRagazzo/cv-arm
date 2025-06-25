"""
cv_pose_to_arm.py – webcam → MediaPipe Pose → DOFBOT angles

First run downloads the chosen model (lite/full/heavy) into
%USERPROFILE%\\.mediapipe\\models\\.
"""

import argparse, time, struct, math, sys
import serial, cv2, numpy as np
from mediapipe.tasks import python as mp
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mediapipe

# -------- CLI ---------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--com",   required=True, help="e.g. COM3")
parser.add_argument("--baud",  type=int, default=2_000_000)
parser.add_argument("--fps",   type=int, default=60)
parser.add_argument("--variant", choices=("lite", "full", "heavy"),
                    default="lite", help="Model size to fetch (default: lite)")
args = parser.parse_args()

# -------- Serial ------------------------------------------------------------
ser   = serial.Serial(args.com, args.baud, timeout=0)
PACK  = struct.Struct(">H6B")
DURMS = int(1_000 / args.fps)

# -------- MediaPipe Pose – let it auto-download -----------------------------
model_file = f"pose_landmarker_{args.variant}.task"
print(f"Loading {model_file} (auto-download if missing)…")
landmarker = mp_vision.PoseLandmarker.create_from_model_path(model_file)
print("Pose model ready.")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    sys.exit("❌ Webcam not found.")

# -------- Helpers -----------------------------------------------------------
def angle_between(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    return math.degrees(math.acos(np.clip(v1 @ v2, -1.0, 1.0)))

# -------- Main loop ---------------------------------------------------------
prev_t = time.time()
while cv2.waitKey(1) != 27:             # Esc quits
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)
    if not result.pose_landmarks:
        cv2.imshow("pose", frame)
        continue

    lm = result.pose_landmarks[0]
    # Use index-based access instead of PoseLandmark enum
    sh = np.array([lm[11].x, lm[11].y, lm[11].z])    # LEFT_SHOULDER = 11
    el = np.array([lm[13].x, lm[13].y, lm[13].z])    # LEFT_ELBOW = 13
    wr = np.array([lm[15].x, lm[15].y, lm[15].z])    # LEFT_WRIST = 15
    hip = np.array([lm[23].x, lm[23].y, lm[23].z])   # LEFT_HIP = 23

    upper, torso, fore = el - sh, hip - sh, wr - el

    j2 = angle_between(upper, torso)                 # shoulder pitch
    j3 = 180 - angle_between(upper, fore)            # elbow flex
    j1 = (math.degrees(math.atan2(-upper[0], -upper[2])) + 360) % 360
    roll = angle_between(np.cross(upper, fore), np.array([0, 0, -1]))
    angles = [
        int(np.clip(j1, 0, 180)),                    # joint-1
        int(np.clip(j2, 0, 180)),                    # joint-2
        int(np.clip(j3, 0, 180)),                    # joint-3
        int(np.clip(j3, 0, 180)),                    # joint-4
        int(np.clip(roll, 0, 270)),                  # joint-5
        90                                           # joint-6 fixed
    ]

    now = time.time()
    if now - prev_t >= 1/args.fps:
        ser.write(PACK.pack(DURMS, *angles))
        prev_t = now

    cv2.putText(frame, f"{angles}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imshow("pose", frame)

cap.release()
ser.close()
cv2.destroyAllWindows()
cv2.putText(frame, f"{angles}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
cv2.imshow("pose", frame)

cap.release()
ser.close()
cv2.destroyAllWindows()
