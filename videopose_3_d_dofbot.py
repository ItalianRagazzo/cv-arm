# videopose_3d_dofbot.py
# Webcam → MMPose VideoPose3D → DOFBOT angles

import pkgutil, zipimport
pkgutil.ImpImporter = zipimport.zipimporter

import os, platform, pathlib
import time, struct, math, sys
import serial, cv2, numpy as np

# Workaround for Torch DLLs on Windows
if platform.system() == 'Windows':
    torch_lib = pathlib.Path(__file__).parent / 'venv' / 'Lib' / 'site-packages' / 'torch' / 'lib'
    if not torch_lib.exists():
        sys.exit(f"⚠️ Torch lib not found at {torch_lib}")
    os.add_dll_directory(str(torch_lib))

# Attempt import of MMPose Inferencer
try:
    from mmpose.apis import MMPoseInferencer
except OSError as e:
    sys.exit(
        f"⚠️ Failed to load Torch dependencies: {e}\n"
        "→ Install Microsoft Visual C++ Redistributable: "
        "https://aka.ms/vs/16/release/vc_redist.x64.exe"
    )

# Serial Configuration
COM_PORT = 'COM3'        # Set to your DOFBOT COM port
BAUD_RATE = 2_000_000
FPS = 30
PACK = struct.Struct(">H6B")
DURMS = int(1000 / FPS)

# Initialize serial connection
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0)
except Exception as e:
    print(f"⚠️ Serial error: {e}")
    ser = None

# Suppress verbose logs
import logging
for logger in ('mmengine', 'mmpose', 'mmdet'):
    logging.getLogger(logger).setLevel(logging.ERROR)

print("Script started.")
print("Initializing VideoPose3D inferencer (may take several seconds)…")
t0 = time.time()

# Load a lighter VideoPose3D model for real-time
MODEL_CFG = 'video-pose-lift_tcn-27frm-supv_8xb128-160e_h36m'
WEIGHTS = r'checkpoints\videopose_h36m_27frames_fullconv_supervised-fe8fbba9_20210527.pth'
try:
    inferencer = MMPoseInferencer(pose3d=MODEL_CFG, pose3d_weights=WEIGHTS, device='cpu')
except Exception as e:
    sys.exit(f"⚠️ Failed to initialize inferencer: {e}")
print(f"Inferencer ready (loaded in {time.time()-t0:.1f}s)")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("❌ Webcam not found.")

# Helper: angle between two vectors
def angle_between(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    return math.degrees(math.acos(np.clip(v1 @ v2, -1.0, 1.0)))

# Main loop
filtered_angles = [90.0] * 6
SMOOTHING_ALPHA = 0.3
prev_t = time.time()

while cv2.waitKey(1) != 27:
    print("----- Loop start -----")
    ok, frame = cap.read()
    print("Frame read status:", ok)
    if not ok:
        break
    print("Frame shape:", getattr(frame, "shape", None))
    print("Calling inferencer...")
    results = list(inferencer(frame))
    print("Inference returned entries:", len(results))
    if not results:
        print("No inference results; skipping frame.")
        continue
    print("First result keys:", list(results[0].keys()))

    first = results[0]

    print("Attempting keypoint extraction...")
    kp = None
    try:
        preds = first.get('predictions', first)
        print(" preds type:", type(preds), "length:", len(preds) if isinstance(preds, (list, np.ndarray)) else "N/A")

        # Handle nested list → dict scenario
        if isinstance(preds, list) and preds and isinstance(preds[0], list):
            nested = preds[0]
            print("  nested list detected, length:", len(nested))
            if nested and isinstance(nested[0], dict):
                print("   nested[0] dict keys:", list(nested[0].keys()))
                kp = np.array(nested[0].get('keypoints', []))

        # Existing dict or array cases
        if kp is None and isinstance(preds, (list, np.ndarray)) and preds:
            if isinstance(preds[0], (list, np.ndarray)):
                kp = np.array(preds[0])
            elif isinstance(preds[0], dict):
                kp = np.array(
                    preds[0].get('keypoints_3d') or
                    preds[0].get('keypoints') or
                    preds[0].get('preds_3d', [])
                )

        # Fallback: top-level keypoints
        if kp is None:
            print(" Fallback to top-level keypoints fields.")
            kp = np.array(
                first.get('keypoints_3d') or
                first.get('keypoints') or
                first.get('preds_3d', [])
            )
    except Exception as e:
        print(" Keypoint extraction exception:", e)
        kp = None

    if kp is None or kp.ndim != 2 or kp.shape[0] < 16 or kp.shape[1] < 3:
        print(" Invalid kp after extraction, kp:", kp)
        continue
    print(" Keypoints extracted; shape:", kp.shape)

    # Optional: draw 2D skeleton overlay (use x,y only)
    for connection in [
        (0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),(9,10),
        (8,11),(11,12),(12,13),(8,14),(14,15),(15,16)
    ]:
        a, b = connection
        x1, y1 = kp[a][:2]
        x2, y2 = kp[b][:2]
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
    for x, y, _ in kp:
        cv2.circle(frame, (int(x), int(y)), 3, (0,0,255), -1)
    print("2D skeleton drawn.")

    # Use same kp for 3D (assumed ordering H36M)
    sh, el, wr = kp[11][:3], kp[13][:3], kp[15][:3]

    # Compute vectors
    sw = np.array(wr) - np.array(sh)
    se = np.array(el) - np.array(sh)
    ew = np.array(wr) - np.array(el)
    print("Vectors computed.")

    # Compute angles in degrees, clamped to [0,180]
    j0 = (math.degrees(math.atan2(-sw[0], -sw[2])) + 90) % 180
    j1 = (90 - math.degrees(math.atan2(sw[1], -sw[2]))) % 180
    j2 = np.clip(angle_between(se, np.array([0,1,0])), 0, 180)
    j3 = np.clip(angle_between(ew, np.array([0,1,0])), 0, 180)
    angles = [int(j0), int(j1), int(j2), int(j3), 90, 90]
    print("Raw angles:", angles)

    # Smooth angles
    for i in range(6):
        filtered_angles[i] = (
            SMOOTHING_ALPHA * angles[i] + (1 - SMOOTHING_ALPHA) * filtered_angles[i]
        )
    print("Smoothed angles:", np.round(filtered_angles).astype(int).tolist())

    # Send via serial
    print("Preparing serial write.")
    if ser:
        now = time.time()
        if now - prev_t >= 1 / FPS:
            try:
                ser.write(PACK.pack(DURMS, *[int(a) for a in filtered_angles]))
                print("Serial write step complete.")
            except Exception as e:
                print(f"⚠️ Serial write failed: {e}")
            prev_t = now
    else:
        print("No serial connection; skipped write.")

    # Display angles
    angle_text = f"Angles: {np.round(filtered_angles).astype(int)}"
    cv2.putText(frame, angle_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imshow("DOFBOT - VideoPose3D", frame)
    print("Frame displayed.")

cap.release()
if ser:
    ser.close()
cv2.destroyAllWindows()
