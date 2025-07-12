import pkgutil, zipimport
pkgutil.ImpImporter = zipimport.zipimporter

import os, platform, pathlib
if platform.system() == 'Windows':
    # Add Torch's DLL folder so dependent .dlls (e.g. asmjit.dll) can be found
    torch_lib = pathlib.Path(__file__).parent / 'venv' / 'Lib' / 'site-packages' / 'torch' / 'lib'
    os.add_dll_directory(str(torch_lib))

import time, struct, math, sys
import serial, cv2, numpy as np

try:
    from mmpose.apis import MMPoseInferencer
except OSError as e:
    sys.exit(f"⚠️ Failed to load Torch dependencies: {e}\n"
             "→ Install Microsoft Visual C++ Redistributable: "
             "https://aka.ms/vs/16/release/vc_redist.x64.exe")

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

import logging
# suppress MMPose/mmengine/mmcv warnings
logging.getLogger('mmengine').setLevel(logging.ERROR)
logging.getLogger('mmpose').setLevel(logging.ERROR)
logging.getLogger('mmdet').setLevel(logging.ERROR)

# Initialize VideoPose3D Inferencer
print("Initializing VideoPose3D inferencer (may take several seconds)…")
t0 = time.time()
try:
    inferencer = MMPoseInferencer(
        det=None,    # disable 2D detector if supported
        pose3d='video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m',
        pose3d_weights=r'checkpoints\videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth',
        device='cpu'
    )
except TypeError:
    # fallback for versions without `det` parameter
    inferencer = MMPoseInferencer(
        pose3d='video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m',
        pose3d_weights=r'checkpoints\videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth',
        device='cpu'
    )
print(f"Inferencer ready (loaded in {time.time()-t0:.1f}s)")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("❌ Webcam not found.")

# Helper function: Calculate angles between vectors
def angle_between(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    return math.degrees(math.acos(np.clip(v1 @ v2, -1.0, 1.0)))

# Main loop
filtered_angles = [90.0] * 6
SMOOTHING_ALPHA = 0.3
prev_t = time.time()

while cv2.waitKey(1) != 27:
    ok, frame = cap.read()
    if not ok:
        break

    results = list(inferencer(frame))
    if not results:
        continue

    first = results[0]

    try:
        # raw keypoints: (N,3) array (x,y,conf)
        kp = np.array(first['predictions'][0]['keypoints'])
        if kp.shape[0] < 16:
            continue

        # Draw 2D skeleton overlay
        connections = [
            (0,1),(1,2),(2,3),(0,4),(4,5),(5,6),
            (0,7),(7,8),(8,9),(9,10),
            (8,11),(11,12),(12,13),
            (8,14),(14,15),(15,16)
        ]
        for a, b in connections:
            x1, y1 = kp[a][:2]
            x2, y2 = kp[b][:2]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        for x, y, _ in kp:
            cv2.circle(frame, (int(x), int(y)), 3, (0,0,255), -1)

        # use same kp array for 3D joint estimates
        joints_3d = kp
    except (IndexError, KeyError, TypeError):
        continue

    # Indices (H36M): Right Shoulder(11), Elbow(13), Wrist(15)
    sh, el, wr = joints_3d[11], joints_3d[13], joints_3d[15]

    # Ensure vectors are numpy arrays
    sw, se, ew = np.array(wr - sh), np.array(el - sh), np.array(wr - el)

    # Angles (explicitly handled)
    j0 = (math.degrees(math.atan2(-sw[0], -sw[2])) + 90) % 180
    j1 = (90 - math.degrees(math.atan2(sw[1], -sw[2]))) % 180
    j2 = np.clip(angle_between(se, np.array([0, 1, 0])), 0, 180)
    j3 = np.clip(angle_between(ew, np.array([0, 1, 0])), 0, 180)

    angles = [int(j0), int(j1), int(j2), int(j3), 90, 90]

    # Smooth angles explicitly as floats
    for i in range(6):
        filtered_angles[i] = (
            SMOOTHING_ALPHA * angles[i] + (1 - SMOOTHING_ALPHA) * filtered_angles[i]
        )

    # Serial write with error handling
    if ser:
        now = time.time()
        if now - prev_t >= 1 / FPS:
            try:
                ser.write(PACK.pack(DURMS, *[int(a) for a in filtered_angles]))
                # Live console output of angles
                print("Live Angles:", np.round(filtered_angles).astype(int).tolist())
            except Exception as e:
                print(f"⚠️ Serial write failed: {e}")
            prev_t = now

    # Display angles
    angle_text = f"Angles: {np.round(filtered_angles).astype(int)}"
    cv2.putText(frame, angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("DOFBOT - VideoPose3D", frame)

cap.release()
if ser:
    ser.close()
cv2.destroyAllWindows()

