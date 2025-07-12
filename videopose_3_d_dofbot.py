# dofbot_videopose3d.py
# Webcam → MMPose VideoPose3D → DOFBOT angles

import time, struct, math, sys
import serial, cv2, numpy as np
from mmpose.apis import MMPoseInferencer

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

# Initialize VideoPose3D Inferencer
inferencer = MMPoseInferencer(
    pose3d='video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m',
    pose3d_weights='checkpoints/video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m.pth',
    device='cuda:0'  # or 'cpu'
)

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
filtered_angles = [90] * 6
SMOOTHING_ALPHA = 0.3
prev_t = time.time()

while cv2.waitKey(1) != 27:
    ok, frame = cap.read()
    if not ok:
        break

    pose_results = inferencer(frame)

    if not pose_results or len(pose_results[0]['predictions']) == 0:
        continue

    joints_3d = pose_results[0]['predictions'][0]['keypoints']

    # Indices (H36M): Right Shoulder(11), Elbow(13), Wrist(15)
    sh, el, wr = joints_3d[11], joints_3d[13], joints_3d[15]

    # Vectors
    sw, se, ew = wr - sh, el - sh, wr - el

    # Angles
    j0 = math.degrees(math.atan2(-sw[0], -sw[2])) + 90
    j1 = 90 - math.degrees(math.atan2(sw[1], -sw[2]))
    j2 = angle_between(se, [0,1,0])
    j3 = angle_between(ew, [0,1,0])

    angles = [
        int(np.clip(j0, 0, 180)),
        int(np.clip(j1, 0, 180)),
        int(np.clip(j2, 0, 180)),
        int(np.clip(j3, 0, 180)),
        90,
        90
    ]

    # Smooth angles
    for i in range(6):
        filtered_angles[i] = SMOOTHING_ALPHA * angles[i] + (1 - SMOOTHING_ALPHA) * filtered_angles[i]

    # Send via serial
    if ser:
        now = time.time()
        if now - prev_t >= 1 / FPS:
            ser.write(PACK.pack(DURMS, *[int(a) for a in filtered_angles]))
            prev_t = now

    # Display angles
    cv2.putText(frame, f"Angles: {np.round(filtered_angles).astype(int)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
    cv2.imshow("DOFBOT - VideoPose3D", frame)

cap.release()
if ser:
    ser.close()
cv2.destroyAllWindows()
