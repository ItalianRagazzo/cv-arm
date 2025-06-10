import cv2
import mediapipe as mp
import numpy as np
import serial
import time

# --- Robot Setup ---
# Adjust port and baud rate as needed
robot = serial.Serial('COM4', 115200)  # For Windows (use /dev/ttyUSB0 for Linux)
time.sleep(2)  # Let the serial port initialize

# --- MediaPipe Setup ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Capture Camera ---
cap = cv2.VideoCapture(0)

# --- Utility: Send joint angles over serial ---
def send_angles(joint_angles):
    """
    Convert joint angles (degrees) to command string and send over serial.
    Format example: "#1P1500#2P1600#3P1700T500\r\n"
    """
    command = "#1P{}#2P{}#3P{}T500\r\n".format(*joint_angles)
    robot.write(command.encode())

# --- Utility: Fake inverse kinematics (example) ---
def compute_inverse_kinematics(x, y, z):
    """
    This is a placeholder for inverse kinematics.
    You need real IK based on DOFBOT's dimensions.
    """
    # Map coordinates to servo angles (simplified)
    base_angle = int(np.clip(1500 + (x - 0.5) * 1000, 1000, 2000))
    shoulder_angle = int(np.clip(1500 - (y - 0.5) * 1000, 1000, 2000))
    elbow_angle = int(np.clip(1500 + z * 1000, 1000, 2000))
    return [base_angle, shoulder_angle, elbow_angle]

# --- Main Loop ---
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Right wrist landmark (normalized coordinates)
            wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            x, y, z = wrist.x, wrist.y, wrist.z

            # Compute joint angles
            angles = compute_inverse_kinematics(x, y, z)
            send_angles(angles)

        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow("DOFBOT Control", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("Interrupted.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    robot.close()
