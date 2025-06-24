"""
pose_fps_test.py – Measure MediaPipe Pose landmark throughput (FPS).

Usage:
    python pose_fps_test.py                 # webcam 0, 10-second test
    python pose_fps_test.py --source 1      # webcam 1
    python pose_fps_test.py --source video.mp4 --duration 5
    python pose_fps_test.py --display       # show pose detection
"""

import cv2
import time
import argparse
import mediapipe as mp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0,
                        help="0/1/… for webcam index or path to video file")
    parser.add_argument("--duration", type=int, default=10,
                        help="seconds to run the test (default: 10)")
    parser.add_argument("--display", action="store_true",
                        help="display video with pose landmarks")
    args = parser.parse_args()

    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit()
                           else args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source {args.source}")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    frame_count = 0
    start_time = time.perf_counter()
    deadline = start_time + args.duration

    print(f"Running for {args.duration} s … Ctrl-C to stop early")

    try:
        while time.perf_counter() < deadline:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR→RGB because MediaPipe expects RGB input
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)            # run inference
            frame_count += 1

            if args.display:
                # Draw pose landmarks on the frame
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                cv2.imshow('Pose Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Print live FPS every second
            if frame_count % 30 == 0:        # adjust for smoother update if needed
                elapsed = time.perf_counter() - start_time
                print(f"\rCurrent FPS: {frame_count / elapsed:.2f}", end="")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    total_elapsed = time.perf_counter() - start_time
    avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0
    print(f"\nProcessed {frame_count} frames in {total_elapsed:.2f} s "
          f"→ average {avg_fps:.2f} FPS")

    cap.release()
    pose.close()
    if args.display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
