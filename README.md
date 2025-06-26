# CV and DOFBOT Arm Project

## This project involves the development of a computer vision system and a robotic arm (DOFBOT) to control the arm's movements based on visual input.
## Install Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/ItalianRagazzo/cv-arm.git
   cd cv-arm
    ```
2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4. **Install torch and torchvision**:
    https://pytorch.org/get-started/locally/
      
5. **Run pose_fps_test.py** to test the FPS of the pose estimation:
    ```bash
    python pose_fps_test.py                 # webcam 0 10-second test
    python pose_fps_test.py --source 1      # webcam 1
    python pose_fps_test.py --source video.mp4 --duration 5
    python pose_fps_test.py --display       # display the video with pose estimation
    ```
