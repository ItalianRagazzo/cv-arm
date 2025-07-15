# CV and DOFBOT Arm Project

A Python package and CLI for driving a DOFBOT robotic arm using real-time computer vision and MediaPipe pose/hand tracking.

## Features
- Detect human pose & hand landmarks via MediaPipe
- Compute joint angles and send commands to DOFBOT over serial
- Smoothing, rate limiting, and confidence filtering
- Configurable model variants (lite, full, heavy)
- Console script entry point: `cv-pose-to-arm` CLI

## Setup

1. Clone the repository:

   ```powershell
   git clone https://github.com/ItalianRagazzo/cv-arm.git
   cd cv-arm
   ```

2. (Optional) Create and activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install Python dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Install the package in editable mode:

   ```powershell
   pip install -e .
   ```

5. Install PyTorch and torchvision (choose correct CUDA/cuDNN version):

   ```none
   # Visit https://pytorch.org/get-started/locally/ for instructions
   ```

6. Install MIM and MMEngine, then download MMpose checkpoints:

   ```powershell
   mim install mmengine mmdet mmcv mmpose
   mim download mmpose \
     --config video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m \
     --dest checkpoints
   ```

## Usage

Run the CLI to stream webcam → pose estimation → DOFBOT:

```powershell
cv-pose-to-arm [--com COM_PORT] [--baud BAUD] [--fps FPS] [--variant lite|full|heavy] [--display]
```

Alternatively, invoke the package directly:

```powershell
python -m cv_arm [--com COM_PORT] [--baud BAUD] [--fps FPS] [--variant lite|full|heavy] [--display]
```

Options:
- `--com`: e.g. `COM3` to enable serial output (omit to disable)
- `--baud`: baud rate (default: 2000000)
- `--fps`: command update rate (default: 60)
- `--variant`: model complexity (`lite`, `full`, `heavy`)
- `--display`: overlay landmarks and diagnostics on video window

Example:

```powershell
cv-pose-to-arm --com COM3 --variant full --display
```

Press <Esc> in the window to exit.

## Performance Testing

To measure pose estimation FPS, run:

```powershell
python pose_fps_test.py --source 0 --duration 10
```

More options:
- `--source`: camera index or video file
- `--duration`: test duration in seconds
- `--display`: show annotated video

## License

This project is licensed under the MIT License. See `LICENSE` for details.
