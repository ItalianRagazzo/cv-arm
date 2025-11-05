# CV and DOFBOT Arm Project

A Python package and CLI for driving a DOFBOT robotic arm using real-time computer vision and MediaPipe pose/hand tracking.

## Features
- Detect human pose & hand landmarks via MediaPipe
- Compute joint angles and send commands to DOFBOT over serial
- Smoothing, rate limiting, and confidence filtering
- Configurable model variants (lite, full, heavy)
- Loadable parameters via `config.yaml` (smoothing, thresholds, ratios)
- Console script entry point: `cv-pose-to-arm` CLI

## Setup

### Using uv (Recommended)

1. Install uv if you haven't already:

   ```powershell
   pip install uv
   ```

2. Clone the repository:

   ```powershell
   git clone https://github.com/ItalianRagazzo/cv-arm.git
   cd cv-arm
   ```

3. Install the project and dependencies:

   ```powershell
   uv sync
   ```

4. (Optional) Run unit tests to verify setup:

   ```powershell
   uv run pytest
   ```

### Using pip (Alternative)

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

3. Install the package in editable mode:

   ```powershell
   pip install -e .
   ```

4. (Optional) Run unit tests to verify setup:

   ```powershell
   pytest
   ```

6. Install PyTorch and torchvision (choose correct CUDA/cuDNN version):

   ```none
   # Visit https://pytorch.org/get-started/locally/ for instructions
   ```

7. Install MIM and MMEngine, then download MMpose checkpoints:

   ```powershell
   mim install mmengine mmdet mmcv mmpose
   mim download mmpose \
     --config video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m \
     --dest checkpoints
   ```

## Usage

### Using uv

Run the CLI to stream webcam → pose estimation → DOFBOT:

```powershell
uv run cv-pose-to-arm [--com COM_PORT] [--baud BAUD] [--fps FPS] \
  [--variant lite|full|heavy] [--display] [--verbose] [--quiet]
```

Alternatively, invoke the package directly:

```powershell
uv run python -m cv_arm [--com COM_PORT] \
  [--baud BAUD] [--fps FPS] [--variant lite|full|heavy] [--display] [--verbose] [--quiet]
```

### Using pip

Run the CLI to stream webcam → pose estimation → DOFBOT:

```powershell
cv-pose-to-arm [--com COM_PORT] [--baud BAUD] [--fps FPS] \
  [--variant lite|full|heavy] [--display] [--verbose] [--quiet]
```

Alternatively, invoke the package directly:

```powershell
python -m cv_arm [--com COM_PORT] \
  [--baud BAUD] [--fps FPS] [--variant lite|full|heavy] [--display] [--verbose] [--quiet]
```

Options:
- `--com`: e.g. `COM3` to enable serial output (omit to disable)
- `--baud`: baud rate (default: 2000000)
- `--fps`: command update rate (default: 60)
- `--variant`: model complexity (`lite`, `full`, `heavy`)
- `--display`: overlay landmarks and diagnostics on video window
- `--verbose` / `--quiet`: set logging level to DEBUG or ERROR

Example with uv:

```powershell
uv run cv-pose-to-arm --com COM3 --variant full --display --verbose
```

Example with pip:

```powershell
cv-pose-to-arm --com COM3 --variant full --display --verbose
```

Press <Esc> in the window to exit.

## Configuration File

You can override default parameters by editing `cv_arm/config.yaml`. Any keys missing will fall back to the built-in defaults.

## Performance Testing

To measure pose estimation FPS, run:

```powershell
# With uv
uv run python pose_fps_test.py --source 0 --duration 10

# With pip
python pose_fps_test.py --source 0 --duration 10
``` 

More options:
- `--source`: camera index or video file
- `--duration`: test duration in seconds
- `--display`: show annotated video

## Testing & CI

- Run `uv run pytest` (or `pytest` with pip) locally to execute unit tests in the `tests/` folder.
- A GitHub Actions workflow (`.github/workflows/ci.yml`) automatically runs tests on each push and pull request.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
