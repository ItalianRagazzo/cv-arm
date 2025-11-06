# CV and DOFBOT Arm Project

A Python package and CLI for driving a DOFBOT robotic arm using real-time computer vision and pose tracking.

## Features
- Real-time human pose & hand landmark detection via MediaPipe
- Experimental 3D pose estimation with VideoPose3D
- Compute joint angles and send commands to DOFBOT over serial
- Smoothing, rate limiting, and confidence filtering
- Configurable model variants (lite, full, heavy)
- Loadable parameters via `config.yaml`
- Console script entry point: `cv-pose-to-arm` CLI

## Setup

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

### Advanced Models Setup

For transformer-based 3D pose estimation, install PyTorch:

```powershell
# Install PyTorch for PoseFormer
uv sync --extra torch
```

For experimental VideoPose3D, install MMPose:

```powershell
# Install MMPose for VideoPose3D (research/experimental)
uv run mim install mmcv==2.1.0 mmdet==3.2.0 mmpose==1.3.2
uv run mim download mmpose --config video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m --dest checkpoints
```

## Usage

### Basic Real-time Control

Run the main CLI to stream webcam → pose estimation → DOFBOT:

```powershell
uv run cv-pose-to-arm [--com COM_PORT] [--baud BAUD] [--fps FPS] \
  [--variant lite|full|heavy] [--display] [--verbose] [--quiet]
```

Alternatively, invoke the package directly:

```powershell
uv run python -m cv_arm [--com COM_PORT] \
  [--baud BAUD] [--fps FPS] [--variant lite|full|heavy] [--display] [--verbose] [--quiet]
```

### Test Scripts

Test different pose estimation models:

```powershell
# Test PoseFormer (transformer-based 3D)
uv run python poseformer_test.py

# Test VideoPose3D (experimental)
uv run python videopose_3_d_dofbot.py

# Performance benchmarking
uv run python pose_fps_test.py --source 0 --duration 10
```

### Command Options
- `--com`: e.g. `COM3` to enable serial output (omit to disable)
- `--baud`: baud rate (default: 2000000)
- `--fps`: command update rate (default: 60)
- `--variant`: model complexity (`lite`, `full`, `heavy`)
- `--display`: overlay landmarks and diagnostics on video window
- `--verbose` / `--quiet`: set logging level to DEBUG or ERROR

### Example Usage

```powershell
# Basic usage with display
uv run cv-pose-to-arm --com COM3 --variant full --display --verbose

# Experimental 3D pose estimation
uv run python videopose_3_d_dofbot.py

# Performance testing
uv run python pose_fps_test.py --source 0 --duration 10 --display
```

Press <Esc> in the video window to exit.

## Configuration

Override default parameters by editing `cv_arm/config.yaml`. Missing keys fall back to built-in defaults.

## Model Comparison

| Model | Z-axis Accuracy | Speed (FPS) | Memory | Use Case |
|-------|----------------|-------------|---------|----------|
| MediaPipe Lite | Good | 30+ | Low | Real-time, responsive |
| MediaPipe Full | Good | 20+ | Medium | Balanced accuracy/speed |
| PoseFormer | Very Good | 10-15 | High | Transformer-based 3D |
| VideoPose3D | Fair | 5-10 | High | Research/experimental |

**Recommendation**: Use **MediaPipe Full** for the best balance of accuracy and real-time performance.

## Performance Testing

Measure pose estimation performance:

```powershell
uv run python pose_fps_test.py --source 0 --duration 10 --display
```

Options:
- `--source`: camera index or video file path
- `--duration`: test duration in seconds
- `--display`: show annotated video output

## Project Structure

```
cv-arm/
├── cv_arm/              # Main package
│   ├── __main__.py      # CLI entry point
│   ├── pose_to_arm.py   # Core pose-to-arm logic
│   ├── config.yaml      # Configuration parameters
│   └── ...
├── tests/               # Unit tests
├── checkpoints/         # Model weights and configs
├── poseformer_test.py   # PoseFormer transformer script
├── videopose_3_d_dofbot.py  # VideoPose3D script
├── pose_fps_test.py     # Performance benchmarking
└── pyproject.toml       # Project configuration
```

## Testing & CI

Run the test suite:

```powershell
uv run pytest
```

A GitHub Actions workflow automatically runs tests on each push and pull request.

## Hardware Requirements

- **CPU**: Modern multi-core processor (Intel i5+ or AMD Ryzen 5+)
- **RAM**: 8GB minimum, 16GB recommended for VideoPose3D
- **GPU**: Optional but recommended for VideoPose3D (NVIDIA GTX 1060+ or RTX series)
- **Camera**: USB webcam or built-in camera
- **Serial**: USB-to-serial adapter for DOFBOT communication

## Troubleshooting

### Common Issues

1. **Serial connection failed**: Check COM port and ensure DOFBOT is connected
2. **Low FPS**: Try `lite` variant or reduce video resolution
3. **Import errors**: Ensure all dependencies installed with `uv sync`
4. **Model loading slow**: First run downloads models, subsequent runs are faster

### Performance Optimization

- Use GPU acceleration for VideoPose3D: Install CUDA-enabled PyTorch
- Reduce video resolution for better FPS
- Use `lite` model variant for real-time applications
- Close other applications to free system resources

## License

This project is licensed under the MIT License. See `LICENSE` for details.