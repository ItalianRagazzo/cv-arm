# Simple CV Arm Control

Control a DOFBOT robotic arm using your webcam and pose detection.

## Quick Start

1. **Install dependencies:**
   ```powershell
   uv sync
   ```

2. **Run the program:**
   ```powershell
   uv run python cv_arm.py --com COM5 --display
   ```

3. **Move your right arm** - the robot will follow your movements
4. **Press ESC** to exit

## Usage

```powershell
uv run python cv_arm.py [options]
```

**Options:**
- `--com COM5` - Serial port for robot (default: COM5)
- `--baud 2000000` - Baud rate (default: 2M)
- `--fps 30` - Update rate (default: 30)
- `--display` - Show video with pose landmarks
- `--no-serial` - Test without robot connected

**Examples:**
```powershell
# Basic usage with display
uv run python cv_arm.py --com COM5 --display

# Test without robot
uv run python cv_arm.py --no-serial --display

# High speed mode
uv run python cv_arm.py --com COM5 --fps 60
```

## Manual Control

Test the robot manually:
```powershell
uv run python arm_test.py
```

## Hardware Setup

1. **Connect DOFBOT** via USB
2. **Check COM port** in Device Manager
3. **Power on robot**
4. **Position camera** to see your upper body

## Troubleshooting

- **"Cannot open camera"** - Check webcam connection
- **"Serial error"** - Verify COM port and robot power
- **Robot not moving** - Check baud rate and connections
- **Jerky movement** - Lower FPS or improve lighting

## Files

- `cv_arm.py` - Main program (single file, ~200 lines)
- `arm_test.py` - Manual robot testing
- `checkpoints/mediapipe/` - Pose detection models

## Requirements

- **Python 3.8+**
- **Webcam** (built-in or USB)
- **DOFBOT** robotic arm
- **Windows/Linux/Mac**

## License

MIT License - see LICENSE file