#!/usr/bin/env python3
"""Simple arm test script - moves DOFBOT through predefined positions without CV"""

import serial
import struct
import time

# Serial Configuration
COM_PORT = 'COM5'  # Updated to COM5
BAUD_RATE = 2_000_000
PACK = struct.Struct(">H6B")

def test_arm():
    """Test arm movement with predefined positions"""
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0)
        print(f"✓ Connected to {COM_PORT}")
    except Exception as e:
        print(f"❌ Serial error: {e}")
        return

    # Test positions: [j0, j1, j2, j3, j4, j5]
    positions = [
        [90, 90, 90, 90, 90, 90],    # Neutral
        [60, 90, 90, 90, 90, 90],    # Base left
        [120, 90, 90, 90, 90, 90],   # Base right
        [90, 60, 90, 90, 90, 90],    # Shoulder up
        [90, 120, 90, 90, 90, 90],   # Shoulder down
        [90, 90, 60, 90, 90, 90],    # Elbow bend
        [90, 90, 120, 90, 90, 90],   # Elbow extend
        [90, 90, 90, 60, 90, 90],    # Wrist up
        [90, 90, 90, 120, 90, 90],   # Wrist down
        [90, 90, 90, 90, 60, 90],    # Wrist rotate
        [90, 90, 90, 90, 120, 90],   # Wrist rotate other
        [90, 90, 90, 90, 90, 30],    # Gripper close
        [90, 90, 90, 90, 90, 150],   # Gripper open
        [90, 90, 90, 90, 90, 90],    # Back to neutral
    ]

    names = [
        "Neutral", "Base Left", "Base Right", "Shoulder Up", "Shoulder Down",
        "Elbow Bend", "Elbow Extend", "Wrist Up", "Wrist Down", 
        "Wrist Rotate 1", "Wrist Rotate 2", "Gripper Close", "Gripper Open", "Final Neutral"
    ]

    print("Starting arm test sequence...")
    print("Press Ctrl+C to stop")

    try:
        for i, (pos, name) in enumerate(zip(positions, names)):
            print(f"{i+1:2d}. {name:15} -> {pos}")
            ser.write(PACK.pack(2000, *pos))  # 2 second movement
            time.sleep(3)  # Wait for movement + buffer
            
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Return to neutral
        print("Returning to neutral...")
        ser.write(PACK.pack(1000, 90, 90, 90, 90, 90, 90))
        ser.close()
        print("✓ Test complete")

if __name__ == "__main__":
    test_arm()