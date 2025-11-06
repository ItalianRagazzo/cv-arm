#!/usr/bin/env python3
"""Test different DOFBOT protocols"""

import serial
import struct
import time

COM_PORT = 'COM3'

def test_protocol_1():
    """Test original protocol: >H6B"""
    print("Testing Protocol 1: >H6B (duration + 6 angles)")
    try:
        ser = serial.Serial(COM_PORT, 2000000, timeout=1)
        pack = struct.Struct(">H6B")
        
        # Send movement command
        data = pack.pack(2000, 45, 90, 90, 90, 90, 90)  # Move base to 45°
        print(f"Sending: {data.hex()}")
        ser.write(data)
        time.sleep(3)
        
        # Return to neutral
        data = pack.pack(1000, 90, 90, 90, 90, 90, 90)
        ser.write(data)
        ser.close()
        
    except Exception as e:
        print(f"Error: {e}")

def test_protocol_2():
    """Test alternative protocol: 6B (just angles)"""
    print("Testing Protocol 2: 6B (angles only)")
    try:
        ser = serial.Serial(COM_PORT, 2000000, timeout=1)
        
        # Send movement command
        data = struct.pack("6B", 45, 90, 90, 90, 90, 90)
        print(f"Sending: {data.hex()}")
        ser.write(data)
        time.sleep(3)
        
        # Return to neutral
        data = struct.pack("6B", 90, 90, 90, 90, 90, 90)
        ser.write(data)
        ser.close()
        
    except Exception as e:
        print(f"Error: {e}")

def test_protocol_3():
    """Test with different baud rates"""
    baud_rates = [9600, 115200, 1000000, 2000000]
    
    for baud in baud_rates:
        print(f"Testing baud rate: {baud}")
        try:
            ser = serial.Serial(COM_PORT, baud, timeout=1)
            pack = struct.Struct(">H6B")
            
            data = pack.pack(1000, 45, 90, 90, 90, 90, 90)
            print(f"Sending at {baud}: {data.hex()}")
            ser.write(data)
            time.sleep(2)
            
            ser.close()
            
        except Exception as e:
            print(f"Error at {baud}: {e}")

def test_protocol_4():
    """Test ASCII protocol"""
    print("Testing ASCII protocol")
    try:
        ser = serial.Serial(COM_PORT, 2000000, timeout=1)
        
        # Try ASCII command
        cmd = "#1P1500T1000\r\n"  # Common servo command format
        print(f"Sending ASCII: {cmd.strip()}")
        ser.write(cmd.encode())
        time.sleep(2)
        
        ser.close()
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("Testing different DOFBOT protocols...")
    print("Watch the arm for any movement between tests")
    print("=" * 50)
    
    test_protocol_1()
    time.sleep(2)
    
    test_protocol_2() 
    time.sleep(2)
    
    test_protocol_3()
    time.sleep(2)
    
    test_protocol_4()
    
    print("=" * 50)
    print("Test complete. Did you see any movement?")

if __name__ == "__main__":
    main()