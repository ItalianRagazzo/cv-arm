#!/usr/bin/env python3
"""Minimal serial test to diagnose DOFBOT connection"""

import serial
import serial.tools.list_ports
import struct
import time

def list_ports():
    """List all available COM ports"""
    ports = serial.tools.list_ports.comports()
    print("Available COM ports:")
    for port in ports:
        print(f"  {port.device}: {port.description}")
    return [port.device for port in ports]

def test_serial(port, baud=2000000):
    """Test basic serial communication"""
    print(f"\nTesting {port} at {baud} baud...")
    try:
        ser = serial.Serial(port, baud, timeout=1)
        print(f"✓ Connected to {port}")
        
        # Send neutral position command
        pack = struct.Struct(">H6B")
        data = pack.pack(1000, 90, 90, 90, 90, 90, 90)
        print(f"Sending: {data.hex()}")
        ser.write(data)
        
        # Try to read response
        time.sleep(0.1)
        if ser.in_waiting > 0:
            response = ser.read(ser.in_waiting)
            print(f"Response: {response.hex()}")
        else:
            print("No response received")
            
        ser.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    ports = list_ports()
    
    if not ports:
        print("No COM ports found!")
        return
    
    # Test each port
    for port in ports:
        if test_serial(port):
            print(f"✓ {port} appears to be working")
        else:
            print(f"❌ {port} failed")
        print("-" * 40)

if __name__ == "__main__":
    main()