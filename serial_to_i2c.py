#!/usr/bin/env python3
import serial, struct
from arm_device import Arm_Device

PORT   = '/dev/ttyGS0'
BAUD   = 2000000          # 2 Mbit/s; both ends must match
FRAME  = struct.Struct('>H6B')   # 2-byte duration + six angles

ser = serial.Serial(PORT, BAUD, timeout=0.01)
arm  = Arm_Device()

buffer = b''

while True:
    buffer += ser.read(64)  # read available chunk
    while len(buffer) >= FRAME.size:
        frame_data = buffer[:FRAME.size]
        buffer = buffer[FRAME.size:]
        dur_ms, *angles = FRAME.unpack(frame_data)
        arm.Arm_serial_servo_write6_array(angles, dur_ms)
