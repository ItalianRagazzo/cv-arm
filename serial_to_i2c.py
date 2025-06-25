#!/usr/bin/env python3
import serial, struct
from arm_device import Arm_Device

PORT   = '/dev/ttyGS0'
BAUD   = 2000000          # 2 Mbit/s; both ends must match
FRAME  = struct.Struct('>H6B')   # 2-byte duration + six angles

ser  = serial.Serial(PORT, BAUD, timeout=0)
arm  = Arm_Device()

while True:
    pkt = ser.read(FRAME.size)
    if len(pkt) != FRAME.size:
        continue
    dur_ms, *angles = FRAME.unpack(pkt)
    arm.Arm_serial_servo_write6_array(angles, dur_ms)

