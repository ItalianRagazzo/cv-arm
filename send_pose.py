import serial, struct, time

COM  = "COM3"          # check Device Manager
BAUD = 2_000_000       # same as Pi script
pack = struct.Struct(">H6B")          # duration + 6 angles

pose = [90, 90, 90, 90, 90, 90]       # any six ints
dur  = 500                            # half-second move

with serial.Serial(COM, BAUD, timeout=1) as ser:
    ser.write(pack.pack(dur, *pose))
    print("sent", pose)
    time.sleep((dur+100)/1000)        # wait till motion ends
