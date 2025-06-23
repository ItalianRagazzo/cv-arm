# -*- coding:UTF-8 -*-
import RPi.GPIO as GPIO
import time
from Arm_Lib import Arm_Device

Arm = Arm_Device()
time.sleep(.1)

# 控制舵机相关的参数
step_value = 5
max_value = 180
min_value = 0

s1_angle = 90
s2_angle = 90
s3_angle = 90
s4_angle = 90
s5_angle = 90
s6_angle = 90

s_time = 500


# 手柄按键定义
PSB_SELECT = 1
PSB_L3 = 2
PSB_R3 = 3
PSB_START = 4
PSB_PAD_UP = 5
PSB_PAD_RIGHT = 6
PSB_PAD_DOWN = 7
PSB_PAD_LEFT = 8
PSB_L2 = 9
PSB_R2 = 10
PSB_L1 = 11
PSB_R1 = 12
PSB_TRIANGLE = 13
PSB_CIRCLE = 14
PSB_CROSS = 15
PSB_SQUARE = 16

# PS2引脚设置
PS2_DAT_PIN = 9  # MOS
PS2_CMD_PIN = 10   # MIS
PS2_SEL_PIN = 7  # CS
PS2_CLK_PIN = 11  # SCK


# 回发过来的后4个数据是摇杆的数据
PSS_RX = 5  # 右摇杆X轴数据
PSS_RY = 6  # 右摇杆Y轴数据
PSS_LX = 7  # 左摇杆X轴数据
PSS_LY = 8  # 右摇杆Y轴数据


global PS2_KEY
global X1
global Y1
global X2
global Y2
global Handkey
scan = [0x01, 0x42, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
Data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
MASK = [PSB_SELECT, PSB_L3, PSB_R3, PSB_START, PSB_PAD_UP, PSB_PAD_RIGHT, PSB_PAD_DOWN,
        PSB_PAD_LEFT, PSB_L2, PSB_R2, PSB_L1, PSB_R1, PSB_TRIANGLE, PSB_CIRCLE, PSB_CROSS, PSB_SQUARE]

# 设置GPIO口为BCM编码方式
GPIO.setmode(GPIO.BCM)
# 忽略警告信息
GPIO.setwarnings(False)


# 初始化
def init():
    GPIO.setup(PS2_CMD_PIN, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(PS2_CLK_PIN, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(PS2_DAT_PIN, GPIO.IN)
    GPIO.setup(PS2_SEL_PIN, GPIO.OUT, initial=GPIO.HIGH)


# 读取PS2摇杆的模拟值
def PS2_AnologaData(button):
    return Data[button]


# 清空接受PS2的数据
def PS2_ClearData():
    Data[:] = []


# 读取PS2的数据
def PS2_ReadData(command):
    res = 0
    j = 1
    i = 0
    for i in range(8):
        if command & 0x01:
            GPIO.output(PS2_CMD_PIN, GPIO.HIGH)
        else:
            GPIO.output(PS2_CMD_PIN, GPIO.LOW)
        command = command >> 1
        time.sleep(0.000008)
        GPIO.output(PS2_CLK_PIN, GPIO.LOW)
        time.sleep(0.000008)
        if GPIO.input(PS2_DAT_PIN):
            res = res + j
        j = j << 1
        GPIO.output(PS2_CLK_PIN, GPIO.HIGH)
        time.sleep(0.000008)
    GPIO.output(PS2_CMD_PIN, GPIO.HIGH)
    time.sleep(0.00004)
    return res


# PS2获取按键类型
def PS2_Datakey():
    global Data
    global scan
    index = 0
    i = 0
    PS2_ClearData()
    GPIO.output(PS2_SEL_PIN, GPIO.LOW)
    for i in range(9):
        Data.append(PS2_ReadData(scan[i]))
    GPIO.output(PS2_SEL_PIN, GPIO.HIGH)

    Handkey = (Data[4] << 8) | Data[3]
    for index in range(16):
        if Handkey & (1 << (MASK[index] - 1)) == 0:
            return index+1
    return 0


try:
    init()
    while True:
        global PS2_KEY
        PS2_KEY = PS2_Datakey()
        # print ("PS2_KEY is %d" % PS2_KEY)
        # PSB_SELECT键按下
        if PS2_KEY == PSB_SELECT:
            print ("PSB_SELECT")
        # PSB_L3键按下，
        elif PS2_KEY == PSB_L3:
            print ("PSB_L3")
            
        # PSB_R3键按下，
        elif PS2_KEY == PSB_R3:
            print ("PSB_R3")

        # PSB_START键按下
        elif PS2_KEY == PSB_START:
            print ("PSB_START")
        
        # PSB_PAD_UP键按下，
        elif PS2_KEY == PSB_PAD_UP:
            print ("PSB_PAD_UP")
            s2_angle += step_value
            if s2_angle > max_value:
                s2_angle = max_value
            Arm.Arm_serial_servo_write(2, s2_angle, s_time)

        # PSB_PAD_RIGHT键按下，
        elif PS2_KEY == PSB_PAD_RIGHT:
            print ("PSB_PAD_RIGHT")
            s1_angle -= step_value
            if s1_angle < min_value:
                s1_angle = min_value
            Arm.Arm_serial_servo_write(1, s1_angle, s_time)

        # PSB_PAD_DOWN键按下，
        elif PS2_KEY == PSB_PAD_DOWN:
            print ("PSB_PAD_DOWN")
            s2_angle -= step_value
            if s2_angle < min_value:
                s2_angle = min_value
            Arm.Arm_serial_servo_write(2, s2_angle, s_time)

        # PSB_PAD_LEFT键按下，
        elif PS2_KEY == PSB_PAD_LEFT:
            print ("PSB_PAD_LEFT")
            s1_angle += step_value
            if s1_angle > max_value:
                s1_angle = max_value
            Arm.Arm_serial_servo_write(1, s1_angle, s_time)

        # L2键按下，
        elif PS2_KEY == PSB_L2:
            print ("PSB_L2")
            s3_angle += step_value
            if s3_angle > max_value:
                s3_angle = max_value
            Arm.Arm_serial_servo_write(3, s3_angle, s_time)

        # R2键按下，
        elif PS2_KEY == PSB_R2:
            print ("PSB_R2")
            s4_angle += step_value
            if s4_angle > max_value:
                s4_angle = max_value
            Arm.Arm_serial_servo_write(4, s4_angle, s_time)

        # L1键按下
        elif PS2_KEY == PSB_L1:
            print ("PSB_L1")
            s3_angle -= step_value
            if s3_angle < min_value:
                s3_angle = min_value
            Arm.Arm_serial_servo_write(3, s3_angle, s_time)

        # R1键按下
        elif PS2_KEY == PSB_R1:
            print ("PSB_R1")
            s4_angle -= step_value
            if s4_angle < min_value:
                s4_angle = min_value
            Arm.Arm_serial_servo_write(4, s4_angle, s_time)

        # 三角形按下，
        elif PS2_KEY == PSB_TRIANGLE:
            print ("PSB_TRIANGLE")
            s6_angle += step_value
            if s6_angle > max_value:
                s6_angle = max_value
            Arm.Arm_serial_servo_write(6, s6_angle, s_time)

        # 圆形键按下，
        elif PS2_KEY == PSB_CIRCLE:
            print ("PSB_CIRCLE")
            s5_angle -= step_value
            if s5_angle < min_value:
                s5_angle = min_value
            Arm.Arm_serial_servo_write(5, s5_angle+18, s_time)

            # time.sleep(0.1)
        # 方形键按下，
        elif PS2_KEY == PSB_SQUARE:
            print ("PSB_SQUARE")
            s5_angle += step_value
            if s5_angle > max_value:
                s5_angle = max_value
            Arm.Arm_serial_servo_write(5, s5_angle+18, s_time)

            # time.sleep(0.1)
        # 当x型按键按下时，
        elif PS2_KEY == PSB_CROSS:
            print ("PSB_CROSS")
            s6_angle -= step_value
            if s6_angle < min_value:
                s6_angle = min_value
            Arm.Arm_serial_servo_write(6, s6_angle, s_time)
        else:
            pass

        # 当L1或者R1按下时，读取摇杆数据的模拟值
        # if PS2_KEY == PSB_L1 or PS2_KEY == PSB_R1:
        #     X1 = PS2_AnologaData(PSS_LX)
        #     Y1 = PS2_AnologaData(PSS_LY)
        #     X2 = PS2_AnologaData(PSS_RX)
        #     Y2 = PS2_AnologaData(PSS_RY)
        #     print("X1=%d,Y1=%d,X2=%d,Y2=%d" % (X1, Y1, X2, Y2))

        # 必要的延时避免过于频繁发送手柄指令
        time.sleep(0.1)
except KeyboardInterrupt:
    GPIO.cleanup()
    pass

GPIO.cleanup()
