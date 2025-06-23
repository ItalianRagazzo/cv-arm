#coding=utf-8
import socket
import os
import time
from Arm_Lib import Arm_Device



global g_sock
Arm = Arm_Device()





# socket客户端
def connect_tcp_server(ip, port):
    Arm.Arm_serial_set_torque(0)
    global g_sock
    g_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    g_sock.connect((ip, port))
    time.sleep(2)
    while True:
        time.sleep(.5)
        
        angle = [0, 0, 0, 0, 0, 0]
        for i in range(6):
            id = i + 1
            angle[i] = Arm.Arm_serial_servo_read(id)
            time.sleep(.005)
        data = "$20{0:0>3}{1:0>3}{2:0>3}{3:0>3}{4:0>3}{5:0>3}#"\
            .format(angle[0], angle[1], angle[2], angle[3], angle[4], angle[5])
        print(data)
        b_data = bytes(data, encoding = "utf8")
        g_sock.send(b_data)



#关闭socket
def waitClose(sock):
    sock.close()
    Arm.Arm_serial_set_torque(1)


if __name__ == '__main__':
    ip = '192.168.2.100'
    port = 6000
    try:
        connect_tcp_server(ip, port)
    except KeyboardInterrupt:
        waitClose(g_sock)
        print(" Program closed! ")
        pass
