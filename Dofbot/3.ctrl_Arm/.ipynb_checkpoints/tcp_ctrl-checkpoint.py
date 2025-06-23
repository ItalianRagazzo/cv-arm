#coding=utf-8
import socket
import os
import time
from Arm_Lib import Arm_Device



global g_init
g_init = False
Arm = Arm_Device()


# 获取本机IP
def getwlanip():
    ip = os.popen("/sbin/ifconfig wlan0 | grep 'inet' | awk '{print $2}'").read()
    ip = ip[0 : ip.find('\n')]
    if(ip == ''):
        ip = 'x.x.x.x'
    return ip


#协议解析部分
def Analysis(socket, str):
    print(str)
    #    $04070113795#
    cmd = str[1:3]
    if cmd == "01": #获取硬件版本号
        data = "$0102100d#"
        socket.send(data.encode(encoding="utf-8"))
    elif cmd == "02": #获取指定舵机的位置
        id = str[5:7]
        pos = Arm.Arm_serial_servo_read(id)
        checksum = (2 + 5 + id + pos)&0xff
        checksumstr = "%02x" % checksum
        posstr = "%02d" % pos
        data = "$0205" + str[5:7] + posstr + checksumstr + "#"
        socket.send(data.encode(encoding="utf-8"))
    elif cmd == "03": #获取前几个舵机的位置
        pos = [0,0,0,0,0,0]
        for i in range(0, 6):
            pos[i] = Arm.Arm_serial_servo_read(i+1)
        checksum = (3 + 20 + 6 + pos[0] + pos[1] + pos[2] + pos[3] + pos[4] + pos[5])&0xff
        pos1str = "%03d" % pos[0]
        pos2str = "%03d" % pos[1]
        pos3str = "%03d" % pos[2]
        pos4str = "%03d" % pos[3]
        pos5str = "%03d" % pos[4]
        pos6str = "%03d" % pos[5]
        checksumstr = "%02x" % checksum
        data = "$0320" + str[5:7] + pos1str+pos2str+pos3str+pos4str+pos5str+pos6str + checksumstr + "#"
        print (data)
        socket.send(data.encode(encoding="utf-8"))
    elif cmd == "04":  #设置指定舵机位置
        id = int(str[5:7])
        angle = int(str[7:10])
        Arm.Arm_serial_servo_write(id, angle, 500)
    elif cmd == "05":  #设置一组舵机位置
        #    $0524060001800001800001809999XX#
        idmax = int(str[5:7])
        if idmax == 6:
            angle1 = int(str[7:10])
            angle2 = int(str[10:13])
            angle3 = int(str[13:16])
            angle4 = int(str[16:19])
            angle5 = int(str[19:22])
            angle6 = int(str[22:25])
            time = int(str[25:29])
            Arm.Arm_serial_servo_write6(angle1, angle2, angle3, angle4, angle5, angle6, time)
    elif cmd == "06":  #设置舵机ID地址
        id = int(str[5:7])
        if id < 1 or id > 250:
            socket.send("$0602020A#".encode(encoding="utf-8")) #编号错误
            return
        Arm.Arm_serial_set_id(id)    
        socket.send("$06020008#".encode(encoding="utf-8")) #返回ok
    elif cmd == "07":  #归中舵机位置
        Arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
    elif cmd == "09":  #急停|恢复舵机，关闭舵机
        state = int(str[5:6])
        Arm.Arm_serial_set_torque(state)
    elif cmd == "0A":  #设置舵机中值id:0 恢复默认  id：1-20 正常舵机中位值
        id = int(str[5:7])
        Arm.Arm_serial_servo_write_offset_switch(id)
    elif cmd == "0B": #动作组模式选择
        mode = int(str[5:7])
        if mode == 1: #学习模式
            Arm.Arm_Button_Mode(1)
        elif mode == 2: #退出学习模式
            Arm.Arm_Button_Mode(0)
        elif mode == 3: #清空动作组
            Arm.Arm_Clear_Action()
        elif mode == 4: #单次运行动作组
            Arm.Arm_Action_Mode(1)
        elif mode == 5: #循环运行动作组
            Arm.Arm_Action_Mode(2)
        elif mode == 6: #循环停止动作组
            Arm.Arm_Action_Mode(0)
    elif cmd == "0C": #控制RGB灯
        color = int(str[5:7])
        if color == 0:
            Arm.Arm_RGB_set(0, 0, 0)
        elif color == 1:
            Arm.Arm_RGB_set(50, 0, 0)
        elif color == 2:
            Arm.Arm_RGB_set(0, 50, 0)
        elif color == 3:
            Arm.Arm_RGB_set(0, 0, 50)
        elif color == 4:
            Arm.Arm_RGB_set(50, 50, 50)
    
    
            

#socket TCP通信建立
def start_tcp_server(ip, port):
    global g_init, g_socket
    g_init = True
    print('start_tcp_server')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((ip, port))
    sock.listen(5)
    while True:
        conn, address = sock.accept()
        g_socket = conn
        while True:
            cmd = g_socket.recv(1024).decode(encoding="utf-8") 
            if not cmd:
                break
            # print('   [-]cmd:', cmd, len(cmd))
            index1 = cmd.find("$")
            index2 = cmd.find("#")
            if index1 >= 0 and index2 > index1:
                Analysis(g_socket, cmd[index1:index2+1])
        g_socket.close()


#关闭socket
def waitClose(sock):
    global g_init
    g_init = False
    time.sleep(10)
    sock.close()


if __name__ == '__main__':
    port = 6000
    if g_init == False:
        while(True):
            ip = getwlanip()
            if(ip == "x.x.x.x"):
                print("get ip error!")
                time.sleep(.1)
                continue
            if(ip != "x.x.x.x"):
                print("%s:%d" % (ip, port))
                break
    start_tcp_server(ip, port)
