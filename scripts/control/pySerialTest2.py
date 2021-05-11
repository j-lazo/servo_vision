import serial
import serial.tools.list_ports
import struct
import random
from time import sleep
import numpy as np
import sys

#Set the COM#

arduino_pro_micro_port_1 = '/dev/ttyACM0'
arduino_pro_micro_port_2 = 'COM6'

arduino_port_1 = serial.Serial(arduino_pro_micro_port_1, 115200, writeTimeout = 0)
#arduino_port_2 = serial.Serial(arduino_pro_micro_port_2, 115200, writeTimeout = 0)

correctionNumber = 100

timeStamp = np.arange(0, 30, 0.1)
sineWave = np.sin(timeStamp) * correctionNumber
i = 0

while True:
    if arduino_port_1.in_waiting:
        sendData = "{:.0f}".format(sineWave[i])
        #sendData = i
        i += 1
        #COMPort.write(struct.pack("l", 123)) # sending one long directly
        #COMPort.write(struct.pack("BBB", r, g, b)) # B means unsigned byte
        arduino_port_1.write(sendData)
        #print COMPort.readline()
        stringReceive = arduino_port_1.readline()
        print("This is what I receive:")
        print(stringReceive)
        print("I will send::")
        print(sendData)
        sleep(.02)

    # if arduino_port_2.in_waiting:
    #     sendData = "258"
    #
    #     #COMPort.write(struct.pack("l", 123)) # sending one long directly
    #     #COMPort.write(struct.pack("BBB", r, g, b)) # B means unsigned byte
    #     arduino_port_2.write(sendData)
    #     #print COMPort.readline()
    #     stringReceive = arduino_port_2.readline()
    #     print("This is what I receive:")
    #     print(stringReceive)
    #     print("I will send::")
    #     print(sendData)
    #     sleep(1)

    if i == 50:
        i = 1
