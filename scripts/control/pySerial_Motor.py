import serial
import serial.tools.list_ports
import struct
import random
from time import sleep
import numpy as np
import sys

#Set the COM#

arduino_pro_micro_port_1 = 'COM12'
arduino_pro_micro_port_2 = 'COM6'

arduino_port_1 = serial.Serial(arduino_pro_micro_port_1, 115200, writeTimeout = 0)
#arduino_port_2 = serial.Serial(arduino_pro_micro_port_2, 115200, writeTimeout = 0)

correctionNumber = 100

#create sine wave input
timeStamp = np.arange(0, 30, 0.1)
sineWave = np.sin(timeStamp) * correctionNumber

#create square wave input:
numberCycle = 6
square_wave_width = 100
square_wave_amplitude = 10
square_wave = np.zeros(square_wave_width).tolist() +\
              (np.ones(square_wave_width)*square_wave_amplitude * 2).tolist() +\
              (np.ones(square_wave_width)*square_wave_amplitude * 3).tolist() +\
              (np.ones(square_wave_width)*square_wave_amplitude * 4).tolist() +\
              (np.ones(square_wave_width)*square_wave_amplitude * 5).tolist() +\
              (np.ones(square_wave_width)*square_wave_amplitude * 6).tolist()
square_wave_six = square_wave*6

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
        encoderValue = arduino_port_1.readline()
        print("This is what python receive:")
        print(encoderValue)
        print("Python will send target point to arduino::")
        print(sendData)
        sleep(.04)

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
    if i == len(sineWave) - 1:
        i = 0