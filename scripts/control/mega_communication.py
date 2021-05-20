import serial
from time import sleep
import random
import find_arduino
# def serial_initialization(arduino_com_port_1='COM4', arduino_com_port_2='COM12', arduino_com_port_3='COM6'):
#     arduino_port_1 = serial.Serial(arduino_com_port_1, 115200, writeTimeout=0)  # port 1 is for the upper motor
#     arduino_port_2 = serial.Serial(arduino_com_port_2, 115200, writeTimeout=0)  # port 2 is for the side motor
#     arduino_port_3 = serial.Serial(arduino_com_port_3, 115200, writeTimeout=0)  # port 3 is for the linear stage motor
#     sleep(2)
#
#     return arduino_port_1, arduino_port_2, arduino_port_3


def serial_initialization(arduino_com_port_1='COM9'):
    arduino_port_1 = serial.Serial(arduino_com_port_1, 115200, writeTimeout=0.5, timeout= 0.5)  # port 1 is for the upper motor
    sleep(2)
    return arduino_port_1


def serial_request(arduino_port_1):

    while arduino_port_1.in_waiting == 0:
        sleep(0.02)
        arduino_port_1.write("r".encode())

    # while arduino_port_1.in_waiting == 0:
    #     print("STUCK")
    #     pass
            # print ("waiting for serial data")
    #receive_data_test = arduino_port_1.readline()
    receive_data_upper = arduino_port_1.readline()
    receive_data_side = arduino_port_1.readline()
    receive_data_stepper = arduino_port_1.readline()

    #print("this is what I receive::")
    #print(receive_data_upper.decode("utf-8"), receive_data_side, type(receive_data_upper), type(receive_data_upper.decode()), receive_data_stepper)

    current_act_joint_variable = [float(receive_data_upper), float(receive_data_side), float(receive_data_stepper)]

    return current_act_joint_variable


def serial_actuate(upper_joint_variable, side_joint_variable, stepper_joint_variable, arduino_port_1):
    arduino_port_1.flushInput()
    arduino_port_1.flushOutput()

    upper_joint_variable = "{:.0f}".format(upper_joint_variable)
    side_joint_variable = "{:.0f}".format(side_joint_variable)
    stepper_joint_variable = "{:.0f}".format(stepper_joint_variable)
    arduino_port_1.write((upper_joint_variable + "," +side_joint_variable + ";" + stepper_joint_variable).encode())

    return upper_joint_variable, side_joint_variable, stepper_joint_variable


def main():

    arduino_port_1 = serial_initialization(str(find_arduino().port))

    while True:
        print(serial_request(arduino_port_1))
        sleep(1)
        a = random.randint(-30,30)
        b = random.randint(-30,30)
        c = random.randint(-5,5)
        print(serial_actuate(a, b, c, arduino_port_1))


if __name__ == '__main__':
    main()