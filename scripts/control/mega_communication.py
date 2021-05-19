import serial
from time import sleep
import random

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
    receive_data_upper = arduino_port_1.readline()
    receive_data_side = arduino_port_1.readline()

        # motor_upper_encoder = arduino_port_1.readline().decode()
        # motor_side_encoder = arduino_port_1.readline().decode()
    print ("this is what I receive::")
    print (receive_data_upper.decode("utf-8"), receive_data_side, type(receive_data_upper), type(receive_data_upper.decode()))
    return #[motor_upper_encoder, motor_side_encoder]


def serial_actuate(upper_joint_variable, side_joint_variable, arduino_port_1):
    arduino_port_1.flushInput()
    arduino_port_1.flushOutput()

    upper_joint_variable = "{:.0f}".format(upper_joint_variable)
    side_joint_variable = "{:.0f}".format(side_joint_variable)
    arduino_port_1.write((upper_joint_variable + "," + side_joint_variable).encode())

    return upper_joint_variable, side_joint_variable


def main():

    arduino_port_1 = serial_initialization()
    while True:
        print(serial_request(arduino_port_1))
        sleep(0.05)
        a = random.randint(-30, 30)
        b = random.randint(-30, 30)
        print(serial_actuate(a, b, arduino_port_1))


if __name__ == "__main__":
    main()



