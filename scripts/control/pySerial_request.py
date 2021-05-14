import serial
import time


def request_encoder_value(arduino_port_1, arduino_port_2, arduino_port_3):

    variable = 're'
    arduino_port_1.write(variable.encode())
    arduino_port_2.write(variable.encode())
    arduino_port_3.write(variable.encode())

    while not arduino_port_1.in_waiting:
        break

    motor_side_encoder = arduino_port_1.readline()
    motor_upper_encoder = arduino_port_2.readline()
    motor_stepper_encoder = arduino_port_3.readline()

    arduino_port_1.flushInput()
    arduino_port_1.flushOutput()
    arduino_port_2.flushInput()
    arduino_port_2.flushOutput()
    arduino_port_3.flushInput()
    arduino_port_3.flushOutput()

    return [motor_side_encoder, motor_upper_encoder, motor_stepper_encoder]


def initialize_ports(arduino_com_port_1='/dev/ttyACM2',
                     arduino_com_port_2='/dev/ttyACM1',
                     arduino_com_port_3='/dev/ttyACM0'):

    arduino_port_1 = serial.Serial(arduino_com_port_1, 115200, writeTimeout=0)  # port 1 is for the upper motor
    arduino_port_2 = serial.Serial(arduino_com_port_2, 115200, writeTimeout=0)  # port 2 is for the side motor
    arduino_port_3 = serial.Serial(arduino_com_port_3, 115200, writeTimeout=0)  # port 3 is for the linear stage motor




    return arduino_port_1, arduino_port_2, arduino_port_3


def main():

    arduino_port_1, arduino_port_2, arduino_port_3 = initialize_ports()
    request_encoder_value(arduino_port_1, arduino_port_2, arduino_port_3)


if __name__ == '__main__':
    main()