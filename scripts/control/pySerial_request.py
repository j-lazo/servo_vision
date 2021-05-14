import serial


def request_encoder_value(arduino_com_port_1='COM4', arduino_com_port_2='COM12', arduino_com_port_3='COM6'):
    arduino_port_1 = serial.Serial(arduino_com_port_1, 115200, writeTimeout=0)  # port 1 is for the upper motor
    arduino_port_2 = serial.Serial(arduino_com_port_2, 115200, writeTimeout=0)  # port 2 is for the side motor
    arduino_port_3 = serial.Serial(arduino_com_port_3, 115200, writeTimeout=0)  # port 3 is for the linear stage motor

    arduino_port_1.flushInput()
    arduino_port_1.flushOutput()
    arduino_port_2.flushInput()
    arduino_port_2.flushOutput()
    arduino_port_3.flushInput()
    arduino_port_3.flushOutput()
    arduino_port_1.write("re")
    arduino_port_2.write("re")
    arduino_port_3.write("re")
    while not arduino_port_1.in_waiting:
        print ("no serial data received yet")
    motor_side_encoder = arduino_port_1.readline()
    motor_upper_encoder = arduino_port_2.readline()
    motor_stepper_encoder = arduino_port_3.readline()

    return [motor_side_encoder, motor_upper_encoder, motor_stepper_encoder]
