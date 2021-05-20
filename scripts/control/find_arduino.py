import warnings
import serial
import serial.tools.list_ports
import platform
import serial.tools.list_ports as port_list


def find_arduino():
    ser = 'naN'
    os_name = platform.system()
    print('OS:', os_name)

    if os_name == 'Windows':

        ports = [p.device for p in serial.tools.list_ports.comports()
                 if 'Arduino' in p.description or 'CH340' in p.description]

        if not ports:
            raise IOError("No Arduino found")

        if len(ports) > 1:
            warnings.warn("Multiple Arduinos found - using the first")

        ser = serial.Serial(ports[0]).port

    elif os_name == 'Linux':
        ports = list(port_list.comports())
        # PID of the arduino we are using
        ports = [p for p in ports if p.pid == 29987]

        if not ports:
            raise IOError("No Arduino found")

        if len(ports) > 1:
            warnings.warn("Multiple Arduinos found - using the first")

        ser = ports[0].device

    else:
        raise IOError("OS detected not compatible with script")

    return ser


if __name__ == '__main__':
    port = find_arduino()
    print(port)


