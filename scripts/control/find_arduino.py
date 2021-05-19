import warnings
import serial
import serial.tools.list_ports
import platform
import glob


def find_arduino():
    ser = 'naN'
    os_name = platform.system()
    print('OS:', os_name)

    if os_name == 'Windows':

        ports = [p.device for p in serial.tools.list_ports.comports()
                 if 'Arduino' in p.description]

        if not ports:
            raise IOError("No Arduino found")

        if len(ports) > 1:
            warnings.warn("Multiple Arduinos found - using the first")

        ser = serial.Serial(ports[0])

    elif os_name == 'Linux':
        glist = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyS*')
        if not glist:
            raise IOError("No Arduino found")

        if len(glist) > 1:
            warnings.warn("Multiple Arduinos found - using the first")

        ser = serial.Serial(glist[0])

    else:
        raise IOError("OS detected not compatible with script")

    return ser


if __name__ == '__main__':
    port = find_arduino()
    print(port)


