import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import cv2
import os
from matplotlib.animation import FuncAnimation
from sksurgerynditracker.nditracker import NDITracker


def update(frame, variables, subplots, init_points, tracker):
    ax1 = subplots[0]
    ax2 = subplots[1]
    ax3 = subplots[2]
    ax4 = subplots[3]

    ts = variables[0]
    xs = variables[1]
    ys = variables[2]
    zs = variables[3]

    init_x = init_points[0]
    init_y = init_points[1]
    init_z = init_points[2]

    # next line is the input from your sensor, if not input use any other input function

    port_handles, timestamps, framenumbers, tracking, quality = tracker.get_frame()
    positions = tracking[0]
    #temp_c = np.random.random()
    # #round(tmp102.read_temp(), 2)

    # Get the information from time and position
    ts.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    xs.append(positions[0][1])
    ys.append(positions[0][2])
    zs.append(positions[0][3])

    # Limit x and y lists to 20 items
    ts = ts[-20:]
    xs = xs[-20:]
    ys = ys[-20:]
    zs = zs[-20:]

    # Draw x and y lists
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    ax1.plot(ts, xs, color='red')
    ax2.plot(ts, ys, color='blue')
    ax3.plot(ts, zs, color='green')
    ax4.plot(xs, ys, 'o')

    # Format plot
    ax1.set_ylabel('x')
    ax1.set_xlabel('t')
    ax1.set_ylim([init_x - 3, init_x + 3])
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.set_xticklabels([])
    ax1.title.set_text('x(t)')

    ax2.set_ylabel('y')
    ax2.set_xlabel('t')
    ax2.set_ylim([init_y - 3, init_y + 3])
    ax2.set_xticklabels([])
    ax2.title.set_text('y(t)')

    ax3.set_ylabel('z')
    ax3.set_xlabel('t')
    ax3.set_ylim([init_z - 15, init_z + 15])
    ax3.tick_params(axis='x', labelrotation=45)
    ax3.title.set_text('z(t)')

    ax4.set_ylabel('y')
    ax4.set_xlabel('x')
    ax4.set_xlim([init_x - 0.5, init_x + 0.5])
    ax4.set_ylim([init_x - 0.5, init_x + 0.5])
    ax4.tick_params(axis='x', labelrotation=45)
    ax4.title.set_text('x vs y')


def main():

    SETTINGS = {
        "tracker type": "aurora",
        "romfiles": [''.join([os.getcwd(), '/scripts/em_tracking/080082.rom'])]
    }
    print('connecting with sensor...')
    TRACKER = NDITracker(SETTINGS)
    TRACKER.start_tracking()

    for wait_time in range(10):
        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
        positions = tracking[0]
        init_x = positions[0][1]
        init_y = positions[0][2]
        init_z = positions[0][3]

    # Define the figure
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    ts = []
    xs = []
    ys = []
    zs = []

    variables = [ts, xs, ys, zs]
    axes = [ax1, ax2, ax3, ax4]
    init_points = [init_x, init_y, init_z]

    animation = FuncAnimation(fig, update, fargs=(variables, axes, init_points, TRACKER), interval=2)
    plt.show()
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        TRACKER.stop_tracking()
        TRACKER.close()
        plt.close()


if __name__ == "__main__":
    main()


