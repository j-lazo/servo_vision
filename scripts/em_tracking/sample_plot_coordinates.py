import numpy as np
import matplotlib.pyplot as plt
import datetime



def plot_figure(fig, xs, ys):

    fig.clear()
    plt.scatter(xs, ys)
    plt.pause(0.05)



fig = plt.figure()
i = 0

xs = []
ys = []


while True:
    i = i + 1
    print(i)
    xs.append(datetime.datetime.now().strftime('%H:%M:%S.%f'))
    y = np.random.random()
    ys.append(i)
    plot_figure(fig, xs, ys)

    plt.show()

"""
from sksurgerynditracker.nditracker import NDITracker
from pylab import plt
import numpy as np
import os
import cv2
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tmp102


# This function is called periodically from Main

def animate(fig, xs, ys):

    ax = fig.add_subplot(1, 1, 1)
    # Read temperature (Celsius) from TMP102
    temp_c = round(tmp102.read_temp(), 2)

    # Add x and y to lists
    xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    ys.append(temp_c)

    # Limit x and y lists to 20 items
    xs = xs[-20:]
    ys = ys[-20:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('TMP102 Temperature over Time')
    plt.ylabel('Temperature (deg C)')


def main():

    settings = {
        "tracker type": "aurora",
        "romfiles": [''.join([os.getcwd(), '/scripts/em_tracking/080082.rom'])],
            }
    TRACKER = NDITracker(settings)
    TRACKER.start_tracking()

    # Create figure for plotting
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    xs = []
    ys = []

    # Set up plot to call animate() function periodically
    ani = animation(fig, animate, fargs=(xs, ys), interval=1000)
    plt.show()

    #path = np.loadtxt("/home/nearlab/Jorge/current_work/robot_vision/scripts/em_tracking/final_path6.txt").reshape(-1, 3)
    #path = path * 0.2645833333
    #x_path = -path[:, 0] #left -x. righy x
    #z_path = path[:, 1]
    #y_path = path[:, 2]
    #y_path = -y_path

    diffx = 0
    diffy = 0
    diffz = 0

    ax1 = plt.subplot(221)
    ax1.title.set_text('X')
    ax1.set_xlim([-60, 60])
    ax1.set_xlabel('t')
    ax1.set_ylim([-50, 50])

    ax2 = plt.subplot(222)
    ax2.title.set_text('Y')
    ax2.set_xlim([-60, 60])
    ax2.set_xlabel('t')
    ax2.set_ylim([-50, 50])

    ax3 = plt.subplot(223)
    ax3.title.set_text('Z')
    ax3.set_xlim([-60, 60])
    ax3.set_xlabel('t')
    ax3.set_ylim([-50, 50])

    ax4 = plt.subplot(224)
    ax4.title.set_text('X vs Y')
    ax4.set_xlim([-60, 60])
    ax4.set_xlabel('t')
    ax4.set_ylim([-50, 50])

    while True:

        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
        t = tracking[0]
        x = t[0][3]
        y = t[1][3]
        z = t[2][3]

        ax1.plot(x, 0, 's')
        ax2.plot(y, 0, 's')
        ax3.plot(z, 0, 's')
        ax4.plot(x, y, 's')

        key = cv2.waitKey(1) & 0xFF
        plt.show()
        if key == ord('q'):
            break

    plt.show()


    TRACKER.stop_tracking()
    TRACKER.close()

    
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    ax11 = fig1.add_subplot(111)
    plt.setp(ax1, xlim=(-160, -60), ylim=(10, 110))
    plt.setp(ax11, xlim=(-160, -60), ylim=(10, 110))
    
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    ax22 = fig2.add_subplot(111)
    plt.setp(ax2, xlim=(-160,-60), ylim=(10,110)) #left: xlim, 60,160, right -160,-60
    plt.setp(ax22, xlim=(-160,-60), ylim=(10,110))
    
    #fig1.suptitle('XZ PLANE' ,  fontsize=16)
    #thismanager = get_current_fig_manager()
    #thismanager.window.SetPosition((600, 0))
    
    
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111)
    ax33 = fig3.add_subplot(111)
    plt.setp(ax3, xlim=(-160,-60), ylim=(-160,-60))
    plt.setp(ax33, xlim=(-160,-60), ylim=(-160,-60)) #right: xlim, 60,160, left -160,-60
    #fig1.suptitle('XY PLANE' ,  fontsize=16)
    #thismanager = get_current_fig_manager()
    #thismanager.window.SetPosition((1200, 0))
    
    
    media = 5
    
    for i in range(media):
        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
        for t in tracking:
            x = t[0][3]
            y = t[1][3]
            z = t[2][3] 
    
    for i in range(media):
        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
        for t in tracking:
            x = t[0][3]
            y = t[1][3]
            z = t[2][3] 
            diffx =  x_path[1] - x
            diffy =  y_path[1] - y
            diffz =  z_path[1] - z 
    
    print(x, y, z)
    #print(x_path, y_path, z_path)
    print(diffx, diffy, diffz)
    
    #x_path = x_path-diffx
    #z_path = z_path-diffz
    #y_path = y_path-diffy
    
    xx = []
    yy = []
    zz = []
    record = []
    
    line1, = ax1.plot(y_path, z_path, 'x')
    line11, = ax11.plot(0, 0, 'o')
    
    line2, = ax2.plot(x_path, z_path, 'x')
    line22, = ax22.plot(0, 0, 'o')
    
    line3, = ax3.plot(y_path, x_path, 'x')
    line33, = ax33.plot(0, 0 , 'o')
    
    
    while True:
        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
        for t in tracking:
    #      x = np.append(x_path, t[0][3])
    #      y = np.append(y_path, t[1][3])
    #      z = np.append(z_path, t[2][3])
          x = t[0][3]
          y = t[1][3]
          z = t[2][3]
          
          x = x + diffx
          z = z + diffz
    
          y = y + diffy
         
          
    #      print(x_path[1], y_path[1], z_path[1])
    
    #      xx.append(x)
    #      yy.append(y)
    #      zz.append(z)
          record = x, y, z
          print(x, y, z)
    #      record.append(recorded_path)
    
    #    line1, = ax1.plot(y, z, 's')    
    #    line2, = ax2.plot(x, z, 's')    
    #    line3, = ax3.plot(x, y, 's')
        
    #    line11.set_data(y, z)
    #    line22.set_data(x, z)
    #    line33.set_data(y, x)
    
        #plt.pause(0.001)
    
        #save path to a txt file
        #a_file = open("Kat_Left_3.txt", "a")
        #np.savetxt(a_file, record)
        #a_file.close
    
    #plt.show()
    

if __name__ == "__main__":
    main()"""