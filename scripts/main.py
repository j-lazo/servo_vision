import cv2
from sksurgerynditracker.nditracker import NDITracker
import numpy as np
import os

def main():

    diffx = 0
    diffy = 0
    diffz = 0

    # matplotlib.use("wx")

    SETTINGS = {
        "tracker type": "aurora",
        "romfiles": ["/home/nearlab/Jorge/current_work/robot_vision/scripts/em_tracking/080082.rom"],
    }
    TRACKER = NDITracker(SETTINGS)
    TRACKER.start_tracking()
    for i in range(5):

        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
        for t in tracking:
            x = t[0][3]
            y = t[1][3]
            z = t[2][3]
            print(x, y, z)

            # This will return video from the first webcam on your computer.
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (300, 300))
    # save_dir = '/home/nearlab/Jorge/current_work/robot_vision/data/calibration/'
    img_id = 0000
    while (True):

        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
        for t in tracking:

            x = t[0][3]
            y = t[1][3]
            z = t[2][3]

            x = x + diffx
            z = z + diffz
            y = y + diffy
            record = x, y, z
            print(x, y, z)

        # reads frames from a camera
        # ret checks return at each frame
        ret, frame = cap.read()
        # read serial
        pattern_size = (7, 5)
        square_size = 0.036
        # Converts to HSV color space, OCV reads colors as BGR
        # frame is converted to hsv
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (450, 300))
        # The original input frame is shown in the window
        cv2.imshow('Original', resized)

        # Wait for 'a' key to stop the program
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    TRACKER.stop_tracking()
    TRACKER.close()
    # Close the window / Release webcam
    cap.release()
    # After we release our webcam, we also release the output
    out.release()
    # De-allocate any associated memory usage
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()