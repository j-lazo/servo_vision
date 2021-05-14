from sksurgerynditracker.nditracker import NDITracker
import os


def test(tracker):

    port_handles, timestamps, framenumbers, tracking, quality = tracker.get_frame()
    for t in tracking:
        print('x:', t[0][3],
              'y:', t[1][3],
              'z:', t[2][3])


def main():

    SETTINGS = {
        "tracker type": "aurora",
        "romfiles": [''.join([os.getcwd(), '/scripts/em_tracking/080082.rom'])]
    }

    TRACKER = NDITracker(SETTINGS)
    TRACKER.start_tracking()
    test(TRACKER)
    TRACKER.stop_tracking()
    TRACKER.close()


if __name__ == '__main__':
    main()