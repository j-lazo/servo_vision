from sksurgerynditracker.nditracker import NDITracker
import os
print(os.getcwd())

SETTINGS = {
    "tracker type": "aurora",
    "romfiles": [''.join([os.getcwd(), '/scripts/em_tracking/080082.rom'])]
        }
TRACKER = NDITracker(SETTINGS)

TRACKER.start_tracking()
port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
print(port_handles, timestamps, framenumbers, tracking, quality)
for t in tracking:
  print(t)

TRACKER.stop_tracking()
TRACKER.close()
