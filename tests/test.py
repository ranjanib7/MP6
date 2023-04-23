import time
import math
from dronekit import connect, VehicleMode, simple_takeoff
from ../scripts/balloon_strategy import BalloonStrategy
from ../scripts/yolov5.detect import run_yolo

strat = BalloonStrategy()

# arm the drone
vehicle.armed   = True

while not vehicle.armed:
    print(" Waiting for arming...")
    time.sleep(1)

# takeoff to 3m
# Copter should arm in GUIDED mode
vehicle.mode    = VehicleMode("GUIDED")
vehicle.simple_takeoff(3)

# Wait until the vehicle reaches a safe height before processing the goto (otherwise the command
#  after Vehicle.simple_takeoff will execute immediately).
while True:
    print " Altitude: ", vehicle.location.global_relative_frame.alt
    #Break and return from function just below target altitude.
    if vehicle.location.global_relative_frame.alt>=3*0.95:
        print("Reached target altitude")
        break
    time.sleep(1)

strat.run()
