import time
import math
import os
import sys
sys.path.insert(0, '/home/pi/mp6/MP6/scripts')

from scripts.balloon_strategy import BalloonStrategy
from scripts.yolov5.detect import run_yolo


strat = BalloonStrategy()
strat.run()
