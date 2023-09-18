import numpy as np
from steaming import adcCapThread
from depthCap import depthCapThread
from get_point_cloud import *
import contextlib
import mmap
from multiprocessing import Process
# Mmwave is millimeter wave object detection, and camera is camera object detection. 
# The camera program is independent of millimeter waves and opens in advance. 
# Note that I am using a Kinect camera here. You can also replace it with code from other camera programs, or simply annotate it. 
# Good luck and God bless this code to run..
if __name__ == "__main__":
    mmWave = adcCapThread(123, "adc")
    camera = depthCapThread(2, "htc")
    camera.start()
    time.sleep(10)
    mmWave.run()
        