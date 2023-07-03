import cv2
import time
import threading
import numpy as np
import open3d as o3d
import os
import datetime
import sys
from config_cyc import BaseConfig as cfg

class depthCapThread(threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.dir_name = datetime.datetime.now().strftime('%Y %m %d %H %M %S')
        self.default_dir = r'F:\mmwave\data\\' + self.dir_name + r'\kinect\\'
        if not os.path.exists(self.default_dir):
            os.makedirs(self.default_dir)
            os.makedirs(self.default_dir + 'color')
            os.makedirs(self.default_dir + 'depth')
            os.makedirs(self.default_dir + 'depth_norm')
        self.config = o3d.io.read_azure_kinect_sensor_config('./config_1.json')
        self.sensor = o3d.io.AzureKinectSensor(self.config)  # 创建传感器对象
        self.device = 0  # 设备编号
        self.recorder = o3d.io.AzureKinectRecorder(self.config, self.device)  # 创建记录器对象
        self.time = []
        self.recorder.init_sensor()  # 初始化传感器
        self.recorder.open_record(self.default_dir + 'recoder.mkv')  # 开启记录器
        self.fps = 30
        # 再确认一下kinect收到的这个图的存储类型 普通的rgb相机是三通道 uint8，这里我暂时改成了1通道uint8
        self.latestReadNum = 0  # 最近get的这一帧的序号
        self.latestReadTimeStamp = 0

    def run(self):

        index = 0
        fps=1200
        for i in range(fps): 
            rec = self.recorder.record_frame(enable_record=True, enable_align_depth_to_color=True)
            # ret, frame = self.cap.read()
            if rec is not None:
                t = datetime.datetime.now()
                color = rec.color
                color = np.asarray(color)
                color = cv2.resize(color,dsize=(640,360))
                cv2.imshow("RGB pic",color)
                cv2.moveWindow("RGB pic", 640+100, 200)
                cv2.waitKey(1)
                index += 1
