import socket
import struct
import threading
import time
import datetime
import numpy as np
import mmap
import contextlib
from config_cyc import PointCloudConfig as cfg
from multiprocessing import  Process
from depthCap import depthCapThread
from get_point_cloud import *
import contextlib
import mmap
import colorsys
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont, Image

from mini_net.yolo import depthpctBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from yolo import YOLO
from utils.utils_bbox import DecodeBox
ADC_PARAMS = {'chirps': 80,
              'rx': 4,
              'tx': 3,
              'samples': 256,
              'IQ': 2,
              'bytes': 2}
# STATIC socket的缓冲区大小和一个UDP包的的大小
MAX_PACKET_SIZE = 4096
BYTES_IN_PACKET = 1456
# 计时
start_time = 0
#time_on_head = 0
#time_end_head = 0
end_time = 0
time_per_frame = 0
frame_time_list = []
# DYNAMIC
BYTES_IN_FRAME = (ADC_PARAMS['chirps'] * ADC_PARAMS['rx'] * ADC_PARAMS['tx'] *
                  ADC_PARAMS['IQ'] * ADC_PARAMS['samples'] * ADC_PARAMS['bytes'])
# 一帧被分成的那些整包所含字节数
BYTES_IN_FRAME_CLIPPED = (BYTES_IN_FRAME // BYTES_IN_PACKET) * BYTES_IN_PACKET
# 一帧被分成的包数
PACKETS_IN_FRAME = BYTES_IN_FRAME / BYTES_IN_PACKET
# 一帧被分成的包数的整数
PACKETS_IN_FRAME_CLIPPED = BYTES_IN_FRAME // BYTES_IN_PACKET

UINT16_IN_PACKET = BYTES_IN_PACKET // 2
UINT16_IN_FRAME = BYTES_IN_FRAME // 2
#threading.Thread
class adcCapThread ():
    def __init__(self, threadID, name, static_ip='192.168.33.30', adc_ip='192.168.33.180',
                 data_port=4098, config_port=4096, bufferSize = 1500):
        super(adcCapThread,self).__init__()
        self.whileSign = True
        self.framesavednum = 0
        self.expanxion =[]
        self.countnum  = []
        self.getbuffertime=[]
        self.laptoptime=[]
        self.threadID = threadID
        self.pretime =0
        self.name = name
        self.recentCapNum = 0 # 最近获得的这一帧的序号
        self.latestReadNum = 0 # 最近get的这一帧的序号
        self.nextReadBufferPosition = 0
        self.nextCapBufferPosition = 0
        self.bufferOverWritten = True
        self.bufferSize = bufferSize
        self.count = 0
        self.packetTimeStamp = 0
        self.frameTimeStamp = 0
        # Create configuration and data destinations
        self.cfg_dest = (adc_ip, config_port)
        self.cfg_recv = (static_ip, config_port)
        self.data_recv = (static_ip, data_port)
        self.a=1
        # Create sockets
        self.config_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

        # Bind data socket to fpga
        self.data_socket.bind(self.data_recv)
        self.data_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,2**27)
        self.data_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)

        # Bind config socket to fpga
        self.config_socket.bind(self.cfg_recv)
        # buffer size是指一次自动存储1500帧
        self.bufferArray = np.zeros((self.bufferSize,BYTES_IN_FRAME//2), dtype = np.int16)
        self.itemNumArray = np.zeros(self.bufferSize, dtype = np.int32)
        self.lostPackeFlagtArray = np.zeros(self.bufferSize,  dtype = bool)
        self.bufferTimeStamp = np.zeros(self.bufferSize, dtype=float)
        self.yolo = YOLO()
    def run(self):
        self._frame_receiver()
    
    def _frame_receiver(self):
        lost_packets = False
        recentframe = np.zeros(UINT16_IN_FRAME, dtype=np.int16)
        self.tp=time.time()
        while self.whileSign:
            
            packet_num, byte_count, packet_data = self._read_data_packet()
            after_packet_count = (byte_count+BYTES_IN_PACKET)% BYTES_IN_FRAME
            # print(packet_data.shape)#检查一下为什么会出现0到400的问题
            # the recent Frame begin at the middle of this packet
            if after_packet_count < BYTES_IN_PACKET :
                # start_time = time.time()
                # 后半个packet里放的是新帧头，拿过来
                recentframe[0:after_packet_count//2] = packet_data[(BYTES_IN_PACKET-after_packet_count)//2:]
                # 总第几帧
                self.recentCapNum = (byte_count+BYTES_IN_PACKET)//BYTES_IN_FRAME
                recentframe_collect_count = after_packet_count
                last_packet_num = packet_num
                break
                
            last_packet_num = packet_num
            #print(1)
        # 找到某一帧的开头了，进行下一步
        
        while self.whileSign:
            self.tp=time.time()
            self.frameTimeStamp = self.packetTimeStamp
            packet_num, byte_count, packet_data = self._read_data_packet()
            # fix up the lost packets
            # 试一下跳过这一帧 继续收下一帧
            #坏帧处理程序
            if last_packet_num < packet_num-1:                
                lost_packets = True
                # print('\a')
                # print("Packet Lost! Please discard this data.")
                #存下来坏帧
                self.lostPackeFlagtArray[self.nextCapBufferPosition] = True
                self.Get_result(recentframe,1)
                self.framesavednum +=1
                # 清空当前帧的缓存
                recentframe = np.zeros(UINT16_IN_FRAME, dtype=np.int16)
                self.packetTimeStamp = 0
                self.frameTimeStamp = 0
                while self.whileSign:
                    packet_num, byte_count, packet_data = self._read_data_packet()
                    time.sleep(10)
                    after_packet_count = (byte_count + BYTES_IN_PACKET) % BYTES_IN_FRAME
                    #print(packet_data.shape)  # 检查一下为什么会出现0到400的问题
                    # the recent Frame begin at the middle of this packet
                    if after_packet_count < BYTES_IN_PACKET:
                        self.frameTimeStamp = self.packetTimeStamp
                        # start_time = time.time()
                        # 后半个packet里放的是新帧头，拿过来
                        
                        recentframe[0:after_packet_count // 2] = packet_data[(BYTES_IN_PACKET - after_packet_count) // 2:]
                        # 总第几帧
                        self.recentCapNum = (byte_count + BYTES_IN_PACKET) // BYTES_IN_FRAME
                        recentframe_collect_count = after_packet_count
                        last_packet_num = packet_num
                        break
                    last_packet_num = packet_num
            # begin to process the recent packet
            # if the frame finished when this packet collected
            if recentframe_collect_count + BYTES_IN_PACKET >= BYTES_IN_FRAME:                
                
                recentframe[recentframe_collect_count//2:]=packet_data[:(BYTES_IN_FRAME-recentframe_collect_count)//2]
                self.lostPackeFlagtArray[self.nextCapBufferPosition] = False
                self.Get_result(recentframe,2)
                self.count += 1
                self.framesavednum +=1
                self.packetTimeStamp = 0
                self.frameTimeStamp = 0
                self.recentCapNum = (byte_count + BYTES_IN_PACKET)//BYTES_IN_FRAME
                recentframe = np.zeros(UINT16_IN_FRAME, dtype=np.int16)
                after_packet_count = (recentframe_collect_count + BYTES_IN_PACKET) % BYTES_IN_FRAME
                #帧发完了就退出
                if packet_data[(BYTES_IN_PACKET-after_packet_count)//2:].size > 0:
                    recentframe[0:after_packet_count//2] = packet_data[(BYTES_IN_PACKET-after_packet_count)//2:]
                    recentframe_collect_count = after_packet_count

                else:
                    # end_time = time.time()
                    # time_per_frame = (end_time - start_time)/self.count
                    # frame_time_list.append(time_per_frame)
                    # print(time_per_frame)
                    # print(sum(frame_time_list)/len(frame_time_list))
                    print('done') # 有可能是因为最后一个包，也有可能是连接出错重发
                    recentframe_collect_count = 0
                    self.count = 0
                    if (self.framesavednum>=300):
                        cv2.destroyAllWindows()
                        # print("平均值如下:")
                        # print(sum(self.expanxion)/len(self.expanxion))
                        # print(sum(self.getbuffertime)/len(self.getbuffertime))
                        # print(sum(self.laptoptime)/len(self.laptoptime))
                        self.countnum.pop(0)
                        # print(sum(self.countnum)/len(self.countnum))
                        # print(self.countnum)
                        return
                    #start_time = time.time()
                    # break
            else:
                after_packet_count = (recentframe_collect_count + BYTES_IN_PACKET)%BYTES_IN_FRAME
                recentframe[recentframe_collect_count//2:after_packet_count//2]=packet_data
                recentframe_collect_count = after_packet_count 
            last_packet_num = packet_num
        
    def getFrame(self):
        return self.framesavednum
    
    def Get_result(self,recentframe,Sign):
        if Sign ==1:
            return
        # Here, get the mmwave bin buffer and change it to pointcloud.
        buffer = np.frombuffer(recentframe,dtype=np.int16)
        point_cloud=[]
        #t0=time.time()
        point_cloud = buffer2PointCloud(buffer)
        #t3=time.time()
        #self.getbuffertime.append(t3-t0)
        if(point_cloud!=[]):
            # Here, get the pointcloud and change it to radar point image.
            img_z       = pointcloudtoimage(point_cloud)
            #t4=time.time()
            #self.expanxion.append(t4-t3)
            img_z = Image.fromarray(img_z)
            #t5=time.time()
            r_image     = self.yolo.detect_image(img_z)
            #t6=time.time()
            #self.laptoptime.append(t6-t5)
            temp = np.array(r_image)
            temp = temp[:,:,::-1]
            cv2.imshow("Pointcloud Expansion",cv2.resize(temp,dsize=(640,360)))
            cv2.moveWindow("Pointcloud Expansion", 20, 200)
            t1=time.time()
            # print("object_detection:",(t1-t0))
            # self.countnum.append(t0-self.pretime)
            # print("CIRCULE:",(t1-self.pretime))
            # self.pretime=t1
            # print("FPS:",1/(t1-t0))
            # print("当前帧是:",self.framesavednum)
            self.a=0
            cv2.waitKey(1);               
        self.itemNumArray[self.nextCapBufferPosition] = self.recentCapNum
        self.bufferTimeStamp[self.nextCapBufferPosition] = self.frameTimeStamp
        # 鸽笼原理
        if((self.nextReadBufferPosition-1+self.bufferSize)%self.bufferSize == self.nextCapBufferPosition):
            self.bufferOverWritten = True  # 避
        self.nextCapBufferPosition += 1
        self.nextCapBufferPosition %= self.bufferSize
    def _read_data_packet(self):
        """
        Returns:
            int: Current packet number, byte count of data that has already been read, raw ADC data in current packet
        """
        data, addr = self.data_socket.recvfrom(MAX_PACKET_SIZE)
        self.packetTimeStamp = datetime.datetime.now().timestamp()
        packet_num = struct.unpack('<1l', data[:4])[0]
        byte_count = struct.unpack('>Q', b'\x00\x00' + data[4:10][::-1])[0]
        packet_data = np.frombuffer(data[10:], dtype=np.uint16)
        return packet_num, byte_count, packet_data
# if __name__ == "__main__":
#     proc = adcCapThread(123, "adc")
#     proc.run()
