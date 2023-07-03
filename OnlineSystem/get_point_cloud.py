import sys
import os

this_path = os.path.dirname(__file__)
sys.path.append(os.path.join(this_path, '../'))
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from config_cyc import PointCloudConfig as cfg
from config_cyc import BaseConfig as bfg
import cv2

att_phase_shift_1443 = np.array([
        [0, 0.2443, 0.2560, 0.3664], 
        [0.7842, 1.0626, 1.0997, 1.1946],
        [0.8840, 1.1286, 1.1252, 1.2505]
    ])

'''============= Read One Frame ============='''
class BufferReader:
    def __init__(self,path):
        self.path = path
        self.bin_fid = open(path,'rb')
       
    def getNextBuffer(self):
        buffer = np.frombuffer(self.bin_fid.read(cfg.FRAME_SIZE * 2 * 2), dtype = np.int16)
        return buffer

    def close(self):
        self.bin_fid.close()

    def __del__(self):
        self.close()

def buffer2Frame(buffer):
    frame = np.zeros(shape = cfg.FRAME_SIZE, dtype = np.complex128)
    frame[0::4] = buffer[0::8] + 1j * buffer[4::8]
    frame[1::4] = buffer[1::8] + 1j * buffer[5::8]
    frame[2::4] = buffer[2::8] + 1j * buffer[6::8]
    frame[3::4] = buffer[3::8] + 1j * buffer[7::8]
    frame = np.reshape(frame, (cfg.CHIRP_NUM, cfg.TX_NUM, cfg.SAMPLE_NUM, cfg.RX_NUM))
    frame = frame.transpose(1, 3, 0 ,2)
    # print("buffer2Frame"+str(type(frame)))
    return frame

def attPhaseComps(frame):
    arr_comps = np.reshape(att_phase_shift_1443, (3, 4, 1, 1))
    arr_comps = np.exp(-1 * 1j * arr_comps, dtype = np.complex128)
    frame = arr_comps * frame
    # print("attphase"+str(type(frame)))
    return frame

def rangeFFT(frame):
    r_win = np.blackman(cfg.SAMPLE_NUM)
    frame = np.fft.fft(frame * r_win, axis = -1)
    # print("ranngefft"+str(type(frame)))
    return frame

def rangeCut(frame):
    frame = frame[:, :, :,  : cfg.MAX_R_I]
    frame[:, :, :, : cfg.MIN_R_I] = 0
    # print("rangecut"+str(type(frame)))
    return frame

def clutterRemoval(frame):
    frame = frame.transpose(2, 1, 0, 3)
    mean = frame.mean(0)
    frame = frame - mean
    # print("cluterremoval"+str(type(frame.transpose(2,1,0,3))))
    return frame.transpose(2, 1, 0, 3)

def dopplerFFT(frame):
    v_win = np.reshape(np.hanning(cfg.CHIRP_NUM), (1, 1, -1, 1))
    frame = np.fft.fft(frame * v_win, axis = 2)
    frame = np.fft.fftshift(frame, axes = 2)
    # print("doopler"+str(type(frame)))
    return frame

def tdmPhaseComps(frame):
    for m in range(cfg.CHIRP_NUM):
        for n in range(1, cfg.TX_NUM):
            frame[n, :, m, :] *= np.exp(-1j * n * 2 * np.pi / 3 * (m - cfg.CHIRP_NUM // 2) / cfg.CHIRP_NUM)
    # print("tdmphase"+str(type(frame)))
    return frame

def frameHPF(frame):
    temp_num = cfg.CHIRP_NUM * cfg.MAX_R_I - cfg.HPF_NUM
    doppler_db = np.abs(np.sum(frame, axis = (0, 1)))
    rss_thr = np.partition(doppler_db.ravel(), temp_num - 1)[temp_num - 1]
    frame_filter =  doppler_db > rss_thr
    filter_indices = np.argwhere(frame_filter == 1)
    return doppler_db, frame_filter, filter_indices

def angleFFT_1(frame):
    ''' deprecated function '''
    azimuth_frame = np.concatenate((frame[0, :, :, :], frame[1, :, :, :]), axis = 0)
    elevation_frame = frame[2, :, :, :]
    azimuth_frame = np.fft.fft(azimuth_frame, cfg.ANGLE_PADDED_NUM, axis = 0)
    azimuth_frame = np.fft.fftshift(azimuth_frame, axes = 0)
    elevation_frame = np.fft.fft(elevation_frame, cfg.ANGLE_PADDED_NUM, axis = 0)
    elevation_frame = np.fft.fftshift(elevation_frame, axes = 0)
    return azimuth_frame, elevation_frame

def getRawPC_1(doppler_db, frame_filter, azimuth_frame, elevation_frame):
    ''' deprecated function '''
    point_set = np.zeros((6, cfg.HPF_NUM))
    point_count = 0
    for r_i in range(cfg.MIN_R_I, cfg.MAX_R_I):
        for v_i in range(cfg.CHIRP_NUM):
            if frame_filter[v_i, r_i]:
                param_R = r_i * cfg.RANGE_RES
                param_V = (v_i - cfg.CHIRP_NUM // 2) * cfg.VELOCITY_RES
                param_RSS = doppler_db[v_i, r_i]
                if param_RSS != 0:
                    wx, wz = getWxWz(azimuth_frame[:, v_i, r_i], elevation_frame[:, v_i, r_i])
                    param_X = param_R * wx / np.pi
                    param_Z = param_R * wz / np.pi
                    y_ = param_R ** 2 - param_X ** 2 - param_Z ** 2
                    if y_ >= 0:
                        param_Y = np.sqrt(y_)
                        point_set[:, point_count] = np.array((
                                param_X, param_Y, param_Z,
                                param_V, param_RSS, param_R
                            ))
                        point_count += 1
    return point_set[:, : point_count]


def getWxWz(azimuth_fft, elevation_fft):
    a_i = np.argmax(np.abs(azimuth_fft))
    e_i = np.argmax(np.abs(elevation_fft))
    wx = (a_i - cfg.ANGLE_PADDED_NUM // 2) / cfg.ANGLE_PADDED_NUM * 2 * np.pi
    wz = np.angle(azimuth_fft[a_i] * elevation_fft[e_i].conj() * np.exp(2j * wx))
    return wx, wz


def angleFFT(frame, frame_filter):
    f_frame = frame[:, :, frame_filter == True]
    f_frame = f_frame.reshape(12, -1)
    azimuth_frame = f_frame[: 2 * cfg.RX_NUM, :]
    elevation_frame = f_frame[2 * cfg.RX_NUM :, :]
    azimuth_frame = np.fft.fft(azimuth_frame, cfg.ANGLE_PADDED_NUM, axis = 0)
    azimuth_frame = np.fft.fftshift(azimuth_frame, axes = 0)
    elevation_frame = np.fft.fft(elevation_frame, 8, axis = 0)
    elevation_frame = np.fft.fftshift(elevation_frame, axes = 0)
    return azimuth_frame, elevation_frame

def getRawPC(doppler_db, frame_filter, filter_indices, azimuth_frame, elevation_frame):
    param_R = filter_indices[:, 1] * cfg.RANGE_RES
    param_V = (filter_indices[:, 0] - cfg.CHIRP_NUM // 2) * cfg.VELOCITY_RES
    list_wx = []
    list_wz = []
    azimuth_frame = azimuth_frame.transpose(1, 0)
    elevation_frame = elevation_frame.transpose(1, 0)
    for a_f, e_f in zip(azimuth_frame, elevation_frame):
        wx, wz = getWxWz(a_f, e_f)
        list_wx.append(wx)
        list_wz.append(wz)
    param_Wx = np.array(list_wx)
    param_Wz = np.array(list_wz)
    param_X = param_Wx / np.pi
    param_Z = param_Wz / np.pi
    param_Y = 1 - param_X ** 2 - param_Z ** 2
    error_xyz = param_Y < 0
    param_X[error_xyz] = 0
    param_Y[error_xyz] = 0
    param_Z[error_xyz] = 0
    param_Y = np.sqrt(param_Y)
    param_X *= param_R
    param_Y *= param_R
    param_Z *= param_R
    param_RSS = np.log10(doppler_db[frame_filter])
    point_cloud = np.concatenate((param_X, param_Y, param_Z, param_R, param_V, param_RSS))
    # print("getrawpc"+str(type(point_cloud.reshape(6, -1))))
    return point_cloud.reshape(6, -1)

def getAxisLimitPC(point_cloud):
    point_cloud = point_cloud[:, np.all([
            point_cloud[0, :] > cfg.LIMIT_X[0],
            point_cloud[1, :] > cfg.LIMIT_Y[0],
            point_cloud[2, :] > cfg.LIMIT_Z[0],
            point_cloud[0, :] < cfg.LIMIT_X[1],
            point_cloud[1, :] < cfg.LIMIT_Y[1],
            point_cloud[2, :] < cfg.LIMIT_Z[1],
            abs(point_cloud[4, :]) < cfg.VelocityMax
        ], axis = 0)]
    # print("getaxislimit"+str(type(point_cloud)))
    return point_cloud

def getKmeansPC(point_cloud):
    kmeans_res = KMeans(n_clusters = cfg.GROUP_NUM).fit_predict(point_cloud[: 3, :].transpose(1, 0))
    kmeans_res = kmeans_res[np.newaxis, :]
    point_cloud = np.concatenate((point_cloud, kmeans_res), axis = 0)
    return point_cloud

def getFilledPC(point_cloud):
    ''' Fill the point clouds to match the rest_num '''
    if point_cloud.shape[1] > cfg.REST_NUM:
        # too many points then only get the top rest_num highest power points
        thr_pos = point_cloud.shape[1] - cfg.REST_NUM
        rss_thr = np.partition(point_cloud[4, :], thr_pos - 1)[thr_pos - 1]
        point_cloud = point_cloud[:, point_cloud[4, :] > rss_thr]
    else:
        # too few points then duplicate the point cloud
        if point_cloud.shape[1] < cfg.REST_NUM // 2:
            return None
        # while point_cloud.shape[1] <= cfg.REST_NUM // 2:
        #     point_cloud = np.concatenate((point_cloud, point_cloud), axis = 1)
        filled_size = cfg.REST_NUM - point_cloud.shape[1]
        if filled_size:
            filled_dims = np.random.choice(point_cloud.shape[1], filled_size, replace = False)
            point_cloud = np.concatenate((point_cloud, point_cloud[:, filled_dims]), axis = 1)
    # print("getfilledpc"+str(type(point_cloud)))
    return point_cloud

def buffer2PointCloud(buffer):
    frame = buffer2Frame(buffer)
    frame = attPhaseComps(frame)
    frame = rangeFFT(frame)
    frame = rangeCut(frame)
    frame = clutterRemoval(frame)
    frame = dopplerFFT(frame)
    frame = tdmPhaseComps(frame)
    doppler_db, frame_filter, filter_indices = frameHPF(frame)
    # x, y = angleFFT_1(frame)
    # point_cloud = getRawPC_1(doppler_db, frame_filter, x, y)
    x, y = angleFFT(frame, frame_filter)
    point_cloud = getRawPC(doppler_db, frame_filter, filter_indices, x, y)
    point_cloud = getAxisLimitPC(point_cloud)
    # print("hh到这里还是好的")
    # print(type(point_cloud))
    return point_cloud

def countGroupKmeans(kmeans):
    g_p_count = np.zeros(cfg.GROUP_NUM)
    for i in range(cfg.REST_NUM):
        g_i = int(kmeans[i])
        g_p_count[g_i] += 1
    print(g_p_count)
def pointcloudtoimage(point_cloud):
    point_cloud = point_cloud[:,np.argsort(point_cloud[1,:])]
    u = (point_cloud[0,:]*bfg.CAM_FX/-point_cloud[1,:]+bfg.CAM_CX)
    v = (point_cloud[2,:]*bfg.CAM_FY/-point_cloud[1,:]+bfg.CAM_CY)
    u=u.astype(np.int16)
    v=v.astype(np.int16)
    img_z =np.zeros((bfg.CAM_HGT,bfg.CAM_WID,3),dtype=np.uint8)
    point_cloud=abs(point_cloud)
    for i in range(point_cloud.shape[1]):
        theta=math.atan(((u[i])-639)/608);
        thetares=1/4/math.cos(theta);
        righturange=abs((math.tan(theta)-math.tan(theta-thetares))*bfg.CAM_FX);
        lefturange =abs((math.tan(theta)-math.tan(theta+thetares))*bfg.CAM_FX);
        pinv=int((lefturange+righturange)/128);
        lefturange=int(lefturange/8)
        img_z[v[i]+1-pinv:v[i]+1+pinv,bfg.CAM_WID-u[i]+1-lefturange:bfg.CAM_WID-u[i]+1+lefturange,0]=point_cloud[3,i]/7.5*255
        img_z[v[i]+1-pinv:v[i]+1+pinv,bfg.CAM_WID-u[i]+1-lefturange:bfg.CAM_WID-u[i]+1+lefturange,1]=point_cloud[4,i]/10*255
        img_z[v[i]+1-pinv:v[i]+1+pinv,bfg.CAM_WID-u[i]+1-lefturange:bfg.CAM_WID-u[i]+1+lefturange,2]+=10
    return img_z
    
def testOneFrame(is_figure = False):
    # file_path = os.path.abspath(os.path.join(this_path, '../../Data/TestDataPath/RawData/2022-01-30_22-16-04'))
    file_path = r'E:\mmwave\Data\finaldataset\Sunday\nfovnear\2023-02-28-13-12-44\\'
    file_name = 'southner_data_Raw_0.bin'
    file = os.path.join(file_path, file_name)
    
    f_reader = BufferReader(file)
    for i in range(1):
        buffer = f_reader.getNextBuffer()
    t_0 = time.time()
    point_cloud = buffer2PointCloud(buffer)
    t_1 = time.time()
    img_z      = pointcloudtoimage(point_cloud)
    t_2 = time.time()
    print("点云处理时间：",t_1 - t_0)
    print("pct expansion时间：",t_2 - t_1)
    cv2.imshow("Pointcloud Expansion",img_z)
    cv2.waitKey(1);
    print('point num: ', point_cloud.shape[1])

    if is_figure:
        cv2.imshow("Pointcloud Expansion",img_z)
        cv2.waitKey(1);
        # ax = plt.subplot(projection = '3d')
        # ax.set_title('point cloud')
        # ax.scatter(point_cloud[0, :], point_cloud[1, :], point_cloud[2, :], c = point_cloud[4, :])
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.show()

if __name__ == '__main__':
    testOneFrame(True)