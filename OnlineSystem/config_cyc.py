
'''========== Base Config =========='''
class BaseConfig:

    CAM_WID = 1280; 
    CAM_HGT =720;
    CAM_FX = 608; 
    CAM_FY = 608;
    CAM_CX = 639;
    CAM_CY = 368;  
    EPS = 1.0e-16;
    # 毫米波数据通道
    PC_CHANNAL = 6
    PC_NUM = 128
    # 点云锚点数与每个锚点聚集的点数
    ANCHOR_NUM = 6
    GROUP_NUM = 8
    GROUPED_POINT_NUM = int((PC_NUM // GROUP_NUM))
    # projection dim
    PROJ_FT_DIM = 3

    ''' =============== Point Cloud Config ============='''
class PointCloudConfig:
    LIGHT_V = 3e8
    F_0 = 77e9
    TOTAL_CHIRP_NUM = 240
    SAMPLE_NUM = 256
    TX_NUM = 3
    RX_NUM = 4
    CHIRP_NUM = TOTAL_CHIRP_NUM // TX_NUM
    LAMDA = LIGHT_V / F_0
    SAMPLE_RATE = 6800e3
    ADC_START_TIME = 6e-6
    CHIRP_RAMP_TIME = 83e-6
    IDLE_TIME = 7e-6
    CHIRP_PERIOD = CHIRP_RAMP_TIME + IDLE_TIME
    CHIRP_RAMP_RATE = 47.990e12
    FRAME_NUM = 450
    # FRAME_PERIOD = 25e-3
    FRAME_PERIOD = 50e-3
    FRAME_SIZE = TOTAL_CHIRP_NUM * SAMPLE_NUM * RX_NUM
    VelocityMax = 5
    # extra calculated parameters
    RANGE_RES = SAMPLE_RATE * LIGHT_V / SAMPLE_NUM / CHIRP_RAMP_RATE / 2
    RANGE_MAX = SAMPLE_RATE * LIGHT_V / CHIRP_RAMP_RATE / 2
    VELOCITY_RES = LAMDA / CHIRP_PERIOD / TX_NUM / CHIRP_NUM / 2
    VELOCITY_MAX = LAMDA / CHIRP_PERIOD / TX_NUM / 4
    ANGLE_PADDED_NUM2 = 8
    # Trick 设定
    IS_KMEANS = False
    # 距离限制设定
    MAX_RANGE = 10
    MAX_R_I = int(MAX_RANGE * 2 * CHIRP_RAMP_RATE * SAMPLE_NUM / SAMPLE_RATE / LIGHT_V) + 1
    MAX_R_I = SAMPLE_NUM
    MIN_RANGE = 2
    MIN_R_I = int(CHIRP_RAMP_RATE * SAMPLE_NUM / SAMPLE_RATE / LIGHT_V) + 1
    # 高能量点过滤个数
    HPF_NUM = 128
    # 每帧保留点数
    REST_NUM = BaseConfig.PC_NUM
    # KMeans 簇个数
    GROUP_NUM = BaseConfig.GROUP_NUM
    # 坐标限制
    LIMIT_X = (-2.5, 2.5)
    LIMIT_Y = (1, 10)
    LIMIT_Z = (-1, 1)

    # angle FFT 填充
    ANGLE_PADDED_NUM = 32
    