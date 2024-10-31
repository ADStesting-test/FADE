
LGSVL_HD_MAP = '12da60a7-2fc9-474d-a62a-5cc08cb97fe8'# San Francisco
# LGSVL_HD_MAP = 'aae03d2a-b7ca-4a88-9e41-9035287a12cc' # Borregas Ave
APOLLO_HD_MAP = 'San Francisco'
# APOLLO_HD_MAP = 'Borregas Ave'
LGSVL_VEHICLE = 'Lincoln2017MKZ LGSVL'
EGO_VEHILCE_TYPE = 'ff12e21b-b8c6-41f0-8f42-a7846f1996b7'  # camera and lidar
# EGO_VEHILCE_TYPE = '2e966a70-4a19-44b5-a5e7-64e00a7bc5de' # 3D ground truth 
DEFAULT_TIMEOUT=30

# 启动模块
MODULES = [ 
        'Localization',
        'Perception',
        'Transform',
        'Routing',
        'Prediction',
        'Planning',
        # 'Camera',
        'Traffic Light',
        'Control',
]

DV_MODULES = [
    'Recorder'
]

ENABLE_RECORD = True # 是否开启录制

TOTAL_SIM_TIME = 30 # 仿真时间
TIME_SLICE_SIZE = 10 # 仿真时间切片


LOOP_NUM = 1 # 仿真次数

SPEED_PEDESTRIAN = 1.55 # 行人速度
SPEED_NPC = 9.1 # NPC速度


FILE_PATH = '/home/zlding/Code/SVL/PythonAPI/Mytest/data'
SAVE_LIDAR = False
SAVE_IMAGE = False
SCENE_STR = 'leftTurn_right_'  