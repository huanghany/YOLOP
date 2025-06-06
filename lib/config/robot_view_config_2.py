import os
from yacs.config import CfgNode as CN


_C = CN()

_C.LOG_DIR = 'runs/'
_C.GPUS = (0, 1, 2, 3, 4, 5, 6, 7)
_C.WORKERS = 8
_C.PIN_MEMORY = False
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = True       # Resume from the last training interrupt
_C.NEED_AUTOANCHOR = False      # Re-select the prior anchor(k-means)    When training from scratch (epoch=0), set it to be ture!
_C.DEBUG = False
_C.num_seg_class = 2

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True


# common params for NETWORK
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = ''
_C.MODEL.STRU_WITHSHARE = False     # add share_block to segbranch 给分割分支共享模块
_C.MODEL.HEADS_NAME = ['']
_C.MODEL.PRETRAINED = ""
_C.MODEL.PRETRAINED_DET = ""
_C.MODEL.IMAGE_SIZE = [320, 320]  # width * height, ex: 192 * 256
_C.MODEL.GRIFING_NUM = 100
_C.MODEL.EXTRA = CN(new_allowed=True)


# loss params
_C.LOSS = CN(new_allowed=True)
_C.LOSS.LOSS_NAME = ''
_C.LOSS.MULTI_HEAD_LAMBDA = None
_C.LOSS.FL_GAMMA = 0.0  # focal loss gamma
_C.LOSS.CLS_POS_WEIGHT = 1.0  # classification loss positive weights
_C.LOSS.OBJ_POS_WEIGHT = 1.0  # object loss positive weights
_C.LOSS.SEG_POS_WEIGHT = 1.0  # segmentation loss positive weights
_C.LOSS.LANE_ROBOT_SIM_WEIGHT = 1.0  # 添加车道线相似损失权重
_C.LOSS.LANE_ROBOT_SHP_WEIGHT = 0.0  # 添加车道线形状损失权重
_C.LOSS.LANE_ROBOT_GAIN = 0.2  # 车道线增益权重
_C.LOSS.BOX_GAIN = 0.05  # box loss gain
_C.LOSS.CLS_GAIN = 0.5  # classification loss gain
_C.LOSS.OBJ_GAIN = 1.0  # object loss gain
_C.LOSS.DA_SEG_GAIN = 0.2  # driving area segmentation loss gain
_C.LOSS.LL_SEG_GAIN = 0.2  # lane line segmentation loss gain
_C.LOSS.LL_IOU_GAIN = 0.2 # lane line iou loss gain

# DATASET related params
_C.DATASET = CN(new_allowed=True)
_C.DATASET.DATAROOT = '/home/huayi/hhy/YOLOP/Datasets/lane_robot_3/imgs'       # the path of images folder
_C.DATASET.LABELROOT = '/home/huayi/hhy/YOLOP/Datasets/lane_robot_3/person_json'      # the path of det_annotations folder
_C.DATASET.MASKROOT = '/home/huayi/hhy/YOLOP/Datasets/lane_robot_3/da_seg'                # the path of da_seg_annotations folder
_C.DATASET.LANEROOT = '/home/huayi/hhy/YOLOP/Datasets/lane_robot_3/gt_instance_image_012'               # the path of ll_seg_annotations folder
_C.DATASET.LANEROBOTROOT = '/home/huayi/hhy/YOLOP/Datasets/lane_robot_3/gt_instance_image_012'               # the path of ll_seg_annotations folder
_C.DATASET.DATASET = 'RobotViewDataset'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'png'
_C.DATASET.SELECT_DATA = False  # 筛选数据
_C.DATASET.ORG_IMG_SIZE = [480, 640]  # 原始图片长宽 640 640

# training data augmentation
_C.DATASET.FLIP = False  # 水平翻转  不能要
_C.DATASET.SCALE_FACTOR = 0  # 缩放范围 0.25
_C.DATASET.ROT_FACTOR = 10  # 旋转角度
_C.DATASET.TRANSLATE = 0.1  # 平移比例
_C.DATASET.SHEAR = 0.0  # 裁减
_C.DATASET.COLOR_RGB = False  # rgb颜色扰动
_C.DATASET.HSV_H = 0.015  # image HSV-Hue augmentation (fraction)
_C.DATASET.HSV_S = 0.7  # image HSV-Saturation augmentation (fraction)
_C.DATASET.HSV_V = 0.4  # image HSV-Value augmentation (fraction)
# TODO: more augmet params to add


# train
_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.LR0 = 0.001  # initial learning rate (SGD=1E-2, Adam=1E-3)
_C.TRAIN.LRF = 0.2  # final OneCycleLR learning rate (lr0 * lrf)
_C.TRAIN.WARMUP_EPOCHS = 10  # warmup轮次
_C.TRAIN.WARMUP_BIASE_LR = 0.1
_C.TRAIN.WARMUP_MOMENTUM = 0.8

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.937
_C.TRAIN.WD = 0.0005
_C.TRAIN.NESTEROV = True
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 300  # 109-300

_C.TRAIN.VAL_FREQ = 1
_C.TRAIN.BATCH_SIZE_PER_GPU = 4
_C.TRAIN.SHUFFLE = True  # 是否打乱训练数据

_C.TRAIN.IOU_THRESHOLD = 0.2  #
_C.TRAIN.ANCHOR_THRESHOLD = 4.0  # 锚框选择阈值

# if training 3 tasks end-to-end, set all parameters as True
# Alternating optimization 交替优化策略 训练部分分支
_C.TRAIN.SEG_ONLY = False           # Only train two segmentation branchs
_C.TRAIN.DET_ONLY = False           # Only train detection branch
_C.TRAIN.ENC_SEG_ONLY = False       # Only train encoder and two segmentation branchs
_C.TRAIN.ENC_DET_ONLY = False       # Only train encoder and detection branch

# Single task 单任务训练
_C.TRAIN.DRIVABLE_ONLY = False      # Only train da_segmentation task
_C.TRAIN.LANE_ONLY = False          # Only train ll_segmentation task
_C.TRAIN.DET_ONLY = False          # Only train detection task
_C.TRAIN.NO_DRIVABLE = False          # 不训练可通行区域
_C.TRAIN.NO_LL = True          # 不训练原车道线区域


_C.TRAIN.PLOT = True                # 

# testing
_C.TEST = CN(new_allowed=True)
_C.TEST.BATCH_SIZE_PER_GPU = 4
_C.TEST.MODEL_FILE = ''
_C.TEST.SAVE_JSON = False
_C.TEST.SAVE_TXT = False
_C.TEST.PLOTS = True
_C.TEST.NMS_CONF_THRESHOLD  = 0.001
_C.TEST.NMS_IOU_THRESHOLD  = 0.6


def update_config(cfg, args):
    cfg.defrost()
    # cfg.merge_from_file(args.cfg)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir
    
    # if args.conf_thres:
    #     cfg.TEST.NMS_CONF_THRESHOLD = args.conf_thres

    # if args.iou_thres:
    #     cfg.TEST.NMS_IOU_THRESHOLD = args.iou_thres
    


    # cfg.MODEL.PRETRAINED = os.path.join(
    #     cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    # )
    #
    # if cfg.TEST.MODEL_FILE:
    #     cfg.TEST.MODEL_FILE = os.path.join(
    #         cfg.DATA_DIR, cfg.TEST.MODEL_FILE
    #     )

    cfg.freeze()
