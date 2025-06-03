import torch
from torch import tensor
import torch.nn as nn
import sys, os
import math
import sys

from lib.models.common3 import LaneParsingHead

sys.path.append(os.getcwd())
from lib.utils import initialize_weights
from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric


YOLOP_lane_robot = [
    [24, 33, 42, 43],  # Det_out_idx, Da_Segout_idx, LL_Segout_idx  添加了新头在索引43
    [-1, Focus, [3, 32, 3]],  # 0            1 12 320x320(输出尺寸）
    [-1, Conv, [32, 64, 3, 2]],  # 1 conv    1 64 160 160
    [-1, BottleneckCSP, [64, 64, 1]],  # 2   1 64 160 160
    [-1, Conv, [64, 128, 3, 2]],  # 3 conv   1 128 80 80
    [-1, BottleneckCSP, [128, 128, 3]],  # 4 1 128 80 80
    [-1, Conv, [128, 256, 3, 2]],  # 5       1 256 40 40
    [-1, BottleneckCSP, [256, 256, 3]],  # 6 1 256 40 40
    [-1, Conv, [256, 512, 3, 2]],  # 7       1 512 20 20
    [-1, SPP, [512, 512, [5, 9, 13]]],  # 8 SPP 1 512 20 20
    [-1, BottleneckCSP, [512, 512, 1, False]],  # 9 1 512 20 20
    [-1, Conv, [512, 256, 1, 1]],  # 10        1 256 20 20
    [-1, Upsample, [None, 2, 'nearest']],  # 11 1 256 40 40
    [[-1, 6], Concat, [1]],  # 12            1 512 40 40
    [-1, BottleneckCSP, [512, 256, 1, False]],  # 13   1 256 40 40
    [-1, Conv, [256, 128, 1, 1]],  # 14               1 128 40 40
    [-1, Upsample, [None, 2, 'nearest']],  # 15       1 128 80 80
    [[-1, 4], Concat, [1]],  # 16         #Encoder  1 256 80 80

    [-1, BottleneckCSP, [256, 128, 1, False]],  # 17    1 128 80 80
    [-1, Conv, [128, 128, 3, 2]],  # 18             1 128 40 40
    [[-1, 14], Concat, [1]],  # 19                 1 256 40 40
    [-1, BottleneckCSP, [256, 256, 1, False]],  # 20    1 256 40 40
    [-1, Conv, [256, 256, 3, 2]],  # 21             1 256 20 20
    [[-1, 10], Concat, [1]],  # 22                     1 512 20 20
    [-1, BottleneckCSP, [512, 512, 1, False]],  # 23    1 512 20 20
    [[17, 20, 23], Detect,
     [1, [[3, 9, 5, 11, 4, 20], [7, 18, 6, 39, 12, 31], [19, 50, 38, 81, 68, 157]], [128, 256, 512]]],
    # Detection head 24 3(20x20+40x40+80x80)  25200

    [16, Conv, [256, 128, 3, 1]],  # 25              1 128 80 80
    [-1, Upsample, [None, 2, 'nearest']],  # 26       1 128 160 160
    [-1, BottleneckCSP, [128, 64, 1, False]],  # 27   1 64 160 160
    [-1, Conv, [64, 32, 3, 1]],  # 28               1 32 160 160
    [-1, Upsample, [None, 2, 'nearest']],  # 29       1 32 320 320
    [-1, Conv, [32, 16, 3, 1]],  # 30               1 16 320 320
    [-1, BottleneckCSP, [16, 8, 1, False]],  # 31   1 8 320 320
    [-1, Upsample, [None, 2, 'nearest']],  # 32       1 8 640 640
    [-1, Conv, [8, 2, 3, 1]],  # 33 Driving area segmentation head 1 2 640 640

    [16, Conv, [256, 128, 3, 1]],  # 34
    [-1, Upsample, [None, 2, 'nearest']],  # 35
    [-1, BottleneckCSP, [128, 64, 1, False]],  # 36
    [-1, Conv, [64, 32, 3, 1]],  # 37
    [-1, Upsample, [None, 2, 'nearest']],  # 38
    [-1, Conv, [32, 16, 3, 1]],  # 39
    [-1, BottleneckCSP, [16, 8, 1, False]],  # 40
    [-1, Upsample, [None, 2, 'nearest']],  # 41
    [-1, Conv, [8, 2, 3, 1]],  # 42 Lane line segmentation head 1 2 640

    # 新添加的车道线头 (Parsing Head)
    # 输入来自模型第9层的输出 (SPP后的BottleneckCSP), 它的通道数是512
    [9, LaneParsingHead, [512, (101, 56, 2)]]  # 43: Lane pasrsing head
]


class YOLOP_Lane_net(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(YOLOP_Lane_net, self).__init__()
        layers, save = [], []
        self.nc = 1
        self.detector_index = -1
        self.detector_index = block_cfg[0][0]
        self.da_seg_index = block_cfg[0][1]
        self.ll_seg_index = block_cfg[0][2]

        # 检查是否存在新的车道线解析头索引
        self.ll_parsing_index = block_cfg[0][3] if len(block_cfg[0]) > 3 else -1

        # Build model 构建模型
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # 层名转换
            if block is Detect:  # 检测
                self.detector_index = i
            block_ = block(*args)  # 构建一层
            block_.index, block_.from_ = i, from_
            layers.append(block_)  # 加入网络
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        assert self.detector_index == block_cfg[0][0]  # 检验

        self.model, self.save = nn.Sequential(*layers), sorted(save)  # 赋值给model
        self.names = [str(i) for i in range(self.nc)]  # 类别

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect):
            s = 320  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, 256, s))  # 前向传播
                detects, _, _, _ = model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # 为相应的比例设置锚点
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()

        initialize_weights(self)  # 初始化权重

    def forward(self, x):
        cache = []  # 用于存储中间层的输出
        # 初始化所有分支的输出
        det_out = None
        da_seg_out = None
        ll_seg_out = None
        ll_parsing_out = None
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                # 根据from_属性获取输入，可以是单个层或多个层（用于Concat）
                x = cache[block.from_] if isinstance(block.from_, int) else \
                    [x if j == -1 else cache[j] for j in block.from_]

            x = block(x)  # 执行当前层的前向传播
            # print(x.shape)
            # 根据索引将输出分配给对应的分支
            if i == self.detector_index:
                det_out = x
            elif i == self.da_seg_index:
                da_seg_out = torch.sigmoid(x)  # 驾驶区域分割通常需要sigmoid
            elif i == self.ll_seg_index:
                ll_seg_out = torch.sigmoid(x)  # 车道线分割通常需要sigmoid
            elif i == self.ll_parsing_index and self.ll_parsing_index != -1:
                ll_parsing_out = x  # 车道线解析头输出通常是原始logits，不直接sigmoid

            # 将当前层的输出加入缓存，如果它被后续层使用
            cache.append(x if block.index in self.save else None)

        # 返回所有分支的输出
        return det_out, da_seg_out, ll_seg_out, ll_parsing_out

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def get_YOLOP_LANE_net(cfg, **kwargs):
    """
    获取网络
    Args:
        cfg:
        **kwargs:

    Returns:

    """
    m_block_cfg = YOLOP_lane_robot
    model = YOLOP_Lane_net(m_block_cfg, **kwargs)
    return model


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter

    model = get_YOLOP_LANE_net(False)  # cfg参数通常用于加载预训练模型，这里为False

    # 假设输入尺寸为640x640，与YOLOP论文中常用尺寸一致
    input_ = torch.randn((1, 3, 256, 320))

    # 执行前向传播，现在应该有4个输出
    det_out, dring_area_seg, lane_line_seg, lane_parsing_out = model(input_)

    # 打印各个输出的形状
    print("Detection outputs:")
    for det in det_out:
        print(det.shape)

    print("\nDriving Area Segmentation shape:", dring_area_seg.shape)
    print("Lane Line Segmentation shape:", lane_line_seg.shape)
    print("Lane Parsing Head output shape:", lane_parsing_out.shape)

    # 验证输出维度 (例如，对于 (37, 10, 4) 的 cls_dim)
    # batch_size * num_gridding * num_cls_per_lane * num_of_lanes
    expected_parsing_dim = (1, 101, 56, 2)
    assert lane_parsing_out.shape == expected_parsing_dim, \
        f"Lane parsing output shape mismatch! Expected {expected_parsing_dim}, got {lane_parsing_out.shape}"
    print("\nLane Parsing Head output shape matches expected dimension:",
          lane_parsing_out.shape == expected_parsing_dim)

    # 如果需要，可以继续进行tensorboard可视化或进一步测试
    # writer = SummaryWriter('./runs/yolop_new_head')
    # writer.add_graph(model, input_)
    # writer.close()
