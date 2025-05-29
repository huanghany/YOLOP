import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw


class LaneParsingHead(nn.Module):
    def __init__(self, in_channels, cls_dim=(37, 10, 4), target_spatial_size=(9, 25)):
        """
        in_channels: 输入特征图的通道数，例如YOLOP的SPP输出(cache[9])是512
        cls_dim: parsingNet的分类维度 (num_gridding, num_cls_per_lane, num_of_lanes)
        target_spatial_size: 特征图经过AdaptiveAvgPool2d后的目标空间尺寸，例如 (9, 25) 对应 1800 的展平尺寸
        """
        super().__init__()
        self.cls_dim = cls_dim
        # 使用 math.prod 替代 np.prod 以保持纯PyTorch/Python风格，兼容Python 3.8+
        # 如果是旧版本Python，可以使用 functools.reduce(operator.mul, cls_dim) 或 np.prod
        self.total_dim = np.prod(cls_dim)

        # 1. Adaptive Pooling: 将输入特征图调整到目标空间尺寸 (9, 25)
        self.avg_pool = nn.AdaptiveAvgPool2d(target_spatial_size)
        self.pool = torch.nn.Conv2d(512, 8, 1)

        # 2. 1x1 卷积层，类似于 parsingNet 中的 self.pool
        # parsingNet的pool层将2048或512通道的特征图转换为8通道
        self.conv_1x1 = nn.Conv2d(in_channels, 8, 1)

        # 3. 计算展平后的维度，确保与Linear层输入匹配
        # 8 * 9 * 25 = 1800
        flattened_size = 8 * target_spatial_size[0] * target_spatial_size[1]

        # 4. 全连接层，与 parsingNet 中的 self.cls 结构相同
        self.cls = nn.Sequential(
            # nn.Linear(flattened_size, 2048),
            nn.Linear(1800, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.total_dim),
        )

        # 初始化权重 (使用YOLOP项目自身的initialize_weights或者parsingNet提供的版本)
        # 这里使用YOLOP项目的initialize_weights，因为它已在当前文件中导入
        # initialize_weights(self.conv_1x1, self.cls)

    def forward(self, x):
        # x 是原始输入图像，而不是中间特征图
        # fea = x
        # print("input:", fea.shape)
        # x 是来自YOLOP骨干网络的特征图 (例如 cache[9])

        # 1. 通过自适应池化调整空间尺寸
        x = self.avg_pool(x)
        # print("after pool:", x.shape)
        # x = self.pool(x)

        # 2. 1x1 卷积
        x = self.conv_1x1(x)

        # 3. 展平特征图，为全连接层做准备
        # 使用x.shape动态获取当前特征图的实际尺寸进行展平
        fea = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        # fea = x.view(-1, 1800)
        # print("after view:", fea.shape)
        # fea = self.pool(x).view(-1, 1800)

        group_cls = self.cls(fea).view(-1, *self.cls_dim)
        # print("group_cls: ", group_cls.shape)
        return group_cls