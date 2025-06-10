import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw


class LaneParsingHead(nn.Module):
    def __init__(self, in_channels, cls_dim=(101, 56, 2)):
        """
        in_channels: 输入特征图的通道数，例如YOLOP的SPP输出(cache[9])是512
        cls_dim: parsingNet的分类维度 (num_gridding, num_cls_per_lane, num_of_lanes)
        """
        super().__init__()
        self.cls_dim = cls_dim
        self.total_dim = math.prod(cls_dim)

        # parsingNet的pool层将2048或512通道的特征图转换为8通道
        self.conv_1x1 = nn.Conv2d(in_channels, 8, 1)

        # 全连接层，与 parsingNet 中的 self.cls 结构相同
        self.cls = nn.Sequential(
            # nn.Linear(flattened_size, 2048),
            nn.Linear(2560, 2048),  # 1, 8,16,20
            nn.ReLU(),
            nn.Linear(2048, self.total_dim),
        )
        # 初始化权重
        # initialize_weights(self.conv_1x1, self.cls)

    def forward(self, x):

        # print("x: ", x.shape)
        # 2. 1x1 卷积
        x = self.conv_1x1(x)

        # 使用x.shape动态获取当前特征图的实际尺寸进行展平
        fea = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])

        # fea = self.pool(x).view(-1, 1800)

        group_cls = self.cls(fea).view(-1, *self.cls_dim)
        # print("group_cls: ", group_cls.shape)
        return group_cls