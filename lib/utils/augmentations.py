# -*- coding: utf-8 -*-

import numpy as np
import cv2
import random
import math


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    """change color hue, saturation, value"""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])


def random_perspective(combination, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    """
    对图像组合进行随机几何变换（仿射或透视），并相应地调整边界框标签。

    Inputs:
    - combination: 图像和掩码的元组，例如 (主图像, 分割掩码, 车道线掩码, [可选] 机器人车道线掩码)。
                   假定第一个元素是彩色图像，其余是单通道掩码。
    - targets: 目标检测的边界框标签，格式为 [class, x1, y1, x2, y2]。
    - degrees: 旋转角度范围。
    - translate: 翻译（平移）比例。
    - scale: 缩放比例。
    - shear: 剪切角度范围。
    - perspective: 透视变换强度。
    - border: 边界填充，格式为 (高度填充, 宽度填充)。

    Returns:
    - transformed_combination: 经过变换后的图像和掩码的元组。
    - targets: 经过变换后调整的边界框标签。
    """

    # 获取主图像，用于确定尺寸
    img_main = combination[0]
    height = img_main.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img_main.shape[1] + border[1] * 2

    # Center (中心化)
    C = np.eye(3)
    C[0, 2] = -img_main.shape[1] / 2  # x 平移 (像素)
    C[1, 2] = -img_main.shape[0] / 2  # y 平移 (像素)

    # Perspective (透视)
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x 透视 (关于 y 轴)
    P[2, 1] = random.uniform(-perspective, perspective)  # y 透视 (关于 x 轴)

    # Rotation and Scale (旋转和缩放)
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear (剪切)
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x 剪切 (度)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y 剪切 (度)

    # Translation (平移)
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x 平移 (像素)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y 平移 (像素)

    # Combined transformation matrix (组合变换矩阵)
    M = T @ S @ R @ P @ C  # 运算顺序 (从右到左) 非常重要

    transformed_combination = []
    # 检查是否需要进行变换 (如果矩阵 M 不是单位矩阵，或者有边界填充)
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # 图像已改变
        for i, item in enumerate(combination):
            # 确定 borderValue: 第一个元素 (主图像) 为彩色填充，其他为黑色填充 (掩码)
            if i == 0:
                border_val = (114, 114, 114)
                interpolation_method = cv2.INTER_LINEAR
            else:
                border_val = 0  # 掩码通常用0填充
                interpolation_method = cv2.INTER_NEAREST
            if perspective:
                transformed_item = cv2.warpPerspective(item, M, dsize=(width, height), borderValue=border_val, flags=interpolation_method)
            else:  # affine (仿射)
                transformed_item = cv2.warpAffine(item, M[:2], dsize=(width, height), borderValue=border_val, flags=interpolation_method)
            transformed_combination.append(transformed_item)
    else:  # 如果没有变换，直接返回原始组合的副本
        transformed_combination = list(combination)  # 转换为列表再转元组，避免直接修改原始元组

    # Transform label coordinates (变换标签坐标)
    n = len(targets)
    if n:
        # warp points (变换点)
        xy = np.ones((n * 4, 3))
        # 从 [x1, y1, x2, y2] 提取四个角点并展平: (x1,y1), (x2,y2), (x1,y2), (x2,y1)
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = xy @ M.T  # 应用变换矩阵
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # 透视变换需要归一化
        else:  # affine (仿射)
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes (创建新的边界框)
        x = xy[:, [0, 2, 4, 6]]  # 提取所有 x 坐标
        y = xy[:, [1, 3, 5, 7]]  # 提取所有 y 坐标
        # 找到新的 x_min, y_min, x_max, y_max
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes (裁剪边界框到图像范围内)
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates (过滤无效的边界框)
        # 这里使用了 _box_candidates 函数，假设它在其他地方定义，用于过滤变换后可能变得过小的框
        # 为了让这个片段可运行，我将暂时注释掉或假设 _box_candidates 的简单行为
        # i = _box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        # targets = targets[i]
        # targets[:, 1:5] = xy[i]

        # 简单过滤：移除宽度或高度小于1像素的框
        valid_indices = (xy[:, 2] - xy[:, 0] > 1) & (xy[:, 3] - xy[:, 1] > 1)
        targets = targets[valid_indices]
        targets[:, 1:5] = xy[valid_indices]

    return tuple(transformed_combination), targets


def cutout(combination, labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    image, gray = combination
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)
        # print('xmin:{},ymin:{},xmax:{},ymax:{}'.format(xmin,ymin,xmax,ymax))

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]
        gray[ymin:ymax, xmin:xmax] = -1

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return image, gray, labels


def letterbox(combination, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    """Resize the input image and automatically padding to suitable shape :https://zhuanlan.zhihu.com/p/172121380"""
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232

    if len(combination) == 4:
        img, gray, line, lane_robot = combination
    else:
        img, gray, line = combination
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        gray = cv2.resize(gray, new_unpad, interpolation=cv2.INTER_LINEAR)
        line = cv2.resize(line, new_unpad, interpolation=cv2.INTER_LINEAR)
        if len(combination) == 4:
            lane_robot = cv2.resize(lane_robot, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    gray = cv2.copyMakeBorder(gray, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)  # add border
    line = cv2.copyMakeBorder(line, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)  # add border
    if len(combination) == 4:
        lane_robot = cv2.copyMakeBorder(lane_robot, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=0)  # add border
    # print(img.shape)

    # combination = (img, gray, line)
    if len(combination) == 4:
        combination = (img, gray, line, lane_robot)
    else:
        combination = (img, gray, line)
    return combination, ratio, (dw, dh)


def letterbox_for_img(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding

    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def _box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates
