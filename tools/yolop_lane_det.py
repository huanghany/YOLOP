import argparse
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import scipy.special
import torch
import torchvision.transforms as transforms
from numpy import random

from multi_task_onnx_det import resize_unscale

# 确保 BASE_DIR 正确指向项目根目录，以便导入 lib
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# 导入自定义库
from lib.config import cfg
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_YOLOP_LANE_net
from lib.dataset import LoadImages  # LoadImages 可以处理图片和视频文件
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box
from lib.core.function import AverageMeter
from tqdm import tqdm

# 定义车道线后处理所需的行锚点
row_anchor = [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
              116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
              168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
              220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
              272, 276, 280, 284]

# 图像标准化转换
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


def visualize_lanes(img, lanes):
    """
    可视化车道线。不同车道类型使用不同颜色绘制。
    args:
        img (np.array): 原始图像（OpenCV格式）。
        lanes (OrderedDict): 包含车道类型和对应点的结构化数据。
    returns:
        np.array: 绘制了车道线的图像。
    """
    # 定义不同车道类型的颜色映射
    color_map = {
        "current_left": (0, 255, 0),  # 绿色 - 当前车道左侧线
        "current_right": (0, 0, 255),  # 红色 - 当前车道右侧线
        # 其他车道类型将默认使用灰色
    }

    img_vis = img.copy()  # 复制图像以避免修改原始图像

    for lane_type, points in lanes.items():
        # 获取车道线颜色，如果不在 color_map 中则默认为灰色
        color = color_map.get(lane_type, (128, 128, 128))  # 默认灰色
        for (x, y) in points:
            # 绘制车道线点
            cv2.circle(img_vis, (x, y), 5, color, -1)  # -1 表示填充圆形
    return img_vis


def detect(cfg, opt):
    """
    主检测函数。
    args:
        cfg (dict): 配置字典。
        opt (argparse.Namespace): 命令行参数。
    """
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'demo')  # 创建日志器

    # 选择设备 (CPU 或 CUDA)
    device = select_device(logger, opt.device)
    os.makedirs(opt.save_dir, exist_ok=True)  # 创建结果保存目录
    half = device.type != 'cpu'  # 是否使用半精度浮点数 (FP16)，仅CUDA支持

    # 加载模型
    model = get_YOLOP_LANE_net(cfg)
    checkpoint = torch.load(opt.weights, map_location=device)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])  # 加载模型权重
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    if half:
        model.half()  # 转换为FP16

    # 设置数据加载器
    # LoadImages 可以处理图片文件、图片文件夹和视频文件
    dataset = LoadImages(opt.source, img_size=opt.img_size)
    # LoadImages 每次只处理一个输入 (一帧或一张图片)，所以 batch_size 始终为 1
    bs = 1

    # 获取类别名称和颜色，用于目标检测框
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # 预热模型 (在第一次推理前运行一次，优化性能)
    # img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)
    img = torch.zeros((1, 3, 256, 320), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    model.eval()  # 设置为评估模式

    # 计时器
    inf_time = AverageMeter()  # 推理时间
    nms_time = AverageMeter()  # NMS时间

    t0 = time.time()  # 记录开始时间

    vid_path, vid_writer = None, None  # 初始化视频写入器变量

    # 遍历数据集中的每张图片/视频帧
    for i, (path, img_tensor, img_orig_det, vid_cap, shapes) in tqdm(enumerate(dataset), total=len(dataset)):
        # img_tensor: 经过预处理 (resize, normalize, to_tensor) 的 PyTorch Tensor 图像
        # img_orig_det: 原始的 OpenCV 图像，用于后续绘制
        # vid_cap: 如果是视频输入，则为 cv2.VideoCapture 对象；否则为 None
        # shapes: 图像缩放的原始比例和填充信息

        # 将图像发送到设备 (GPU/CPU) 并调整精度
        img_tensor = transform(img_tensor).to(device)
        img_tensor = img_tensor.half() if half else img_tensor.float()  # uint8 to fp16/32
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)  # 增加批次维度

        # 模型推理
        t1 = time_synchronized()
        det_out, drive_area_result, _, lane_robot_result = model(img_tensor)  # 只取目标检测和车道线输出
        t2 = time_synchronized()
        inf_time.update(t2 - t1, img_tensor.size(0))  # 更新推理时间

        # 目标检测后处理 (NMS)
        t3 = time_synchronized()
        det_pred = non_max_suppression(det_out[0], conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None,
                                       agnostic=False)
        t4 = time_synchronized()
        nms_time.update(t4 - t3, img_tensor.size(0))  # 更新 NMS 时间
        det = det_pred[0]  # 获取当前图像的检测结果

        h_orig, w_orig, _ = img_orig_det.shape

        result_img = img_orig_det.copy()

        if drive_area_result is not None:
            # da_output = drive_area_result[0]
            pad_dh, pad_dw = 8, 0
            new_unpad_h, new_unpad_w = 240, 320
            da_output = drive_area_result[0, :, pad_dh: pad_dh + new_unpad_h, pad_dw: pad_dw + new_unpad_w]

            da_seg_mask = torch.argmax(da_output, dim=0).byte().cpu().numpy()
            # Resize mask to original image size
            da_seg_mask_resized = cv2.resize(da_seg_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

            # Create a color overlay for the drivable area
            drivable_area_color_bgr = [0, 200, 0]  # Light green in BGR format

            # Create an image for the overlay. Initialize with zeros (black).
            color_overlay_da = np.zeros_like(img_orig_det, dtype=np.uint8)
            # Where the mask is 1 (drivable), set the color.
            color_overlay_da[da_seg_mask_resized == 1] = drivable_area_color_bgr

            # Blend the overlay with the result_img
            # result_img = original_img * (1-alpha) + color_overlay_da * alpha
            alpha_da = 0.3  # Transparency of the drivable area
            cv2.addWeighted(color_overlay_da, alpha_da, result_img, 1 - alpha_da, 0, result_img)

        # 但在实际项目中，请直接修改顶部的 postprocess_lanes 函数
        def _postprocess_lanes_with_dims(output, griding_num, actual_img_w, actual_img_h):
            out = output[0].data.cpu().numpy()
            out = out[:, ::-1, :]

            prob = scipy.special.softmax(out[:-1, :, :], axis=0)
            idx = np.arange(100).reshape(-1, 1, 1) + 1
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out, axis=0)
            loc[out_j == 100] = 0

            lanes = OrderedDict({
                "current_left": [],
                "current_right": [],
            })
            model_w, model_h = 320, 256  # 模型输入尺寸
            col_sample = np.linspace(0, model_w - 1, griding_num)
            col_sample_w = col_sample[1] - col_sample[0]

            for lane_idx in range(out.shape[2]):
                lane = []
                for point_idx in range(out.shape[1]):
                    if loc[point_idx, lane_idx] > 0:
                        x = int(loc[point_idx, lane_idx] * col_sample_w * actual_img_w / model_w)
                        y = int(row_anchor[::-1][point_idx] * actual_img_h / 288)
                        # lane.append((x, y))
                        y_anchor_on_288 = row_anchor[point_idx]
                        y_model = y_anchor_on_288 * (256 / 288)
                        # 步骤 2: 反向Padding，转换到(240, 320)空间
                        y_resized = y_model - 8
                        # 步骤 3: 过滤无效点 (检查点是否在padding区域之外)
                        if 0 <= y_resized < 240:
                            # 步骤 4: 反向缩放，转换到最终的原图(480, 640)空间
                            y_final = y_resized * (actual_img_h / 240)
                            lane.append((int(x), int(y_final)))
                        else:
                            lane.append((None, None))  # 点在padding区域，舍弃

                lane_type = {
                    0: "current_left",
                    1: "current_right",
                }.get(lane_idx, f"other_lane_{lane_idx}")
                lanes[lane_type] = lane
            return lanes

        # 使用新的后处理函数
        lanes = _postprocess_lanes_with_dims(lane_robot_result, opt.griding_num, w_orig, h_orig)

        # 复制原始图像，以便在其上绘制所有结果
        # result_img = img_orig_det.copy()

        # 绘制目标检测框
        if len(det):
            # 将检测框坐标缩放回原始图像尺寸
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], result_img.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, result_img, label=label_det_pred, color=colors[int(cls)], line_thickness=2)

        # 绘制车道线
        result_img = visualize_lanes(result_img, lanes)

        # --- 视频写入逻辑 ---
        # 检查是否是视频文件，且是否是首次处理
        is_video = vid_cap is not None
        if is_video and opt.save:
            if vid_writer is None:
                # 构建输出视频路径，保留原视频文件名，但确保后缀为.mp4
                save_video_path = str(Path(opt.save_dir) / (Path(path).stem + '.mp4'))

                # 获取原始视频的帧率、宽度和高度
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                # 使用 img_orig_det 的实际宽高，而不是 vid_cap 的原始宽高，
                # 因为 LoadImages 可能已经对 img_orig_det 进行了一些隐式调整
                output_width, output_height = result_img.shape[1], result_img.shape[0]

                # 定义视频编码器
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'XVID'/'MJPG'

                vid_writer = cv2.VideoWriter(save_video_path, fourcc, fps, (output_width, output_height))
                print(f"创建输出视频文件: {save_video_path}, FPS: {fps}, 尺寸: {output_width}x{output_height}")

            # 写入当前帧
            vid_writer.write(result_img)
        elif opt.save:  # 如果是图片且 opt.save 为 True
            save_image_path = str(Path(opt.save_dir) / Path(path).name)
            cv2.imwrite(save_image_path, result_img)
            # print(f"结果已保存至: {save_image_path}") # 避免在 tqdm 中频繁打印

        # 显示结果
        if opt.show:
            cv2.imshow('YOLOP Inference Result', result_img)
            # 对于视频，等待1毫秒以显示帧并响应按键；对于图片，也等待1毫秒
            # 按 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户按下'q'键，停止处理。")
                break  # 退出循环

    print('所有文件/帧处理完成。')
    print('总耗时: (%.3fs)' % (time.time() - t0))
    print('平均推理时间: (%.4fs/frame)   平均NMS时间: (%.4fs/frame)' % (inf_time.avg, nms_time.avg))

    # 在所有图像/帧处理完毕后，释放视频写入器和关闭窗口
    if vid_writer is not None:
        vid_writer.release()
        print(f"输出视频已保存到: {str(Path(opt.save_dir) / (Path(opt.source).stem + '.mp4'))}")

    if opt.show:
        print("按任意键或关闭窗口以结束程序...")
        cv2.waitKey(0)  # 最终的阻塞等待，直到用户手动关闭窗口或按键
        cv2.destroyAllWindows()  # 确保关闭所有OpenCV窗口


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='/home/huayi/hhy/YOLOP/runs/RobotViewDataset/_2025-06-19-17-50(backbone+lane_head)/final_state.pth',
                        # 请替换为你的模型权重路径
                        help='模型权重文件路径，例如: runs/RobotViewDataset/_2025-05-29-17-00(warmup)/final_state.pth')
    parser.add_argument('--source', type=str,
                        # default='/home/huayi/hhy/Datasets/Lane_robot/shanxing/20250121-165847_山行左侧双向植保_auto.mp4',
                        default='/home/huayi/hhy/YOLOP/inference/robot_1',
                        # /home/huayi/hhy/Datasets/Lane_robot/aiwei_test_video/2025-01-17-10-42-49_front.mp4
                        # /home/huayi/hhy/Datasets/Lane_robot/aiwei_test_video/20250415-181616_全流程产量巡检_5m_auto.mp4
                        help='输入源：可以是图像文件路径 (例如: inference/images/0304.png) 或包含图像的文件夹路径 (例如: inference/images/)')
    parser.add_argument('--img-size', type=int, default=320, help='推理时模型输入的图像尺寸 (正方形像素)')
    parser.add_argument('--save', type=bool, default=False, help='是否保存处理后的图像到 --save-dir 指定的目录')
    parser.add_argument('--show', type=bool, default=False, help='是否显示处理后的图像窗口')

    parser.add_argument('--griding-num', type=int, default=100, help='车道线模型输出的栅格数量')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='目标检测的置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='目标检测的IOU阈值 (用于NMS)')
    parser.add_argument('--device', default='0, 1', help='运行设备，例如: "0" (GPU 0), "0,1,2,3" (多GPU), 或 "cpu"')
    parser.add_argument('--save-dir', type=str, default='/home/huayi/hhy/YOLOP/inference/robot_result_layer_16_lane',
                        help='保存推理结果的目录')
    parser.add_argument('--augment', action='store_true', help='是否使用数据增强进行推理 (通常不用于推理)')
    parser.add_argument('--update', action='store_true', help='是否更新所有模型 (通常不用于推理)')
    opt = parser.parse_args()

    # 运行检测
    with torch.no_grad():  # 在推理时禁用梯度计算，节省内存并加速
        detect(cfg, opt)
