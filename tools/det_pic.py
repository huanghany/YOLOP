import json
import scipy.special
from collections import OrderedDict
import torch
import cv2
import numpy as np
import scipy.special
import argparse

from lib.models import get_YOLOP_LANE_net

lane_num =2
tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]
class LaneDetector:
    def __init__(self, cfg, model_path):
        self.cfg = cfg
        self.img_w, self.img_h = (1640, 590) if cfg.dataset == 'CULane' else (640, 480)
        self.row_anchor = tusimple_row_anchor
        self.cls_num_per_lane = 18 if cfg.dataset == 'CULane' else 56  # 每条车道线上采样点数量

        # 初始化模型
        self.net = get_YOLOP_LANE_net(cfg)
        self._load_model(model_path)
        self.net.eval()

        # 预处理参数
        self.col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        self.col_sample_w = self.col_sample[1] - self.col_sample[0]
        self.scale_x = self.img_w / 800
        self.scale_y = self.img_h / 288


    def _load_model(self, model_path):
        # state_dict = torch.load(model_path, map_location='cpu')['model']
        state_dict = torch.load(model_path, map_location='cpu')
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.net.load_state_dict(state_dict, strict=False)

    def preprocess(self, img):
        # 使用OpenCV进行预处理，提升效率
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (800, 288))
        img = img.astype(np.float32) / 255.0
        img -= np.array([0.485, 0.456, 0.406])
        img /= np.array([0.229, 0.224, 0.225])
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()

    def postprocess(self, output):
        """后处理增强：返回带车道类型标识的结构化数据"""
        out = output[0].data.cpu().numpy()
        out = out[:, ::-1, :]

        prob = scipy.special.softmax(out[:-1, :, :], axis=0)
        idx = np.arange(self.cfg.griding_num).reshape(-1, 1, 1) + 1
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out, axis=0)
        loc[out_j == self.cfg.griding_num] = 0

        # 初始化结构化存储
        lanes = OrderedDict({
            "current_left": [],
            "current_right": [],
            "adjacent_left": [],
            "adjacent_right": []
        })

        for lane_idx in range(out.shape[2]):
            print(lane_idx)
            lane = []
            valid_points = 0
            for point_idx in range(out.shape[1]):
                if loc[point_idx, lane_idx] > 0:
                    x = int(loc[point_idx, lane_idx] * self.col_sample_w * self.scale_x)
                    y = int(self.row_anchor[self.cls_num_per_lane - 1 - point_idx] * self.scale_y)
                    lane.append((x, y))
                    valid_points += 1


            if valid_points > 2:
                # 根据位置分配车道类型（假设索引0-3对应类型）
                lane_type = {
                    0: "adjacent_left",
                    1: "current_left",
                    2: "current_right",
                    3: "adjacent_right"
                }.get(lane_idx, "unknown")

                # 如果类型不存在则动态生成（处理超过4车道的情况）
                if lane_type not in lanes:
                    lane_type = f"lane_{lane_idx}"
                    lanes[lane_type] = []

                lanes[lane_type] = lane

        return lanes

    def visualize(self, img, lanes):
        """增强可视化：不同车道类型使用不同颜色"""
        color_map = {
            "current_left": (0, 255, 0),  # 绿色-当前左车道
            "current_right": (0, 0, 255),  # 红色-当前右车道
            "adjacent_left": (255, 255, 0),  # 青色-左侧邻道
            "adjacent_right": (0, 255, 255)  # 黄色-右侧邻道
        }

        img_vis = img.copy()
        for lane_type, points in lanes.items():
            color = color_map.get(lane_type, (128, 128, 128))  # 默认灰色
            for (x, y) in points:
                cv2.circle(img_vis, (x, y), 5, color, -1)
        return img_vis

    def detect_single_image(self, img_path, save_path=None, json_path=None):
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            return None

        # 调整图像大小
        img = cv2.resize(img, (self.img_w, self.img_h))

        # 处理图像
        input_tensor = self.preprocess(img)  # 图像预处理 输出(1, 3, 288, 800)
        with torch.no_grad():
            output = self.net(input_tensor)  # 将图像输入网络 输出(1, 101, 56, 4)
        lanes = self.postprocess(output)
        result_img = self.visualize(img, lanes)  # 可视化 输出结果图片

        # 保存结果
        if save_path:
            cv2.imwrite(save_path, result_img)
            print(f"结果已保存至: {save_path}")
        # 保存结构化数据
        if json_path:
            with open(json_path, 'w') as f:
                json.dump({
                    "image_size": (self.img_w, self.img_h),
                    "lanes": [
                        {
                            "type": lane_type,
                            "points": points,
                            "count": len(points)
                        }
                        for lane_type, points in lanes.items()
                        if len(points) > 0
                    ]
                }, f, indent=2)
            print(f"结构化数据已保存至: {json_path}")

        return result_img, lanes


# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lane Detection on Single Image')
    parser.add_argument('--img_path', type=str, default='./Datasets/test/51.jpg',
                        help='Path to input image')
    parser.add_argument('--save_path', type=str, default='./save/test/shanxing_3.jpg', help='Path to save result')
    # parser.add_argument('--model_path', type=str, default='./result/20250226_164326_lr_4e-04_b_8/ep399.pth',
    parser.add_argument('--model_path', type=str, default='./model/model_0226.pth',
                        help='Path to model weights')

    args_cmd = parser.parse_args()
    # args, cfg = merge_config()
    # 初始化检测器
    detector = LaneDetector(None, args_cmd.model_path)

    # 处理图片并保存结果
    img_result, lane_data = detector.detect_single_image(
        args_cmd.img_path, args_cmd.save_path
    )

    # 打印结构化数据
    print("检测到车道线：")
    for lane_type, points in lane_data.items():
        if len(points) > 0:
            print(f"- {lane_type}: 包含{len(points)}个坐标点")
