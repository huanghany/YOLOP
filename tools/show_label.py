import os
import json
from PIL import Image, ImageDraw
import numpy as np


def visualize_labels(img_path, lane_path, da_seg_path, json_path, output_path):
    # 读取原始图像并转为RGBA
    img = Image.open(img_path).convert("RGBA")
    width, height = img.size

    # 创建透明覆盖层
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    # ===== 修复1：正确加载灰度可通行区域标签 =====
    if os.path.exists(da_seg_path):
        # 强制转换为单通道灰度图
        area_mask = np.array(Image.open(da_seg_path).convert('L'))  # 关键修改点
        # 精确识别255像素（仅处理可通行区域）
        area_mask = (area_mask == 255)
        if np.any(area_mask):
            area_color = (255, 255, 0, 80)  # 黄色半透明
            layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            # 使用二值化掩码（仅255区域生效）
            layer.paste(area_color, (0, 0, width, height), mask=Image.fromarray((area_mask * 255).astype(np.uint8)))
            overlay = Image.alpha_composite(overlay, layer)

    # 可视化车道线标签（处理逻辑不变）
    if os.path.exists(lane_path):
        lane_mask = np.array(Image.open(lane_path))
        colors = [(255, 0, 0, 100), (0, 255, 0, 100), (0, 0, 255, 100)]
        for i in range(1, 3):
            mask = (lane_mask == i)
            if np.any(mask):
                layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
                layer.paste(colors[i], (0, 0, width, height), Image.fromarray(mask.astype(np.uint8) * 255))
                overlay = Image.alpha_composite(overlay, layer)

    # 合并图像与覆盖层
    visualized = Image.alpha_composite(img, overlay)

    # 行人检测框绘制（逻辑不变）
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        draw = ImageDraw.Draw(visualized)
        try:
            for frame in data["frames"]:
                for obj in frame["objects"]:
                    if obj["category"] == "person":
                        box = obj["box2d"]
                        x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
                        draw.rectangle([x1, y1, x2, y2], outline="yellow", width=4)
                        draw.text((x1 + 5, y1 + 5), "person", fill="yellow")
        except Exception as e:
            print(f"JSON处理错误 {json_path}: {str(e)}")

    visualized.convert("RGB").save(output_path)
    print(f"已保存: {output_path}")


if __name__ == "__main__":
    # 定义路径（根据实际位置调整）
    root_path = "/home/huayi/hhy/YOLOP/Datasets/lane_robot_3/"
    img_dir = root_path + "imgs" + "/train"  # 原始图片目录
    lane_dir = root_path + "gt_instance_image_012" + "/train"  # 车道线标签目录
    da_seg_dir = root_path + "da_seg" + "/train"  # 可通行区域标签目录
    person_dir = root_path + "person_json" + "/train"  # 行人检测标签目录
    output_dir = root_path + "visualize"  # 输出目录

    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有原始图片
    for img_name in os.listdir(img_dir):
        if img_name.lower().endswith(".png"):
            base_name = os.path.splitext(img_name)[0]

            # 构建完整路径
            img_path = os.path.join(img_dir, img_name)
            lane_path = os.path.join(lane_dir, img_name)
            da_seg_path = os.path.join(da_seg_dir, img_name)
            json_path = os.path.join(person_dir, f"{base_name}.json")  # JSON无.png后缀
            output_path = os.path.join(output_dir, img_name)

            visualize_labels(img_path, lane_path, da_seg_path, json_path, output_path)
