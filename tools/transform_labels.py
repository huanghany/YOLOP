import os
import cv2
import numpy as np


def convert_label_values(label_dir, backup=True):
    """
    参数说明：
    label_dir: 需要转换的标签文件目录路径
    backup: 是否创建备份目录（默认True）
    """
    # 定义映射规则
    value_map = {
        0: 0,  # 保持背景为0
        4: 1,  # 4 → 1
        9: 2  # 9 → 2
    }

    # 创建备份目录
    if backup:
        backup_dir = os.path.join(label_dir, "original_backup")
        os.makedirs(backup_dir, exist_ok=True)

    # 遍历目录
    for filename in os.listdir(label_dir):
        if filename.endswith(".png"):
            filepath = os.path.join(label_dir, filename)

            # 读取标签文件
            label = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # 检查是否有非法值
            unique_vals = np.unique(label)
            illegal_vals = [v for v in unique_vals if v not in value_map]
            if illegal_vals:
                print(f"警告: {filename} 包含非法值 {illegal_vals}，已自动置为背景0")

            # 创建新标签
            new_label = np.zeros_like(label)
            for old_val, new_val in value_map.items():
                new_label[label == old_val] = new_val

            # 处理非法值（设置为0）
            for bad_val in illegal_vals:
                new_label[label == bad_val] = 0

            # 备份原始文件
            if backup:
                backup_path = os.path.join(backup_dir, filename)
                cv2.imwrite(backup_path, label)

            # 保存新标签
            cv2.imwrite(filepath, new_label)
            print(f"已转换: {filename}")


def verify_conversion(label_dir):
    """验证转换结果"""
    print("\n验证转换结果：")
    for filename in os.listdir(label_dir):
        if filename.endswith(".png") and not filename.startswith("original_backup"):
            filepath = os.path.join(label_dir, filename)
            label = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            unique_vals = np.unique(label)
            if set(unique_vals) - {0, 1, 2}:
                print(f"错误: {filename} 包含非法值 {unique_vals}")
            else:
                print(f"验证通过: {filename} -> 唯一值 {unique_vals}")


if __name__ == "__main__":
    # 配置参数
    LABEL_DIR = "/home/huayi/hhy/YOLOP/Datasets/lane_robot_2/gt_instance_image/val"  # 替换为你的标签目录路径

    # 执行转换
    convert_label_values(LABEL_DIR)

    # 验证结果
    verify_conversion(LABEL_DIR)
