import os
import shutil
import random
import argparse


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in IMG_EXTS


def main(folder_a, folder_b, folder_c, n):
    # 1. 创建输出文件夹 C
    os.makedirs(folder_c, exist_ok=True)

    # 2. 列出 B 中的所有图片文件名（仅文件名，不含路径）
    b_filenames = set()
    for root, dirs, files in os.walk(folder_b):
        for f in files:
            if is_image_file(f):
                b_filenames.add(f)

    # 3. 遍历 A，找出 “A 有但 B 没有的” 图片路径
    diff_paths = []
    for root, dirs, files in os.walk(folder_a):
        for f in files:
            if not is_image_file(f):
                continue
            if f not in b_filenames:
                full_path = os.path.join(root, f)
                diff_paths.append(full_path)

    if not diff_paths:
        print("没有找到 A 中独有的图片（相对于 B）。")
        return

    print(f"共有 {len(diff_paths)} 张图片在 A 中存在但 B 中不存在。")

    # 4. 根据 n 抽取
    if n <= 0 or n >= len(diff_paths):
        selected = diff_paths  # 0 或大于总数则选全部
        print("将复制全部符合条件的图片。")
    else:
        selected = random.sample(diff_paths, n)
        print(f"随机抽取 {n} 张图片。")

    # 5. 复制到 C，保持文件名（不保留原有目录结构）
    for src in selected:
        filename = os.path.basename(src)
        dst = os.path.join(folder_c, filename)
        # 若重名，简单处理：在前面加一个前缀避免覆盖
        if os.path.exists(dst):
            name, ext = os.path.splitext(filename)
            i = 1
            new_filename = f"{name}_{i}{ext}"
            new_dst = os.path.join(folder_c, new_filename)
            while os.path.exists(new_dst):
                i += 1
                new_filename = f"{name}_{i}{ext}"
                new_dst = os.path.join(folder_c, new_filename)
            dst = new_dst
        shutil.copy2(src, dst)

    print(f"已复制 {len(selected)} 张图片到 {folder_c}。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从文件夹A中抽取在B中没有的图片，复制到文件夹C。"
    )
    parser.add_argument("--folder_a", default="/home/huanghanyang/下载/project-140/images", help="文件夹 A 路径")
    parser.add_argument("--folder_b", default="/home/huanghanyang/Datasets/rack_datasets/origin_v1/images", help="文件夹 B 路径")
    parser.add_argument("--folder_c", default="/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v1/images/test", help="输出文件夹 C 路径")
    parser.add_argument(
        "-n",
        type=int,
        default=0,
        help="抽取的图片数量，0 或不填表示全部"
    )

    args = parser.parse_args()
    main(args.folder_a, args.folder_b, args.folder_c, args.n)