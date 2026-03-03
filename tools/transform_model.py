import torch
from collections import OrderedDict
import re


def filter_and_remap_state_dict(input_pth, output_pth, start_idx_to_delete, end_idx_to_delete):
    """
    直接在 state_dict 层面按索引筛选层，并重新映射后续层的索引。

    Args:
        input_pth (str): 原始权重文件路径 (.pth)。
        output_pth (str): 处理后要保存的新文件路径。
        start_idx_to_delete (int): 要删除的起始层索引（包含）。
        end_idx_to_delete (int): 要删除的结束层索引（包含）。
    """
    print(f"[*] 加载 state_dict 从: {input_pth}")
    try:
        # 加载 state_dict，处理可能存在的嵌套
        checkpoint = torch.load(input_pth, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, OrderedDict):
            print(f"[!] 错误: 加载的对象不是 OrderedDict，而是 {type(state_dict)}。")
            return

    except Exception as e:
        print(f"[!] 加载文件时出错: {e}")
        return

    new_state_dict = OrderedDict()

    # 计算索引偏移量
    num_deleted_layers = end_idx_to_delete - start_idx_to_delete + 1

    print(f"[*] 准备删除索引在 [{start_idx_to_delete}, {end_idx_to_delete}] 范围内的层。")
    print(f"[*] 后续层的索引将被向前移动 {num_deleted_layers} 位。")

    # 用于从键名中提取第一个数字的正则表达式
    # 匹配 'model.数字.' 或 '数字.'
    pattern = re.compile(r'^(?:model\.)?(\d+)\.')

    for key, value in state_dict.items():
        match = pattern.match(key)

        if not match:
            print(f"  - 保留 (非索引层): {key}")
            new_state_dict[key] = value
            continue

        # 提取层索引
        layer_idx = int(match.group(1))

        # --- 决策逻辑 ---
        if layer_idx < start_idx_to_delete:
            # 1. 保留删除范围之前的层，键名不变
            # print(f"  - 保留 (索引 {layer_idx} < {start_idx_to_delete}): {key}")
            new_state_dict[key] = value

        elif layer_idx > end_idx_to_delete:
            # 2. 保留删除范围之后的层，并重新映射索引
            new_layer_idx = layer_idx - num_deleted_layers

            # 创建新的键名，替换旧的索引
            # 例如: 'model.42.conv.weight' -> 'model.33.conv.weight'
            old_prefix = f"{match.group(0)}"  # e.g., "model.42."
            new_prefix = f"{'model.' if 'model.' in old_prefix else ''}{new_layer_idx}."
            new_key = key.replace(old_prefix, new_prefix, 1)

            print(f"  - 重新映射: {key} -> {new_key}")
            new_state_dict[new_key] = value

        else:  # start_idx_to_delete <= layer_idx <= end_idx_to_delete
            # 3. 丢弃在删除范围内的层
            # print(f"  - 删除 (索引 {layer_idx} 在范围内): {key}")
            pass

    # --- 保存结果 ---
    original_keys = len(state_dict)
    new_keys = len(new_state_dict)
    print("\n[*] 筛选完成。")
    print(f"    原始 state_dict 条目数: {original_keys}")
    print(f"    新的 state_dict 条目数: {new_keys}")
    print(f"    共删除了 {original_keys - new_keys} 个条目。")

    try:
        torch.save(new_state_dict, output_pth)
        print(f"[+] 成功将新的 state_dict 保存到: {output_pth}")
    except Exception as e:
        print(f"[!] 保存文件时出错: {e}")


if __name__ == '__main__':
    # =================================================================================
    #                           你需要配置的部分
    # =================================================================================

    # 1. 设置输入和输出文件路径
    INPUT_PTH_PATH = '/home/huanghanyang/Project/YOLOP/weights/yolop-256-320-lane-simv3.pth'  # 你的原始模型文件
    OUTPUT_PTH_PATH = '/home/huanghanyang/Project/YOLOP/weights/yolop-256-320-lane-simv3_no_ll_seg.pth'  # 你想保存的新模型文件

    # 2. 指定要删除的层的索引范围 (从0开始计数)
    #    删除第34层到第42层，对应的索引是 33 到 41
    START_INDEX_TO_DELETE = 34
    END_INDEX_TO_DELETE = 42

    # =================================================================================
    #                           执行脚本
    # =================================================================================
    filter_and_remap_state_dict(
        input_pth=INPUT_PTH_PATH,
        output_pth=OUTPUT_PTH_PATH,
        start_idx_to_delete=START_INDEX_TO_DELETE,
        end_idx_to_delete=END_INDEX_TO_DELETE
    )
