# -*- coding: utf-8 -*-
# @Author      : huanghany
# @File        : multi_task_onnx_det.py
# @Create      : 2025/6/11-09:55 (Mod: Current Date)
# @Contact     : huanghanyang345@

import os
import time
import argparse

import cv2
import numpy as np
import onnxruntime as ort

from multi_task_onnx_det import resize_unscale, non_max_suppression_np, postprocess_lanes_from_onnx, visualize_lanes


def infer_yolop(onnx_path: str,
                img_path: str,
                batch_size: int,
                warmup_runs: int,
                save_image: bool,
                griding_num_param: int,
                model_input_shape: tuple):
    """
    :param onnx_path: onnx 模型路径
    :param img_path:  测试图片路径
    :param batch_size: 1 或 2（需与导出的 ONNX 模型兼容）
    :param warmup_runs: 预热轮数
    :param save_image: 是否保存可视化结果
    :param griding_num_param: 车道线 griding_num
    :param model_input_shape: (H, W)，例如 (256, 320)
    """

    assert batch_size in (1, 2), "batch_size 只支持 1 或 2"

    # ----------------- 1. 构建 onnxruntime session -----------------
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX 模型不存在: {onnx_path}")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # 必须开 FULL 优化

    # 可以尝试打开 intra_op_num_threads / inter_op_num_threads
    so.intra_op_num_threads = 0  # 0 = 让 ORT 自己决定（通常就是用满 CPU）
    so.inter_op_num_threads = 0

    # 按优先级使用 GPU，否则 CPU
    # providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)

    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    print(f"模型输入: name={input_name}, shape={input_meta.shape}, type={input_meta.type}")

    output_metas = sess.get_outputs()
    for i, o in enumerate(output_metas):
        print(f"模型输出[{i}]: name={o.name}, shape={o.shape}, type={o.type}")

    # ----------------- 2. 读入图像并预处理（单张） -----------------
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"测试图片不存在: {img_path}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise RuntimeError(f"无法读取图片: {img_path}")

    original_height, original_width = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 这里假设你项目中已有 resize_unscale，返回 pad 后图像及缩放信息
    # canvas: pad 后图像 (H_in, W_in, 3)
    model_input_h, model_input_w = model_input_shape
    canvas, r_scale, pad_dw, pad_dh, new_unpad_w, new_unpad_h = resize_unscale(
        img_rgb, new_shape=(model_input_h, model_input_w)
    )

    # 归一化到 0~1
    img_norm = canvas.astype(np.float32) / 255.0
    # 标准化（你实际训练时的 mean/std 为准）
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_norm - mean) / std

    # HWC -> CHW
    img_chw = img_norm.transpose(2, 0, 1)  # [3, H, W]

    # ----------------- 3. 构造 batch 输入 -----------------
    if batch_size == 1:
        # [1, 3, H, W]
        img_batch = np.expand_dims(img_chw, 0).astype(np.float32)
    else:  # batch_size == 2
        # 简单起见：用同一张图复制两份；如果你有两张图，就各自预处理后 stack 即可
        img_batch = np.stack([img_chw, img_chw], axis=0).astype(np.float32)  # [2, 3, H, W]

    print(f"输入 batch 形状: {img_batch.shape}")

    # ----------------- 4. 预热 -----------------
    print(f"开始预热 {warmup_runs} 次 ...")
    for _ in range(warmup_runs):
        _ = sess.run(None, {input_name: img_batch})
    print("预热完成。")

    # ----------------- 5. 正式推理并测速 -----------------
    runs = 50  # 正式测速次数，可按需调整
    print(f"开始测速，总共 {runs} 次（batch={batch_size}）...")
    t_start = time.time()
    for _ in range(runs):
        outputs = sess.run(None, {input_name: img_batch})
    t_end = time.time()

    avg_time = (t_end - t_start) / runs
    fps = batch_size / avg_time
    print(f"平均单次耗时: {avg_time * 1000:.2f} ms, 等效吞吐: {fps:.2f} images/s")

    # ----------------- 6. 取一次输出做后处理和可视化 -----------------
    # 为了演示，这里再正向一次，取 outputs
    outputs = sess.run(None, {input_name: img_batch})

    # 假定输出顺序为 [det, da_seg, lane]，如果你的模型不同，请按实际修改
    if len(outputs) >= 3:
        last_det_out, last_da_seg_out, last_lane_robot_out = outputs[:3]
    elif len(outputs) == 2:
        last_det_out, last_da_seg_out = outputs
        last_lane_robot_out = None
    else:
        last_det_out = outputs[0]
        last_da_seg_out = None
        last_lane_robot_out = None

    # ----------------- 7. 检测后处理（只展示 batch[0]） -----------------
    img_result = img_bgr.copy()  # 用原图 BGR 做可视化

    if last_det_out is not None:
        # last_det_out: [B, N, 5+nc]
        det_predictions_np = last_det_out.astype(np.float32)
        boxes_list = non_max_suppression_np(det_predictions_np, conf_thres=0.25, iou_thres=0.45)

        # 这里只可视化 batch[0] 的结果
        boxes = boxes_list[0]
        if boxes is not None and boxes.shape[0] > 0:
            # 还原到原图坐标：先去 pad，再除缩放
            boxes[:, 0:4:2] -= pad_dw
            boxes[:, 1:4:2] -= pad_dh
            boxes[:, :4] /= r_scale
            # 限制到图像范围
            boxes[:, 0:4:2] = np.clip(boxes[:, 0:4:2], 0, original_width)
            boxes[:, 1:4:2] = np.clip(boxes[:, 1:4:2], 0, original_height)

            print(f"[batch0] 检测到 {boxes.shape[0]} 个框")
            for i in range(boxes.shape[0]):
                x1, y1, x2, y2, conf, cls_id = boxes[i]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cls_id = int(cls_id)
                cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img_result,
                    f"id:{cls_id} {conf:.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

    # ----------------- 8. 分割后处理（只用 batch[0]） -----------------
    if last_da_seg_out is not None:
        # last_da_seg_out: [B, C, H_in, W_in]
        da_seg = last_da_seg_out[0]  # 取 batch[0]

        # 去掉 padding 部分
        da_seg_valid_region = da_seg[:, pad_dh: pad_dh + new_unpad_h,
        pad_dw: pad_dw + new_unpad_w]  # [C, new_h, new_w]

        # 假设第 0 通道是 road mask
        road_prob = da_seg_valid_region[0]  # [new_h, new_w]

        # resize 回原图大小
        road_prob = cv2.resize(
            road_prob,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR
        )

        # road_prob 一般是 float32，取阈值得到 0/255 的单通道 mask（uint8）
        road_mask = (road_prob > 0.5).astype(np.uint8) * 255  # [H, W], dtype=uint8

        # 如果你想用概率值做颜色图，而不是二值 mask，可以先归一化到 0~255 再转 uint8：
        # road_prob_norm = np.clip(road_prob, 0.0, 1.0)
        # road_mask = (road_prob_norm * 255).astype(np.uint8)

        road_mask_color = cv2.applyColorMap(road_mask, cv2.COLORMAP_JET)
        img_result = cv2.addWeighted(img_result, 0.7, road_mask_color, 0.3, 0)

    # ----------------- 9. 车道线后处理（只用 batch[0]） -----------------
    if last_lane_robot_out is not None:
        # last_lane_robot_out: [B, ...]，这里只取 batch[0]
        lane_pred_0 = last_lane_robot_out[0]
        processed_lanes = postprocess_lanes_from_onnx(
            lane_pred_0,
            griding_num=griding_num_param,
            model_input_w=model_input_w,
            model_input_h=model_input_h
        )
        img_result = visualize_lanes(img_result, processed_lanes)

    # ----------------- 10. 保存 / 显示 -----------------
    if save_image:
        save_dir = "./onnx_vis"
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.basename(img_path)
        save_path = os.path.join(save_dir, f"result_bs{batch_size}_" + base)
        cv2.imwrite(save_path, img_result)
        print(f"结果已保存到: {save_path}")
    else:
        cv2.imshow(f"result_bs{batch_size}", img_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def parse_args():
    parser = argparse.ArgumentParser(description="YOLOP 多任务 ONNX 推理与测速")
    # parser.add_argument("--onnx_path", type=str, default="./weights/yolop_lane_b2_256x320_op13.onnx", help="ONNX 模型路径")
    parser.add_argument("--onnx_path", type=str, default="./weights/yolop-256-320-lane-v4-1-1.onnx", help="ONNX 模型路径")
    parser.add_argument("--img_path", type=str, default="./inference/data_test/左边行间有人_front_560.png", help="测试图片路径")
    parser.add_argument("--model_input_height", type=int, default=256, help="模型输入高度")
    parser.add_argument("--model_input_width", type=int, default=320, help="模型输入宽度")
    parser.add_argument("--warmup_runs", type=int, default=5, help="预热次数")
    parser.add_argument("--save_image", action="store_true", help="是否保存可视化结果")
    parser.add_argument("--griding_num", type=int, default=56, help="车道线 griding_num")
    # 新增：batch_size 参数
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        choices=[1, 2],
        help="测试时的 batch 大小，只支持 1 或 2（须与导出的 ONNX 模型兼容）"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer_yolop(
        onnx_path=args.onnx_path,
        img_path=args.img_path,
        batch_size=args.batch_size,
        warmup_runs=args.warmup_runs,
        save_image=args.save_image,
        griding_num_param=args.griding_num,
        model_input_shape=(args.model_input_height, args.model_input_width)
    )