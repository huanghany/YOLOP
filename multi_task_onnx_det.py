# -*- coding: utf-8 -*-
# @Author      : huanghany
# @File        : multi_task_onnx_det.py
# @Create      : 2025/6/11-09:55 (Mod: Current Date)
# @Contact     : huanghanyang345@163.com
# @Copyright   : Copyright (c) 2024, Huayi Robot Inc. All Rights Reserved.
# @Description : YOLOP ONNX 推理，使用 NumPy 实现 NMS, 支持预热和可选图片保存
import os
import time
import cv2
import argparse
import onnxruntime as ort
import numpy as np
import scipy.special
from collections import OrderedDict

# --- NumPy NMS 辅助函数 (与之前相同) ---
def xywh2xyxy_np(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def box_iou_np(box1, box2, epsilon=1e-7):
    (N, M) = (box1.shape[0], box2.shape[0])
    iou_matrix = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        for j in range(M):
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[i, 0], box1[i, 1], box1[i, 2], box1[i, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[j, 0], box2[j, 1], box2[j, 2], box2[j, 3]
            inter_x1 = np.maximum(b1_x1, b2_x1)
            inter_y1 = np.maximum(b1_y1, b2_y1)
            inter_x2 = np.minimum(b1_x2, b2_x2)
            inter_y2 = np.minimum(b1_y2, b2_y2)
            inter_w = np.maximum(0.0, inter_x2 - inter_x1)
            inter_h = np.maximum(0.0, inter_y2 - inter_y1)
            intersection = inter_w * inter_h
            area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
            area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
            union = area1 + area2 - intersection
            iou_matrix[i, j] = intersection / (union + epsilon)
    return iou_matrix

def nms_core_numpy(boxes, scores, iou_threshold):
    if not boxes.size:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int64)

def non_max_suppression_np(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=None, labels=()):
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres
    max_wh = 7680
    max_det = 300
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    merge = False
    if multi_label is None:
        multi_label = nc > 1
    t_start_nms = time.time()
    output = [np.zeros((0, 6), dtype=np.float32) for _ in range(prediction.shape[0])]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            l = np.array(labels[xi]) if not isinstance(labels[xi], np.ndarray) else labels[xi]
            v = np.zeros((len(l), nc + 5), dtype=np.float32)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].astype(np.int32) + 5] = 1.0
            x = np.concatenate((x, v), axis=0)
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy_np(x[:, :4])
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero()
            x = np.concatenate((box[i], x[i, j + 5, np.newaxis], j[:, np.newaxis].astype(np.float32)), axis=1)
        else:
            conf_flat = np.max(x[:, 5:], axis=1)
            conf = conf_flat[:, np.newaxis]
            j_flat = np.argmax(x[:, 5:], axis=1)
            j = j_flat[:, np.newaxis]
            x = np.concatenate((box, conf, j.astype(np.float32)), axis=1)
            x = x[conf.flatten() > conf_thres]
        if classes is not None:
            target_classes = np.array(classes, dtype=np.float32)
            x = x[(x[:, 5:6] == target_classes).any(axis=1)]
        n = x.shape[0]
        if not n: continue
        if n > max_nms:
            x = x[x[:, 4].argsort()[::-1][:max_nms]]
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes_for_nms, scores_for_nms = x[:, :4] + c, x[:, 4]
        i = nms_core_numpy(boxes_for_nms, scores_for_nms, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge:
            if n > 0 and 1 < n < 3E3:
                try:
                    iou_matrix = box_iou_np(boxes_for_nms[i], boxes_for_nms)
                    iou_mask = iou_matrix > iou_thres
                    weights = iou_mask * scores_for_nms[np.newaxis, :]
                    x_coords_to_merge = x[:, :4]
                    merged_coords_numerator = np.dot(weights, x_coords_to_merge)
                    sum_weights = weights.sum(axis=1, keepdims=True)
                    x[i, :4] = merged_coords_numerator / np.maximum(sum_weights, 1e-7)
                    if redundant:
                        i = i[weights.sum(axis=1) > 1]
                except Exception as e:
                    print(f"警告: Merge NMS 过程中发生错误: {e}")
                    pass
        output[xi] = x[i]
        if (time.time() - t_start_nms) > time_limit:
            print(f'警告: NMS 处理图像 {xi} 超时 ({time_limit}s)')
            break
    return output
# --- NumPy NMS 结束 ---

row_anchor = [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
              116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
              168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
              220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
              272, 276, 280, 284]

def visualize_lanes(img, lanes):
    color_map = {"current_left": (0, 255, 0), "current_right": (0, 0, 255)}
    img_vis = img.copy()
    for lane_type, points in lanes.items():
        color = color_map.get(lane_type, (128, 128, 128))
        for (x, y) in points:
            if x is None or y is None: continue
            cv2.circle(img_vis, (int(x), int(y)), 5, color, -1)
    return img_vis

def postprocess_lanes_from_onnx(lane_robot_output_onnx, griding_num,
                                actual_img_w, actual_img_h,
                                model_input_w=320, model_input_h=256):
    out = lane_robot_output_onnx[0]
    prob = scipy.special.softmax(out[:-1, :, :], axis=0)
    idx = np.arange(griding_num).reshape(-1, 1, 1) + 1
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out, axis=0)
    loc[out_j == griding_num] = 0
    lanes = OrderedDict({"current_left": [], "current_right": []})
    col_sample = np.linspace(0, model_input_w - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]
    for lane_idx in range(loc.shape[1]):
        lane_points = []
        for point_idx in range(loc.shape[0]):
            if loc[point_idx, lane_idx] > 0:
                x_model_space = loc[point_idx, lane_idx] * col_sample_w
                x_orig_space = int(x_model_space * actual_img_w / model_input_w)
                y_orig_space = int(row_anchor[point_idx] * actual_img_h / 288.0)
                lane_points.append((x_orig_space, y_orig_space))
            else:
                lane_points.append((None, None))
        if lane_idx == 0: lane_type = "current_left"
        elif lane_idx == 1: lane_type = "current_right"
        else: lane_type = f"other_lane_{lane_idx}"
        if lane_type in lanes:
            lanes[lane_type] = [p for p in lane_points if p[0] is not None]
        elif "other_lane" in lane_type:
            lanes[lane_type] = [p for p in lane_points if p[0] is not None]
    return lanes

def resize_unscale(img, new_shape=(640, 640), color=114):
    shape = img.shape[:2]
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad_h, new_unpad_w = int(round(shape[0] * r)), int(round(shape[1] * r))
    pad_h, pad_w = new_shape[0] - new_unpad_h, new_shape[1] - new_unpad_w
    dh, dw = pad_h // 2, pad_w // 2
    if shape[0] != new_unpad_h or shape[1] != new_unpad_w:
        img_resized = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img
    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img_resized
    return canvas, r, dw, dh, new_unpad_w, new_unpad_h

def infer_yolop(onnx_model_path="yolop-256-320-lane.onnx",
                img_path="./inference/images/some_image.jpg",
                output_dir="./inference_onnx_output",
                num_runs=1,
                warmup_runs=0, # 新增预热次数参数
                save_image=True, # 新增是否保存图片参数
                griding_num_param=100,
                model_input_shape=(256, 320)
                ):
    ort.set_default_logger_severity(3)
    if not os.path.exists(onnx_model_path):
        print(f"错误: 模型文件未找到: {onnx_model_path}")
        return

    execution_provider = "CUDAExecutionProvider" if ort.get_device() == 'GPU' else "CPUExecutionProvider"
    try:
        ort_session = ort.InferenceSession(onnx_model_path, providers=[execution_provider])
        print(f"成功加载模型: {onnx_model_path}，使用 {execution_provider}。")
    except Exception as e:
        print(f"错误: 加载ONNX模型失败: {e}")
        return

    print("模型输入:")
    for inp in ort_session.get_inputs(): print(f"  {inp.name}: {inp.shape} ({inp.type})")
    print("模型输出:")
    output_node_names_from_model = [o.name for o in ort_session.get_outputs()]
    for name, shape, type_ in zip(output_node_names_from_model,
                                 [o.shape for o in ort_session.get_outputs()],
                                 [o.type for o in ort_session.get_outputs()]):
        print(f"  {name}: {shape} ({type_})")


    if save_image: # 仅当需要保存图片时创建目录和文件名
        os.makedirs(output_dir, exist_ok=True)
        base_img_name = os.path.splitext(os.path.basename(img_path))[0]
        save_merge_path = os.path.join(output_dir, f"{base_img_name}_output_onnx.jpg")
    else:
        save_merge_path = None # 如果不保存，路径设为None

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"错误: 无法加载图片: {img_path}")
        return
    original_height, original_width, _ = img_bgr.shape
    img_rgb = img_bgr[:, :, ::-1].copy()

    canvas, r_scale, pad_dw, pad_dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb, new_shape=model_input_shape)
    img_norm = canvas.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_norm - mean) / std
    img_chw = img_norm.transpose(2, 0, 1)
    img_batch = np.expand_dims(img_chw, 0).astype(np.float32)

    input_name = ort_session.get_inputs()[0].name
    if len(output_node_names_from_model) >= 3:
        output_names_to_run = [output_node_names_from_model[0], output_node_names_from_model[4], output_node_names_from_model[5]]
        print(f"将使用以下输出节点进行推理: {output_names_to_run}")
    else:
        print(f"错误：模型输出节点数量 ({len(output_node_names_from_model)}) 少于预期的3个。请检查模型。")
        return

    # --- 模型预热 ---
    if warmup_runs > 0:
        print(f"开始进行 {warmup_runs} 次模型预热...")
        for _ in range(warmup_runs):
            _ = ort_session.run(output_names_to_run, {input_name: img_batch})
        print("模型预热完成。")

    # --- 正式推理 ---
    total_inference_time = 0.0
    last_det_out, last_da_seg_out, last_lane_robot_out = None, None, None
    print(f"开始进行 {num_runs} 次正式推理...")
    for i_run in range(num_runs):
        time_start = time.time()
        try:
            outputs = ort_session.run(output_names_to_run, {input_name: img_batch})
            # 仅在最后一次运行时保留输出，以减少内存占用，除非num_runs=1
            if i_run == num_runs - 1 or num_runs == 1:
                 last_det_out = outputs[0]
                 last_da_seg_out = outputs[1]
                 last_lane_robot_out = outputs[2]
        except Exception as e:
            print(f"推理过程中发生错误: {e}")
            return
        total_inference_time += (time.time() - time_start)

    if num_runs > 0:
        average_inference_time = total_inference_time / num_runs
        print(f"平均推理时间: {average_inference_time:.4f} 秒 (基于 {num_runs} 次正式推理)")
    else:
        print("未执行正式推理 (num_runs=0)")
        if not save_image: # 如果不保存图片且不推理，则直接退出
             print("所有任务完成（无推理，无保存）。")
             return


    if last_det_out is None and num_runs > 0 : # 确保有输出用于后处理
        print("警告: 推理未产生有效输出用于后处理（可能是因为num_runs > 1且只保留了最后一次）。")
        if num_runs > 1: # 如果多次运行但没有保留最后输出，则重新运行一次以获取输出
            print("为进行后处理，将额外运行一次推理...")
            outputs = ort_session.run(output_names_to_run, {input_name: img_batch})
            last_det_out = outputs[0]
            last_da_seg_out = outputs[1]
            last_lane_robot_out = outputs[2]
        else: # num_runs=0 或 num_runs=1 但 last_det_out 仍为 None
            print("无法获取推理输出进行后处理。")
            if not save_image: # 如果不保存图片，则退出
                print("所有任务完成（无后处理，无保存）。")
                return


    if not save_image and (last_det_out is None or last_da_seg_out is None or last_lane_robot_out is None):
        print("无需保存图片且无有效推理输出，任务结束。")
        return


    # --- 后处理 (仅当需要保存图片或num_runs=1时，last_*_out才会有值) ---
    if last_det_out is not None: # 确保有数据进行后处理
        img_result = img_bgr.copy() # 开始绘制前复制原始图像

        # 1. 目标检测
        det_predictions_np = last_det_out.astype(np.float32)
        boxes_list = non_max_suppression_np(det_predictions_np, conf_thres=0.25, iou_thres=0.45)
        boxes = boxes_list[0]
        if boxes is not None and boxes.shape[0] > 0:
            boxes[:, 0:4:2] -= pad_dw
            boxes[:, 1:4:2] -= pad_dh
            boxes[:, :4] /= r_scale
            boxes[:, 0:4:2] = np.clip(boxes[:, 0:4:2], 0, original_width)
            boxes[:, 1:4:2] = np.clip(boxes[:, 1:4:2], 0, original_height)
            print(f"检测到 {boxes.shape[0]} 个边界框。")
            for i in range(boxes.shape[0]):
                x1, y1, x2, y2, conf, cls_id = boxes[i]
                cv2.rectangle(img_result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img_result, f"cls:{int(cls_id)} {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 2. 可行驶区域分割
        if last_da_seg_out is not None:
            da_seg_valid_region = last_da_seg_out[0, :, pad_dh : pad_dh + new_unpad_h, pad_dw : pad_dw + new_unpad_w]
            da_seg_mask = np.argmax(da_seg_valid_region, axis=0).astype(np.uint8)
            da_seg_mask_resized = cv2.resize(da_seg_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            drivable_color_bgr = [0, 200, 0]
            overlay_da = np.zeros_like(img_result, dtype=np.uint8)
            overlay_da[da_seg_mask_resized == 1] = drivable_color_bgr
            alpha_da = 0.3
            cv2.addWeighted(overlay_da, alpha_da, img_result, 1 - alpha_da, 0, img_result)

        # 3. 车道线
        if last_lane_robot_out is not None:
            processed_lanes = postprocess_lanes_from_onnx(
                last_lane_robot_out,
                griding_num=griding_num_param,
                actual_img_w=original_width, actual_img_h=original_height,
                model_input_w=model_input_shape[1], model_input_h=model_input_shape[0]
            )
            img_result = visualize_lanes(img_result, processed_lanes)

        if save_image and save_merge_path is not None:
            cv2.imwrite(save_merge_path, img_result)
            print(f"合并结果已保存到: {save_merge_path}")
        elif save_image and save_merge_path is None: # 逻辑上不应发生，但作为保险
            print("警告：请求保存图片但保存路径未设置。")

    elif save_image: # 如果请求保存但没有推理结果
        print("警告: 请求保存图片，但没有有效的推理输出进行处理和保存。")


    print("所有任务完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOP ONNX模型推理脚本 (NumPy NMS)，支持预热和可选图片保存。")
    parser.add_argument('--model', type=str, default="./weights/yolop-256-320-lane-v1-3.onnx", help='ONNX模型权重路径')
    # v1.0    ./weights/yolop-256-320-lane-v1-0.onnx
    # v1.1    ./weights/yolop-256-320-lane-v1-1.onnx
    # v1.2    ./weights/yolop-256-320-lane-v1-2.onnx
    parser.add_argument('--img', type=str, default="./inference/robot_1/0114.png", help='推理图片路径')
    parser.add_argument('--output_dir', type=str, default="./inference/robot_1_result_v1.0", help='保存结果的目录')
    parser.add_argument('--num_runs', type=int, default=100, help='计算平均推理时间的正式运行次数')
    parser.add_argument('--warmup_runs', type=int, default=10, help='模型预热运行次数 (0表示不预热)')
    parser.add_argument('--save_image', type=float, default=True, help='是否保存输出图片 (默认不保存)')

    parser.add_argument('--griding_num', type=int, default=100, help='车道线后处理的栅格数量')
    parser.add_argument('--model_input_height', type=int, default=256, help='模型输入高度')
    parser.add_argument('--model_input_width', type=int, default=320, help='模型输入宽度')

    args = parser.parse_args()

    os.makedirs("./weights", exist_ok=True)
    img_dir_check = os.path.dirname(args.img)
    if img_dir_check and not os.path.exists(img_dir_check):
        print(f"警告: 图片目录 '{img_dir_check}' 不存在。")
    if not os.path.exists(args.img):
         print(f"错误: 输入图片 '{args.img}' 不存在。请检查路径。")
         exit()

    infer_yolop(
        onnx_model_path=args.model,
        img_path=args.img,
        output_dir=args.output_dir,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs, # 传递预热参数
        save_image=args.save_image,   # 传递保存图片参数
        griding_num_param=args.griding_num,
        model_input_shape=(args.model_input_height, args.model_input_width)
    )
