# -*- coding: utf-8 -*-
import torch
import argparse
import onnx
import onnxruntime as ort
import onnxsim

from lib.models.YOLOP_LANE import YOLOP_Lane_net_export, YOLOP_lane_robot_no_ll_seg


def export_yolop_lane_onnx(
    batch_size: int = 1,
    height: int = 256,
    width: int = 320,
    weight_path: str = "",
    onnx_path: str = "",
    simplify: bool = True,
    opset: int = 13,
    device: str = ""
):
    assert batch_size in (1, 2), "当前导出脚本只支持 batch_size = 1 或 2（便于极致优化）"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 构建模型
    model = YOLOP_Lane_net_export(YOLOP_lane_robot_no_ll_seg).to(device)

    # 2. 加载权重
    if weight_path:
        try:
            ckpt = torch.load(weight_path, map_location=device)
            if "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
            else:
                model.load_state_dict(ckpt)
            print(f"[Info] 权重加载完成: {weight_path}")
        except FileNotFoundError:
            print(f"[Warn] 未找到权重文件 {weight_path}，使用随机初始化权重导出。")
    else:
        print("[Warn] 未指定权重路径，将导出随机初始化模型。")

    model.eval()

    # 3. 构造固定 batch 的 dummy input
    dummy_input = torch.randn(batch_size, 3, height, width, device=device)

    # 4. ONNX 输出路径
    if not onnx_path:
        onnx_path = f"./weights/yolop_lane_b{batch_size}_{height}x{width}_op{opset}.onnx"
    print(f"[Info] 导出 ONNX 至: {onnx_path}")

    # 5. 导出（固定 batch，关闭 dynamic_axes）
    input_names = ["images"]
    output_names = ["det_out", "da_seg", "lane_parsing"]

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            verbose=False,
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,  # 常量折叠，利于优化
            dynamic_axes=None          # 关键：关闭动态轴，固定 batch
        )

    print("[Info] ONNX 导出完成，开始检查模型...")

    # 6. ONNX 检查
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    print("[Info] ONNX 检查通过。")

    # 7. 可选：onnx-simplifier 简化
    if simplify:
        print(f"[Info] 使用 onnx-simplifier {onnxsim.__version__} 进行模型简化...")
        model_onnx, check = onnxsim.simplify(
            model_onnx,
            check_n=3,
            input_shapes={"images": (batch_size, 3, height, width)}
        )
        assert check, "onnx-simplifier 简化检查失败"
        onnx.save(model_onnx, onnx_path)
        print("[Info] 模型简化完成。")

    # 8. 用 onnxruntime 快速验证一次
    print("\n[Info] 使用 ONNX Runtime 验证推理...")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    print("--- 模型输入信息 ---")
    for ii in sess.get_inputs():
        print(f"  输入名称: {ii.name}, 形状: {ii.shape}, 类型: {ii.type}")
    print("--- 模型输出信息 ---")
    for oo in sess.get_outputs():
        print(f"  输出名称: {oo.name}, 形状: {oo.shape}, 类型: {oo.type}")

    test_input = torch.randn(batch_size, 3, height, width).cpu().numpy()
    ort_inputs = {sess.get_inputs()[0].name: test_input}
    ort_outs = sess.run(None, ort_inputs)
    print(f"\n[Info] 测试推理成功，第一个输出 det_out 的形状: {ort_outs[0].shape}")

    print("[Info] 导出 + 验证完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOP_Lane ONNX 导出脚本（极致优化版：固定 batch）")
    parser.add_argument("--height", type=int, default=256, help="模型输入高度")
    parser.add_argument("--width", type=int, default=320, help="模型输入宽度")
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        choices=[1, 2],
        help="导出时的 batch 大小，仅允许 1 或 2（固定，便于后端极致优化）"
    )
    parser.add_argument(
        "--weights",
        type=str,
        # default="/home/huanghanyang/Project/YOLOP/weights/yolop-256-320-lane-simv3_no_ll_seg.pth",  # sim
        # default="/home/huanghanyang/Project/YOLOP/weights/yolop-256-320-lane-simv3_no_ll_seg.pth",  # v4-1
        default="/home/huanghanyang/Project/YOLOP/weights/yolop_lane_v6_0_final_state_no_ll_seg.pth",  # v6-0
        help="PyTorch 权重路径"
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="",
        help="ONNX 保存路径（不填则自动按规则生成）"
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="不使用 onnx-simplifier 简化模型"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset 版本（建议 13 或 16）"
    )

    args = parser.parse_args()

    export_yolop_lane_onnx(
        batch_size=args.batch,
        height=args.height,
        width=args.width,
        weight_path=args.weights,
        onnx_path=args.onnx,
        simplify=not args.no_simplify,
        opset=args.opset
    )