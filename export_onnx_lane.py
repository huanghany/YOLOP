# -*- coding: utf-8 -*-
import torch

from lib.models.YOLOP_LANE import YOLOP_Lane_net, YOLOP_lane_robot_no_ll_seg
import argparse
import onnx
import onnxruntime as ort
import onnxsim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOP ONNX 导出脚本")
    parser.add_argument('--height', type=int, default=256, help='模型输入高度')
    parser.add_argument('--width', type=int, default=320, help='模型输入宽度')
    args = parser.parse_args()

    # 是否进行ONNX模型简化
    do_simplify = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 注意：这里的模型定义需要与你的项目路径匹配
    model = YOLOP_Lane_net(YOLOP_lane_robot_no_ll_seg)  # 读取不包含车道线分割的网络

    # 请确保您的权重路径是正确的
    try:
        # checkpoint = torch.load('/home/hhy/huayi/YOLOP/weights/save.pth', map_location=device)  # 要转换的模型
        # 为了方便演示，这里假设权重文件存在。如果不存在，会打印警告。
        checkpoint_path = '/home/hhy/huayi/YOLOP/weights/save.pth'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"权重加载完成: {checkpoint_path}")
    except FileNotFoundError:
        print("警告: 未找到权重文件。将使用随机初始化的模型进行导出。")

    model.eval()

    height = args.height
    width = args.width

    # 建议为动态批处理模型起一个新名字以作区分
    onnx_path = f'./weights/yolop-{height}-{width}-lane-v1-2.onnx'

    # 创建一个批处理大小为1的示例输入，导出的模型将是动态的
    inputs = torch.randn(1, 3, height, width)

    # ------------------- 关键改动：定义动态轴 -------------------
    # 定义输入和输出的名称
    input_names = ['images']
    output_names = ['det_out', 'det_big', 'det_middle', 'det_small', 'da_seg', 'lane_robot']

    # 为输入和所有输出的第0维（batch_size）指定一个动态轴
    # 'batch_size' 是一个自定义的名称，你可以换成其他名字，如 'batch'
    dynamic_axes = {'images': {0: 'batch_size'}}
    for name in output_names:
        dynamic_axes[name] = {0: 'batch_size'}
    # -----------------------------------------------------------

    print(f"正在转换为支持动态批处理的ONNX模型: {onnx_path}...")
    torch.onnx.export(
        model,
        inputs,
        onnx_path,
        verbose=False,
        opset_version=12,
        input_names=input_names,
        output_names=output_names,
        # ------------------- 关键改动：在导出时传入 dynamic_axes -------------------
        dynamic_axes=dynamic_axes
    )
    print(f"成功将模型转换为ONNX: {onnx_path}")

    # 检查导出的模型
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    # print(onnx.helper.printable_graph(model_onnx.graph))  # 打印图谱会很长，通常不需要，可以注释掉

    if do_simplify:
        print(f'正在使用 onnx-simplifier {onnxsim.__version__} 进行模型简化...')
        model_onnx, check = onnxsim.simplify(model_onnx, check_n=3)
        assert check, '模型简化检查失败'
        onnx.save(model_onnx, onnx_path)
        print('模型简化完成!!!')

    # 使用 onnxruntime 验证动态特性
    print("\n使用 ONNX Runtime 进行验证...")
    try:
        sess = ort.InferenceSession(onnx_path)

        print("--- 模型输入/输出信息 ---")
        for ii in sess.get_inputs():
            # 这里的形状会显示 'batch_size' 或 'unk' 等动态维度
            print(f"输入名称: {ii.name}, 形状: {ii.shape}, 类型: {ii.type}")
        for oo in sess.get_outputs():
            # 这里的形状也会显示动态维度
            print(f"输出名称: {oo.name}, 形状: {oo.shape}, 类型: {oo.type}")

        print('\n--- 使用不同批处理大小进行测试 ---')
        # 测试 batch_size = 1
        print("正在测试 batch_size = 1...")
        inputs_b1 = torch.randn(1, 3, height, width).cpu().numpy()
        ort_inputs_b1 = {sess.get_inputs()[0].name: inputs_b1}
        ort_outs_b1 = sess.run(None, ort_inputs_b1)
        print(f"batch_size=1 时，第一个输出的形状: {ort_outs_b1[0].shape}")

        # 测试 batch_size = 4
        print("\n正在测试 batch_size = 4...")
        inputs_b4 = torch.randn(4, 3, height, width).cpu().numpy()
        ort_inputs_b4 = {sess.get_inputs()[0].name: inputs_b4}
        ort_outs_b4 = sess.run(None, ort_inputs_b4)
        print(f"batch_size=4 时，第一个输出的形状: {ort_outs_b4[0].shape}")

        print('\n成功使用 onnxruntime 读取并测试ONNX模型。')
    except Exception as e:
        print('读取或测试失败')
        raise e
