import torch

from lib.models.YOLOP_LANE import YOLOP_Lane_net, YOLOP_lane_robot_no_ll_seg
import argparse
import onnx
import onnxruntime as ort
import onnxsim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=256)  # height
    parser.add_argument('--width', type=int, default=320)  # width
    args = parser.parse_args()

    do_simplify = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLOP_Lane_net(YOLOP_lane_robot_no_ll_seg)  # 读取不包含车道线分割的网络
    checkpoint = torch.load('/home/hhy/huayi/YOLOP/weights/save.pth', map_location=device)  # 要转换模型
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    height = args.height
    width = args.width
    print("Load ./weights/End-to-end.pth done!")
    onnx_path = f'./weights/yolop-{height}-{width}-lane-v1-2.onnx'  # onnx模型保存路径和名字
    inputs = torch.randn(1, 3, height, width)

    print(f"Converting to {onnx_path}")
    torch.onnx.export(model, inputs, onnx_path,
                      verbose=False, opset_version=12, input_names=['images'],
                      output_names=['det_out', 'det_big', 'det_middle', 'det_small', 'da_seg', 'lane_robot'])  # 命名onnx模型输出层名字
    print('convert', onnx_path, 'to onnx finish!!!')
    # Checks
    model_onnx = onnx.load(onnx_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    print(onnx.helper.printable_graph(model_onnx.graph))  # print

    if do_simplify:
        print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
        model_onnx, check = onnxsim.simplify(model_onnx, check_n=3)
        assert check, 'assert check failed'
        onnx.save(model_onnx, onnx_path)

    x = inputs.cpu().numpy()
    try:
        sess = ort.InferenceSession(onnx_path)

        for ii in sess.get_inputs():
            print("Input: ", ii)
        for oo in sess.get_outputs():
            print("Output: ", oo)

        print('read onnx using onnxruntime sucess')
    except Exception as e:
        print('read failed')
        raise e

