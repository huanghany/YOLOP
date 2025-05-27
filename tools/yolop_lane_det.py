import argparse
import os, sys
import shutil
import time
from collections import OrderedDict
from pathlib import Path
import imageio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net, get_YOLOP_LANE_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm

row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def visualize(img, lanes):
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


def postprocess(output):
    """后处理增强：返回带车道类型标识的结构化数据"""
    out = output[0].data.cpu().numpy()
    out = out[:, ::-1, :]

    prob = scipy.special.softmax(out[:-1, :, :], axis=0)
    idx = np.arange(100).reshape(-1, 1, 1) + 1
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out, axis=0)
    loc[out_j == 100] = 0

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
                x = int(loc[point_idx, lane_idx] * (800/99) * (640/800))
                y = int(row_anchor[56 - 1 - point_idx] * (480/288))
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


def detect(cfg,opt):

    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger,opt.device)
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = get_YOLOP_LANE_net(cfg)
    checkpoint = torch.load(opt.weights, map_location= device)
    # model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(checkpoint)
    model = model.to(device)
    if half:
        model.half()  # to FP16

    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1  # batch_size


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    
    for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
        img = transform(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        det_out, da_seg_out, ll_seg_out, lane_robot_result = model(img)
        print(lane_robot_result)
        lanes = postprocess(lane_robot_result)



        t2 = time_synchronized()
        # if i == 0:
        #     print(det_out)
        inf_out, _ = det_out
        inf_time.update(t2-t1,img.size(0))

        # Apply NMS
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
        t4 = time_synchronized()

        nms_time.update(t4-t3,img.size(0))
        det=det_pred[0]

        save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")

        _, _, height, width = img.shape
        h,w,_=img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)

        
        ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        # Lane line post-processing
        #ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
        #ll_seg_mask = connect_lane(ll_seg_mask)

        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

        if len(det):
            det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
            for *xyxy,conf,cls in reversed(det):
                label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
        
        if dataset.mode == 'images':
            cv2.imwrite(save_path,img_det)
            # 保存结果
            if save_path:
                result_img = visualize(img_det, lanes)  # 可视化 输出结果图片
                cv2.imwrite(save_path, result_img)
                print(f"结果已保存至: {save_path}")

        elif dataset.mode == 'video':
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h,w,_=img_det.shape
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(img_det)
        
        else:
            cv2.imshow('image', img_det)
            cv2.waitKey(1)  # 1 millisecond

    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    # parser.add_argument('--weights', nargs='+', type=str, default='weights/train_1_epoch-100.pth', help='model.pth path(s)')
    # parser.add_argument('--weights', nargs='+', type=str, default='runs/BddDataset/checkpoint.pth', help='model.pth path(s)')
    parser.add_argument('--weights', nargs='+', type=str,
                        default='/home/huayi/hhy/YOLOP/runs/RobotViewDataset/_2025-05-26-21-15/epoch-615.pth', help='model.pth path(s)')
                        # default='/home/huayi/hhy/YOLOP/runs/RobotViewDataset/model_0526.pth', help='model.pth path(s)')
    # parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--source', type=str, default='/home/huayi/hhy/YOLOP/inference/huayi_1', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.05, help='object confidence threshold')  # 0.25
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')  # 0.45
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/hy_result_test_1', help='directory to save results')
    # parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg,opt)
