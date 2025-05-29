import cv2
import numpy as np
import pdb
# np.set_printoptions(threshold=np.inf)
import random
import torch
import torchvision.transforms as transforms
# from visualization import plot_img_and_mask,plot_one_box,show_seg_result
from pathlib import Path
from torch.utils.data import Dataset
from ..utils import letterbox, augment_hsv, random_perspective, xyxy2xywh


def find_start_pos(row_sample,start_line):
    # row_sample = row_sample.sort()
    # for i,r in enumerate(row_sample):
    #     if r >= start_line:
    #         return i
    l,r = 0,len(row_sample)-1
    while True:
        mid = int((l+r)/2)
        if r - l == 1:
            return r
        if row_sample[mid] < start_line:
            l = mid
        if row_sample[mid] > start_line:
            r = mid
        if row_sample[mid] == start_line:
            return mid

class AutoDriveDataset(Dataset):
    """
    A general Dataset for some common function
    """
    def __init__(self, cfg, is_train, inputsize=640, transform=None, griding_num=50, num_lanes=2, row_anchor=None):
        """
        initial all the characteristic

        Inputs:
        -cfg: configurations
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize
        
        Returns:
        None
        """
        self.griding_num = griding_num
        self.num_lanes = num_lanes
        self.row_anchor = row_anchor
        self.row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]
        if self.row_anchor is not None:
            self.row_anchor.sort()

        # self.is_train = is_train
        self.is_train = False
        self.cfg = cfg
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()
        img_root = Path(cfg.DATASET.DATAROOT)
        label_root = Path(cfg.DATASET.LABELROOT)
        mask_root = Path(cfg.DATASET.MASKROOT)
        lane_root = Path(cfg.DATASET.LANEROOT)

        # 使用 getattr 获取 LANEROBOTROOT，如果不存在则为 None
        lane_robot_root = getattr(cfg.DATASET, 'LANEROBOTROOT', None)  # cfg.DATASET.LANEROBOTROOT

        if is_train:
            indicator = cfg.DATASET.TRAIN_SET
        else:
            indicator = cfg.DATASET.TEST_SET
        self.img_root = img_root / indicator
        self.label_root = label_root / indicator
        self.mask_root = mask_root / indicator
        self.lane_root = lane_root / indicator
        if lane_robot_root:
            lane_robot_root = Path(lane_robot_root)
            self.lane_robot_root = lane_robot_root / indicator
        else:
            self.lane_robot_root = None
        # self.label_list = self.label_root.iterdir()
        self.mask_list = self.mask_root.iterdir()

        self.db = []

        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        # self.target_type = cfg.MODEL.TARGET_TYPE
        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)

    def _get_index(self, label):
        # label 现在是cv2读取的numpy数组
        h, w = label.shape[:2]  # 从label.size 更改为 label.shape

        if h != 288:  # 假设原始row_anchor基于288高度
            scale_f = lambda x: int((x * 1.0 / 288) * h)
            sample_tmp = list(map(scale_f, self.row_anchor))
        else:  # 如果图像高度已经是288，则直接使用原始row_anchor
            sample_tmp = self.row_anchor

        all_idx = np.zeros((self.num_lanes, len(sample_tmp), 2))
        for i, r in enumerate(sample_tmp):
            label_r = label[int(round(r))]  # 从 np.asarray(label)[int(round(r))] 更改
            for lane_idx in range(1, self.num_lanes + 1):
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1
                    continue
                pos = np.mean(pos)
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = pos

        # data augmentation: extend the lane to the boundary of image
        all_idx_cp = all_idx.copy()
        for i in range(self.num_lanes):
            if np.all(all_idx_cp[i, :, 1] == -1):
                continue

            valid = all_idx_cp[i, :, 1] != -1
            valid_idx = all_idx_cp[i, valid, :]

            if len(valid_idx) < 2:  # 如果有效点少于2个，无法进行多项式拟合，跳过
                continue

            # polyfit需要至少2个点，确保valid_idx_half也有足够多的点
            valid_idx_half = valid_idx[len(valid_idx) // 2:, :]
            if len(valid_idx_half) < 2:
                if len(valid_idx_half) == 1:  # 如果只有一个点，近似为水平线
                    p = [0, valid_idx_half[0, 1]]  # 斜率0，截距为y值
                else:  # 没有点，跳过
                    continue
            else:
                p = np.polyfit(valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1)

            start_line = valid_idx_half[-1, 0]
            pos = find_start_pos(all_idx_cp[i, :, 0], start_line) + 1

            fitted = np.polyval(p, all_idx_cp[i, pos:, 0])
            fitted = np.array([-1 if y < 0 or y > w - 1 else y for y in fitted])

            # 将拟合结果填充到-1的位置
            for k in range(pos, all_idx_cp.shape[1]):
                if k - pos < len(fitted) and all_idx_cp[i, k, 1] == -1:  # 确保索引不越界，只填充-1的位置
                    all_idx_cp[i, k, 1] = fitted[k - pos]

        if -1 in all_idx_cp[:, :, 0]:
            pdb.set_trace()
        return all_idx_cp  # 输出

    def _grid_pts(self, pts, num_cols, w):
        # pts : numlane,n,2
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)  # 生成一个等间距num_cols的列采样点

        assert n2 == 2
        to_pts = np.zeros((n, num_lane))  # 每个采样点在网格中的列索引
        for i in range(num_lane):  # 每条车道线，计算每个采样点在网格中的列索引
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)  # 返回int

    def _get_db(self):
        """
        finished on children Dataset(for dataset which is not in Bdd100k format, rewrite children Dataset)
        """
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir):
        """
        finished on children dataset
        """
        raise NotImplementedError
    
    def __len__(self,):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def __getitem__(self, idx):
        """
        Get input and groud-truth from database & add data augmentation on input

        Inputs:
        -idx: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """
        data = self.db[idx]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # seg_label = cv2.imread(data["mask"], 0)
        if self.cfg.num_seg_class == 3:
            seg_label = cv2.imread(data["mask"])
        else:
            seg_label = cv2.imread(data["mask"], 0)
        lane_label = cv2.imread(data["lane"], 0)

        lane_robot_label = None
        if "lane_robot" in data:
            lane_robot_label = cv2.imread(data["lane_robot"], -1)

        # print("lane_robot_label.shape:", lane_robot_label.shape) # 可以删除
        # print("seg_label.shape：", seg_label.shape) # 可以删除
        # print("lane_label.shape:", lane_label.shape) # 可以删除
        # print("seg_label.shape:", seg_label.shape) # 可以删除

        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # orig hw 480 640
        r = resized_shape / max(h0, w0)  # resize image to img_size 0.5
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            seg_label = cv2.resize(seg_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
            lane_label = cv2.resize(lane_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
            if lane_robot_label is not None:
                # lane_robot_label = cv2.resize(lane_robot_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
                lane_robot_label = cv2.resize(lane_robot_label, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_NEAREST)  # 用最近临

        # 获取letterbox前的图像尺寸，用于后续计算ratio
        h, w = img.shape[:2]  # 240 320

        if lane_robot_label is not None:
            (img, seg_label, lane_label, lane_robot_label), ratio, pad = letterbox(
                (img, seg_label, lane_label, lane_robot_label), resized_shape, auto=True, scaleup=self.is_train)
        else:
            (img, seg_label, lane_label), ratio, pad = letterbox(
                (img, seg_label, lane_label), resized_shape, auto=True, scaleup=self.is_train)

        # 获取letterbox后的图像尺寸，用于车道线网格计算
        h_letterbox, w_letterbox = img.shape[:2]  # 256 320

        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling (480 640) 0.5 0.5 0 8
        # ratio = (w / w0, h / h0)
        # print(resized_shape)

        # 处理det_label
        det_label = data["label"]
        labels = []
        if det_label.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = det_label.copy()
            labels[:, 1] = ratio[0] * w * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w * (det_label[:, 1] + det_label[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (det_label[:, 2] + det_label[:, 4] / 2) + pad[1]

        if self.is_train:
            if lane_robot_label is not None:
                combination = (img, seg_label, lane_label, lane_robot_label)
                (img, seg_label, lane_label, lane_robot_label), labels = random_perspective(
                    combination=combination,
                    targets=labels,
                    degrees=self.cfg.DATASET.ROT_FACTOR,
                    translate=self.cfg.DATASET.TRANSLATE,
                    scale=self.cfg.DATASET.SCALE_FACTOR,
                    shear=self.cfg.DATASET.SHEAR
                )
            else:
                combination = (img, seg_label, lane_label)
                (img, seg_label, lane_label), labels = random_perspective(  # 随机透视
                    combination=combination,
                    targets=labels,
                    degrees=self.cfg.DATASET.ROT_FACTOR,
                    translate=self.cfg.DATASET.TRANSLATE,
                    scale=self.cfg.DATASET.SCALE_FACTOR,
                    shear=self.cfg.DATASET.SHEAR
                )
            # print(labels.shape) # 可以删除
            augment_hsv(img, hgain=self.cfg.DATASET.HSV_H, sgain=self.cfg.DATASET.HSV_S,
                        vgain=self.cfg.DATASET.HSV_V)  # hsv增强
            # img, seg_label, labels = cutout(combination=combination, labels=labels)

            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

            # if self.is_train:
            # random left-right flip
            lr_flip = False  # 左右翻转  不能要
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                seg_label = np.fliplr(seg_label)
                lane_label = np.fliplr(lane_label)
                if lane_robot_label is not None:  # 对robot lane标签也进行翻转
                    lane_robot_label = np.fliplr(lane_robot_label)  # 翻转图像
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False  # 未启用上下翻转
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                seg_label = np.flipud(seg_label)  #
                lane_label = np.flipud(lane_label)  #
                if lane_robot_label is not None:  # 对robot lane标签也进行翻转
                    lane_robot_label = np.flipud(lane_robot_label)
                if len(labels):
                    labels[:, 2] = 1 - labels[:, 2]

        else:
            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

        labels_out = torch.zeros((len(labels), 6))  # 标签转换
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)
        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        # seg_label = np.ascontiguousarray(seg_label)
        # if idx == 0:
        #     print(seg_label[:,:,0])

        # seg标签处理
        if self.cfg.num_seg_class == 3:
            _, seg0 = cv2.threshold(seg_label[:, :, 0], 128, 255, cv2.THRESH_BINARY)
            _, seg1 = cv2.threshold(seg_label[:, :, 1], 1, 255, cv2.THRESH_BINARY)
            _, seg2 = cv2.threshold(seg_label[:, :, 2], 1, 255, cv2.THRESH_BINARY)
        else:
            _, seg1 = cv2.threshold(seg_label, 1, 255, cv2.THRESH_BINARY)
            _, seg2 = cv2.threshold(seg_label, 1, 255, cv2.THRESH_BINARY_INV)
        _, lane1 = cv2.threshold(lane_label, 1, 255, cv2.THRESH_BINARY)
        _, lane2 = cv2.threshold(lane_label, 1, 255, cv2.THRESH_BINARY_INV)
        #        _,seg2 = cv2.threshold(seg_label[:,:,2],1,255,cv2.THRESH_BINARY)
        # # seg1[cutout_mask] = 0
        # # seg2[cutout_mask] = 0

        # seg_label /= 255
        # seg0 = self.Tensor(seg0)
        if self.cfg.num_seg_class == 3:
            seg0 = self.Tensor(seg0)
        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        # seg1 = self.Tensor(seg1)
        # seg2 = self.Tensor(seg2)
        lane1 = self.Tensor(lane1)  # 转换为tensor
        lane2 = self.Tensor(lane2)

        # seg_label = torch.stack((seg2[0], seg1[0]),0)
        if self.cfg.num_seg_class == 3:
            seg_label = torch.stack((seg0[0], seg1[0], seg2[0]), 0)
        else:
            seg_label = torch.stack((seg2[0], seg1[0]), 0)

        lane_label = torch.stack((lane2[0], lane1[0]), 0)

        # lane robot标签处理 (原RobotViewDataset第70行下面的逻辑移到这里)
        cls_label = None  # 初始化 cls_label
        if lane_robot_label is not None and \
                self.griding_num is not None and \
                self.row_anchor is not None and \
                self.num_lanes is not None:
            # lane_robot_label 已经是 cv2 图像 (numpy 数组)
            scale_label = lane_robot_label.copy()
            if lane_robot_label.max()>0:
                scale_label = (scale_label/scale_label.max()*255).astype(np.uint8)
            else:
                scale_label = np.zeros_like(lane_robot_label, dtype=np.uint8)
            # print("max:", lane_robot_label.max())
            colored_mask = cv2.applyColorMap(scale_label, cv2.COLORMAP_JET)
            is_labled_mask = (lane_robot_label > 0).astype(np.uint8)*255
            combined_image = img.copy().astype(np.float32)
            colored_mask_float = colored_mask.astype(np.float32)
            rows, cols =np.where(is_labled_mask>0)
            combined_mask = cv2.addWeighted(img[rows, cols], 0.5, colored_mask[rows, cols], 0.5, 0)
            combined_image[rows, cols] = combined_mask

            combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
            cv2.imwrite(f'./label_pic/lane_label_{idx}.png', combined_image)
            lane_pts = self._get_index(lane_robot_label)  # (num_lanes, n, (y, x))
            # print("lane_points:", lane_pts)
            # 获取车道线在行锚点处的坐标
            h, w = img.shape[:2]
            cls_label = self._grid_pts(lane_pts, self.griding_num, w)  # 车道线标签 图像中每个采样点的列索引 需要处理后的图像宽度
            # (n, num_lanes) n 是采样点的数量，num_lanes 是车道线的数量。每个元素是一个整数，表示该采样点在网格中的列索引

            cls_label = torch.from_numpy(cls_label)  # 转换为Tensor

        if cls_label is not None:
            target = [labels_out, seg_label, lane_label, cls_label]  # 最后返回处理后标签

        else:
            target = [labels_out, seg_label, lane_label]

        img = self.transform(img)

        return img, target, data["image"], shapes

    def select_data(self, db):
        """
        You can use this function to filter useless images in the dataset

        Inputs:
        -db: (list)database

        Returns:
        -db_selected: (list)filtered dataset
        """
        db_selected = ...
        return db_selected

    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes = zip(*batch)
        label_det, label_seg, label_lane, label_cls_lane = [], [], [], []  # 添加 label_cls_lane
        has_cls_lane = False

        for i, l in enumerate(label):
            l_det, l_seg, l_lane = l[0], l[1], l[2]  # 解包前三个标签
            l_det[:, 0] = i  # 为 build_targets() 添加目标图像索引
            label_det.append(l_det)
            label_seg.append(l_seg)
            label_lane.append(l_lane)

            if len(l) > 3 and l[3] is not None:  # 检查是否有cls_lane标签
                label_cls_lane.append(l[3])
                has_cls_lane = True

        # 根据是否存在 cls_lane 标签决定返回的格式
        if has_cls_lane:
            return torch.stack(img, 0), [torch.cat(label_det, 0), torch.stack(label_seg, 0), torch.stack(label_lane, 0),
                                         torch.stack(label_cls_lane, 0)], paths, shapes
        else:
            return torch.stack(img, 0), [torch.cat(label_det, 0), torch.stack(label_seg, 0),
                                         torch.stack(label_lane, 0)], paths, shapes
