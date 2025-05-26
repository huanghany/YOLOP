import numpy as np
import json
import pdb
from PIL import Image
from .AutoDriveDataset import AutoDriveDataset
from .convert import convert, id_dict, id_dict_single
from tqdm import tqdm

single_cls = False  # just detect vehicle 只检测车辆


def loader_func(path):
    return Image.open(path)


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


class RobotViewDataset(AutoDriveDataset):
    def __init__(self, cfg, is_train, inputsize, transform=None, griding_num=50, load_name=False,
                 row_anchor=None, use_aux=False, segment_transform=None, num_lanes=4):
        super().__init__(cfg, is_train, inputsize, transform,
                         griding_num=griding_num, row_anchor=row_anchor, num_lanes=num_lanes)
        self.db = self._get_db()
        # self.row_anchor = row_anchor
        # if self.row_anchor is not None:
        #     self.row_anchor.sort()
        self.cfg = cfg

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        mask: path of the segmetation label
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        print('building database...')
        gt_db = []
        height, width = self.shapes
        for mask in tqdm(list(self.mask_list)):
            mask_path = str(mask)
            label_path = mask_path.replace(str(self.mask_root), str(self.label_root)).replace(".png", ".json")
            # image_path = mask_path.replace(str(self.mask_root), str(self.img_root)).replace(".png", ".jpg")
            image_path = mask_path.replace(str(self.mask_root), str(self.img_root)).replace(".png", ".png")
            lane_path = mask_path.replace(str(self.mask_root), str(self.lane_root))
            lane_robot_path = mask_path.replace(str(self.mask_root), str(self.lane_robot_root))  # 车道线数据路径

            # 处理车道线数据 (转移至 AutoDriveDataset.__getitem__ ）
            # label_path = lane_robot_path
            # label = loader_func(label_path)
            #
            # img = loader_func(image_path)
            # lane_pts = self._get_index(label)  # (num_lanes, n, (y, x))
            # # get the coordinates of lanes at row anchors
            # w, h = img.size
            # cls_label = self._grid_pts(lane_pts, self.griding_num, w)  # 车道线标签 图像中每个采样点的列索引
            # (n, num_lanes) n 是采样点的数量，num_lanes 是车道线的数量。每个元素是一个整数，表示该采样点在网格中的列索引

            # 处理det
            with open(label_path, 'r') as f:
                label = json.load(f)
            data = label['frames'][0]['objects']
            data = self.filter_data(data)
            gt = np.zeros((len(data), 5))
            for idx, obj in enumerate(data):
                category = obj['category']
                if category == "traffic light":
                    color = obj['attributes']['trafficLightColor']
                    category = "tl_" + color
                if category in id_dict.keys():
                    x1 = float(obj['box2d']['x1'])
                    y1 = float(obj['box2d']['y1'])
                    x2 = float(obj['box2d']['x2'])
                    y2 = float(obj['box2d']['y2'])
                    cls_id = id_dict[category]
                    if single_cls:
                        cls_id = 0
                    gt[idx][0] = cls_id
                    box = convert((width, height), (x1, x2, y1, y2))
                    gt[idx][1:] = list(box)

            rec = [{
                'image': image_path,
                'label': gt,
                'mask': mask_path,
                'lane': lane_path,
                'lane_robot': lane_robot_path  # 车道线数据
            }]

            gt_db += rec
        print('database build finish')
        return gt_db

    def _get_index(self, label):
        w, h = label.size

        if h != 288:
            scale_f = lambda x: int((x * 1.0 / 288) * h)
            sample_tmp = list(map(scale_f, self.row_anchor))

        all_idx = np.zeros((self.num_lanes, len(sample_tmp), 2))
        for i, r in enumerate(sample_tmp):
            label_r = np.asarray(label)[int(round(r))]
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
            # if there is no lane

            valid = all_idx_cp[i, :, 1] != -1
            # get all valid lane points' index
            valid_idx = all_idx_cp[i, valid, :]
            # get all valid lane points
            if valid_idx[-1, 0] == all_idx_cp[0, -1, 0]:
                # if the last valid lane point's y-coordinate is already the last y-coordinate of all rows
                # this means this lane has reached the bottom boundary of the image
                # so we skip
                continue
            if len(valid_idx) < 6:
                continue
            # if the lane is too short to extend

            valid_idx_half = valid_idx[len(valid_idx) // 2:, :]
            p = np.polyfit(valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1)
            start_line = valid_idx_half[-1, 0]
            pos = find_start_pos(all_idx_cp[i, :, 0], start_line) + 1

            fitted = np.polyval(p, all_idx_cp[i, pos:, 0])
            fitted = np.array([-1 if y < 0 or y > w - 1 else y for y in fitted])

            assert np.all(all_idx_cp[i, pos:, 1] == -1)
            all_idx_cp[i, pos:, 1] = fitted
        if -1 in all_idx[:, :, 0]:
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

    def filter_data(self, data):
        remain = []
        for obj in data:
            if 'box2d' in obj.keys():  # obj.has_key('box2d'):
                if single_cls:
                    if obj['category'] in id_dict_single.keys():
                        remain.append(obj)
                else:
                    remain.append(obj)
        return remain

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """  
        """
        pass
