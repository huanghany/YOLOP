import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from .general import bbox_iou
from .postprocess import build_targets
from lib.core.evaluate import SegmentationMetric


# 定义新的损失函数类
class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)  # NLLLoss expects log-probabilities

    def forward(self, logits, labels):
        # logits: (N, C, H, W)
        # labels: (N, H, W) or (N, 1, H, W)
        # Ensure labels are long type for NLLLoss
        labels = labels.squeeze(1).long() if labels.dim() == 4 else labels.long()

        scores = F.softmax(logits, dim=1)  # (N, C, H, W)
        factor = torch.pow(1. - scores, self.gamma)  # (N, C, H, W)
        log_score = F.log_softmax(logits, dim=1)  # (N, C, H, W)

        # Apply factor element-wise
        log_score_weighted = factor * log_score

        # Reshape for NLLLoss: (N, C, H, W) -> (N*H*W, C) and (N, H, W) -> (N*H*W)
        loss = self.nll(log_score_weighted, labels)
        return loss


class ParsingRelationLoss(nn.Module):  # 相似损失
    def __init__(self):
        super(ParsingRelationLoss, self).__init__()

    def forward(self, logits):
        # logits: (N, C, H, W)
        n, c, h, w = logits.shape  # 批量大小 通道数 高度 宽度
        if torch.isinf(logits).any():
            print('logits has infinity')
        if torch.isnan(logits).any():
            print('logits has nan')
        loss_all = []
        for i in range(0, h - 1):
            # 计算相邻行差值 (n,c,w)
            loss_all.append(logits[:, :, i, :] - logits[:, :, i + 1, :])
        # loss0 : n,c,w
        loss = torch.cat(loss_all)  # 拼接差值张量
        # 计算平滑L1损失，目标是差值为0
        return torch.nn.functional.smooth_l1_loss(loss, torch.zeros_like(loss))


class ParsingRelationDis(nn.Module):  # 形状损失
    def __init__(self):
        super(ParsingRelationDis, self).__init__()
        self.l1 = torch.nn.L1Loss()  # L1损失函数
        # self.l1 = torch.nn.MSELoss()

    def forward(self, x):
        # x: (N, C, H, W)
        n, dim, num_rows, num_cols = x.shape
        # 使用softmax归一化前dim-1个通道 (N, C-1, H, W)
        x_processed = torch.nn.functional.softmax(x[:, :dim - 1, :, :], dim=1)

        # 计算位置差异的 embedding
        # embedding: (1, C-1, 1, 1)
        embedding = torch.Tensor(np.arange(dim - 1)).float().to(x.device).view(1, -1, 1, 1)

        # 计算加权和，得到每个像素的“位置”
        # pos: (N, H, W)
        pos = torch.sum(x_processed * embedding, dim=1)

        diff_list1 = []
        # 计算相邻行之间的位置差异
        for i in range(0, num_rows // 2):  # 只比较上半部分和下半部分的对应行
            diff_list1.append(pos[:, i, :] - pos[:, i + 1, :])

        loss = 0
        # 计算L1损失并累加到总损失中
        if len(diff_list1) > 1:  # 确保有至少两项可以比较
            for i in range(len(diff_list1) - 1):
                loss += self.l1(diff_list1[i], diff_list1[i + 1])
            loss /= (len(diff_list1) - 1)  # 平均损失
        else:  # 如果只有一项或没有，则损失为0
            loss = torch.tensor(0.0, device=x.device)

        return loss

class MultiHeadLoss(nn.Module):
    """
    collect all the loss we need
    """
    def __init__(self, losses, cfg, lambdas=None):
        """
        Inputs:
        - losses: (list)[nn.Module, nn.Module, ...]
        - cfg: config object
        - lambdas: (list) + IoU loss, weight for each loss
        """
        super().__init__()
        # lambdas: [cls, obj, iou, la_seg, ll_seg, ll_iou]
        if not lambdas:
            lambdas = [1.0 for _ in range(len(losses) + 3)]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = nn.ModuleList(losses)
        self.lambdas = lambdas
        self.cfg = cfg

    def forward(self, head_fields, head_targets, shapes, model):
        """
        Inputs:
        - head_fields: (list) output from each task head
        - head_targets: (list) ground-truth for each task head
        - model:

        Returns:
        - total_loss: sum of all the loss
        - head_losses: (tuple) contain all loss[loss1, loss2, ...]

        """
        # head_losses = [ll
        #                 for l, f, t in zip(self.losses, head_fields, head_targets)
        #                 for ll in l(f, t)]
        #
        # assert len(self.lambdas) == len(head_losses)
        # loss_values = [lam * l
        #                for lam, l in zip(self.lambdas, head_losses)
        #                if l is not None]
        # total_loss = sum(loss_values) if loss_values else None
        # print(model.nc)
        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model)

        return total_loss, head_losses

    def _forward_impl(self, predictions, targets, shapes, model):
        """

        Args:
            predictions: predicts of [[det_head1, det_head2, det_head3], drive_area_seg_head, lane_line_seg_head]
            targets: gts [det_targets, segment_targets, lane_targets]
            model:

        Returns:
            total_loss: sum of all the loss
            head_losses: list containing losses

        """
        cfg = self.cfg
        device = targets[0].device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = build_targets(cfg, predictions[0], targets[0], model)  # targets

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.0)  #

        BCEcls, BCEobj, BCEseg = self.losses

        # Calculate Losses
        nt = 0  # number of targets
        no = len(predictions[0])  # number of outputs
        balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6

        # calculate detection loss
        for i, pi in enumerate(predictions[0]):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                nt += n  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness ojb损失 正样本保留iou 负样本为0
                tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                # print(model.nc)
                if model.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                    t[range(n), tcls[i]] = cp  # 为正样本分配标签cp=1
                    lcls += BCEcls(ps[:, 5:], t)  # BCE 只有正样本的分类损失
            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

        drive_area_seg_predicts = predictions[1].view(-1)
        drive_area_seg_targets = targets[1].view(-1)
        lseg_da = BCEseg(drive_area_seg_predicts, drive_area_seg_targets)  # 可行驶区域损失

        lane_line_seg_predicts = predictions[2].view(-1)
        lane_line_seg_targets = targets[2].view(-1)
        lseg_ll = BCEseg(lane_line_seg_predicts, lane_line_seg_targets)  # 车道线损失

        metric = SegmentationMetric(2)
        nb, _, height, width = targets[1].shape
        pad_w, pad_h = shapes[0][1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        _,lane_line_pred=torch.max(predictions[2], 1)
        _,lane_line_gt=torch.max(targets[2], 1)
        lane_line_pred = lane_line_pred[:, pad_h:height-pad_h, pad_w:width-pad_w]
        lane_line_gt = lane_line_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]
        metric.reset()
        metric.addBatch(lane_line_pred.cpu(), lane_line_gt.cpu())  #
        IoU = metric.IntersectionOverUnion()
        liou_ll = 1 - IoU

        s = 3 / no  # output count scaling
        lcls *= cfg.LOSS.CLS_GAIN * s * self.lambdas[0]
        lobj *= cfg.LOSS.OBJ_GAIN * s * (1.4 if no == 4 else 1.) * self.lambdas[1]
        lbox *= cfg.LOSS.BOX_GAIN * s * self.lambdas[2]

        lseg_da *= cfg.LOSS.DA_SEG_GAIN * self.lambdas[3]
        lseg_ll *= cfg.LOSS.LL_SEG_GAIN * self.lambdas[4]
        liou_ll *= cfg.LOSS.LL_IOU_GAIN * self.lambdas[5]

        
        if cfg.TRAIN.DET_ONLY or cfg.TRAIN.ENC_DET_ONLY or cfg.TRAIN.DET_ONLY:  # 单独训练时不计算损失
            lseg_da = 0 * lseg_da
            lseg_ll = 0 * lseg_ll
            liou_ll = 0 * liou_ll
            
        if cfg.TRAIN.SEG_ONLY or cfg.TRAIN.ENC_SEG_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox

        if cfg.TRAIN.LANE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_da = 0 * lseg_da

        if cfg.TRAIN.DRIVABLE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_ll = 0 * lseg_ll
            liou_ll = 0 * liou_ll

        loss = lbox + lobj + lcls + lseg_da + lseg_ll + liou_ll
        # loss = lseg
        # return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
        return loss, (lbox.item(), lobj.item(), lcls.item(), lseg_da.item(), lseg_ll.item(), liou_ll.item(), loss.item())


class YolopLaneLoss(nn.Module):
    def __init__(self, cfg, device, lambdas=None):
        super().__init__()
        self.cfg = cfg
        # self.model = model
        # lambdas 列表中应该包含新的损失项的权重
        # 例如，如果之前有6项，现在应该有7项
        if not lambdas:
            lambdas = [1.0 for _ in range(4 + 3)]
        assert all(lam >= 0.0 for lam in lambdas)
        self.lambdas = lambdas

        self.losses = nn.ModuleList([
            nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.LOSS.CLS_POS_WEIGHT])).to(device),  # Cls loss
            nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.LOSS.OBJ_POS_WEIGHT])).to(device),  # Obj loss
            nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)  # Seg loss
        ])

        # 新增的 lane_robot 损失函数实例
        self.lane_robot_losses = nn.ModuleList([
            SoftmaxFocalLoss(gamma=2, ignore_lb=0),  # Focal Loss for lane classification
            ParsingRelationLoss(),  # Relation Loss for lane similarity
            ParsingRelationDis()  # Relation Disparity Loss for lane shape
        ])

        # 确保 cfg 中有这些配置项
        # cfg.LOSS.CLS_PW, cfg.LOSS.OBJ_PW
        # cfg.LOSS.CLS_GAIN, cfg.LOSS.OBJ_GAIN, cfg.LOSS.BOX_GAIN
        # cfg.LOSS.DA_SEG_GAIN, cfg.LOSS.LL_SEG_GAIN, cfg.LOSS.LL_IOU_GAIN
        # 新增
        # cfg.sim_loss_w, cfg.shp_loss_w (来自 get_loss_dict)
        # cfg.LOSS.LANE_ROBOT_GAIN (新的总的 lane_robot 损失的增益)

    def forward(self, head_fields, head_targets, shapes, model):
        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model)

        return total_loss, head_losses

    def _forward_impl(self, predictions, targets, shapes, model):
        """

        Args:
            predictions: predicts of [[det_head1, det_head2, det_head3], drive_area_seg_head, lane_line_seg_head]
            targets: gts [det_targets, segment_targets, lane_targets]
            model:

        Returns:
            total_loss: sum of all the loss
            head_losses: list containing losses

        """
        cfg = self.cfg
        device = targets[0].device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        # 0. Detection Loss Calculation
        tcls, tbox, indices, anchors = build_targets(cfg, predictions[0], targets[0], model)  # targets

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.0)  #

        BCEcls, BCEobj, BCEseg = self.losses  # Use the pre-instantiated losses

        nt = 0  # number of targets
        no = len(predictions[0])  # number of outputs
        balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6

        # calculate detection loss
        for i, pi in enumerate(predictions[0]):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                nt += n  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness ojb损失 正样本保留iou 负样本为0
                tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                # print(model.nc)
                if model.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                    t[range(n), tcls[i]] = cp  # 为正样本分配标签cp=1
                    lcls += BCEcls(ps[:, 5:], t)  # BCE 只有正样本的分类损失
            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

        drive_area_seg_predicts = predictions[1].view(-1)
        drive_area_seg_targets = targets[1].view(-1)
        lseg_da = BCEseg(drive_area_seg_predicts, drive_area_seg_targets)  # 可行驶区域损失

        lane_line_seg_predicts = predictions[2].view(-1)
        lane_line_seg_targets = targets[2].view(-1)
        lseg_ll = BCEseg(lane_line_seg_predicts, lane_line_seg_targets)  # 车道线损失

        metric = SegmentationMetric(2)
        nb, _, height, width = targets[1].shape
        pad_w, pad_h = shapes[0][1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        _, lane_line_pred = torch.max(predictions[2], 1)
        _, lane_line_gt = torch.max(targets[2], 1)
        lane_line_pred = lane_line_pred[:, pad_h:height - pad_h, pad_w:width - pad_w]
        lane_line_gt = lane_line_gt[:, pad_h:height - pad_h, pad_w:width - pad_w]
        metric.reset()
        metric.addBatch(lane_line_pred.cpu(), lane_line_gt.cpu())  #
        IoU = metric.IntersectionOverUnion()
        liou_ll = 1 - IoU

        # 4. Lane Robot Loss Calculation (New Addition)
        lane_robot_predicts = predictions[3]
        lane_robot_targets = targets[3]  # Ensure target is long type for FocalLoss

        # Calculate individual lane robot loss components
        l_cls_robot = self.lane_robot_losses[0](lane_robot_predicts, lane_robot_targets)  #
        l_sim_robot = self.lane_robot_losses[1](lane_robot_predicts)
        l_shp_robot = self.lane_robot_losses[2](lane_robot_predicts)

        # Combine lane robot losses with their internal weights (from get_loss_dict)
        # Assuming cfg has sim_loss_w and shp_loss_w
        lane_robot_total_loss = (1.0 * l_cls_robot +
                                 cfg.LOSS.LANE_ROBOT_SIM_WEIGHT * l_sim_robot +
                                 cfg.LOSS.LANE_ROBOT_SHP_WEIGHT * l_shp_robot)

        # Apply gains and lambdas for all losses
        s = 3 / no  # output count scaling for detection losses
        lcls *= cfg.LOSS.CLS_GAIN * s * self.lambdas[0]
        lobj *= cfg.LOSS.OBJ_GAIN * s * (1.4 if no == 4 else 1.) * self.lambdas[1]
        lbox *= cfg.LOSS.BOX_GAIN * s * self.lambdas[2]

        lseg_da *= cfg.LOSS.DA_SEG_GAIN * self.lambdas[3]
        lseg_ll *= cfg.LOSS.LL_SEG_GAIN * self.lambdas[4]
        liou_ll *= cfg.LOSS.LL_IOU_GAIN * self.lambdas[5]

        # Apply gain and lambda for the new lane_robot_total_loss
        # Assuming lambdas list has been extended to include a 7th item (index 6) for this loss
        lane_robot_total_loss *= cfg.LOSS.LANE_ROBOT_GAIN * self.lambdas[6]  # New gain for lane_robot loss

        if cfg.TRAIN.NO_DRIVABLE:  # 不训练可通行区域时
            lseg_da = 0 * lseg_da
        if cfg.TRAIN.NO_LL:  # 不训练可通行区域时
            lseg_ll = 0 * lseg_ll
            liou_ll = 0 * liou_ll
        # Conditional Loss Calculation based on training mode
        if cfg.TRAIN.DET_ONLY or cfg.TRAIN.ENC_DET_ONLY:  # cfg.TRAIN.DET_ONLY is duplicated in original code, fixed one instance
            lseg_da = 0 * lseg_da
            lseg_ll = 0 * lseg_ll
            liou_ll = 0 * liou_ll
            lane_robot_total_loss = 0 * lane_robot_total_loss  # Zero out new loss if only training detection

        if cfg.TRAIN.SEG_ONLY or cfg.TRAIN.ENC_SEG_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lane_robot_total_loss = 0 * lane_robot_total_loss  # Zero out new loss if only training segmentation

        if cfg.TRAIN.LANE_ONLY:  # This mode trains lane line related tasks
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_da = 0 * lseg_da
            # lane_robot_total_loss should NOT be zeroed out here, as it's part of lane tasks

        if cfg.TRAIN.DRIVABLE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_ll = 0 * lseg_ll
            liou_ll = 0 * liou_ll
            lane_robot_total_loss = 0 * lane_robot_total_loss  # Zero out new loss if only training drivable area

        if cfg.TRAIN.ROBOT_LANE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_ll = 0 * lseg_ll
            liou_ll = 0 * liou_ll
            lseg_da = 0 * lseg_da

        # Sum all losses
        loss = lbox + lobj + lcls + lseg_da + lseg_ll + liou_ll + lane_robot_total_loss

        # Return total loss and individual loss components for logging/monitoring
        return loss, (lbox.item(), lobj.item(), lcls.item(),
                      lseg_da.item(), lseg_ll.item(), liou_ll.item(),
                      lane_robot_total_loss.item(),  # Add the total lane_robot loss here
                      loss.item())


def get_loss(cfg, device):
    """
    get MultiHeadLoss

    Inputs:
    -cfg: configuration use the loss_name part or 
          function part(like regression classification)
    -device: cpu or gpu device

    Returns:
    -loss: (MultiHeadLoss)

    """
    # class loss criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.CLS_POS_WEIGHT])).to(device)
    # object loss criteria
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.OBJ_POS_WEIGHT])).to(device)
    # segmentation loss criteria
    BCEseg = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)
    # Focal loss
    gamma = cfg.LOSS.FL_GAMMA  # focal loss gamma
    if gamma > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, gamma), FocalLoss(BCEobj, gamma)

    loss_list = [BCEcls, BCEobj, BCEseg]
    loss = MultiHeadLoss(loss_list, cfg=cfg, lambdas=cfg.LOSS.MULTI_HEAD_LAMBDA)
    return loss

# example
# class L1_Loss(nn.Module)


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        # alpha  balance positive & negative samples
        # gamma  focus on difficult samples
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
