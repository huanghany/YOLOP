import argparse
import os, sys
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import time
import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import amp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
from lib.utils import DataLoaderX, torch_distributed_zero_first
from tensorboardX import SummaryWriter

import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import train
from lib.core.function import validate
from lib.core.general import fitness
from lib.models import get_YOLOP_LANE_net
from lib.utils import is_parallel
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger, select_device
from lib.utils import run_anchor


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multitask network')
    # general
    # parser.add_argument('--cfg',
    #                     help='experiment configure file name',
    #                     required=True,
    #                     type=str)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='runs/')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    args = parser.parse_args()

    return args


def main():
    """
    主训练函数。
    """
    # 设置所有配置
    args = parse_args()  # 解析命令行参数
    update_config(cfg, args)  # 使用解析的参数更新全局配置

    # 设置DDP（分布式数据并行）变量
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1  # 全局进程数量
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1  # 当前进程的全局排名

    rank = global_rank  # 当前进程的排名
    #print(rank)
    # TODO: 处理分布式训练的日志记录
    # 设置日志器，tb_log_dir表示TensorBoard的日志目录
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'train', rank=rank)

    # 仅在主进程（rank -1表示非分布式，rank 0表示分布式主进程）创建TensorBoard写入器
    if rank in [-1, 0]:
        logger.info(pprint.pformat(args))  # 打印命令行参数
        logger.info(cfg)  # 打印配置

        writer_dict = {
            'writer': SummaryWriter(log_dir=tb_log_dir),  # TensorBoard写入器
            'train_global_steps': 0,  # 训练的全局步数
            'valid_global_steps': 0,  # 验证的全局步数

        }
    else:
        writer_dict = None  # 从属进程不创建写入器

    # cuDNN相关设置
    cudnn.benchmark = cfg.CUDNN.BENCHMARK  # 启用cuDNN自动寻找最优算法
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC  # 确保cuDNN算法具有确定性
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED  # 启用cuDNN

    # 构建模型
    # start_time = time.time()
    print("开始构建模型...")
    # DP (DataParallel) 模式
    # 如果不是DEBUG模式，则选择GPU设备，否则选择CPU
    device = select_device(logger, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS)) if not cfg.DEBUG \
        else select_device(logger, 'cpu')

    # 如果是DDP（DistributedDataParallel）模式
    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank  # 确保有足够的GPU
        torch.cuda.set_device(args.local_rank)  # 设置当前进程使用的GPU
        device = torch.device('cuda', args.local_rank)  # 定义设备
        dist.init_process_group(backend='nccl', init_method='env://')  # 初始化分布式后端

    print("加载模型到设备")
    model = get_YOLOP_LANE_net(cfg).to(device)  # 根据配置获取模型并将其移动到指定设备

    # 定义损失函数和优化器
    criterion = get_loss(cfg, device=device)  # 获取损失函数实例
    optimizer = get_optimizer(cfg, model)  # 获取优化器实例

    # 加载检查点模型
    best_perf = 0.0  # 最佳性能指标
    best_model = False  # 是否为最佳模型
    last_epoch = -1  # 上一个训练的epoch

    # 定义模型各部分的参数索引，用于选择性地加载或冻结参数
    Encoder_para_idx = [str(i) for i in range(0, 17)]
    Det_Head_para_idx = [str(i) for i in range(17, 25)]
    Da_Seg_Head_para_idx = [str(i) for i in range(25, 34)]  # 可通行区域
    Ll_Seg_Head_para_idx = [str(i) for i in range(34, 43)]  # 车道线

    # 学习率调整函数：余弦退火
    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                   (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # 学习率调度器
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH  # 训练开始的epoch

    # 仅在主进程（rank -1 或 0）处理模型加载和冻结逻辑
    if rank in [-1, 0]:
        checkpoint_file = os.path.join(
            os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET), 'checkpoint.pth'
        )
        # 加载预训练的完整模型
        if os.path.exists(cfg.MODEL.PRETRAINED):
            logger.info("=> 正在加载模型 '{}'".format(cfg.MODEL.PRETRAINED))
            checkpoint = torch.load(cfg.MODEL.PRETRAINED)
            begin_epoch = checkpoint['epoch']  # 读取 epoch
            # best_perf = checkpoint['perf']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])  # 加载模型状态字典
            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器状态
            logger.info("=> 已加载检查点 '{}' (epoch {})".format(
                cfg.MODEL.PRETRAINED, checkpoint['epoch']))
            #cfg.NEED_AUTOANCHOR = False     # 禁用自动锚框计算

        # 加载预训练的检测分支模型权重
        if os.path.exists(cfg.MODEL.PRETRAINED_DET):
            logger.info("=> 正在从 '{}' 加载检测分支模型权重".format(cfg.MODEL.PRETRAINED_DET))
            det_idx_range = [str(i) for i in range(0, 25)]  # 检测分支相关的参数索引
            model_dict = model.state_dict()  # 获取当前模型的状态字典
            checkpoint_file = cfg.MODEL.PRETRAINED_DET
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            last_epoch = checkpoint['epoch']
            # 过滤检查点中只与检测分支相关的权重
            checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.split(".")[1] in det_idx_range}
            model_dict.update(checkpoint_dict)  # 更新模型状态字典
            model.load_state_dict(model_dict)  # 加载模型权重
            logger.info("=> 已加载检测分支检查点 '{}' ".format(checkpoint_file))

        # 自动恢复训练
        if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
            logger.info("=> 正在加载检查点 '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            # best_perf = checkpoint['perf']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=False)  # 加载模型状态字典
            # optimizer = get_optimizer(cfg, model) # 如果需要，可以重新获取优化器
            # optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器状态
            logger.info("=> 已加载检查点 '{}' (epoch {})".format(
                checkpoint_file, checkpoint['epoch']))
            #cfg.NEED_AUTOANCHOR = False     # 禁用自动锚框计算
        # model = model.to(device) # 模型已在前面移动到设备

        # 根据配置冻结特定层的参数
        if cfg.TRAIN.SEG_ONLY:  # 只训练两个分割分支
            logger.info('冻结编码器和检测头...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # 默认所有层都可训练
                # 如果参数属于编码器或检测头，则冻结
                if k.split(".")[1] in Encoder_para_idx + Det_Head_para_idx:
                    print('冻结 %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.DET_ONLY:  # 只训练检测分支
            logger.info('冻结编码器和两个分割头...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # 默认所有层都可训练
                # 如果参数属于编码器或两个分割头，则冻结
                if k.split(".")[1] in Encoder_para_idx + Da_Seg_Head_para_idx + Ll_Seg_Head_para_idx:
                    print('冻结 %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.ENC_SEG_ONLY:  # 只训练编码器和两个分割分支
            logger.info('冻结检测头...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # 默认所有层都可训练
                # 如果参数属于检测头，则冻结
                if k.split(".")[1] in Det_Head_para_idx:
                    print('冻结 %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.ENC_DET_ONLY or cfg.TRAIN.DET_ONLY:  # 只训练编码器和检测分支
            logger.info('冻结两个分割头...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # 默认所有层都可训练
                # 如果参数属于两个分割头，则冻结
                if k.split(".")[1] in Da_Seg_Head_para_idx + Ll_Seg_Head_para_idx:
                    print('冻结 %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.LANE_ONLY:  # 只训练车道线分割分支
            logger.info('冻结编码器、检测头和可行驶区域分割头...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # 默认所有层都可训练
                # 如果参数属于编码器、检测头或可行驶区域分割头，则冻结
                if k.split(".")[1] in Encoder_para_idx + Da_Seg_Head_para_idx + Det_Head_para_idx:
                    print('冻结 %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.DRIVABLE_ONLY:  # 只训练可行驶区域分割分支 NO_DRIVABLE
            logger.info('冻结编码器、检测头和车道线分割头...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # 默认所有层都可训练
                # 如果参数属于编码器、车道线分割头或检测头，则冻结
                if k.split(".")[1] in Encoder_para_idx + Ll_Seg_Head_para_idx + Det_Head_para_idx:
                    print('冻结 %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.NO_DRIVABLE:  # 不训练可行驶区域分割分支
            logger.info('冻结可行驶区域分割头...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # 默认所有层都可训练
                # 如果参数属于编码器、检测头或可行驶区域分割头，则冻结
                if k.split(".")[1] in Da_Seg_Head_para_idx:
                    print('冻结 %s' % k)
                    v.requires_grad = False
    # 模型并行化设置
    if rank == -1 and torch.cuda.device_count() > 1:
        # 如果不是DDP模式且有多个GPU，使用DataParallel
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS)
        # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda() # 已经通过.to(device)移动
    # DDP 模式
    if rank != -1:
        # 如果是DDP模式，使用DistributedDataParallel
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # 分配模型参数
    model.gr = 1.0  # 可能与一些模型内部计算相关
    model.nc = 1  # 类别数量  单任务  huayi 2

    print("开始加载数据")
    # 数据加载
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # 标准化参数
    )

    # 实例化训练数据集
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=True,  # 训练模式
        inputsize=cfg.MODEL.IMAGE_SIZE,  # 输入图像大小 [320, 320]
        transform=transforms.Compose([  # 图像变换
            transforms.ToTensor(),  # 转换为Tensor
            normalize,  # 归一化
        ])
    )
    # 分布式采样器，仅在DDP模式下使用
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if rank != -1 else None

    # 训练数据加载器
    train_loader = DataLoaderX(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),  # 总批次大小
        shuffle=(cfg.TRAIN.SHUFFLE & rank == -1),  # 非分布式模式下随机打乱
        num_workers=cfg.WORKERS,  # 工作进程数
        sampler=train_sampler,  # 采样器
        pin_memory=cfg.PIN_MEMORY,  # 是否将数据加载到CUDA的固定内存
        collate_fn=dataset.AutoDriveDataset.collate_fn  # 批量处理函数
    )
    num_batch = len(train_loader)  # 每个epoch的批次数量

    # 仅在主进程（rank -1 或 0）加载验证数据集和数据加载器
    if rank in [-1, 0]:
        # 实例化验证数据集
        valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
            cfg=cfg,
            is_train=False,  # 验证模式
            inputsize=cfg.MODEL.IMAGE_SIZE,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

        # 验证数据加载器
        valid_loader = DataLoaderX(
            valid_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
            shuffle=False,  # 验证集不打乱
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=dataset.AutoDriveDataset.collate_fn
        )
        print('数据加载完成')

    # 仅在主进程（rank -1 或 0）处理锚框生成或加载
    if rank in [-1, 0]:
        if cfg.NEED_AUTOANCHOR:  # 如果需要自动生成锚框
            logger.info("开始检查锚框")
            run_anchor(logger, train_dataset, model=model, thr=cfg.TRAIN.ANCHOR_THRESHOLD,
                       imgsz=min(cfg.MODEL.IMAGE_SIZE))  # 运行锚框聚类
        else:
            logger.info("锚框加载成功")
            # 获取检测头并打印其锚框
            det = model.module.model[model.module.detector_index] if is_parallel(model) \
                else model.model[model.detector_index]
            logger.info(str(det.anchors))

    # 训练
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000)  # 学习率预热步数
    scaler = amp.GradScaler(enabled=device.type != 'cpu')  # 自动混合精度 GradScaler，CPU模式下禁用
    print('=> 开始训练...')
    # 训练循环
    for epoch in range(begin_epoch + 1, cfg.TRAIN.END_EPOCH + 1):
        if rank != -1:  # DDP模式下，每个epoch开始时设置采样器，以确保数据不同
            train_loader.sampler.set_epoch(epoch)
        # 训练一个epoch
        train(cfg, train_loader, model, criterion, optimizer, scaler,
              epoch, num_batch, num_warmup, writer_dict, logger, device, rank)  # 调用训练函数

        lr_scheduler.step()  # 更新学习率

        # 在验证集上评估
        # 定期或在训练结束时进行验证，且仅在主进程（rank -1 或 0）执行
        if (epoch % cfg.TRAIN.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH) and rank in [-1, 0]:
            # print('validate')
            da_segment_results, ll_segment_results, detect_results, total_loss, maps, times = validate(
                epoch, cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, writer_dict,
                logger, device, rank
            )  # 调用验证函数
            fi = fitness(np.array(detect_results).reshape(1, -1))  # 计算目标检测的fitness指标

            # 格式化并记录评估结果
            msg = 'Epoch: [{0}]    Loss({loss:.3f})\n' \
                  '可行驶区域分割: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                  '车道线分割: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
                  '检测: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n' \
                  '时间: 推理({t_inf:.4f}s/帧)  NMS({t_nms:.4f}s/帧)'.format(
                epoch, loss=total_loss, da_seg_acc=da_segment_results[0], da_seg_iou=da_segment_results[1],
                da_seg_miou=da_segment_results[2],
                ll_seg_acc=ll_segment_results[0], ll_seg_iou=ll_segment_results[1], ll_seg_miou=ll_segment_results[2],
                p=detect_results[0], r=detect_results[1], map50=detect_results[2], map=detect_results[3],
                t_inf=times[0], t_nms=times[1])
            logger.info(msg)  # 记录日志

            # if perf_indicator >= best_perf:
            #     best_perf = perf_indicator
            #     best_model = True
            # else:
            #     best_model = False

        # 保存检查点模型和最佳模型 (仅在主进程保存)
        if rank in [-1, 0]:
            if epoch % 20 == 0:
                savepath = os.path.join(final_output_dir, f'epoch-{epoch}.pth')  # 当前epoch的模型保存路径
                logger.info('=> 正在保存检查点到 {}'.format(savepath))
                save_checkpoint(
                    epoch=epoch,
                    name=cfg.MODEL.NAME,
                    model=model,
                    # 'best_state_dict': model.module.state_dict(),
                    # 'perf': perf_indicator,
                    optimizer=optimizer,
                    output_dir=final_output_dir,
                    filename=f'epoch-{epoch}.pth'  # 保存当前epoch的模型
                )
            save_checkpoint(
                epoch=epoch,
                name=cfg.MODEL.NAME,
                model=model,
                # 'best_state_dict': model.module.state_dict(),
                # 'perf': perf_indicator,
                optimizer=optimizer,
                output_dir=os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET),
                filename='checkpoint.pth'  # 保存最新的检查点（用于自动恢复）
            )

    # 保存最终模型 (仅在主进程保存)
    if rank in [-1, 0]:
        final_model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> 正在保存最终模型状态到 {}'.format(
            final_model_state_file)
        )
        # 获取模型的state_dict，如果是并行模式则获取module的state_dict
        model_state = model.module.state_dict() if is_parallel(model) else model.state_dict()
        torch.save(model_state, final_model_state_file)  # 保存最终模型状态
        writer_dict['writer'].close()  # 关闭TensorBoard写入器
    else:
        dist.destroy_process_group()  # 从属进程销毁分布式进程组


if __name__ == '__main__':
    main()
