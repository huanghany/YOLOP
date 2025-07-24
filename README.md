# YoloP-UFast

## 说明

* 本工作将YOLOP和U-FAST结合，用U-FAST的车道线检测替换YOLOP中车道线分割部分

## 项目结构
- 只重点描述本项目相关部分（其他详见`README _CH.md`）
- 重点记录tools中各脚本作用
```
├─inference  # 推理结果图片/视频存放位置
├─lib
│ ├─config/default   # 训练配置文件
│ ├─core
│ │ ├─evaluate.py   # 评价指标计算
│ │ ├─function.py   # 训练和验证函数
│ │ ├─loss.py   # 损失函数计算
│ ├─dataset 数据集相关
│ │ ├─AutoDriveDataset.py   # 数据集调用迭代器
│ │ ├─robot_view.py   # 自定义数据集
│ ├─models 网络结构
│ │ ├── common3.py 车道线检测网络
│ │ ├── YOLOP_LANE.py 合并后网络总框架
├── tools
│ ├── load_pth.py 读取模型（用于检验）
│ ├── show_label.py 标签可视化
│ ├── train_yolop_lane.py
│ ├── transform_labels.py
│ ├── transform_model.py
│ └── yolop_lane_det.py
├─weights    # 存放模型位置
```

---

### Requirement

整个代码库是在 python 3.版本, PyTorch 1.7+版本和 torchvision 0.8+版本上开发的:

```
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch
```

其他依赖库的版本要求详见`requirements.txt`：

```setup
pip install -r requirements.txt
```

### Data preparation

我们推荐按照如下图片数据集文件结构:

```
├─dataset root
│ ├─images
│ │ ├─train
│ │ ├─val
│ ├─det_annotations
│ │ ├─train
│ │ ├─val
│ ├─da_seg_annotations
│ │ ├─train
│ │ ├─val
│ ├─ll_annotations
│ │ ├─train
│ │ ├─val
```

在 `./lib/config/default.py`下更新数据集的路径配置。

### 模型训练

你可以在 `./lib/config/default.py`设定训练配置. (包括:  预训练模型的读取，损失函数， 数据增强，optimizer，训练预热和余弦退火，自动anchor，训练轮次epoch, batch_size)



如果你想尝试交替优化或者单一任务学习，可以在`./lib/config/default.py` 中将对应的配置选项修改为 `True`。(如下，所有的配置都是 `False`, which means training multiple tasks end to end)。

```python
# Alternating optimization
_C.TRAIN.SEG_ONLY = False           # Only train two segmentation branchs
_C.TRAIN.DET_ONLY = False           # Only train detection branch
_C.TRAIN.ENC_SEG_ONLY = False       # Only train encoder and two segmentation branchs
_C.TRAIN.ENC_DET_ONLY = False       # Only train encoder and detection branch

# Single task 
_C.TRAIN.DRIVABLE_ONLY = False      # Only train da_segmentation task
_C.TRAIN.LANE_ONLY = False          # Only train ll_segmentation task
_C.TRAIN.DET_ONLY = False          # Only train detection task
```

开始训练:

```shell
python tools/train.py
```
多GPU训练:
```
python -m torch.distributed.launch --nproc_per_node=N tools/train.py  # N: the number of GPUs
```

