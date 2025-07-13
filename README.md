## 1. 项目结构
```
deepvision-exp1/
├── configs/                     # ★ YAML等超参、路径配置
│   └── coco_lite10k_det.yaml    #   ↳ 目标检测 YAML
|
├── data/                        # ★ 所有数据文件
│   └── coco_lite10k/            #   ↳ 抽样后的实验子集
│       ├── images/
|       ├── labels/
│       ├── annotations/
|       ├── cls_train.txt        #
|       ├── cls_val.txt
│       └── cls_test.txt
│
├── scripts/                     # ★ 可复现的 Python 脚本
│   ├── build_coco_lite10k.py    #   ↳ 抽样 & 生成标注 对于分类生成txt文件，检测生成json文件
│   ├── convert_data.py          # 对于json文件格式的coco数据集转为yolo格式
│   ├── instances_train_fix.py   # 对于train.json文件的修复
│   ├── instances_val_fix.py     # 对于val.json文件的修复
│   ├── dataset_coco_multilabel.py
│   ├── train_multilabel.py
│   ├── viz_cls_samples.py       # 模型分类可视化脚本
│   └── viz_det_samples.py       # 模型检测可视化脚本
│
├── runs/                        # （模型权重、日志）
│   ├── cls/best.pt
│   └── det/...
|
├── cls_vis_val10.png            # 模型分类可视化结果
|
├── det_vis_val10.png            # 模型检测可视化结果
|
└── README.md                    # 项目简介、复现步骤
```

## 2. 多标签分类（ResNet-50）

### 虚拟环境的安装
```
conda create -n cocoCls python=3.11 -y
conda activate cocoCls
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm torchmetrics lightning tqdm
```
### 训练指令
```
python scripts/train_multilabel.py
```
### 结果可视化
```
python scripts/viz_val_samples.py
```
## 3. 目标检测（YOLOv8-nano）
### 虚拟环境的安装

```
conda activate cvexp
$Env:KMP_DUPLICATE_LIB_OK="TRUE"
```

### 训练指令
### 关键参数解释

| 参数                   | 作用 / 建议                                              |
| -------------------- | ---------------------------------------------------- |
| **model=yolov8n.pt** | nano 版 \~3.2 M 参数，显存最低，便于快速测试。                       |
| **imgsz=512**        | COCO 默认 640；降到 512 和批量 4 可把训练显存压到 3 GB。             |
| **save\_period**     | 长训时可设 10；快速验证设 1 方便随时检查权重文件是否写入。                     |
| **optimizer / lr0**  | 留空即用默认 AdamW + 0.01；调参时可加 `optimizer=SGD lr0=0.005`。 |

### 初始训练指令
```
yolo detect train `
    model=yolov8n.pt `
    data=config/coco10k_det.yaml `
    epochs=1 `
    imgsz=512 `
    batch=16 `
    amp=True `
    workers=3 `
    project=runs/det `
    name=final_attempt `
    format=coco
```
### 优化后的训练指令
```
yolo detect train `
    model=yolov8n.pt `
    data=config/coco10k_det.yaml `
    epochs=50 imgsz=640 batch=4 `
    optimizer=SGD lr0=0.005 `
    freeze=10 mosaic=0.8 close_mosaic=5 cos_lr=True `
    workers=2 project=runs/det name=quick_det
```

### 结果可视化
```

```