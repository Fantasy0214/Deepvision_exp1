# -*- coding: utf-8 -*-
"""
随机挑 10 张验证图 → 推理 → 可视化真标签 vs 预测 (Top-3)
保存到 vis_val10.png
"""
import random, math, matplotlib.pyplot as plt, torch, timm
from dataset_coco_multilabel import CocoMultiLabel
import pathlib, os
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]  # 上两级
ROOT = PROJECT_ROOT / "data" / "coco_lite10k"
MODEL_PATH  = "runs/cls/best.pt"
NUM_SAMPLES = 10
TOPK        = 3
THRESH      = 0.50           # > THRESH 判为正
IMG_SIZE    = 224

COCO80_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# ---------- 数据 ----------
val_ds = CocoMultiLabel(f"{ROOT}/cls_val.txt", ROOT,
                        img_size=IMG_SIZE, map91to80=True, train=False)
idxs = random.sample(range(len(val_ds)), NUM_SAMPLES)

# ---------- 模型 ----------
backbone = timm.create_model("resnet50", pretrained=False, num_classes=0)
model = torch.nn.Sequential(
    backbone,
    torch.nn.Linear(backbone.num_features, 80)
).cuda().eval()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cuda"))

# ---------- 反标准化函数 ----------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
def denorm(x):       # x: Tensor(C, H, W)
    return (x.cpu()*IMAGENET_STD + IMAGENET_MEAN).clamp(0,1)

# ---------- 可视化 ----------
ncols = 5
plt.figure(figsize=(ncols*3, math.ceil(NUM_SAMPLES/ncols)*3))

for i, idx in enumerate(idxs, 1):
    img, gt = val_ds[idx]
    with torch.no_grad():
        probs = model(img.unsqueeze(0).cuda()).sigmoid()[0].cpu()

    # 真标签
    gt_ids   = torch.where(gt==1)[0].tolist()
    gt_names = [COCO80_NAMES[j] for j in gt_ids]

    # 预测 Top-k
    conf, pred_ids = torch.topk(probs, TOPK)
    pred_str = [f"{COCO80_NAMES[p.item()]}({conf[k]:.2f})"
                for k, p in enumerate(pred_ids)]

    title =  ("GT: " + (', '.join(gt_names) if gt_names else '∅') + "\n" +
              "PR: " + ', '.join(pred_str))

    plt.subplot(math.ceil(NUM_SAMPLES/ncols), ncols, i)
    plt.imshow(denorm(img).permute(1,2,0))
    plt.axis('off')
    plt.title(title, fontsize=8)

plt.tight_layout()
plt.savefig("cls_vis_val10.png", dpi=200)
print("✓ 已保存 cls_vis_val10.png")
