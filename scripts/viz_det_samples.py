# ── scripts/viz_det_samples.py ───────────────────────────────
import random, math, matplotlib.pyplot as plt, torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import pathlib, os

ROOT = "data/coco_lite10k"
VAL_DIR = f"{ROOT}/images/val"
MODEL_WEIGHTS = "runs/det/quick_det2/weights/best.pt"
NUM_SAMPLES = 10
CONF_THRES  = 0.25

# ------------- 1. 随机采样 10 张验证图 ------------------------
all_imgs = [f"{VAL_DIR}/{name}" for name in os.listdir(VAL_DIR) if name.endswith(".jpg")]
sample_imgs = random.sample(all_imgs, NUM_SAMPLES)

# ------------- 2. 加载 YOLOv8n 探测模型 ----------------------
model = YOLO(MODEL_WEIGHTS)

# ------------- 3. 推理 + 可视化 ------------------------------
ncols = 5
plt.figure(figsize=(ncols*4, math.ceil(NUM_SAMPLES/ncols)*4))
font = ImageFont.truetype("arial.ttf", 12)  

for i, path in enumerate(sample_imgs, 1):
    # 推理
    res = model(path, conf=CONF_THRES, verbose=False)[0]

    # 画框到 PIL Image
    im = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(im)
    for box, cls, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
        x1,y1,x2,y2 = box.tolist()
        label = model.names[int(cls)]
        draw.rectangle([x1,y1,x2,y2], outline="red", width=2)
        draw.text((x1, y1-12), f"{label}:{conf:.2f}", fill="red", font=font)

    # Matplotlib 显示
    plt.subplot(math.ceil(NUM_SAMPLES/ncols), ncols, i)
    plt.imshow(im); plt.axis('off')

plt.tight_layout()
plt.savefig("det_vis_val10.png", dpi=200)
print("✓ 已保存 det_vis_val10.png")
