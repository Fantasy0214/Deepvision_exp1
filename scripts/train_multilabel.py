# -*- coding: utf-8 -*-
import os, torch, timm
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelF1Score
from tqdm import tqdm
from dataset_coco_multilabel import CocoMultiLabel

def main():
    # ------------ 配置 ------------
    ROOT       = "data/coco_lite10k"
    IMG_SIZE   = 224
    BATCH      = 32
    EPOCHS     = 25
    LR         = 3e-4
    NUM_CLASS  = 80
    SAVE_DIR   = "runs/cls"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ------------ 数据 ------------
    train_ds = CocoMultiLabel(f"{ROOT}/cls_train.txt", ROOT, IMG_SIZE, True, True)
    val_ds   = CocoMultiLabel(f"{ROOT}/cls_val.txt",   ROOT, IMG_SIZE, True, False)
    train_loader = DataLoader(train_ds, BATCH, True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   64,   False, num_workers=4, pin_memory=True)

    # ------------ 模型 ------------
    backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
    model = nn.Sequential(backbone,
                          nn.Linear(backbone.num_features, NUM_CLASS)).cuda()

    # ------------ 优化器 & lr 计划 ------------
    crit = nn.BCEWithLogitsLoss()
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # ------------ 指标 ------------
    mAP = MultilabelAveragePrecision(num_labels=NUM_CLASS, average="micro").cuda()
    F1  = MultilabelF1Score(num_labels=NUM_CLASS, average="macro",
                            threshold=0.5).cuda()

    best_map = 0.0
    for ep in range(1, EPOCHS+1):
        # ---- 训练 ----
        model.train()
        pbar = tqdm(train_loader, desc=f"[Epoch {ep}/{EPOCHS}]")
        for x, y in pbar:
            x, y = x.cuda(), y.cuda()
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        # ---- 验证 ----
        model.eval(); mAP.reset(); F1.reset()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                logits = model(x)
                mAP.update(logits, y.int())
                F1.update((logits.sigmoid() > 0.5).int(), y.int())

        cur_map = mAP.compute().item()
        cur_f1  = F1.compute().item()
        print(f"[VAL] micro-mAP={cur_map:.4f}  macro-F1={cur_f1:.4f}")

        # ---- 保存权重 ----
        torch.save(model.state_dict(),
                   f"{SAVE_DIR}/epoch{ep:02d}.pt")
        if cur_map > best_map:
            best_map = cur_map
            torch.save(model.state_dict(),
                       f"{SAVE_DIR}/best.pt")
            print(f"★ 新最佳 micro-mAP={best_map:.4f} → 保存为 best.pt")

if __name__ == "__main__":
    main()
