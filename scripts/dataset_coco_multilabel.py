# -*- coding: utf-8 -*-
"""
把 cls_train/val/test.txt 读成 (image, multi-hot) 样本
"""
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T

# 91 → 80 连续 id 映射（官方 COCO）
MAP91_80 = {
    1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10,
    13:11, 14:12, 15:13, 16:14, 17:15, 18:16, 19:17, 20:18, 21:19, 22:20,
    23:21, 24:22, 25:23, 27:24, 28:25, 31:26, 32:27, 33:28, 34:29, 35:30,
    36:31, 37:32, 38:33, 39:34, 40:35, 41:36, 42:37, 43:38, 44:39, 46:40,
    47:41, 48:42, 49:43, 50:44, 51:45, 52:46, 53:47, 54:48, 55:49, 56:50,
    57:51, 58:52, 59:53, 60:54, 61:55, 62:56, 63:57, 64:58, 65:59, 67:60,
    70:61, 72:62, 73:63, 74:64, 75:65, 76:66, 77:67, 78:68, 79:69, 80:70,
    81:71, 82:72, 84:73, 85:74, 86:75, 87:76, 88:77, 89:78, 90:79
}

class CocoMultiLabel(torch.utils.data.Dataset):
    def __init__(self, txt_file:str, root:str|Path,
                 img_size:int=224, map91to80:bool=True, train:bool=True):
        self.root   = Path(root)
        self.lines  = [l.strip().split() for l in open(txt_file, encoding='utf-8')]
        self.map91  = map91to80
        self.C      = 80 if map91to80 else 91

        # 图像变换
        aug = [T.Resize((img_size, img_size))]
        if train:
            aug += [T.RandomHorizontalFlip(),
                    T.ColorJitter(0.2, 0.2, 0.2, 0.1)]
        aug += [T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],
                            [0.229,0.224,0.225])]
        self.tf = T.Compose(aug)

    def __len__(self): return len(self.lines)

    def __getitem__(self, idx):
        path, *cats = self.lines[idx]
        img = Image.open(self.root / path).convert('RGB')
        y   = torch.zeros(self.C)
        for c in cats:
            cid = int(c)
            if self.map91:
                if cid in MAP91_80:
                    y[MAP91_80[cid]] = 1.
            else:
                y[cid] = 1.
        return self.tf(img), y
