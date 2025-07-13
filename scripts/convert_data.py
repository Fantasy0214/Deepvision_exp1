from ultralytics.data.converter import convert_coco   # ← 位置就在这里！

convert_coco(
    labels_dir="data/coco_lite10k/annotations",   # 放 *.json 的目录
    save_dir="data/coco_lite10k",                 # 会生成 labels/{train,val,test}
    use_segments=False,       # 只要 bbox
    use_keypoints=False,      # 不要关键点
    cls91to80=True            # COCO 91 → 80 类映射
)

print("✅ 已生成 YOLO-txt 标签（data/coco_lite10k/labels/*）")
