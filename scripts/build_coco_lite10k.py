from __future__ import annotations
from pathlib import Path
import json, random, shutil, collections, tqdm, argparse, math


def split_indices(imgs: list[dict], ratios: tuple[float, float, float], seed: int):
    random.seed(seed)
    random.shuffle(imgs)
    n = len(imgs)
    n_train = math.floor(ratios[0] * n)
    n_val   = math.floor(ratios[1] * n)
    return imgs[:n_train], imgs[n_train:n_train + n_val], imgs[n_train + n_val:]


def save_subset_json(full_ann: dict, subset_imgs: list[dict], out_file: Path):
    ids = {im["id"] for im in subset_imgs}
    sub = {k: [] if isinstance(v, list) else v for k, v in full_ann.items()}
    sub["images"] = subset_imgs
    sub["annotations"] = [a for a in full_ann["annotations"] if a["image_id"] in ids]
    json.dump(sub, open(out_file, "w"))


def save_cls_txt(subset_imgs: list[dict], id2cats: dict[int, set[int]],
                 subset_name: str, dst: Path):
    lines = [
        f"images/{subset_name}/{im['file_name']} "
        + " ".join(map(str, sorted(id2cats[im['id']])))
        for im in subset_imgs
    ]
    (dst / f"cls_{subset_name}.txt").write_text("\n".join(lines))


def main(src: str | Path,
         dst: str | Path = "data/coco_lite10k",
         n: int = 10_000,
         split: tuple[float, float, float] = (0.8, 0.1, 0.1),
         seed: int = 42):
    src, dst = Path(src), Path(dst)
    img_dst  = dst / "images"
    for sub in ("train", "val", "test"):
        (img_dst / sub).mkdir(parents=True, exist_ok=True)

    print("📖  loading annotations …")
    ann_full = json.load(open(src / "annotations/instances_train2017.json"))

    imgs_sampled = random.sample(ann_full["images"], n)
    imgs_train, imgs_val, imgs_test = split_indices(imgs_sampled, split, seed)
    subsets = dict(train=imgs_train, val=imgs_val, test=imgs_test)

    # 预先把 10k id 做成 set，加速后续查找
    sampled_ids = {im["id"] for im in imgs_sampled}

    # image_id → set(category_id)（只扫一次 annotation）
    id2cats: dict[int, set[int]] = collections.defaultdict(set)
    for a in ann_full["annotations"]:
        if a["image_id"] in sampled_ids:
            id2cats[a["image_id"]].add(a["category_id"])

    # 复制图片 + 导出 JSON / txt
    for sub, imgs in subsets.items():
        print(f"\n🗂  {sub}: {len(imgs)} images")
        for im in tqdm.tqdm(imgs, desc=f"copy {sub}", ncols=70):
            shutil.copy(src / f"train2017/{im['file_name']}",
                        img_dst / sub / im["file_name"])

        save_subset_json(ann_full, imgs, dst / f"instances_{sub}.json")
        save_cls_txt(imgs, id2cats, sub, dst)

    print(f"\n✅  LiteCOCO ready ➜  {dst.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="COCO root  e.g.  E:/coco_temp/coco2017")
    ap.add_argument("--dst", default="data/coco_lite10k")
    ap.add_argument("--num", type=int, default=10_000)
    ap.add_argument("--split", nargs=3, type=float, default=(0.8, 0.1, 0.1),
                    metavar=("TRAIN", "VAL", "TEST"),
                    help="ratios, must sum to 1.0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    assert abs(sum(args.split) - 1) < 1e-6, "--split must sum to 1.0"
    main(args.src, args.dst, args.num, tuple(args.split), args.seed)
