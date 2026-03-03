import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm


def list_by_stem(folder: Path, suffix: str):
    return {p.stem: p for p in folder.rglob(f"*{suffix}") if p.is_file()}


def load_meta(meta_json: Path):
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    return (
        meta["global"]["suggested_num_classes"],
        meta["global"]["suggested_ignore_index"],
        meta["global"]["unique_labels"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/train_dataset")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--img_dir", type=str, default="img")
    ap.add_argument("--mask_dir", type=str, default="labels")
    ap.add_argument("--img_suffix", type=str, default=".jpg")
    ap.add_argument("--mask_suffix", type=str, default=".png")
    ap.add_argument(
        "--meta_json",
        type=str,
        default="practicum_work/supplementary/classes_inferred.json",
    )
    ap.add_argument(
        "--out_dir", type=str, default="practicum_work/supplementary/viz/stage1/eda"
    )
    args = ap.parse_args()

    root = Path(args.data_root)
    img_folder = root / args.img_dir / args.split
    mask_folder = root / args.mask_dir / args.split

    imgs = list_by_stem(img_folder, args.img_suffix)
    msks = list_by_stem(mask_folder, args.mask_suffix)
    common = sorted(set(imgs) & set(msks))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_json = Path(args.meta_json)
    num_classes, ignore_index, uniq_global = (None, None, None)
    if meta_json.exists():
        num_classes, ignore_index, uniq_global = load_meta(meta_json)

    if not num_classes:
        if uniq_global:
            candidates = [x for x in uniq_global if x != 255]
            num_classes = (max(candidates) + 1) if candidates else 1
        else:
            num_classes = 256

    pixel_counts = np.zeros(int(num_classes), dtype=np.int64)
    present_counts = np.zeros(int(num_classes), dtype=np.int64)
    widths, heights = [], []

    for stem in tqdm(common, desc=f"EDA ({args.split})"):
        with Image.open(imgs[stem]) as im:
            w, h = im.size
        widths.append(w)
        heights.append(h)

        with Image.open(msks[stem]) as im:
            mask = np.array(im)

        if mask.ndim != 2:
            continue

        if ignore_index is not None and ignore_index >= 0:
            valid = mask[mask != ignore_index]
        else:
            valid = mask

        if valid.size == 0:
            continue

        valid = np.clip(valid, 0, int(num_classes) - 1)
        cnt = np.bincount(valid.flatten(), minlength=int(num_classes))
        pixel_counts += cnt
        present_counts += (cnt > 0).astype(np.int64)

    widths = np.array(widths)
    heights = np.array(heights)

    plt.figure()
    plt.hist(widths, bins=30)
    plt.title("Image widths")
    plt.xlabel("width")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "image_widths_hist.png")
    plt.close()

    plt.figure()
    plt.hist(heights, bins=30)
    plt.title("Image heights")
    plt.xlabel("height")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "image_heights_hist.png")
    plt.close()

    plt.figure()
    plt.scatter(widths, heights, s=8)
    plt.title("Image sizes (W vs H)")
    plt.xlabel("width")
    plt.ylabel("height")
    plt.tight_layout()
    plt.savefig(out_dir / "image_sizes_scatter.png")
    plt.close()

    total = int(pixel_counts.sum())
    frac = pixel_counts / max(total, 1)

    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(int(num_classes)), frac)
    plt.title("Class pixel fraction")
    plt.xlabel("class_id")
    plt.ylabel("fraction")
    plt.tight_layout()
    plt.savefig(out_dir / "class_pixel_fraction.png")
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(int(num_classes)), present_counts)
    plt.title("Class presence (#images where class appears)")
    plt.xlabel("class_id")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "class_presence_count.png")
    plt.close()

    df = pd.DataFrame(
        {
            "class_id": np.arange(int(num_classes)),
            "pixel_count": pixel_counts,
            "pixel_frac": frac,
            "present_in_images": present_counts,
        }
    ).sort_values("pixel_frac", ascending=False)
    df.to_csv(out_dir / "class_stats.csv", index=False)

    print(f"[OK] out_dir: {out_dir}")
    print(f"[INFO] num_classes={num_classes}, ignore_index={ignore_index}")


if __name__ == "__main__":
    main()
