from collections import Counter
import cv2
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import random
import re
from typing import Union
import matplotlib.pyplot as plt
import math


def draw_distribution(json_path: str, data_split_name: str) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_and_name = {category["id"]: category["name"] for category in coco["categories"]}
    counts = Counter(annotation["category_id"] for annotation in coco["annotations"])
    rows = [{"class": id_and_name[k], "count": v} for k, v in counts.items()]
    df = pd.DataFrame(rows).sort_values("count", ascending=False)

    plt.figure(figsize=(12, 5))
    plt.bar(df["class"], df["count"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title(f"{data_split_name.upper()} DATASET: Classes distribution")
    plt.tight_layout()
    plt.show()


def visualize_images_with_boxes(
    coco_json_path: str | Path,
    images_root: str | Path,
    thickness: int = 2,
):
    coco_json_path = Path(coco_json_path)
    images_root = Path(images_root)

    with coco_json_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    id_and_name = {category["id"]: category["name"] for category in coco["categories"]}
    img_by_id = {img["id"]: img for img in coco["images"]}

    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    candidate_ids = [
        im_id
        for im_id in img_by_id
        if im_id in anns_by_img and len(anns_by_img[im_id]) > 0
    ]
    if len(candidate_ids) < 4:
        candidate_ids = list(img_by_id.keys())

    chosen_ids = random.sample(candidate_ids, k=min(4, len(candidate_ids)))

    rendered = []
    titles = []

    for image_id in chosen_ids:
        im_meta = img_by_id[image_id]
        rel_path = Path(im_meta["file_name"])
        img_path = images_root / rel_path.name

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            rendered.append(None)
            titles.append(f"READ FAIL: {im_meta['file_name']}")
            continue

        anns = anns_by_img.get(image_id, [])

        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1 = int(round(x))
            y1 = int(round(y))
            x2 = int(round(x + w))
            y2 = int(round(y + h))

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), thickness)

            cls = id_and_name.get(ann["category_id"], str(ann["category_id"]))
            label = cls

            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            y_text = max(th + 4, y1)
            cv2.rectangle(
                img_bgr,
                (x1, y_text - th - 6),
                (x1 + tw + 6, y_text + baseline),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                img_bgr,
                label,
                (x1 + 3, y_text - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rendered.append(img_rgb)
        titles.append(im_meta["file_name"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for i in range(4):
        ax = axes[i]
        ax.axis("off")
        if i < len(rendered) and rendered[i] is not None:
            ax.imshow(rendered[i])
            ax.set_title(titles[i])
        elif i < len(titles):
            ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()


def draw_pricture(image_path: str) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(Image.open(image_path))
    plt.axis("off")
    plt.show()


def draw_fcos_metrics(path_to_log: str) -> None:
    log_path = Path(path_to_log)
    text = log_path.read_text(errors="ignore")
    patterns = re.compile(
        r"Epoch\(train\)\s+\[(\d+)\]\[\s*(\d+)/(\d+)\].*?"
        r"lr:\s*([0-9.eE+-]+).*?"
        r"grad_norm:\s*([0-9.eE+-]+)\s+"
        r"loss:\s*([0-9.eE+-]+)\s+"
        r"loss_cls:\s*([0-9.eE+-]+)\s+"
        r"loss_bbox:\s*([0-9.eE+-]+)\s+"
        r"loss_centerness:\s*([0-9.eE+-]+)"
    )

    rows = []
    for m in patterns.finditer(text):
        ep, it, it_total, lr, gn, l, lcls, lb, lc = m.groups()
        rows.append(
            dict(
                epoch=int(ep),
                iter=int(it),
                iter_total=int(it_total),
                lr=float(lr),
                grad_norm=float(gn),
                loss=float(l),
                loss_cls=float(lcls),
                loss_bbox=float(lb),
                loss_centerness=float(lc),
            )
        )

    df = pd.DataFrame(rows)
    df["global_step"] = (df["epoch"] - 1) * df["iter_total"] + df["iter"]
    df.to_csv("metrics.csv", index=False)

    plt.figure()
    plt.plot(df.global_step, df.loss)
    plt.title("loss")
    plt.show()
    plt.figure()
    plt.plot(df.global_step, df.lr)
    plt.title("lr")
    plt.show()


def _plot(title: str, cols: list[str], df: pd.DataFrame):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        print(f"[skip] нет колонок для: {title}")
        return
    plt.figure(figsize=(10, 5))
    for c in cols:
        plt.plot(df["epoch"], df[c], label=c)
    plt.title(title)
    plt.xlabel("epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def draw_yolo_metrics(results_csv_path: str | Path) -> None:
    results_csv_path = Path(results_csv_path)

    df = pd.read_csv(results_csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "epoch" not in df.columns:
        df.insert(0, "epoch", range(len(df)))

    _plot(
        "Losses",
        [
            "train/box_loss",
            "train/cls_loss",
            "train/dfl_loss",
            "val/box_loss",
            "val/cls_loss",
            "val/dfl_loss",
        ],
        df,
    )

    _plot(
        "Metrics (BBox)",
        [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
        ],
        df,
    )

    lr_cols = [c for c in df.columns if c.startswith("lr/")]
    _plot("Learning rate", lr_cols, df)


def show_n_images(
    images_dir_path: Union[str, Path], n: int = 5, title: str | None = None
):
    images_dir_path = Path(images_dir_path)

    img_paths = sorted(
        [p for p in images_dir_path.iterdir() if p.suffix.lower() == ".jpg"]
    )

    total = min(n, len(img_paths))
    rows = math.ceil(total / 5)

    for r in range(rows):
        chunk = img_paths[r * 5 : (r + 1) * 5]
        plt.figure(figsize=(15, 3))
        for i, p in enumerate(chunk, 1):
            plt.subplot(1, len(chunk), i)
            plt.imshow(Image.open(p))
            plt.axis("off")
            plt.title(p.name, fontsize=8)

        if title and r == 0:
            plt.suptitle(title)

        plt.tight_layout()
        plt.show()
