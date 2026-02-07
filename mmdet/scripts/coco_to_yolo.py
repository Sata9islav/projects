import json
from pathlib import Path
from typing import Union


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def coco_to_yolo(coco_json: Union[Path | str], out_labels_dir: Union[Path | str]):
    out_labels_dir = Path(out_labels_dir)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    coco_json = Path(coco_json)
    coco = json.loads(coco_json.read_text(encoding="utf-8"))
    images = {img["id"]: img for img in coco["images"]}

    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    for image_id, img in images.items():
        w = float(img["width"])
        h = float(img["height"])
        anns = anns_by_img.get(image_id, [])

        img_name = Path(img["file_name"]).name
        txt_path = out_labels_dir / (Path(img_name).stem + ".txt")

        lines = []
        for ann in anns:
            x, y, bw, bh = map(float, ann["bbox"])
            if bw <= 0 or bh <= 0:
                continue

            cls = int(ann["category_id"]) - 1
            if cls < 0:
                continue

            xc = (x + bw / 2.0) / w
            yc = (y + bh / 2.0) / h
            bw_n = bw / w
            bh_n = bh / h

            xc = _clip(xc)
            yc = _clip(yc)
            bw_n = _clip(bw_n)
            bh_n = _clip(bh_n)

            lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw_n:.6f} {bh_n:.6f}")

        txt_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
