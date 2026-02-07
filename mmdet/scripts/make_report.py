import numpy as np
from pathlib import Path


def box_iou_xyxy(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if b.ndim == 1:
        b = b[None, :]
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)

    union = area_a + area_b - inter + 1e-9
    return inter / union


def coco_ap_from_pr(rec, prec):
    rec = np.asarray(rec, dtype=np.float32)
    prec = np.asarray(prec, dtype=np.float32)

    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])

    recall_thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    for rt in recall_thresholds:
        p = prec[rec >= rt].max() if np.any(rec >= rt) else 0.0
        ap += p
    return ap / 101.0


def img_to_label_path(image_path: str | Path):
    p = Path(image_path)
    parts = list(p.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return p.with_suffix(".txt")


def load_yolo_gt_xyxy(image_path: str | Path, img_w: int, img_h: int):
    lab_path = img_to_label_path(image_path)
    if not lab_path.exists():
        return []

    lines = lab_path.read_text().strip().splitlines()
    gts = []
    for line in lines:
        if not line.strip():
            continue
        cls, x, y, w, h = line.strip().split()
        cls = int(float(cls))
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)

        cx = x * img_w
        cy = y * img_h
        bw = w * img_w
        bh = h * img_h

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        gts.append({"cls": cls, "xyxy": np.array([x1, y1, x2, y2], dtype=np.float32)})
    return gts


def yolo_results_to_preds(yolo_preds, conf_thr=0.25):
    preds_by_img = {}
    shapes_by_img = {}

    for r in yolo_preds:
        img_path = str(r.path)
        h, w = r.orig_shape  # (h,w)
        shapes_by_img[img_path] = (int(h), int(w))

        preds = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls):
                if float(s) < conf_thr:
                    continue
                preds.append(
                    {
                        "cls": int(c),
                        "score": float(s),
                        "xyxy": np.array([x1, y1, x2, y2], dtype=np.float32),
                    }
                )
        preds_by_img[img_path] = preds

    return preds_by_img, shapes_by_img


def fps_from_ultralytics_speed(yolo_preds):
    inf_ms = [r.speed.get("inference", None) for r in yolo_preds if hasattr(r, "speed")]
    inf_ms = [x for x in inf_ms if x is not None and x > 0]
    if not inf_ms:
        return None
    return 1000.0 / float(np.mean(inf_ms))


def fcos_outputs_to_preds(fcos_preds, conf_thr=0.25):
    preds_list = fcos_preds.get("predictions", fcos_preds)
    preds_by_img = {}

    for ds in preds_list:
        img_path = None
        if hasattr(ds, "metainfo"):
            mi = ds.metainfo
            img_path = mi.get("img_path", mi.get("ori_filename", None))
        if img_path is None and hasattr(ds, "img_path"):
            img_path = ds.img_path

        inst = getattr(ds, "pred_instances", None)
        if inst is None:
            preds_by_img[str(img_path)] = []
            continue

        bboxes = inst.bboxes.detach().cpu().numpy()
        scores = inst.scores.detach().cpu().numpy()
        labels = inst.labels.detach().cpu().numpy().astype(int)

        preds = []
        for box, s, c in zip(bboxes, scores, labels):
            if float(s) < conf_thr:
                continue
            preds.append(
                {"cls": int(c), "score": float(s), "xyxy": box.astype(np.float32)}
            )

        preds_by_img[str(img_path)] = preds

    return preds_by_img


def evaluate_detection_map_pr(
    preds_by_img,
    shapes_by_img,
    num_classes: int,
    conf_thr: float = 0.25,
    iou_thresholds=None,
    pr_iou_thr: float = 0.5,
):
    if iou_thresholds is None:
        iou_thresholds = np.round(np.arange(0.50, 0.96, 0.05), 2)

    gts_by_img = {}
    for img_path, (h, w) in shapes_by_img.items():
        gts_by_img[img_path] = load_yolo_gt_xyxy(img_path, img_w=w, img_h=h)

    ap_per_thr = {thr: [] for thr in iou_thresholds}

    for thr in iou_thresholds:
        for cls_id in range(num_classes):
            gt_count = 0
            gt_used = {}

            for img_path, gts in gts_by_img.items():
                cls_gts = [g for g in gts if g["cls"] == cls_id]
                gt_count += len(cls_gts)
                gt_used[img_path] = np.zeros(len(cls_gts), dtype=bool)

            dets = []
            for img_path, preds in preds_by_img.items():
                for p in preds:
                    if p["cls"] == cls_id and p["score"] >= conf_thr:
                        dets.append((img_path, p["score"], p["xyxy"]))
            dets.sort(key=lambda x: x[1], reverse=True)

            if gt_count == 0:
                continue

            tp = np.zeros(len(dets), dtype=np.float32)
            fp = np.zeros(len(dets), dtype=np.float32)

            for i, (img_path, score, box) in enumerate(dets):
                cls_gts = [g for g in gts_by_img[img_path] if g["cls"] == cls_id]
                if len(cls_gts) == 0:
                    fp[i] = 1.0
                    continue
                gt_boxes = np.stack([g["xyxy"] for g in cls_gts], axis=0)

                ious = box_iou_xyxy(box, gt_boxes)
                best_j = int(np.argmax(ious))
                best_iou = float(ious[best_j])

                if best_iou >= thr and not gt_used[img_path][best_j]:
                    tp[i] = 1.0
                    gt_used[img_path][best_j] = True
                else:
                    fp[i] = 1.0

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / (gt_count + 1e-9)
            prec = tp_cum / (tp_cum + fp_cum + 1e-9)

            ap = coco_ap_from_pr(rec, prec)
            ap_per_thr[thr].append(ap)

    map_50 = float(np.mean(ap_per_thr[0.50])) if ap_per_thr[0.50] else 0.0
    map_all = (
        float(np.mean([np.mean(v) for v in ap_per_thr.values() if len(v) > 0]))
        if any(len(v) > 0 for v in ap_per_thr.values())
        else 0.0
    )

    TP = FP = FN = 0
    for img_path, (h, w) in shapes_by_img.items():
        gts = gts_by_img[img_path]
        preds = [p for p in preds_by_img.get(img_path, []) if p["score"] >= conf_thr]
        preds.sort(key=lambda p: p["score"], reverse=True)

        gts_by_cls = {}
        for g in gts:
            gts_by_cls.setdefault(g["cls"], []).append(g["xyxy"])
        used_by_cls = {
            c: np.zeros(len(bxs), dtype=bool) for c, bxs in gts_by_cls.items()
        }

        for p in preds:
            c = p["cls"]
            if c not in gts_by_cls or len(gts_by_cls[c]) == 0:
                FP += 1
                continue
            gt_boxes = np.stack(gts_by_cls[c], axis=0)
            ious = box_iou_xyxy(p["xyxy"], gt_boxes)
            j = int(np.argmax(ious))
            if float(ious[j]) >= pr_iou_thr and not used_by_cls[c][j]:
                TP += 1
                used_by_cls[c][j] = True
            else:
                FP += 1

        for c, mask in used_by_cls.items():
            FN += int((~mask).sum())

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "mAP": map_all,
        "mAP_50": map_50,
        "Precision": float(precision),
        "Recall": float(recall),
        "F1-score": float(f1),
    }
