import cv2
from mmdet.apis import init_detector, inference_detector
from mmengine.structures import InstanceData
from pathlib import Path
from tqdm import tqdm


def _to_label(lab, label_map):
    if isinstance(lab, int) and label_map is not None:
        if isinstance(label_map, dict):
            return str(label_map.get(lab, lab))
        if 0 <= lab < len(label_map):
            return str(label_map[lab])
    return str(lab)


def _draw_label_box(img, x1, y1, text, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    y_top = max(0, y1 - th - baseline - 6)
    x_left = max(0, x1)
    x_right = min(img.shape[1] - 1, x_left + tw + 6)
    y_bottom = min(img.shape[0] - 1, y_top + th + baseline + 6)

    cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom), color, -1)
    cv2.putText(
        img,
        text,
        (x_left + 3, y_bottom - baseline - 3),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


def process_video_generic(
    infer_on_frame_fn,
    input_video_path,
    output_video_path,
    label_map=None,
    confidence_threshold=0.25,
    draw=True,
    imgsz=512,
) -> None:
    input_video_path = str(input_video_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Can't open video: {input_video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, float(fps), (w, h))
    if not out.isOpened():
        raise RuntimeError(f"Can't create output video: {output_video_path}")

    pbar_total = total_frames if total_frames is not None else 0
    with tqdm(
        total=pbar_total,
        desc=f"Inference video -> {output_video_path.name}",
        disable=(total_frames is None),
    ) as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            boxes = infer_on_frame_fn(frame, imgsz=imgsz, conf=confidence_threshold)

            if draw:
                for b in boxes:
                    score = float(b.get("score", 0.0))
                    if score < confidence_threshold:
                        continue

                    x1, y1, x2, y2 = map(int, [b["x1"], b["y1"], b["x2"], b["y2"]])
                    lab = b.get("label", b.get("cls", "obj"))
                    lab = _to_label(lab, label_map)

                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    text = f"{lab} {score:.2f}"
                    _draw_label_box(frame, x1, y1, text, color=color)

            out.write(frame)

            if total_frames is not None:
                pbar.update(1)

    cap.release()
    out.release()
    print("VIDEO SAVED:", str(output_video_path))


def make_yolo_infer_fn(model_yolo):
    def _infer(frame, imgsz=512, conf=0.25):
        results = model_yolo.predict(
            source=frame, imgsz=imgsz, conf=conf, verbose=False
        )
        out = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0].item())
                cls = int(box.cls[0].item())
                out.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "score": score,
                        "label": model_yolo.names[cls],
                    }
                )
        return out

    return _infer


def make_fcos_infer_fn(
    config_path, checkpoint_path, device="cpu", score_thr=0.25, class_names=None
):
    model = init_detector(str(config_path), str(checkpoint_path), device=device)

    def _infer(frame, imgsz=None, conf=0.25):
        pred = inference_detector(model, frame)
        inst: InstanceData = pred.pred_instances

        if inst is None or len(inst) == 0:
            return []

        bboxes = inst.bboxes.detach().cpu().numpy()
        scores = inst.scores.detach().cpu().numpy()
        labels = inst.labels.detach().cpu().numpy()

        out = []
        for (x1, y1, x2, y2), s, lab in zip(bboxes, scores, labels):
            if float(s) < conf:
                continue
            name = int(lab)
            if class_names is not None:
                name = class_names[int(lab)]
            out.append(
                {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "score": float(s),
                    "label": name,
                }
            )
        return out

    return _infer
