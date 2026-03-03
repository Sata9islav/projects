import cv2
import glob
import numpy as np
import os
from tqdm import tqdm

DATA_ROOT = "data/train_dataset"
SPLIT = "test"

IMG_DIR = os.path.join(DATA_ROOT, "img", SPLIT)
GT_DIR = os.path.join(DATA_ROOT, "labels", SPLIT)

# PRED_DIR = "work_dirs_eval/segformer_b0/raw"
# OUT_DIR = "work_dirs_eval/segformer_b0/quality_report"

PRED_DIR = "work_dirs_eval/deeplab_r50/raw"
OUT_DIR = "work_dirs_eval/deeplab_r50/quality_report"

# PRED_DIR = "work_dirs_eval/deeplab_101/raw"
# OUT_DIR = "work_dirs_eval/deeplab_101/quality_report"

N_SHOW = 10

NUM_CLASSES = 3
IGNORE_INDEX = 255

PALETTE = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
}

os.makedirs(OUT_DIR, exist_ok=True)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, color in PALETTE.items():
        out[mask == k] = color
    out[mask == IGNORE_INDEX] = (127, 127, 127)
    return out


def dice_per_class(pred, gt, num_classes=NUM_CLASSES):
    dices = []
    valid = gt != IGNORE_INDEX
    for c in range(num_classes):
        p = (pred == c) & valid
        g = (gt == c) & valid
        inter = (p & g).sum()
        denom = p.sum() + g.sum()
        if denom == 0:
            dices.append(np.nan)
        else:
            dices.append(2.0 * inter / denom)
    return np.array(dices, dtype=np.float32)


def mean_dice_ignore_nan(dices: np.ndarray):
    x = dices[~np.isnan(dices)]
    return float(x.mean()) if len(x) else float("nan")


def make_diff_map(pred, gt):
    valid = gt != IGNORE_INDEX
    gt_fg = (gt != 0) & valid
    pr_fg = (pred != 0) & valid

    fp = pr_fg & (~gt_fg)
    fn = (~pr_fg) & gt_fg

    diff = np.zeros((*gt.shape, 3), dtype=np.uint8)
    diff[fp] = (255, 0, 0)
    diff[fn] = (0, 255, 255)
    return diff


def find_pred_path(stem):
    cand = [
        os.path.join(PRED_DIR, stem + ".png"),
        os.path.join(PRED_DIR, stem + "_pred.png"),
        os.path.join(PRED_DIR, stem),
    ]
    for c in cand:
        if os.path.exists(c):
            return c

    hits = glob.glob(os.path.join(PRED_DIR, "**", stem + ".png"), recursive=True)
    if hits:
        return hits[0]
    return None


rows = []
img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))

for img_path in tqdm(img_paths, desc="Scoring"):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    gt_path = os.path.join(GT_DIR, stem + ".png")
    pred_path = find_pred_path(stem)
    if not os.path.exists(gt_path) or pred_path is None:
        continue

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    pr = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)

    if gt is None or pr is None or img is None:
        continue

    if pr.ndim == 3:
        pr = pr[:, :, 0]

    dices = dice_per_class(pr, gt)
    mdice = mean_dice_ignore_nan(dices)

    rows.append((mdice, stem, img, gt, pr, dices))

rows.sort(key=lambda x: x[0])

worst = rows[:N_SHOW]
best = rows[-N_SHOW:][::-1]


def save_examples(block, name):
    block_dir = os.path.join(OUT_DIR, name)
    os.makedirs(block_dir, exist_ok=True)

    for rank, (mdice, stem, img, gt, pr, dices) in enumerate(block, 1):
        gt_c = colorize_mask(gt)
        pr_c = colorize_mask(pr)
        diff = make_diff_map(pr, gt)

        title = (
            f"{rank:02d}_{stem}_mDice={mdice:.3f}_c1={dices[1]:.3f}_c2={dices[2]:.3f}"
        )
        canvas = np.concatenate([img, gt_c, pr_c, diff], axis=1)

        out_path = os.path.join(block_dir, title + ".png")
        cv2.imwrite(out_path, canvas)


save_examples(best, "best")
save_examples(worst, "worst")

csv_path = os.path.join(OUT_DIR, "summary.csv")
with open(csv_path, "w") as f:
    f.write("stem,mdice,dice_bg,dice_c1,dice_c2\n")
    for mdice, stem, *_rest, dices in [(r[0], r[1], r[5]) for r in rows]:
        f.write(f"{stem},{mdice:.6f},{dices[0]:.6f},{dices[1]:.6f},{dices[2]:.6f}\n")

print("Saved report to:", OUT_DIR)
print("Best examples:", os.path.join(OUT_DIR, "best"))
print("Worst examples:", os.path.join(OUT_DIR, "worst"))
print("CSV:", csv_path)
