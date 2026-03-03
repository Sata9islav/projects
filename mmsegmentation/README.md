
# Selecting and training a model from mmsegmentation for the task of multi-class semantic segmentation

## Task description

In this project, you need to select a semantic segmentation model from mmsegementation and then train it to achieve the specified metrics on the test dataset.

## Goal

The target metric is mDice score. The project is considered completed when mDIce > 0.75

## Work

### 1.EDA

1. Identify classes/ignore + check pairs

```
python practicum_work/src/data/scan_and_infer_classes.py \
  --data_root data/train_dataset
```

2. Quality check train

```
python practicum_work/src/data/quality_check.py \
  --data_root data/train_dataset --split train
```

3. Overlays for bad stems

```
python practicum_work/src/data/dump_overlays.py \
  --data_root data/train_dataset --split train \
  --stems_file practicum_work/supplementary/bad_stems.txt --take 30
```

4. EDA for train

```
python practicum_work/src/analysis/eda.py \
  --data_root data/train_dataset --split train
```

#### Report

###### Data quality analysis:

- The dataset is organized in a format compatible with BaseSegDataset: img/{split} and labels/{split}; for each image there is a corresponding mask with the same stem.
- Only values of classes 0, 1, and 2 are found in the markup; ignore_index was not detected. Number of classes: 3. ￼
- Based on the results of automatic checks (image/mask size matching, correctness of the id range, absence of broken files), no critical problems requiring deletion/additional markup have been identified.
- Cleaning strategy: no cleaning is required at this stage; the focus is on compensating for class imbalances during the learning phase.

###### EDA

- Image sizes: all images are the same size 256×256, which simplifies the pipeline.
  ![image heights hist](/Users/satanislav/Documents/projects/mmsegmentation/practicum_work/supplementary/viz/stage1/eda/image_heights_hist.png)
  ![image widths hist](/Users/satanislav/Documents/projects/mmsegmentation/practicum_work/supplementary/viz/stage1/eda/image_widths_hist.png)
  ![image size scatter](/Users/satanislav/Documents/projects/mmsegmentation/practicum_work/supplementary/viz/stage1/eda/image_sizes_scatter.png)
- Class balance:

  - there is a strong imbalance in pixels: the background (class 0) dominates (~90%), while classes 1 and 2 take up a small share (~5% and ~4% according to your graphs);
    ![class pixel count](/Users/satanislav/Documents/projects/mmsegmentation/practicum_work/supplementary/viz/stage1/eda/class_pixel_fraction.png)
  - by presence in images, classes 1 and 2 occur in about half of train-samples (by your graph - about 100 images each), i.e. the problem is more “in the area of objects” than “in the rarity of images”.
    ![class presence count](/Users/satanislav/Documents/projects/mmsegmentation/practicum_work/supplementary/viz/stage1/eda/class_presence_count.png)
- basic CrossEntropy can “stick” to the background → it is reasonable to try DiceLoss / CE+Dice and/or class weights right away;
- due to the small proportion of objects, augmentations and/or crops are useful to see the object more often in patches.

### 2.Primary hepotisis

#### Hypothesis 1: SegFormer (MiT-B0) + CE + Dice + лёгкие аугментации

Why:

- SegFormer often gives a very strong baseline on multiclass segmentation “out of the box”
- tolerates imbalance well, especially with Dice

Config: configs/practicum_baselines/segformer_mit-b0_train_dataset_256x256.py
ClearML: https://app.clear.ml/projects/61fbf6c04fee4cf4ab525a5423281757/experiments/5079d78742e747089092765a18d94926/output/execution

```
python tools/train.py configs/practicum_baselines/segformer_mit-b0_train_dataset_256x256.py
```

##### Quality analysis

```
python tools/test.py \
  configs/practicum_baselines/segformer_mit-b0_train_dataset_256x256.py \
  work_dirs/segformer_mit-b0_train_dataset_256x256/best_mDice_iter_16000.pth \
  --work-dir work_dirs_eval/segformer_b0 \
  --out work_dirs_eval/segformer_b0/raw
```

```
python practicum_work/src/analysis/analyze_seg.py 
```

Metrics on test (120 images):
 • aAcc = 90.92, mDice = 56.33, mAcc = 51.99
 • Dice by class: background 95.69, class_1 34.09, class_2 39.20

Observations:
 • The model confidently segments the background, but has a lower performance on the target classes.
 • High overall accuracy with low mDice indicates pixel imbalance (background is dominant).

Examples of correct predictions:
![best example 1](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/segformer_b0/quality_report/best/01_000000414495_3471_mDice=0.929_c1=0.875_c2=nan.png)
![best example 2](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/segformer_b0/quality_report/best/02_000000272153_1436_mDice=0.929_c1=nan_c2=0.869.png)
![best example 3](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/segformer_b0/quality_report/best/03_000000497010_7263_mDice=0.915_c1=0.843_c2=nan.png)
![best example 4](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/segformer_b0/quality_report/best/04_000000502680_4731_mDice=0.915_c1=nan_c2=0.851.png)
![best example 5](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/segformer_b0/quality_report/best/05_000000284148_7566_mDice=0.911_c1=0.844_c2=nan.png)

Examples of errors (fails):
![worst example 1](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/segformer_b0/quality_report/worst/01_000000364566_4776_mDice=0.295_c1=0.000_c2=0.000.png)
![worst example 2](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/segformer_b0/quality_report/worst/02_000000287567_4976_mDice=0.301_c1=0.000_c2=0.019.png)
![worst example 3](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/segformer_b0/quality_report/worst/03_000000366787_5428_mDice=0.305_c1=0.000_c2=0.000.png)
![worst example 4](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/segformer_b0/quality_report/worst/04_000000485148_1904_mDice=0.309_c1=0.001_c2=0.000.png)
![worst example 5](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/segformer_b0/quality_report/worst/05_000000380117_1580_mDice=0.310_c1=0.003_c2=0.000.png)

Hypotheses of the reasons:
 • class imbalance / small objects / similar textures between class_1 and class_2
 • augmentations are not “varied” enough for rare classes
 • crop/resize could reduce informative details

#### Hypothesis 2: DeepLabV3+ (R50) + CE + Dice + crop (focus on objects)

Why:

- CNN-baseline, often stable
- add RandomCrop with category control to see the object more often (important when the background is dominant)

Config: configs/practicum_baselines/deeplabv3plus_r50_train_dataset_256x256.py
ClearML: https://app.clear.ml/projects/61fbf6c04fee4cf4ab525a5423281757/experiments/58dc071d7ea94103a89492de43fd939a/output/execution

```
python tools/train.py configs/practicum_baselines/deeplabv3plus_r50_train_dataset_256x256.py
```

##### Quality analysis

```
python tools/test.py \
  configs/practicum_baselines/deeplabv3plus_r50_train_dataset_256x256.py \
  work_dirs/deeplabv3plus_r50_train_dataset_256x256/best_mDice_iter_2000.pth \
  --work-dir work_dirs_eval/deeplab_r50 \
  --out work_dirs_eval/segformer_b0/raw
```

```
python practicum_work/src/analysis/analyze_seg.py 
```

Metrics on test (120 images):

- aAcc = 97.95 mDice = 92.04 mAcc = 91.30
- Dice by classes: background 98.98, class_1 90.45, class_2 86.68

Observations:
The model demonstrates high and balanced quality on both target classes (class_1 and class_2), while the background is segmented with almost no errors.
The most challenging remains class_2 (Dice lower than class_1), which may be due to fewer examples/complex object geometry or heterogeneous labeling.

Examples of correct predictions:

- the contours of class_1 and class_2 objects match the markup well;
- the model correctly separates the background from the objects;
- there is no “flowing” of classes into each other.

![best example 1](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r50/quality_report/best/01_000000446604_4215_mDice=0.984_c1=0.973_c2=nan.png)
![best example 2](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r50/quality_report/best/02_000000414495_3471_mDice=0.983_c1=0.971_c2=nan.png)
![best example 3](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r50/quality_report/best/03_000000395290_2371_mDice=0.983_c1=0.971_c2=nan.png)
![best example 4](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r50/quality_report/best/04_000000322321_6994_mDice=0.982_c1=0.967_c2=nan.png)
![best example 5](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r50/quality_report/best/05_000000563342_7436_mDice=0.981_c1=0.966_c2=nan.png)

Examples of errors (fails):

- Errors on object boundaries
- “ragged” edges of the mask, especially on thin/small objects
- small contour displacements (boundary mismatch)
- Confusion between class_1 and class_2
- parts of the object belong to an adjacent class
- different classes are predicted in “spots” within the same area
- Missing small objects
- the model does not see small segments or thin details
- False positives in the background
- small islands of class_1/class_2 in the background (FP), often in textured areas

![worst example 1](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r50/quality_report/worst/01_000000531614_4831_mDice=0.478_c1=0.434_c2=0.000.png)
![worst example 2](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r50/quality_report/worst/02_000000436539_4321_mDice=0.525_c1=0.000_c2=0.591.png)
![worst example 3](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r50/quality_report/worst/03_000000443499_491_mDice=0.600_c1=0.807_c2=0.000.png)
![worst example 4](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r50/quality_report/worst/04_000000355603_7766_mDice=0.624_c1=0.000_c2=0.879.png)
![worst example 5](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r50/quality_report/worst/05_000000282137_4359_mDice=0.637_c1=0.927_c2=0.000.png)

#### Report: Model Comparison

DeepLabV3+ R50 significantly outperforms SegFormer-B0 in terms of target class segmentation quality: model #1 performs well on background, but significantly worse on class_1 and class_2.
Model #2 shows consistent quality across all classes and is suitable as the main model.

### Quality improvement experiments

#### Easy post-processing

**Description of the experiment**

1. Take the best checkpoint of the second model (best_mDice_iter_2000.pth).
2. Run inference on test and save the raw predictions.
3. Perform a simple post-processing.

```
python tools/test.py \
  configs/practicum_baselines/deeplabv3plus_r50_train_dataset_256x256.py \
  work_dirs/deeplabv3plus_r50_train_dataset_256x256/best_mDice_iter_2000.pth \
  --work-dir work_dirs_stage3/deeplab_r50 \
  --out work_dirs_stage3/deeplab_r50/preds.pkl
```

```
python practicum_work/src/data/postprocess_and_eval.py \
  --pred_dir work_dirs_stage3/deeplab_r50/preds.pkl \
  --gt_dir data/train_dataset/labels/test \
  --out_dir work_dirs_stage3/deeplab_r50/preds_pp \
  --num_classes 3 \
  --min_area 0,120,0 \
  --close_k 0 \
  --open_k 0
```

**Quality analysis**

```
=== RAW (до постобработки, dataset-level) ===
Dice per class: 98.98% | 90.45% | 86.68%
IoU  per class: 97.98% | 82.57% | 76.50%
mDice: 92.04%
mIoU : 85.68%

=== POSTPROCESSED (после постобработки, dataset-level) ===
Dice per class: 98.98% | 90.43% | 86.68%
IoU  per class: 97.98% | 82.54% | 76.50%
mDice: 92.03%
mIoU : 85.67%

=== DELTA (pp - raw) ===
ΔDice per class: -0.00% | -0.02% | +0.00%
ΔIoU  per class: -0.00% | -0.03% | +0.00%
ΔmDice: -0.01%
ΔmIoU : -0.01%

=== CLASS PIXEL SHARE (valid pixels) ===
GT  : 89.72% | 5.81% | 4.47%
RAW : 90.06% | 5.36% | 4.58%
PP  : 90.06% | 5.36% | 4.58%
```

##### Report

For this DeeplabV3+ R50 model, the predictions are already quite “clean”, so simple post-processing heuristics in their current form do not provide a benefit and may slightly “eat up” useful small segments.
To improve, it is worth trying more meaningful techniques (TTA/multi-scale, boundary-aware loss, training on larger crops, targeted augmentation, or post-processing based on classes/context).

#### Experiment 2 - Architecture Complexification (DeepLabV3+ ResNet-101)

**Experiment description**

The goal is to check whether a more powerful backbone (ResNet-101 instead of ResNet-50) will give a quality gain due to richer features. We change **only the architecture**, leaving the rest of the training and augmentation parameters as in the baseline.

**Changes (illustration)**

In the config, change:

```
_base_ = [
    "../_base_/models/deeplabv3plus_r50-d8.py",
    ...
]

model = dict(
    backbone=dict(depth=101),
    pretrained="open-mmlab://resnet101_v1c",
)
work_dir = "./work_dirs/deeplabv3plus_r101_train_dataset_256x256"

visualizer = dict(
    vis_backends=[
        ...
        dict(
            type="ClearMLVisBackend",
            init_kwargs=dict(
                project_name="Practicum",
                task_name="H2_deeplabv3plus_r101_ce+dice_crop",
            ),
        ),
    ]
)
```

```
python tools/train.py configs/practicum_baselines/deeplabv3plus_r101_train_dataset_256x256.py
```

```
python tools/test.py \
  configs/practicum_baselines/deeplabv3plus_r50_train_dataset_256x256.py \
  work_dirs/deeplabv3plus_r50_train_dataset_256x256/best_mDice_iter_2000.pth \
  --work-dir work_dirs_eval/deeplab_r50 \
  --out work_dirs_eval/deeplab_101/raw
```

```
python practicum_work/src/analysis/analyze_seg.py 
```

Learning outcomes:

- Config: configs/practicum_baselines/deeplabv3plus_r101_train_dataset_256x256.py
- ClearML: https://app.clear.ml/projects/61fbf6c04fee4cf4ab525a5423281757/experiments/899ecb37cce547e4aa7b336193801dd7/output/execution

**Quality analysis**
Test (120 images): aAcc: 97.97, mDice: 91.75, mAcc: 91.40
Dice by classes: background: 99.06, class_1: 90.05, class_2: 86.14

**Comparison with the best model (DeepLabV3+ R50)**

Difference (R101 − R50):
 • mDice: 91.75 − 92.04 = −0.29
 • aAcc: 97.97 − 97.95 = +0.02
 • mAcc: 91.40 − 91.30 = +0.10
 • background Dice: +0.08
 • class_1 Dice: −0.40
 • class_2 Dice: −0.54

Examples of correct predictions:

![best example 1](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r101/quality_report/best/01_000000556500_5818_mDice=0.989_c1=0.983_c2=nan.png)
![best example 2](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r101/quality_report/best/02_000000543836_507_mDice=0.989_c1=0.983_c2=nan.png)
![best example 3](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r101/quality_report/best/03_000000406211_2388_mDice=0.988_c1=0.980_c2=nan.png)
![best example 4](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r101/quality_report/best/04_000000284148_7566_mDice=0.986_c1=0.975_c2=nan.png)
![best example 5](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r101/quality_report/best/05_000000415604_7522_mDice=0.984_c1=0.973_c2=nan.png)

Examples of errors (fails):

![worst example 1](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r101/quality_report/worst/01_000000284884_6459_mDice=0.332_c1=0.000_c2=0.000.png)
![worst example 2](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r101/quality_report/worst/02_000000445187_3686_mDice=0.342_c1=0.041_c2=0.000.png)
![worst example 3](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r101/quality_report/worst/03_000000308083_5809_mDice=0.359_c1=0.000_c2=0.087.png)
![worst example 4](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r101/quality_report/worst/04_000000502680_4731_mDice=0.418_c1=0.000_c2=0.262.png)
![worst example 5](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r101/quality_report/worst/05_000000315467_1972_mDice=0.459_c1=0.383_c2=0.000.png)

##### Report

The complexity of the architecture (R50 → R101) did not result in an increase in the target metric mDice, but rather led to a slight decrease. However, the background quality improved slightly, but the classes 1/2 experienced a decline.

### Conclusion and selection of the best experiment

During the experiments, two main segmentation architectures and one option for complicating the architecture were compared:

- SegFormer MIT-B0 (CE + Dice, augmentation)
  Result on test subset: mDice = 56.33 (much weaker in 1/2 classes).
- DeepLabV3+ ResNet50 (CE + Dice, crop + resize, standard augmentation)
  Result on test subset: mDice = 92.04.
- DeepLabV3+ ResNet101 (experiment with architecture complication)
  The result on the test subset: mDice = 91.75 (slightly worse than the base R50).

Choosing the best experiment:

The best turned out to be DeepLabV3+ with backbone ResNet50, because it gave the maximum mDice among all tested variants (92.04) and at the same time provides a balanced quality on both target classes (class_1 and class_2). An attempt to complicate the architecture to ResNet101 did not give an increase in the target metric (−0.29 mDice), and SegFormer B0 lost significantly in quality on classes.

mDice (test subset) = 92.04

Examples of correct predictions:

![best example 1](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r50/quality_report/best/01_000000446604_4215_mDice=0.984_c1=0.973_c2=nan.png)
![best example 2](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r50/quality_report/best/02_000000414495_3471_mDice=0.983_c1=0.971_c2=nan.png)
![best example 3](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r50/quality_report/best/03_000000395290_2371_mDice=0.983_c1=0.971_c2=nan.png)
![best example 4](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r50/quality_report/best/04_000000322321_6994_mDice=0.982_c1=0.967_c2=nan.png)
![best example 5](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r50/quality_report/best/05_000000563342_7436_mDice=0.981_c1=0.966_c2=nan.png)

Examples of errors (fails):

![worst example 1](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r50/quality_report/worst/01_000000531614_4831_mDice=0.478_c1=0.434_c2=0.000.png)
![worst example 2](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r50/quality_report/worst/02_000000436539_4321_mDice=0.525_c1=0.000_c2=0.591.png)
![worst example 3](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r50/quality_report/worst/03_000000443499_491_mDice=0.600_c1=0.807_c2=0.000.png)
![worst example 4](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r50/quality_report/worst/04_000000355603_7766_mDice=0.624_c1=0.000_c2=0.879.png)
![worst example 5](/Users/satanislav/Documents/projects/mmsegmentation/work_dirs_eval/deeplab_r50/quality_report/worst/05_000000282137_4359_mDice=0.637_c1=0.927_c2=0.000.png)

Opportunities for improvement:

From the results of metrics it is seen that the base model is already strong (mDice ~92), and simple morphological post-processing did not give a noticeable increase. Potential directions of improvements:

1. Working with errors on the borders

- add boundary-aware loss (for example, BoundaryLoss/LevelSet, or strengthen the role of Dice/IoU on the borders),
- use more “fine” augmentations for the edges (RandomRotate, RandomAffine, Elastic, but carefully).

2. Strengthening small objects and class imbalance

- increase sampling of “complex” patches (crop around objects of classes 1/2),
- try class-weighted CE or Focal Loss (if there are many misses/false positives for small classes).

3. Changing the scale and context

- multi-scale training/tta (multiple scales on inference),
- increase crop_size (if VRAM allows) or add RandomResize of a wider range.

4. Architectural improvements without a significant increase in cost

- try the backbone “middle” (ResNet50 → ResNet101
