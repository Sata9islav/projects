_base_ = [
    "/home/ubuntu/mmdetection/configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py",
]

load_from = "/home/ubuntu/mmdetection/mmdet/checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth"

num_classes = 17
model = dict(
    bbox_head=dict(num_classes=num_classes),
    test_cfg=dict(
        nms_pre=2000,
        score_thr=0.0,
        min_bbox_size=0,
        max_per_img=300,
        nms=dict(type="nms", iou_threshold=0.5),
    ),
)


metainfo = dict(
    classes=(
        "bee",
        "chicken",
        "cow",
        "creeper",
        "enderman",
        "fox",
        "frog",
        "ghast",
        "goat",
        "llama",
        "pig",
        "sheep",
        "skeleton",
        "spider",
        "turtle",
        "wolf",
        "zombie",
    )
)


visualizer = dict(
    vis_backends=[dict(type="LocalVisBackend", save_dir="artifacts/inference")],
    name="visualizer",
)


data_root = "/home/ubuntu/mmdetection/mmdet/datasets/minecraft/"
train_ann = "annotations/train.json"
val_ann = "annotations/valid.json"
test_ann = "annotations/test.json"

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        _delete_=True,
        type="ClassBalancedDataset",
        oversample_thr=0.001,
        dataset=dict(
            type="CocoDataset",
            data_root=data_root,
            ann_file=train_ann,
            data_prefix=dict(img="images/"),
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=0),
            pipeline=[
                dict(type="LoadImageFromFile"),
                dict(type="LoadAnnotations", with_bbox=True),
                dict(type="Resize", scale=(768, 768), keep_ratio=True),
                dict(type="RandomFlip", prob=0.5),
                dict(type="Pad", size_divisor=32),
                dict(type="PackDetInputs"),
            ],
        ),
    ),
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file=val_ann,
        data_prefix=dict(img="images/"),
        metainfo=metainfo,
        test_mode=True,
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="Resize", scale=(768, 768), keep_ratio=True),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(type="Pad", size_divisor=32),
            dict(type="PackDetInputs"),
        ],
    ),
)

test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file=test_ann,
        data_prefix=dict(img="images/"),
        metainfo=metainfo,
        test_mode=True,
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="Resize", scale=(768, 768), keep_ratio=True),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(type="Pad", size_divisor=32),
            dict(type="PackDetInputs"),
        ],
    ),
)


val_evaluator = dict(type="CocoMetric", ann_file=data_root + val_ann, metric="bbox")
test_evaluator = dict(type="CocoMetric", ann_file=data_root + test_ann, metric="bbox")

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=2e-4, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=10, norm_type=2),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type="MultiStepLR", by_epoch=True, milestones=[8, 11], gamma=0.1),
]

work_dir = "artifacts/fcos"

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1),
)
