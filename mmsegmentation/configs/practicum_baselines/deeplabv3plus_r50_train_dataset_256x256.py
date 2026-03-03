_base_ = [
    "../_base_/models/deeplabv3plus_r50-d8.py",
    "../_base_/datasets/my_dataset.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/my_dataset_schedule.py",
]

crop_size = (256, 256)
model = dict(
    data_preprocessor=dict(size=crop_size),
    decode_head=dict(
        num_classes=3,
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            dict(type="DiceLoss", loss_weight=1.0),
        ],
    ),
    auxiliary_head=dict(num_classes=3),
    test_cfg=dict(mode="whole"),
)


train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomResize", scale=(256, 256), ratio_range=(0.8, 1.2), keep_ratio=True
    ),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline), batch_size=8)

visualizer = dict(
    type="Visualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
        dict(
            type="ClearMLVisBackend",
            init_kwargs=dict(
                project_name="Practicum",
                task_name="H2_deeplabv3plus_r50_ce+dice_crop",
                reuse_last_task_id=False,
                continue_last_task=False,
                output_uri=None,
                auto_connect_arg_parser=True,
                auto_connect_frameworks=True,
                auto_resource_monitoring=True,
                auto_connect_streams=True,
            ),
        ),
    ],
)
