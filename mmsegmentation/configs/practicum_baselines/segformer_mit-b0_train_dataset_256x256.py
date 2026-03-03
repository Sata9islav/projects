_base_ = [
    "../_base_/models/segformer_mit-b0.py",
    "../_base_/datasets/my_dataset.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/my_dataset_schedule.py",
]

input_size = (256, 256)
model = dict(
    data_preprocessor=dict(size=input_size),
    decode_head=dict(
        num_classes=3,
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            dict(type="DiceLoss", loss_weight=1.0),
        ],
    ),
    test_cfg=dict(mode="whole"),
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
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
                task_name="H1_segformer_mit-b0_ce+dice_aug",
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
