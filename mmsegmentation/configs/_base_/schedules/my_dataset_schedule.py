optimizer = dict(type="AdamW", lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=optimizer,
    clip_grad=dict(max_norm=1.0),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=300,
    ),
    dict(
        type="PolyLR",
        eta_min=1e-6,
        power=1.0,
        begin=300,
        end=3000,
        by_epoch=False,
    ),
]


train_cfg = dict(type="IterBasedTrainLoop", max_iters=3000, val_interval=500)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")


default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=500,
        save_best="mDice",
        rule="greater",
        max_keep_ckpts=3,
        save_last=True,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook", interval=500, draw=True),
)