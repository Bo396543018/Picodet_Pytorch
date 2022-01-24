_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']

# model settings
model = dict(
    type='PicoDet',
    backbone=dict(
        type='ESNet',
        model_size='s',
        out_indices=[2, 9, 12],
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        act_cfg=dict(type='HSwish'),
        se_cfg=dict(conv_cfg=None, 
                    ratio=4,
                    act_cfg=(dict(type='ReLU'), dict(type='HSigmoid', bias=3.0, divisor=6.0, max_value=6.0))),
        init_cfg=dict(type='Pretrained', 
                      checkpoint='MODEL_DIR/ESNet_x0_75_pretrained_mmdet_format.pth')),
    neck=dict(
        type='CSPPAN',
        in_channels=[96, 192, 384],
        act_cfg=dict(type='HSwish'),
        norm_cfg=dict(type='BN', requires_grad=True),
        out_channels=96,
        num_features=4,
        expansion=1,
        num_csp_blocks=1),
    bbox_head=dict(
        type='PicoDetHead',
        num_classes=80,
        in_channels=96,
        feat_channels=96,
        stacked_convs=2,
        kernel_size=5,
        share_cls_reg=True,
        use_depthwise=True,
        reg_max=7,
        act_cfg=dict(type='HSwish'),
        strides=[8, 16, 32, 64],
        norm_cfg=dict(type='BN', requires_grad=True),
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='SimOTAAssigner', num_classes=80, 
                      use_vfl=True, iou_weight=6),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.025,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# dataset settings
data_root = 'data/coco/'
dataset_type = 'CocoDataset'

img_scales = [(256, 256), (288, 288), (320, 320), (352, 352), (384, 384)]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MinIoURandomCrop'),
    dict(type='Resize', img_scale=img_scales, multiscale_mode='value', keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scales[2],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'images/train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/val2017/',
        pipeline=test_pipeline))

evaluation = dict(interval=10, metric='bbox')

# optimizer
# default 4gpus
optimizer = dict(type='SGD', lr=0.1 * 4, momentum=0.9, weight_decay=0.00004)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=0.1)

runner = dict(type='EpochBasedRunner', max_epochs=300)

checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

resume_from = None
# yapf:enable
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CycleEMAHook', cycle_epoch=40, resume_from=resume_from, priority=49)
    ]