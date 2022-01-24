_base_ = 'picodet_s_320_coco.py'

model = dict(
    type='PicoDet',
    backbone=dict(
        model_size='m',
        init_cfg=dict(type='Pretrained', 
                      checkpoint='MODEL_DIR/ESNet_x1_0_pretrained_mmdet_format.pth')),
    neck=dict(
        in_channels=[128, 256, 512],
        out_channels=128,),
    bbox_head=dict(
        in_channels=128,
        feat_channels=128,
        stacked_convs=4,))

img_scales = [(352, 352), (384, 384), (416, 416), (448, 448), (480, 480)]

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
    samples_per_gpu=64,
    workers_per_gpu=8, 
    train=dict(
        pipeline=train_pipeline),
    val=dict(
        pipeline=test_pipeline),
    test=dict(
        pipeline=test_pipeline)) 