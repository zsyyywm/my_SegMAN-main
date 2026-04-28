# DataB 二分类，256×256（方案3 可用）。请在 ``segmentation/`` 下启动训练。
# 其它路径：``export SEG_DATAB_ROOT=/绝对路径/DataB``。
import os
import os.path as osp

dataset_type = 'CustomDataset'
_data_root = os.environ.get(
    'SEG_DATAB_ROOT', osp.join('..', '..', '..', 'DataA-B', 'DataB'))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
crop_size = (256, 256)
_DATAB_SHORT_EDGES = [int(256 * x * 0.1) for x in range(5, 21)]
_train_img_scales = [(s, s) for s in _DATAB_SHORT_EDGES]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='Resize',
        img_scale=_train_img_scales,
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
val_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 256),
        flip=False,
        transforms=[
            dict(type='AlignedResize', keep_ratio=True, size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = val_test_pipeline
segman_val_sup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(1024, 256), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type=dataset_type,
            data_root=_data_root,
            img_dir='image/train',
            ann_dir='mask/train',
            img_suffix='.jpg',
            seg_map_suffix='.png',
            pipeline=train_pipeline,
            classes=('background', 'foreground'),
            palette=[[0, 0, 0], [255, 255, 255]])),
    val=dict(
        type=dataset_type,
        data_root=_data_root,
        img_dir='image/val',
        ann_dir='mask/val',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=val_test_pipeline,
        classes=('background', 'foreground'),
        palette=[[0, 0, 0], [255, 255, 255]]),
    test=dict(
        type=dataset_type,
        data_root=_data_root,
        img_dir='image/test',
        ann_dir='mask/test',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=val_test_pipeline,
        classes=('background', 'foreground'),
        palette=[[0, 0, 0], [255, 255, 255]]))
