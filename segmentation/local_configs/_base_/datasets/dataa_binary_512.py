# DataA 二分类。
# 默认 ``../../../DataA-B/DataA`` 相对 **训练启动时的 cwd**，请在 ``segmentation/`` 下执行
# ``python tools/train.py ...``，此时即 ``<mamba-main>/DataA-B/DataA``（与两层 ``SegMAN-main`` 同级）。
# 数据在其它盘或路径时：``export SEG_DATAA_ROOT=/绝对路径/DataA``（须含 ``image/train``、``mask/train`` 等）。
import os
import os.path as osp

dataset_type = 'CustomDataset'
_data_root = os.environ.get(
    'SEG_DATAA_ROOT', osp.join('..', '..', '..', 'DataA-B', 'DataA'))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
crop_size = (512, 512)
# 与 sci/TransNeXt ``RandomChoiceResize`` + ``ResizeShortestEdge`` 离散尺度同公式（mmseg 用正方形多尺度近似）
_DATAA_SHORT_EDGES = [int(512 * x * 0.1) for x in range(5, 21)]
_train_img_scales = [(s, s) for s in _DATAA_SHORT_EDGES]
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
# 验证集：mmseg ``CustomDataset`` 在 ``test_mode=True`` 时不带 ``ann_info``，GT 由 ``pre_eval`` 路径按索引加载；
# 与 TransNeXt 输入尺寸一致：``img_scale=(2048,512)`` + ``keep_ratio``（经 ``AlignedResize``）。
val_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
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
# 带 GT，仅供 SegmanEvalHook 计算 val loss（与 mIoU 验证 pipeline 分离）
segman_val_sup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 512), keep_ratio=True),
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
