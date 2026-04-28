log_config = dict(
    interval=1, hooks=[dict(type='TextLoggerHook', by_epoch=True)])
dist_params = dict(backend='gloo')
log_level = 'INFO'
load_from = 'pretrained/segman_t_ade.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
dataset_type = 'CustomDataset'
_data_root = '../../../DataA-B/DataA'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (256, 256)
_DATAA_SHORT_EDGES = [
    128, 153, 179, 204, 230, 256, 281, 307, 332, 358, 384, 409, 435, 460, 486,
    512
]
_train_img_scales = [
    (128, 128), (153, 153), (179, 179), (204, 204), (230, 230), (256, 256),
    (281, 281), (307, 307), (332, 332), (358, 358), (384, 384), (409, 409),
    (435, 435), (460, 460), (486, 486), (512, 512)
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='Resize',
        img_scale=[(128, 128), (153, 153), (179, 179), (204, 204), (230, 230),
                   (256, 256), (281, 281), (307, 307), (332, 332), (358, 358),
                   (384, 384), (409, 409), (435, 435), (460, 460), (486, 486),
                   (512, 512)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
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
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 256),
        flip=False,
        transforms=[
            dict(type='AlignedResize', keep_ratio=True, size_divisor=32),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
segman_val_sup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(1024, 256), keep_ratio=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='CustomDataset',
            data_root='../../../DataA-B/DataA',
            img_dir='image/train',
            ann_dir='mask/train',
            img_suffix='.jpg',
            seg_map_suffix='.png',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', reduce_zero_label=False),
                dict(
                    type='Resize',
                    img_scale=[(128, 128), (153, 153), (179, 179), (204, 204),
                               (230, 230), (256, 256), (281, 281), (307, 307),
                               (332, 332), (358, 358), (384, 384), (409, 409),
                               (435, 435), (460, 460), (486, 486), (512, 512)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_size=(256, 256),
                    cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ],
            classes=('background', 'foreground'),
            palette=[[0, 0, 0], [255, 255, 255]])),
    val=dict(
        type='CustomDataset',
        data_root='../../../DataA-B/DataA',
        img_dir='image/val',
        ann_dir='mask/val',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 256),
                flip=False,
                transforms=[
                    dict(
                        type='AlignedResize', keep_ratio=True,
                        size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('background', 'foreground'),
        palette=[[0, 0, 0], [255, 255, 255]]),
    test=dict(
        type='CustomDataset',
        data_root='../../../DataA-B/DataA',
        img_dir='image/test',
        ann_dir='mask/test',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 256),
                flip=False,
                transforms=[
                    dict(
                        type='AlignedResize', keep_ratio=True,
                        size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('background', 'foreground'),
        palette=[[0, 0, 0], [255, 255, 255]]))
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(type='SegMANEncoder_t', pretrained=None, style='pytorch'),
    decode_head=dict(
        type='SegMANDecoder',
        in_channels=[32, 64, 144, 192],
        in_index=[0, 1, 2, 3],
        channels=128,
        feat_proj_dim=192,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            class_weight=[1.0, 8.0],
            avg_non_ignore=True)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1e-06,
    power=0.9,
    min_lr=0.0,
    by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=200)
evaluation = dict(
    interval=1,
    metric=['mIoU', 'mFscore'],
    rule='greater',
    save_best='val/IoU',
    greater_keys=['mIoU', 'aAcc', 'mFscore', 'IoU.foreground', 'val/IoU'],
    pre_eval=True)
segman_use_binary_ckpt_layout = True
segman_enable_val_loss_best = False
segman_wire_scheme3_256 = True
segman_wire_fg = 1
segman_wire_th = 0.5
segman_iou_early_stop_patience = 50
segman_console_summary_interval = 1
checkpoint_config = dict(by_epoch=True, interval=1000000000, save_last=False)
optimizer_config = dict()
custom_hooks = [
    dict(
        type='SegmanIoUPatienceEarlyStopHook',
        monitor='val/IoU',
        patience=50,
        rule='greater',
        priority=75),
    dict(type='SegmanConsoleSummaryHook', interval=1, priority=78)
]
find_unused_parameters = False
segman_segmentation_root = '/root/autodl-tmp/my_MambaVision-main/my_SegMAN-main-main/my_SegMAN-main-main'
segman_run_ts = 'train_20260428_101215'
work_dir = '/root/autodl-tmp/my_MambaVision-main/my_SegMAN-main-main/my_SegMAN-main-main/data/checkpoints1/train_20260428_101215'
gpu_ids = [0]
auto_resume = False
