# SegMAN-Tiny + DataB 二分类 —— 与 ``segman_t_dataa_512_iou.py`` / sci TransNeXt DataB IoU 版同策略，仅数据根目录不同。

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/datab_binary_512.py',
]

load_from = 'pretrained/segman_t_ade.pth'

dist_params = dict(backend='gloo')

log_config = dict(
    interval=1,
    hooks=[dict(type='TextLoggerHook', by_epoch=True)])

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SegMANEncoder_t',
        pretrained=None,
        style='pytorch'),
    decode_head=dict(
        type='SegMANDecoder',
        in_channels=[32, 64, 144, 192],
        in_index=[0, 1, 2, 3],
        channels=128,
        feat_proj_dim=192,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
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
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

optimizer_config = dict()

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1e-6,
    power=0.9,
    min_lr=0.0,
    by_epoch=True)

runner = dict(type='EpochBasedRunner', max_epochs=200)

evaluation = dict(
    interval=1,
    metric=['mIoU', 'mFscore'],
    rule='greater',
    save_best='val/IoU',
    greater_keys=[
        'mIoU', 'aAcc', 'mDice', 'mFscore', 'mPrecision', 'mRecall',
        'IoU.foreground', 'IoU.background',
        'Fscore.foreground', 'Precision.foreground', 'Recall.foreground',
        'val/IoU', 'val/mIoU', 'val/F1', 'val/Precision', 'val/Recall',
    ],
    less_keys=['val/loss', 'loss'],
    pre_eval=True)

segman_use_binary_ckpt_layout = True
segman_binary_ckpt_root = 'checkpoints'
segman_enable_val_loss_best = False

segman_iou_early_stop_patience = 50
segman_console_summary_interval = 1

checkpoint_config = dict(by_epoch=True, interval=10**9, save_last=False)

data = dict(train=dict(times=1))
