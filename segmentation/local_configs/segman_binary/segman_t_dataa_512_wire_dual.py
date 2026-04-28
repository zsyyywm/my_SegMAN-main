# SegMAN-Tiny + DataA 512：方案1/2 同训（单卡；``SegmanWireDualEvalHook``）
# 方案1=argmax；方案2=前景 P>0.55；``save_best``/早停仅盯 ``val/IoU``（方案1）。
# 在 segmentation/ 下: python tools/train.py local_configs/segman_binary/segman_t_dataa_512_wire_dual.py
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/dataa_binary_512.py',
]

# 预训练：同 ``segman_t_dataa_512_iou.py``（ADE20K SegMAN-T 整网）— Google Drive:
# https://drive.google.com/file/d/1d0wp7C83YjImeQmL5_CIo1qMRjPd_-8a/view?usp=sharing
load_from = 'pretrained/segman_t_ade.pth'

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
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

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
        'IoU.foreground', 'IoU.background', 'Fscore.foreground',
        'val/IoU', 'val_t055/IoU', 'val_t055/mIoU', 'val/mIoU', 'val/F1',
    ],
    less_keys=['val/loss', 'loss'],
    pre_eval=True)

segman_use_binary_ckpt_layout = True
segman_enable_val_loss_best = False
segman_wire_dual_512 = True
segman_wire_subdir1 = 'scheme1_argmax512'
segman_wire_subdir2 = 'scheme2_t055_512'
segman_wire_fg = 1
segman_wire_th = 0.55
segman_iou_early_stop_patience = 50
segman_console_summary_interval = 1

checkpoint_config = dict(by_epoch=True, interval=10**9, save_last=False)

# 同方案二/三：须 samples_per_gpu >= 2（见 segman_t_dataa_512_scheme2.py 注释）。
data = dict(train=dict(times=1), samples_per_gpu=4, workers_per_gpu=4)

log_config = dict(
    interval=1, hooks=[dict(type='TextLoggerHook', by_epoch=True)])

dist_params = dict(backend='gloo')
optimizer_config = dict()
