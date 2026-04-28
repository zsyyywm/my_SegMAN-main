# 方案三（BEVANet 对齐）：**256×256** 训练；验证 / save_best / 早停 基于 **前景 P>0.5**
#（``SegmanWireScheme3EvalHook``，``segman_wire_th=0.5``）。与方案一、方案二、双路同训互斥。
# 在 segmentation/ 下: python tools/train.py local_configs/segman_binary/segman_t_dataa_256_scheme3.py
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/dataa_binary_256.py',
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
    greater_keys=['mIoU', 'aAcc', 'mFscore', 'IoU.foreground', 'val/IoU'],
    pre_eval=True)

segman_use_binary_ckpt_layout = True
segman_enable_val_loss_best = False
segman_wire_scheme3_256 = True
segman_wire_fg = 1
segman_wire_th = 0.5
segman_iou_early_stop_patience = 50
segman_console_summary_interval = 1

checkpoint_config = dict(by_epoch=True, interval=10**9, save_last=False)

# 同方案二：``image_pool`` 分支下 BN 要求每卡 batch > 1。
data = dict(train=dict(times=1), samples_per_gpu=4, workers_per_gpu=4)

log_config = dict(
    interval=1, hooks=[dict(type='TextLoggerHook', by_epoch=True)])

dist_params = dict(backend='gloo')
optimizer_config = dict()
