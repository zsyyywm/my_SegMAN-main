# 方案二（BEVANet 对齐）：**512×512** 训练；验证 / save_best / 早停 均基于 **前景 P>0.55** 单路解码
#（``SegmanWireScheme3EvalHook`` + ``single_gpu_test_wire_scheme3_t05``，``th=0.55``）。
# 与方案一（argmax）、方案三（256 + P>0.5）**互斥**；需在 ``segmentation/`` 下执行。
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/dataa_binary_512.py',
]

# 预训练：同 ``segman_t_dataa_512_iou.py``
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
        'mIoU', 'aAcc', 'mFscore', 'IoU.foreground', 'val/IoU',
        'val/mIoU', 'val/F1',
    ],
    pre_eval=True)

segman_use_binary_ckpt_layout = True
segman_binary_ckpt_root = 'data/checkpoints1'
segman_enable_val_loss_best = False
segman_wire_scheme2_512 = True
segman_wire_scheme3_256 = False
segman_wire_fg = 1
segman_wire_th = 0.55
segman_iou_early_stop_patience = 50
segman_console_summary_interval = 1

checkpoint_config = dict(by_epoch=True, interval=10**9, save_last=False)

# SegMANDecoder.forward_winssm 中 ``image_pool → BN``：空间被压成 1×1 且 N=GPU batch；
# samples_per_gpu=1 会令 BatchNorm.training 报错（Expected more than 1 value…），须 >=2。
data = dict(train=dict(times=1), samples_per_gpu=4, workers_per_gpu=4)

log_config = dict(
    interval=1, hooks=[dict(type='TextLoggerHook', by_epoch=True)])

dist_params = dict(backend='gloo')
optimizer_config = dict()
