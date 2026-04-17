# SegMAN-Tiny + DataA 二分类 —— **200 epoch + 每 epoch 验证 + 与 TransNeXt 同策略**
# （对齐 sci/TransNeXt ``mask2former_transnext_tiny_dataa_512x512_iou.py``，仅骨干/解码器换为 SegMAN）
#
# - ``runner``: ``EpochBasedRunner``，``max_epochs=200``
# - ``evaluation.interval=1``；``save_best='val/IoU'``（``SegmanEvalHook`` 将前景 IoU 同步为
#   TransNeXt ``BinaryForegroundIoUMetric`` 主键；best 权重文件名形如 ``best_val/IoU_epoch*.pth``）
# - ``segman_iou_early_stop_patience=50``：连续 50 次验证未刷新 ``save_best`` 则早停
#   （``tools/train.py`` 注入 ``SegmanIoUPatienceEarlyStopHook``；与 val_loss 双轨互斥）
# - ``segman_console_summary_interval=1``：TransNeXt 风格彩色终端（loss/lr/验证表）；
#   设为 ``0`` 可关闭；也可用 ``mask2former_console_summary_interval``。
# - ``data.train.times=1``：每 epoch 扫一遍 RepeatDataset 内层，避免沿用 160k 的 times=50
# - 产出目录见文件头注释（``checkpoints/train_<时间>/``）
#
# 在 ``segmentation`` 目录执行:
#   python tools/train.py local_configs/segman_binary/segman_t_dataa_512_iou.py
# 或: ``bash scripts/train_dataa_segman_t_iou.sh``
#
# 初始化（对齐 TransNeXt 的 ``load_from``）：使用 ADE20K 上训好的 **整网**
# ``pretrained/segman_t_ade.pth``；``mmcv`` 加载为 ``strict=False``，二分类头最后一层与
# 150 类不兼容处会跳过、保持随机初始化。勿再让骨干单独读 ``SegMAN_Encoder_t.pth.tar``，避免重复加载。
#
# 双轨（IoU best + val loss best）请用 ``segman_t_dataa_512_val_loss.py``（该模式下不注入 IoU 早停）。

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/dataa_binary_512.py',
]

# 相对 **训练启动 cwd**（应在 ``segmentation/`` 下执行 ``tools/train.py``）
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
        # 对齐 TransNeXt DataA：更强 CE 权重 + 前景类加权，减轻「全预测背景」
        # （Mask2Former 侧为 loss_weight=2.0 + class_weight；此处为 2 类 softmax CE）
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
    # 与 TransNeXt ``save_best='val/IoU'`` 一致（由 ``segman_eval_hooks`` 注入同名标量）
    save_best='val/IoU',
    greater_keys=[
        'mIoU', 'aAcc', 'mDice', 'mFscore', 'mPrecision', 'mRecall',
        'IoU.foreground', 'IoU.background',
        'Fscore.foreground', 'Precision.foreground', 'Recall.foreground',
        'val/IoU', 'val/mIoU', 'val/F1', 'val/Precision', 'val/Recall',
    ],
    # 与 TransNeXt val/loss 早停一致（IoU 单轨时无 val/loss 键，不影响）
    less_keys=['val/loss', 'loss'],
    pre_eval=True)

segman_use_binary_ckpt_layout = True
segman_binary_ckpt_root = 'checkpoints'
segman_enable_val_loss_best = False

# 也可用 TransNeXt 同名键 ``mask2former_iou_early_stop_patience``（二选一）
segman_iou_early_stop_patience = 50
segman_console_summary_interval = 1

# 与 TransNeXt：不按轮次周期存盘，仅 best；不另存 last（``save_last=False``）
checkpoint_config = dict(by_epoch=True, interval=10**9, save_last=False)

data = dict(train=dict(times=1))
