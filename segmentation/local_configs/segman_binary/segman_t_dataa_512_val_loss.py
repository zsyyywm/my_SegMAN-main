# **IoU best + val loss best 双轨**（对齐 sci/TransNeXt ``mask2former_transnext_tiny_dataa_512x512_val_loss.py``）
#
# - ``segman_binary_ckpt_root='checkpoints2'``：与 TransNeXt ``work_dir=data/checkpoints2`` 同层级命名
# - 主目录：``segmentation/checkpoints2/train_<时间>/``（``save_best='val/IoU'`` 存 IoU 最优）
# - ``segmentation/checkpoints2/val_loss_best/<run_ts>/``：按更小 val loss 另存权重
# - ``mask2former_val_loss_early_stop_patience=50``：连续 50 次验证 val/loss 未刷新最优则早停（``tools/train.py`` 注入）
# - 与 TransNeXt 一致：本模式 **不** 注入 IoU patience 早停

_base_ = ['./segman_t_dataa_512_iou.py']

segman_binary_ckpt_root = 'checkpoints2'

segman_enable_val_loss_best = True
# mask2former_enable_val_loss_best = True

mask2former_val_loss_early_stop_patience = 50
