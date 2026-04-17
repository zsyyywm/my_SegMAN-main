# DataB：**IoU + val loss 双轨**（对齐 ``segman_t_dataa_512_val_loss.py`` / sci TransNeXt DataB val_loss 版）

_base_ = ['./segman_t_datab_512_iou.py']

segman_binary_ckpt_root = 'checkpoints2'

segman_enable_val_loss_best = True

mask2former_val_loss_early_stop_patience = 50
