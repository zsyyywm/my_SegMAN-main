# Copyright (c) OpenMMLab. All rights reserved.
"""训练中途若 ``work_dir`` 被删除（误删、清理脚本等），mmcv ``TextLoggerHook`` 追加
``*.log.json`` 会 ``FileNotFoundError``。在写日志前每次都确保目录存在。"""
import mmcv
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SegmanEnsureWorkDirHook(Hook):
    """仅 rank0 执行；须以 ``priority='VERY_HIGH'`` 注册，先于 TextLoggerHook（VERY_LOW）写 json。"""

    def after_train_iter(self, runner):
        if getattr(runner, 'rank', 0) == 0:
            mmcv.mkdir_or_exist(runner.work_dir)

    def after_val_iter(self, runner):
        if getattr(runner, 'rank', 0) == 0:
            mmcv.mkdir_or_exist(runner.work_dir)
