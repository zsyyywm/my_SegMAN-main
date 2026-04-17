# Copyright (c) OpenMMLab-style extension for SegMAN binary-seg experiments.
# 与 TransNeXt mask2former 的 IoU 早停一致：连续若干次验证未刷新最优则结束训练（EpochBasedRunner）。
import math

import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, EpochBasedRunner, Hook, get_dist_info


@HOOKS.register_module()
class SegmanIoUPatienceEarlyStopHook(Hook):
    """在 ``EpochBasedRunner`` 下，按验证指标做 patience 早停。

    须在 EvalHook（priority LOW）之后执行，默认 priority=75。
    """

    def __init__(self,
                 monitor='IoU.foreground',
                 patience=50,
                 rule='greater',
                 min_delta=0.0):
        self.monitor = str(monitor)
        self.patience = int(patience)
        self.rule = str(rule).lower()
        self.min_delta = float(min_delta)
        self._best = None
        self._epochs_no_improve = 0

    def after_train_epoch(self, runner):
        if not isinstance(runner, EpochBasedRunner):
            return
        if runner.max_epochs is None:
            return
        rank, world_size = get_dist_info()
        cur = None
        if rank == 0:
            cur = runner.log_buffer.output.get(self.monitor)
        if world_size > 1:
            t = torch.tensor(
                [float('nan')],
                dtype=torch.float64,
                device=torch.device('cpu'))
            if rank == 0 and cur is not None:
                try:
                    t[0] = float(cur)
                except (TypeError, ValueError):
                    t[0] = float('nan')
            dist.broadcast(t, src=0)
            cur = None if math.isnan(float(t[0])) else float(t[0])
        else:
            if cur is not None:
                try:
                    cur = float(cur)
                except (TypeError, ValueError):
                    cur = None
        if cur is None:
            return
        if self._best is None:
            self._best = cur
            self._epochs_no_improve = 0
            return
        if self.rule == 'less':
            improved = cur < (self._best - self.min_delta)
        else:
            improved = cur > (self._best + self.min_delta)
        if improved:
            self._best = cur
            self._epochs_no_improve = 0
        else:
            self._epochs_no_improve += 1
        if self._epochs_no_improve < self.patience:
            return
        if rank == 0:
            runner.logger.warning(
                f'SegmanIoUPatienceEarlyStopHook: 连续 {self.patience} 次验证 '
                f'「{self.monitor}」未优于当前最优 {self._best:.6f}，触发早停。')
        runner._max_epochs = runner.epoch + 1
