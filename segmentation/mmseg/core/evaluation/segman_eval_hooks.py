# Copyright (c) OpenMMLab-style extension for SegMAN binary-seg experiments.
# Dual best: existing EvalHook save_best (e.g. IoU.foreground) under work_dir;
# optional minimum val loss checkpoint under <seg_root>/<segman_binary_ckpt_root>/val_loss_best/<run_ts>/.
import os.path as osp
import warnings
from copy import deepcopy

import mmcv
import torch
from mmcv.parallel import scatter
from mmcv.runner.checkpoint import save_checkpoint as mmcv_save_checkpoint
from torch.nn.parallel import DataParallel

from mmseg.core.evaluation.eval_hooks import DistEvalHook, EvalHook


def _inject_transnext_style_val_aliases(eval_res):
    """与 sci/TransNeXt ``BinaryForegroundIoUMetric`` + ``prefix='val'`` 的键名对齐（数值与 mmseg 一致，0–1）。"""
    if not isinstance(eval_res, dict):
        return
    if 'IoU.foreground' in eval_res:
        eval_res['val/IoU'] = eval_res['IoU.foreground']
    if 'Fscore.foreground' in eval_res:
        eval_res['val/F1'] = eval_res['Fscore.foreground']
    if 'Precision.foreground' in eval_res:
        eval_res['val/Precision'] = eval_res['Precision.foreground']
    if 'Recall.foreground' in eval_res:
        eval_res['val/Recall'] = eval_res['Recall.foreground']
    if 'mIoU' in eval_res:
        eval_res['val/mIoU'] = eval_res['mIoU']


def _evaluate_with_val_aliases(eval_hook_self, runner, results):
    """等价 mmcv EvalHook.evaluate，并在写入 log_buffer 前注入 ``val/*`` 别名。"""
    eval_res = eval_hook_self.dataloader.dataset.evaluate(
        results, logger=runner.logger, **eval_hook_self.eval_kwargs)
    _inject_transnext_style_val_aliases(eval_res)
    for name, val in eval_res.items():
        runner.log_buffer.output[name] = val
    runner.log_buffer.ready = True
    if eval_hook_self.save_best is not None:
        if not eval_res:
            warnings.warn(
                'Since `eval_res` is an empty dict, the behavior to save '
                'the best checkpoint will be skipped in this evaluation.',
                UserWarning)
            return None
        if eval_hook_self.key_indicator == 'auto':
            eval_hook_self._init_rule(eval_hook_self.rule,
                                      list(eval_res.keys())[0])
        ki = eval_hook_self.key_indicator
        if ki not in eval_res:
            warnings.warn(
                f'save_best key {ki!r} not in eval results; keys: '
                f'{list(eval_res.keys())}',
                UserWarning)
            return None
        return eval_res[ki]
    return None


def _unwrap_model(runner):
    m = runner.model
    if isinstance(m, DataParallel):
        return m.module
    if hasattr(m, 'module'):
        return m.module
    return m


class SegmanEvalHook(EvalHook):
    """EvalHook + optional second checkpoint by lowest mean val loss."""

    def evaluate(self, runner, results):
        return _evaluate_with_val_aliases(self, runner, results)

    def __init__(self, *args, segman_enable_val_loss_best=False, **kwargs):
        self.segman_enable_val_loss_best = bool(
            kwargs.pop('segman_enable_val_loss_best', segman_enable_val_loss_best))
        self._segman_best_val_loss = float('inf')
        self._segman_val_sup_loader = None
        super().__init__(*args, **kwargs)

    def _segman_get_val_sup_loader(self, runner):
        # 延迟导入，避免 mmseg.core.evaluation 与 mmseg.datasets 循环依赖
        from mmseg.datasets import build_dataloader, build_dataset

        if self._segman_val_sup_loader is False:
            return None
        if self._segman_val_sup_loader is not None:
            return self._segman_val_sup_loader
        cfg = getattr(runner, 'cfg', None)
        if cfg is None or not cfg.get('segman_val_sup_pipeline'):
            runner.logger.warning(
                'cfg.segman_val_sup_pipeline 未设置，跳过按 val loss 存 best')
            self._segman_val_sup_loader = False
            return None
        vcfg = deepcopy(cfg.data.val)
        vcfg['pipeline'] = cfg.segman_val_sup_pipeline
        ds = build_dataset(vcfg, dict(test_mode=False))
        self._segman_val_sup_loader = build_dataloader(
            ds,
            samples_per_gpu=1,
            workers_per_gpu=min(4, cfg.data.get('workers_per_gpu', 2)),
            num_gpus=1,
            dist=False,
            shuffle=False,
            seed=cfg.get('seed'),
            drop_last=False,
            pin_memory=True,
            persistent_workers=False)
        return self._segman_val_sup_loader

    def _do_evaluate(self, runner):
        super()._do_evaluate(runner)
        if not self.segman_enable_val_loss_best:
            return
        self._segman_save_val_loss_best(runner)

    def _segman_save_val_loss_best(self, runner):
        cfg = getattr(runner, 'cfg', None)
        if cfg is None:
            return
        seg_root = cfg.get('segman_segmentation_root')
        run_ts = cfg.get('segman_run_ts')
        if not seg_root or not run_ts:
            return
        bin_root = str(cfg.get('segman_binary_ckpt_root', 'checkpoints'))
        out_dir = osp.join(seg_root, bin_root, 'val_loss_best', str(run_ts))
        mmcv.mkdir_or_exist(out_dir)

        loader = self._segman_get_val_sup_loader(runner)
        if not loader:
            return

        device = next(runner.model.parameters()).device
        model = _unwrap_model(runner)
        was_training = model.training
        model.train()
        s, c = 0.0, 0
        try:
            with torch.no_grad():
                for data in loader:
                    sdata = scatter(data, [device])[0]
                    img = sdata['img']
                    img_metas = sdata['img_metas']
                    gt = sdata['gt_semantic_seg']
                    if hasattr(img, 'data'):
                        img = img.data[0]
                    if hasattr(img_metas, 'data'):
                        img_metas = img_metas.data[0]
                    if hasattr(gt, 'data'):
                        gt = gt.data[0]
                    losses = model.forward_train(img, img_metas, gt)
                    if not isinstance(losses, dict):
                        continue
                    batch_total = 0.0
                    for v in losses.values():
                        if isinstance(v, torch.Tensor):
                            batch_total += float(v.detach().mean().cpu())
                    s += batch_total
                    c += 1
        finally:
            model.train(was_training)

        if c <= 0:
            return
        avg = s / c
        runner.logger.info(
            f'SegmanEvalHook: val mean loss (train forward) = {avg:.6f} '
            f'over {c} batches')
        try:
            runner.log_buffer.output['val/loss'] = float(avg)
        except Exception:
            pass
        if avg >= self._segman_best_val_loss - 1e-12:
            return
        self._segman_best_val_loss = avg
        ckpt = osp.join(out_dir, f'best_val_loss_iter_{runner.iter}.pth')
        mmcv_save_checkpoint(
            _unwrap_model(runner),
            ckpt,
            meta=getattr(runner, 'meta', None))
        runner.logger.info(
            f'SegmanEvalHook: new best val loss {avg:.6f} -> saved {ckpt}')


class SegmanDistEvalHook(DistEvalHook):
    """DistEvalHook + optional val-loss-best (rank 0 only)."""

    def evaluate(self, runner, results):
        return _evaluate_with_val_aliases(self, runner, results)

    def __init__(self, *args, segman_enable_val_loss_best=False, **kwargs):
        self.segman_enable_val_loss_best = bool(
            kwargs.pop('segman_enable_val_loss_best', segman_enable_val_loss_best))
        self._segman_best_val_loss = float('inf')
        self._segman_val_sup_loader = None
        super().__init__(*args, **kwargs)

    def _segman_get_val_sup_loader(self, runner):
        # 延迟导入，避免 mmseg.core.evaluation 与 mmseg.datasets 循环依赖
        from mmseg.datasets import build_dataloader, build_dataset

        if self._segman_val_sup_loader is False:
            return None
        if self._segman_val_sup_loader is not None:
            return self._segman_val_sup_loader
        cfg = getattr(runner, 'cfg', None)
        if cfg is None or not cfg.get('segman_val_sup_pipeline'):
            runner.logger.warning(
                'cfg.segman_val_sup_pipeline 未设置，跳过按 val loss 存 best')
            self._segman_val_sup_loader = False
            return None
        vcfg = deepcopy(cfg.data.val)
        vcfg['pipeline'] = cfg.segman_val_sup_pipeline
        ds = build_dataset(vcfg, dict(test_mode=False))
        self._segman_val_sup_loader = build_dataloader(
            ds,
            samples_per_gpu=1,
            workers_per_gpu=min(4, cfg.data.get('workers_per_gpu', 2)),
            num_gpus=1,
            dist=False,
            shuffle=False,
            seed=cfg.get('seed'),
            drop_last=False,
            pin_memory=True,
            persistent_workers=False)
        return self._segman_val_sup_loader

    def _do_evaluate(self, runner):
        super()._do_evaluate(runner)
        if not self.segman_enable_val_loss_best:
            return
        if runner.rank != 0:
            return
        self._segman_save_val_loss_best(runner)

    def _segman_save_val_loss_best(self, runner):
        cfg = getattr(runner, 'cfg', None)
        if cfg is None:
            return
        seg_root = cfg.get('segman_segmentation_root')
        run_ts = cfg.get('segman_run_ts')
        if not seg_root or not run_ts:
            return
        bin_root = str(cfg.get('segman_binary_ckpt_root', 'checkpoints'))
        out_dir = osp.join(seg_root, bin_root, 'val_loss_best', str(run_ts))
        mmcv.mkdir_or_exist(out_dir)

        loader = self._segman_get_val_sup_loader(runner)
        if not loader:
            return

        device = next(runner.model.parameters()).device
        model = _unwrap_model(runner)
        was_training = model.training
        model.train()
        s, c = 0.0, 0
        try:
            with torch.no_grad():
                for data in loader:
                    sdata = scatter(data, [device])[0]
                    img = sdata['img']
                    img_metas = sdata['img_metas']
                    gt = sdata['gt_semantic_seg']
                    if hasattr(img, 'data'):
                        img = img.data[0]
                    if hasattr(img_metas, 'data'):
                        img_metas = img_metas.data[0]
                    if hasattr(gt, 'data'):
                        gt = gt.data[0]
                    losses = model.forward_train(img, img_metas, gt)
                    if not isinstance(losses, dict):
                        continue
                    batch_total = 0.0
                    for v in losses.values():
                        if isinstance(v, torch.Tensor):
                            batch_total += float(v.detach().mean().cpu())
                    s += batch_total
                    c += 1
        finally:
            model.train(was_training)

        if c <= 0:
            return
        avg = s / c
        runner.logger.info(
            f'SegmanDistEvalHook: val mean loss = {avg:.6f} over {c} batches')
        try:
            runner.log_buffer.output['val/loss'] = float(avg)
        except Exception:
            pass
        if avg >= self._segman_best_val_loss - 1e-12:
            return
        self._segman_best_val_loss = avg
        ckpt = osp.join(out_dir, f'best_val_loss_iter_{runner.iter}.pth')
        mmcv_save_checkpoint(
            _unwrap_model(runner),
            ckpt,
            meta=getattr(runner, 'meta', None))
        runner.logger.info(
            f'SegmanDistEvalHook: new best val loss {avg:.6f} -> saved {ckpt}')
