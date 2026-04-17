# Copyright (c) OpenMMLab-style extension for SegMAN binary-seg experiments.
# 终端彩色指标摘要，风格对齐 TransNeXt ``ConsoleSummaryHook``（ANSI 色码 + loss/lr/验证表）。
import numbers

import torch
from mmcv.runner import HOOKS, EpochBasedRunner, Hook, IterBasedRunner
from mmcv.runner import get_dist_info


@HOOKS.register_module()
class SegmanConsoleSummaryHook(Hook):
    """按间隔打印训练 loss / lr / 显存 / 进度；验证后打印彩色指标表。

    训练标量来自 ``runner.log_buffer.val_history``（与 TextLoggerHook 无关）；验证指标在
    EvalHook 写入 ``log_buffer.output`` 之后由本 hook 的 ``after_train_epoch``（建议
    ``priority=78``，高于 EvalHook 的 LOW=70）读取。
    """

    _MAX_REASONABLE_EPOCH_LEN = 100000

    def __init__(self, interval=1):
        self.interval = max(1, int(interval))

    @staticmethod
    def _c(text, color):
        colors = {
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'magenta': '\033[95m',
            'bold': '\033[1m',
            'white': '\033[97m',
        }
        end = '\033[0m'
        return f'{colors.get(color, "")}{text}{end}'

    def _cell(self, text, width, color):
        s = str(text)
        if len(s) > width:
            s = s[: max(1, width - 2)] + '..'
        s = s.ljust(width)
        return self._c(s, color)

    @staticmethod
    def _last_scalar(runner, key):
        hist = runner.log_buffer.val_history.get(key)
        if hist:
            v = hist[-1]
            if hasattr(v, 'item'):
                return float(v.item())
            return float(v)
        out = runner.log_buffer.output.get(key)
        if out is None:
            return None
        if hasattr(out, 'item'):
            return float(out.item())
        try:
            return float(out)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _current_lr(runner):
        try:
            lr = runner.current_lr()
        except Exception:
            return None
        if isinstance(lr, (list, tuple)) and len(lr) > 0:
            return float(lr[0])
        if isinstance(lr, dict) and lr:
            first = next(iter(lr.values()))
            if isinstance(first, (list, tuple)) and first:
                return float(first[0])
            if isinstance(first, (int, float)):
                return float(first)
        return None

    @staticmethod
    def _image_size_from_batch(data_batch):
        if not isinstance(data_batch, dict):
            return None
        img = data_batch.get('img')
        if img is None:
            return None
        if hasattr(img, 'data'):
            img = img.data
        if isinstance(img, (list, tuple)) and len(img) > 0:
            t = img[0]
        else:
            t = img
        if not hasattr(t, 'shape') or len(t.shape) < 2:
            return None
        return int(t.shape[-1])

    @staticmethod
    def _pick_metric(buf, keys):
        out = buf.output if hasattr(buf, 'output') else {}
        for key in keys:
            if key in out:
                return out[key]
        for key in keys:
            for metric_key, metric_val in out.items():
                if str(metric_key).endswith(key):
                    return metric_val
        return None

    @staticmethod
    def _fmt_pct(v):
        if not isinstance(v, numbers.Real):
            # mmseg 验证写入 log_buffer 的键因版本/配置而异，未输出时并非「无训练参数」
            return '未记录'
        x = float(v)
        if x <= 1.0 + 1e-6:
            x = x * 100.0
        return f'{x:.2f}'

    def _progress_tags(self, runner, batch_idx=None):
        it = runner.iter + 1
        tags = []
        if isinstance(runner, EpochBasedRunner):
            ep = runner.epoch + 1
            me = runner.max_epochs
            tags.append(f'epoch={ep}' + (f'/{me}' if me is not None else ''))
            if batch_idx is not None:
                try:
                    n = len(runner.data_loader)
                    if n and n < self._MAX_REASONABLE_EPOCH_LEN:
                        tags.append(f'batch={batch_idx + 1}/{n}')
                except (TypeError, AttributeError):
                    pass
            tags.append(f'global_iter={it}')
        elif isinstance(runner, IterBasedRunner):
            mi = runner.max_iters
            tags.append(f'iter={it}' + (f'/{mi}' if mi is not None else ''))
        else:
            tags.append(f'global_iter={it}')
        return ' | '.join(tags)

    def _epoch_label(self, runner):
        if not isinstance(runner, EpochBasedRunner):
            return str(runner.epoch + 1)
        cur = runner.epoch + 1
        me = runner.max_epochs
        if me:
            return f'{cur}/{me}'
        return str(cur)

    def _print_epoch_train_footer(self, runner, batch_idx):
        w_ep, w_dn, w_mem, w_loss, w_lr, w_img = 10, 12, 14, 14, 14, 10
        loss = self._last_scalar(runner, 'loss')
        lr = self._current_lr(runner)
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024**2)
        else:
            gpu_mem = 0.0
        try:
            total = len(runner.data_loader)
            data_num = f'{batch_idx + 1}/{total}'
        except (TypeError, AttributeError):
            data_num = f'{batch_idx + 1}/?'
        data_batch = getattr(runner, 'data_batch', None)
        img = self._image_size_from_batch(data_batch)
        loss_str = f'{loss:.8f}' if loss is not None else '?'
        lr_str = f'{lr:.8f}' if lr is not None else '?'
        img_str = str(img) if img is not None else '?'
        row1 = (
            self._cell('Epoch', w_ep, 'green')
            + self._cell('data_num', w_dn, 'yellow')
            + self._cell('GPU Mem', w_mem, 'yellow')
            + self._cell('Loss', w_loss, 'yellow')
            + self._cell('LR', w_lr, 'yellow')
            + self._cell('Image_size', w_img, 'yellow'))
        row2 = (
            self._cell(self._epoch_label(runner), w_ep, 'bold')
            + self._cell(data_num, w_dn, 'white')
            + self._cell(f'{gpu_mem:.2f} MB', w_mem, 'white')
            + self._cell(loss_str, w_loss, 'white')
            + self._cell(lr_str, w_lr, 'white')
            + self._cell(img_str, w_img, 'white'))
        print(self._c(f'[本轮训练结束] {self._progress_tags(runner, batch_idx)}', 'cyan'),
              flush=True)
        print(row1, flush=True)
        print(row2, flush=True)

    def after_train_iter(self, runner):
        rank, _ = get_dist_info()
        if rank != 0:
            return
        batch_idx = getattr(runner, 'inner_iter', 0)
        if isinstance(runner, EpochBasedRunner):
            try:
                total = len(runner.data_loader)
            except (TypeError, AttributeError):
                total = None
            epoch_end = (
                total is not None and 0 < total < self._MAX_REASONABLE_EPOCH_LEN
                and (batch_idx + 1) == total)
            if epoch_end:
                loss = self._last_scalar(runner, 'loss')
                lr = self._current_lr(runner)
                loss_str = f'{loss:.6f}' if loss is not None else '?'
                lr_str = f'{lr:.8f}' if lr is not None else '?'
                ep_done = runner.epoch + 1
                me = runner.max_epochs
                ep_tag = f'{ep_done}/{me}' if me is not None else str(ep_done)
                it = runner.iter + 1
                print(flush=True)
                print(
                    self._c('[Epoch]', 'magenta')
                    + f' 第 {ep_tag} 轮训练阶段结束 | global_iter={it} | '
                    f'loss≈{loss_str} | lr={lr_str} | 随后验证集…',
                    flush=True)
                print(flush=True)
                self._print_epoch_train_footer(runner, batch_idx)
                return

        if (runner.iter + 1) % self.interval != 0:
            return

        loss = self._last_scalar(runner, 'loss')
        lr = self._current_lr(runner)
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024**2)
        else:
            gpu_mem = 0.0
        try:
            total = len(runner.data_loader)
            data_num = f'{batch_idx + 1}/{total}'
        except (TypeError, AttributeError):
            data_num = f'{batch_idx + 1}/?'
        data_batch = getattr(runner, 'data_batch', None)
        image_size = self._image_size_from_batch(data_batch)
        loss_str = f'{loss:.6f}' if loss is not None else '?'
        lr_str = f'{lr:.8f}' if lr is not None else '?'
        image_size_str = str(image_size) if image_size is not None else '?'
        prog = self._progress_tags(runner, batch_idx=batch_idx)
        print(
            f"{self._c('[训练]', 'green')} {prog} | "
            f"{self._c('batch进度', 'yellow')} {data_num} | "
            f"{self._c('GPU Mem', 'magenta')} {gpu_mem:.2f} MB | "
            f"{self._c('Loss', 'red')} {loss_str} | "
            f"{self._c('LR', 'yellow')} {lr_str} | "
            f"{self._c('Img', 'blue')} {image_size_str}",
            flush=True)

    def after_train_epoch(self, runner):
        rank, _ = get_dist_info()
        if rank != 0:
            return
        cfg = getattr(runner, 'cfg', None)
        if cfg is None:
            return
        rtype = cfg.get('runner', {}).get('type', '')
        if rtype != 'EpochBasedRunner':
            return
        ev = cfg.get('evaluation') or {}
        interval = int(ev.get('interval', 1))
        if interval <= 0:
            return
        if (runner.epoch + 1) % interval != 0:
            return
        buf = runner.log_buffer
        if not any(
                k in buf.output
                for k in ('mIoU', 'aAcc', 'mDice', 'IoU.foreground', 'val/IoU')):
            return
        self._print_val_block(runner)

    def _print_val_block(self, runner):
        buf = runner.log_buffer
        vtag = self._progress_tags(runner, batch_idx=None)
        print(self._c(f'[验证] {vtag}', 'red'), flush=True)

        w_dn, w_iou, w_f1, w_p, w_r, w_a = 12, 14, 10, 12, 12, 10
        row3 = (
            self._cell('phase', w_dn, 'red')
            + self._cell('IoU(fg)', w_iou, 'red')
            + self._cell('F1', w_f1, 'red')
            + self._cell('Precision', w_p, 'red')
            + self._cell('Recall', w_r, 'red')
            + self._cell('aAcc', w_a, 'red'))

        iou_fg = buf.output.get('val/IoU',
                               buf.output.get('IoU.foreground',
                                              self._pick_metric(buf, ['IoU', 'mIoU'])))
        # 与 ``evaluation.metric=['mIoU','mFscore']`` 对齐：优先前景类，其次 mean
        mf1 = self._pick_metric(buf, [
            'val/F1', 'Fscore.foreground', 'mFscore', 'F1', 'mF1', 'Fscore'])
        mp = self._pick_metric(buf, [
            'val/Precision', 'Precision.foreground', 'mPrecision', 'Precision'])
        mr = self._pick_metric(buf, [
            'val/Recall', 'Recall.foreground', 'mRecall', 'Recall'])
        aacc = self._pick_metric(buf, ['aAcc'])

        row4 = (
            self._cell('val', w_dn, 'white')
            + self._cell(self._fmt_pct(iou_fg), w_iou, 'white')
            + self._cell(self._fmt_pct(mf1), w_f1, 'white')
            + self._cell(self._fmt_pct(mp), w_p, 'white')
            + self._cell(self._fmt_pct(mr), w_r, 'white')
            + self._cell(self._fmt_pct(aacc), w_a, 'white'))
        print(row3, flush=True)
        print(row4, flush=True)
        print(flush=True)
