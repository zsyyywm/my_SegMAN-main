# Copyright (c) OpenMMLab extension — 二分类线材 512 双路（argmax + 阈值）与 256 方案3（仅阈值 0.5）验证/测试
import warnings

import torch
import torch.nn as nn
from mmcv.parallel import DataContainer as DC

def _unwrap_model(model):
    m = model
    if isinstance(m, nn.DataParallel) or (
            m.__class__.__name__ in ('MMSeparateDistributedDataParallel',)):
        return m.module
    if hasattr(m, 'module'):
        return m.module
    return m


def _flatten_img_metas_to_list_dict(md):
    """将 mmcv Collect/collate 后的 ``meta`` 展成 ``EncoderDecoder.inference`` 所需的
    ``list[dict]``（每元素对应 batch 内一张图）。

    - 训练：**``DataContainer.data``** 常为 ``[[d0,...,dB]]``（外层按 GPU）。
    - 验证（``MultiScaleFlipAug``）：**``data['img_metas']``** 为 **list[DC]**，
      第一层 list 对应 TTA 分支；单层增强时亦为 ``list[DC]``，须先取 **[0]** 再 unwrap，
      与 ``mmseg/apis/test.py`` 中 ``data['img_metas'][0].data[0]`` 一致。"""
    cur = md
    # 常见形态：
    # - dict
    # - list[dict]
    # - list[list[dict]]（每卡一组）
    # - list[DC]（每增强一组）
    # - DC(...) 的任意嵌套
    while True:
        if isinstance(cur, dict):
            return [cur]
        if isinstance(cur, DC):
            cur = cur.data
            continue
        if isinstance(cur, tuple):
            cur = list(cur)
            continue
        if not isinstance(cur, list) or len(cur) == 0:
            raise TypeError(
                'wire_test: img_metas unexpected structure, '
                f'type={type(cur)!r}, value_preview={str(cur)[:160]}')
        head = cur[0]
        if isinstance(head, dict):
            return cur
        # 继续下钻到第一条分支（与 mmseg single_gpu_test 取第 0 增强分支一致）
        cur = head


def _img_and_metas_from_data(data, device):
    """从 ``DataLoader`` 的 ``data`` 中取出与 ``inference`` 一致的 batch 图像张量及
    ``img_meta``: ``list[dict]``。"""
    img = data['img']
    # 验证集 ``MultiScaleFlipAug``：**``forward_test``** 要求 ``img`` 为 list[Tensor]；
    # Collate 后常为 ``list`` 第一项即本批次张量。
    if isinstance(img, torch.Tensor):
        img0 = img
    elif isinstance(img, list) and len(img) > 0:
        img0 = img[0]
        if isinstance(img0, DC):
            img0 = img0.data[0]
    elif isinstance(img, DC):
        blk = img.data
        img0 = blk[0] if isinstance(blk, list) and len(blk) else blk
    else:
        img0 = img
    if isinstance(img0, list) and len(img0) > 0:
        img0 = img0[0]

    raw_meta = data['img_metas']
    # ``list[DataContainer]``：测试时 TTA，只取第一条增强分支（与 ``single_gpu_test`` 对齐）。
    if isinstance(raw_meta, list) and len(raw_meta) > 0 and isinstance(
            raw_meta[0], DC):
        md = raw_meta[0].data
    elif isinstance(raw_meta, DC):
        md = raw_meta.data
    else:
        md = raw_meta
    meta0 = _flatten_img_metas_to_list_dict(md)
    if not isinstance(img0, torch.Tensor):
        raise TypeError('wire_test: expected tensor img, got ' + str(type(img0)))
    if device is not None and img0.device != device:
        img0 = img0.to(device, non_blocking=True)
    if not isinstance(meta0, (list, tuple)) or (len(meta0) and
                                                 not isinstance(meta0[0], dict)):
        raise TypeError(
            'wire_test: expected img_meta as list[dict], got '
            f'{type(meta0)!r} first={type(meta0[0]).__name__ if isinstance(meta0, (list, tuple)) and len(meta0) else "EMPTY"}')
    return img0, meta0


@torch.no_grad()
def single_gpu_test_wire_dual(model, data_loader, fg_index=1, th055=0.55):
    """单卡评估：对同一 ``inference`` 概率图分别用 argmax 与 ``P(fg) > th055`` 生成
    两套 pre_eval 结果，供 ``CustomDataset.evaluate`` 各算一套指标。仅支持
    ``samples_per_gpu=1``。"""
    m = _unwrap_model(model)
    m.eval()
    device = next(m.parameters()).device
    r_argmax, r_t = [], []
    dataset = data_loader.dataset
    for batch_indices, data in zip(data_loader.batch_sampler, data_loader):
        with torch.no_grad():
            x, metas = _img_and_metas_from_data(data, device)
        prob = m.inference(x, metas, rescale=True)
        bsz = int(x.size(0))
        if bsz > 1:
            warnings.warn(
                'wire_test: samples_per_gpu>1 未全测，请保持 val 上 samples_per_gpu=1',
                UserWarning, stacklevel=2)
        if m.out_channels == 1:
            s = prob.squeeze(1)
            a = (s > 0.5).long().cpu().numpy()
            b = (s > th055).long().cpu().numpy()
        else:
            a = prob.argmax(1).cpu().numpy()
            b = (prob[:, fg_index] > th055).long().cpu().numpy()
        if not isinstance(batch_indices, (list, tuple)):
            inds = [int(batch_indices)]
        else:
            inds = [int(b) for b in batch_indices]
        if len(inds) < bsz:
            inds = [inds[0] + j for j in range(bsz)]
        for j in range(bsz):
            pa = a[j] if a.ndim == 3 else a
            pb = b[j] if b.ndim == 3 else b
            r_argmax.extend(dataset.pre_eval([pa], [inds[j]]))
            r_t.extend(dataset.pre_eval([pb], [inds[j]]))
    return r_argmax, r_t


@torch.no_grad()
def single_gpu_test_wire_scheme3_t05(model, data_loader, fg_index=1, th=0.5):
    """单路阈值验证：仅用 ``P(fg) > th`` 生成预测再 ``pre_eval``（方案二 512/th0.55 与方案三 256/th0.5 共用）。"""
    m = _unwrap_model(model)
    m.eval()
    device = next(m.parameters()).device
    results = []
    dataset = data_loader.dataset
    for batch_indices, data in zip(data_loader.batch_sampler, data_loader):
        with torch.no_grad():
            x, metas = _img_and_metas_from_data(data, device)
        prob = m.inference(x, metas, rescale=True)
        bsz = int(x.size(0))
        if m.out_channels == 1:
            s = prob.squeeze(1)
            b = (s > th).long().cpu().numpy()
        else:
            b = (prob[:, fg_index] > th).long().cpu().numpy()
        if not isinstance(batch_indices, (list, tuple)):
            inds = [int(batch_indices)]
        else:
            inds = [int(b) for b in batch_indices]
        for j in range(min(bsz, len(inds))):
            pb = b[j] if b.ndim == 3 else b
            results.extend(dataset.pre_eval([pb], [inds[j]]))
    return results


