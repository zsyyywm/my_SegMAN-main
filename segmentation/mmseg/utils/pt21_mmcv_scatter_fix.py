# Copyright (c) OpenMMLab. All rights reserved.
# PyTorch 2.1+ torch.nn.parallel._functions._get_stream expects torch.device;
# mmcv<=1.7 Scatter.forward passes int GPU indices -> AttributeError: 'int' has no attribute 'type'.
import torch
from torch.nn.parallel import _functions as torch_parallel_fn


def apply_mmcv_scatter_pt21_hotfix():
    """Idempotent monkey-patch for mmcv.parallel._functions.Scatter.forward."""
    import mmcv.parallel._functions as mmcv_pf

    if getattr(mmcv_pf.Scatter, '_mmseg_pt21_hotfix_applied', False):
        return

    _orig_forward = mmcv_pf.Scatter.forward

    @staticmethod
    def _patched_forward(target_gpus, input):
        input_device = mmcv_pf.get_input_device(input)
        streams = None
        if input_device == -1 and target_gpus != [-1]:
            stream_devices = []
            for d in target_gpus:
                if isinstance(d, int):
                    stream_devices.append(
                        torch.device('cpu') if d == -1 else torch.device(
                            'cuda', d))
                else:
                    stream_devices.append(d)
            streams = [
                torch_parallel_fn._get_stream(dev) for dev in stream_devices
            ]

        outputs = mmcv_pf.scatter(input, target_gpus, streams)
        if streams is not None:
            mmcv_pf.synchronize_stream(outputs, target_gpus, streams)

        return tuple(outputs) if isinstance(outputs, list) else (outputs, )

    mmcv_pf.Scatter.forward = _patched_forward
    mmcv_pf.Scatter._mmseg_pt21_hotfix_applied = True

