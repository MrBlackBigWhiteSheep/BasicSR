from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim
from .perception import calculate_lpips, calculate_uiqm, calculate_uciqe, calculate_entropy, calculate_lpips

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe', 'calculate_lpips', 'calculate_uiqm', 'calculate_uciqe', 'calculate_entropy']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
