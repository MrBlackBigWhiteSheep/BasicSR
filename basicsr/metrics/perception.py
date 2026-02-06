import math
import numpy as np
import cv2
import torch
import logging
from skimage import color, filters

from basicsr.metrics.metric_util import reorder_image
from basicsr.utils.registry import METRIC_REGISTRY
from basicsr.utils import get_root_logger

# 尝试导入 LPIPS，如果没安装则跳过，防止报错
try:
    import lpips
    from basicsr.utils import img2tensor
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False

# =====================================================================
# Helper Functions (来源于你提供的核心算法)
# =====================================================================

def eme(ch, blocksize=8, eps=1e-6):
    """向量化计算 EME"""
    if ch.ndim != 2:
        raise ValueError("eme 要求输入为 2D 单通道数组")

    h, w = ch.shape
    arr = ch.astype(np.float32)

    pad_h = (blocksize - (h % blocksize)) % blocksize
    pad_w = (blocksize - (w % blocksize)) % blocksize
    if pad_h != 0 or pad_w != 0:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='reflect')
    
    H, W = arr.shape
    n_blocks_h = H // blocksize
    n_blocks_w = W // blocksize
    
    arr_blocks = arr.reshape(n_blocks_h, blocksize, n_blocks_w, blocksize)
    arr_blocks = arr_blocks.transpose(0, 2, 1, 3) 

    block_min = arr_blocks.min(axis=(2, 3))
    block_max = arr_blocks.max(axis=(2, 3))

    block_min = np.maximum(block_min, eps)
    block_max = np.maximum(block_max, eps)

    log_ratio = np.log(block_max / block_min)
    wgt = 2.0 / (n_blocks_h * n_blocks_w)
    eme_val = wgt * np.sum(log_ratio)
    return float(eme_val)

def plipsum(i, j, gamma=1026):
    return i + j - i * j / gamma

def plipsub(i, j, k=1026):
    return k * (i - j) / (k - j)

def plipmult(c, j, gamma=1026):
    return gamma - gamma * (1 - j / gamma)**c

def logamee(ch, blocksize=8, pad_mode='reflect', eps=1e-12):
    """纯 NumPy 向量化实现 logAMEE"""
    arr = np.asarray(ch, dtype=np.float64)
    H, W = arr.shape
    nx = math.ceil(H / blocksize)
    ny = math.ceil(W / blocksize)
    H2 = nx * blocksize
    W2 = ny * blocksize

    if H2 != H or W2 != W:
        pad_h = H2 - H
        pad_w = W2 - W
        arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode=pad_mode)

    blocks = arr.reshape(nx, blocksize, ny, blocksize).transpose(0, 2, 1, 3)
    block_min = blocks.min(axis=(2, 3))
    block_max = blocks.max(axis=(2, 3))

    top = plipsub(block_max, block_min)
    bottom = plipsum(block_max, block_min)

    with np.errstate(divide='ignore', invalid='ignore'):
        m = np.where(np.abs(bottom) <= eps, 0.0, top / bottom)

    positive = m > eps
    terms = np.zeros_like(m, dtype=np.float64)
    terms[positive] = m[positive] * np.log(m[positive])

    s = terms.sum()
    w = 1.0 / (nx * ny)
    return plipmult(w, s)

def _prepare_data(img, crop_border, input_order):
    """通用数据预处理：重排 -> 裁剪 -> 转uint8 [0,255]"""
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)

    if crop_border != 0:
        h, w = img.shape[:2]
        img = img[crop_border:h-crop_border, crop_border:w-crop_border, ...]

    # 你的算法大多依赖 [0, 255] 范围，这里强制转换
    if img.max() <= 1.01: # 稍微放宽一点防止浮点误差
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    
    return img

# =====================================================================
# Registered Metric: UIQM
# =====================================================================

@METRIC_REGISTRY.register()
def calculate_uiqm(img, img2=None, crop_border=0, input_order='HWC', **kwargs):
    """
    Calculate UIQM (Underwater Image Quality Measure).
    Reference-free: img2 is ignored.
    Args:
        img (ndarray): Image, usually float32 [0, 1] from BasicSR validation loop.
    """
    # 1. 数据准备 (转为 uint8 HWC)
    img = _prepare_data(img, crop_border, input_order)
    
    # 2. 转换为 RGB (因为 BasicSR 读取可能是 BGR，但你的算法基于 rgb2lab/rgb2gray，假设输入是 RGB)
    # BasicSR 默认由 opencv 读取为 BGR。skimage 的 color 转换函数默认输入是 RGB。
    # 这里我们统一假定输入为 BGR (BasicSR标准)，转为 RGB 供你的算法使用。
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = color.rgb2gray(img_rgb) # skimage 接收 RGB 返回 [0,1] float gray

    # --- UIQM 核心逻辑 ---
    p1, p2, p3 = 0.0282, 0.2953, 3.5753

    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]
    
    # UICM
    rg = (R.astype(float) - G.astype(float)).ravel()
    yb = ((R.astype(float) + G.astype(float)) * 0.5 - B.astype(float)).ravel()
    
    urg, s2rg = np.mean(rg), np.var(rg)
    uyb, s2yb = np.mean(yb), np.var(yb)
    uicm = -0.0268 * math.hypot(urg, uyb) + 0.1586 * math.hypot(s2rg, s2yb)

    # UISM (Sharpness)
    def _to_uint8(mat):
        m = np.clip(mat, 0.0, 1.0)
        return np.rint(m * 255.0).astype(np.uint8)

    # filters.sobel 期望浮点输入，输出也是浮点
    Rsobel = R.astype(float) * filters.sobel(R)
    Gsobel = G.astype(float) * filters.sobel(G)
    Bsobel = B.astype(float) * filters.sobel(B)

    Reme = eme(_to_uint8(Rsobel))
    Geme = eme(_to_uint8(Gsobel))
    Beme = eme(_to_uint8(Bsobel))

    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    # UIConM (Contrast) - 使用 logamee
    uiconm = logamee(img_gray)

    uiqm = p1 * uicm + p2 * uism + p3 * uiconm
    return uiqm

# =====================================================================
# Registered Metric: UCIQE
# =====================================================================

@METRIC_REGISTRY.register()
def calculate_uciqe(img, img2=None, crop_border=0, input_order='HWC', **kwargs):
    """
    Calculate UCIQE (Underwater Color Image Quality Evaluation).
    Reference-free.
    """
    img = _prepare_data(img, crop_border, input_order)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BasicSR BGR -> RGB for skimage
    
    # skimage rgb2lab 期望 [0,1] float RGB，或者 [0, 255] uint8 RGB
    # 这里我们直接传 uint8，它会处理
    lab = color.rgb2lab(img_rgb) 
    
    L = lab[:, :, 0]
    a_chan = lab[:, :, 1]
    b_chan = lab[:, :, 2]

    # Chroma
    chroma = np.hypot(a_chan, b_chan)
    sc = float(np.std(chroma))

    # Contrast of Luminance (Top 1% logic)
    H, W = L.shape
    Npix = H * W
    top_k = int(round(0.01 * Npix))
    
    if top_k <= 0 or top_k >= Npix:
        conl = 0.0
    else:
        flatL = L.ravel()
        # 使用排序替代 partition，更稳定
        sl = np.sort(flatL)
        conl = float(np.mean(sl[-top_k:]) - np.mean(sl[:top_k]))

    # Saturation
    flat_chroma = chroma.ravel()
    flat_L = L.ravel()
    # 避免除以 0
    with np.errstate(divide='ignore', invalid='ignore'):
        satur = flat_chroma / flat_L
        satur = np.where(np.isfinite(satur), satur, 0.0)
    
    us = float(np.mean(satur))

    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    uciqe = c1 * sc + c2 * conl + c3 * us
    return uciqe

# =====================================================================
# Registered Metric: Entropy
# =====================================================================

@METRIC_REGISTRY.register()
def calculate_entropy(img, img2=None, crop_border=0, input_order='HWC', **kwargs):
    """
    Calculate Image Entropy.
    """
    img = _prepare_data(img, crop_border, input_order) # 获取 uint8 BGR
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 你的逻辑：计算直方图 -> 概率 -> 熵
    hist = np.histogram(img_gray, bins=256, range=(0, 256))[0]
    hist = hist.astype(np.float32) / hist.sum()

    hist_nonzero = hist[hist > 0]
    ent = -np.sum(hist_nonzero * np.log2(hist_nonzero))
    return float(ent)

# =====================================================================
# Registered Metric: LPIPS (Optional, included for completeness)
# =====================================================================
try:
    import lpips
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False

# ==========================================================
# 缺失的部分 1: 全局变量定义
# ==========================================================
_LPIPS_VGG = None

# ==========================================================
# 缺失的部分 2: 获取模型的辅助函数
# ==========================================================
def _get_lpips_model(device):
    global _LPIPS_VGG
    if _LPIPS_VGG is None:
        logger = get_root_logger()
        if logger is None:
            logger = logging.getLogger('basicsr')
        logger.info('Loading LPIPS model (VGG)...')
        # 确保 lpips 已安装
        if not _LPIPS_AVAILABLE:
             raise ImportError("LPIPS not installed. Please run: pip install lpips")
             
        _LPIPS_VGG = lpips.LPIPS(net='vgg').to(device)
        _LPIPS_VGG.eval()
    return _LPIPS_VGG
@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, crop_border=0, input_order='HWC', **kwargs):
    """Calculate LPIPS.
    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image.
        input_order (str): 'HWC' or 'CHW'.
    """
    if not _LPIPS_AVAILABLE:
        raise ImportError("LPIPS not installed. Run 'pip install lpips'")
        
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # LPIPS 需要 float32 [0,1]
    img = img.astype(np.float32) / 255. if img.max() > 1 else img.astype(np.float32)
    img2 = img2.astype(np.float32) / 255. if img2.max() > 1 else img2.astype(np.float32)

    device_opt = kwargs.get('device', None)
    if device_opt is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        if device_opt == 'cuda' and not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device(device_opt)

    img_tensor = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0).to(device)
    img2_tensor = img2tensor(img2, bgr2rgb=True, float32=True).unsqueeze(0).to(device)

    # Normalize to [-1, 1]
    img_tensor = (img_tensor - 0.5) * 2
    img2_tensor = (img2_tensor - 0.5) * 2

    lpips_model = _get_lpips_model(device)
    with torch.no_grad():
        lpips_val = lpips_model(img_tensor, img2_tensor)

    return lpips_val.item()