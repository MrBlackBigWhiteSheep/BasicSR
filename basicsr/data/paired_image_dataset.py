from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    # def __getitem__(self, index):
    #     if self.file_client is None:
    #         self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

    #     scale = self.opt['scale']

    #     # Load gt and lq images. Dimension order: HWC; channel order: BGR;
    #     # image range: [0, 1], float32.
    #     gt_path = self.paths[index]['gt_path']
    #     img_bytes = self.file_client.get(gt_path, 'gt')
    #     img_gt = imfrombytes(img_bytes, float32=True)
    #     lq_path = self.paths[index]['lq_path']
    #     img_bytes = self.file_client.get(lq_path, 'lq')
    #     img_lq = imfrombytes(img_bytes, float32=True)

    #     # augmentation for training
    #     if self.opt['phase'] == 'train':
    #         gt_size = self.opt['gt_size']
    #         # random crop
    #         img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
    #         # flip, rotation
    #         img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

    #     # color space transform
    #     if 'color' in self.opt and self.opt['color'] == 'y':
    #         img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
    #         img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

    #     # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
    #     # TODO: It is better to update the datasets, rather than force to crop
    #     if self.opt['phase'] != 'train':
    #         img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

    #     # BGR to RGB, HWC to CHW, numpy to tensor
    #     img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
    #     # normalize
    #     if self.mean is not None or self.std is not None:
    #         normalize(img_lq, self.mean, self.std, inplace=True)
    #         normalize(img_gt, self.mean, self.std, inplace=True)

    #     return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}
    
    def __getitem__(self, index):
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

            scale = self.opt['scale']

            # Load gt and lq images. Dimension order: HWC; channel order: BGR;
            # image range: [0, 1], float32.
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)

            # augmentation for training
            # if self.opt['phase'] == 'train':
            #     gt_size = self.opt['gt_size']
                
            #     # ================== 【新增 1】: 自动填充逻辑 (解决 ValueError) ==================
            #     import cv2 
            #     # 计算 LQ 需要的最小尺寸
            #     required_lq_size = gt_size // scale
            #     lq_h, lq_w, _ = img_lq.shape
                
            #     # 如果图片小于裁剪尺寸，进行镜像填充
            #     if lq_h < required_lq_size or lq_w < required_lq_size:
            #         pad_h = max(0, required_lq_size - lq_h)
            #         pad_w = max(0, required_lq_size - lq_w)
            #         # 填充 LQ
            #         img_lq = cv2.copyMakeBorder(img_lq, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            #         # 填充 GT (注意比例)
            #         img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h * scale, 0, pad_w * scale, cv2.BORDER_REFLECT_101)
            #     # ================== 【新增 1 结束】 ==================

            #     # random crop
            #     img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            #     # flip, rotation
            #     img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
            
            # -------------------------------------------------------------------------
            #  【SOTA 终极修正版】: 自动缩放对齐 + 自动镜像填充 + 随机裁剪
            # -------------------------------------------------------------------------
            if self.opt['phase'] == 'train':
                gt_size = self.opt['gt_size']
                import cv2 

                # --- 步骤 1: 强制对齐尺寸 (解决 ValueError: Scale mismatches) ---
                # 你的报错是因为 Input 只有 GT 的一半大，这里强制把 Input 拉伸到和 GT 一样大
                if self.opt.get('scale', 1) == 1:
                    h_gt, w_gt = img_gt.shape[:2]
                    h_lq, w_lq = img_lq.shape[:2]
                    if h_gt != h_lq or w_gt != w_lq:
                        # 使用三次插值放大 LQ，保证不报错且保留细节
                        img_lq = cv2.resize(img_lq, (w_gt, h_gt), interpolation=cv2.INTER_CUBIC)

                # --- 步骤 2: 自动填充 (你提供的代码，解决图片小于 Crop Size 问题) ---
                # 计算 LQ 需要的最小尺寸
                required_lq_size = gt_size // scale
                lq_h, lq_w, _ = img_lq.shape
                
                # 如果图片小于裁剪尺寸，进行镜像填充
                if lq_h < required_lq_size or lq_w < required_lq_size:
                    pad_h = max(0, required_lq_size - lq_h)
                    pad_w = max(0, required_lq_size - lq_w)
                    # 填充 LQ
                    img_lq = cv2.copyMakeBorder(img_lq, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                    # 填充 GT (注意 scale 比例，虽然 scale=1 但为了兼容性保留乘法)
                    img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h * scale, 0, pad_w * scale, cv2.BORDER_REFLECT_101)

                # --- 步骤 3: 标准裁剪与增强 ---
                # random crop
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
                # flip, rotation
                img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
            
            # -------------------------------------------------------------------------
            # color space transform
            if 'color' in self.opt and self.opt['color'] == 'y':
                img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
                img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

            # ================== 【新增 2】: 验证/测试集裁剪逻辑 (解决 Batch Size 问题) ==================
            if self.opt['phase'] != 'train':
                # 1. 尝试从配置中获取 gt_size
                val_gt_size = self.opt.get('gt_size', 0)
                cropped = False

                # 2. 如果配置了有效的 gt_size，尝试中心裁剪
                if val_gt_size > 0:
                    target_lq_size = val_gt_size // scale
                    h, w, _ = img_lq.shape
                    
                    # 只有当原图尺寸足够大时才裁剪（不够大这里不处理，由 dataloader 自己 pad 或报错）
                    if h >= target_lq_size and w >= target_lq_size:
                        start_h = (h - target_lq_size) // 2
                        start_w = (w - target_lq_size) // 2
                        img_lq = img_lq[start_h : start_h + target_lq_size, start_w : start_w + target_lq_size, :]
                        
                        gt_start_h = start_h * scale
                        gt_start_w = start_w * scale
                        img_gt = img_gt[gt_start_h : gt_start_h + val_gt_size, gt_start_w : gt_start_w + val_gt_size, :]
                        cropped = True
                
                # 3. 原有逻辑保底
                if not cropped:
                    img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]
            # ================== 【新增 2 结束】 ==================

            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
            # normalize
            if self.mean is not None or self.std is not None:
                normalize(img_lq, self.mean, self.std, inplace=True)
                normalize(img_gt, self.mean, self.std, inplace=True)

            return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
