import os.path as osp
from torch.utils import data as data
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class RESIDEDataset(data.Dataset):
    def __init__(self, opt):
        super(RESIDEDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        
        # 扫描目录下所有雾图路径
        self.lq_paths = sorted(list(scandir(self.lq_folder, full_path=True)))

    def __getitem__(self, index):
        # 1. 初始化 FileClient (懒加载模式)
        if self.file_client is None:
            curr_opt = self.io_backend_opt.copy()
            client_type = curr_opt.pop('type')
            self.file_client = FileClient(client_type, **curr_opt)
            
        lq_path = self.lq_paths[index]
        lq_filename = osp.basename(lq_path)

        # -----------------------------------------------------------
        # 核心配对逻辑：根据你的描述定制
        # 例子：lq_filename = "1_1_0.90179.png"
        # -----------------------------------------------------------
        
        # 第一步：提取 ID (获取 "1")
        gt_id = lq_filename.split('_')[0]
        
        # 第二步：拼接 GT 文件名 (拼接成 "1.png")
        # 注意：RESIDE ITS 的 GT 都是 .png
        gt_filename = f'{gt_id}.png'
        
        # 第三步：组合完整路径
        gt_path = osp.join(self.gt_folder, gt_filename)

        # [保险措施] 快速检查文件是否存在
        # 如果你不想每次都检查拖慢速度，可以注释掉下面这块 try-except
        # 但强烈建议保留，直到跑通为止
        # if not osp.exists(gt_path):
            # 兼容性尝试：万一 GT 有前导零 (如 0001.png)
            # gt_filename_padded = f'{int(gt_id):04d}.png' # 尝试 0001.png
            # gt_path = osp.join(self.gt_folder, gt_filename_padded)
            # if not osp.exists(gt_path):
            #    raise FileNotFoundError(f"找不到 GT 图片: {gt_filename} (对应雾图 {lq_filename})")

        # -----------------------------------------------------------

        # 2. 读取图像
        try:
            img_bytes_lq = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes_lq, float32=True)
            
            img_bytes_gt = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes_gt, float32=True)
        except Exception as e:
            raise IOError(f"读取图片失败: \nLQ: {lq_path}\nGT: {gt_path}\n错误信息: {e}")

        # 3. 训练时的增强操作 (随机裁剪、翻转)
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, 1, lq_path)
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # 4. 格式转换 (BGR -> RGB, HWC -> CHW, Numpy -> Tensor)
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.lq_paths)