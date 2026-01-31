import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel


@MODEL_REGISTRY.register()
class MambaWaterModel(SRModel):

    # test by partitioning
    # def test(self):
    #     _, C, h, w = self.lq.size()
    #     split_token_h = h // 200 + 1  # number of horizontal cut sections
    #     split_token_w = w // 200 + 1  # number of vertical cut sections
    #     # padding
    #     mod_pad_h, mod_pad_w = 0, 0
    #     if h % split_token_h != 0:
    #         mod_pad_h = split_token_h - h % split_token_h
    #     if w % split_token_w != 0:
    #         mod_pad_w = split_token_w - w % split_token_w
    #     img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    #     _, _, H, W = img.size()
    #     split_h = H // split_token_h  # height of each partition
    #     split_w = W // split_token_w  # width of each partition
    #     # overlapping
    #     shave_h = split_h // 10
    #     shave_w = split_w // 10
    #     scale = self.opt.get('scale', 1)
    #     ral = H // split_h
    #     row = W // split_w
    #     slices = []  # list of partition borders
    #     for i in range(ral):
    #         for j in range(row):
    #             if i == 0 and i == ral - 1:
    #                 top = slice(i * split_h, (i + 1) * split_h)
    #             elif i == 0:
    #                 top = slice(i*split_h, (i+1)*split_h+shave_h)
    #             elif i == ral - 1:
    #                 top = slice(i*split_h-shave_h, (i+1)*split_h)
    #             else:
    #                 top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)
    #             if j == 0 and j == row - 1:
    #                 left = slice(j*split_w, (j+1)*split_w)
    #             elif j == 0:
    #                 left = slice(j*split_w, (j+1)*split_w+shave_w)
    #             elif j == row - 1:
    #                 left = slice(j*split_w-shave_w, (j+1)*split_w)
    #             else:
    #                 left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)
    #             temp = (top, left)
    #             slices.append(temp)
    #     img_chops = []  # list of partitions
    #     for temp in slices:
    #         top, left = temp
    #         img_chops.append(img[..., top, left])
    #     if hasattr(self, 'net_g_ema'):
    #         self.net_g_ema.eval()
    #         with torch.no_grad():
    #             outputs = []
    #             for chop in img_chops:
    #                 out = self.net_g_ema(chop)  # image processing of each partition
    #                 outputs.append(out)
    #             _img = torch.zeros(1, C, H * scale, W * scale)
    #             # merge
    #             for i in range(ral):
    #                 for j in range(row):
    #                     top = slice(i * split_h * scale, (i + 1) * split_h * scale)
    #                     left = slice(j * split_w * scale, (j + 1) * split_w * scale)
    #                     if i == 0:
    #                         _top = slice(0, split_h * scale)
    #                     else:
    #                         _top = slice(shave_h*scale, (shave_h+split_h)*scale)
    #                     if j == 0:
    #                         _left = slice(0, split_w*scale)
    #                     else:
    #                         _left = slice(shave_w*scale, (shave_w+split_w)*scale)
    #                     _img[..., top, left] = outputs[i * row + j][..., _top, _left]
    #             self.output = _img
    #     else:
    #         self.net_g.eval()
    #         with torch.no_grad():
    #             outputs = []
    #             for chop in img_chops:
    #                 out = self.net_g(chop)  # image processing of each partition
    #                 outputs.append(out)
    #             _img = torch.zeros(1, C, H * scale, W * scale)
    #             # merge
    #             for i in range(ral):
    #                 for j in range(row):
    #                     top = slice(i * split_h * scale, (i + 1) * split_h * scale)
    #                     left = slice(j * split_w * scale, (j + 1) * split_w * scale)
    #                     if i == 0:
    #                         _top = slice(0, split_h * scale)
    #                     else:
    #                         _top = slice(shave_h * scale, (shave_h + split_h) * scale)
    #                     if j == 0:
    #                         _left = slice(0, split_w * scale)
    #                     else:
    #                         _left = slice(shave_w * scale, (shave_w + split_w) * scale)
    #                     _img[..., top, left] = outputs[i * row + j][..., _top, _left]
    #             self.output = _img
    #         self.net_g.train()
    #     _, _, h, w = self.output.size()
    #     self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def test(self):
            # ---------------- 1. 获取配置参数 ----------------
            # 尝试从 YAML 的 val 部分读取 center_crop_size
            # 如果 YAML 里没写，或者写了 ~, 这里就会得到 None
            crop_size = self.opt['val'].get('center_crop_size', None)
            scale = self.opt.get('scale', 1)
            
            # ---------------- 2. 模式 A: 指定了裁剪大小 (Center Crop Mode) ----------------
            if crop_size is not None:
                # 计算中心坐标
                _, _, h, w = self.lq.size()
                
                # 如果图片比裁剪尺寸还小，就用整张图，防止报错
                if h < crop_size or w < crop_size:
                    top, left = 0, 0
                    actual_h, actual_w = h, w
                else:
                    top = (h - crop_size) // 2
                    left = (w - crop_size) // 2
                    actual_h, actual_w = crop_size, crop_size
                
                # 裁剪 LQ (输入)
                self.lq = self.lq[..., top:top+actual_h, left:left+actual_w]
                
                # 【重要】同步裁剪 GT (如果存在)，确保 PSNR 计算区域一致
                if hasattr(self, 'gt'):
                    self.gt = self.gt[..., top*scale:(top+actual_h)*scale, left*scale:(left+actual_w)*scale]
                
                # 直接推理 (通常 256 很小，不需要切块)
                self.net_g.eval()
                with torch.no_grad():
                    if hasattr(self, 'net_g_ema'):
                        self.output = self.net_g_ema(self.lq)
                    else:
                        self.output = self.net_g(self.lq)

            # ---------------- 3. 模式 B: 全图推理 (Full Image Mode) ----------------
            else:
                # 这里就是我们之前写的防 OOM 全图逻辑
                tile_size = 512
                tile_overlap = 32
                window_size = 16 
                
                # Padding
                _, _, h, w = self.lq.size()
                mod_pad_h = (window_size - h % window_size) % window_size
                mod_pad_w = (window_size - w % window_size) % window_size
                img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
                _, _, H, W = img.size()

                # 策略判断：全图 vs 切块
                if H * W <= tile_size * tile_size * 1.5:
                    try:
                        self.net_g.eval()
                        with torch.no_grad():
                            if hasattr(self, 'net_g_ema'):
                                output = self.net_g_ema(img)
                            else:
                                output = self.net_g(img)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            output = self.tile_inference(img, tile_size, tile_overlap, scale)
                        else:
                            raise e
                else:
                    output = self.tile_inference(img, tile_size, tile_overlap, scale)

                # 裁剪掉 Padding
                _, _, h_out, w_out = output.size()
                self.output = output[:, :, 0:h_out - mod_pad_h * scale, 0:w_out - mod_pad_w * scale]
                
            self.net_g.train()

    # 别忘了保留 tile_inference 函数，全图模式需要它
    def tile_inference(self, img, tile_size, tile_overlap, scale):
        # ... (保持之前的代码不变) ...
        # 如果需要我再贴一遍 tile_inference 请告诉我
        b, c, h, w = img.size()
        current_tile_h = min(tile_size, h)
        current_tile_w = min(tile_size, w)
        stride_h = current_tile_h - tile_overlap
        stride_w = current_tile_w - tile_overlap
        if stride_h <= 0: stride_h = current_tile_h // 2
        if stride_w <= 0: stride_w = current_tile_w // 2
        h_idx_list = list(range(0, h - current_tile_h, stride_h)) + [h - current_tile_h]
        w_idx_list = list(range(0, w - current_tile_w, stride_w)) + [w - current_tile_w]
        E = torch.zeros(b, c, h*scale, w*scale).type_as(img)
        W = torch.zeros_like(E)
        self.net_g.eval()
        with torch.no_grad():
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img[..., h_idx:h_idx+current_tile_h, w_idx:w_idx+current_tile_w]
                    if hasattr(self, 'net_g_ema'):
                        out_patch = self.net_g_ema(in_patch)
                    else:
                        out_patch = self.net_g(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)
                    E[..., h_idx*scale:(h_idx+current_tile_h)*scale, w_idx*scale:(w_idx+current_tile_w)*scale].add_(out_patch)
                    W[..., h_idx*scale:(h_idx+current_tile_h)*scale, w_idx*scale:(w_idx+current_tile_w)*scale].add_(out_patch_mask)
        return E.div_(W)