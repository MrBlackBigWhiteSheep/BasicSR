import random
from collections import OrderedDict
from copy import deepcopy
from os import path as osp

import torch
from torch.nn import functional as F

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


class MixingAugment:
    """Simple mixup for paired restoration."""

    def __init__(self, mixup_beta: float, use_identity: bool, device: torch.device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device
        self.use_identity = use_identity

    def __call__(self, target: torch.Tensor, input_: torch.Tensor):
        # optionally keep identity (probability 1/(n+1))
        if self.use_identity and random.randint(0, 1) == 0:
            return target, input_

        lam = self.dist.rsample((1, 1)).item()
        r_index = torch.randperm(target.size(0)).to(self.device)
        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]
        return target, input_


@MODEL_REGISTRY.register()
class RestormerModel(BaseModel):
    """Restormer image restoration model with optional mixup and windowed testing."""

    def __init__(self, opt):
        super().__init__(opt)

        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    # ----------------- training ----------------- #
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # mixup augmentation
        mixing_opt = train_opt.get('mixing_augs', {})
        self.mixing_flag = mixing_opt.get('mixup', False)
        if self.mixing_flag:
            mixup_beta = mixing_opt.get('mixup_beta', 1.2)
            use_identity = mixing_opt.get('use_identity', False)
            self.mixing_augmentation = MixingAugment(mixup_beta, use_identity, self.device)

        self.use_grad_clip = train_opt.get('use_grad_clip', False)
        self.grad_clip_norm = train_opt.get('grad_clip_norm', 0.01)

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        # losses
        self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device) if train_opt.get('pixel_opt') else None
        self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device) if train_opt.get('perceptual_opt') else None
        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # optimizers & schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                get_root_logger().warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.is_train and getattr(self, 'mixing_flag', False):
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_pix:
            l_pix = 0.0
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt)
            loss_dict['l_pix'] = l_pix
            l_total += l_pix

        if self.cri_perceptual:
            l_percep = 0.0
            l_style = 0.0
            for pred in preds:
                lp, ls = self.cri_perceptual(pred, self.gt)
                if lp is not None:
                    l_percep = l_percep + lp
                if ls is not None:
                    l_style = l_style + ls
            if l_percep:
                loss_dict['l_percep'] = l_percep
                l_total += l_percep
            if l_style:
                loss_dict['l_style'] = l_style
                l_total += l_style

        l_total.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.grad_clip_norm)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    # ----------------- inference / validation ----------------- #
    def _infer(self, img):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            self.net_g.train()

        if isinstance(pred, list):
            pred = pred[-1]
        return pred

    def test(self):
        val_opt = self.opt.get('val', {})
        window_size = val_opt.get('window_size', 0)
        pad_multiple = val_opt.get('pad_multiple', 8)
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0

        # 如果未指定 window_size，也保证尺寸对齐 pad_multiple（默认 8，对应 Restormer 下采样因子）
        effective = window_size if window_size else pad_multiple

        if effective:
            _, _, h, w = self.lq.size()
            if h % effective != 0:
                mod_pad_h = effective - h % effective
            if w % effective != 0:
                mod_pad_w = effective - w % effective
            img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        else:
            img = self.lq

        pred = self._infer(img)

        if effective and (mod_pad_h or mod_pad_w):
            _, _, h, w = pred.size()
            pred = pred[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

        self.output = pred

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if use_pbar:
            from tqdm import tqdm
            pbar = tqdm(total=len(dataloader), unit='image')

        num_samples = 0
        for idx, val_data in enumerate(dataloader):
            num_samples += 1
            metric_data = dict()
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}.png')
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics and 'img2' in metric_data:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            del self.lq
            del self.output
            if 'img2' in metric_data:
                del metric_data['img2']
            torch.cuda.empty_cache()

        if use_pbar:
            pbar.close()

        if with_metrics and num_samples > 0:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= num_samples
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    # ----------------- visuals / save ----------------- #
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'
            if tb_logger:
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
        logger = get_root_logger()
        logger.info(log_str)

