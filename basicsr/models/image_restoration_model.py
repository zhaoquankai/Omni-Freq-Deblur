import importlib
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import math
import gc 

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


def create_window(window_size, channel):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim_torch(img1, img2, window_size=11, window=None, size_average=True):
    val_range = 1.0
    padd = window_size // 2
    (_, channel, height, width) = img1.size()
    
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel).to(img1.device).type(img1.dtype)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def psnr_torch(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


class ImageRestorationModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel, self).__init__(opt)

        self.net_g = define_network(deepcopy(opt['network_g']))

        self.net_g = self.model_to_device(self.net_g)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(
                self.net_g,
                load_path,
                self.opt['path'].get('strict_load_g', True),
                param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])
        self.ssim_window = None


    def model_to_device(self, network):
        network = network.to(self.device)
        if self.opt['dist']:
            network = torch.nn.parallel.DistributedDataParallel(
                network, 
                device_ids=[torch.cuda.current_device()], 
                find_unused_parameters=True
            )
        return network

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']


        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None


        if train_opt.get('fft_loss_opt'):
            fft_type = train_opt['fft_loss_opt'].pop('type')
            cri_fft_cls = getattr(loss_module, fft_type)
            self.cri_fft = cri_fft_cls(**train_opt['fft_loss_opt']).to(self.device)
        else:
            self.cri_fft = None

        if train_opt.get('edge_opt'):
            edge_type = train_opt['edge_opt'].pop('type')
            cri_edge_cls = getattr(loss_module, edge_type)
            self.cri_edge = cri_edge_cls(**train_opt['edge_opt']).to(self.device)
        else:
            self.cri_edge = None


        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None


        if train_opt.get('msssim_opt'):
            msssim_type = train_opt['msssim_opt'].pop('type')
            cri_msssim_cls = getattr(loss_module, msssim_type)
            self.cri_msssim = cri_msssim_cls(**train_opt['msssim_opt']).to(self.device)
        else:
            self.cri_msssim = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_msssim is None:
            raise ValueError('All losses are None.')

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                 **train_opt['optim_g'])
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)


    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()


        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if self.opt['train'].get('mixup', False):
                self.mixup_aug()

            lq1, lq2 = self.lq.chunk(2, dim=0)
            gt1, gt2 = self.gt.chunk(2, dim=0)

            preds = self.net_g(lq1)
            if not isinstance(preds, list):
                preds = [preds]
            self.output = preds[-1]

            l_total = 0
            loss_dict = OrderedDict()

            if self.cri_pix:
                l_pix = sum(self.cri_pix(pred, gt1) for pred in preds)
                l_total += l_pix
                loss_dict['l_pix'] = l_pix

            if self.cri_fft:
                with torch.cuda.amp.autocast(enabled=False):
                    pred_fft = preds[-1].float().clamp(0, 1)
                    gt_fft = gt1.float().clamp(0, 1)
                    l_fft = self.cri_fft(pred_fft, gt_fft)
                l_total += l_fft
                loss_dict['l_fft'] = l_fft

            if self.cri_edge:
                l_edge = self.cri_edge(preds[-1], gt1)
                l_total += l_edge
                loss_dict['l_edge'] = l_edge


            if self.cri_msssim:
                with torch.cuda.amp.autocast(enabled=False):
                    pred32 = preds[-1].float().clamp(0, 1)
                    gt32 = gt1.float().clamp(0, 1)
                    l_msssim = self.cri_msssim(pred32, gt32)
                l_total += l_msssim
                loss_dict['l_msssim'] = l_msssim

            l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())
            l_total = l_total / 2
        l_total.backward()


        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            preds = self.net_g(lq2)
            if not isinstance(preds, list):
                preds = [preds]
            self.output = preds[-1]

            l_total = 0

            if self.cri_pix:
                l_pix = sum(self.cri_pix(pred, gt2) for pred in preds)
                l_total += l_pix
                loss_dict['l_pix'] = l_pix

            if self.cri_fft:
                with torch.cuda.amp.autocast(enabled=False):
                    pred_fft = preds[-1].float().clamp(0, 1)
                    gt_fft = gt2.float().clamp(0, 1) 
                    l_fft = self.cri_fft(pred_fft, gt_fft)
                l_total += l_fft
                loss_dict['l_fft'] = l_fft

            if self.cri_edge:
                l_edge = self.cri_edge(preds[-1], gt2)
                l_total += l_edge
                loss_dict['l_edge'] = l_edge


            if self.cri_msssim:
                with torch.cuda.amp.autocast(enabled=False):
                    pred32 = preds[-1].float().clamp(0, 1)
                    gt32 = gt2.float().clamp(0, 1) 
                    l_msssim = self.cri_msssim(pred32, gt32)
                l_total += l_msssim
                loss_dict['l_msssim'] = l_msssim

            l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())
            l_total = l_total / 2

        l_total.backward()

        if self.opt['train'].get('use_grad_clip', True):
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):

        original_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        
        self.net_g.eval()
        try:

            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                n = len(self.lq)
                outs = []
                m = self.opt['val'].get('max_minibatch', n)
                i = 0
                while i < n:
                    j = i + m
                    if j >= n:
                        j = n
                    

                    lq_batch = self.lq[i:j].contiguous()
                    b, c, h, w = lq_batch.size()

                    factor = 64
                    h_pad = (math.ceil(h / factor) * factor) - h
                    w_pad = (math.ceil(w / factor) * factor) - w
                    
                    if h_pad != 0 or w_pad != 0:
                        img_padded = F.pad(lq_batch, (0, w_pad, 0, h_pad), mode='replicate')
                    else:
                        img_padded = lq_batch

                    img_padded = img_padded.contiguous()
                    

                    torch.cuda.synchronize()


                    pred = self.net_g(img_padded)
                    
                    if isinstance(pred, list):
                        pred = pred[-1]


                    pred = pred[..., :h, :w]
                    
                    outs.append(pred)
                    i = j
                
                self.output = torch.cat(outs, dim=0)
        finally:

            torch.backends.cudnn.benchmark = original_benchmark
            self.net_g.train()


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}

        rank, world_size = get_dist_info()
        
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0
        if self.ssim_window is None:
            self.ssim_window = create_window(11, 3).to(self.device)

        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data, is_val=True)
            

            self.test()


            if with_metrics:
  
                sr_tensor = self.output.float().clamp(0, 1)
                gt_tensor = self.gt.float().clamp(0, 1)
                
                if 'psnr' in self.metric_results:
                    self.metric_results['psnr'] += psnr_torch(sr_tensor, gt_tensor).item()
                if 'ssim' in self.metric_results:
                    self.metric_results['ssim'] += ssim_torch(sr_tensor, gt_tensor, window=self.ssim_window).item()

            if save_img:
                visuals = self.get_current_visuals()
                sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
                if 'gt' in visuals:
                    gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                    del self.gt

                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                             f'{img_name}.png')
                imwrite(sr_img, save_img_path)

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        
        if rank == 0:
            pbar.close()

        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)
            self.collected_metrics = collected_metrics

        keys, metrics = [], []
        if hasattr(self, 'collected_metrics'):
            for name, value in self.collected_metrics.items():
                keys.append(name)
                metrics.append(value)
            metrics = torch.stack(metrics, 0)
            torch.distributed.reduce(metrics, dst=0)

            if self.opt['rank'] == 0:
                metrics_dict = {}
                cnt_val = 0
                for key, metric in zip(keys, metrics):
                    if key == 'cnt':
                        cnt_val = float(metric)
                        continue
                    metrics_dict[key] = float(metric)

                if cnt_val > 0:
                    for key in metrics_dict:
                        metrics_dict[key] /= cnt_val

                self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                                  tb_logger, metrics_dict)
        

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        torch.cuda.empty_cache()
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}

        pbar = tqdm(total=len(dataloader), unit='image')
        cnt = 0
        if self.ssim_window is None:
            self.ssim_window = create_window(11, 3).to(self.device)

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data, is_val=True)
            
            self.test()

            if with_metrics:
                sr_tensor = self.output.float().clamp(0, 1)
                gt_tensor = self.gt.float().clamp(0, 1)
                
                if 'psnr' in self.metric_results:
                    self.metric_results['psnr'] += psnr_torch(sr_tensor, gt_tensor).item()
                if 'ssim' in self.metric_results:
                    self.metric_results['ssim'] += ssim_torch(sr_tensor, gt_tensor, window=self.ssim_window).item()

            if save_img:
                visuals = self.get_current_visuals()
                sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
                if 'gt' in visuals:
                    gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                    del self.gt
                
                if self.opt['is_train']:
                      save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                 f'{img_name}_{current_iter}.png')
                else:
                      save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')
                imwrite(sr_img, save_img_path)

            del self.lq
            del self.output
            torch.cuda.empty_cache()
            
            cnt += 1
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            metrics_dict = {}
            for key, metric in self.metric_results.items():
                metrics_dict[key] = metric
            if cnt > 0:
                for key in metrics_dict:
                    metrics_dict[key] /= cnt
            self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                     tb_logger, metrics_dict)
        

        gc.collect()
        torch.cuda.empty_cache()
        return 0.

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value
        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)