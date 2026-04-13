import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast # 引入 autocast
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if weight is not None:
            loss = loss * weight
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss * self.loss_weight


class EdgeLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.register_buffer('kernel', torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1))
        self.weight = loss_weight

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')

        return F.conv2d(img, self.kernel.float(), groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y, weight=None, **kwargs):

        with autocast(enabled=False):
            x = x.float()
            y = y.float()
            loss = l1_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss * self.weight


class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):

        with autocast(enabled=False):
            pred = pred.float()
            target = target.float()
            pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
            pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
            target_fft = torch.fft.fft2(target, dim=(-2, -1))
            target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
            loss = l1_loss(pred_fft, target_fft, weight, reduction=self.reduction)
        return self.loss_weight * loss


class MSELoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super().__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False
            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            pred, target = pred / 255., target / 255.
        return self.loss_weight * self.scale * torch.log(
            ((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.reshape(1, 1, size)


def _gaussian_filter(input, win):
    C = input.shape[1]
    out = F.conv2d(input, win[..., None].expand(C, 1, -1, 1), groups=C, padding=(win.shape[-1] // 2, 0))
    out = F.conv2d(out, win[..., None, :].expand(C, 1, 1, -1), groups=C, padding=(0, win.shape[-1] // 2))
    return out


def _ssim(pred, target, win, data_range=1.0, size_average=True):
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = _gaussian_filter(pred, win)
    mu2 = _gaussian_filter(target, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _gaussian_filter(pred * pred, win) - mu1_sq
    sigma2_sq = _gaussian_filter(target * target, win) - mu2_sq
    sigma12 = _gaussian_filter(pred * target, win) - mu1_mu2

    v1 = 2 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.flatten(1).mean(-1)


def _msssim(pred, target, data_range=1.0, size_average=True, win_size=11, win_sigma=1.5, weights=None):
    if weights is None:
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
                               device=pred.device, dtype=pred.dtype)
    win = _fspecial_gauss_1d(win_size, win_sigma).to(pred.device, dtype=pred.dtype)
    levels = weights.shape[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        ssim_map = _ssim(pred, target, win=win, data_range=data_range, size_average=False)
        cs_map = _ssim(pred, target, win=win, data_range=data_range, size_average=False)
        mssim.append(ssim_map)
        mcs.append(cs_map)
        pred = F.avg_pool2d(pred, kernel_size=2, padding=0)
        target = F.avg_pool2d(target, kernel_size=2, padding=0)

    mssim = torch.stack(mssim, dim=0)
    mcs = torch.stack(mcs, dim=0)
    pow1 = mcs[:-1] ** weights[:-1].unsqueeze(1)
    pow2 = mssim[-1] ** weights[-1]
    score = torch.prod(pow1, dim=0) * pow2
    if size_average:
        return score.mean()
    else:
        return score


class MSSSIMLoss(nn.Module):
    """(1 - MS-SSIM) loss, expects input in [0,1]."""
    def __init__(self, loss_weight=1.0, channel=3):
        super().__init__()
        self.loss_weight = loss_weight
        self.channel = channel

    def forward(self, pred, target, **kwargs):

        with autocast(enabled=False):
            pred = pred.float().clamp(0, 1)
            target = target.float().clamp(0, 1)
            loss = self.loss_weight * (1.0 - _msssim(pred, target, data_range=1.0, size_average=True))
        return loss