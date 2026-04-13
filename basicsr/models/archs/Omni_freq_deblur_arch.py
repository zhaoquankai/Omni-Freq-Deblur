import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple

def _pad_to_patch_size(x: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    b, c, h, w = x.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    mode = 'reflect' if h > 1 and w > 1 else 'replicate'
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode), (h, w)

def _crop_to_original_size(x: torch.Tensor, original_size: Tuple[int, int]) -> torch.Tensor:
    h0, w0 = original_size
    return x[:, :, :h0, :w0]

def s_scan_flatten(x):
    B, C, H, W = x.shape
    return x.reshape(B, C, -1).transpose(1, 2)

def s_scan_unflatten(x, H, W):
    B, L, C = x.shape
    return x.transpose(1, 2).reshape(B, C, H, W)

try:
    from mamba_ssm import Mamba
except ImportError:
    print("Warning: mamba_ssm not found. Using nn.Identity().")
    class Mamba(nn.Module):
        def __init__(self, d_model, **kwargs): 
            super().__init__()
            self.id = nn.Identity()
        def forward(self, x, **kwargs): 
            return self.id(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, x):
        if x.ndim == 4:
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, (self.weight.shape[0],), weight=self.weight, bias=self.bias, eps=self.eps)
            x = x.permute(0, 3, 1, 2)
        else:
            x = F.layer_norm(x, (self.weight.shape[0],), weight=self.weight, bias=self.bias, eps=self.eps)
        return x

class LayerNormCLast(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, x): 
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)
    
    def forward(self, x): 
        return self.proj(x)

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Conv2d(n_feat, n_feat * 2, 3, 2, 1, bias=False)
    
    def forward(self, x): 
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False)
        )
    
    def forward(self, x): 
        return self.body(x)

class DWT(nn.Module):
    def __init__(self): 
        super().__init__()
        self.requires_grad = False
    
    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2.0
        x02 = x[:, :, 1::2, :] / 2.0
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        ll = x1 + x2 + x3 + x4
        high = torch.cat([-x1-x2+x3+x4, -x1+x2-x3+x4, x1-x2-x3+x4], dim=1)
        return ll, high

class IWT(nn.Module):
    def __init__(self): 
        super().__init__()
        self.requires_grad = False
    
    def forward(self, ll, high):
        C = ll.shape[1]
        hl, lh, hh = torch.split(high, C, dim=1)
        x1 = (ll - hl - lh + hh) / 2.0
        x2 = (ll - hl + lh - hh) / 2.0
        x3 = (ll + hl - lh - hh) / 2.0
        x4 = (ll + hl + lh + hh) / 2.0
        out = torch.zeros(ll.size(0), C, ll.size(2)*2, ll.size(3)*2, device=ll.device, dtype=ll.dtype)
        out[:,:,0::2,0::2] = x1
        out[:,:,1::2,0::2] = x2
        out[:,:,0::2,1::2] = x3
        out[:,:,1::2,1::2] = x4
        return out

class MDTA(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)

class ShiftedWindowMHSA(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8, shift_size=0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords = torch.stack(torch.meshgrid([torch.arange(window_size), torch.arange(window_size)], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1) 
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = rearrange(shifted_x, 'b (h p1) (w p2) c -> (b h w) (p1 p2) c', 
                              p1=self.window_size, p2=self.window_size)
        qkv = self.qkv(x_windows)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.num_heads).unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        ).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        x_attn = (attn @ v).transpose(1, 2).reshape(x_windows.shape[0], x_windows.shape[1], C)
        shifted_x = rearrange(x_attn, '(b h w) (p1 p2) c -> b (h p1) (w p2) c', 
                              h=Hp // self.window_size, w=Wp // self.window_size, 
                              p1=self.window_size, p2=self.window_size)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]
        x = self.proj(x)
        return x.permute(0, 3, 1, 2) 

class LHMSA(nn.Module):
    def __init__(self, dim, bias, mode='local_struct'):
        super().__init__()
        self.mode = mode
        self.dim = dim
        self.dwt = DWT()
        self.iwt = IWT()
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)
        
        self.gamma_main = nn.Parameter(torch.ones(1) * 1e-4)
        self.gamma_aux = nn.Parameter(torch.ones(1) * 0.5)
        self.gamma_cross = nn.Parameter(torch.ones(1) * 1e-5)

        if mode == 'local_struct': 
            self.op_main = MDTA(dim, num_heads=max(1, dim//32), bias=bias)
            self.norm_main = LayerNorm(dim) 
            
            self.op_aux = nn.Sequential(
                nn.Conv2d(dim*3, dim*3, 3, 1, 1, groups=dim*3, bias=bias),
                nn.GELU(),
                nn.Conv2d(dim*3, dim*3, 1, bias=bias)
            )
            
            self.cross_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim*3, dim//4, 1, bias=bias),
                nn.GELU(),
                nn.Conv2d(dim//4, dim, 1, bias=bias),
                nn.Sigmoid()
            )
            
        elif mode == 'high_freq_attention': 
            self.op_main = ShiftedWindowMHSA(dim*3, num_heads=max(1, (dim*3)//32), window_size=8, shift_size=4)
            self.norm_main = LayerNorm(dim*3)
            
            self.op_aux = nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=bias),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, bias=bias)
            )
            
            self.cross_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, (dim*3)//4, 1, bias=bias),
                nn.GELU(),
                nn.Conv2d((dim*3)//4, dim*3, 1, bias=bias),
                nn.Sigmoid()
            )

    def forward(self, x):
        ll, high = self.dwt(x)
        
        if self.mode == 'local_struct': 
            ll_norm = self.norm_main(ll)
            refined_ll = ll + self.gamma_main * self.op_main(ll_norm)
            
            refined_high = high + self.gamma_aux * self.op_aux(high)
            
            cross_mod = self.cross_gate(high)
            refined_ll = refined_ll * (1.0 + self.gamma_cross * (cross_mod - 0.5) * 2.0)
            
        elif self.mode == 'high_freq_attention': 
            high_norm = self.norm_main(high)
            refined_high = high + self.gamma_main * self.op_main(high_norm)
            
            refined_ll = ll + self.gamma_aux * self.op_aux(ll)
            
            cross_mod = self.cross_gate(ll)
            refined_high = refined_high * (1.0 + self.gamma_cross * (cross_mod - 0.5) * 2.0)
            
        else:
            refined_ll = ll
            refined_high = high
        
        out = self.iwt(refined_ll, refined_high)
        if out.shape != x.shape:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear')
        
        return self.project_out(out - x)

class ChannelAttentionFuse(nn.Module):
    def __init__(self, n_feat, reduction=8, bias=False):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_gate = nn.Sequential(
            nn.Conv2d(n_feat * 4, n_feat // reduction, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat // reduction, n_feat, 1, bias=bias),
            nn.Sigmoid()
        )
        
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat * 2, 7, padding=3, groups=n_feat * 2, bias=bias),
            nn.GELU(), 
            nn.Conv2d(n_feat * 2, 1, 1, bias=bias),
            nn.Sigmoid()
        )
        
        self.enc_align = nn.Conv2d(n_feat, n_feat, 1, bias=bias)

    def forward(self, enc, dec):
        cat_feat = torch.cat([enc, dec], dim=1)
        stats = torch.cat([self.avg_pool(cat_feat), self.max_pool(cat_feat)], dim=1)
        c_mask = self.channel_gate(stats)
        s_mask = self.spatial_gate(cat_feat)
        final_mask = c_mask * s_mask
        enc_aligned = self.enc_align(enc)
        enc_refined = enc_aligned * final_mask
        out = dec + enc_refined
        return out

class PSFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, mode='full', patch_size=8,
                 radius_low=3.0, radius_high=3.0):
        super().__init__()
        self.mode = mode
        self.patch_size = patch_size
        
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden*2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden*2, hidden*2, 3, 1, 1, groups=hidden*2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)
        
        self.W_base_real = nn.Parameter(torch.ones(hidden, 1, 1, patch_size, patch_size//2+1))
        self.W_base_imag = nn.Parameter(torch.zeros(hidden, 1, 1, patch_size, patch_size//2+1))
        
        if mode == 'full':
            self.use_boost = False
            
        elif mode == 'low_emphasis':
            self.use_boost = True
            boost_mask = self._create_low_pass_mask(patch_size, radius_low)
            self.register_buffer('boost_mask', boost_mask)
            self.channel_scale = nn.Parameter(torch.ones(hidden) * 0.5)
            self.dynamic_modulator = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden, hidden//16, 1, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden//16, 1, 1, bias=bias),
                nn.Sigmoid()
            )
            
        elif mode == 'high_emphasis':
            self.use_boost = True
            boost_mask = self._create_high_pass_mask(patch_size, radius_high)
            self.register_buffer('boost_mask', boost_mask)
            self.channel_scale = nn.Parameter(torch.ones(hidden) * 0.5)
            self.dynamic_modulator = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden, hidden//16, 1, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden//16, 1, 1, bias=bias),
                nn.Sigmoid()
            )
    
    def _create_low_pass_mask(self, patch_size, radius):
        y = torch.arange(patch_size).view(-1, 1).repeat(1, patch_size//2+1)
        x = torch.arange(patch_size//2+1).view(1, -1).repeat(patch_size, 1)
        dist = torch.sqrt(y.float()**2 + x.float()**2)
        mask = torch.exp(-dist**2 / (2 * radius**2))
        return mask.view(1, 1, 1, patch_size, patch_size//2+1)
    
    def _create_high_pass_mask(self, patch_size, radius):
        return 1.0 - self._create_low_pass_mask(patch_size, radius)
    
    def forward(self, x):
        x_in = self.project_in(x)
        x1, x2 = self.dwconv(x_in).chunk(2, dim=1)
        feat = F.gelu(x1) * x2  
        
        feat_pad, orig = _pad_to_patch_size(feat, self.patch_size)
        if feat_pad.shape[-2] == 0 or feat_pad.shape[-1] == 0:
            return self.project_out(feat)
        
        patch = rearrange(feat_pad, 'b c (h p1) (w p2) -> b c h w p1 p2',
                          p1=self.patch_size, p2=self.patch_size)
        
        fft = torch.fft.rfft2(patch.float())
        W_base = torch.complex(self.W_base_real, self.W_base_imag)
        
        if self.use_boost:
            B, C, n_h, n_w, p, p_freq = fft.shape
            base_boost = self.channel_scale.view(1, C, 1, 1, 1, 1)
            global_mod = self.dynamic_modulator(feat).view(B, 1, 1, 1, 1, 1)
            adaptive_scale = base_boost * (0.5 + global_mod)
            boost_mask_6d = self.boost_mask.view(1, 1, 1, 1, p, p_freq)
            boost_weight = 1.0 + boost_mask_6d * adaptive_scale
        else:
            boost_weight = 1.0
        
        fft = fft * W_base * boost_weight
        
        spatial = torch.fft.irfft2(fft, s=(self.patch_size, self.patch_size))
        spatial = rearrange(spatial, 'b c h w p1 p2 -> b c (h p1) (w p2)',
                            p1=self.patch_size, p2=self.patch_size)
        spatial = _crop_to_original_size(spatial, orig)
        
        return self.project_out(spatial)

class SimpleEncoderBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3.0, bias=False):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.ffn = PSFFN(dim, ffn_expansion_factor, bias, mode='full', patch_size=8)
    
    def forward(self, x):
        x = x + self.ffn(self.norm(x))
        return x

class formerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3.0, bias=False, ffn_mode='full',
                 radius_low=None, radius_high=None, mode='local_struct'):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = LHMSA(dim, bias, mode=mode)
        self.norm2 = LayerNorm(dim)
        self.ffn = PSFFN(dim, ffn_expansion_factor, bias, ffn_mode, 8,
                                     radius_low=radius_low, radius_high=radius_high)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = LayerNormCLast(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
    
    def forward(self, x_flat):
        return self.mamba(self.norm(x_flat))

class MOLS(nn.Module):
    def __init__(self, dim=192, d_state=16, d_conv=4, expand=2, overlap_ratio=0.25):
        super().__init__()
        self.dim = dim
        self.step = dim // 4
        raw_group = self.step / (1 - overlap_ratio)
        self.group_dim = int((raw_group + 7) // 8 * 8)
        self.expanded_dim = self.group_dim + 3 * self.step
        
        self.expand = nn.Conv2d(dim, self.expanded_dim, 1, bias=False)
        self.mamba_fwd = MambaBlock(self.group_dim, d_state, d_conv, expand)
        self.mamba_bwd = MambaBlock(self.group_dim, d_state, d_conv, expand)
        self.mamba_fwd_t = MambaBlock(self.group_dim, d_state, d_conv, expand)
        self.mamba_bwd_t = MambaBlock(self.group_dim, d_state, d_conv, expand)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(self.group_dim * 4, dim * 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1, bias=False)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        if H * W == 0: return x
        x_expanded = self.expand(x)
        
        x1 = x_expanded[:, 0 : self.group_dim]
        x2 = x_expanded[:, self.step : self.step + self.group_dim]
        x3 = x_expanded[:, self.step*2 : self.step*2 + self.group_dim]
        x4 = x_expanded[:, self.step*3 : self.step*3 + self.group_dim]
        
        x1_flat = s_scan_flatten(x1)
        y1_flat = self.mamba_fwd(x1_flat)
        y1 = s_scan_unflatten(y1_flat, H, W)
        
        x2_flat = s_scan_flatten(x2)
        x2_bwd = torch.flip(x2_flat, dims=[1])
        y2_bwd = self.mamba_bwd(x2_bwd)
        y2_flat = torch.flip(y2_bwd, dims=[1])
        y2 = s_scan_unflatten(y2_flat, H, W)
        
        x3_t = x3.transpose(-1, -2).contiguous()
        x3_flat = s_scan_flatten(x3_t)
        y3_flat = self.mamba_fwd_t(x3_flat)
        y3_t = s_scan_unflatten(y3_flat, W, H)
        y3 = y3_t.transpose(-1, -2).contiguous()
        
        x4_t = x4.transpose(-1, -2).contiguous()
        x4_flat = s_scan_flatten(x4_t)
        x4_bwd = torch.flip(x4_flat, dims=[1])
        y4_bwd = self.mamba_bwd_t(x4_bwd)
        y4_flat = torch.flip(y4_bwd, dims=[1])
        y4_t = s_scan_unflatten(y4_flat, W, H)
        y4 = y4_t.transpose(-1, -2).contiguous()
        
        y = torch.cat([y1, y2, y3, y4], dim=1)
        return self.fusion(y)

class LCmamba(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3.0, bias=False, att=True,
                 patch=128, ffn_mode='full', overlap_ratio=0.25):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MOLS(dim=dim, overlap_ratio=overlap_ratio) if att else nn.Identity()
        self.norm2 = LayerNorm(dim)
        self.ffn = PSFFN(dim, ffn_expansion_factor, bias, ffn_mode)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class DCFiLM(nn.Module):
    def __init__(self, source_channels, target_channels, expansion=4, gamma_scale=0.1, beta_scale=0.1):
        super().__init__()
        self.gamma_scale = gamma_scale
        self.beta_scale = beta_scale
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(source_channels, source_channels*expansion),
            nn.GELU(),
            nn.Linear(source_channels*expansion, 2*target_channels)
        )
    
    def forward(self, x):
        out = self.mlp(self.pool(x).flatten(1))
        c = out.shape[1] // 2
        gamma = 1 + self.gamma_scale * torch.tanh(out[:, :c]).view(-1, c, 1, 1)
        beta = self.beta_scale * torch.tanh(out[:, c:]).view(-1, c, 1, 1)
        return gamma, beta

class Omni_freq_deblur_arch(nn.Module):
    def __init__(self, 
                 inp_channels=3, 
                 out_channels=3, 
                 dim=48,
                 num_blocks=(6, 6, 12),
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3.0,
                 bias=False,
                 radius_low=3.0,
                 radius_high=3.0,
                 dc_expansion=4,
                 dc_gamma_scale=0.1,
                 dc_beta_scale=0.1,
                 enable_dc=True,
                 mamba_overlap_ratio=0.25):
        super().__init__()
        self.enable_dc = enable_dc
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        print(f"\n{'='*70}")
        print(f"EVSSM Final Hybrid Architecture (Level 3.0 - Heterogeneous)")
        print(f"{'='*70}")
        print(f"L2 Strategy (Low-Freq): MDTA (Global Structure)")
        print(f"L1 Strategy (High-Freq): Shifted Window MHSA (Local Texture)")
        print(f"{'='*70}\n")

        self.encoder_level1 = nn.Sequential(*[
            SimpleEncoderBlock(dim, ffn_expansion_factor, bias)
            for _ in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(dim)
        
        self.encoder_level2 = nn.Sequential(*[
            SimpleEncoderBlock(dim*2, ffn_expansion_factor, bias)
            for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(dim*2)

        self.encoder_level3 = nn.Sequential(*[
            LCmamba(dim*4, ffn_expansion_factor, bias, att=True, patch=96, ffn_mode='full',
                overlap_ratio=mamba_overlap_ratio)
            for _ in range(num_blocks[2])
        ])

        self.decoder_level3 = nn.Sequential(*[
            LCmamba(dim*4, ffn_expansion_factor, bias, att=True, patch=96, ffn_mode='full',
                overlap_ratio=mamba_overlap_ratio)
            for _ in range(num_blocks[2])
        ])
        self.up3_2 = Upsample(dim * 4)

        if enable_dc:
            self.dc_film = DCFiLM(dim, dim*4, expansion=dc_expansion,
                                   gamma_scale=dc_gamma_scale, beta_scale=dc_beta_scale)

        self.fuse2 = ChannelAttentionFuse(dim * 2, reduction=16, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            formerBlock(dim*2, ffn_expansion_factor, bias, ffn_mode='low_emphasis',
                           radius_low=radius_low, mode='local_struct')
            for _ in range(num_blocks[1])
        ])
        self.up2_1 = Upsample(dim * 2)

        self.fuse1 = ChannelAttentionFuse(dim, reduction=8, bias=bias)
        self.decoder_level1 = nn.Sequential(*[
            formerBlock(dim, ffn_expansion_factor, bias, ffn_mode='high_emphasis',
                           radius_high=radius_high, mode='high_freq_attention')
            for _ in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            formerBlock(dim, ffn_expansion_factor, bias, ffn_mode='high_emphasis',
                           radius_high=radius_high, mode='high_freq_attention')
            for _ in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=bias)

    def forward(self, inp_img):
        l1_in = self.patch_embed(inp_img)
        l1_enc = self.encoder_level1(l1_in)
        
        l2_in = self.down1_2(l1_enc)
        l2_enc = self.encoder_level2(l2_in)
        
        l3_in = self.down2_3(l2_enc)
        l3_enc = self.encoder_level3(l3_in)

        l3_dec = self.decoder_level3(l3_enc)
        
        if self.enable_dc:
            gamma, beta = self.dc_film(l1_enc)
            l3_dec = l3_dec * gamma + beta

        l2_dec = self.decoder_level2(self.fuse2(l2_enc, self.up3_2(l3_dec)))
        l1_dec = self.decoder_level1(self.fuse1(l1_enc, self.up2_1(l2_dec)))
        l1_dec = self.refinement(l1_dec)

        return self.output(l1_dec) + inp_img

