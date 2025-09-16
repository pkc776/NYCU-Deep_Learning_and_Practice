# unet_model.py
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler

class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for scalar time t -> vector dim `dim`.
    Produces (B, dim).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1).float()  
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / float(half)
        )  
        args = t[:, None] * freqs[None, :] 
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), value=0.0)
        emb = self.proj(emb)  
        return emb

class AdaptiveGroupNorm(nn.Module):
    """
    Adaptive GroupNorm that produces per-channel scale and shift from conditioning vectors
    (time embedding and optional external conditioning like multi-hot).

    Applies: out = GN(x) * (1 + gamma) + beta
    where [gamma, beta] are produced by a small linear projection from [time_emb, cond_emb].
    """
    def __init__(self, num_groups: int, num_channels: int, time_emb_dim: int, cond_dim: Optional[int] = None, eps: float = 1e-6):
        super().__init__()
        self.num_channels = num_channels
        self.base_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps)
        in_dim = time_emb_dim + (cond_dim if cond_dim is not None and cond_dim > 0 else 0)
        self.proj = nn.Linear(in_dim, num_channels * 2)
        self.act = nn.SiLU()
        self.cond_dim = cond_dim

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, cond_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        if self.cond_dim is not None and cond_emb is not None:
            inp = torch.cat([t_emb, cond_emb], dim=-1)
        else:
            inp = t_emb
        params = self.proj(self.act(inp))  
        params = params.view(B, 2, self.num_channels)
        gamma = params[:, 0].view(B, self.num_channels, 1, 1)
        beta = params[:, 1].view(B, self.num_channels, 1, 1)
        out = self.base_norm(x)
        out = out * (1.0 + gamma) + beta
        return out


class ResidualBlock(nn.Module):
    """
    Residual block with time conditioning and optional additional conditioning vector.
    GroupNorm -> SiLU -> Conv -> add time (+ cond) -> GroupNorm -> SiLU -> Dropout -> Conv -> skip
    """
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int,
                 groups: int = 8, dropout: float = 0.0, cond_dim: Optional[int] = None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        # make the first norm adaptive as well so conditioning can modulate inputs before conv1
        self.norm1 = AdaptiveGroupNorm(num_groups=groups, num_channels=in_ch, time_emb_dim=time_emb_dim, cond_dim=cond_dim)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        # project time embedding to out_ch and add (broadcast)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        # optional condition projection (e.g., one-hot / multi-hot)
        self.cond_dim = cond_dim
        if cond_dim is not None and cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, out_ch)
        else:
            self.cond_proj = None
        # replace plain GroupNorm with AdaptiveGroupNorm to make normalization conditional-aware
        self.norm2 = AdaptiveGroupNorm(num_groups=groups, num_channels=out_ch, time_emb_dim=time_emb_dim, cond_dim=cond_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, cond_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, in_ch, H, W)
        t_emb: (B, time_emb_dim)
        cond_emb: optional (B, cond_dim)
        returns: (B, out_ch, H, W)
        """
        h = self.conv1(self.act(self.norm1(x, t_emb, cond_emb)))
        # time conditioning
        t = self.time_proj(self.act(t_emb))  # (B, out_ch)
        h = h + t[:, :, None, None]
        # optional extra conditioning (e.g., multi-hot one-hot vector)
        if cond_emb is not None and self.cond_proj is not None:
            c = self.cond_proj(self.act(cond_emb))  # (B, out_ch)
            h = h + c[:, :, None, None]
        # AdaptiveGroupNorm requires t_emb and cond_emb
        h = self.conv2(self.dropout(self.act(self.norm2(h, t_emb, cond_emb))))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    Multi-head self attention over spatial locations.
    Norm is adaptive so conditioning and time can modulate attention inputs.
    Input: (B, C, H, W) -> returns same shape. Forward accepts optional t_emb and cond_emb.
    """
    def __init__(self, n_channels: int, n_heads: int = 4, d_k: Optional[int] = None, norm_groups: int = 8, time_emb_dim: Optional[int] = None, cond_dim: Optional[int] = None):
        super().__init__()
        self.n_channels = n_channels
        self.n_heads = n_heads
        # choose d_k per head; default evenly split
        if d_k is None:
            assert n_channels % n_heads == 0, "n_channels must be divisible by n_heads when d_k is None"
            d_k = n_channels // n_heads
        self.d_k = d_k
        # use adaptive norm so we can modulate attention with condition/time
        if time_emb_dim is not None:
            self.norm = AdaptiveGroupNorm(num_groups=norm_groups, num_channels=n_channels, time_emb_dim=time_emb_dim, cond_dim=cond_dim)
        else:
            self.norm = nn.GroupNorm(num_groups=norm_groups, num_channels=n_channels, eps=1e-6)
        # qkv projection (applied to channel dim)
        self.qkv = nn.Linear(n_channels, n_heads * d_k * 3)
        self.out = nn.Linear(n_heads * d_k, n_channels)

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None, cond_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, C, H, W)
        t_emb: optional (B, time_emb_dim)
        cond_emb: optional (B, cond_dim)
        """
        B, C, H, W = x.shape
        seq_len = H * W
        # normalize in channel space (adaptive if available)
        if isinstance(self.norm, AdaptiveGroupNorm):
            xn = self.norm(x, t_emb, cond_emb)  # (B, C, H, W)
        else:
            xn = self.norm(x)
        xn = xn.view(B, C, seq_len).permute(0, 2, 1)  # (B, seq, C)
        qkv = self.qkv(xn)  # (B, seq, 3 * n_heads * d_k)
        qkv = qkv.reshape(B, seq_len, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, n_heads, seq, d_k)

        scale = 1.0 / math.sqrt(self.d_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_heads, seq, seq)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, n_heads, seq, d_k)
        out = out.permute(0, 2, 1, 3).reshape(B, seq_len, self.n_heads * self.d_k)  # (B, seq, C)
        out = self.out(out)  # (B, seq, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out + x  # residual


class Downsample(nn.Module):
    """Conv downsample by factor 2"""
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """ConvTranspose upsample by factor 2"""
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class UNet(nn.Module):
    """
    UNet variant for diffusion / conditional image generation.
    Added `cond_dim` to support one-hot / multi-hot conditioning vectors.
    """
    def __init__(self,
                 image_channels: int = 3,
                 base_channels: int = 32,
                 channel_mults: Optional[List[int]] = None,
                 num_res_blocks: int = 2,
                 time_emb_factor: int = 4,
                 num_heads: int = 4,
                 attn_at: Optional[List[int]] = None,
                 dropout: float = 0.0,
                 cond_dim: int = 0):
        super().__init__()
        if channel_mults is None:
            channel_mults = [1, 2, 4, 8]
        if attn_at is None:
            attn_at = [2]  # default: apply attention at 3rd stage (indexing from 0)
        self.image_proj = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)
        self.base_channels = base_channels
        self.time_dim = base_channels * time_emb_factor
        self.time_emb = SinusoidalEmbedding(self.time_dim)
        # conditioning dimension (e.g., one-hot 24d). If 0 -> no conditioning
        self.cond_dim = cond_dim if cond_dim and cond_dim > 0 else None

        # build down path
        self.downs = nn.ModuleList()
        in_ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            stage = nn.ModuleList()
            for r in range(num_res_blocks):
                stage.append(ResidualBlock(in_ch, out_ch, self.time_dim, dropout=dropout, cond_dim=self.cond_dim))
                in_ch = out_ch
            # optional attention block at this stage
            if i in attn_at:
                stage.append(AttentionBlock(out_ch, n_heads=num_heads, time_emb_dim=self.time_dim, cond_dim=self.cond_dim))
            # downsample if not last
            if i != len(channel_mults) - 1:
                stage.append(Downsample(out_ch))
            self.downs.append(stage)

        # middle / bottleneck
        self.mid = nn.ModuleList([
            ResidualBlock(in_ch, in_ch, self.time_dim, dropout=dropout, cond_dim=self.cond_dim),
            AttentionBlock(in_ch, n_heads=num_heads, time_emb_dim=self.time_dim, cond_dim=self.cond_dim),
            ResidualBlock(in_ch, in_ch, self.time_dim, dropout=dropout, cond_dim=self.cond_dim),
        ])

        # build up path (reverse)
        self.ups = nn.ModuleList()
        for i, mult in list(enumerate(reversed(channel_mults))):
            out_ch = base_channels * mult
            stage = nn.ModuleList()
            for r in range(num_res_blocks):
                if r == 0:
                    stage.append(ResidualBlock(in_ch + out_ch, out_ch, self.time_dim, dropout=dropout, cond_dim=self.cond_dim))
                else:
                    stage.append(ResidualBlock(out_ch, out_ch, self.time_dim, dropout=dropout, cond_dim=self.cond_dim))
            # potential attention
            down_stage_idx = len(channel_mults) - 1 - i
            if down_stage_idx in attn_at:
                stage.append(AttentionBlock(out_ch, n_heads=num_heads, time_emb_dim=self.time_dim, cond_dim=self.cond_dim))
            # upsample if not last
            if i != len(channel_mults) - 1:
                stage.append(Upsample(out_ch))
            in_ch = out_ch
            self.ups.append(stage)

        # final normalization and projection back to image channels
        # make final norm adaptive if model is conditional
        if self.cond_dim is not None:
            self.final_norm = AdaptiveGroupNorm(num_groups=8, num_channels=in_ch, time_emb_dim=self.time_dim, cond_dim=self.cond_dim)
        else:
            self.final_norm = nn.GroupNorm(num_groups=8, num_channels=in_ch, eps=1e-6)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(in_ch, image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, C_in, H, W)
        t: (B,) or (B,1)
        cond: optional conditioning tensor of shape (B, cond_dim) - can be multi-hot
        returns: (B, C_in, H, W)
        """
        B = x.shape[0]
        t_emb = self.time_emb(t)  # (B, time_dim)

        # if cond provided, expect shape (B, cond_dim)
        if cond is not None:
            if self.cond_dim is None:
                raise ValueError("Model was not initialized with cond_dim but cond was passed to forward()")
            if cond.dim() == 1:
                cond = cond.view(B, -1)

        h = self.image_proj(x)
        skips = []

        # down path
        for stage in self.downs:
            for module in stage:
                if isinstance(module, ResidualBlock):
                    h = module(h, t_emb, cond)
                elif isinstance(module, AttentionBlock):
                    h = module(h, t_emb, cond)
                elif isinstance(module, Downsample):
                    skips.append(h)
                    h = module(h)
                else:
                    # unknown module: just call
                    h = module(h)
            if not any(isinstance(m, Downsample) for m in stage):
                skips.append(h)

        # middle
        for m in self.mid:
            if isinstance(m, ResidualBlock):
                h = m(h, t_emb, cond)
            elif isinstance(m, AttentionBlock):
                h = m(h, t_emb, cond)
            else:
                h = m(h)

        # up path
        for stage in self.ups:
            used_skip = False
            for module in stage:
                if isinstance(module, ResidualBlock):
                    # only the first ResidualBlock in this stage consumes a skip connection
                    if (not used_skip) and skips:
                        s = skips.pop()
                        # ensure spatial sizes match (upsample h if needed)
                        if h.shape[2:] != s.shape[2:]:
                            h = F.interpolate(h, size=s.shape[2:], mode='nearest')
                        h = torch.cat([h, s], dim=1)
                        used_skip = True
                    h = module(h, t_emb, cond)
                elif isinstance(module, AttentionBlock):
                    h = module(h, t_emb, cond)
                elif isinstance(module, Upsample):
                    h = module(h)
                else:
                    h = module(h)

        # final norm may be adaptive; call accordingly
        if isinstance(self.final_norm, AdaptiveGroupNorm):
            hn = self.final_norm(h, t_emb, cond)
        else:
            hn = self.final_norm(h)
        out = self.final_conv(self.final_act(hn))
        return out


# Diffusion / DDPM helper using Hugging Face diffusers

class DiffusionDDPM:
    """
    Minimal DDPM helper that uses a provided UNet-like model and
    Hugging Face's `DDPMScheduler` to (1) add noise during training and
    (2) run the reverse sampling loop for generation.
    """

    def __init__(self, model: nn.Module, num_train_timesteps: int = 1000, device: Optional[str] = None,
                 scheduler: Optional[object] = None):
        if DDPMScheduler is None:
            raise ImportError("diffusers not found. Install with `pip install diffusers` to use DiffusionDDPM`.")
        self.model = model
        self.device = device or (next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else "cpu")
        self.num_train_timesteps = num_train_timesteps
        self.scheduler = scheduler or DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule="linear")

    def add_noise(self, clean_images: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Add Gaussian noise to clean_images according to the scheduler at given timesteps.
        Returns (noisy_images, noise)
        timesteps: tensor of shape (B,) with ints in [0, num_train_timesteps-1]
        """
        if noise is None:
            noise = torch.randn_like(clean_images)
        noisy = self.scheduler.add_noise(clean_images, noise, timesteps)
        return noisy, noise

# end of diffusion helper