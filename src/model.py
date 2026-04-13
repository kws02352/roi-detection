"""
Card/ID ROI Detection - CardDetectionViT

Architecture
------------
Input (B, 3, 180, 180)
    ↓
PatchEmbed  (patch_size=6 → 30×30 tokens)
    ↓
Stage 1 - Area Blocks (Local Attention)
    6 × AreaBlock (row/col alternating)
    - RowAreaAttention: each row attends to itself  → O(W²) per row
    - ColAreaAttention: each col attends to itself  → O(H²) per col
    ↓
Stage 2 - Global Blocks (Global Context)
    prepend cls_token → 2 × GlobalBlock
    - GlobalAttention  : full softmax O(N²)
    - LinearAttention  : ELU kernel   O(N·d²)  ← default (ONNX-friendly)
    - PerformerAttention: FAVOR+      O(N·m·d)
    ↓
Decoder
    DenseNet concat [f0, f3, f5, spatial] → dense_proj
    → PixelShuffle ×2 → ×2 → RevDWSepConv → seg_head
    → 3-channel output (card_mask, title_mask, corner_mask)
    ↓
Classification Head
    cls_token + Seg-Guided Masked Pool → cls_head
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

def posemb_sincos_2d(h: int, w: int, dim: int, temperature: float = 10000.0) -> torch.Tensor:
    """2D sincos positional embedding — (H*W, dim)."""
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D sincos"
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    omega = torch.arange(dim // 4) / (dim // 4 - 1 + 1e-6)
    omega = 1.0 / (temperature ** omega)
    y_enc = y.flatten().float().unsqueeze(1) * omega
    x_enc = x.flatten().float().unsqueeze(1) * omega
    return torch.cat([y_enc.sin(), y_enc.cos(),
                      x_enc.sin(), x_enc.cos()], dim=1)


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.act(self.norm(x))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ---------------------------------------------------------------------------
# Area Attention  (local, row/col)
# ---------------------------------------------------------------------------

class RowAreaAttention(nn.Module):
    """Each row of patch tokens attends only to tokens in the same row."""

    def __init__(self, dim: int, num_heads: int, patch_h: int, patch_w: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.patch_h, self.patch_w = patch_h, patch_w
        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H, W    = self.patch_h, self.patch_w
        x  = x.view(B, H, W, C).reshape(B * H, W, C)
        G  = x.shape[0]
        qkv = self.qkv(x).reshape(G, W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(G, W, C)
        x = self.proj(x)
        return x.view(B, H, W, C).reshape(B, N, C)


class ColAreaAttention(nn.Module):
    """Each column of patch tokens attends only to tokens in the same column."""

    def __init__(self, dim: int, num_heads: int, patch_h: int, patch_w: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.patch_h, self.patch_w = patch_h, patch_w
        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H, W    = self.patch_h, self.patch_w
        x = x.view(B, H, W, C).permute(0, 2, 1, 3).contiguous().reshape(B * W, H, C)
        G = x.shape[0]
        qkv = self.qkv(x).reshape(G, H, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(G, H, C)
        x = self.proj(x)
        return x.view(B, W, H, C).permute(0, 2, 1, 3).contiguous().reshape(B, N, C)


class AreaBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float,
                 patch_h: int, patch_w: int, direction: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn  = (RowAreaAttention(dim, num_heads, patch_h, patch_w)
                      if direction == 'row'
                      else ColAreaAttention(dim, num_heads, patch_h, patch_w))
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp   = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Global Attention  (softmax / linear / performer)
# ---------------------------------------------------------------------------

class GlobalAttention(nn.Module):
    """Standard full self-attention — O(N²·d)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class LinearAttention(nn.Module):
    """
    Linear Attention — Katharopoulos et al. (2020)
    φ(x) = elu(x) + 1  →  O(N·d²), ONNX-friendly, deterministic.

    Complexity comparison (N=901, head_dim=5):
        Softmax  : O(N²·d)  = 4.1M FLOPs
        Linear   : O(N·d²)  =  22k FLOPs  ← ~186× faster
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = self._phi(qkv[0]), self._phi(qkv[1]), qkv[2]
        kv    = torch.einsum('bhnd,bhne->bhde', k, v)
        numer = torch.einsum('bhnd,bhde->bhne', q, kv)
        k_sum = k.sum(dim=2)
        denom = torch.einsum('bhnd,bhd->bhn', q, k_sum).unsqueeze(-1).clamp(min=1e-6)
        return self.proj((numer / denom).transpose(1, 2).reshape(B, N, C))


class PerformerAttention(nn.Module):
    """
    FAVOR+ Attention — Choromanski et al. (2021)
    φ(x) = exp(x·ω/√d − ||x||²/2d) / √m
    O(N·m·d), ONNX-friendly, no custom kernels.
    """

    def __init__(self, dim: int, num_heads: int,
                 num_features: int = 64, seed: int = 42):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads    = num_heads
        self.head_dim     = dim // num_heads
        self.num_features = num_features
        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        torch.manual_seed(seed)
        self.register_buffer('omega', self._ortho_features(self.head_dim, num_features))

    @staticmethod
    def _ortho_features(d: int, m: int) -> torch.Tensor:
        blocks = []
        for _ in range(m // d):
            q, _ = torch.linalg.qr(torch.randn(d, d))
            blocks.append(q)
        rem = m % d
        if rem:
            q, _ = torch.linalg.qr(torch.randn(d, d))
            blocks.append(q[:, :rem])
        return torch.cat(blocks, dim=1)

    def _favor(self, x: torch.Tensor) -> torch.Tensor:
        proj = torch.einsum('bhnd,dm->bhnm', x, self.omega) / math.sqrt(self.head_dim)
        norm = (x * x).sum(-1, keepdim=True) / (2.0 * self.head_dim)
        return torch.exp(proj - norm) / math.sqrt(self.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q_f, k_f, v = self._favor(qkv[0]), self._favor(qkv[1]), qkv[2]
        kv    = torch.einsum('bhnm,bhnd->bhmd', k_f, v)
        numer = torch.einsum('bhnm,bhmd->bhnd', q_f, kv)
        denom = torch.einsum('bhnm,bhm->bhn', q_f, k_f.sum(2)).unsqueeze(-1).clamp(min=1e-6)
        return self.proj((numer / denom).transpose(1, 2).reshape(B, N, C))


class GlobalBlock(nn.Module):
    """
    Global attention block with pluggable attention type.

    attn_type : 'softmax'   → GlobalAttention    O(N²·d)
                'linear'    → LinearAttention     O(N·d²)  ← default
                'performer' → PerformerAttention  O(N·m·d)
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float,
                 attn_type: str = 'linear'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        if attn_type == 'linear':
            self.attn = LinearAttention(dim, num_heads)
        elif attn_type == 'performer':
            self.attn = PerformerAttention(dim, num_heads)
        else:
            self.attn = GlobalAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp   = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Decoder Blocks
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    def __init__(self, channel: int, reduction: int = 20):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        y = self.pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1)


class RevDWSepConv(nn.Module):
    """Reverse Depthwise Separable Conv: pointwise expand → depthwise → SE."""

    def __init__(self, channels: int, expand: int = 2):
        super().__init__()
        inter = channels * expand
        self.pw    = nn.Conv2d(channels, inter, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(inter)
        self.act1  = nn.GELU()
        self.dw    = nn.Conv2d(inter, channels, kernel_size=3, padding=1, groups=channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.act2  = nn.GELU()
        self.se    = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.pw(x)))
        x = self.act2(self.norm2(self.dw(x)))
        return self.se(x)


class UpsampleBlock(nn.Module):
    """Pixel-shuffle ×2 upsample block."""

    def __init__(self, in_channels: int, up_scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2,
                              kernel_size=3, padding=1)
        self.ps   = nn.PixelShuffle(up_scale)
        self.act  = nn.GELU()
        self.se   = SEBlock(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.act(self.ps(self.conv(x))))


# ---------------------------------------------------------------------------
# CardDetectionViT  (main backbone)
# ---------------------------------------------------------------------------

class CardDetectionViT(nn.Module):
    """
    Two-stage ViT backbone for card/ID ROI detection.

    Stage 1 (Local)  — 6 × AreaBlock (row/col alternating)
    Stage 2 (Global) — 2 × GlobalBlock + cls_token

    Decoder — DenseNet concat [f0, f3, f5, spatial] + PixelShuffle ×4
    Heads   — seg_head (3-ch map) + cls_head (card type)
    """

    def __init__(self, input_shape: tuple, patch_size: int, embed_dim: int,
                 num_classes: int, output_channels: int,
                 num_heads: int = 8, global_num_heads: int = None,
                 mlp_ratio: float = 2.0,
                 global_attn_type: str = 'linear'):
        super().__init__()
        img_h, img_w, in_chans = input_shape
        self.patch_h   = img_h // patch_size
        self.patch_w   = img_w // patch_size
        self.embed_dim = embed_dim
        _gnh = global_num_heads if global_num_heads is not None else num_heads

        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)
        pe = posemb_sincos_2d(self.patch_h, self.patch_w, embed_dim)
        self.register_buffer('pos_embed', pe.unsqueeze(0))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.area_blocks = nn.ModuleList([
            AreaBlock(embed_dim, num_heads, mlp_ratio,
                      self.patch_h, self.patch_w,
                      'row' if i % 2 == 0 else 'col')
            for i in range(6)
        ])

        self.global_blocks = nn.ModuleList([
            GlobalBlock(embed_dim, _gnh, mlp_ratio, attn_type=global_attn_type)
            for _ in range(2)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.dense_proj = nn.Linear(embed_dim * 4, embed_dim)
        self.upscale    = UpsampleBlock(embed_dim)
        self.upscale2   = UpsampleBlock(embed_dim)
        self.seg_refine = RevDWSepConv(embed_dim)
        self.seg_head   = nn.Conv2d(embed_dim, output_channels, kernel_size=3, padding=1)

        self.cls_head      = nn.Linear(embed_dim, num_classes)
        self.cls_pool_head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, images: torch.Tensor):
        B = images.shape[0]

        x  = self.patch_embed(images) + self.pos_embed
        f0 = x

        for i, blk in enumerate(self.area_blocks):
            x = blk(x)
            if i == 2:
                f3 = x
        f5 = x

        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        for blk in self.global_blocks:
            x = blk(x)
        x = self.norm(x)

        cls_feat = x[:, 0]
        spatial  = x[:, 1:]

        combined   = torch.cat([f0, f3, f5, spatial], dim=-1)
        spatial_2d = self.dense_proj(combined).permute(0, 2, 1) \
                         .reshape(B, self.embed_dim, self.patch_h, self.patch_w)
        spatial_2d = self.upscale(spatial_2d)
        spatial_2d = self.upscale2(spatial_2d)
        spatial_2d = self.seg_refine(spatial_2d)
        seg_out    = torch.sigmoid(self.seg_head(spatial_2d))

        seg_w  = F.interpolate(seg_out[:, 0:1].detach(),
                               size=(self.patch_h, self.patch_w),
                               mode='bilinear', align_corners=False).flatten(2)
        guided = (spatial * seg_w.transpose(1, 2)).sum(1) / (seg_w.sum(-1) + 1e-6)
        cls_out = self.cls_head(cls_feat) + self.cls_pool_head(guided)

        return seg_out, cls_out


# ---------------------------------------------------------------------------
# RoiDetectionModel  (training wrapper)
# ---------------------------------------------------------------------------

class RoiDetectionModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vit = CardDetectionViT(
            input_shape      = args.input_shape,
            patch_size       = args.patch_size,
            embed_dim        = args.embed_dim,
            num_classes      = args.num_classes,
            output_channels  = args.output_channels,
            global_attn_type = getattr(args, 'global_attn_type', 'linear'),
            global_num_heads = getattr(args, 'global_num_heads', None),
        )

    def forward(self, images: torch.Tensor):
        if torch.jit.is_tracing():
            return self.vit(images)
        if self.training:
            return checkpoint(self.vit, images, use_reentrant=False)
        return self.vit(images)


# ---------------------------------------------------------------------------
# ModelEMA
# ---------------------------------------------------------------------------

class ModelEMA:
    """
    Exponential Moving Average of model parameters.
    Stabilises inference performance during training.

    Usage
    -----
        ema = ModelEMA(model, decay=0.999)
        # after each training step:
        ema.update(model)
        # for evaluation:
        ema.module.eval()
        pred = ema.module(x)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ep, mp in zip(self.module.parameters(), model.parameters()):
            ep.data.mul_(self.decay).add_(mp.data, alpha=1.0 - self.decay)