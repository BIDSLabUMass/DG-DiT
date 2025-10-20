import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_, to_2tuple

################################################################################
# Helper blocks
################################################################################

class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """Split x (B,H,W,C) into non‑overlapping windows of size window_size×window_size."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Inverse of window_partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi‑head self‑attention (W‑MSA) with relative positional bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Pair‑wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2,Wh,Ww
        coords_flat = torch.flatten(coords, 1)                      # 2, Wh*Ww
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()      # Wh*Ww,Wh*Ww,2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)                     # Wh*Ww,Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        """x: (num_windows*B, N, C) where N = window_size*window_size"""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)      # 3, B_, heads, N, dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))                 # B_, heads, N, N

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # heads,N,N
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # B_,N,C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """A basic Swin Transformer block (with optional window shift)."""

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size if min(input_resolution) > window_size else 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, drop)
        # self.kernel_attn = nn.Sequential(*[
        #     nn.Conv2d(2, 2, 3, 1, 1),
        #     nn.GELU(),
        #     nn.Flatten(),
        #     nn.Linear(latent_size, dim * 2, bias=False)
        # ])
        # self.kernel_ffn = nn.Sequential(*[
        #     nn.Conv2d(2, 2, 3, 1, 1),
        #     nn.GELU(),
        #     nn.Flatten(),
        #     nn.Linear(latent_size, dim * 2, bias=False)
        # ])

        # pre‑compute attention mask for SW‑MSA
        if self.shift_size > 0:
            H, W = input_resolution
            img_mask = torch.zeros((1, H, W, 1))          # 1 H W 1
            h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt; cnt += 1
            mask_windows = window_partition(img_mask, window_size)  # nW,win,win,1
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # z_attn = self.kernel_attn(z).view(B, 1, 1, self.dim * 2)
        # z1_attn, z2_attn = z_attn.chunk(2, 3)
        # x = x * z1_attn + z2_attn

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)               # nW*B,win,win,C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B,N,C

        # W‑MSA / SW‑MSA
        attn_windows = self.attn(x_windows) if self.attn_mask is None else self.attn(x_windows + 0.0)  # mask is baked in

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)        # B,H,W,C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)

        # z_ffn = self.kernel_ffn(z).view(B, 1, self.dim * 2)
        # z1_ffn, z2_ffn = z_ffn.chunk(2, 2)
        # x = x * z1_ffn + z2_ffn

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    """A sequence of SwinTransformerBlocks (no down‑sampling)."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0., use_checkpoint=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, input_resolution, num_heads, window_size,
                                  shift_size=0 if (i % 2 == 0) else window_size // 2,
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  drop=drop, attn_drop=attn_drop,
                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)])
        self.use_checkpoint = use_checkpoint

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        return x


class PatchEmbed(nn.Module):
    """Flatten (patch_size=1) feature map to sequence."""
    def __init__(self, img_size=192, patch_size=1, in_chans=64, embed_dim=64):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.proj = nn.Identity()  # no conv when patch_size==1

    def forward(self, x):  # x: B,C,H,W
        return x.flatten(2).transpose(1, 2)  # B,H*W,C


class PatchUnEmbed(nn.Module):
    """Recover (H,W) feature map from sequence."""
    def __init__(self, img_size=192, patch_size=1, embed_dim=64):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        return x.transpose(1, 2).view(B, C, x_size[0], x_size[1])


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class LatentProjector(nn.Module):
    """Upsample + conv stack to map z (B, z_chans, 24,24) -> (B, embed_dim, 192,192)."""
    def __init__(self, z_chans: int, embed_dim: int, up_factor=8):
        super().__init__()
        self.proj = nn.Sequential(*[
            nn.Conv2d(z_chans, embed_dim, 3, 1, 1),
            nn.GELU(),
            Upsample(embed_dim, with_conv=True),
            nn.GELU(),
            Upsample(embed_dim, with_conv=False),
            Upsample(embed_dim, with_conv=True),
            nn.GELU(),
        ])

    def forward(self, z):
        return self.proj(z)


class GateFusion(nn.Module):
    """Per-token gated fusion of current stream and latent stream."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gx = nn.Sequential(*[nn.Linear(embed_dim, embed_dim), nn.GELU()])
        self.gz = nn.Sequential(*[nn.Linear(embed_dim, embed_dim), nn.GELU()])

    def forward(self, x_tokens, z_tokens) :
        ax = self.gx(x_tokens)
        az = self.gz(z_tokens)
        return ax * x_tokens + az * z_tokens

################################################################################
#                                        DiT                                   #
################################################################################

class SwinIR(nn.Module):
    def __init__(self, img_size=192, patch_size=1, in_chans=2, out_chans=1, embed_dim=64, z_chans=2,
                 depths=(4, 4, 4, 4), num_heads=(4, 4, 4, 4), window_size=8,
                 mlp_ratio=4., drop_path_rate=0.1, use_checkpoint=False):
        super().__init__()

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.latent_proj = LatentProjector(z_chans, embed_dim)

        # Split into patches
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim, embed_dim)
        self.latent_patch_embed = PatchEmbed(img_size, patch_size, embed_dim, embed_dim)
        self.patch_unembed = PatchUnEmbed(img_size, patch_size, embed_dim)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        pos = 0
        self.layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        patches_resolution = (img_size, img_size)

        for i_layer in range(len(depths)):
            layer = BasicLayer(dim=embed_dim,
                               input_resolution=patches_resolution,
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               drop_path=dpr[pos:pos + depths[i_layer]],
                               use_checkpoint=use_checkpoint)
            pos += depths[i_layer]
            self.layers.append(layer)
            self.gates.append(GateFusion(embed_dim))

        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim, out_chans, 3, 1, 1)

        self.apply(self._init_weights)

    # ---------------------------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ---------------------------------------------------------------------
    def forward_features(self, x, z):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        z = self.latent_patch_embed(z)
        for gate, layer in zip(self.gates, self.layers):
            x = gate(x, z)
            x = layer(x, x_size)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    # ---------------------------------------------------------------------
    def forward(self, x, z):
        x_shallow = self.conv_first(x)
        z_shallow = self.latent_proj(z)
        res = self.conv_after_body(self.forward_features(x_shallow, z_shallow))
        out = self.conv_last(res+x_shallow)
        return out


def get_timestep_embedding(timesteps: torch.Tensor,
                           dim: int,
                           max_period: int = 10_000):
    """
    Sinusoidal timestep embedding identical to DiT / ADM.
      timesteps: (B,)  int 0 … n_steps-1
      returns  : (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    ).to(timesteps.device)                                   # (half,)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:                                               # zero-pad if odd
        emb = F.pad(emb, (0, 1))
    return emb                                                # (B, dim)


class AdaLayerNorm(nn.Module):
    """
    Adaptive LayerNorm-Zero (DiT-style):
      x_norm = (x-μ)/σ
      out    = γ*(1+Δγ)*x_norm + (β+Δβ)
    """
    def __init__(self, normalized_shape: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta  = nn.Parameter(torch.zeros(normalized_shape))
        self.mod   = nn.Linear(cond_dim, 2 * normalized_shape)  # Δγ, Δβ
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        x   : (B, *, C)
        cond: (B, cond_dim)
        """
        B, C = x.shape[0], x.shape[-1]
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        mod = self.mod(cond).view(B, 2, C)         # (B, 2, C)
        d_gamma, d_beta = mod[:, 0], mod[:, 1]     # (B, C) each
        return (self.gamma * (1 + d_gamma)).unsqueeze(1) * x_hat + \
               (self.beta  +       d_beta).unsqueeze(1)


class DiTSwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads,
                 window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., cond_dim=128):
        super().__init__()
        self.dim  = dim
        self.ws   = window_size
        self.shift_size = shift_size if min(input_resolution) > window_size else 0
        self.input_resolution = input_resolution

        self.norm1 = AdaLayerNorm(dim, cond_dim)
        self.attn  = WindowAttention(dim, to_2tuple(window_size), num_heads,
                                     qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = AdaLayerNorm(dim, cond_dim)
        self.mlp   = Mlp(dim, mlp_ratio, drop)

        # pre-compute attention mask for SW-MSA
        if self.shift_size > 0:
            H, W = input_resolution
            img_mask = torch.zeros(1, H, W, 1)
            h_slices = (slice(0, -window_size),
                        slice(-window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -window_size),
                        slice(-window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0) \
                                   .masked_fill(attn_mask == 0,   0.0)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, x_size, cond):
        H, W = x_size
        B, L, C = x.shape
        shortcut = x

        # ---- Attention ----
        x = self.norm1(x, cond).view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x_windows = window_partition(x, self.ws).view(-1, self.ws * self.ws, C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.ws, self.ws, C)
        x = window_reverse(attn_windows, self.ws, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(B, L, C)
        x = shortcut + self.drop_path(x)

        # ---- FFN ----
        x = x + self.drop_path(self.mlp(self.norm2(x, cond)))
        return x


class DiTBasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., use_checkpoint=False, cond_dim=128):
        super().__init__()
        self.blocks = nn.ModuleList([
            DiTSwinTransformerBlock(dim, input_resolution, num_heads,
                                 window_size, 0 if i % 2 == 0 else window_size // 2,
                                 mlp_ratio, qkv_bias, drop, attn_drop,
                                 drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 cond_dim)
            for i in range(depth)
        ])
        self.use_checkpoint = use_checkpoint

    def forward(self, x, x_size, cond):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x, x_size, cond)
            else:
                x = blk(x, x_size, cond)
        return x


class DiT(nn.Module):

    def __init__(self,
                 img_size=24,
                 in_chans=4,
                 z_chans=4,
                 out_chans=4,
                 # num_classes=1000,
                 embed_dim=128,
                 depths=(2, 2, 6, 2),
                 num_heads=(4, 4, 4, 4),
                 window_size=6,
                 mlp_ratio=4.,
                 drop_path_rate=0.1,
                 use_checkpoint=False):
        super().__init__()
        self.embed_dim = embed_dim
        # self.num_classes = num_classes

        # shallow feature
        self.conv_x = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.conv_z = nn.Conv2d(z_chans, embed_dim, 3, 1, 1)

        # patches
        self.patch_embed = PatchEmbed(img_size, 1, embed_dim, embed_dim)
        self.patch_unembed = PatchUnEmbed(img_size, 1, embed_dim)

        # time / class embeddings
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        # self.label_emb = nn.Embedding(num_classes, embed_dim)

        # transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        pos = 0
        patches_resolution = (img_size, img_size)
        self.layers = nn.ModuleList()
        self.gates  = nn.ModuleList()
        for i_layer, depth in enumerate(depths):
            layer = DiTBasicLayer(embed_dim, patches_resolution, depth,
                               num_heads[i_layer], window_size, mlp_ratio,
                               drop_path=dpr[pos:pos + depth],
                               use_checkpoint=use_checkpoint,
                               cond_dim=embed_dim)
            pos += depth
            self.layers.append(layer)
            self.gates.append(GateFusion(embed_dim))

        self.norm = nn.LayerNorm(embed_dim)
        self.conv_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim, out_chans, 3, 1, 1)

        self.apply(self._init_weights)

    # ------------------------------------------------------------------ #
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias) if m.bias is not None else None
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    # ------------------------------------------------------------------ #
    def _condition_vector(self, t, y):

        t_emb = self.time_mlp(get_timestep_embedding(t, self.embed_dim))
        y_emb = self.label_emb(y) if self.num_classes > 0 else 0
        return t_emb + y_emb

    # ------------------------------------------------------------------ #
    def time_embedding(self, t):

        t_emb = self.time_mlp(get_timestep_embedding(t, self.embed_dim))
        return t_emb

    # ------------------------------------------------------------------ #
    def forward_features(self, x_feat, z_feat, cond):
        x_size = (x_feat.shape[2], x_feat.shape[3])          # (24, 24)
        x_tok  = self.patch_embed(x_feat)
        z_tok  = self.patch_embed(z_feat)

        for gate, layer in zip(self.gates, self.layers):
            x_tok = gate(x_tok, z_tok)                      # 融合 x & z
            x_tok = layer(x_tok, x_size, cond)

        x_tok = self.norm(x_tok)
        x_feat = self.patch_unembed(x_tok, x_size)
        return x_feat

    # ------------------------------------------------------------------ #
    def forward(self, x, t, z):
        """
        x,z : (B, 4, 24, 24)
        t,y : (B,)
        """
        x_shallow = self.conv_x(x)
        z_shallow = self.conv_z(z)
        cond_vec  = self.time_embedding(t)

        body_out  = self.conv_body(self.forward_features(x_shallow, z_shallow, cond_vec))
        out       = self.conv_last(body_out + x_shallow)
        return out


class DiT_concate(nn.Module):

    def __init__(self,
                 img_size=24,
                 in_chans=4,
                 z_chans=4,
                 out_chans=4,
                 # num_classes=1000,
                 embed_dim=128,
                 depths=(2, 2, 6, 2),
                 num_heads=(4, 4, 4, 4),
                 window_size=6,
                 mlp_ratio=4.,
                 drop_path_rate=0.1,
                 use_checkpoint=False):
        super().__init__()
        self.embed_dim = embed_dim
        # self.num_classes = num_classes

        # shallow feature
        self.conv_x = nn.Conv2d(in_chans+z_chans, embed_dim, 3, 1, 1)
        # self.conv_z = nn.Conv2d(z_chans, embed_dim, 3, 1, 1)

        # patches
        self.patch_embed = PatchEmbed(img_size, 1, embed_dim, embed_dim)
        self.patch_unembed = PatchUnEmbed(img_size, 1, embed_dim)

        # time / class embeddings
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        # self.label_emb = nn.Embedding(num_classes, embed_dim)

        # transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        pos = 0
        patches_resolution = (img_size, img_size)
        self.layers = nn.ModuleList()
        # self.gates  = nn.ModuleList()
        for i_layer, depth in enumerate(depths):
            layer = DiTBasicLayer(embed_dim, patches_resolution, depth,
                               num_heads[i_layer], window_size, mlp_ratio,
                               drop_path=dpr[pos:pos + depth],
                               use_checkpoint=use_checkpoint,
                               cond_dim=embed_dim)
            pos += depth
            self.layers.append(layer)
            # self.gates.append(GateFusion(embed_dim))

        self.norm = nn.LayerNorm(embed_dim)
        self.conv_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim, out_chans, 3, 1, 1)

        self.apply(self._init_weights)

    # ------------------------------------------------------------------ #
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias) if m.bias is not None else None
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    # ------------------------------------------------------------------ #
    def _condition_vector(self, t, y):

        t_emb = self.time_mlp(get_timestep_embedding(t, self.embed_dim))
        y_emb = self.label_emb(y) if self.num_classes > 0 else 0
        return t_emb + y_emb

    # ------------------------------------------------------------------ #
    def time_embedding(self, t):

        t_emb = self.time_mlp(get_timestep_embedding(t, self.embed_dim))
        return t_emb

    # ------------------------------------------------------------------ #
    def forward_features(self, x_feat, cond):
        x_size = (x_feat.shape[2], x_feat.shape[3])          # (24, 24)
        x_tok  = self.patch_embed(x_feat)
        # z_tok  = self.patch_embed(z_feat)

        for layer in self.layers:
            # x_tok = gate(x_tok, z_tok)                      # 融合 x & z
            x_tok = layer(x_tok, x_size, cond)

        x_tok = self.norm(x_tok)
        x_feat = self.patch_unembed(x_tok, x_size)
        return x_feat

    # ------------------------------------------------------------------ #
    def forward(self, x, t, z):
        """
        x,z : (B, 4, 24, 24)
        t,y : (B,)
        """
        x_shallow = self.conv_x(torch.cat((x, z), dim=1))
        # z_shallow = self.conv_z(z)
        # x_concat = torch.cat((x_shallow, z_shallow), dim=1)
        cond_vec  = self.time_embedding(t)

        body_out  = self.conv_body(self.forward_features(x_shallow, cond_vec))
        out       = self.conv_last(body_out + x_shallow)
        return out


class CrossWindowAttention(nn.Module):
    """
    Window-based multi-head CROSS attention (W-XCA).
    Query from x-window tokens, Key/Value from z-window tokens.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table (same as self-attn, windows对齐时可复用)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        # pair-wise relative position index for tokens inside one window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flat = torch.flatten(coords, 1)                      # 2,Wh*Ww
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # 2, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()      # N,N,2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)                     # N,N
        self.register_buffer("relative_position_index", relative_position_index)

        # Separate projections for Q (from x) and KV (from z)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x_win, z_win):
        """
        x_win: (num_windows*B, N, C)  - queries
        z_win: (num_windows*B, N, C)  - keys/values
        """
        B_, N, C = x_win.shape

        q = self.q(x_win).reshape(B_, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)  # B_, heads, N, dim

        kv = self.kv(z_win).reshape(B_, N, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)  # 2, B_, heads, N, dim
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # B_, heads, N, N

        # add relative position bias
        rel_pos = self.relative_position_bias_table[self.relative_position_index.view(-1)] \
                    .view(N, N, -1).permute(2, 0, 1).contiguous()  # heads, N, N
        attn = attn + rel_pos.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # B_, N, C
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class CrossAttnFusion(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 qkv_bias=True, attn_drop=0., proj_drop=0., drop_path=0.,
                 cond_dim=128):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.ws = window_size

        self.norm_x = AdaLayerNorm(dim, cond_dim)
        self.norm_z = nn.LayerNorm(dim)

        self.xca = CrossWindowAttention(dim, to_2tuple(window_size), num_heads,
                                        qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_tok, z_tok, x_size, cond):
        """
        x_tok, z_tok: (B, H*W, C)
        x_size: (H, W)
        cond:   (B, cond_dim)
        """
        H, W = x_size
        B, L, C = x_tok.shape
        assert L == H * W, "Token length must match H*W for window partition."

        # AdaLN on x; LN on z
        x = self.norm_x(x_tok, cond).view(B, H, W, C)
        z = self.norm_z(z_tok).view(B, H, W, C)

        # partition windows
        x_win = window_partition(x, self.ws).view(-1, self.ws * self.ws, C)  # (nW*B, N, C)
        z_win = window_partition(z, self.ws).view(-1, self.ws * self.ws, C)

        # window cross-attention
        fused = self.xca(x_win, z_win)  # (nW*B, N, C)

        # reverse windows
        fused = fused.view(-1, self.ws, self.ws, C)
        fused = window_reverse(fused, self.ws, H, W).view(B, L, C)

        # residual
        return x_tok + self.drop_path(fused)


class DiT_crossattn(nn.Module):

    def __init__(self,
                 img_size=24,
                 in_chans=4,
                 z_chans=4,
                 out_chans=4,
                 embed_dim=128,
                 depths=(2, 2, 6, 2),
                 num_heads=(4, 4, 4, 4),
                 window_size=6,
                 mlp_ratio=4.,
                 drop_path_rate=0.1,
                 use_checkpoint=False):
        super().__init__()
        self.embed_dim = embed_dim

        # shallow feature
        self.conv_x = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.conv_z = nn.Conv2d(z_chans, embed_dim, 3, 1, 1)

        # patches
        self.patch_embed   = PatchEmbed(img_size, 1, embed_dim, embed_dim)
        self.patch_unembed = PatchUnEmbed(img_size, 1, embed_dim)

        # time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        pos = 0
        patches_resolution = (img_size, img_size)
        self.layers = nn.ModuleList()
        self.fusions = nn.ModuleList()
        for i_layer, depth in enumerate(depths):
            self.fusions.append(
                CrossAttnFusion(embed_dim, patches_resolution,
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                qkv_bias=True, attn_drop=0., proj_drop=0.,
                                drop_path=0.,
                                cond_dim=embed_dim)
            )
            layer = DiTBasicLayer(embed_dim, patches_resolution, depth,
                                  num_heads[i_layer], window_size, mlp_ratio,
                                  drop_path=dpr[pos:pos + depth],
                                  use_checkpoint=use_checkpoint,
                                  cond_dim=embed_dim)
            pos += depth
            self.layers.append(layer)

        self.norm = nn.LayerNorm(embed_dim)
        self.conv_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim, out_chans, 3, 1, 1)

        self.apply(self._init_weights)

    # ------------------------------- utils ---------------------------------- #
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def time_embedding(self, t):
        return self.time_mlp(get_timestep_embedding(t, self.embed_dim))

    # ------------------------------- forward -------------------------------- #
    def forward_features(self, x_feat, z_feat, cond):
        H, W = x_feat.shape[2], x_feat.shape[3]
        x_tok = self.patch_embed(x_feat)  # (B, H*W, C)
        z_tok = self.patch_embed(z_feat)  # (B, H*W, C)

        for fusion, layer in zip(self.fusions, self.layers):
            x_tok = fusion(x_tok, z_tok, (H, W), cond)
            x_tok = layer(x_tok, (H, W), cond)

        x_tok = self.norm(x_tok)
        x_feat = self.patch_unembed(x_tok, (H, W))
        return x_feat

    def forward(self, x, t, z):
        """
        x,z : (B, in_chans/z_chans, img_size, img_size)
        t   : (B,)
        """
        x_shallow = self.conv_x(x)
        z_shallow = self.conv_z(z)
        cond_vec  = self.time_embedding(t)

        body_out  = self.conv_body(self.forward_features(x_shallow, z_shallow, cond_vec))
        out       = self.conv_last(body_out + x_shallow)
        return out
