# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


__all__ = [
    "window_partition",
    "window_unpartition",
    "add_decomposed_rel_pos",
    "get_abs_pos",
    "PatchEmbed",
]


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, X, Y, Z, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, window_size, C].
        (Xp, Yp, Zp): padded height and width before partition
    """
    B, X, Y, Z, C = x.shape

    pad_x = (window_size - X % window_size) % window_size
    pad_y = (window_size - Y % window_size) % window_size
    pad_z = (window_size - Z % window_size) % window_size
    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        x = F.pad(x, (0, 0, 0, pad_x, 0, pad_y))

    Xp, Yp, Zp = X + pad_x, Y + pad_y, Z + pad_z

    x = x.view(B, Xp // window_size, window_size, Yp // window_size, window_size, Zp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows, (Xp, Yp, Zp)


def window_unpartition(windows, window_size, pad_xyz, xyz):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, window_size, C].
        window_size (int): window size.
        pad_xyz (Tuple): padded height and width (Xp, Yp, Zp).
        xyz (Tuple): original height and width (X, Y, Z) before padding.

    Returns:
        x: unpartitioned sequences with [B, X, Y, Z, C].
    """
    Xp, Yp, Zp = pad_xyz
    X, Y, Z = xyz
    B = windows.shape[0] // (Xp * Yp * Zp // window_size // window_size // window_size)
    x = windows.view(B, Xp // window_size, Yp // window_size, Zp // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Xp, Yp, Zp, -1)

    if Xp > X or Yp > Y or Zp > Z:
        x = x[:, :X, :Y, :Z, :].contiguous()
    return x


def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_x, rel_pos_y, rel_pos_z, q_size, k_size):
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_x * q_y * q_z, C).
        rel_pos_x (Tensor): relative position embeddings (Lx, C) for height axis.
        rel_pos_y (Tensor): relative position embeddings (Ly, C) for width axis.
        rel_pos_z (Tensor): relative position embeddings (Lz, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_x, q_y, q_z = q_size
    k_x, k_y, k_z = k_size

    Rx = get_rel_pos(q_x, k_x, rel_pos_x)
    Ry = get_rel_pos(q_y, k_y, rel_pos_y)
    Rz = get_rel_pos(q_z, k_z, rel_pos_z)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_x, q_y, q_z, dim)
    rel_x = torch.einsum("bxyzc,xkc->bxyzk", r_q, Rx)
    rel_y = torch.einsum("bxyzc,ykc->bxyzk", r_q, Ry)
    rel_z = torch.einsum("bxyzc,zkc->bxyzk", r_q, Rz)

    attn = (
        attn.view(B, q_x, q_y, q_z, k_x, k_y, k_z) + rel_x[:, :, :, :, None] + rel_y[:, :, :, :, :, None] + + rel_z[:, :, :, :, :, :, None]
    ).view(B, q_x * q_y * q_z, k_x * k_y * k_z)

    return attn


def get_abs_pos(abs_pos, has_cls_token, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), in_chans=3, embed_dim=768
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True, attn_drop=0., proj_drop=0., eta=None):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if eta is not None: # LayerScale Initialization (no layerscale when None)
            self.gamma1 = nn.Parameter(eta * torch.ones(dim * 3), requires_grad=True)
            self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

    def forward(self, x):
        B, N, C = x.shape

        # qkv
        qkv = (self.gamma1 * self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # mask torchscript happy (cannot use tensor as tuple)

        # attention matrix
        attn = (q @ k.transpose(-2, -1)) * self.scale # batch x head x seq x seq'
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v) # batch x head x seq x embed_chunk
        x = x.transpose(1, 2) # batch x seq x head x embed_chunk
        x = x.reshape(B, N, C)

        # projection
        x = self.gamma2 * self.proj(x)
        x = self.proj_drop(x)

        return x
    

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0., eta=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

        if eta is not None: # LayerScale Initialization (no layerscale when None)
            self.gamma1 = nn.Parameter(eta * torch.ones(hidden_features), requires_grad=True)
            self.gamma2 = nn.Parameter(eta * torch.ones(out_features), requires_grad=True)
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

    def forward(self, x):
        x = self.gamma1 * self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gamma2 * self.fc2(x)
        x = self.drop2(x)
        return x