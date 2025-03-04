"""
Vision Transformer (ViT) implementation for GRAFT framework.

This module implements a Vision Transformer backbone with support
for loading MAE pre-trained weights and incorporating a multi-label
classification head.
"""
from typing import Dict, List, Optional, Tuple, Union, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.

    This module splits an image into patches and projects them into an embedding space.
    """

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_channels: int = 3,
            embed_dim: int = 768
    ):
        """
        Initialize patch embedding.

        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_channels: Number of input channels.
            embed_dim: Embedding dimension.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Tensor of shape [B, N, D] where N is the number of patches
            and D is the embedding dimension.
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."

        # [B, C, H, W] -> [B, D, H/P, W/P]
        x = self.proj(x)
        # [B, D, H/P, W/P] -> [B, D, N]
        x = x.flatten(2)
        # [B, D, N] -> [B, N, D]
        x = x.transpose(1, 2)

        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for multi-label image classification.

    This implementation is based on the original ViT paper with
    modifications for multi-label classification and integration
    with the GRAFT framework.
    """

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_channels: int = 3,
            num_classes: int = 20,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            norm_layer: nn.Module = nn.LayerNorm,
            **kwargs
    ):
        """
        Initialize Vision Transformer.

        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_channels: Number of input channels.
            num_classes: Number of output classes.
            embed_dim: Embedding dimension.
            depth: Number of transformer layers.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden dimension ratio.
            qkv_bias: Whether to use bias in QKV projection.
            drop_rate: Dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.n_patches

        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer encoder
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Multi-label classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        )

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize model weights.
        """
        # Initialize patch_embed like nn.Linear
        nn.init.trunc_normal_(self.patch_embed.proj.weight, std=0.02)

        # Initialize cls_token
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize pos_embed with sine-cosine positional encoding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.n_patches ** 0.5),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize all other linear layers and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize individual module weights.

        Args:
            m: Module to initialize.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Feature tensor of shape [B, D] where D is the embedding dimension.
        """
        B = x.shape[0]

        # Extract patch embeddings
        x = self.patch_embed(x)

        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        # Return [CLS] token features
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Logits tensor of shape [B, num_classes].
        """
        # Extract features
        x = self.forward_features(x)

        # Apply classification head
        x = self.head(x)

        return x

    def load_mae_weights(self, checkpoint_path: str, strict: bool = False):
        """
        Load weights from a MAE pre-trained checkpoint.

        Args:
            checkpoint_path: Path to the MAE checkpoint.
            strict: Whether to strictly enforce that the keys in state_dict match.
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        print(f"Loading MAE pre-trained weights from: {checkpoint_path}")

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Filter out decoder weights
        state_dict = {k: v for k, v in state_dict.items() if 'decoder' not in k}

        # Filter out head weights
        state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}

        # Load weights
        msg = self.load_state_dict(state_dict, strict=strict)
        print(f"Loaded MAE weights with message: {msg}")

        return msg


class Block(nn.Module):
    """
    Transformer encoder block with attention and MLP.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            drop_path: float = 0.0,
            norm_layer: nn.Module = nn.LayerNorm
    ):
        """
        Initialize transformer block.

        Args:
            dim: Input dimension.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden dimension ratio.
            qkv_bias: Whether to use bias in QKV projection.
            drop: Dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        # Attention block with residual connection
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # MLP block with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Attention(nn.Module):
    """
    Multi-head self-attention module.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0
    ):
        """
        Initialize attention module.

        Args:
            dim: Input dimension.
            num_heads: Number of attention heads.
            qkv_bias: Whether to use bias in QKV projection.
            attn_drop: Attention dropout rate.
            proj_drop: Output projection dropout rate.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, N, D].

        Returns:
            Output tensor of shape [B, N, D].
        """
        B, N, C = x.shape

        # Project input to Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D/H]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each of shape [B, H, N, D/H]

        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention weights
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, D]

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """
    Multi-layer perceptron module.
    """

    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            drop: float = 0.0
    ):
        """
        Initialize MLP.

        Args:
            in_features: Input feature dimension.
            hidden_features: Hidden feature dimension.
            out_features: Output feature dimension.
            drop: Dropout rate.
        """
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """

    def __init__(self, drop_prob: float = 0.0):
        """
        Initialize drop path.

        Args:
            drop_prob: Probability of dropping a path.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with drop path.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with randomly dropped paths.
        """
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor

        return output


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """
    Generate 2D sine-cosine positional embedding.

    Args:
        embed_dim: Embedding dimension.
        grid_size: Grid size.
        cls_token: Whether to include a class token.

    Returns:
        Positional embedding of shape [grid_size*grid_size+(1 if cls_token else 0), embed_dim].
    """
    import numpy as np

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """
    Generate 2D sine-cosine positional embedding from grid.

    Args:
        embed_dim: Embedding dimension.
        grid: Grid of shape [2, H, W].

    Returns:
        Positional embedding of shape [H*W, embed_dim].
    """
    import numpy as np

    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Generate 1D sine-cosine positional embedding from grid.

    Args:
        embed_dim: Embedding dimension.
        pos: Position grid of shape [H, W].

    Returns:
        Positional embedding of shape [H*W, embed_dim].
    """
    import numpy as np

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (H*W,)
    out = np.einsum('i,j->ij', pos, omega)  # (H*W, D/2), outer product

    emb_sin = np.sin(out)  # (H*W, D/2)
    emb_cos = np.cos(out)  # (H*W, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (H*W, D)
    return emb


def vit_base_patch16_224(
        pretrained: bool = False,
        pretrained_weights: Optional[str] = None,
        img_size: int = 224,
        num_classes: int = 20,
        **kwargs
) -> VisionTransformer:
    """
    Create a ViT-Base/16 model.

    Args:
        pretrained: Whether to use pre-trained weights.
        pretrained_weights: Path to pre-trained weights.
        img_size: Input image size.
        num_classes: Number of output classes.

    Returns:
        ViT-Base/16 model.
    """
    model = VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        num_classes=num_classes,
        **kwargs
    )

    if pretrained and pretrained_weights is not None:
        model.load_mae_weights(pretrained_weights, strict=False)

    return model


def vit_large_patch16_224(
        pretrained: bool = False,
        pretrained_weights: Optional[str] = None,
        img_size: int = 224,
        num_classes: int = 20,
        **kwargs
) -> VisionTransformer:
    """
    Create a ViT-Large/16 model.

    Args:
        pretrained: Whether to use pre-trained weights.
        pretrained_weights: Path to pre-trained weights.
        img_size: Input image size.
        num_classes: Number of output classes.

    Returns:
        ViT-Large/16 model.
    """
    model = VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        num_classes=num_classes,
        **kwargs
    )

    if pretrained and pretrained_weights is not None:
        model.load_mae_weights(pretrained_weights, strict=False)

    return model


def create_vit_model(
        variant: str = "base",
        patch_size: int = 16,
        img_size: int = 224,
        num_classes: int = 20,
        pretrained: bool = True,
        pretrained_weights: Optional[str] = None,
        **kwargs
) -> VisionTransformer:
    """
    Create a Vision Transformer model with the specified variant.

    Args:
        variant: Model variant, either "base" or "large".
        patch_size: Patch size.
        img_size: Input image size.
        num_classes: Number of output classes.
        pretrained: Whether to use pre-trained weights.
        pretrained_weights: Path to pre-trained weights.

    Returns:
        Vision Transformer model.
    """
    assert patch_size == 16, "Only patch size 16 is supported for now."

    if variant == "base":
        return vit_base_patch16_224(
            pretrained=pretrained,
            pretrained_weights=pretrained_weights,
            img_size=img_size,
            num_classes=num_classes,
            **kwargs
        )
    elif variant == "large":
        return vit_large_patch16_224(
            pretrained=pretrained,
            pretrained_weights=pretrained_weights,
            img_size=img_size,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown ViT variant: {variant}")

