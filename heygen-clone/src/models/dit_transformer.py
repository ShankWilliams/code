"""
Diffusion Transformer (DiT) Architecture for Video Generation

This implementation provides a state-of-the-art DiT architecture optimized for video generation,
incorporating the latest research insights from leading models like Sora and CogVideoX.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math


class TimestepEmbedding(nn.Module):
    """Timestep embedding for diffusion process."""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        ).to(timesteps.device)
        
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding


class PositionalEncoding3D(nn.Module):
    """3D positional encoding for video tokens (time, height, width)."""
    
    def __init__(self, dim: int, max_len: int = 1000):
        super().__init__()
        self.dim = dim
        
        # Create positional encoding for each dimension
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, t: int, h: int, w: int) -> torch.Tensor:
        """Generate 3D positional encoding for video tokens."""
        # Temporal encoding
        t_pe = self.pe[:t, :self.dim//3]
        
        # Spatial encoding
        h_pe = self.pe[:h, self.dim//3:2*self.dim//3]
        w_pe = self.pe[:w, 2*self.dim//3:]
        
        # Combine encodings
        pos_encoding = torch.zeros(t, h, w, self.dim)
        for i in range(t):
            for j in range(h):
                for k in range(w):
                    pos_encoding[i, j, k] = torch.cat([t_pe[i], h_pe[j], w_pe[k]])
                    
        return pos_encoding.view(-1, self.dim)


class MultiHeadAttention3D(nn.Module):
    """Multi-head attention optimized for 3D video data."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class CrossAttention(nn.Module):
    """Cross-attention for text conditioning."""
    
    def __init__(self, dim: int, context_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(context_dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        _, M, _ = context.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    """Diffusion Transformer block with self-attention and cross-attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention3D(dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        if context_dim is not None:
            self.cross_attn = CrossAttention(dim, context_dim, num_heads, dropout)
            self.norm_cross = nn.LayerNorm(dim)
        else:
            self.cross_attn = None
            
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        x = x + self.attn(self.norm1(x), mask)
        
        # Cross-attention (if context provided)
        if self.cross_attn is not None and context is not None:
            x = x + self.cross_attn(self.norm_cross(x), context)
            
        # Feed-forward
        x = x + self.mlp(self.norm3(x))
        
        return x


class VideoTokenizer(nn.Module):
    """Convert video patches to tokens and back."""
    
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (2, 16, 16),  # (t, h, w)
        in_channels: int = 4,  # VAE latent channels
        embed_dim: int = 1024
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Output projection
        self.output_proj = nn.Conv3d(
            embed_dim,
            in_channels,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert video to patches."""
        B, C, T, H, W = x.shape
        
        # Apply patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, T', H', W')
        
        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # (B, T'*H'*W', embed_dim)
        
        return x
        
    def unpatchify(self, x: torch.Tensor, output_shape: Tuple[int, ...]) -> torch.Tensor:
        """Convert patches back to video."""
        B, N, C = x.shape
        T, H, W = output_shape[2:]
        
        # Calculate patch grid dimensions
        T_p = T // self.patch_size[0]
        H_p = H // self.patch_size[1]
        W_p = W // self.patch_size[2]
        
        # Reshape to 3D
        x = x.transpose(1, 2).view(B, C, T_p, H_p, W_p)
        
        # Apply output projection (transposed convolution)
        x = F.interpolate(x, size=(T, H, W), mode='trilinear', align_corners=False)
        x = self.output_proj(x)
        
        return x


class DiffusionTransformer(nn.Module):
    """
    Diffusion Transformer for Video Generation.
    
    This implementation follows the DiT architecture with optimizations for video data,
    including 3D positional encoding, temporal attention, and cross-attention for text conditioning.
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int, int] = (16, 32, 32),  # (T, H, W) in latent space
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        in_channels: int = 4,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        context_dim: int = 2048,  # T5-XL embedding dimension
        dropout: float = 0.1,
        learn_sigma: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.learn_sigma = learn_sigma
        
        # Calculate number of patches
        self.num_patches = (
            (input_size[0] // patch_size[0]) *
            (input_size[1] // patch_size[1]) *
            (input_size[2] // patch_size[2])
        )
        
        # Video tokenizer
        self.tokenizer = VideoTokenizer(patch_size, in_channels, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding3D(embed_dim)
        
        # Timestep embedding
        self.time_embed = TimestepEmbedding(embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio, dropout, context_dim)
            for _ in range(depth)
        ])
        
        # Output layers
        self.norm_final = nn.LayerNorm(embed_dim)
        out_channels = in_channels * 2 if learn_sigma else in_channels
        self.output = nn.Linear(embed_dim, out_channels)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights following DiT paper."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
            
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Diffusion Transformer.
        
        Args:
            x: Input video tensor (B, C, T, H, W)
            timesteps: Diffusion timesteps (B,)
            context: Text conditioning (B, seq_len, context_dim)
            mask: Attention mask (optional)
            
        Returns:
            Predicted noise (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        
        # Convert to patches
        x = self.tokenizer.patchify(x)  # (B, N, embed_dim)
        
        # Add positional encoding
        pos_enc = self.pos_encoding(
            T // self.patch_size[0],
            H // self.patch_size[1],
            W // self.patch_size[2]
        ).unsqueeze(0).expand(B, -1, -1).to(x.device)
        x = x + pos_enc
        
        # Add timestep embedding
        t_emb = self.time_embed(timesteps)
        t_emb = self.time_proj(t_emb).unsqueeze(1)  # (B, 1, embed_dim)
        x = x + t_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, context, mask)
            
        # Final normalization and output projection
        x = self.norm_final(x)
        x = self.output(x)
        
        # Convert back to video format
        if self.learn_sigma:
            # Split into noise prediction and variance
            noise, log_var = x.chunk(2, dim=-1)
            noise = self.tokenizer.unpatchify(noise, (B, C, T, H, W))
            log_var = self.tokenizer.unpatchify(log_var, (B, C, T, H, W))
            return torch.cat([noise, log_var], dim=1)
        else:
            x = self.tokenizer.unpatchify(x, (B, C, T, H, W))
            return x


# Model configurations for different scales
MODEL_CONFIGS = {
    "dit_small": {
        "embed_dim": 512,
        "depth": 12,
        "num_heads": 8,
        "mlp_ratio": 4.0
    },
    "dit_base": {
        "embed_dim": 768,
        "depth": 16,
        "num_heads": 12,
        "mlp_ratio": 4.0
    },
    "dit_large": {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0
    },
    "dit_xlarge": {
        "embed_dim": 1536,
        "depth": 32,
        "num_heads": 24,
        "mlp_ratio": 4.0
    }
}


def create_dit_model(config_name: str = "dit_base", **kwargs) -> DiffusionTransformer:
    """Create a DiT model with predefined configuration."""
    config = MODEL_CONFIGS[config_name].copy()
    config.update(kwargs)
    return DiffusionTransformer(**config)


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_dit_model("dit_base").to(device)
    
    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 4, 16, 32, 32).to(device)  # Latent video
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    context = torch.randn(batch_size, 77, 2048).to(device)  # T5 embeddings
    
    # Forward pass
    with torch.no_grad():
        output = model(x, timesteps, context)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
