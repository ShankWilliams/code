"""
Video Variational Auto-Encoder (VAE) for efficient video compression and reconstruction.
This implementation provides a 3D VAE optimized for video data, enabling efficient processing in latent space while maintaining high-quality reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class Conv3dBlock(nn.Module):
    """3D Convolution block with normalization and activation."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int,int,int]=(3,3,3), stride: Tuple[int,int,int]=(1,1,1), padding: Tuple[int,int,int]=(1,1,1), use_norm: bool=True, activation: str="silu"):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.GroupNorm(32, out_channels) if use_norm else nn.Identity()
        if activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))

class ResBlock3D(nn.Module):
    def __init__(self, channels: int, dropout: float=0.0, use_conv_shortcut: bool=False):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.conv_shortcut = nn.Conv3d(channels, channels, 1) if use_conv_shortcut else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.conv_shortcut(x) + h

class TemporalAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int=8):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels*3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C,T,H,W = x.shape
        x_r = x.permute(0,3,4,1,2).reshape(B*H*W, C, T)
        h = self.norm(x_r.view(B,H,W,C,T).permute(0,3,4,1,2).reshape(B,C,T,H,W))
        h = h.permute(0,3,4,1,2).reshape(B*H*W, C, T)
        q,k,v = self.qkv(h).chunk(3, dim=1)
        q = q.view(B*H*W, self.num_heads, self.head_dim, T).transpose(-2,-1)
        k = k.view(B*H*W, self.num_heads, self.head_dim, T).transpose(-2,-1)
        v = v.view(B*H*W, self.num_heads, self.head_dim, T).transpose(-2,-1)
        attn = torch.softmax((q @ k.transpose(-2,-1)) * (self.head_dim**-0.5), dim=-1)
        out = attn @ v
        out = out.transpose(-2,-1).reshape(B*H*W, C, T)
        out = self.proj(out)
        out = out.view(B,H,W,C,T).permute(0,3,4,1,2)
        return x + out

class DownsampleBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample_time: bool=True, num_res_blocks: int=2):
        super().__init__()
        self.res_blocks = nn.ModuleList([ResBlock3D(in_channels if i==0 else out_channels) for i in range(num_res_blocks)])
        self.conv_in = nn.Conv3d(in_channels, out_channels, 1) if in_channels!=out_channels else nn.Identity()
        if downsample_time:
            self.downsample = nn.Conv3d(out_channels, out_channels, 3, stride=2, padding=1)
        else:
            self.downsample = nn.Conv3d(out_channels, out_channels, (1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.attention = TemporalAttention(out_channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for rb in self.res_blocks:
            x = rb(x)
        x = self.attention(x)
        x = self.downsample(x)
        return x

class UpsampleBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample_time: bool=True, num_res_blocks: int=2):
        super().__init__()
        if upsample_time:
            self.upsample = nn.ConvTranspose3d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1)
        else:
            self.upsample = nn.ConvTranspose3d(in_channels, in_channels, (1,3,3), stride=(1,2,2), padding=(0,1,1), output_padding=(0,1,1))
        self.res_blocks = nn.ModuleList([ResBlock3D(in_channels if i==0 else out_channels) for i in range(num_res_blocks)])
        self.conv_out = nn.Conv3d(in_channels, out_channels, 1) if in_channels!=out_channels else nn.Identity()
        self.attention = TemporalAttention(out_channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        for rb in self.res_blocks:
            x = rb(x)
        x = self.conv_out(x)
        x = self.attention(x)
        return x

class VideoEncoder(nn.Module):
    def __init__(self, in_channels: int=3, latent_channels: int=4, channel_mult=(1,2,4,8), num_res_blocks: int=2, temporal_downsample=(True,True,False,False)):
        super().__init__()
        base_ch = 128
        self.conv_in = Conv3dBlock(in_channels, base_ch)
        self.down_blocks = nn.ModuleList()
        in_ch = base_ch
        for i,mult in enumerate(channel_mult):
            out_ch = base_ch*mult
            down_t = temporal_downsample[i] if i<len(temporal_downsample) else False
            self.down_blocks.append(DownsampleBlock3D(in_ch, out_ch, down_t, num_res_blocks))
            in_ch = out_ch
        self.mid_block1 = ResBlock3D(in_ch)
        self.mid_attn = TemporalAttention(in_ch)
        self.mid_block2 = ResBlock3D(in_ch)
        self.norm_out = nn.GroupNorm(32, in_ch)
        self.conv_out = nn.Conv3d(in_ch, latent_channels*2, 3, padding=1)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_in(x)
        for db in self.down_blocks:
            h = db(h)
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h.chunk(2, dim=1)

class VideoDecoder(nn.Module):
    def __init__(self, out_channels: int=3, latent_channels: int=4, channel_mult=(1,2,4,8), num_res_blocks: int=2, temporal_upsample=(False,False,True,True)):
        super().__init__()
        base_ch = 128
        in_ch = base_ch*channel_mult[-1]
        self.conv_in = Conv3dBlock(latent_channels, in_ch)
        self.mid_block1 = ResBlock3D(in_ch)
        self.mid_attn = TemporalAttention(in_ch)
        self.mid_block2 = ResBlock3D(in_ch)
        self.up_blocks = nn.ModuleList()
        for i,mult in enumerate(reversed(channel_mult)):
            out_ch = base_ch*mult
            up_t = temporal_upsample[-(i+1)] if i < len(temporal_upsample) else False
            self.up_blocks.append(UpsampleBlock3D(in_ch, out_ch, up_t, num_res_blocks))
            in_ch = out_ch
        self.norm_out = nn.GroupNorm(32, in_ch)
        self.conv_out = nn.Conv3d(in_ch, out_channels, 3, padding=1)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        for ub in self.up_blocks:
            h = ub(h)
        h = self.norm_out(h)
        h = F.silu(h)
        return self.conv_out(h)

class VideoVAE(nn.Module):
    def __init__(self, in_channels: int=3, out_channels: int=3, latent_channels: int=4, channel_mult=(1,2,4,8), num_res_blocks: int=2, temporal_downsample=(True,True,False,False), temporal_upsample=(False,False,True,True), sample_posterior: bool=True):
        super().__init__()
        self.sample_posterior = sample_posterior
        self.encoder = VideoEncoder(in_channels, latent_channels, channel_mult, num_res_blocks, temporal_downsample)
        self.decoder = VideoDecoder(out_channels, latent_channels, channel_mult, num_res_blocks, temporal_upsample)
        self.quant_conv = nn.Conv3d(latent_channels*2, latent_channels*2, 1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, 1)
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self.encoder(x)
        mean, logvar = self.quant_conv(torch.cat([mean, logvar], dim=1)).chunk(2, dim=1)
        return mean, logvar
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(z))
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.sample_posterior:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return {"reconstruction": recon, "mean": mean, "logvar": logvar, "z": z}
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.encode(x)
        return self.reparameterize(mean, logvar)
    def compute_loss(self, x: torch.Tensor, reconstruction: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor, kl_weight: float=1e-6) -> Dict[str, torch.Tensor]:
        recon_loss = F.mse_loss(reconstruction, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl_loss = kl_loss / (x.shape[0]*x.shape[2]*x.shape[3]*x.shape[4])
        total = recon_loss + kl_weight * kl_loss
        return {"total_loss": total, "recon_loss": recon_loss, "kl_loss": kl_loss}

def create_video_vae(config_name: str="base") -> VideoVAE:
    configs = {
        "small": {"latent_channels": 4, "channel_mult": (1,2,4), "num_res_blocks": 1, "temporal_downsample": (True,False,False), "temporal_upsample": (False,False,True)},
        "base": {"latent_channels": 4, "channel_mult": (1,2,4,8), "num_res_blocks": 2, "temporal_downsample": (True,True,False,False), "temporal_upsample": (False,False,True,True)},
        "large": {"latent_channels": 8, "channel_mult": (1,2,4,8,16), "num_res_blocks": 3, "temporal_downsample": (True,True,True,False,False), "temporal_upsample": (False,False,True,True,True)}
    }
    return VideoVAE(**configs.get(config_name, configs["base"]))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = create_video_vae().to(device)
    x = torch.randn(1,3,16,64,64).to(device)
    with torch.no_grad():
        out = vae(x)
        print(out["reconstruction"].shape)
        print(out["z"].shape)
        print(sum(p.numel() for p in vae.parameters()))
