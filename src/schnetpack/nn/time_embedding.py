import math
import torch
from torch import nn

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, scale: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, t: torch.Tensor):
        # t: [B] or [B,1]
        device = t.device
        half_dim = self.dim // 2

        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t * emb[None, :] * self.scale
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return emb  # [B, dim]

class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size=256, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi

        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# vi:ts=4 sw=4 et
