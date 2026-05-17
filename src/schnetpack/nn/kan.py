import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ShiftedChebyKANLayer", "normalize_chebyshev"]


def normalize_chebyshev(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize input to [0, 1] per sample so inputs
    lie within the shifted Chebyshev polynomial domain.
    """
    mn  = x.min(dim=-1, keepdim=True).values
    mx  = x.max(dim=-1, keepdim=True).values
    rng = (mx - mn).clamp(min=1e-8)
    return (x - mn) / rng


@torch.compile
def shifted_chebyshev_basis(x: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Compute shifted Chebyshev polynomials T_0 … T_degree evaluated at x.
 
    Shifted variant maps the natural domain [0, 1] to [-1, 1] via the substitution
    2x-1.
 
        T_0(x)     = 1
        T_1(x)     = 2x - 1
        T_{n+1}(x) = 2(2x-1)·T_n(x) - T_{n-1}(x)
 
    Args:
        x      : Tensor of shape (*, ) — values assumed in [0, 1]
        degree : maximum polynomial degree N
 
    Returns:
        Tensor of shape (*, degree+1)
    """
    T = []
    T0 = torch.ones_like(x)                # T_0 = 1
    T1 = 2.0 * x - 1.0                     # T_1 = 2x - 1
    T.append(T0)
    if degree >= 1:
        T.append(T1)
    for _ in range(2, degree + 1):
        Tn_next = 2.0 * (2.0 * x - 1.0) * T[-1] - T[-2]
        T.append(Tn_next)
    return torch.stack(T, dim=-1)           # (*, degree+1)
 
 
class ShiftedChebyKANLayer(nn.Module):
    """
    One Cheby-KAN layer.
 
    Replaces a linear (MLP) layer with a learnable Chebyshev polynomial
    expansion on every edge of the KAN graph:
 
        y_j = Σ_{i=0}^{degree} c_{j,·,i} · T_i(x)
 
    where c ∈ R^{out_features × in_features × (degree+1)} are the only
    trainable parameters — no bias, no grid.
 
    Parameter count: in_features × (degree+1) × out_features
 
    Args:
        in_features  : dimensionality of input
        out_features : dimensionality of output
        degree       : Chebyshev polynomial degree (paper uses 5 → 6 bases)
    """
 
    def __init__(self, in_features: int, out_features: int, degree: int = 5):
        super().__init__()
        self.in_features  = in_features 
        self.out_features = out_features
        self.degree       = degree
 
        # Learnable coefficients c_i
        # Shape: (out_features, in_features, degree+1)
        self.coeffs = nn.Parameter(
            torch.empty(out_features, in_features, degree + 1)
        )
        self._reset_parameters()
 
    def _reset_parameters(self):
        # Xavier-like init scaled by degree for stable polynomial sums
        nn.init.xavier_uniform_(
            self.coeffs.view(self.out_features, -1)
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, in_features)
 
        Returns:
            (batch, out_features)
        """
        x = normalize_chebyshev(x)

        # (batch, in_features, degree+1)
        T = shifted_chebyshev_basis(x, self.degree)
 
        # Σ_i c_{j,k,i} · T_i(x_k)   summed over k (in_features) and i (degree)
        # einsum: b=batch, k=in, d=degree+1, o=out
        out = torch.einsum('...kd,okd->...o', T, self.coeffs)
        return out
 
    def extra_repr(self) -> str:
        return (
          f'in={self.in_features}, out={self.out_features}, '
          f'degree={self.degree}, '
          f'params={self.in_features*(self.degree+1)*self.out_features}'
        )
