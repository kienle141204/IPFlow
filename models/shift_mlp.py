"""
ShiftMLP: Lightweight MLP that maps per-atom IPNet interaction features and
timestep t to a per-atom 3D shift vector.

Architecture (~50K parameters):
    - Sinusoidal time embedding: t -> R^{time_emb_dim}
    - Fusion: concat(F_M_i, time_emb) -> R^{d + time_emb_dim}
    - MLP: Linear -> LayerNorm -> SiLU -> Linear -> LayerNorm -> SiLU -> Linear(3)

The shift output is NOT multiplied by g(t) here — the caller is responsible
for applying the g(t) = t*(1-t) gate so that boundary conditions are preserved.
"""

import numpy as np
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for a scalar timestep t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) or (N,) float tensor of timesteps in [0, 1].
        Returns:
            emb: (..., dim) float tensor.
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=device, dtype=torch.float32) / (half - 1)
        )
        # t: (...,) -> (..., half)
        angles = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (..., dim)
        return emb


class ShiftMLP(nn.Module):
    """
    Per-atom 3D shift MLP.

    For each ligand atom i:
        delta_i = ShiftMLP(F_M_i, t)  in R^3

    Called during training and inference to produce the shift vector.
    The curved trajectory is then:
        x_t = (1-t)*x_0 + t*x_1 + g(t) * delta
    where g(t) = t*(1-t) is applied externally.
    """

    def __init__(self, feat_dim: int = 128, time_emb_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.feat_dim = feat_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim

        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

        in_dim = feat_dim + time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

        # Small initialization for the last linear to start near zero shift
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)

    def forward(self, f_m: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_m:  (N_atoms, feat_dim)  — per-atom IPNet ligand features.
            t:    (N_atoms,)           — timestep broadcast to each atom.
                  (The caller should index t via batch_ligand before passing.)
        Returns:
            delta: (N_atoms, 3)        — per-atom 3D shift (without g(t) gate).
        """
        t_emb = self.time_emb(t)                        # (N_atoms, time_emb_dim)
        x = torch.cat([f_m, t_emb], dim=-1)             # (N_atoms, feat_dim + time_emb_dim)
        return self.net(x)                               # (N_atoms, 3)

    @staticmethod
    def g(t: torch.Tensor) -> torch.Tensor:
        """
        Boundary-preserving gate: g(t) = t * (1 - t).
        g(0) = g(1) = 0, so endpoints x_0 and x_1 are unchanged.

        Args:
            t: (N_atoms,) float tensor.
        Returns:
            (N_atoms, 1) float tensor.
        """
        return (t * (1.0 - t)).unsqueeze(-1)
