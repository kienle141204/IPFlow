"""
FlowMatchingModel: Main IPFM model for curved-trajectory flow matching.

Architecture recap:
  - IPNet (BAPNet, frozen): extracts per-atom interaction features F_M, F_P.
  - ShiftMLP (psi_theta): maps (F_M, t) -> 3D shift vector delta.
  - UniTransformerO2 backbone: SE(3)-equivariant velocity network.
  - Prior-conditioning: F_M and F_P are concatenated into hidden states.
  - Velocity output head for positions, classifier head for atom types.

Curved trajectory (training):
  x_t = (1-t)*x_0 + t*x_1 + g(t)*delta     where g(t) = t*(1-t)

Velocity target:
  u_t = (x_0 - x_1) + (1-2t)*delta

Loss:
  L = MSE(v_pred, u_t) + lambda_v * CE(type_pred, true_types)
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from models.common import compose_context, ShiftedSoftplus
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral
from .shift_mlp import ShiftMLP


# ---------------------------------------------------------------------------
# Time embedding (sinusoidal), reused from IPDiff pattern
# ---------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) or (N,) — timestep in [0, 1].
        Returns:
            (len(t), dim)
        """
        device = t.device
        half = self.dim // 2
        emb = np.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ---------------------------------------------------------------------------
# Center-of-mass utility
# ---------------------------------------------------------------------------
def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    if mode == 'none':
        offset = torch.zeros(batch_protein.max().item() + 1, 3, device=protein_pos.device)
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError(f'center_pos mode: {mode}')
    return protein_pos, ligand_pos, offset


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------
class FlowMatchingModel(nn.Module):
    """
    Interaction-Prior guided Flow Matching model (IPFM).

    Core responsibilities:
      1. Compute curved trajectory interpolant x_t during training.
      2. Predict velocity field v_theta(x_t, t, protein, F_M, F_P).
      3. Compute training loss: MSE(velocity) + lambda_v * CE(atom types).
      4. Run Euler ODE sampling at inference.
    """

    def __init__(self, config, protein_atom_feature_dim: int, ligand_atom_feature_dim: int):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_classes = ligand_atom_feature_dim  # number of ligand atom types
        self.loss_v_weight = config.loss_v_weight
        self.center_pos_mode = config.center_pos_mode
        self.cond_dim = config.cond_dim            # IPNet feature dim (128)
        self.time_emb_dim = config.time_emb_dim    # sinusoidal embedding dim

        # ----------------------------------------------------------------
        # Embedding layers
        # ----------------------------------------------------------------
        node_indicator = config.node_indicator     # bool: append 0/1 indicator
        if node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)

        # Time embedding for ligand features
        if self.time_emb_dim > 0:
            self.time_emb = nn.Sequential(
                SinusoidalPosEmb(self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                nn.GELU(),
                nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
            )
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim, emb_dim)
        else:
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)

        # Prior-conditioning projection (concat h + IPNet features -> h)
        self.emb_mlp = nn.Linear(emb_dim + self.cond_dim, emb_dim)

        # ----------------------------------------------------------------
        # SE(3)-equivariant backbone
        # ----------------------------------------------------------------
        self.backbone = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=self.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup,
        )

        # ----------------------------------------------------------------
        # Output heads
        # ----------------------------------------------------------------
        # Atom-type prediction head
        self.type_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )

        # ----------------------------------------------------------------
        # ShiftMLP for curved trajectory
        # ----------------------------------------------------------------
        self.shift_mlp = ShiftMLP(
            feat_dim=self.cond_dim,
            time_emb_dim=config.shift_time_emb_dim,
            hidden_dim=config.shift_hidden_dim,
        )

    # ------------------------------------------------------------------
    # Forward: predict velocity and atom types
    # ------------------------------------------------------------------
    def forward(
        self,
        protein_pos, protein_v, batch_protein,
        ligand_pos, ligand_v, batch_ligand,
        t_continuous,         # (num_graphs,) float in [0, 1]
        f_m=None,             # (N_ligand, cond_dim) — IPNet ligand features
        f_p=None,             # (N_protein, cond_dim) — IPNet protein features
    ):
        """
        Forward pass: embed, compose context, run backbone, decode outputs.

        Args:
            protein_pos:   (N_protein, 3)
            protein_v:     (N_protein, protein_feat_dim)
            batch_protein: (N_protein,) int batch indices
            ligand_pos:    (N_ligand, 3)   — already at x_t (interpolated)
            ligand_v:      (N_ligand,) int — atom type indices (perturbed)
            batch_ligand:  (N_ligand,) int
            t_continuous:  (num_graphs,) float in [0, 1]
            f_m:           (N_ligand, cond_dim) or None
            f_p:           (N_protein, cond_dim) or None

        Returns:
            dict with 'pred_vel' (N_ligand, 3) and 'pred_type' (N_ligand, K).
        """
        device = protein_pos.device

        # One-hot encode atom types
        ligand_v_onehot = F.one_hot(ligand_v, self.num_classes).float()

        # Time embedding per atom (indexed by batch_ligand)
        t_per_atom = t_continuous[batch_ligand]   # (N_ligand,)
        if self.time_emb_dim > 0:
            t_feat = self.time_emb(t_per_atom)    # (N_ligand, time_emb_dim)
            input_ligand_feat = torch.cat([ligand_v_onehot, t_feat], dim=-1)
        else:
            input_ligand_feat = torch.cat(
                [ligand_v_onehot, t_per_atom.unsqueeze(-1)], dim=-1
            )

        # Embed
        h_protein = self.protein_atom_emb(protein_v)
        h_ligand = self.ligand_atom_emb(input_ligand_feat)

        # Prior-conditioning: concat IPNet features then project back to emb_dim
        if f_p is None:
            f_p = torch.zeros(h_protein.shape[0], self.cond_dim, device=device)
        if f_m is None:
            f_m = torch.zeros(h_ligand.shape[0], self.cond_dim, device=device)

        h_protein = self.emb_mlp(torch.cat([h_protein, f_p], dim=-1))
        h_ligand = self.emb_mlp(torch.cat([h_ligand, f_m], dim=-1))

        # Node-type indicator: protein=0, ligand=1
        if self.config.node_indicator:
            h_protein = torch.cat(
                [h_protein, torch.zeros(len(h_protein), 1, device=device)], dim=-1
            )
            h_ligand = torch.cat(
                [h_ligand, torch.ones(len(h_ligand), 1, device=device)], dim=-1
            )

        # Compose joint context (protein + ligand, sorted by batch)
        h_all, pos_all, batch_all, mask_ligand, _ = compose_context(
            h_protein=h_protein,
            h_ligand=h_ligand,
            pos_protein=protein_pos,
            pos_ligand=ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        # SE(3)-equivariant backbone
        outputs = self.backbone(h_all, pos_all, mask_ligand, batch_all)
        final_pos = outputs['x']          # (N_all, 3) — updated positions
        final_h = outputs['h']            # (N_all, hidden_dim)

        # Extract ligand outputs
        final_ligand_pos = final_pos[mask_ligand]  # (N_ligand, 3)
        final_ligand_h = final_h[mask_ligand]      # (N_ligand, hidden_dim)

        # Position velocity: difference between backbone output pos and input pos
        pred_vel = final_ligand_pos - ligand_pos   # (N_ligand, 3)

        # Atom type logits
        pred_type = self.type_head(final_ligand_h)  # (N_ligand, K)

        return {
            'pred_vel': pred_vel,
            'pred_type': pred_type,
            'final_ligand_h': final_ligand_h,
        }

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------
    def get_flow_loss(
        self,
        net_cond,            # BAPNet/IPNet (frozen)
        protein_pos, protein_v, batch_protein,
        ligand_pos, ligand_v, batch_ligand,
        t=None,              # optional fixed timestep per graph
    ):
        """
        Compute the curved-trajectory flow matching loss.

        Steps:
          1. Center around protein CoM.
          2. Extract IPNet features F_M, F_P (frozen, with noise augmentation).
          3. Sample t ~ Uniform(0,1) per graph (or use provided t).
          4. Sample x_1 ~ N(0, I).
          5. Compute shift delta = ShiftMLP(F_M, t).
          6. Interpolate: x_t = (1-t)*x_0 + t*x_1 + g(t)*delta.
          7. Perturb atom types (uniform noise as in standard FM discrete).
          8. Forward pass -> (pred_vel, pred_type).
          9. Velocity target: u_t = (x_0 - x_1) + (1-2t)*delta.
          10. Loss = MSE(pred_vel, u_t) + lambda_v * CE(pred_type, true_types).

        Returns dict with 'loss', 'loss_pos', 'loss_v'.
        """
        num_graphs = batch_protein.max().item() + 1
        device = protein_pos.device

        # --- 1. Center ---
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand,
            mode=self.center_pos_mode,
        )

        # --- 2. IPNet features (frozen, no grad) ---
        gt_protein_a_h = torch.argmax(protein_v[:, :6], dim=1)
        gt_protein_r_h = torch.argmax(protein_v[:, 6:26], dim=1)
        gt_lig_a_h = ligand_v  # integer indices

        # Add noise to ligand positions for IPNet (sigma=0.5, see idea.md §3.1)
        ipnet_noise_sigma = getattr(self.config, 'ipnet_noise_sigma', 0.5)
        lig_pos_noisy = ligand_pos + torch.randn_like(ligand_pos) * ipnet_noise_sigma

        with torch.no_grad():
            f_m, f_p = net_cond.extract_features(
                lig_pos_noisy, protein_pos,
                gt_lig_a_h, gt_protein_a_h, gt_protein_r_h,
                batch_ligand, batch_protein,
            )

        # --- 3. Sample t ---
        if t is None:
            t = torch.rand(num_graphs, device=device)  # (B,) uniform [0,1]
        t_per_atom = t[batch_ligand]                   # (N_ligand,)

        # --- 4. Noise x_1 ~ N(0, I) ---
        x_0 = ligand_pos                               # (N_ligand, 3)
        x_1 = torch.randn_like(x_0)                   # (N_ligand, 3)

        # --- 5. Shift ---
        delta = self.shift_mlp(f_m, t_per_atom)        # (N_ligand, 3)
        g_t = ShiftMLP.g(t_per_atom)                   # (N_ligand, 1)

        # --- 6. Curved interpolation ---
        t_col = t_per_atom.unsqueeze(-1)               # (N_ligand, 1)
        x_t = (1.0 - t_col) * x_0 + t_col * x_1 + g_t * delta

        # --- 7. Atom type perturbation (uniform noise / pass-through) ---
        # For simplicity use the ground-truth atom types during training.
        # The type head is trained with CE loss directly.
        ligand_v_noisy = ligand_v                      # (N_ligand,)

        # --- 8. Forward ---
        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            ligand_pos=x_t,
            ligand_v=ligand_v_noisy,
            batch_ligand=batch_ligand,
            t_continuous=t,
            f_m=f_m,
            f_p=f_p,
        )
        pred_vel = preds['pred_vel']    # (N_ligand, 3)
        pred_type = preds['pred_type']  # (N_ligand, K)

        # --- 9. Velocity target ---
        one_minus_2t = (1.0 - 2.0 * t_per_atom).unsqueeze(-1)   # (N_ligand, 1)
        u_t = (x_0 - x_1) + one_minus_2t * delta                 # (N_ligand, 3)

        # --- 10. Loss ---
        # Position loss: mean over atoms then mean over batch
        loss_pos_per_atom = ((pred_vel - u_t) ** 2).sum(-1)       # (N_ligand,)
        loss_pos = scatter_mean(loss_pos_per_atom, batch_ligand, dim=0).mean()

        # Type loss: cross-entropy
        loss_v = F.cross_entropy(pred_type, ligand_v)

        loss = loss_pos + self.loss_v_weight * loss_v

        return {
            'loss': loss,
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'x_t': x_t,
            'x_0': x_0,
            'u_t': u_t,
            'pred_vel': pred_vel,
            'pred_type': pred_type,
        }

    # ------------------------------------------------------------------
    # Inference: Euler ODE sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_flow(
        self,
        net_cond,
        protein_pos, protein_v, batch_protein,
        init_ligand_pos, init_ligand_v, batch_ligand,
        num_steps: int = 100,
        center_pos_mode: str = 'protein',
    ):
        """
        Euler ODE integration from t=1 (noise) to t=0 (data).

        Args:
            init_ligand_pos: (N_ligand, 3) — initial noise positions.
            init_ligand_v:   (N_ligand,) int — initial atom type guesses.
            num_steps:       number of Euler steps (default 100).

        Returns:
            dict with 'pos', 'v', 'pos_traj', 'v_traj'.
        """
        device = protein_pos.device
        num_graphs = batch_protein.max().item() + 1

        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand,
            mode=center_pos_mode,
        )

        x_t = init_ligand_pos.clone()
        v_t = init_ligand_v.clone()

        pos_traj, v_traj = [], []

        # Initial IPNet features with zeros
        f_m = torch.zeros(x_t.shape[0], self.cond_dim, device=device)
        f_p = torch.zeros(protein_pos.shape[0], self.cond_dim, device=device)

        dt = 1.0 / num_steps

        from tqdm.auto import tqdm

        for step in tqdm(range(num_steps), desc='sampling', total=num_steps):
            # t goes from 1.0 -> 0.0 (we integrate backwards)
            t_val = 1.0 - step * dt
            t = torch.full((num_graphs,), t_val, device=device, dtype=torch.float32)
            t_per_atom = t[batch_ligand]

            # Predict velocity
            preds = self(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                ligand_pos=x_t,
                ligand_v=v_t,
                batch_ligand=batch_ligand,
                t_continuous=t,
                f_m=f_m,
                f_p=f_p,
            )
            vel = preds['pred_vel']       # (N_ligand, 3)
            type_logits = preds['pred_type']  # (N_ligand, K)

            # Euler step: x_{t - dt} = x_t - dt * v_theta
            x_t = x_t - dt * vel

            # Estimate x_0 for IPNet feature update:
            #   x_0_est = x_t - t * vel  (one-step denoising estimate)
            # Clamp t to avoid division issues at t~0
            t_safe = t_per_atom.clamp(min=1e-4).unsqueeze(-1)
            x_0_est = x_t - t_safe * vel

            # Update atom type estimate
            v_t_new = type_logits.argmax(dim=-1)

            # Update IPNet features from current estimate
            gt_protein_a_h = torch.argmax(protein_v[:, :6], dim=1)
            gt_protein_r_h = torch.argmax(protein_v[:, 6:26], dim=1)
            lig_a_h_est = v_t_new

            # Add small noise for robustness (ipnet_noise_sigma at inference can be 0 or small)
            infer_noise = getattr(self.config, 'infer_ipnet_noise_sigma', 0.1)
            x_0_noisy = x_0_est + torch.randn_like(x_0_est) * infer_noise
            f_m, f_p = net_cond.extract_features(
                x_0_noisy, protein_pos,
                lig_a_h_est, gt_protein_a_h, gt_protein_r_h,
                batch_ligand, batch_protein,
            )

            v_t = v_t_new

            # Record trajectory (un-center)
            ori_ligand_pos = x_t + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(v_t.clone().cpu())

        # Un-center final positions
        x_t = x_t + offset[batch_ligand]

        return {
            'pos': x_t,
            'v': v_t,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
        }
