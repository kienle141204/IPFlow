# IPFlow (IPFM)

**Interaction-Prior guided Flow Matching** for structure-based drug design (SBDD).

IPFlow replaces IPDiff's 1000-step diffusion process with ~100-step deterministic ODE integration via Flow Matching, while preserving the interaction-prior guidance that makes IPDiff outperform other SBDD baselines. The core idea is to bend the straight-line flow trajectory using per-atom features from a pretrained binding-affinity network (IPNet/BAPNet), guiding generation toward protein-ligand configurations with high binding affinity.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Environment Setup](#environment-setup)
5. [Data Preparation](#data-preparation)
6. [Training](#training)
7. [Sampling](#sampling)
8. [Evaluation](#evaluation)
9. [Sample for a Custom Pocket](#sample-for-a-custom-pocket)
10. [Key Hyperparameters](#key-hyperparameters)
11. [Comparison with IPDiff](#comparison-with-ipdiff)

---

## Overview

Given a **protein binding pocket** (3D structure), IPFlow generates **drug-like ligand molecules** that bind tightly to that pocket. The model learns by integrating two components:

- **IPNet (BAPNet)** — a pretrained network that encodes interaction knowledge from thousands of real protein-ligand pairs with Vina Score supervision. Its per-atom features capture *where each atom should ideally be* to maximize binding affinity. The network is **frozen** throughout IPFlow training.

- **Flow Matching backbone** — an SE(3)-equivariant Transformer that learns a velocity field mapping Gaussian noise to valid ligand conformations. The velocity field is **conditioned on IPNet features** and further **shaped by a curved trajectory** that bends the ODE path toward high-affinity regions.

### Curved Trajectory Formulation

Standard flow matching uses a straight-line interpolant:

```
x_t = (1-t) * x_0 + t * x_1
u_t = x_0 - x_1
```

IPFlow bends this trajectory using a per-atom shift predicted from IPNet features:

```
x_t = (1-t) * x_0 + t * x_1 + g(t) * ψ_θ(F_M, t)
u_t = (x_0 - x_1) + (1-2t) * ψ_θ(F_M, t)
```

where `g(t) = t*(1-t)` is a bell-shaped gate ensuring the endpoints (`t=0` data, `t=1` noise) are unmodified, and `ψ_θ` (ShiftMLP) is a lightweight ~50K-parameter MLP that maps IPNet features and timestep to a 3D steering vector per atom.

---

## Architecture

```
Input: protein pocket (3D atoms) + noisy ligand x_t at time t
                          │
              ┌───────────┴───────────┐
              │                       │
       BAPNet / IPNet           FlowMatchingModel
       (FROZEN pretrained)      (trainable)
              │                       │
    F_M (ligand features)      SE(3)-equivariant
    F_P (protein features)     UniTransformerO2
    128-dim per atom                  │
              │                       │
              └───────────┬───────────┘
                          │
              Conditioning: concat F_M, F_P
              into hidden states → project
                          │
                    ShiftMLP ψ_θ
              F_M + t → delta (3D shift)
                          │
              Curved interpolant x_t
              Velocity target u_t
                          │
              Loss = MSE(v_pred, u_t)
                   + 100 * CE(type_pred, atom_types)
```

### Components

| Component | File | Description |
|---|---|---|
| **BAPNet / IPNet** | `../IPDiff/graphbap/bapnet.py` | Pretrained binding affinity network. Frozen. Outputs `F_M` (ligand) and `F_P` (protein) per-atom features. |
| **FlowMatchingModel** | `models/flow_model.py` | Main model. Embeds protein+ligand, injects IPNet features, runs backbone, predicts velocity and atom types. |
| **UniTransformerO2** | `../IPDiff/models/uni_transformer.py` | SE(3)-equivariant k-NN Transformer backbone. Reused directly from IPDiff. |
| **ShiftMLP** | `models/shift_mlp.py` | Lightweight MLP: `(F_M, t) → delta ∈ R^3`. Applies the interaction-prior curve to the ODE trajectory. |

---

## Project Structure

```
IPFlow/
├── models/
│   ├── flow_model.py           # FlowMatchingModel (main model)
│   ├── shift_mlp.py            # ShiftMLP — curved trajectory component
│   └── __init__.py
├── configs/
│   └── training.yml            # All hyperparameters
├── scripts/
│   ├── data_preparation/
│   │   ├── clean_crossdocked.py    # Step 1: RMSD filtering
│   │   ├── extract_pockets.py      # Step 2: pocket extraction
│   │   └── split_pl_dataset.py     # Step 3: train/test split
│   ├── evaluate_flow.py            # Compute QED / SA / Vina metrics
│   └── sample_for_pocket.py        # Generate for a custom .pdb pocket
├── train.py                    # Training loop
├── sample.py                   # Batch sampling over test set
└── README.md
```

IPFlow **imports utilities directly from IPDiff** via `sys.path` — no code duplication:
- `IPDiff/datasets/` — CrossDocked2020 dataset loader (LMDB)
- `IPDiff/utils/` — featurization, reconstruction, evaluation
- `IPDiff/graphbap/bapnet.py` — frozen IPNet

---

## Environment Setup

IPFlow uses the same conda environment as IPDiff:

```bash
conda env create -f ../IPDiff/ipdiff.yml
conda activate ipdiff
```

Key dependencies: PyTorch 1.10.1 + CUDA 11.3, PyTorch Geometric 2.0.4, RDKit 2022.03.5, OpenBabel 3.1.1, AutoDock Vina 1.2.2, LMDB.

---

## Data Preparation

All data preparation is identical to IPDiff. If you have already processed CrossDocked2020 for IPDiff, **skip this section entirely** and point `configs/training.yml` to the existing data paths.

**Step 1 — Filter by RMSD:**
```bash
python scripts/data_preparation/clean_crossdocked.py \
    --source /path/to/CrossDocked2020 \
    --dest   /path/to/crossdocked_v1.1_rmsd1.0 \
    --rmsd_thr 1.0
```

**Step 2 — Extract pockets (10 Å radius around ligand):**
```bash
python scripts/data_preparation/extract_pockets.py \
    --source /path/to/crossdocked_v1.1_rmsd1.0 \
    --dest   /path/to/crossdocked_pocket10
```

**Step 3 — Create train/test split:**
```bash
python scripts/data_preparation/split_pl_dataset.py \
    --path /path/to/crossdocked_pocket10 \
    --dest /path/to/crossdocked_pocket10_pose_split.pt \
    --fixed_split
```

Then edit `configs/training.yml`:
```yaml
data:
  path:  /path/to/crossdocked_pocket10
  split: /path/to/crossdocked_pocket10_pose_split.pt
```

---

## Training

**Edit `configs/training.yml`** to set:
- `data.path` — path to pocket-extracted CrossDocked2020
- `data.split` — path to split `.pt` file
- `net_cond.ckpt_path` — path to IPNet pretrained checkpoint (from IPDiff)

**Run (from `E:/BIO/IPFlow/`):**
```bash
python train.py \
    --config configs/training.yml \
    --device cuda \
    --logdir logs/
```

Optional flags:
```bash
--tag my_experiment          # append tag to log directory name
--train_report_iter 200      # log training loss every N iterations
```

**Output structure:**
```
logs/
└── training_<timestamp>_<tag>/
    ├── training.yml             # config snapshot
    ├── training.log             # full log
    └── checkpoints/
        ├── 5000.pt
        ├── 10000.pt
        └── ...                  # best validation checkpoint saved
```

**What the training loop does:**
1. Loads protein+ligand pairs from CrossDocked2020 LMDB.
2. For each batch, calls frozen IPNet to get `F_M`, `F_P` (with σ=0.5 noise on ligand positions to bridge the train/inference gap).
3. Samples `t ~ Uniform(0, 1)`, noise `x_1 ~ N(0, I)`.
4. Computes the curved interpolant `x_t` and velocity target `u_t`.
5. Runs the FM backbone, computes `loss_pos (MSE) + 100 * loss_v (CE)`.
6. Validates every 5000 iterations at 10 uniformly-spaced timesteps; saves checkpoint if validation loss improves.

---

## Sampling

Generate ligands for the 100 test proteins in CrossDocked2020:

```bash
python sample.py \
    --config     configs/training.yml \
    --checkpoint logs/<run>/checkpoints/<iter>.pt \
    --device     cuda:0 \
    --start_index 0 \
    --end_index   99 \
    --batch_size  25 \
    --num_samples 100 \
    --num_steps   100 \
    --result_path sampled_results/
```

Key arguments:
| Argument | Default | Description |
|---|---|---|
| `--num_steps` | 100 | Euler ODE integration steps (vs 1000 for IPDiff) |
| `--num_samples` | 100 | Ligands generated per protein |
| `--batch_size` | 25 | Ligands generated in one GPU forward pass |
| `--sample_num_atoms` | `prior` | Atom count sampling (`prior` / `ref` / `range`) |
| `--center_pos_mode` | `protein` | Center coordinates around protein CoM |

**Output:** one `.pt` file per protein in `result_path/`:
```
sampled_results/
├── result_0.pt
├── result_1.pt
└── ...   (result_99.pt)
```

Each `.pt` contains:
```python
{
  'data':                 <ProteinLigandData>,
  'pred_ligand_pos':      [array(N_atoms, 3), ...],   # 100 final positions
  'pred_ligand_v':        [array(N_atoms,), ...],     # 100 final atom types
  'pred_ligand_pos_traj': [array(100, N_atoms, 3), ...],  # full ODE trajectory
  'pred_ligand_v_traj':   [array(100, N_atoms,), ...],
  'time':                 [float, ...],               # wall-clock per batch
}
```

---

## Evaluation

Compute QED, SA, Vina Score, validity, uniqueness, and diversity:

```bash
python scripts/evaluate_flow.py sampled_results/ \
    --protein_root /path/to/crossdocked_v1.1_rmsd1.0 \
    --docking_mode vina_score \
    --atom_enc_mode add_aromatic
```

Arguments:
| Argument | Default | Description |
|---|---|---|
| `--docking_mode` | `vina_score` | `vina_score` (fast, no pose search) / `vina_dock` / `none` |
| `--eval_step` | `-1` | Trajectory step to evaluate; `-1` = final output |
| `--exhaustiveness` | `16` | Vina exhaustiveness (higher = slower, more accurate) |
| `--eval_num_examples` | all | Evaluate only first N result files |

**Output metrics:**
- **Validity** — fraction of generated molecules that are chemically valid
- **Uniqueness** — fraction of unique SMILES among valid molecules
- **Diversity** — average pairwise Tanimoto dissimilarity
- **QED** — drug-likeness score (0–1, higher is better)
- **SA** — synthetic accessibility score (lower is easier to synthesize)
- **Vina Score** — predicted binding affinity (kcal/mol, lower = tighter binding)
- **Vina < -6 / < -8** — fraction of molecules below affinity thresholds

Results are saved to `sampled_results/eval_results/metrics.pt`.

---

## Sample for a Custom Pocket

To generate ligands for any protein pocket given a `.pdb` file:

```bash
python scripts/sample_for_pocket.py \
    --config     configs/training.yml \
    --checkpoint logs/<run>/checkpoints/<iter>.pt \
    --pdb_path   /path/to/my_pocket.pdb \
    --num_samples 100 \
    --result_path ./outputs_pdb \
    --device      cuda
```

The `.pdb` file should contain only the binding pocket residues (≤ 10 Å from the expected ligand binding site). Use standard tools (PyMOL, MDAnalysis) to extract the pocket before running this script.

---

## Key Hyperparameters

All hyperparameters live in `configs/training.yml`.

| Parameter | Value | Notes |
|---|---|---|
| `hidden_dim` | 128 | Same as IPDiff |
| `num_layers` | 9 | Same as IPDiff |
| `n_heads` | 16 | Same as IPDiff |
| `knn` | 32 | k-NN graph connectivity |
| `batch_size` | 4 | Per-GPU |
| `lr` | 1e-3 | Adam |
| `loss_v_weight` | 100 | CE weight for atom types |
| `num_steps` (inference) | 100 | Euler ODE steps (vs 1000 diffusion steps) |
| `shift_hidden_dim` | 128 | ShiftMLP hidden size (~50K params) |
| `shift_time_emb_dim` | 64 | Sinusoidal time embedding in ShiftMLP |
| `ipnet_noise_sigma` | 0.5 | Ligand position noise for IPNet (training) |
| `infer_ipnet_noise_sigma` | 0.1 | Reduced noise for IPNet at inference |
| `cond_dim` | 128 | IPNet feature dimension |

---

## Comparison with IPDiff

| | IPDiff | IPFlow |
|---|---|---|
| **Generative model** | Diffusion (DDPM) | Flow Matching (ODE) |
| **Inference steps** | 1000 | ~100 |
| **Trajectory** | Stochastic reverse SDE | Deterministic Euler ODE |
| **Interaction prior** | Shift + feature concat | Curved trajectory + feature concat |
| **Shift gate** | `k_t = √ᾱ_t (1-√ᾱ_t)` | `g(t) = t(1-t)` |
| **Atom type prediction** | C0 direct prediction | CE on ground-truth types |
| **IPNet** | Frozen (same checkpoint) | Frozen (same checkpoint) |
| **Dataset** | CrossDocked2020 | CrossDocked2020 (same) |
| **Target metric** | Vina Score −6.42 avg | ≥ −6.42 (target to beat) |
#   I P F l o w  
 