"""
IPFlow sampling script.

Usage (from E:/BIO/IPFlow/):
    python sample.py \\
        --config configs/training.yml \\
        --checkpoint logs/<run>/checkpoints/<iter>.pt \\
        --device cuda \\
        --start_index 0 --end_index 99 \\
        --batch_size 25 \\
        --result_path sampled_results

The script runs Euler ODE sampling (t: 1 -> 0) for each test protein and saves
results as .pt files compatible with IPDiff's evaluation pipeline.
"""

import argparse
import os
import shutil
import time

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_mean, scatter_sum
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from graphbap.bapnet import BAPNet
from utils.evaluation import atom_num

# IPFlow model
from models.flow_model import FlowMatchingModel


# ---------------------------------------------------------------------------
# Unbatch trajectory helper
# ---------------------------------------------------------------------------
def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    return [np.stack(sv) for sv in all_step_v]


# ---------------------------------------------------------------------------
# Per-protein batch sampling
# ---------------------------------------------------------------------------
def sample_flow_ligand(
    model,
    data,
    num_samples: int,
    batch_size: int = 16,
    device: str = 'cuda:0',
    num_steps: int = 100,
    center_pos_mode: str = 'protein',
    sample_num_atoms: str = 'prior',
    net_cond=None,
):
    """
    Generate `num_samples` ligands for a single protein `data`.

    Returns lists parallel to IPDiff's sample_diffusion_ligand output for
    compatibility with eval_split.py.
    """
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0

    for i in tqdm(range(num_batch), desc='batch', leave=False):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch = Batch.from_data_list(
            [data.clone() for _ in range(n_data)],
            follow_batch=FOLLOW_BATCH,
        ).to(device)

        t1 = time.time()
        with torch.no_grad():
            batch_protein = batch.protein_element_batch

            # --- Determine number of ligand atoms per sample ---
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(batch.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [
                    atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)
                ]
                batch_ligand = torch.repeat_interleave(
                    torch.arange(n_data), torch.tensor(ligand_num_atoms)
                ).to(device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(
                    torch.arange(n_data), torch.tensor(ligand_num_atoms)
                ).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(
                    torch.ones_like(batch_ligand), batch_ligand, dim=0
                ).tolist()
            else:
                raise ValueError(f'Unknown sample_num_atoms: {sample_num_atoms}')

            # --- Initial positions: protein CoM + Gaussian noise ---
            center = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center = center[batch_ligand]
            init_ligand_pos = batch_center + torch.randn_like(batch_center)

            # --- Initial atom types: uniform random ---
            init_ligand_v = torch.randint(
                0, model.num_classes, (len(batch_ligand),), device=device
            )

            # --- ODE sampling ---
            r = model.sample_flow(
                net_cond=net_cond,
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,
                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                center_pos_mode=center_pos_mode,
            )

            ligand_pos = r['pos']
            ligand_v = r['v']
            ligand_pos_traj = r['pos_traj']
            ligand_v_traj = r['v_traj']

            # --- Unbatch results ---
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [
                ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]]
                for k in range(n_data)
            ]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_pred_pos_traj += [np.stack(sp) for sp in all_step_pos]

            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [
                ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]]
                for k in range(n_data)
            ]
            all_pred_v_traj += unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)

        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data

    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, time_list


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training.yml',
                        help='Training config (for model arch + data paths)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint .pt file')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of ligands to generate per protein')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Number of Euler ODE steps (default 100)')
    parser.add_argument('--result_path', type=str, default='sampled_results')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=99)
    parser.add_argument('--center_pos_mode', type=str, default='protein')
    parser.add_argument('--sample_num_atoms', type=str, default='prior')
    args = parser.parse_args()

    logger = misc.get_logger('sampling')

    # ----------------------------------------------------------------
    # Config & checkpoint
    # ----------------------------------------------------------------
    config = misc.load_config(args.config)
    logger.info(f'Config: {config}')

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    logger.info(f'Loaded checkpoint from {args.checkpoint}')

    # ----------------------------------------------------------------
    # Transforms & dataset
    # ----------------------------------------------------------------
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    dataset, subsets = get_dataset(config=config.data, transform=transform)
    _, test_set = subsets['train'], subsets['test']
    logger.info(f'Test set size: {len(test_set)}')

    # ----------------------------------------------------------------
    # Models
    # ----------------------------------------------------------------
    net_cond = BAPNet(
        ckpt_path=config.net_cond.ckpt_path,
        hidden_nf=config.net_cond.hidden_dim,
    ).to(args.device)

    model = FlowMatchingModel(
        config=config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    logger.info(f'Model loaded. Starting sampling [{args.start_index}, {args.end_index}]...')

    # ----------------------------------------------------------------
    # Sampling loop
    # ----------------------------------------------------------------
    os.makedirs(args.result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(args.result_path, 'training.yml'))

    for data_id in range(args.start_index, args.end_index + 1):
        data = test_set[data_id]

        pred_pos, pred_v, pred_pos_traj, pred_v_traj, time_list = sample_flow_ligand(
            model=model,
            data=data,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=args.device,
            num_steps=args.num_steps,
            center_pos_mode=args.center_pos_mode,
            sample_num_atoms=args.sample_num_atoms,
            net_cond=net_cond,
        )

        result = {
            'data': data,
            'pred_ligand_pos': pred_pos,
            'pred_ligand_v': pred_v,
            'pred_ligand_pos_traj': pred_pos_traj,
            'pred_ligand_v_traj': pred_v_traj,
            'time': time_list,
        }

        out_path = os.path.join(args.result_path, f'result_{data_id}.pt')
        torch.save(result, out_path)
        logger.info(f'Saved result for data_id={data_id} -> {out_path}')
        print(f'sampled data_id: {data_id}')
