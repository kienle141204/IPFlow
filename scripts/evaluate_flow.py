"""
Evaluate IPFlow generated ligands.

Loads .pt result files from sample.py, reconstructs molecules via RDKit,
computes QED, SA, Vina Score, and diversity metrics.

Usage (from E:/BIO/IPFlow/):
    python scripts/evaluate_flow.py <sample_path> \
        --protein_root /path/to/crossdocked_v1.1_rmsd1.0 \
        --docking_mode vina_score \
        --atom_enc_mode add_aromatic
"""

import argparse
import os
import sys
from glob import glob
from collections import Counter

import numpy as np
import torch
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

_IPFLOW_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(_IPFLOW_DIR))

from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_vina import VinaDockingTask


def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = sum(1 for c in all_ring_sizes if ring_size in c)
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', type=str)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--eval_step', type=int, default=-1,
                        help='Which trajectory step to evaluate (-1 = final)')
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--protein_root', type=str,
                        default='./data/crossdocked_v1.1_rmsd1.0')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str,
                        choices=['vina_score', 'vina_dock', 'none'], default='vina_score')
    parser.add_argument('--exhaustiveness', type=int, default=16)
    args = parser.parse_args()

    result_path = os.path.join(args.sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # Load .pt result files
    results_fn_list = sorted(
        glob(os.path.join(args.sample_path, '*result_*.pt')),
        key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]),
    )
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]
    num_examples = len(results_fn_list)
    logger.info(f'Loaded {num_examples} result files from {args.sample_path}')

    num_samples = 0
    all_mol_stable, all_atom_stable = 0, 0
    num_mol_stable, num_atoms = 0, 0
    all_results = []
    all_smiles_list = []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()

    for example_idx, result_fn in enumerate(tqdm(results_fn_list, desc='Evaluate')):
        result = torch.load(result_fn, map_location='cpu')
        data = result['data']

        # Pick trajectory step
        if args.eval_step == -1:
            pred_pos  = result['pred_ligand_pos']
            pred_v    = result['pred_ligand_v']
        else:
            pred_pos  = [t[args.eval_step] for t in result['pred_ligand_pos_traj']]
            pred_v    = [t[args.eval_step] for t in result['pred_ligand_v_traj']]

        num_samples += len(pred_pos)

        for sample_idx, (pos, v) in enumerate(zip(pred_pos, pred_v)):
            try:
                pred_atom_type = transforms.get_atomic_number_from_index(
                    v, mode=args.atom_enc_mode
                )
                all_atom_types += Counter(pred_atom_type)

                # Bond length stats
                pair_dist = eval_bond_length.pair_distance_from_pos_v(pos, pred_atom_type)
                success_pair_dist.append(pair_dist)
                success_atom_types += Counter(pred_atom_type)

                # Reconstruct molecule
                mol = reconstruct.reconstruct_from_generated(pos, pred_atom_type)
                smiles = Chem.MolToSmiles(mol)

                # Validity check
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                largest_frag = max(mol_frags, key=lambda m: m.GetNumAtoms())
                smiles = Chem.MolToSmiles(largest_frag)

                all_smiles_list.append(smiles)

                # Compute molecular properties
                qed_score = scoring_func.compute_qed(largest_frag)
                sa_score  = scoring_func.compute_sa(largest_frag)
                logp      = scoring_func.compute_logp(largest_frag)
                lipinski  = scoring_func.compute_lipinski(largest_frag)

                # Docking
                vina_score = None
                if args.docking_mode == 'vina_score':
                    protein_fn = os.path.join(
                        args.protein_root, data.protein_filename
                    )
                    if os.path.exists(protein_fn):
                        try:
                            vina_task = VinaDockingTask.from_generated_mol(
                                largest_frag, protein_fn
                            )
                            vina_results = vina_task.run(mode='score_only',
                                                         exhaustiveness=args.exhaustiveness)
                            vina_score = vina_results[0]['affinity']
                        except Exception:
                            pass

                all_results.append({
                    'mol': largest_frag,
                    'smiles': smiles,
                    'qed': qed_score,
                    'sa': sa_score,
                    'logp': logp,
                    'lipinski': lipinski,
                    'vina': vina_score,
                })

            except Exception:
                all_smiles_list.append(None)
                continue

    # Aggregate metrics
    logger.info(f'Total samples: {num_samples}')
    valid_results = [r for r in all_results if r is not None]
    valid_smiles  = [r['smiles'] for r in valid_results]

    validity  = len(valid_results) / num_samples
    diversity = analyze.get_diversity(valid_smiles) if len(valid_smiles) > 1 else 0.0
    uniqueness = len(set(valid_smiles)) / len(valid_smiles) if valid_smiles else 0.0

    logger.info(f'Validity:   {validity:.4f}')
    logger.info(f'Uniqueness: {uniqueness:.4f}')
    logger.info(f'Diversity:  {diversity:.4f}')

    for metric in ['qed', 'sa', 'logp', 'lipinski']:
        vals = [r[metric] for r in valid_results if r[metric] is not None]
        if vals:
            logger.info(f'{metric.upper()}: mean={np.mean(vals):.4f}  median={np.median(vals):.4f}')

    vina_vals = [r['vina'] for r in valid_results if r['vina'] is not None]
    if vina_vals:
        logger.info(f'Vina Score: mean={np.mean(vina_vals):.4f}  median={np.median(vina_vals):.4f}')
        logger.info(f'Vina < -6:  {sum(1 for v in vina_vals if v < -6) / len(vina_vals):.4f}')
        logger.info(f'Vina < -8:  {sum(1 for v in vina_vals if v < -8) / len(vina_vals):.4f}')

    if args.save:
        save_path = os.path.join(result_path, 'metrics.pt')
        torch.save({
            'all_results': all_results,
            'validity': validity,
            'uniqueness': uniqueness,
            'diversity': diversity,
        }, save_path)
        logger.info(f'Saved metrics to {save_path}')
