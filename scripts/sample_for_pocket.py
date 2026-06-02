"""
Generate ligands for a single protein pocket given a raw PDB file.

Unlike sample.py (which runs over the CrossDocked2020 test set),
this script accepts any .pdb file directly — useful for custom targets.

Usage (from E:/BIO/IPFlow/):
    python scripts/sample_for_pocket.py \
        --config    configs/training.yml \
        --checkpoint logs/<run>/checkpoints/<iter>.pt \
        --pdb_path  /path/to/pocket.pdb \
        --num_samples 100 \
        --result_path ./outputs_pdb \
        --device cuda
"""

import argparse
import os
import sys
import time

import torch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_mean

_IPFLOW_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(_IPFLOW_DIR))

import utils.misc as misc
import utils.transforms as trans
from datasets.pl_data import ProteinLigandData, torchify_dict
from graphbap.bapnet import BAPNet
from utils.data import PDBProtein
from utils.evaluation import atom_num

from models.flow_model import FlowMatchingModel
from sample import sample_flow_ligand


def pdb_to_pocket_data(pdb_path, protein_featurizer):
    pocket_dict = PDBProtein(pdb_path).to_dict_atom()
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict={
            'element':      torch.empty([0], dtype=torch.long),
            'pos':          torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index':   torch.empty([2, 0], dtype=torch.long),
            'bond_type':    torch.empty([0], dtype=torch.long),
        },
    )
    data = protein_featurizer(data)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',      type=str, required=True)
    parser.add_argument('--checkpoint',  type=str, required=True)
    parser.add_argument('--pdb_path',    type=str, required=True)
    parser.add_argument('--device',      type=str, default='cuda:0')
    parser.add_argument('--batch_size',  type=int, default=16)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--num_steps',   type=int, default=100)
    parser.add_argument('--result_path', type=str, default='./outputs_pdb')
    parser.add_argument('--center_pos_mode',  type=str, default='protein')
    parser.add_argument('--sample_num_atoms', type=str, default='prior')
    args = parser.parse_args()

    logger = misc.get_logger('sample_for_pocket')

    config = misc.load_config(args.config)
    ckpt   = torch.load(args.checkpoint, map_location=args.device)

    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer  = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)

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

    data = pdb_to_pocket_data(args.pdb_path, protein_featurizer)
    logger.info(f'Pocket: {args.pdb_path}  ({data.protein_pos.shape[0]} atoms)')

    os.makedirs(args.result_path, exist_ok=True)

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
        'data':                 data,
        'pred_ligand_pos':      pred_pos,
        'pred_ligand_v':        pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj':   pred_v_traj,
        'time':                 time_list,
    }
    pocket_name = os.path.splitext(os.path.basename(args.pdb_path))[0]
    out_path = os.path.join(args.result_path, f'result_{pocket_name}.pt')
    torch.save(result, out_path)
    logger.info(f'Saved {args.num_samples} samples to {out_path}')
    logger.info(f'Avg sampling time per batch: {sum(time_list)/len(time_list):.2f}s')
