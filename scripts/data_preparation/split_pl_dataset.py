"""
Data preparation step 3: Create train/test split (CrossDocked2020 standard split).

Usage (from E:/BIO/IPFlow/):
    python scripts/data_preparation/split_pl_dataset.py \
        --path        /path/to/crossdocked_pocket10 \
        --dest        /path/to/crossdocked_pocket10_pose_split.pt \
        --fixed_split /path/to/split_by_name.pt
"""

import os
import sys
import argparse
import random

import torch
from torch.utils.data import Subset
from tqdm.auto import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datasets.pl_pair_dataset import PocketLigandPairDataset


def get_chain_name(fn):
    return os.path.basename(fn)[:6]


def get_pdb_name(fn):
    return os.path.basename(fn)[:4]


def get_unique_pockets(dataset, raw_id, used_pdb, num_pockets, seed):
    unique_id = []
    pdb_visited = set()
    for idx in tqdm(raw_id, 'Filter'):
        pdb_name = get_pdb_name(dataset[idx].ligand_filename)
        if pdb_name not in used_pdb and pdb_name not in pdb_visited:
            unique_id.append(idx)
            pdb_visited.add(pdb_name)

    print('Number of Pairs: %d' % len(unique_id))
    print('Number of PDBs:  %d' % len(pdb_visited))
    random.Random(seed).shuffle(unique_id)
    unique_id = unique_id[:num_pockets]
    print('Number of selected: %d' % len(unique_id))
    return unique_id, pdb_visited.union(used_pdb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./data/crossdocked_v1.1_rmsd1.0_pocket10')
    parser.add_argument('--dest', type=str, default='./data/crossdocked_pocket10_pose_split.pt')
    parser.add_argument('--fixed_split', type=str, default='./data/split_by_name.pt')
    parser.add_argument('--train', type=int, default=100000)
    parser.add_argument('--val', type=int, default=1000)
    parser.add_argument('--test', type=int, default=20000)
    parser.add_argument('--val_num_pockets', type=int, default=-1)
    parser.add_argument('--test_num_pockets', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()

    dataset = PocketLigandPairDataset(args.path)
    print('Load dataset successfully!')

    if args.fixed_split:
        fixed_split = torch.load(args.fixed_split)
        print('Load fixed split successfully!')
        name_id_dict = {}
        for idx, data in enumerate(tqdm(dataset, desc='Indexing')):
            name_id_dict[data.protein_filename + data.ligand_filename] = idx

        selected_ids = {'train': [], 'test': []}
        for split in ['train', 'test']:
            print(f'Selecting {split} split...')
            for fn in fixed_split[split]:
                key = fn[0] + fn[1]
                if key in name_id_dict:
                    selected_ids[split].append(name_id_dict[key])
                else:
                    print(f'Warning: {fn[0]} / {fn[1]} not found!')
        train_id, val_id, test_id = selected_ids['train'], [], selected_ids['test']
    else:
        allowed_elements = {1, 6, 7, 8, 9, 15, 16, 17}
        elements = {i: set() for i in range(90)}
        for i, data in enumerate(tqdm(dataset, desc='Filter')):
            for e in data.ligand_element:
                elements[e.item()].add(i)

        all_id = set(range(len(dataset)))
        blocked_id = set().union(*[elements[i] for i in elements if i not in allowed_elements])
        allowed_id = list(all_id - blocked_id)
        random.Random(args.seed).shuffle(allowed_id)
        print('Allowed: %d' % len(allowed_id))

        train_id = allowed_id[:args.train]
        train_set = Subset(dataset, indices=train_id)
        train_pdb = {get_pdb_name(d.ligand_filename) for d in tqdm(train_set)}

        if args.val_num_pockets == -1:
            val_id = allowed_id[args.train: args.train + args.val]
            used_pdb = train_pdb
        else:
            raw_val_id = allowed_id[args.train: args.train + args.val]
            val_id, used_pdb = get_unique_pockets(dataset, raw_val_id, train_pdb, args.val_num_pockets, args.seed)

        if args.test_num_pockets == -1:
            test_id = allowed_id[args.train + args.val: args.train + args.val + args.test]
        else:
            raw_test_id = allowed_id[args.train + args.val: args.train + args.val + args.test]
            test_id, _ = get_unique_pockets(dataset, raw_test_id, used_pdb, args.test_num_pockets, args.seed)

    torch.save({'train': train_id, 'val': val_id, 'test': test_id}, args.dest)
    print('Train %d, Val %d, Test %d.' % (len(train_id), len(val_id), len(test_id)))
    print('Done.')
