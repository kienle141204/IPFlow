"""
IPFlow training script.

Usage (from E:/BIO/IPFlow/):
    python train.py --config configs/training.yml --device cuda --logdir logs/

The script:
  1. Loads CrossDocked2020 via PocketLigandPairDataset (reused from IPDiff).
  2. Loads the frozen IPNet/BAPNet checkpoint.
  3. Trains FlowMatchingModel with curved-trajectory FM loss.
  4. Saves best checkpoint based on validation loss.
"""

import argparse
import os
import shutil

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from graphbap.bapnet import BAPNet

# IPFlow model
from models.flow_model import FlowMatchingModel


# ---------------------------------------------------------------------------
# AUROC helper (for validation logging)
# ---------------------------------------------------------------------------
def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.0
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL,
        }
        print(f'atom: {mapping[feat_mode][c]}  auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Config & logging
    # ----------------------------------------------------------------
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)

    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    # ----------------------------------------------------------------
    # Transforms & dataset
    # ----------------------------------------------------------------
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(config=config.data, transform=transform)
    train_set, val_set = subsets['train'], subsets['test']
    logger.info(f'Training: {len(train_set)}  Validation: {len(val_set)}')

    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys,
    ))
    val_loader = DataLoader(
        val_set,
        batch_size=config.train.batch_size,
        shuffle=False,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys,
    )

    # ----------------------------------------------------------------
    # Models
    # ----------------------------------------------------------------
    logger.info('Building models...')

    # Frozen IPNet
    net_cond = BAPNet(
        ckpt_path=config.net_cond.ckpt_path,
        hidden_nf=config.net_cond.hidden_dim,
    ).to(args.device)

    # IPFlow model
    model = FlowMatchingModel(
        config=config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(args.device)

    logger.info(
        f'protein feature dim: {protein_featurizer.feature_dim}  '
        f'ligand feature dim: {ligand_featurizer.feature_dim}'
    )
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer & scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    # ----------------------------------------------------------------
    # Training step
    # ----------------------------------------------------------------
    def train(it):
        model.train()
        optimizer.zero_grad()

        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator).to(args.device)

            # Optional protein position noise augmentation
            protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
            gt_protein_pos = batch.protein_pos + protein_noise

            results = model.get_flow_loss(
                net_cond=net_cond,
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,
                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch,
            )
            loss = results['loss'] / config.train.n_acc_batch
            loss.backward()

        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        if it % args.train_report_iter == 0:
            logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                    it,
                    results['loss'].item(),
                    results['loss_pos'].item(),
                    results['loss_v'].item(),
                    optimizer.param_groups[0]['lr'],
                    orig_grad_norm,
                )
            )

    # ----------------------------------------------------------------
    # Validation step
    # ----------------------------------------------------------------
    def validate(it):
        sum_loss, sum_loss_pos, sum_loss_v, sum_n = 0.0, 0.0, 0.0, 0
        all_pred_v, all_true_v = [], []

        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
                batch_size = batch.num_graphs

                # Evaluate at 10 uniformly spaced timesteps
                for t_val in np.linspace(0.05, 0.95, 10):
                    t = torch.full((batch_size,), t_val, device=args.device, dtype=torch.float32)

                    results = model.get_flow_loss(
                        net_cond=net_cond,
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch.protein_element_batch,
                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,
                        t=t,
                    )

                    sum_loss += float(results['loss']) * batch_size
                    sum_loss_pos += float(results['loss_pos']) * batch_size
                    sum_loss_v += float(results['loss_v']) * batch_size
                    sum_n += batch_size

                    pred_type_prob = torch.softmax(results['pred_type'], dim=-1)
                    all_pred_v.append(pred_type_prob.detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        atom_auroc = get_auroc(
            np.concatenate(all_true_v),
            np.concatenate(all_pred_v, axis=0),
            feat_mode=config.data.transform.ligand_atom_mode,
        )

        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Avg atom auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, atom_auroc,
            )
        )
        return avg_loss

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    try:
        best_loss, best_iter = None, None
        for it in range(1, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = validate(it)
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, f'{it}.pt')
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                else:
                    logger.info(
                        f'[Validate] Val loss not improved. '
                        f'Best: {best_loss:.6f} at iter {best_iter}'
                    )
    except KeyboardInterrupt:
        logger.info('Terminating...')
