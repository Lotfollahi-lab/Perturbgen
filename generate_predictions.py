import pytorch_lightning as pl

import argparse
import os

import torch
from T_perturb.Model.trainer import CellGenTrainer, CountDecoderTrainer
from T_perturb.Modules.T_model_GFE import CellGen, CountDecoder
from tqdm import tqdm
from T_perturb.Dataloaders.datamodule import CellGenDataModule
from T_perturb.src.utils import read_dataset_files

from datasets import load_from_disk
import scanpy as sc
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_layers', type=int, default=6, help='number of decoder layers'
    )
    parser.add_argument(
        '--src_adata',
        type=str,
        default='./T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_src/Day 00-03.h5ad',

        help='path to src',
    )
    parser.add_argument(
        '--tgt_adata',
        type=str,
        default='./T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_tgt',

        help='path to tgt',
    )

    parser.add_argument(
        '--output_dir',
        type=str,

        default='./T_perturb/T_perturb/plt/res/norman',
        help='store dataset name',
    )
    parser.add_argument(
        '--splitting_mode',
        type=str,
        default='random',

        choices=['random', 'stratified', 'unseen_cond'],
        help='splitting mode',
    )
    parser.add_argument(
        '--ckpt_masking_path',
        type=str,
        default=None,
        help='path to checkpoint',
    )

    parser.add_argument(
        '--src_dataset',
        type=str,
        default='./T_perturb/T_perturb/pp/res/eb/dataset_hvg_src/Day 00-03.dataset',
        help='path to tokenised resting data',
    )
    parser.add_argument(
        '--tgt_dataset',
        type=str,
        default='./T_perturb/T_perturb/pp/res/eb/dataset_hvg_tgt',
        help='path to tokenised activated data',
    )
    parser.add_argument(
        '--cellgen_ckpt_path',
        type=str,
        default='/lustre/scratch126/cellgen/team205/bair/diseaseCG/T_perturb/T_perturb/Model/checkpoints/20240917_2121_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_32_mlmp_0.15_ntask_2_s_100-epoch=19.ckpt',
        help='path to cellgen_ckpt_path',
    )
    parser.add_argument(
        '--count_decoder_ckpt_path',
        type=str,
        default='/lustre/scratch126/cellgen/team205/bair/diseaseCG/T_perturb/T_perturb/Model/checkpoints/20240918_2227_cellgen_train_count_lr_0.0001_wd_0.0001_batch_32_zinb_ntask_2_s_42-epoch=19.ckpt',
        help='path to count_decoder_ckpt_path',
    )
    parser.add_argument(
        '--max_len',
        type=int,
        default=263,
        help='max sequence length',
    )  
    parser.add_argument(
        '--tgt_vocab_size',
        type=int,
        default=2001,
        help='vocab size (max token id + 1) in dataset for padding',
    )
    parser.add_argument(
        '--cellgen_lr', type=float, default=0.0001, help='learning rate'
    )
    parser.add_argument('--d_ff', type=int, default=128, help='feed forward dimension')

    parser.add_argument('--mlm_prob', type=float, default=0.15, help='mlm probability')
    parser.add_argument(
        '--n_workers', type=int, default=32, help='number of workers'
    )  
    parser.add_argument(
        '--loss_mode', type=str, default='zinb', help='loss mode [zinb, nb, mse]'
    )
    parser.add_argument('--cellgen_dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--count_dropout', type=float, default=0.0, help='dropout')
    parser.add_argument(
        '--condition_keys',
        nargs='+',
        default=None,

        type=str,
        help='Selection of condition keys to use for model',
    )
    parser.add_argument('--conditions', type=dict, default=None, help='conditions')
    parser.add_argument(
        '--conditions_combined', type=list, default=None, help='conditions combined'
    )
    parser.add_argument(
        '--n_task_conditions',

        type=int,
        default=2,
        help='Number of task tokens corresponds to number of MoE classes',
    )
    parser.add_argument(
        '--var_list',

        nargs='+',
        type=str,

        default=['Cell_population', 'Cell_type', 'Time_point', 'Donor'],
        help='List of variables to keep in the dataset',
    )
    parser.add_argument(
        '--train_prop',
        type=float,
        default=0.8,
    )
    parser.add_argument(
        '--test_prop',
        type=float,
        default=0.1,
    )
    parser.add_argument(
        '--encoder_type',
        default='GF_frozen',
        type=str,
        choices=[
            'GF_fine_tuned',
            'GF_frozen',
            'Transformer_encoder',
        ],
        help='mode of encoder',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='seed for reproducibility',
    )
    args = parser.parse_args()
    return args

def main() -> None:
    args = get_args()
    pl.seed_everything(args.seed)
    torch.manual_seed(args.seed)

    tgt_dataset = load_from_disk(args.tgt_dataset)
    src_dataset = load_from_disk(args.src_dataset)

    src_adata = sc.read_h5ad(args.src_adata)
    src_counts = src_adata.X

    print("start generating predictions")

    cellgen_model = CellGen(
        tgt_vocab_size=args.tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_length=args.max_len,
        dropout=0.0,
        mlm_probability=args.mlm_prob,
        n_task_conditions=args.n_task_conditions,
        encoder_type=args.encoder_type,
        moe_type='none',
    )

    cellgen_checkpoint = torch.load(args.cellgen_ckpt_path, map_location='cpu')
    cellgen_model.load_state_dict(cellgen_checkpoint['state_dict'], strict=False)
    cellgen_model.eval()

    print("Loaded the masking")

    count_decoder = CountDecoder(
        pretrained_model=cellgen_model,
        loss_mode=args.loss_mode,
        n_genes=1424,
        tgt_vocab_size=args.tgt_vocab_size,
        d_model=512,
        dropout=0.25,
        time_steps=[1],
        n_task_conditions=2,
    )

    count_decoder_checkpoint = torch.load(args.count_decoder_ckpt_path, map_location='cpu')
    count_decoder.load_state_dict(count_decoder_checkpoint['state_dict'], strict=False)
    count_decoder.eval()

    print("Loaded the count decoder")

    tgt_adatas = read_dataset_files(args.tgt_adata, 'h5ad')

    tgt_counts_dict = {}
    for keys, tgt_adata in tgt_adatas.items():
        tgt_counts_dict[keys] = tgt_adata.X

    data_module = CellGenDataModule(
        src_dataset=src_dataset,
        tgt_dataset=tgt_dataset,
        tgt_adata=tgt_adata,
        src_counts=src_counts,
        tgt_counts_dict=tgt_counts_dict,
        batch_size=32,
        num_workers=16,
        shuffle=False,
        max_len=args.max_len,
        split=False,
        var_list=['Time_point'],
    )

    print("Checkpoint 1")

    data_module.setup('test')

    print("Checkpoint 1.5")

    test_dataloader = data_module.test_dataloader()

    print("Checkpoint 2")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cellgen_model.to(device)
    count_decoder.to(device)

    predicted_counts = []
    true_counts = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):

            src_input_id = batch['src_input_ids'].to(device)

            cellgen_outputs = cellgen_model(
                src_input_id=src_input_id,
                apply_attn_mask=False,
                generate=False,
            )

            embeddings = cellgen_outputs['mean_embedding']

            count_outputs = count_decoder.count_decoder(embeddings)

            pred_counts = count_outputs['count_mean']  

            predicted_counts.append(pred_counts.cpu())
            true_counts.append(batch['tgt_counts'].cpu())

    print("Checkpoint 3")

    predicted_counts = torch.cat(predicted_counts, dim=0)
    true_counts = torch.cat(true_counts, dim=0)

    print("Checkpoint 4")

    torch.save({'predicted_counts': predicted_counts, 'true_counts': true_counts}, 'predictions.pt')

    print("Checkpoint 5")

    predicted_counts_np = predicted_counts.numpy()
    true_counts_np = true_counts.numpy()

    mse = mean_squared_error(true_counts_np.flatten(), predicted_counts_np.flatten())
    print(f"Mean Squared Error (MSE): {mse}")

    mae = mean_absolute_error(true_counts_np.flatten(), predicted_counts_np.flatten())
    print(f"Mean Absolute Error (MAE): {mae}")

    pearson_corr, _ = pearsonr(true_counts_np.flatten(), predicted_counts_np.flatten())
    print(f"Pearson Correlation Coefficient: {pearson_corr}")

    spearman_corr, _ = spearmanr(true_counts_np.flatten(), predicted_counts_np.flatten())
    print(f"Spearman Rank Correlation: {spearman_corr}")

    r2 = r2_score(true_counts_np.flatten(), predicted_counts_np.flatten())
    print(f"R-squared: {r2}")

if __name__ == '__main__':
    main()