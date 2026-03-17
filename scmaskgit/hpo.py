import argparse
import os
import uuid
from datetime import datetime

import pytorch_lightning as pl
import scanpy as sc
import torch
from datasets import load_from_disk, Dataset

# from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy  # ,DeepSpeedStrategy
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from tests.test_cellgen_training import dummy_dataset
import wandb
from scmaskgit.Dataloaders.datamodule import CellGenDataModule
from scmaskgit.Model.trainer import CellGenTrainer, CountDecoderTrainer
from scmaskgit.src.utils import (
    condition_for_count_loss,
    randomised_split,
    read_dataset_files,
    str2bool,
    stratified_split,
)
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

if os.getcwd().split('/')[-1] != 'scmaskgit':
    # set working directory to root of repository
    os.chdir('/lustre/scratch126/cellgen/team361/av13/scmaskgit')
    print('Changed working directory to root of repository')

os.environ["WANDB_CACHE_DIR"] = os.getcwd()
os.environ["WANDB_CONFIG_DIR"] = os.getcwd()
os.environ["WANDB_ARTIFACT_DIR"] = os.getcwd()
os.environ["WANDB_DATA_DIR"] = os.getcwd()
os.environ["WANDB_ARTIFACT_LOCATION"] = os.getcwd()

class _TuneReportCallback(TuneReportCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_mode',
        type=str,
        default='masking',
        help='Mode [masking, count]',
    )
    parser.add_argument(
        '--split',
        type=bool,
        default=False,
        help='split data for extrapolation',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/lustre/scratch126/cellgen/team361/av13/scmaskgit/output',
        help='store dataset name',
    )
    parser.add_argument(
        '--splitting_mode',
        type=str,
        default='random',
        # default='stratified',
        choices=['random', 'stratified', 'unseen_cond'],
        help='splitting mode',
    )
    parser.add_argument(
        '--split_obs',
        type=str,
        nargs='+',
        # default=['Donor', 'Cell_type'],
        default=['celltype_v2'],
    )
    parser.add_argument('--split_value', type=str, default='D351')
    parser.add_argument(
        '--ckpt_masking_path',
        type=str,
        default=None,
        help='path to checkpoint',
    )
    parser.add_argument(
        '--src_dataset',
        type=str,
        default='./T_perturb/perturbgen/pp/res/eb/dataset_hvg_src/Day 00-03.dataset',
        # default=(
        #     './T_perturb/perturbgen/pp/res/eb/'
        #     'dataset_all_src/eb_all_Day 00-03.dataset'
        # ),
        # default='./T_perturb/perturbgen/pp/res/cytoimmgen/dataset_hvg_src/0h.dataset',
        help='path to tokenised resting data',
    )
    parser.add_argument(
        '--src_adata',
        type=str,
        default='./T_perturb/perturbgen/pp/res/eb/h5ad_pairing_hvg_src/Day 00-03.h5ad',
        # default=(
        #     './T_perturb/perturbgen/pp/'
        #     'res/eb/h5ad_pairing_all_src/eb_all_Day 00-03.h5ad'
        # ),
        # default='./T_perturb/perturbgen/pp/res/cytoimmgen/'
        # 'h5ad_pairing_hvg_src/0h.h5ad',
        help='path to src',
    )
    parser.add_argument(
        '--tgt_adata_folder',
        type=str,
        default='./T_perturb/perturbgen/pp/res/eb/h5ad_pairing_hvg_tgt',
        # default='./T_perturb/perturbgen/pp/res/eb/h5ad_pairing_all_tgt',
        # default='./T_perturb/perturbgen/pp/res/cytoimmgen/h5ad_pairing_hvg_tgt',
        help='path to tgt',
    )
    parser.add_argument(
        '--mapping_dict_path',
        type=str,
        # default='./T_perturb/perturbgen/pp/res/eb/token_id_to_genename_hvg.pkl',
        # default='./T_perturb/perturbgen/pp/res/eb/token_id_to_genename_all.pkl'
        default='./T_perturb/perturbgen/pp/res/cytoimmgen/token_id_to_genename_hvg.pkl',
    )
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle')
    parser.add_argument(
        '--epochs', type=int, default=1, help='number of training epochs'
    )
    parser.add_argument(
        '--log_dir', type=str, default='logs', help='path to data directory'
    )
    parser.add_argument(
        '--max_len',
        type=int,
        # default=300,
        # default=2048,
        default=4096,
        help='max sequence length',
    )  # check how many genes there are
    parser.add_argument(
        '--tgt_vocab_size',
        type=int,
        # default=1261,
        # default=15280,
        default=24939,
        help='vocab size (max token id + 1) in dataset for padding',
    )
    parser.add_argument(
        '--cellgen_lr', type=float, default=0.0001, help='learning rate'
    )
    
    parser.add_argument('--count_lr', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--cellgen_wd', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--count_wd', type=float, default=0.01, help='weight decay')
    parser.add_argument(
        '--num_layers', type=int, default=6, help='number of decoder layers'
    )
    parser.add_argument('--d_ff', type=int, default=128, help='feed forward dimension')

    parser.add_argument('--mlm_prob', type=float, default=0.15, help='mlm probability')
    parser.add_argument(
        '--n_workers', type=int, default=8, help='number of workers'
    )  # 64
    parser.add_argument(
        '--loss_mode', type=str, default='zinb', help='loss mode [zinb, nb, mse]'
    )
    parser.add_argument('--cellgen_dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--count_dropout', type=float, default=0.0, help='dropout')
    parser.add_argument(
        '--condition_keys',
        nargs='+',
        default=None,
        # default='Cell_culture_batch',
        type=str,
        help='Selection of condition keys to use for model',
    )
    parser.add_argument(
        '--mask_scheduler',
        type=str,
        default='pow',
        help='mask scheduler [cosine, exp, pow]',
    )
    parser.add_argument('--temperature', type=float, default=1.5, help='temperature')
    parser.add_argument('--iterations', type=int, default=19, help='iterations')
    parser.add_argument('--conditions', type=dict, default=None, help='conditions')
    parser.add_argument(
        '--conditions_combined', type=list, default=None, help='conditions combined'
    )
    parser.add_argument(
        '--pred_tps',
        nargs='+',
        type=int,
        default=[1, 2, 3],
        help='time steps which are predicted',
    )
    parser.add_argument(
        '--context_tps',
        nargs='+',
        type=int,
        default=None,
        help='context time steps in cross-attn',
    )
    parser.add_argument(
        '--var_list',
        # type=list,
        nargs='+',
        type=str,
        # default=['Time_point'],
        # default=['Cell_population', 'Cell_type', 'Time_point', 'Donor'],
        # default=['celltype_v2', 'sex', 'phase', 'tissue', 'diff_state'],
        default=[],
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
        '--encoder',
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
        '--pos_encoding_mode',
        type=str,
        default='time_pos_sin',
        choices=['time_pos_sin', 'comb_sin', 'sin_learnt', 'time_pos_learnt'],
        help='positional encoding',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='seed for reproducibility',
    )
    parser.add_argument(
        '--context_mode',
        type=bool,
        default=True,
        help='context mode for timepoints',
    )
    args = parser.parse_args()
    return args

def train_model(config, args):
    """Training function for Ray Tune."""
    pl.seed_everything(args.seed)
    torch.manual_seed(args.seed)

    # Load and preprocess data
    print('Loading and preprocessing data...')
    src_dataset = load_from_disk("/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/perturbgen/pp/res/concatenated_all_25k_all_subsetted.dataset")
    src_dataset = src_dataset.select(range(1000000))
    # length = []
    # for i in range(len(src_dataset)):
    #     length.append(max(src_dataset[i]["input_ids"]))
    # print(max(length))
    # raise
    train_indices = list(range(len(src_dataset)))
    val_indices = None

    data_module_kwargs = {
        'src_dataset': src_dataset,
        'batch_size': int(config["batch_size"]),
        'num_workers': args.n_workers,
        'shuffle': args.shuffle,
        'max_len': args.max_len,
        'split': args.split,
        # 'train_indices': train_indices,
        # 'val_indices': val_indices,
        'test_indices': [],
        'pred_tps': args.pred_tps,
        'context_tps': args.context_tps,
        'var_list': args.var_list,
    }
    data_module = CellGenDataModule(**data_module_kwargs)

    trainer_kwargs = {
        'tgt_vocab_size': args.tgt_vocab_size,
        'd_model': int(config["d_model"]),
        'num_heads': int(config["num_heads"]),
        'num_layers': int(config["num_layers"]),
        'd_ff': int(config["d_ff"]),
        'max_seq_length': args.max_len + 100,
        'mask_scheduler': args.mask_scheduler,
        'pred_tps': args.pred_tps,
        'context_tps': args.context_tps,
        'encoder': args.encoder,
        'output_dir': args.output_dir,
        'context_mode': args.context_mode,
        'pos_encoding_mode': args.pos_encoding_mode,
        'dropout': config["dropout"],
        'mlm_probability': args.mlm_prob,
        'end_lr': config["learning_rate"],
        'weight_decay': config["weight_decay"],
        'mapping_dict_path': args.mapping_dict_path,
    }
    pretrained_module = CellGenTrainer(**trainer_kwargs)

    # Initialize logger
    run_id = datetime.now().strftime('%Y%m%d_%H%M_cellgen')
    log_path = os.path.join(args.log_dir, run_id)
    print(os.getcwd())
    os.makedirs(os.path.join(os.getcwd(), log_path), exist_ok=True)

    run_name = (
        f'{run_id}_{str(uuid.uuid4())[:6]}' if torch.cuda.device_count() > 1 else run_id
    )
    wandb_logger = WandbLogger(
        project="Moscf",
        name=run_name,
        save_dir=args.log_dir,
        settings=wandb.Settings(save_code=False),
        log_model=True,
    )
    tune_callback = _TuneReportCallback(
        {"train_perplexity": "train/perplexity"}, on="train_epoch_end"
    )

    # Early stopping and checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        save_top_k=1,
        monitor="train/perplexity_epoch",
        mode="min",
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="train/perplexity_epoch",
        min_delta=0.01,
        patience=5,
        mode="min",
    )

    # Initialize Trainer
    trainer = pl.Trainer(
        # logger=wandb_logger,
        callbacks=[tune_callback],
        max_epochs=2,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="bf16-mixed",
        devices=-1 if torch.cuda.is_available() else 0,
    )
    print(trainer.callback_metrics)
    # Training
    trainer.fit(pretrained_module, data_module)

def main():
    args = get_args()

    # Define hyperparameter search space

    search_space = {
    "d_model": tune.choice([256, 512, 768]),
    "num_heads": tune.choice([6, 8, 10]),
    "num_layers": tune.choice([4, 6, 8]),
    "d_ff": tune.choice([32, 64, 96]),
    "dropout": tune.choice([0.01, 0.02, 0.03, 0.07, 0.1]),  # Discretized values
    "learning_rate": tune.choice([1e-5, 5e-5, 1e-4, 5e-4]),  # Discrete values
    "weight_decay": tune.choice([1e-6, 1e-5, 1e-4]),  # Discrete values
    "batch_size": tune.choice([8, 16, 32]),
    # "max_epochs": tune.choice([5, 10, 15]),
}


    # search_space = {
    #     "d_model": 256,
    #     "num_heads": 4,
    #     "num_layers": 2,
    #     "d_ff": 512,
    #     "dropout": 0.1,
    #     "learning_rate": 1e-5,
    #     "weight_decay": 1e-6,
    #     "batch_size": 16,
    # }
    # train_model(search_space, args)
    # Set up BOHB
    bohb_search = TuneBOHB()
    bohb_scheduler = HyperBandForBOHB(time_attr="training_iteration", max_t=1)

    # # Run Ray Tune
    tune.run(
        tune.with_parameters(train_model, args=args),
        config=search_space,
        metric="train_perplexity",
        mode="min",
        scheduler=bohb_scheduler,
        search_alg=bohb_search,
        storage_path="/lustre/scratch126/cellgen/team361/av13/scmaskgit",
        num_samples=20,
        resources_per_trial={"cpu": 10, "gpu":1},
    )

if __name__ == "__main__":
    main()
