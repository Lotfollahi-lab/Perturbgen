#!/bin/bash
#SBATCH -J 3_train_masking_Norman2019
#SBATCH -o ../../logs/3_Norman2019_train_masking.out
#SBATCH -e ../../logs/3_Norman2019_train_masking.err
#SBATCH -t 24:00:00
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_normal
#SBATCH -c 20
#SBATCH --mem=160G
#SBATCH --nice=10000
#SBATCH --constraint=a100_80gb

# activate conda environment
source ~/.bashrc
conda activate Tperturb
cd /lustre/groups/imm01/workspace/irene.bonafonte/Projects/2024Mar_Tperturb/T_perturb/T_perturb
cwd=$(pwd)
export WANDB_DIR=$cwd/wandb
export WANDB_API_KEY=$(cat ~/wandb_api_key.txt)
ulimit -n 4096

# run script
echo "--- Start computing model"
# # Run python script for rna
python3 $cwd/train.py \
--train_mode masking \
--num_cells 0 \
--split True \
--splitting_mode gears-simulation \
--src_dataset_folder ../../datasets/Norman2019/dataset/filtered_tokenised_hvg_pairing_GFpert_control.dataset \
--tgt_dataset_folder ../../datasets/Norman2019/dataset/filtered_tokenised_hvg_pairing_perturbed.dataset \
--src_adata_folder ../../datasets/Norman2019/adata/filtered_tokenised_hvg_pairing_control.h5ad \
--tgt_adata_folder ../../datasets/Norman2019/adata/filtered_tokenised_hvg_pairing_perturbed.h5ad \
--batch_size 32 \
--epochs 50 \
--max_len 1750 \
--petra_lr 0.001 \
--petra_wd 0.001 \
--count_wd 0.001 \
--mlm_probability 0.3 \
--n_workers 20 \
--seed 1 \
--loss_mode mse \
--base_path /lustre/groups/imm01/workspace/irene.bonafonte/
echo "--- Finished computing model"
