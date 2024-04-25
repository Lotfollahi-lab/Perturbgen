#!/bin/bash
#SBATCH -J 1_train_counts_Norman2019
#SBATCH -o ../../logs/1_Norman2019_train_counts.out
#SBATCH -e ../../logs/1_Norman2019_train_counts.err
#SBATCH -t 24:00:00
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_normal
#SBATCH -c 20
#SBATCH --mem=160G
#SBATCH --nice=10000


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
--train_mode count \
--ckpt_path "/lustre/groups/imm01/workspace/irene.bonafonte/Projects/2024Mar_Tperturb/T_perturb/T_perturb/Model/checkpoints/20240424_2248_petra_mode_masking_lr_0.001_wd_0.001_batch_16_mlmp_0.3_hvg_pairing_GFpert_epoch13.ckpt" \
--num_cells 0 \
--src_dataset_folder ../../datasets/Norman2019/dataset/filtered_tokenised_hvg_pairing_GFpert_control.dataset \
--tgt_dataset_folder ../../datasets/Norman2019/dataset/filtered_tokenised_hvg_pairing_perturbed.dataset \
--src_adata_folder ../../datasets/Norman2019/adata/filtered_tokenised_hvg_pairing_control.h5ad \
--tgt_adata_folder ../../datasets/Norman2019/adata/filtered_tokenised_hvg_pairing_perturbed.h5ad \
--batch_size 16 \
--split True \
--splitting_mode gears-simulation \
--epochs 15 \
--max_len 1750 \
--mlm_probability 0.3 \
--n_workers 16 \
--loss_mode zinb \
--petra_lr 0.001 \
--count_lr 0.001 \
--petra_wd 0.001 \
--count_wd 0.001
echo "--- Finished computing model"
