#!/bin/bash
#SBATCH -J 3.4_test_counts_Norman2019
#SBATCH -o ../../logs/3.4_Norman2019_test_counts.out
#SBATCH -e ../../logs/3.4_Norman2019_test_counts.err
#SBATCH -t 12:00:00
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_normal
#SBATCH -c 20
#SBATCH --mem=160G
#SBATCH --nice=10000
#SBATCH --constraint=a100_80gb|a100_40gb

# activate conda environment
source ~/.bashrc
conda activate Tperturb
cd /lustre/groups/imm01/workspace/irene.bonafonte/Projects/2024Mar_Tperturb/T_perturb/T_perturb
cwd=$(pwd)
export WANDB_DIR=$cwd/wandb
export WANDB_API_KEY=$(cat ~/wandb_api_key.txt)
ulimit -n 4096

# run script
echo "--- Start testing model"
# # Run python script for rna
python3 $cwd/val.py \
--ckpt_masking_file 20240428_2344_petra_mode_masking_lr_0.001_wd_0.001_batch_32_mlmp_0.3_hvg_pairing_GFpert.ckpt \
--ckpt_count_file 20240506_1319_petra_mode_count_lr_0.001_wd_0.001_batch_55_zinb_hvg_pairing_GFpert.ckpt \
--num_cells 0 \
--src_dataset_folder ../../datasets/Norman2019/dataset/filtered_tokenised_hvg_pairing_GFpert_control.dataset \
--tgt_dataset_folder ../../datasets/Norman2019/dataset/filtered_tokenised_hvg_pairing_perturbed.dataset \
--src_adata_folder ../../datasets/Norman2019/adata/filtered_tokenised_hvg_pairing_control.h5ad \
--tgt_adata_folder ../../datasets/Norman2019/adata/filtered_tokenised_hvg_pairing_perturbed.h5ad \
--batch_size 32 \
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
--count_wd 0.001 \
--seed 1 \
--base_path /lustre/groups/imm01/workspace/irene.bonafonte/
echo "--- Finished computing model"
