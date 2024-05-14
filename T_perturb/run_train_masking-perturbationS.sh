#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=3' # request for exclusive access to gpu
#BSUB -n 32 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team361/ip14/Projects/2024Mar_Tperturb/T_perturb/T_perturb/ # working directory
#BSUB -o ../../logs/%J.out # output file
#BSUB -e ../../logs/%J.err # error file
#BSUB -M 64GB  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>64GB] rusage[mem=64GB]' # RAM memory part 1. Default: 100MB
#BSUB -J 7_train_masking_Norman2019 # job name

# activate conda environment
source ~/.bashrc
conda activate Tperturb

module load cuda-12.1.1
cd /lustre/scratch126/cellgen/team361/ip14/Projects/2024Mar_Tperturb/T_perturb/T_perturb
cwd=$(pwd)
export WANDB_DIR=$cwd/wandb

# run script
echo "--- Start computing model"
# # Run python script for rna
python3 $cwd/train.py \
--train_mode masking \
--split True \
--splitting_mode gears-simulation \
--src_dataset_folder ../../datasets/Norman2019/dataset/filtered_tokenised_hvg_pairing_GFpert_control.dataset \
--tgt_dataset_folder ../../datasets/Norman2019/dataset/filtered_tokenised_hvg_pairing_perturbed.dataset \
--src_adata_folder ../../datasets/Norman2019/adata/filtered_tokenised_hvg_pairing_control.h5ad \
--tgt_adata_folder ../../datasets/Norman2019/adata/filtered_tokenised_hvg_pairing_perturbed.h5ad \
--batch_size 45 \
--epochs 50 \
--max_len 1750 \
--petra_lr 0.001 \
--petra_wd 0.0001 \
--num_layers 5 \
--d_ff 16 \
--mlm_probability 0.3 \
--n_workers 32 \
--seed 1 \
--base_path /lustre/scratch126/cellgen/team361/ip14
echo "--- Finished computing model"
