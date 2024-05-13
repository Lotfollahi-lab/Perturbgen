#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=2' # request for exclusive access to gpu
#BSUB -n 32 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team361/ip14/Projects/2024Mar_Tperturb/T_perturb/T_perturb/ # working directory
#BSUB -o ../../logs/%J.out # output file
#BSUB -e ../../logs/%J.err # error file
#BSUB -M 64GB  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>64GB] rusage[mem=64GB]' # RAM memory part 1. Default: 100MB
#BSUB -J 4.1_test_count_Norman2019 # job name

# activate conda environment
source ~/.bashrc
conda activate Tperturb
module load cuda-12.1.1
cd /lustre/groups/imm01/workspace/irene.bonafonte/Projects/2024Mar_Tperturb/T_perturb/T_perturb
cwd=$(pwd)
export WANDB_DIR=$cwd/wandb

# run script
echo "--- Start testing model"
# # Run python script for rna
python3 $cwd/val.py \
--ckpt_masking_file  20240512_1803_petra_mode_masking_lr_0.001_wd_0.001_batch_45_mlmp_0.3_seed1_hvg_pairing_GFpert.ckpt \
--ckpt_count_file  20240513_0955_petra_mode_count_lr_0.001_wd_0.001_batch_55_mse_seed1_hvg_pairing_GFpert.ckpt \
--src_dataset_folder ../../datasets/Norman2019/dataset/filtered_tokenised_hvg_pairing_GFpert_control.dataset \
--tgt_dataset_folder ../../datasets/Norman2019/dataset/filtered_tokenised_hvg_pairing_perturbed.dataset \
--src_adata_folder ../../datasets/Norman2019/adata/filtered_tokenised_hvg_pairing_control.h5ad \
--tgt_adata_folder ../../datasets/Norman2019/adata/filtered_tokenised_hvg_pairing_perturbed.h5ad \
--batch_size 55 \
--split True \
--splitting_mode gears-simulation \
--max_len 1750 \
--n_workers 32 \
--loss_mode mse \
--seed 1 \
--base_path /lustre/scratch126/cellgen/team361/ip14
echo "--- Finished computing model"
