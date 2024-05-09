#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=2' # request for exclusive access to gpu
#BSUB -n 32 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team361/ip14/Projects/2024Mar_Tperturb/T_perturb/T_perturb/ # working directory
#BSUB -o ../../logs/%J.out # output file
#BSUB -e ../../logs/%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J 4_train_masking_Norman2019 # job name

# activate conda environment
source ~/.bashrc
conda activate Tperturb

module load cuda-12.1.1
cd /lustre/scratch126/cellgen/team361/ip14/Projects/2024Mar_Tperturb/T_perturb/T_perturb
export WANDB_DIR=$cwd/wandb

# run script
echo "--- Start testing model"
# # Run python script for rna
python3 $cwd/val.py \
--ckpt_masking_file "20240428_2344_petra_mode_masking_lr_0.001_wd_0.001_batch_32_mlmp_0.3_hvg_pairing_GFpert.ckpt" \
--ckpt_count_file "20240506_1234_petra_mode_count_lr_0.001_wd_0.001_batch_55_mse_hvg_pairing_GFpert.ckpt" \
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
--loss_mode mse \
--petra_lr 0.001 \
--count_lr 0.001 \
--petra_wd 0.001 \
--count_wd 0.001 \
--seed 1 \
--base_path /lustre/groups/imm01/workspace/irene.bonafonte/
echo "--- Finished computing model"