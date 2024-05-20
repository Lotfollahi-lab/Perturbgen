#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=4' # request for exclusive access to gpu
#BSUB -n 16 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team361/ip14/Projects/2024Mar_Tperturb/T_perturb/T_perturb/ # working directory
#BSUB -o ../../logs/%J.out # output file
#BSUB -e ../../logs/%J.err # error file
#BSUB -M 40GB  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>40GB] rusage[mem=40GB]' # RAM memory part 1. Default: 100MB
#BSUB -J 21_G2VMlpSrcCsub # job name

# activate conda environment
source ~/.bashrc
conda activate Tperturb

module load cuda-12.1.1
cd /lustre/scratch126/cellgen/team361/ip14/Projects/2024Mar_Tperturb/T_perturb/T_perturb
cwd=$(pwd)
export WANDB_DIR=$cwd/wandb
export WANDB_ARTIFACT_LOCATION=$cwd/wandb/artifacts
export WANDB_ARTIFACT_DIR=$cwd/wandb/artifacts
export WANDB_CACHE_DIR=$cwd/wandb/artifacts
export WANDB_CONFIG_DIR=$cwd/wandb
export WANDB_DATA_DIR=$cwd/wandb

# run script
echo "--- Start computing model"
# # Run python script for rna
python3 $cwd/train.py \
--train_mode masking \
--split True \
--splitting_mode gears-simulation \
--src_dataset_folder ../../datasets/Norman2019/dataset/subsetted_filtered_tokenised_hvg_pairing_gene2vecpert_control.dataset \
--tgt_dataset_folder ../../datasets/Norman2019/dataset/subsetted_filtered_tokenised_hvg_pairing_perturbed.dataset \
--src_adata_folder ../../datasets/Norman2019/adata/subsetted_filtered_tokenised_hvg_pairing_control.h5ad \
--tgt_adata_folder ../../datasets/Norman2019/adata/subsetted_filtered_tokenised_hvg_pairing_perturbed.h5ad \
--batch_size 40 \
--epochs 100 \
--max_len 1750 \
--petra_lr 0.001 \
--petra_wd 0.0001 \
--num_layers 5 \
--d_ff 16 \
--mlm_probability 0.5 \
--n_workers 16 \
--seed 1 \
--perturbation_encoding_mode mlp compress_src \
--base_path /lustre/scratch126/cellgen/team361/ip14 \
--tune_geneformer False
echo "--- Finished computing model"
