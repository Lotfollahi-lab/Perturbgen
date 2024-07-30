#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=exclusive_process:num=1" # request for exclusive access to gpu
#BSUB -n 32 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/covid_embed_%J.out # output file
#BSUB -e logs/covid_embed_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'span[ptile=32]'  # Allocate 4 CPU cores per node
#BSUB -R "select[mem>50000] rusage[mem=50000]" # RAM memory part 1. Default: 100MB
#BSUB -J covid_embed # job name

# load cuda
module load cuda-12.1.1

# activate pyenv
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.petra_cuda12/bin/activate
cwd=$(pwd)

# export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# # Run python script for rna
# python3 $cwd/val.py \
python3 /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/val.py \
--test_mode masking \
--split False \
--splitting_mode stratified \
--return_embed True \
--generate True \
--ckpt_masking_path "./T_perturb/T_perturb/Model/checkpoints/20240730_1353_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_64_mlmp_0.15_ntask_1_s_42-epoch=49.ckpt" \
--output_dir "./T_perturb/T_perturb/plt/res/covid_19" \
--src_dataset "./CellGen-reproducibility/covid/processed_data/dataset_hvg_src_transformer/normal.dataset" \
--tgt_dataset_folder "./CellGen-reproducibility/covid/processed_data/dataset_hvg_tgt/" \
--src_adata "./CellGen-reproducibility/covid/processed_data/h5ad_pairing_hvg_src/normal.h5ad" \
--tgt_adata_folder "./CellGen-reproducibility/covid/processed_data/h5ad_pairing_hvg_tgt" \
--cell_pairing_dir "./CellGen-reproducibility/covid/processed_data/cell_pairing" \
--batch_size 32 \
--max_len 900 \
--tgt_vocab_size 25428 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--count_lr 0.00005 \
--count_wd 0.01 \
--d_ff 128 \
--num_layers 6 \
--n_workers 32 \
--n_task_conditions 1 \
--var_list sex disease donor_id development_stage cell_type \
--encoder_type GF_frozen
echo "--- Finished computing model"
