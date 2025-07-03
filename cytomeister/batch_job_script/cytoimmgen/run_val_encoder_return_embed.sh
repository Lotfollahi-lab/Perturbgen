#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=exclusive_process:num=1" # request for exclusive access to gpu
#BSUB -n 32 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/cytomeister # working directory
#BSUB -o logs/return_embed_%J.out # output file
#BSUB -e logs/return_embed_%J.err # error file
#BSUB -M 150000  # RAM memory part 2. Default: 100MB
#BSUB -R "select[mem>150000] rusage[mem=150000]" # RAM memory part 1. Default: 100MB
#BSUB -J cytoimmgen_return_embed # job name

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
python3 /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/cytomeister/val.py \
--test_mode masking \
--split False \
--splitting_mode stratified \
--return_embed True \
--generate False \
--ckpt_masking_path "./T_perturb/cytomeister/Model/checkpoints/20240809_1245_wo_residual_masking_lr_0.0001_wd_0.0001_batch_64_mlmp_0.15_tp_1-2_s_100-epoch=19.ckpt" \
--output_dir "./T_perturb/cytomeister/plt/res/cytoimmgen" \
--src_dataset "./T_perturb/tokenized_data/cytoimmgen/dataset_hvg_src_transformer/0h.dataset" \
--tgt_dataset_folder "./T_perturb/tokenized_data/cytoimmgen/dataset_hvg_tgt" \
--src_adata "./T_perturb/tokenized_data/cytoimmgen/h5ad_pairing_hvg_src/0h.h5ad" \
--tgt_adata_folder "./T_perturb/tokenized_data/cytoimmgen/h5ad_pairing_hvg_tgt" \
--batch_size 64 \
--max_len 291 \
--tgt_vocab_size 1261 \
--cellgen_lr 0.001 \
--cellgen_wd 0.001 \
--count_lr 0.00005 \
--count_wd 0.01 \
--d_ff 128 \
--num_layers 6 \
--n_workers 32 \
--condition_keys Cell_culture_batch \
--time_steps 1 2 \
--mode Transformer_encoder \
--var_list Cell_population Cell_type Time_point Donor \
--seed 100
echo "--- Finished computing model"
