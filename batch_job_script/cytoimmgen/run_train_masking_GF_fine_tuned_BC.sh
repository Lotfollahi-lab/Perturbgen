#!/bin/bash
# BSUB -q gpu-huge # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
# BSUB -gpu 'mode=exclusive_process:num=4:block=yes' # request for exclusive access to gpu
# BSUB -n 32 # number of cores
# BSUB -G team361 # groupname for billing
# BSUB -cwd /lustre/scratch126/cellgen/team205/bair/CellGen_HK/T_perturb/T_perturb # working directory
# BSUB -o logs/bc_random_masking_%J.out # output file
# BSUB -e logs/bc_random_masking_%J.err # error file
# BSUB -M 50000  # RAM memory part 2. Default: 100MB
# BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
# BSUB -J random_cytoimmgen_masking # job name
# BSUB -m "farm22-gpu0303 farm22-gpu0304 farm22-gpu0305" # select specific nodes

# load cuda
module load cuda-12.1.1

# activate pyenv
source /lustre/scratch126/cellgen/team361/hk11/cytoimmgen/.venv/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb

export WANDB_API_KEY=ae236eefd11e9090974d0d303f9271c7a8e117dd
wandb login

# run script
echo "--- Start computing model"

# # Interpolation
# python3 $cwd/train.py \
# --train_mode masking \
# --split False \
# --splitting_mode stratified \
# --output_dir "./T_perturb/T_perturb/plt/res/cytoimmgen" \
# --src_dataset "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_src/0h.dataset" \
# --tgt_dataset_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_tgt" \
# --src_adata "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_src/0h.h5ad" \
# --tgt_adata_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_tgt" \
# --mapping_dict_path  "./T_perturb/T_perturb/pp/res/cytoimmgen/token_id_to_genename_hvg.pkl" \
# --batch_size 64 \
# --max_len 300 \
# --epochs 50 \
# --tgt_vocab_size 1261 \
# --cellgen_lr 0.0001 \
# --cellgen_wd 0.0001 \
# --mlm_prob 0.15 \
# --n_workers 32 \
# --d_ff 128 \
# --num_layers 6 \
# --condition_keys Cell_culture_batch \
# --time_steps 1 3 \
# --var_list Cell_population Cell_type Time_point Donor \
# --mode GF_fine_tuned
# echo "--- Finished computing model"

# Extrapolation
python3 $cwd/train_bc.py \
--train_mode masking \
--split False \
--splitting_mode stratified \
--output_dir "./T_perturb/T_perturb/plt/res/cytoimmgen" \
--src_dataset "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_src/0h.dataset" \
--tgt_dataset_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_tgt" \
--src_adata "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_src/0h.h5ad" \
--tgt_adata_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_tgt" \
--mapping_dict_path  "./T_perturb/T_perturb/pp/res/cytoimmgen/token_id_to_genename_hvg.pkl" \
--batch_size 64 \
--max_len 300 \
--epochs 20 \
--tgt_vocab_size 1261 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--mlm_prob 0.15 \
--n_workers 32 \
--d_ff 128 \
--num_layers 6 \
--condition_keys Cell_culture_batch \
--time_steps 1 \
--var_list Cell_population Cell_type Time_point Donor \
--mode GF_fine_tuned \
--context_mode True
echo "--- Finished computing model"
