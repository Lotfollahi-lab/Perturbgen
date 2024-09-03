#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=exclusive_process:num=1" # request for exclusive access to gpu
#BSUB -n 4 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team361/hk11/cytoimmgen/T_perturb/T_perturb # working directory
#BSUB -o logs/return_embed_%J.out # output file
#BSUB -e logs/return_embed_%J.err # error file
#BSUB -M 150000  # RAM memory part 2. Default: 100MB
#BSUB -R "select[mem>150000] rusage[mem=150000]" # RAM memory part 1. Default: 100MB
#BSUB -J cytoimmgen_return_embed # job name

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

# # Run python script for rna
# python3 $cwd/val.py \
python3 /lustre/scratch126/cellgen/team361/hk11/cytoimmgen/T_perturb/T_perturb/val.py \
--test_mode masking \
--split False \
--splitting_mode stratified \
--return_embeddings True \
--generate False \
--ckpt_masking_path "./T_perturb/T_perturb/Model/checkpoints/20240828_1720_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_64_mlmp_0.15_tp_1_s_42-epoch=49.ckpt" \
--output_dir "./T_perturb/T_perturb/plt/res/cytoimmgen" \
--src_dataset "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_src/0h.dataset" \
--tgt_dataset_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_tgt" \
--src_adata "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_src/0h.h5ad" \
--tgt_adata_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_tgt" \
--batch_size 64 \
--max_len 300 \
--tgt_vocab_size 1261 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--count_lr 0.00005 \
--count_wd 0.01 \
--d_ff 128 \
--num_layers 6 \
--n_workers 16 \
--condition_keys Cell_culture_batch \
--time_steps 1 \
--var_list Cell_population Cell_type Time_point Donor \
--mode GF_fine_tuned \
--context_mode False \
--perturbation_mode KO \
--perturbation_genes first_gene
echo "--- Finished computing model"
# ENO1 TNFRSF4 FABP5 IL2RA TUBA1B HSP90AA1 STAT1 BATF PIM3 NFKBIA (tt)
# IL3 IL9 EGR2 ZBED2 EGR3 CXCL10 IL21 IL2 BATF3 TNFRSF9 (top10)

# # # Run python script for rna
# # python3 $cwd/val.py \
# python3 /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/val.py \
# --test_mode masking \
# --split True \
# --splitting_mode stratified \
# --return_embed True \
# --generate False \
# --ckpt_count_path "./T_perturb/T_perturb/Model/checkpoints/20240517_1225_petra_train_masking_lr_0.0001_wd_0.0001_batch_64_mlmp_0.15_tp_1-2-3.ckpt" \
# --output_dir "./T_perturb/T_perturb/plt/res/cytoimmgen" \
# --src_dataset "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_src/0h.dataset" \
# --tgt_dataset_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_tgt" \
# --src_adata "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_src/0h.h5ad" \
# --tgt_adata_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_tgt" \
# --batch_size 64 \
# --max_len 300 \
# --tgt_vocab_size 1261 \
# --cellgen_lr 0.001 \
# --cellgen_wd 0.0001 \
# --count_lr 0.00005 \
# --count_wd 0.01 \
# --d_ff 32 \
# --num_layers 1 \
# --n_workers 32 \
# --condition_keys Cell_culture_batch \
# --time_steps 1 2 3 \
# --var_list Cell_population Cell_type Time_point Donor
# echo "--- Finished computing model"
