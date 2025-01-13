#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=4:block=yes' # request for exclusive access to gpu
#BSUB -n 8 # number of cores
#BSUB -R "span[ptile=8]"     # split X cores per host
#BSUB -G team361 # groupname for billing
##BSUB -cwd /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb 
#BSUB -o %J.out # output file
#BSUB -e %J.err # error file
#BSUB -M 140000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>140000] rusage[mem=140000]' # RAM memory part 1. Default: 100MB
#BSUB -J interpolation # job name

# load cuda
module load cuda-12.1.1

# activate conda environment
source /lustre/scratch126/cellgen/team361/av13/scmaskgit/.venv/bin/activate
# cwd=$(pwd)

# export WANDB_DIR=$cwd/wandb
# Run python script to train count decoder
echo "--- Start computing model"

# #interpolation
# python3 $cwd/train.py \
# --train_mode count \
# --split False \
# --splitting_mode stratified \
# --output_dir "./T_perturb/T_perturb/plt/res/cytoimmgen" \
# --ckpt_masking_path "./T_perturb/T_perturb/Model/checkpoints/20240522_0248_tcell_extrapol_lr_0.0001_wd_0.0001_batch_64_mlmp_0.15_tp_1-2_s_100-epoch=149.ckpt" \
# --src_dataset "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_src_transformer/0h.dataset" \
# --tgt_dataset_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_tgt" \
# --src_adata "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_src/0h.h5ad" \
# --tgt_adata_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_tgt" \
# --mapping_dict_path  "./T_perturb/T_perturb/pp/res/cytoimmgen/token_id_to_genename_hvg.pkl" \
# --batch_size 64 \
# --max_len 300 \
# --epochs 20 \
# --tgt_vocab_size 1261 \
# --cellgen_lr 0.0001 \
# --count_lr 0.00005 \
# --cellgen_wd 0.0001 \
# --count_wd 0.01 \
# --mlm_prob 0.15 \
# --n_workers 32 \
# --d_ff 128 \
# --num_layers 6 \
# --loss_mode zinb \
# --condition_keys Cell_culture_batch \
# --time_steps 1 3 \
# --var_list Cell_population Cell_type Time_point Donor \
# --mode Transformer_encoder \
# --seed 100
# echo "--- Finished computing model"
cd /lustre/scratch126/cellgen/team361/av13/T_perturb/T_perturb/

# extrapolation
python3 train.py \
--train_mode count \
--split False \
--splitting_mode stratified \
--output_dir "./output_count" \
--ckpt_masking_path "/lustre/scratch126/cellgen/team361/av13/T_perturb/T_perturb/output/checkpoints/20250111_2058_cellgen_train_masking_lr_5e-05_wd_1e-05_batch_16_psin_learnt_m_cosine_tp_1-3_s_42-epoch=13.ckpt" \
--src_dataset "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/pp/res/lps_10k_ds/dataset_hvg_subsetted_src/normal.dataset" \
--tgt_dataset_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/pp/res/lps_10k_ds/dataset_hvg_subsetted_tgt" \
--src_adata "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/pp/res/lps_10k_ds/h5ad_pairing_hvg_src/normal.h5ad" \
--tgt_adata_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/pp/res/lps_10k_ds/h5ad_pairing_hvg_tgt" \
--mapping_dict_path  "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/pp/res/token_id_to_genename_hvg.pkl" \
--batch_size 16 \
--max_len 3600 \
--epochs 20 \
--tgt_vocab_size 20274 \
--cellgen_lr 0.0001 \
--count_lr 0.00005 \
--cellgen_wd 0.0001 \
--count_wd 0.01 \
--mlm_prob 0.15 \
--n_workers 32 \
--d_ff 128 \
--num_layers 6 \
--loss_mode zinb \
--time_steps 1 2 \
--var_list Cell_population Cell_type Time_point Donor \
--mode Transformer_encoder \
--seed 100
echo "--- Finished computing model"
