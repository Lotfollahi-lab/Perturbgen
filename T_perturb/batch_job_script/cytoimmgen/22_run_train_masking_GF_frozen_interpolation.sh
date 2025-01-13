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
# module load cuda-12.1.1

# activate pyenv
# source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.cellgen_4096/bin/activate
# cwd=$(pwd)
source /lustre/scratch126/cellgen/team361/av13/scmaskgit/.venv/bin/activate

# export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# # ----------------- Create folder to save results and copy the script -----------------
# RES_DIR="/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/nature"
# RES_NAME="LPS/opt_hparam_10k_final_ws"

# if directory does not exist, create it with the name $RES_NAME
# mkdir -p $RES_DIR/$RES_NAME
# Get the current timestamp
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# copy the current script to the result directory
# cp $0 $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_interpolation_$TIMESTAMP.sh
# echo "Copying script to $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_interpolation_$TIMESTAMP.sh"


cd /lustre/scratch126/cellgen/team361/av13/T_perturb/T_perturb/
# export PYTHONPATH="/lustre/scratch126/cellgen/team361/av13:${PYTHONPATH}"
# ----------------- Interpolation -----------------
python3 train.py \
--train_mode masking \
--split False \
--splitting_mode stratified \
--output_dir ./output/ \
--src_dataset "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/pp/res/lps_new_median/dataset_all_subsetted_src/normal.dataset" \
--tgt_dataset_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/pp/res/lps_new_median/dataset_all_subsetted_tgt" \
--src_adata "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/pp/res/lps_new_median/h5ad_pairing_all_src/normal.h5ad" \
--tgt_adata_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_repo/T_perturb/T_perturb/pp/res/lps_new_median/h5ad_pairing_all_tgt" \
--mapping_dict_path "/lustre/scratch126/cellgen/team298/dv8/trace_paper/T_perturb/token_dictionary_gc95M.pkl" \
--encoder_path "/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit/output1/checkpoints/20250107_1024_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=08.ckpt" \
--batch_size 8 \
--max_len 4096 \
--epochs 18 \
--tgt_vocab_size 26717 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.00001 \
--mlm_prob 0.3 \
--n_workers 32 \
--d_ff 32 \
--num_layers 6 \
--time_steps 1 3 \
--var_list cell_type_cellgen_harm donor_cellgen_harm time_after_LPS \
--mode Transformer_encoder \
--positional_encoding 'sin_learnt' \
--context_mode True \
--seed 42 \
--mask_scheduler 'cosine'
echo "--- Finished computing model"
