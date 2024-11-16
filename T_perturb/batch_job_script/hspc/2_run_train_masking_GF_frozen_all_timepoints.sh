#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=2:block=yes' # request for exclusive access to gpu :gmodel=NVIDIAA100_SXM4_80GB
#BSUB -n 16 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/hspc_masking_%J.out # output file
#BSUB -e logs/hspc_masking_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J hspc_masking # job name

# load cuda
module load cuda-12.1.1

# activate pyenv
source /lustre/scratch126/cellgen/team361/kl11/t_generative/.cellgen_4096/bin/activate
#activate wandb for logging
# wandb online
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# # # ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/plt/res"
RES_NAME="hspc/embeddings"
# if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# # Get the current timestamp
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# # copy the current script to the result directory
# cp $0 $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_all_timepoints_$TIMESTAMP.sh
# echo "Copying script to $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_all_timepoints_$TIMESTAMP.sh"

# ----------------- all_timepoints -----------------
# python3 $cwd/train.py \
python3 /lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/train.py \
--train_mode masking \
--split True \
--splitting_mode stratified \
--split_obs celltype_v2 \
--output_dir $RES_DIR/$RES_NAME/res \
--src_dataset "./T_perturb/T_perturb/pp/res/hspc/dataset_10000_hvg_src/stem.dataset" \
--tgt_dataset_folder "./T_perturb/T_perturb/pp/res/hspc/dataset_10000_hvg_tgt" \
--src_adata "./T_perturb/T_perturb/pp/res/hspc/h5ad_pairing_10000_hvg_src/stem.h5ad" \
--tgt_adata_folder "./T_perturb/T_perturb/pp/res/hspc/h5ad_pairing_10000_hvg_tgt" \
--mapping_dict_path  "./T_perturb/T_perturb/pp/res/hspc/token_id_to_genename_10000_hvg.pkl" \
--batch_size 64 \
--max_len 2200 \
--epochs 10 \
--tgt_vocab_size 22044 \
--cellgen_lr 0.00001 \
--cellgen_wd 0.00001 \
--mlm_prob 0.15 \
--n_workers 16 \
--d_ff 64 \
--num_layers 6 \
--pred_tps 1 2 \
--var_list sex phase tissue celltype_v2 diff_state \
--encoder GF_frozen \
--context_mode True \
--mask_scheduler 'cosine' \
--pos_encoding_mode 'time_pos_learnt'
echo "--- Finished computing model"

# 2k hvgs
# --src_dataset "./T_perturb/T_perturb/pp/res/hspc/dataset_hvg_src/stem.dataset" \
# --tgt_dataset_folder "./T_perturb/T_perturb/pp/res/hspc/dataset_hvg_tgt" \
# --src_adata "./T_perturb/T_perturb/pp/res/hspc/h5ad_pairing_hvg_src/stem.h5ad" \
# --tgt_adata_folder "./T_perturb/T_perturb/pp/res/hspc/h5ad_pairing_hvg_tgt" \
# --mapping_dict_path  "./T_perturb/T_perturb/pp/res/hspc/token_id_to_genename_hvg.pkl" \

# --max_len 450 \
# --tgt_vocab_size 1187 \

# 5k hvgs
# --max_len 1040 \
# --tgt_vocab_size 3015 \

# 10k hvgs
# --max_len 2200 \
# --tgt_vocab_size 22044 \
