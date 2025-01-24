#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=exclusive_process:num=1" # request for exclusive access to gpu :gmodel=NVIDIAA100_SXM4_80GB :gmodel=NVIDIA_H100_HBM3_80GB
#BSUB -n 8 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/return_embed_%J.out # output file
#BSUB -e logs/return_embed_%J.err # error file
#BSUB -M 200000  # RAM memory part 2. Default: 100MB
#BSUB -R "select[mem>200000] rusage[mem=200000]" # RAM memory part 1. Default: 100MB
#BSUB -J return_embed_scmaskgit # job name

# load cuda
module load cuda-12.1.1

# activate pyenv
source /lustre/scratch126/cellgen/team361/kl11/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)

# export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/plt/res"
RES_NAME="hspc/pbmc_median/"
# # if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# # Get the current timestamp
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# # copy the current script to the result directory
# cp $0 $RES_DIR/$RES_NAME/2.1_run_val_return_embed.sh_$TIMESTAMP.sh
# echo "Copying script to $RES_DIR/$RES_NAME/2.1_run_val_return_embed.sh_$TIMESTAMP.sh"

# # Run python script for rna
# python3 $cwd/val.py \
python3 /lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/val.py \
--test_mode masking \
--split False \
--splitting_mode stratified \
--return_embed True \
--return_attn False \
--generate False \
--ckpt_masking_path "T_perturb/T_perturb/plt/res/hspc/pbmc_median/checkpoints/20250123_1633_cellgen_train_masking_lr_1e-05_wd_1e-05_batch_64_ptime_pos_sin_m_cosine_tp_1-2_s_42-epoch=19.ckpt" \
--output_dir $RES_DIR/$RES_NAME/embeddings \
--src_dataset "T_perturb/T_perturb/pp/res/hspc_pbmc_median/dataset_10000_hvg_src/stem.dataset" \
--tgt_dataset_folder "T_perturb/T_perturb/pp/res/hspc_pbmc_median/dataset_10000_hvg_tgt" \
--src_adata "T_perturb/T_perturb/pp/res/hspc_pbmc_median/h5ad_pairing_10000_hvg_src/stem.h5ad" \
--tgt_adata_folder "T_perturb/T_perturb/pp/res/hspc_pbmc_median/h5ad_pairing_10000_hvg_tgt" \
--mapping_dict_path  "T_perturb/T_perturb/pp/res/hspc_pbmc_median/token_id_to_genename_10000_hvg.pkl" \
--batch_size 64 \
--max_len 2200 \
--tgt_vocab_size 5710 \
--cellgen_lr 0.00001 \
--cellgen_wd 0.00001 \
--count_lr 0.00005 \
--count_wd 0.01 \
--d_ff 64 \
--num_layers 6 \
--n_workers 8 \
--pred_tps 1 2 \
--var_list sex phase tissue celltype_v2 diff_state \
--cond_list celltype_v2 diff_state \
--encoder scmaskgit \
--encoder_path "/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit/output3/checkpoints/20250113_1104_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=06.ckpt" \
--tokenid_to_rowid 'T_perturb/T_perturb/pp/res/hspc_pbmc_median/tokenid_to_rowid_10000_hvg.pkl' \
--context_mode True \
--mask_scheduler 'cosine' \
--pos_encoding_mode 'time_pos_sin' \
--d_model 768
echo "--- Finished computing model"

# PBMC median
# --encoder_path '/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit/output3/checkpoints/20250113_1104_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=06.ckpt' \

# 20k GF median
# --encoder_path '/lustre/scratch126/cellgen/team361/av13/scmaskgit/scmaskgit/output2/checkpoints/20250110_2325_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=06.ckpt' \

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
