#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=exclusive_process:num=1" # request for exclusive access to gpu :gmodel=NVIDIAA100_SXM4_80GB :gmodel=NVIDIA_H100_HBM3_80GB
#BSUB -n 4 # number of cores
#BSUB -G cellulargenetics-priority # groupname for billing team361
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11 # working directory
#BSUB -o T_perturb/cytomeister/logs/return_gene_embed_%J.out # output file
#BSUB -e T_perturb/cytomeister/logs/return_gene_embed_%J.err # error file
#BSUB -M 250000  # RAM memory part 2. Default: 100MB
#BSUB -R "select[mem>250000] rusage[mem=250000]" # RAM memory part 1. Default: 100MB
#BSUB -J return_gene_embed # job name

# load cuda
module load cuda-12.1.1

# activate pyenv
source /nfs/team361/cytomeister/.cytomeister/bin/activate
cwd=$(pwd)

# export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/res/"
RES_NAME="hspc/embeddings"
# # if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# # Get the current timestamp
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# # copy the current script to the result directory
# cp $0 $RES_DIR/$RES_NAME/2.1_run_val_return_embed.sh_$TIMESTAMP.sh
# echo "Copying script to $RES_DIR/$RES_NAME/2.1_run_val_return_embed.sh_$TIMESTAMP.sh"

# # Run python script for rna
# python3 $cwd/val.py \
python3 /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister/val.py \
--test_mode masking \
--split False \
--splitting_mode stratified \
--return_embed True \
--return_attn False \
--generate False \
--ckpt_masking_path "T_perturb/res/hspc/fine_tuning/checkpoints/20250623_2256_cellgen_train_masking_lr_1e-05_wd_1e-05_batch_64_ptime_pos_sin_m_pow_tp_1-2_s_42-epoch=19.ckpt" \
--output_dir $RES_DIR/$RES_NAME/embeddings \
--src_dataset "T_perturb/tokenized_data/hspc_pbmc_median_all_tissue_all_tf/dataset_5000_hvg_src/stem.dataset" \
--tgt_dataset_folder "T_perturb/tokenized_data/hspc_pbmc_median_all_tissue_all_tf/dataset_5000_hvg_tgt" \
--src_adata "T_perturb/tokenized_data/hspc_pbmc_median_all_tissue_all_tf/h5ad_pairing_5000_hvg_src/stem.h5ad" \
--tgt_adata_folder "T_perturb/tokenized_data/hspc_pbmc_median_all_tissue_all_tf/h5ad_pairing_5000_hvg_tgt" \
--mapping_dict_path "T_perturb/tokenized_data/hspc_pbmc_median_all_tissue_all_tf/token_id_to_genename_5000_hvg.pkl" \
--batch_size 32 \
--max_len 2275 \
--tgt_vocab_size 5685 \
--cellgen_lr 0.00001 \
--cellgen_wd 0.00001 \
--count_lr 0.00005 \
--count_wd 0.01 \
--d_ff 64 \
--num_layers 6 \
--n_workers 4 \
--pred_tps 1 2 \
--var_list sex phase tissue celltype_v2 diff_state \
--encoder scmaskgit \
--encoder_path "/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/output2/checkpoints/20250620_1508_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=07.ckpt" \
--tokenid_to_rowid 'T_perturb/tokenized_data/hspc_pbmc_median_all_tissue_all_tf/token_id_to_genename_5000_hvg.pkl' \
--context_mode True \
--mask_scheduler 'pow' \
--pos_encoding_mode 'time_pos_sin' \
--d_model 768 \
--return_gene_embs True \
--gene_embs_condition 'diff_state'
echo "--- Finished computing model"

# --deg_pkl_path 'T_perturb/res/hspc/figures/20250126_top250_DEG_lmpptissue_v_lmpprest_10k.pkl'

# PBMC median
# --encoder_path '/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/output2/checkpoints/20250620_1508_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=07.ckpt' \

# 20k GF median
# --encoder_path '/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/scmaskgit/output2/checkpoints/20250110_2325_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=06.ckpt' \

# 2k hvgs
# --src_dataset "./T_perturb/tokenized_data/hspc/dataset_hvg_src/stem.dataset" \
# --tgt_dataset_folder "./T_perturb/tokenized_data/hspc/dataset_hvg_tgt" \
# --src_adata "./T_perturb/tokenized_data/hspc/h5ad_pairing_hvg_src/stem.h5ad" \
# --tgt_adata_folder "./T_perturb/tokenized_data/hspc/h5ad_pairing_hvg_tgt" \
# --mapping_dict_path  "./T_perturb/tokenized_data/hspc/token_id_to_genename_hvg.pkl" \

# --max_len 450 \
# --tgt_vocab_size 1187 \

# 5k hvgs
# --max_len 1040 \
# --tgt_vocab_size 3015 \
