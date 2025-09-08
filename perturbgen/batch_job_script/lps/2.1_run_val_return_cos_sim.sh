#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=exclusive_process:num=1" # request for exclusive access to gpu :gmodel=NVIDIAA100_SXM4_80GB :gmodel=NVIDIA_H100_HBM3_80GB
#BSUB -n 8 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb # working directory
#BSUB -o T_perturb/log/return_embed_%J.out # output file
#BSUB -e T_perturb/log/return_embed_%J.err # error file
#BSUB -M 80000  # RAM memory part 2. Default: 100MB
#BSUB -R "select[mem>80000] rusage[mem=80000]" # RAM memory part 1. Default: 100MB
#BSUB -J return_embed_scmaskgit # job name

# load cuda
module load cuda-12.1.1

# activate pyenv
cwd=$(pwd)

# export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/T_perturb/plt/res"
RES_NAME="lps/embedding_100mMedE0e7_int_2k_all_tps_lps"
## if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# # Get the current timestamp
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# # copy the current script to the result directory
# cp $0 $RES_DIR/$RES_NAME/2.1_run_val_return_embed.sh_$TIMESTAMP.sh
# echo "Copying script to $RES_DIR/$RES_NAME/2.1_run_val_return_embed.sh_$TIMESTAMP.sh"

# # Run python script for rna
# python3 $cwd/val.py \

python3 /lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/T_perturb/val.py \
--test_mode masking \
--split False \
--splitting_mode stratified \
--return_embed True \
--return_attn False \
--generate False \
--ckpt_masking_path "/lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/T_perturb/plt/res/lps/Our_median/interpolation_LPS_e0100M_CondTime_cosine_withControl_2k/res/checkpoints/20250714_2311_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_64_ptime_pos_sin_m_cosine_tp_1-2-3_s_0-epoch=07.ckpt" \
--output_dir $RES_DIR/$RES_NAME/embeddings \
--src_dataset "/lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/T_perturb/pp/res/OURmed100m_LPS_all_tps_2k_hvg/dataset_2000_hvg_src/normal.dataset" \
--tgt_dataset_folder "/lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/T_perturb/pp/res/OURmed100m_LPS_all_tps_2k_hvg/dataset_2000_hvg_tgt" \
--src_adata "/lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/T_perturb/pp/res/OURmed100m_LPS_all_tps_2k_hvg/h5ad_pairing_2000_hvg_src/normal.h5ad" \
--tgt_adata_folder "/lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/T_perturb/pp/res/OURmed100m_LPS_all_tps_2k_hvg/h5ad_pairing_2000_hvg_tgt" \
--mapping_dict_path "/lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/T_perturb/pp/res/OURmed100m_LPS_all_tps_2k_hvg/token_id_to_genename_2000_hvg.pkl" \
--batch_size 64 \
--max_len 797 \
--tgt_vocab_size 2006 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--count_lr 0.00005 \
--count_wd 0.01 \
--d_ff 32 \
--num_layers 6 \
--n_workers 32 \
--pred_tps 1 2 3 \
--var_list cell_pairing_index time_after_LPS cell_type_harmonized \
--tokenid_to_rowid '/lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/T_perturb/pp/res/OURmed100m_LPS_all_tps_2k_hvg/tokenid_to_rowid_2000_hvg.pkl' \
--encoder scmaskgit \
--encoder_path "/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/foundation_107m/checkpoints/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt" \
--context_mode True \
--mask_scheduler 'cosine' \
--return_gene_embs True \
--gene_embs_condition 'time_after_LPS' \
--pos_encoding_mode 'time_pos_sin' \
--d_model 768 
echo "--- Finished computing model"
