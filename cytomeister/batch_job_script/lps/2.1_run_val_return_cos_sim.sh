#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=exclusive_process:num=1" # request for exclusive access to gpu :gmodel=NVIDIAA100_SXM4_80GB :gmodel=NVIDIA_H100_HBM3_80GB
#BSUB -n 8 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb # working directory
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
RES_DIR="/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/cytomeister/results"
RES_NAME="lps/embedding_analysis_int_2k_all_tps_cond_celltype"
# # if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# # Get the current timestamp
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# # copy the current script to the result directory
# cp $0 $RES_DIR/$RES_NAME/2.1_run_val_return_embed.sh_$TIMESTAMP.sh
# echo "Copying script to $RES_DIR/$RES_NAME/2.1_run_val_return_embed.sh_$TIMESTAMP.sh"

# # Run python script for rna
# python3 $cwd/val.py \
python3 /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/cytomeister/val.py \
--test_mode masking \
--split False \
--splitting_mode stratified \
--return_embed True \
--return_attn False \
--generate False \
--ckpt_masking_path "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/cytomeister/results/lps/interpolation_2k_all_tps_cond_celltype/res/checkpoints/20250216_1825_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_64_ptime_pos_sin_m_cosine_tp_1-2-3_s_42-epoch=15.ckpt" \
--output_dir $RES_DIR/$RES_NAME/embeddings \
--src_dataset "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/tokenized_data/2k_hvg_ourMED_all_tps/dataset_2000_hvg_src/normal.dataset" \
--tgt_dataset_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/tokenized_data/2k_hvg_ourMED_all_tps/dataset_2000_hvg_tgt" \
--src_adata "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/tokenized_data/2k_hvg_ourMED_all_tps/h5ad_pairing_2000_hvg_src/normal.h5ad" \
--tgt_adata_folder "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/tokenized_data/2k_hvg_ourMED_all_tps/h5ad_pairing_2000_hvg_tgt" \
--mapping_dict_path "/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/tokenized_data/2k_hvg_ourMED_all_tps/token_id_to_genename_2000_hvg.pkl" \
--batch_size 64 \
--max_len 666 \
--tgt_vocab_size 20274 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--count_lr 0.00005 \
--count_wd 0.01 \
--d_ff 32 \
--num_layers 6 \
--n_workers 32 \
--pred_tps 1 2 3 \
--var_list cell_type_cellgen_harm donor_cellgen_harm time_after_LPS \
--cond_list cell_type_cellgen_harm \
--encoder scmaskgit \
--encoder_path "/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/output2/checkpoints/20250620_1508_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=07.ckpt" \
--tokenid_to_rowid '/lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/tokenized_data/2k_hvg_ourMED_all_tps/tokenid_to_rowid_2000_hvg.pkl' \
--context_mode True \
--mask_scheduler 'cosine' \
--return_gene_embs True \
--gene_embs_condition 'time_after_LPS' \
--pos_encoding_mode 'time_pos_sin' \
--d_model 768
echo "--- Finished computing model"

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

#--cond_list cell_type_cellgen_harm \
#--gene_embs_condition 'cell_type_cellgen_harm' \

