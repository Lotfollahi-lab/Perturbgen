#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=exclusive_process:num=1" # request for exclusive access to gpu :gmodel=NVIDIAA100_SXM4_80GB :gmodel=NVIDIA_H100_HBM3_80GB
#BSUB -n 8 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/return_embed_%J.out # output file
#BSUB -e logs/return_embed_%J.err # error file
#BSUB -M 400000  # RAM memory part 2. Default: 100MB
#BSUB -R "select[mem>400000] rusage[mem=400000]" # RAM memory part 1. Default: 100MB
#BSUB -J return_embed # job name

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
RES_NAME="hspc/embeddings/"
# # if directory does not exist, create it with the name $RES_NAME
# mkdir -p $RES_DIR/$RES_NAME
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
--ckpt_masking_path "./T_perturb/T_perturb/plt/res/hspc/embeddings/res/checkpoints/20241203_1059_cellgen_train_masking_lr_0.0001_wd_1e-05_batch_64_ptime_pos_learnt_m_cosine_tp_1-2_s_42-epoch=29.ckpt" \
--output_dir $RES_DIR/$RES_NAME/res \
--src_dataset "./T_perturb/T_perturb/pp/res/hspc/dataset_10000_hvg_src/stem.dataset" \
--tgt_dataset_folder "./T_perturb/T_perturb/pp/res/hspc/dataset_10000_hvg_tgt" \
--src_adata "./T_perturb/T_perturb/pp/res/hspc/h5ad_pairing_10000_hvg_src/stem.h5ad" \
--tgt_adata_folder "./T_perturb/T_perturb/pp/res/hspc/h5ad_pairing_10000_hvg_tgt" \
--mapping_dict_path  "./T_perturb/T_perturb/pp/res/hspc/token_id_to_genename_10000_hvg.pkl" \
--marker_genes_path "./T_perturb/T_perturb/plt/res/hspc/embeddings/res/cos_sim/20241204_top_250genes_wilcoxon_intermediate_10k.pkl" \
--batch_size 64 \
--max_len 2200 \
--tgt_vocab_size 5805 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.00001 \
--count_lr 0.00005 \
--count_wd 0.01 \
--d_ff 32 \
--num_layers 6 \
--n_workers 8 \
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
