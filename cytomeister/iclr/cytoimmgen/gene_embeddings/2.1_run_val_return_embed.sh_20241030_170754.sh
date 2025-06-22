#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=exclusive_process:num=1" # request for exclusive access to gpu
#BSUB -n 8 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister # working directory
#BSUB -o logs/return_embed_%J.out # output file
#BSUB -e logs/return_embed_%J.err # error file
#BSUB -M 150000  # RAM memory part 2. Default: 100MB
#BSUB -R "select[mem>150000] rusage[mem=150000]" # RAM memory part 1. Default: 100MB
#BSUB -J return_embed # job name

# load cuda
module load cuda-12.1.1

# activate pyenv
source /nfs/team361/cytomeister/.cytomeister/bin/activate
cwd=$(pwd)

# export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister/iclr"
RES_NAME="cytoimmgen/gene_embeddings/"
# if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# copy the current script to the result directory
cp $0 $RES_DIR/$RES_NAME/2.1_run_val_return_embed.sh_$TIMESTAMP.sh
echo "Copying script to $RES_DIR/$RES_NAME/2.1_run_val_return_embed.sh_$TIMESTAMP.sh"

# # Run python script for rna
# python3 $cwd/val.py \
python3 /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister/val.py \
--test_mode masking \
--split False \
--splitting_mode stratified \
--return_embed True \
--generate False \
--ckpt_masking_path "./T_perturb/T_perturb/iclr/cytoimmgen/embedding_analysis/res/checkpoints/20240928_1101_cellgen_train_masking_lr_1e-05_wd_1e-05_batch_64_psin_learnt_m_cosine_tp_1_s_42-epoch=19.ckpt" \
--output_dir $RES_DIR/$RES_NAME/res \
--src_dataset "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_subsetted_src/0h.dataset" \
--tgt_dataset_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/dataset_hvg_subsetted_tgt" \
--src_adata "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_src/0h.h5ad" \
--tgt_adata_folder "./T_perturb/T_perturb/pp/res/cytoimmgen/h5ad_pairing_hvg_tgt" \
--batch_size 256 \
--max_len 300 \
--tgt_vocab_size 1254 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--count_lr 0.00005 \
--count_wd 0.01 \
--d_ff 64 \
--num_layers 6 \
--n_workers 8 \
--pred_tps 1 2 \
--var_list sex phase tissue celltype_v2 diff_state \
--mode GF_frozen \
--context_mode True \
--mask_scheduler 'cosine' \
--positional_encoding 'sin_learnt'
echo "--- Finished computing model"
