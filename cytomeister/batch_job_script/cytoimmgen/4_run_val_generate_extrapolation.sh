#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=1' # request for exclusive access to gpu
#BSUB -n 4 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister # working directory
#BSUB -o logs/cyto_generate_extra_s100_%J.out # output file
#BSUB -e logs/cyto_generate_extra_s100_%J.err # error file
#BSUB -M 25000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>25000] rusage[mem=25000]' # RAM memory part 1. Default: 100MB
#BSUB -J cyto_generate_extra_s100 # job name

# load cuda
module load cuda-12.1.1

# activate conda environment
source /nfs/team361/cytomeister/.cytomeister/bin/activate
RES_DIR="/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/res/"
RES_NAME="cytoimmgen/pbmc_median/extrapolation"
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# Run python script to train count decoder
echo "--- Start computing model"

# if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# # Get the current timestamp
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# # copy the current script to the result directory
# cp $0 $RES_DIR/$RES_NAME/3_run_train_count_GF_frozen_extrapolation_$TIMESTAMP.sh
# echo "Copying script to $RES_DIR/$RES_NAME/3_run_train_count_GF_frozen_extrapolation_$TIMESTAMP.sh"

# ----------------- Extrapolation -----------------
# python3 $cwd/val.py \
python3 $cwd/val.py \
--test_mode count \
--split False \
--splitting_mode stratified \
--generate True \
--ckpt_count_path 'T_perturb/cytomeister/plt/res/cytoimmgen/pbmc_median/extrapolation/res/checkpoints/20250514_1753_cellgen_train_count_lr_0.001_wd_0.001_batch_64_drop_0.05_zinb_tp_1-2_s_100_pos_time_pos_sin_m_pow-epoch=01.ckpt' \
--output_dir $RES_DIR/$RES_NAME \
--src_dataset "T_perturb/tokenized_data/cytoimmgen_pbmc_median/dataset_2000_hvg_src/0h.dataset" \
--tgt_dataset_folder "T_perturb/tokenized_data/cytoimmgen_pbmc_median/dataset_2000_hvg_tgt" \
--src_adata "T_perturb/tokenized_data/cytoimmgen_pbmc_median/h5ad_pairing_2000_hvg_src/0h.h5ad" \
--tgt_adata_folder "T_perturb/tokenized_data/cytoimmgen_pbmc_median/h5ad_pairing_2000_hvg_tgt" \
--mapping_dict_path  "T_perturb/tokenized_data/cytoimmgen_pbmc_median/token_id_to_genename_2000_hvg.pkl" \
--batch_size 128 \
--max_len 400 \
--tgt_vocab_size 1360 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--count_lr 0.005 \
--count_wd 0.001 \
--d_ff 64 \
--num_layers 6 \
--loss_mode zinb \
--n_workers 4 \
--condition_keys Cell_culture_batch \
--pred_tps 3 \
--context_tps 1 2 \
--var_list Cell_population Cell_type Time_point Donor \
--cond_list Time_point \
--encoder scmaskgit \
--encoder_path "/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/output2/checkpoints/20250620_1508_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=07.ckpt" \
--pos_encoding_mode time_pos_sin \
--d_model 768 \
--temperature 0.75 \
--sequence_length 125 \
--iterations 20 \
--n_samples 2 \
--mask_scheduler 'pow' \
--seed 100 \
--context_mode True
echo '--- Finished computing model'
