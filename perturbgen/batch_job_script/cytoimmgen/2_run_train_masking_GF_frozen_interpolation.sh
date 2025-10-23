#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=4' # request for exclusive access to gpu
#BSUB -n 4 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/perturbgen # working directory
#BSUB -o logs/cytoimmgen_masking_%J.out # output file
#BSUB -e logs/cytoimmgen_masking_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J cytoimmgen_masking # job name

# load cuda
module load cuda-12.1.1

# activate pyenv
source /nfs/team361/cytomeister/.cytomeister/bin/activate
cwd=$(pwd)

# export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# # ----------------- Create folder to save results and copy the script -----------------
RES_DIR="/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/res/"
RES_NAME="cytoimmgen/interpolation"
# if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# Get the current timestamp
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# copy the current script to the result directory
# cp $0 $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_interpolation_$TIMESTAMP.sh
# echo "Copying script to $RES_DIR/$RES_NAME/2_run_train_masking_GF_frozen_interpolation_$TIMESTAMP.sh"

# ----------------- Interpolation -----------------
python3 $cwd/train.py \
--train_mode masking \
--split False \
--splitting_mode stratified \
--output_dir $RES_DIR/$RES_NAME \
--src_dataset "T_perturb/tokenized_data/cytoimmgen_100M_cellpopulation/dataset_2000_hvg_src/0h.dataset" \
--tgt_dataset_folder "T_perturb/tokenized_data/cytoimmgen_100M_cellpopulation/dataset_2000_hvg_tgt" \
--src_adata "T_perturb/tokenized_data/cytoimmgen_100M_cellpopulation/h5ad_pairing_2000_hvg_src/0h.h5ad" \
--tgt_adata_folder "T_perturb/tokenized_data/cytoimmgen_100M_cellpopulation/h5ad_pairing_2000_hvg_tgt" \
--mapping_dict_path  "T_perturb/tokenized_data/cytoimmgen_100M_cellpopulation/token_id_to_genename_2000_hvg.pkl" \
--batch_size 64 \
--epochs 20 \
--cellgen_lr 0.00001 \
--cellgen_wd 0.00001 \
--n_workers 4 \
--d_ff 64 \
--num_layers 6 \
--condition_keys Cell_culture_batch \
--pred_tps 1 3 \
--var_list Cell_population Cell_type Time_point Donor \
--encoder scmaskgit \
--context_mode True \
--encoder_path "/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/foundation_107m/checkpoints/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt" \
--pos_encoding_mode time_pos_sin \
--seed 42 \
--mask_scheduler 'pow' \
--d_model 768 \
--use_weighted_sampler True \
--sampling_keys Cell_type \
--ckpt_every_n_epochs 5
echo "--- Finished computing model"
