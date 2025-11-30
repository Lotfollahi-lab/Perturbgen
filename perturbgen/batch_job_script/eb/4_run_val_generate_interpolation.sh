#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=1 ' # request for exclusive access to gpu
#BSUB -n 4 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/perturbgen # working directory
#BSUB -o logs/eb_generate_inter_s42_%J.out # output file
#BSUB -e logs/eb_generate_inter_s42_%J.err # error file
#BSUB -M 20000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>20000] rusage[mem=20000]' # RAM memory part 1. Default: 100MB
#BSUB -J eb_generate_inter_s42 # job name

# load cuda
module load cuda-12.1.1

# activate pyenv
source /nfs/team361/cytomeister/.cytomeister/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run script
echo '--- Start computing model'

RES_DIR="/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/res/"
RES_NAME="eb/interpolation"

# # if directory does not exist, create it with the name $RES_NAME
# mkdir -p $RES_DIR/$RES_NAME
# # Get the current timestamp
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# # copy the current script to the result directory
# cp $0 $RES_DIR/$RES_NAME/4_run_val_generate_interpolation_$TIMESTAMP.sh
# echo "Copying script to $RES_DIR/$RES_NAME/4_run_val_generate_interpolation_$TIMESTAMP.sh"

# export WANDB_DIR=$cwd/wandb
# Run python script to train count decoder


# interpolation
python3 $cwd/val.py \
--test_mode count \
--split False \
--splitting_mode random \
--generate True \
--output_dir $RES_DIR/$RES_NAME \
--ckpt_count_path 'T_perturb/res/eb/interpolation/checkpoints/20250828_2252_cellgen_train_count_lr_0.0001_wd_0.0001_batch_64_drop_0.1_zinb_tp_1-2-4_s_100_pos_time_pos_sin_m_pow-epoch=99.ckpt' \
--src_dataset 'T_perturb/tokenized_data/eb_100M/dataset_2000_hvg_src/Day 00-03.dataset' \
--tgt_dataset_folder 'T_perturb/tokenized_data/eb_100M/dataset_2000_hvg_tgt' \
--src_adata 'T_perturb/tokenized_data/eb_100M/h5ad_pairing_2000_hvg_src/Day 00-03.h5ad' \
--tgt_adata_folder 'T_perturb/tokenized_data/eb_100M/h5ad_pairing_2000_hvg_tgt' \
--mapping_dict_path  'T_perturb/tokenized_data/eb_100M/token_id_to_genename_2000_hvg.pkl' \
--batch_size 64 \
--cellgen_lr 0.001 \
--cellgen_wd 0.0001 \
--count_lr 0.0001 \
--count_wd 0.0001 \
--num_layers 3 \
--d_ff 32 \
--loss_mode zinb \
--n_workers 4 \
--pred_tps 3 \
--context_tps 1 2 4 \
--var_list Time_point \
--encoder scmaskgit \
--encoder_path "/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/res/eb/extrapolation/checkpoints/20250828_1430_cellgen_train_masking_lr_0.001_wd_0.0001_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_0-epoch=49.ckpt" \
--temperature 1.0 \
--sequence_length 125 \
--iterations 20 \
--n_samples 2 \
--context_mode True \
--pos_encoding_mode time_pos_sin \
--mask_scheduler 'pow' \
--d_model 768 \
--seed 100
echo '--- Finished computing model'
