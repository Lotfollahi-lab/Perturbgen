#!/bin/bash
#BSUB -q gpu-lotfollahi-train # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=1' # request for exclusive access to gpu
#BSUB -n 4 # number of cores
#BSUB -R "span[ptile=4]"     # split X cores per host
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister # working directory
#BSUB -o logs/lps_extra_generate_s100_mask09_s200_%J.out # output file
#BSUB -e logs/lps_extra_generate_s100_mask09_s200_%J.err # error file
#BSUB -M 100000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>100000] rusage[mem=100000]' # RAM memory part 1. Default: 100MB
#BSUB -J lps_extra_generate_s100_mask09_s200 # job name

# load cuda
module load cuda-12.1.1

# activate pyenv
source /nfs/team361/cytomeister/.cytomeister/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run script
echo '--- Start computing model'

RES_DIR="/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/res/"
RES_NAME="lps/extrapolation"

python3 $cwd/val.py \
--test_mode count \
--split False \
--splitting_mode stratified \
--split_obs cell_type_harmonized \
--generate True \
--output_dir $RES_DIR/$RES_NAME \
--ckpt_count_path 'T_perturb/res/lps/extrapolation/checkpoints/20250902_1851_cellgen_train_count_lr_0.001_wd_0.0001_batch_64_drop_0.1_zinb_tp_1-2_s_100_pos_time_pos_sin_m_pow-epoch=03.ckpt' \
--src_dataset "T_perturb/tokenized_data/lps_100M/dataset_2000_hvg_src/normal.dataset" \
--tgt_dataset_folder "T_perturb/tokenized_data/lps_100M/dataset_2000_hvg_tgt" \
--src_adata "T_perturb/tokenized_data/lps_100M/h5ad_pairing_2000_hvg_src/normal.h5ad" \
--tgt_adata_folder "T_perturb/tokenized_data/lps_100M/h5ad_pairing_2000_hvg_tgt" \
--mapping_dict_path "T_perturb/tokenized_data/lps_100M/token_id_to_genename_2000_hvg.pkl" \
--batch_size 128 \
--count_lr 0.001 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--sequence_length 200 \
--count_wd 0.001 \
--num_layers 6 \
--d_ff 64 \
--loss_mode zinb \
--n_workers 4 \
--pred_tps 3 \
--context_tps 1 2 \
--var_list cell_type_harmonized cell_pairing_index time_after_LPS \
--encoder scmaskgit \
--encoder_path "/lustre/scratch126/cellgen/lotfollahi/av13/scmaskgit/foundation_107m/checkpoints/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt" \
--add_cell_time False \
--use_positional_encoding False \
--context_mode True \
--pos_encoding_mode time_pos_sin \
--mask_scheduler 'pow' \
--iterations 20 \
--temperature 0.25 \
--n_samples 2 \
--num_node 1 \
--d_model 768 \
--seed 100
echo '--- Finished computing model'
