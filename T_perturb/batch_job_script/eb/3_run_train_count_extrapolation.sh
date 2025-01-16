#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=2' # request for exclusive access to gpu :gmodel=NVIDIAA100_SXM4_80GB
#BSUB -n 8 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/eb_count_extra_cont_generation_%J.out # output file
#BSUB -e logs/eb_count_extra_cont_generation_%J.err # error file
#BSUB -M 20000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>20000] rusage[mem=20000]' # RAM memory part 1. Default: 100MB
#BSUB -J eb_count_extra_cont_generation # job name

# load cuda
module load cuda-12.1.1

# activate python environment
source /lustre/scratch126/cellgen/team361/kl11/t_generative/.cellgen_4096/bin/activate
# cwd=$(pwd)

# results directory
RES_DIR="/lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/plt/res"
RES_NAME="eb/extrapolation/"

# # if directory does not exist, create it with the name $RES_NAME
# mkdir -p $RES_DIR/$RES_NAME
# # Get the current timestamp
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# # copy the current script to the result directory
# cp $0 $RES_DIR/$RES_NAME/3_run_train_count_extrapolation_$TIMESTAMP.sh
# echo "Copying script to $RES_DIR/$RES_NAME/3_run_train_count_extrapolation_$TIMESTAMP.sh"

# export WANDB_DIR=$cwd/wandb
# Run python script to train count decoder
echo '--- Start computing model'

# # python3 $cwd/train.py \
# Extrapolation
python3 /lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/train.py \
--train_mode count \
--split False \
--splitting_mode random \
--output_dir $RES_DIR/$RES_NAME/res \
--ckpt_masking_path 'T_perturb/T_perturb/plt/res/eb/extrapolation/res/checkpoints/20250116_1353_cellgen_train_masking_lr_0.001_wd_0.0001_batch_64_ptime_pos_sin_m_cosine_tp_1-2-3_s_42-epoch=49.ckpt' \
--src_dataset './T_perturb/T_perturb/pp/res/eb/dataset_hvg_subsetted_src/Day 00-03.dataset' \
--tgt_dataset_folder './T_perturb/T_perturb/pp/res/eb/dataset_hvg_subsetted_tgt' \
--src_adata './T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_src/Day 00-03.h5ad' \
--tgt_adata_folder './T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_tgt' \
--mapping_dict_path  './T_perturb/T_perturb/pp/res/eb/token_id_to_genename_hvg.pkl' \
--batch_size 64 \
--max_len 270 \
--epochs 100 \
--tgt_vocab_size 1730 \
--cellgen_lr 0.001 \
--count_lr 0.0001 \
--cellgen_wd 0.0001 \
--count_wd 0.0001 \
--count_dropout 0.25 \
--mlm_prob 0.15 \
--n_workers 8 \
--num_layers 3 \
--d_ff 32 \
--loss_mode zinb \
--pred_tps 1 2 3 \
--var_list Time_point \
--encoder scmaskgit \
--context_mode True \
--pos_encoding_mode time_pos_sin \
--mask_scheduler 'cosine' \
--d_model 768
echo '--- Finished computing model'
