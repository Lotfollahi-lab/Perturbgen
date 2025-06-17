#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=1' # request for exclusive access to gpu
#BSUB -n 32 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/eb_generate_extra_%J.out # output file
#BSUB -e logs/eb_generate_extra_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J eb_generate_extra # job name

# load cuda
module load cuda-12.1.1

# activate pyenv
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run script
echo '--- Start computing model'

RES_DIR="/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/iclr"
RES_NAME="eb/extrapolation/"

# if directory does not exist, create it with the name $RES_NAME
mkdir -p $RES_DIR/$RES_NAME
# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# copy the current script to the result directory
cp $0 $RES_DIR/$RES_NAME/4_run_val_generate_extrapolation_$TIMESTAMP.sh
echo "Copying script to $RES_DIR/$RES_NAME/4_run_val_generate_extrapolation_$TIMESTAMP.sh"

# extrapolation
# python3 /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/val.py \
python3 $cwd/val.py \
--test_mode count \
--split False \
--splitting_mode random \
--generate True \
--ckpt_count_path './T_perturb/T_perturb/iclr/eb/extrapolation/res/checkpoints/20240917_1437_cellgen_train_count_lr_0.0001_wd_0.0001_batch_32_zinb_tp_1-2_s_42-epoch=49.ckpt' \
--output_dir $RES_DIR/$RES_NAME/res \
--src_dataset './T_perturb/T_perturb/pp/res/eb/dataset_hvg_src_random_pairing_4096/Day 00-03.dataset' \
--tgt_dataset_folder './T_perturb/T_perturb/pp/res/eb/dataset_hvg_tgt_random_pairing_4096' \
--src_adata './T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_src_random_pairing_4096/Day 00-03.h5ad' \
--tgt_adata_folder './T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_tgt_random_pairing_4096' \
--batch_size 32 \
--max_len 270 \
--tgt_vocab_size 2001 \
--cellgen_lr 0.001 \
--cellgen_wd 0.0001 \
--count_lr 0.0001 \
--count_wd 0.0001 \
--num_layers 2 \
--d_ff 16 \
--loss_mode zinb \
--n_workers 32 \
--time_steps 3 \
--var_list Time_point
echo '--- Finished computing model'
