#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=1' # request for exclusive access to gpu
#BSUB -n 32 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/eb_generate_%J.out # output file
#BSUB -e logs/eb_generate_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J eb_generate # job name

# load cuda
module load cuda-12.1.1

# activate pyenv
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.petra_cuda12/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run script
echo '--- Start computing model'


# interpolation
# python3 /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/val.py \
python3 $cwd/val.py \
--test_mode count \
--split False \
--splitting_mode random \
--generate True \
--ckpt_count_path './T_perturb/T_perturb/Model/checkpoints/'\
'20240520_0936_eb_no_context_lr_0.0001'\
'_wd_0.0001_batch_32_zinb_tp_1-2-4-epoch=99.ckpt' \
--output_dir './T_perturb/T_perturb/plt/res/eb' \
--src_dataset './T_perturb/T_perturb/pp/res/eb/dataset_hvg_src/Day 00-03.dataset' \
--tgt_dataset_folder './T_perturb/T_perturb/pp/res/eb/dataset_hvg_tgt' \
--src_adata './T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_src/Day 00-03.h5ad' \
--tgt_adata_folder './T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_tgt' \
--batch_size 32 \
--max_len 263 \
--tgt_vocab_size 2001 \
--petra_lr 0.001 \
--petra_wd 0.0001 \
--count_lr 0.0001 \
--count_wd 0.0001 \
--num_layers 2 \
--d_ff 16 \
--loss_mode zinb \
--n_workers 32 \
--time_steps 3 \
--var_list Time_point
echo '--- Finished computing model'

# # extrapolation
# # python3 /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/val.py \
# python3 $cwd/val.py \
# --test_mode count \
# --split False \
# --splitting_mode random \
# --generate True \
# --ckpt_count_path './T_perturb/T_perturb/Model/checkpoints/20240518_2103_eb_extrapol'\
# '_count_lr_0.0001_wd_0.0001'\
# '_batch_32_zinb_tp_1-2-epoch=99.ckpt' \
# --output_dir './T_perturb/T_perturb/plt/res/eb' \
# --src_dataset './T_perturb/T_perturb/pp/res/eb/dataset_hvg_src/Day 00-03.dataset' \
# --tgt_dataset_folder './T_perturb/T_perturb/pp/res/eb/dataset_hvg_tgt' \
# --src_adata './T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_src/Day 00-03.h5ad' \
# --tgt_adata_folder './T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_tgt' \
# --batch_size 32 \
# --max_len 263 \
# --tgt_vocab_size 2001 \
# --petra_lr 0.001 \
# --petra_wd 0.0001 \
# --count_lr 0.0001 \
# --count_wd 0.0001 \
# --num_layers 2 \
# --d_ff 16 \
# --loss_mode zinb \
# --n_workers 32 \
# --time_steps 3 \
# --var_list Time_point
# echo '--- Finished computing model'
