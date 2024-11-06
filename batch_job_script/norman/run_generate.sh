#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=2' # request for exclusive access to gpu
#BSUB -n 64 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/generation_out_%J.out # output file
#BSUB -e logs/generation_error_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J eb_count_extrapolation_s100 # job name

# load cuda
module load cuda-12.1.1

echo '--- Start computing model'

export PYTHONPATH=/lustre/scratch126/cellgen/team361/chang/CellGen:$PYTHONPATH
export WANDB_DIR=$cwd/wandb



# interpolation
# python3 /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/val.py \
# python3 $cwd/val.py \
# --test_mode count \
# --split False \
# --splitting_mode random \
# --generate True \
# --ckpt_count_path './T_perturb/T_perturb/Model/checkpoints/'\
# '20240520_1446_extrapolate_lr_0.0001'\
# '_wd_0.0001_batch_32_zinb_tp_1-epoch=99.ckpt' \
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

# extrapolation
# python3 /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/val.py \
python3 /lustre/scratch126/cellgen/team361/chang/CellGen/T_perturb/val.py \
--test_mode count \
--split False \
--splitting_mode random \
--generate True \
--ckpt_count_path '/lustre/scratch126/cellgen/team361/chang/CellGen/T_perturb/Model/checkpoints/20240918_2227_cellgen_train_count_lr_0.0001_wd_0.0001_batch_32_zinb_ntask_2_s_42-epoch=19.ckpt' \
--output_dir '/lustre/scratch126/cellgen/team361/chang/CellGen/T_perturb/generation/' \
--src_dataset '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/dataset_hvg_src_random_pairing/control.dataset' \
--tgt_dataset_folder '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/dataset_hvg_tgt_random_pairing/' \
--tgt_dataset '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/dataset_hvg_tgt_random_pairing/perturbation.dataset' \
--src_adata '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/h5ad_pairing_hvg_src_random_pairing/control.h5ad' \
--tgt_adata_folder '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/h5ad_pairing_hvg_tgt_random_pairing/' \
--batch_size 32 \
--max_len 1650 \
--tgt_vocab_size 25426 \
--cellgen_lr 0.001 \
--cellgen_wd 0.0001 \
--count_lr 0.0001 \
--count_wd 0.0001 \
--num_layers 2 \
--d_ff 128 \
--loss_mode zinb \
--n_workers 32 \
--var_list Time_point true_counts ctrl_counts pred_counts cls_embeddings

echo '--- Finished computing model'