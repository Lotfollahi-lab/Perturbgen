#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=2' # request for exclusive access to gpu
#BSUB -n 64 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/eb_count_s100_%J.out # output file
#BSUB -e logs/eb_count_s100_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J eb_count_extrapolation_s100 # job name

# load cuda
module load cuda-12.1.1


# FOR PERTURBATION MODELING

export PYTHONPATH=/lustre/scratch126/cellgen/team361/chang/CellGen:$PYTHONPATH


# activate conda environment
# source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.petra_cuda12/bin/activate
# cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
export WANDB_API_KEY=1083c9998690575185cfb3235dba7d5be067c59a


# Run python script to train count decoder
echo '--- Start computing model'
# # python3 $cwd/train.py \
# # Interpolation
# python3 /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/train.py \
# --train_mode count \
# --split False \
# --splitting_mode random \
# --output_dir './T_perturb/T_perturb/plt/res/eb' \
# --ckpt_masking_path './T_perturb/T_perturb/Model/checkpoints/20240522_1054_petra_train_masking_lr_0.001_wd_0.0001_batch_32_mlmp_0.15_tp_1-2-4_s_100-epoch=99.ckpt' \
# --src_dataset './T_perturb/T_perturb/pp/res/eb/dataset_hvg_src/Day 00-03.dataset' \
# --tgt_dataset_folder './T_perturb/T_perturb/pp/res/eb/dataset_hvg_tgt' \
# --src_adata './T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_src/Day 00-03.h5ad' \
# --tgt_adata_folder './T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_tgt' \
# --mapping_dict_path  './T_perturb/T_perturb/pp/res/eb/token_id_to_genename_hvg.pkl' \
# --batch_size 32 \
# --max_len 263 \
# --epochs 100 \
# --tgt_vocab_size 2001 \
# --cellgen_lr 0.001 \
# --count_lr 0.0001 \
# --cellgen_wd 0.0001 \
# --count_wd 0.0001 \
# --count_dropout 0.25 \
# --mlm_prob 0.15 \
# --n_workers 64 \
# --num_layers 2 \
# --d_ff 16 \
# --loss_mode zinb \
# --time_steps 1 2 4 \
# --var_list Time_point \
# --mode GF_frozen \
# --seed 100
# echo '--- Finished computing model'

# # Extrapolation
python3 /lustre/scratch126/cellgen/team361/chang/CellGen/T_perturb/train.py \
--train_mode count \
--split False \
--splitting_mode random \
--output_dir '/lustre/scratch126/cellgen/team361/chang/CellGen/T_perturb/plt/res/norman/' \
--ckpt_masking_path '/lustre/scratch126/cellgen/team361/chang/CellGen/T_perturb/Model/checkpoints/20241029_1236_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_16_mlmp_0.15_ntask_2_s_100-epoch=19.ckpt' \
--src_dataset '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/dataset_hvg_src_random_pairing/control.dataset' \
--tgt_dataset '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/dataset_hvg_tgt_random_pairing/perturbation.dataset' \
--src_adata '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/h5ad_pairing_hvg_src_random_pairing/control.h5ad' \
--tgt_adata '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/h5ad_pairing_hvg_tgt_random_pairing/perturbation.h5ad' \
--tgt_adata_path '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/h5ad_pairing_hvg_tgt_random_pairing/' \
--mapping_dict_path  '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/token_id_to_genename_hvg.pkl' \
--batch_size 32 \
--max_len 4000 \
--epochs 20 \
--tgt_vocab_size 25426 \
--cellgen_lr 0.001 \
--count_lr 0.0001 \
--cellgen_wd 0.0001 \
--count_wd 0.0001 \
--count_dropout 0.25 \
--mlm_prob 0.15 \
--n_workers 16 \
--num_layers 2 \
--d_ff 128 \
--loss_mode zinb \
--var_list Time_point \
--encoder_type GF_frozen
echo '--- Finished computing model'
