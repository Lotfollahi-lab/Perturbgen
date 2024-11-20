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

cwd=/lustre/scratch126/cellgen/team361/chang/CellGen/T_perturb

export WANDB_DIR=$cwd/wandb
# run script
echo '--- Start computing model'

export PYTHONPATH=/lustre/scratch126/cellgen/team361/chang/CellGen:$PYTHONPATH

# interpolation
# python3 /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/val.py \
python3 $cwd/val.py \
--test_mode count \
--split False \
--splitting_mode random \
--generate True \
--ckpt_count_path '/lustre/scratch126/cellgen/team361/chang/CellGen/T_perturb/Model/checkpoints/20241120_1536_cellgen_train_count_lr_0.0001_wd_0.0001_batch_32_zinb_ntask_1_s_42-epoch=19.ckpt' \
--output_dir '/lustre/scratch126/cellgen/team361/chang/CellGen/T_perturb/plt/res/norman' \
--src_dataset '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/dataset_hvg_src_random_pairing/control.dataset' \
--src_adata '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/h5ad_pairing_hvg_src_random_pairing/control.h5ad' \
--tgt_adata_folder '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/h5ad_pairing_hvg_tgt_random_pairing' \
--tgt_dataset '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/dataset_hvg_tgt_random_pairing/perturbation.dataset' \
--tokenid_to_genename_dict '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/token_id_to_genename_hvg.pkl' \
--tgt_adata_folder '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/h5ad_pairing_hvg_tgt_random_pairing' \
--tgt_dataset_folder '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/dataset_hvg_tgt_random_pairing' \
--mapping_dict_path  '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/token_id_to_genename_hvg.pkl' \
--batch_size 32 \
--max_len 4000 \
--tgt_vocab_size 25426 \
--cellgen_lr 0.001 \
--cellgen_wd 0.0001 \
--count_lr 0.0001 \
--count_wd 0.0001 \
--num_layers 2 \
--d_ff 128 \
--loss_mode zinb \
--n_workers 16 \
--var_list Time_point
echo '--- Finished computing model'