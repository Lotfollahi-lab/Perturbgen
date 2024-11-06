#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-cellgeni-a100)
#BSUB -gpu 'mode=exclusive_process:num=1' # request for exclusive access to gpu
#BSUB -n 64 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/eb_masking_extra_s100_%J.out # output file
#BSUB -e logs/eb_masking_extra_s100_%J.err # error file
#BSUB -M 80000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>80000] rusage[mem=80000]' # RAM memory part 1. Default: 100MB
#BSUB -J eb_masking_extra_s100 # job name

# load cuda
module load cuda-12.1.1

# activate python environment
# source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.petra_cuda12/bin/activate
# cwd=$(pwd)

# export WANDB_DIR=$cwd/wandb
# Run python script to PETRA
echo '--- Start computing model'

# # # interpolation
# # python3 $cwd/train.py \
# python3 /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/train.py \
# --train_mode masking \
# --split False \
# --splitting_mode random \
# --src_dataset './T_perturb/T_perturb/pp/res/eb/dataset_hvg_src/Day 00-03.dataset' \
# --tgt_dataset_folder './T_perturb/T_perturb/pp/res/eb/dataset_hvg_tgt' \
# --src_adata './T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_src/Day 00-03.h5ad' \
# --tgt_adata_folder './T_perturb/T_perturb/pp/res/eb/h5ad_pairing_hvg_tgt' \
# --mapping_dict_path  './T_perturb/T_perturb/pp/res/eb/token_id_to_genename_hvg.pkl' \
# --batch_size 32 \
# --max_len 263 \
# --epochs 100 \
# --tgt_vocab_size 2001 \
# --petra_lr 0.001 \
# --petra_wd 0.0001 \
# --mlm_prob 0.15 \
# --n_workers 64 \
# --num_layers 2 \
# --d_ff 16 \
# --time_steps 1 2 4 \
# --var_list Time_point \
# --mode GF_frozen \
# --seed 100

# echo '--- Finished computing model'

# extrapolation
# python3 $cwd/train.py \
CUDA_LAUNCH_BLOCKING=1 python3 /lustre/scratch126/cellgen/team361/chang/CellGen/T_perturb/train.py \
--train_mode masking \
--split False \
--splitting_mode random \
--src_dataset '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/dataset_hvg_src_random_pairing/control.dataset' \
--tgt_dataset '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/dataset_hvg_tgt_random_pairing/perturbation.dataset' \
--src_adata '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/h5ad_pairing_hvg_src_random_pairing/control.h5ad' \
--tgt_adata '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/h5ad_pairing_hvg_tgt_random_pairing/perturbation.h5ad' \
--tgt_adata_path '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/h5ad_pairing_hvg_tgt_random_pairing/' \
--mapping_dict_path '/lustre/scratch126/cellgen/team361/chang/CellGen/perturbench/perturbench_data/norman/token_id_to_genename_hvg.pkl' \
--batch_size 16 \
--max_len 4000 \
--epochs 20 \
--tgt_vocab_size 25426 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--mlm_prob 0.15 \
--n_workers 16 \
--d_ff 128 \
--num_layers 2 \
--n_task_conditions 2 \
--var_list good_coverage cell_type perturbation_type celltype perturbation nperts ngenes condition ncounts percent_mito total_counts \
--encoder_type GF_frozen \
--alpha 0.5 \
--seed 100

echo '--- Finished computing model'